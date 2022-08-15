//===- Inliner.cpp - Code common to all inliners --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the mechanics required to implement inlining without
// missing any calls and updating the call graph.  The decisions of which calls
// are profitable to inline are implemented elsewhere.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "inline"

STATISTIC(NumInlined, "Number of functions inlined");
STATISTIC(NumCallsDeleted, "Number of call sites deleted, not inlined");
STATISTIC(NumDeleted, "Number of functions deleted because all callers found");
// STATISTIC(NumMergedAllocas, "Number of allocas merged together"); // HLSL Change - unused

// This weirdly named statistic tracks the number of times that, when attempting
// to inline a function A into B, we analyze the callers of B in order to see
// if those would be more profitable and blocked inline steps.
STATISTIC(NumCallerCallersAnalyzed, "Number of caller-callers analyzed");

#if 0 // HLSL Change Starts
static cl::opt<int>
InlineLimit("inline-threshold", cl::Hidden, cl::init(225), cl::ZeroOrMore,
        cl::desc("Control the amount of inlining to perform (default = 225)"));

static cl::opt<int>
HintThreshold("inlinehint-threshold", cl::Hidden, cl::init(325),
              cl::desc("Threshold for inlining functions with inline hint"));

// We instroduce this threshold to help performance of instrumentation based
// PGO before we actually hook up inliner with analysis passes such as BPI and
// BFI.
static cl::opt<int>
ColdThreshold("inlinecold-threshold", cl::Hidden, cl::init(225),
              cl::desc("Threshold for inlining functions with cold attribute"));
#else
struct NullOpt {
  NullOpt(int val) : _val(val) {}
  int _val;
  int getNumOccurrences() const { return 0; }
  operator int() const {
    return _val;
  }
};
static const NullOpt InlineLimit(225);
static const NullOpt HintThreshold(325);
static const NullOpt ColdThreshold(225);
#endif // HLSL Change Ends

// Threshold to use when optsize is specified (and there is no -inline-limit).
const int OptSizeThreshold = 75;

Inliner::Inliner(char &ID) 
  : CallGraphSCCPass(ID), InlineThreshold(InlineLimit), InsertLifetime(true) {}

Inliner::Inliner(char &ID, int Threshold, bool InsertLifetime)
  : CallGraphSCCPass(ID), InlineThreshold(InlineLimit.getNumOccurrences() > 0 ?
                                          unsigned(InlineLimit) : Threshold),
    InsertLifetime(InsertLifetime) {}

/// For this class, we declare that we require and preserve the call graph.
/// If the derived class implements this method, it should
/// always explicitly call the implementation here.
void Inliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<AssumptionCacheTracker>();
  CallGraphSCCPass::getAnalysisUsage(AU);
}


typedef DenseMap<ArrayType*, std::vector<AllocaInst*> >
InlinedArrayAllocasTy;

/// \brief If the inlined function had a higher stack protection level than the
/// calling function, then bump up the caller's stack protection level.
static void AdjustCallerSSPLevel(Function *Caller, Function *Callee) {
  // If upgrading the SSP attribute, clear out the old SSP Attributes first.
  // Having multiple SSP attributes doesn't actually hurt, but it adds useless
  // clutter to the IR.
  AttrBuilder B;
  B.addAttribute(Attribute::StackProtect)
    .addAttribute(Attribute::StackProtectStrong)
    .addAttribute(Attribute::StackProtectReq);
  AttributeSet OldSSPAttr = AttributeSet::get(Caller->getContext(),
                                              AttributeSet::FunctionIndex,
                                              B);

  if (Callee->hasFnAttribute(Attribute::SafeStack)) {
    Caller->removeAttributes(AttributeSet::FunctionIndex, OldSSPAttr);
    Caller->addFnAttr(Attribute::SafeStack);
  } else if (Callee->hasFnAttribute(Attribute::StackProtectReq) &&
             !Caller->hasFnAttribute(Attribute::SafeStack)) {
    Caller->removeAttributes(AttributeSet::FunctionIndex, OldSSPAttr);
    Caller->addFnAttr(Attribute::StackProtectReq);
  } else if (Callee->hasFnAttribute(Attribute::StackProtectStrong) &&
             !Caller->hasFnAttribute(Attribute::SafeStack) &&
             !Caller->hasFnAttribute(Attribute::StackProtectReq)) {
    Caller->removeAttributes(AttributeSet::FunctionIndex, OldSSPAttr);
    Caller->addFnAttr(Attribute::StackProtectStrong);
  } else if (Callee->hasFnAttribute(Attribute::StackProtect) &&
             !Caller->hasFnAttribute(Attribute::SafeStack) &&
             !Caller->hasFnAttribute(Attribute::StackProtectReq) &&
             !Caller->hasFnAttribute(Attribute::StackProtectStrong))
    Caller->addFnAttr(Attribute::StackProtect);
}

/// If it is possible to inline the specified call site,
/// do so and update the CallGraph for this operation.
///
/// This function also does some basic book-keeping to update the IR.  The
/// InlinedArrayAllocas map keeps track of any allocas that are already
/// available from other functions inlined into the caller.  If we are able to
/// inline this call site we attempt to reuse already available allocas or add
/// any new allocas to the set if not possible.
static bool InlineCallIfPossible(CallSite CS, InlineFunctionInfo &IFI,
                                 InlinedArrayAllocasTy &InlinedArrayAllocas,
                                 int InlineHistory, bool InsertLifetime) {
  Function *Callee = CS.getCalledFunction();
  Function *Caller = CS.getCaller();

  // Try to inline the function.  Get the list of static allocas that were
  // inlined.
  if (!InlineFunction(CS, IFI, InsertLifetime))
    return false;

  AdjustCallerSSPLevel(Caller, Callee);

  // HLSL Change Begin- not merge allocas.
  // Merge alloca will make alloca which has one def become multi def.
  // SROA will fail to remove the merged allocas.
  return true;
  // HLSL Change End.
#if 0 // HLSL Change - disable unused code.
  // Look at all of the allocas that we inlined through this call site.  If we
  // have already inlined other allocas through other calls into this function,
  // then we know that they have disjoint lifetimes and that we can merge them.
  //
  // There are many heuristics possible for merging these allocas, and the
  // different options have different tradeoffs.  One thing that we *really*
  // don't want to hurt is SRoA: once inlining happens, often allocas are no
  // longer address taken and so they can be promoted.
  //
  // Our "solution" for that is to only merge allocas whose outermost type is an
  // array type.  These are usually not promoted because someone is using a
  // variable index into them.  These are also often the most important ones to
  // merge.
  //
  // A better solution would be to have real memory lifetime markers in the IR
  // and not have the inliner do any merging of allocas at all.  This would
  // allow the backend to do proper stack slot coloring of all allocas that
  // *actually make it to the backend*, which is really what we want.
  //
  // Because we don't have this information, we do this simple and useful hack.
  //
  SmallPtrSet<AllocaInst*, 16> UsedAllocas;
  
  // When processing our SCC, check to see if CS was inlined from some other
  // call site.  For example, if we're processing "A" in this code:
  //   A() { B() }
  //   B() { x = alloca ... C() }
  //   C() { y = alloca ... }
  // Assume that C was not inlined into B initially, and so we're processing A
  // and decide to inline B into A.  Doing this makes an alloca available for
  // reuse and makes a callsite (C) available for inlining.  When we process
  // the C call site we don't want to do any alloca merging between X and Y
  // because their scopes are not disjoint.  We could make this smarter by
  // keeping track of the inline history for each alloca in the
  // InlinedArrayAllocas but this isn't likely to be a significant win.
  if (InlineHistory != -1)  // Only do merging for top-level call sites in SCC.
    return true;
  
  // Loop over all the allocas we have so far and see if they can be merged with
  // a previously inlined alloca.  If not, remember that we had it.
  for (unsigned AllocaNo = 0, e = IFI.StaticAllocas.size();
       AllocaNo != e; ++AllocaNo) {
    AllocaInst *AI = IFI.StaticAllocas[AllocaNo];
    
    // Don't bother trying to merge array allocations (they will usually be
    // canonicalized to be an allocation *of* an array), or allocations whose
    // type is not itself an array (because we're afraid of pessimizing SRoA).
    ArrayType *ATy = dyn_cast<ArrayType>(AI->getAllocatedType());
    if (!ATy || AI->isArrayAllocation())
      continue;
    
    // Get the list of all available allocas for this array type.
    std::vector<AllocaInst*> &AllocasForType = InlinedArrayAllocas[ATy];
    
    // Loop over the allocas in AllocasForType to see if we can reuse one.  Note
    // that we have to be careful not to reuse the same "available" alloca for
    // multiple different allocas that we just inlined, we use the 'UsedAllocas'
    // set to keep track of which "available" allocas are being used by this
    // function.  Also, AllocasForType can be empty of course!
    bool MergedAwayAlloca = false;
    for (AllocaInst *AvailableAlloca : AllocasForType) {

      unsigned Align1 = AI->getAlignment(),
               Align2 = AvailableAlloca->getAlignment();
      
      // The available alloca has to be in the right function, not in some other
      // function in this SCC.
      if (AvailableAlloca->getParent() != AI->getParent())
        continue;
      
      // If the inlined function already uses this alloca then we can't reuse
      // it.
      if (!UsedAllocas.insert(AvailableAlloca).second)
        continue;
      
      // Otherwise, we *can* reuse it, RAUW AI into AvailableAlloca and declare
      // success!
      DEBUG(dbgs() << "    ***MERGED ALLOCA: " << *AI << "\n\t\tINTO: "
                   << *AvailableAlloca << '\n');
      
      AI->replaceAllUsesWith(AvailableAlloca);

      if (Align1 != Align2) {
        if (!Align1 || !Align2) {
          const DataLayout &DL = Caller->getParent()->getDataLayout();
          unsigned TypeAlign = DL.getABITypeAlignment(AI->getAllocatedType());

          Align1 = Align1 ? Align1 : TypeAlign;
          Align2 = Align2 ? Align2 : TypeAlign;
        }

        if (Align1 > Align2)
          AvailableAlloca->setAlignment(AI->getAlignment());
      }

      AI->eraseFromParent();
      MergedAwayAlloca = true;
      ++NumMergedAllocas;
      IFI.StaticAllocas[AllocaNo] = nullptr;
      break;
    }

    // If we already nuked the alloca, we're done with it.
    if (MergedAwayAlloca)
      continue;
    
    // If we were unable to merge away the alloca either because there are no
    // allocas of the right type available or because we reused them all
    // already, remember that this alloca came from an inlined function and mark
    // it used so we don't reuse it for other allocas from this inline
    // operation.
    AllocasForType.push_back(AI);
    UsedAllocas.insert(AI);
  }
  
  return true;
#endif
}

unsigned Inliner::getInlineThreshold(CallSite CS) const {
  int thres = InlineThreshold; // -inline-threshold or else selected by
                               // overall opt level

  // If -inline-threshold is not given, listen to the optsize attribute when it
  // would decrease the threshold.
  Function *Caller = CS.getCaller();
  bool OptSize = Caller && !Caller->isDeclaration() &&
                 Caller->hasFnAttribute(Attribute::OptimizeForSize);
  if (!(InlineLimit.getNumOccurrences() > 0) && OptSize &&
      OptSizeThreshold < thres)
    thres = OptSizeThreshold;

  // Listen to the inlinehint attribute when it would increase the threshold
  // and the caller does not need to minimize its size.
  Function *Callee = CS.getCalledFunction();
  bool InlineHint = Callee && !Callee->isDeclaration() &&
                    Callee->hasFnAttribute(Attribute::InlineHint);
  if (InlineHint && HintThreshold > thres &&
      !Caller->hasFnAttribute(Attribute::MinSize))
    thres = HintThreshold;

  // Listen to the cold attribute when it would decrease the threshold.
  bool ColdCallee = Callee && !Callee->isDeclaration() &&
                    Callee->hasFnAttribute(Attribute::Cold);
  // Command line argument for InlineLimit will override the default
  // ColdThreshold. If we have -inline-threshold but no -inlinecold-threshold,
  // do not use the default cold threshold even if it is smaller.
  if ((InlineLimit.getNumOccurrences() == 0 ||
       ColdThreshold.getNumOccurrences() > 0) && ColdCallee &&
      ColdThreshold < thres)
    thres = ColdThreshold;

  return thres;
}

static void emitAnalysis(CallSite CS, const Twine &Msg) {
  Function *Caller = CS.getCaller();
  LLVMContext &Ctx = Caller->getContext();
  DebugLoc DLoc = CS.getInstruction()->getDebugLoc();
  emitOptimizationRemarkAnalysis(Ctx, DEBUG_TYPE, *Caller, DLoc, Msg);
}

/// Return true if the inliner should attempt to inline at the given CallSite.
bool Inliner::shouldInline(CallSite CS) {
  InlineCost IC = getInlineCost(CS);
  
  if (IC.isAlways()) {
    DEBUG(dbgs() << "    Inlining: cost=always"
          << ", Call: " << *CS.getInstruction() << "\n");
    emitAnalysis(CS, Twine(CS.getCalledFunction()->getName()) +
                         " should always be inlined (cost=always)");
    return true;
  }
  
  if (IC.isNever()) {
    DEBUG(dbgs() << "    NOT Inlining: cost=never"
          << ", Call: " << *CS.getInstruction() << "\n");
    emitAnalysis(CS, Twine(CS.getCalledFunction()->getName() +
                           " should never be inlined (cost=never)"));
    return false;
  }
  
  Function *Caller = CS.getCaller();
  if (!IC) {
    DEBUG(dbgs() << "    NOT Inlining: cost=" << IC.getCost()
          << ", thres=" << (IC.getCostDelta() + IC.getCost())
          << ", Call: " << *CS.getInstruction() << "\n");
    emitAnalysis(CS, Twine(CS.getCalledFunction()->getName() +
                           " too costly to inline (cost=") +
                         Twine(IC.getCost()) + ", threshold=" +
                         Twine(IC.getCostDelta() + IC.getCost()) + ")");
    return false;
  }
  
  // Try to detect the case where the current inlining candidate caller (call
  // it B) is a static or linkonce-ODR function and is an inlining candidate
  // elsewhere, and the current candidate callee (call it C) is large enough
  // that inlining it into B would make B too big to inline later. In these
  // circumstances it may be best not to inline C into B, but to inline B into
  // its callers.
  //
  // This only applies to static and linkonce-ODR functions because those are
  // expected to be available for inlining in the translation units where they
  // are used. Thus we will always have the opportunity to make local inlining
  // decisions. Importantly the linkonce-ODR linkage covers inline functions
  // and templates in C++.
  //
  // FIXME: All of this logic should be sunk into getInlineCost. It relies on
  // the internal implementation of the inline cost metrics rather than
  // treating them as truly abstract units etc.
  if (Caller->hasLocalLinkage() || Caller->hasLinkOnceODRLinkage()) {
    int TotalSecondaryCost = 0;
    // The candidate cost to be imposed upon the current function.
    int CandidateCost = IC.getCost() - (InlineConstants::CallPenalty + 1);
    // This bool tracks what happens if we do NOT inline C into B.
    bool callerWillBeRemoved = Caller->hasLocalLinkage();
    // This bool tracks what happens if we DO inline C into B.
    bool inliningPreventsSomeOuterInline = false;
    for (User *U : Caller->users()) {
      CallSite CS2(U);

      // If this isn't a call to Caller (it could be some other sort
      // of reference) skip it.  Such references will prevent the caller
      // from being removed.
      if (!CS2 || CS2.getCalledFunction() != Caller) {
        callerWillBeRemoved = false;
        continue;
      }

      InlineCost IC2 = getInlineCost(CS2);
      ++NumCallerCallersAnalyzed;
      if (!IC2) {
        callerWillBeRemoved = false;
        continue;
      }
      if (IC2.isAlways())
        continue;

      // See if inlining or original callsite would erase the cost delta of
      // this callsite. We subtract off the penalty for the call instruction,
      // which we would be deleting.
      if (IC2.getCostDelta() <= CandidateCost) {
        inliningPreventsSomeOuterInline = true;
        TotalSecondaryCost += IC2.getCost();
      }
    }
    // If all outer calls to Caller would get inlined, the cost for the last
    // one is set very low by getInlineCost, in anticipation that Caller will
    // be removed entirely.  We did not account for this above unless there
    // is only one caller of Caller.
    if (callerWillBeRemoved && !Caller->use_empty())
      TotalSecondaryCost += InlineConstants::LastCallToStaticBonus;

    if (inliningPreventsSomeOuterInline && TotalSecondaryCost < IC.getCost()) {
      DEBUG(dbgs() << "    NOT Inlining: " << *CS.getInstruction() <<
           " Cost = " << IC.getCost() <<
           ", outer Cost = " << TotalSecondaryCost << '\n');
      emitAnalysis(
          CS, Twine("Not inlining. Cost of inlining " +
                    CS.getCalledFunction()->getName() +
                    " increases the cost of inlining " +
                    CS.getCaller()->getName() + " in other contexts"));
      return false;
    }
  }

  DEBUG(dbgs() << "    Inlining: cost=" << IC.getCost()
        << ", thres=" << (IC.getCostDelta() + IC.getCost())
        << ", Call: " << *CS.getInstruction() << '\n');
  emitAnalysis(
      CS, CS.getCalledFunction()->getName() + Twine(" can be inlined into ") +
              CS.getCaller()->getName() + " with cost=" + Twine(IC.getCost()) +
              " (threshold=" + Twine(IC.getCostDelta() + IC.getCost()) + ")");
  return true;
}

/// Return true if the specified inline history ID
/// indicates an inline history that includes the specified function.
static bool InlineHistoryIncludes(Function *F, int InlineHistoryID,
            const SmallVectorImpl<std::pair<Function*, int> > &InlineHistory) {
  while (InlineHistoryID != -1) {
    assert(unsigned(InlineHistoryID) < InlineHistory.size() &&
           "Invalid inline history ID");
    if (InlineHistory[InlineHistoryID].first == F)
      return true;
    InlineHistoryID = InlineHistory[InlineHistoryID].second;
  }
  return false;
}

bool Inliner::runOnSCC(CallGraphSCC &SCC) {
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  AssumptionCacheTracker *ACT = &getAnalysis<AssumptionCacheTracker>();
  auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  const TargetLibraryInfo *TLI = TLIP ? &TLIP->getTLI() : nullptr;
  AliasAnalysis *AA = &getAnalysis<AliasAnalysis>();

  SmallPtrSet<Function*, 8> SCCFunctions;
  DEBUG(dbgs() << "Inliner visiting SCC:");
  for (CallGraphNode *Node : SCC) {
    Function *F = Node->getFunction();
    if (F) SCCFunctions.insert(F);
    DEBUG(dbgs() << " " << (F ? F->getName() : "INDIRECTNODE"));
  }

  // Scan through and identify all call sites ahead of time so that we only
  // inline call sites in the original functions, not call sites that result
  // from inlining other functions.
  SmallVector<std::pair<CallSite, int>, 16> CallSites;
  
  // When inlining a callee produces new call sites, we want to keep track of
  // the fact that they were inlined from the callee.  This allows us to avoid
  // infinite inlining in some obscure cases.  To represent this, we use an
  // index into the InlineHistory vector.
  SmallVector<std::pair<Function*, int>, 8> InlineHistory;

  for (CallGraphNode *Node : SCC) {
    Function *F = Node->getFunction();
    if (!F) continue;
    
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB) {
        CallSite CS(cast<Value>(&I));
        // If this isn't a call, or it is a call to an intrinsic, it can
        // never be inlined.
        if (!CS || isa<IntrinsicInst>(I))
          continue;
        
        // If this is a direct call to an external function, we can never inline
        // it.  If it is an indirect call, inlining may resolve it to be a
        // direct call, so we keep it.
        if (CS.getCalledFunction() && CS.getCalledFunction()->isDeclaration())
          continue;
        
        CallSites.push_back(std::make_pair(CS, -1));
      }
  }

  DEBUG(dbgs() << ": " << CallSites.size() << " call sites.\n");

  // If there are no calls in this function, exit early.
  if (CallSites.empty())
    return false;

  // Now that we have all of the call sites, move the ones to functions in the
  // current SCC to the end of the list.
  unsigned FirstCallInSCC = CallSites.size();
  for (unsigned i = 0; i < FirstCallInSCC; ++i)
    if (Function *F = CallSites[i].first.getCalledFunction())
      if (SCCFunctions.count(F))
        std::swap(CallSites[i--], CallSites[--FirstCallInSCC]);

  
  InlinedArrayAllocasTy InlinedArrayAllocas;
  InlineFunctionInfo InlineInfo(&CG, AA, ACT);

  // Now that we have all of the call sites, loop over them and inline them if
  // it looks profitable to do so.
  bool Changed = false;
  bool LocalChange;
  do {
    LocalChange = false;
    // Iterate over the outer loop because inlining functions can cause indirect
    // calls to become direct calls.
    // CallSites may be modified inside so ranged for loop can not be used.
    for (unsigned CSi = 0; CSi != CallSites.size(); ++CSi) {
      CallSite CS = CallSites[CSi].first;
      
      Function *Caller = CS.getCaller();
      Function *Callee = CS.getCalledFunction();

      // If this call site is dead and it is to a readonly function, we should
      // just delete the call instead of trying to inline it, regardless of
      // size.  This happens because IPSCCP propagates the result out of the
      // call and then we're left with the dead call.
      if (isInstructionTriviallyDead(CS.getInstruction(), TLI)) {
        DEBUG(dbgs() << "    -> Deleting dead call: "
                     << *CS.getInstruction() << "\n");
        // Update the call graph by deleting the edge from Callee to Caller.
        CG[Caller]->removeCallEdgeFor(CS);
        CS.getInstruction()->eraseFromParent();
        ++NumCallsDeleted;
      } else {
        // We can only inline direct calls to non-declarations.
        if (!Callee || Callee->isDeclaration()) continue;
      
        // If this call site was obtained by inlining another function, verify
        // that the include path for the function did not include the callee
        // itself.  If so, we'd be recursively inlining the same function,
        // which would provide the same callsites, which would cause us to
        // infinitely inline.
        int InlineHistoryID = CallSites[CSi].second;
        if (InlineHistoryID != -1 &&
            InlineHistoryIncludes(Callee, InlineHistoryID, InlineHistory))
          continue;
        
        LLVMContext &CallerCtx = Caller->getContext();

        // Get DebugLoc to report. CS will be invalid after Inliner.
        DebugLoc DLoc = CS.getInstruction()->getDebugLoc();

        // If the policy determines that we should inline this function,
        // try to do so.
        if (!shouldInline(CS)) {
          emitOptimizationRemarkMissed(CallerCtx, DEBUG_TYPE, *Caller, DLoc,
                                       Twine(Callee->getName() +
                                             " will not be inlined into " +
                                             Caller->getName()));
          continue;
        }

        // Attempt to inline the function.
        if (!InlineCallIfPossible(CS, InlineInfo, InlinedArrayAllocas,
                                  InlineHistoryID, InsertLifetime)) {
          emitOptimizationRemarkMissed(CallerCtx, DEBUG_TYPE, *Caller, DLoc,
                                       Twine(Callee->getName() +
                                             " will not be inlined into " +
                                             Caller->getName()));
          continue;
        }
        ++NumInlined;

        // Report the inline decision.
        emitOptimizationRemark(
            CallerCtx, DEBUG_TYPE, *Caller, DLoc,
            Twine(Callee->getName() + " inlined into " + Caller->getName()));

        // If inlining this function gave us any new call sites, throw them
        // onto our worklist to process.  They are useful inline candidates.
        if (!InlineInfo.InlinedCalls.empty()) {
          // Create a new inline history entry for this, so that we remember
          // that these new callsites came about due to inlining Callee.
          int NewHistoryID = InlineHistory.size();
          InlineHistory.push_back(std::make_pair(Callee, InlineHistoryID));

          for (Value *Ptr : InlineInfo.InlinedCalls)
            CallSites.push_back(std::make_pair(CallSite(Ptr), NewHistoryID));
        }
      }
      
      // If we inlined or deleted the last possible call site to the function,
      // delete the function body now.
      if (Callee && Callee->use_empty() && Callee->hasLocalLinkage() &&
          // TODO: Can remove if in SCC now.
          !SCCFunctions.count(Callee) &&
          
          // The function may be apparently dead, but if there are indirect
          // callgraph references to the node, we cannot delete it yet, this
          // could invalidate the CGSCC iterator.
          CG[Callee]->getNumReferences() == 0) {
        DEBUG(dbgs() << "    -> Deleting dead function: "
              << Callee->getName() << "\n");
        CallGraphNode *CalleeNode = CG[Callee];

        // Remove any call graph edges from the callee to its callees.
        CalleeNode->removeAllCalledFunctions();
        
        // Removing the node for callee from the call graph and delete it.
        delete CG.removeFunctionFromModule(CalleeNode);
        ++NumDeleted;
      }

      // Remove this call site from the list.  If possible, use 
      // swap/pop_back for efficiency, but do not use it if doing so would
      // move a call site to a function in this SCC before the
      // 'FirstCallInSCC' barrier.
      if (SCC.isSingular()) {
        CallSites[CSi] = CallSites.back();
        CallSites.pop_back();
      } else {
        CallSites.erase(CallSites.begin()+CSi);
      }
      --CSi;

      Changed = true;
      LocalChange = true;
    }
  } while (LocalChange);

  return Changed;
}

/// Remove now-dead linkonce functions at the end of
/// processing to avoid breaking the SCC traversal.
bool Inliner::doFinalization(CallGraph &CG) {
  return removeDeadFunctions(CG);
}

/// Remove dead functions that are not included in DNR (Do Not Remove) list.
bool Inliner::removeDeadFunctions(CallGraph &CG, bool AlwaysInlineOnly) {
  SmallVector<CallGraphNode*, 16> FunctionsToRemove;
  SmallVector<CallGraphNode *, 16> DeadFunctionsInComdats;
  SmallDenseMap<const Comdat *, int, 16> ComdatEntriesAlive;

  auto RemoveCGN = [&](CallGraphNode *CGN) {
    // Remove any call graph edges from the function to its callees.
    CGN->removeAllCalledFunctions();

    // Remove any edges from the external node to the function's call graph
    // node.  These edges might have been made irrelegant due to
    // optimization of the program.
    CG.getExternalCallingNode()->removeAnyCallEdgeTo(CGN);

    // Removing the node for callee from the call graph and delete it.
    FunctionsToRemove.push_back(CGN);
  };

  // Scan for all of the functions, looking for ones that should now be removed
  // from the program.  Insert the dead ones in the FunctionsToRemove set.
  for (const auto &I : CG) {
    CallGraphNode *CGN = I.second.get();
    Function *F = CGN->getFunction();
    if (!F || F->isDeclaration())
      continue;

    // Handle the case when this function is called and we only want to care
    // about always-inline functions. This is a bit of a hack to share code
    // between here and the InlineAlways pass.
    if (AlwaysInlineOnly && !F->hasFnAttribute(Attribute::AlwaysInline))
      continue;

    // If the only remaining users of the function are dead constants, remove
    // them.
    F->removeDeadConstantUsers();

    if (!F->isDefTriviallyDead())
      continue;

    // It is unsafe to drop a function with discardable linkage from a COMDAT
    // without also dropping the other members of the COMDAT.
    // The inliner doesn't visit non-function entities which are in COMDAT
    // groups so it is unsafe to do so *unless* the linkage is local.
    if (!F->hasLocalLinkage()) {
      if (const Comdat *C = F->getComdat()) {
        --ComdatEntriesAlive[C];
        DeadFunctionsInComdats.push_back(CGN);
        continue;
      }
    }

    RemoveCGN(CGN);
  }
  if (!DeadFunctionsInComdats.empty()) {
    // Count up all the entities in COMDAT groups
    auto ComdatGroupReferenced = [&](const Comdat *C) {
      auto I = ComdatEntriesAlive.find(C);
      if (I != ComdatEntriesAlive.end())
        ++(I->getSecond());
    };
    for (const Function &F : CG.getModule())
      if (const Comdat *C = F.getComdat())
        ComdatGroupReferenced(C);
    for (const GlobalVariable &GV : CG.getModule().globals())
      if (const Comdat *C = GV.getComdat())
        ComdatGroupReferenced(C);
    for (const GlobalAlias &GA : CG.getModule().aliases())
      if (const Comdat *C = GA.getComdat())
        ComdatGroupReferenced(C);
    for (CallGraphNode *CGN : DeadFunctionsInComdats) {
      Function *F = CGN->getFunction();
      const Comdat *C = F->getComdat();
      int NumAlive = ComdatEntriesAlive[C];
      // We can remove functions in a COMDAT group if the entire group is dead.
      assert(NumAlive >= 0);
      if (NumAlive > 0)
        continue;

      RemoveCGN(CGN);
    }
  }

  if (FunctionsToRemove.empty())
    return false;

  // Now that we know which functions to delete, do so.  We didn't want to do
  // this inline, because that would invalidate our CallGraph::iterator
  // objects. :(
  //
  // Note that it doesn't matter that we are iterating over a non-stable order
  // here to do this, it doesn't matter which order the functions are deleted
  // in.
  array_pod_sort(FunctionsToRemove.begin(), FunctionsToRemove.end());
  FunctionsToRemove.erase(std::unique(FunctionsToRemove.begin(),
                                      FunctionsToRemove.end()),
                          FunctionsToRemove.end());
  for (CallGraphNode *CGN : FunctionsToRemove) {
    delete CG.removeFunctionFromModule(CGN);
    ++NumDeleted;
  }
  return true;
}
