//===- GlobalsModRef.cpp - Simple Mod/Ref Analysis for Globals ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This simple pass provides alias and mod/ref information for global values
// that do not have their address taken, and keeps track of whether functions
// read or write memory (are "pure").  For this simple (but very common) case,
// we can provide pretty accurate and useful information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <set>
using namespace llvm;

#define DEBUG_TYPE "globalsmodref-aa"

STATISTIC(NumNonAddrTakenGlobalVars,
          "Number of global vars without address taken");
STATISTIC(NumNonAddrTakenFunctions,"Number of functions without address taken");
STATISTIC(NumNoMemFunctions, "Number of functions that do not access memory");
STATISTIC(NumReadMemFunctions, "Number of functions that only read memory");
STATISTIC(NumIndirectGlobalVars, "Number of indirect global objects");

namespace {
/// FunctionRecord - One instance of this structure is stored for every
/// function in the program.  Later, the entries for these functions are
/// removed if the function is found to call an external function (in which
/// case we know nothing about it.
struct FunctionRecord {
  /// GlobalInfo - Maintain mod/ref info for all of the globals without
  /// addresses taken that are read or written (transitively) by this
  /// function.
  std::map<const GlobalValue *, unsigned> GlobalInfo;

  /// MayReadAnyGlobal - May read global variables, but it is not known which.
  bool MayReadAnyGlobal;

  unsigned getInfoForGlobal(const GlobalValue *GV) const {
    unsigned Effect = MayReadAnyGlobal ? AliasAnalysis::Ref : 0;
    std::map<const GlobalValue *, unsigned>::const_iterator I =
        GlobalInfo.find(GV);
    if (I != GlobalInfo.end())
      Effect |= I->second;
    return Effect;
  }

  /// FunctionEffect - Capture whether or not this function reads or writes to
  /// ANY memory.  If not, we can do a lot of aggressive analysis on it.
  unsigned FunctionEffect;

  FunctionRecord() : MayReadAnyGlobal(false), FunctionEffect(0) {}
};

/// GlobalsModRef - The actual analysis pass.
class GlobalsModRef : public ModulePass, public AliasAnalysis {
  /// NonAddressTakenGlobals - The globals that do not have their addresses
  /// taken.
  std::set<const GlobalValue *> NonAddressTakenGlobals;

  /// IndirectGlobals - The memory pointed to by this global is known to be
  /// 'owned' by the global.
  std::set<const GlobalValue *> IndirectGlobals;

  /// AllocsForIndirectGlobals - If an instruction allocates memory for an
  /// indirect global, this map indicates which one.
  std::map<const Value *, const GlobalValue *> AllocsForIndirectGlobals;

  /// FunctionInfo - For each function, keep track of what globals are
  /// modified or read.
  std::map<const Function *, FunctionRecord> FunctionInfo;

public:
  static char ID;
  GlobalsModRef() : ModulePass(ID) {
    initializeGlobalsModRefPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    InitializeAliasAnalysis(this, &M.getDataLayout());

    // Find non-addr taken globals.
    AnalyzeGlobals(M);

    // Propagate on CG.
    AnalyzeCallGraph(getAnalysis<CallGraphWrapperPass>().getCallGraph(), M);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AliasAnalysis::getAnalysisUsage(AU);
    AU.addRequired<CallGraphWrapperPass>();
    AU.setPreservesAll(); // Does not transform code
  }

  //------------------------------------------------
  // Implement the AliasAnalysis API
  //
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;
  ModRefResult getModRefInfo(ImmutableCallSite CS,
                             const MemoryLocation &Loc) override;
  ModRefResult getModRefInfo(ImmutableCallSite CS1,
                             ImmutableCallSite CS2) override {
    return AliasAnalysis::getModRefInfo(CS1, CS2);
  }

  /// getModRefBehavior - Return the behavior of the specified function if
  /// called from the specified call site.  The call site may be null in which
  /// case the most generic behavior of this function should be returned.
  ModRefBehavior getModRefBehavior(const Function *F) override {
    ModRefBehavior Min = UnknownModRefBehavior;

    if (FunctionRecord *FR = getFunctionInfo(F)) {
      if (FR->FunctionEffect == 0)
        Min = DoesNotAccessMemory;
      else if ((FR->FunctionEffect & Mod) == 0)
        Min = OnlyReadsMemory;
    }

    return ModRefBehavior(AliasAnalysis::getModRefBehavior(F) & Min);
  }

  /// getModRefBehavior - Return the behavior of the specified function if
  /// called from the specified call site.  The call site may be null in which
  /// case the most generic behavior of this function should be returned.
  ModRefBehavior getModRefBehavior(ImmutableCallSite CS) override {
    ModRefBehavior Min = UnknownModRefBehavior;

    if (const Function *F = CS.getCalledFunction())
      if (FunctionRecord *FR = getFunctionInfo(F)) {
        if (FR->FunctionEffect == 0)
          Min = DoesNotAccessMemory;
        else if ((FR->FunctionEffect & Mod) == 0)
          Min = OnlyReadsMemory;
      }

    return ModRefBehavior(AliasAnalysis::getModRefBehavior(CS) & Min);
  }

  void deleteValue(Value *V) override;
  void addEscapingUse(Use &U) override;

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it
  /// should override this to adjust the this pointer as needed for the
  /// specified pass info.
  void *getAdjustedAnalysisPointer(AnalysisID PI) override {
    if (PI == &AliasAnalysis::ID)
      return (AliasAnalysis *)this;
    return this;
  }

private:
  /// getFunctionInfo - Return the function info for the function, or null if
  /// we don't have anything useful to say about it.
  FunctionRecord *getFunctionInfo(const Function *F) {
    std::map<const Function *, FunctionRecord>::iterator I =
        FunctionInfo.find(F);
    if (I != FunctionInfo.end())
      return &I->second;
    return nullptr;
  }

  void AnalyzeGlobals(Module &M);
  void AnalyzeCallGraph(CallGraph &CG, Module &M);
  bool AnalyzeUsesOfPointer(Value *V, std::vector<Function *> &Readers,
                            std::vector<Function *> &Writers,
                            GlobalValue *OkayStoreDest = nullptr);
  bool AnalyzeIndirectGlobalMemory(GlobalValue *GV);
};
}

char GlobalsModRef::ID = 0;
INITIALIZE_AG_PASS_BEGIN(GlobalsModRef, AliasAnalysis, "globalsmodref-aa",
                         "Simple mod/ref analysis for globals", false, true,
                         false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_AG_PASS_END(GlobalsModRef, AliasAnalysis, "globalsmodref-aa",
                       "Simple mod/ref analysis for globals", false, true,
                       false)

Pass *llvm::createGlobalsModRefPass() { return new GlobalsModRef(); }

/// AnalyzeGlobals - Scan through the users of all of the internal
/// GlobalValue's in the program.  If none of them have their "address taken"
/// (really, their address passed to something nontrivial), record this fact,
/// and record the functions that they are used directly in.
void GlobalsModRef::AnalyzeGlobals(Module &M) {
  std::vector<Function *> Readers, Writers;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasLocalLinkage()) {
      if (!AnalyzeUsesOfPointer(I, Readers, Writers)) {
        // Remember that we are tracking this global.
        NonAddressTakenGlobals.insert(I);
        ++NumNonAddrTakenFunctions;
      }
      Readers.clear();
      Writers.clear();
    }

  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E;
       ++I)
    if (I->hasLocalLinkage()) {
      if (!AnalyzeUsesOfPointer(I, Readers, Writers)) {
        // Remember that we are tracking this global, and the mod/ref fns
        NonAddressTakenGlobals.insert(I);

        for (unsigned i = 0, e = Readers.size(); i != e; ++i)
          FunctionInfo[Readers[i]].GlobalInfo[I] |= Ref;

        if (!I->isConstant()) // No need to keep track of writers to constants
          for (unsigned i = 0, e = Writers.size(); i != e; ++i)
            FunctionInfo[Writers[i]].GlobalInfo[I] |= Mod;
        ++NumNonAddrTakenGlobalVars;

        // If this global holds a pointer type, see if it is an indirect global.
        if (I->getType()->getElementType()->isPointerTy() &&
            AnalyzeIndirectGlobalMemory(I))
          ++NumIndirectGlobalVars;
      }
      Readers.clear();
      Writers.clear();
    }
}

/// AnalyzeUsesOfPointer - Look at all of the users of the specified pointer.
/// If this is used by anything complex (i.e., the address escapes), return
/// true.  Also, while we are at it, keep track of those functions that read and
/// write to the value.
///
/// If OkayStoreDest is non-null, stores into this global are allowed.
bool GlobalsModRef::AnalyzeUsesOfPointer(Value *V,
                                         std::vector<Function *> &Readers,
                                         std::vector<Function *> &Writers,
                                         GlobalValue *OkayStoreDest) {
  if (!V->getType()->isPointerTy())
    return true;

  for (Use &U : V->uses()) {
    User *I = U.getUser();
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      Readers.push_back(LI->getParent()->getParent());
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (V == SI->getOperand(1)) {
        Writers.push_back(SI->getParent()->getParent());
      } else if (SI->getOperand(1) != OkayStoreDest) {
        return true; // Storing the pointer
      }
    } else if (Operator::getOpcode(I) == Instruction::GetElementPtr) {
      if (AnalyzeUsesOfPointer(I, Readers, Writers))
        return true;
    } else if (Operator::getOpcode(I) == Instruction::BitCast) {
      if (AnalyzeUsesOfPointer(I, Readers, Writers, OkayStoreDest))
        return true;
    } else if (auto CS = CallSite(I)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      if (!CS.isCallee(&U)) {
        // Detect calls to free.
        if (isFreeCall(I, TLI))
          Writers.push_back(CS->getParent()->getParent());
        else
          return true; // Argument of an unknown call.
      }
    } else if (ICmpInst *ICI = dyn_cast<ICmpInst>(I)) {
      if (!isa<ConstantPointerNull>(ICI->getOperand(1)))
        return true; // Allow comparison against null.
    } else {
      return true;
    }
  }

  return false;
}

/// AnalyzeIndirectGlobalMemory - We found an non-address-taken global variable
/// which holds a pointer type.  See if the global always points to non-aliased
/// heap memory: that is, all initializers of the globals are allocations, and
/// those allocations have no use other than initialization of the global.
/// Further, all loads out of GV must directly use the memory, not store the
/// pointer somewhere.  If this is true, we consider the memory pointed to by
/// GV to be owned by GV and can disambiguate other pointers from it.
bool GlobalsModRef::AnalyzeIndirectGlobalMemory(GlobalValue *GV) {
  // Keep track of values related to the allocation of the memory, f.e. the
  // value produced by the malloc call and any casts.
  std::vector<Value *> AllocRelatedValues;

  // Walk the user list of the global.  If we find anything other than a direct
  // load or store, bail out.
  for (User *U : GV->users()) {
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // The pointer loaded from the global can only be used in simple ways:
      // we allow addressing of it and loading storing to it.  We do *not* allow
      // storing the loaded pointer somewhere else or passing to a function.
      std::vector<Function *> ReadersWriters;
      if (AnalyzeUsesOfPointer(LI, ReadersWriters, ReadersWriters))
        return false; // Loaded pointer escapes.
      // TODO: Could try some IP mod/ref of the loaded pointer.
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // Storing the global itself.
      if (SI->getOperand(0) == GV)
        return false;

      // If storing the null pointer, ignore it.
      if (isa<ConstantPointerNull>(SI->getOperand(0)))
        continue;

      // Check the value being stored.
      Value *Ptr = GetUnderlyingObject(SI->getOperand(0),
                                       GV->getParent()->getDataLayout());

      if (!isAllocLikeFn(Ptr, TLI))
        return false; // Too hard to analyze.

      // Analyze all uses of the allocation.  If any of them are used in a
      // non-simple way (e.g. stored to another global) bail out.
      std::vector<Function *> ReadersWriters;
      if (AnalyzeUsesOfPointer(Ptr, ReadersWriters, ReadersWriters, GV))
        return false; // Loaded pointer escapes.

      // Remember that this allocation is related to the indirect global.
      AllocRelatedValues.push_back(Ptr);
    } else {
      // Something complex, bail out.
      return false;
    }
  }

  // Okay, this is an indirect global.  Remember all of the allocations for
  // this global in AllocsForIndirectGlobals.
  while (!AllocRelatedValues.empty()) {
    AllocsForIndirectGlobals[AllocRelatedValues.back()] = GV;
    AllocRelatedValues.pop_back();
  }
  IndirectGlobals.insert(GV);
  return true;
}

/// AnalyzeCallGraph - At this point, we know the functions where globals are
/// immediately stored to and read from.  Propagate this information up the call
/// graph to all callers and compute the mod/ref info for all memory for each
/// function.
void GlobalsModRef::AnalyzeCallGraph(CallGraph &CG, Module &M) {
  // We do a bottom-up SCC traversal of the call graph.  In other words, we
  // visit all callees before callers (leaf-first).
  for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    assert(!SCC.empty() && "SCC with no functions?");

    if (!SCC[0]->getFunction()) {
      // Calls externally - can't say anything useful.  Remove any existing
      // function records (may have been created when scanning globals).
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        FunctionInfo.erase(SCC[i]->getFunction());
      continue;
    }

    FunctionRecord &FR = FunctionInfo[SCC[0]->getFunction()];

    bool KnowNothing = false;
    unsigned FunctionEffect = 0;

    // Collect the mod/ref properties due to called functions.  We only compute
    // one mod-ref set.
    for (unsigned i = 0, e = SCC.size(); i != e && !KnowNothing; ++i) {
      Function *F = SCC[i]->getFunction();
      if (!F) {
        KnowNothing = true;
        break;
      }

      if (F->isDeclaration()) {
        // Try to get mod/ref behaviour from function attributes.
        if (F->doesNotAccessMemory()) {
          // Can't do better than that!
        } else if (F->onlyReadsMemory()) {
          FunctionEffect |= Ref;
          if (!F->isIntrinsic())
            // This function might call back into the module and read a global -
            // consider every global as possibly being read by this function.
            FR.MayReadAnyGlobal = true;
        } else {
          FunctionEffect |= ModRef;
          // Can't say anything useful unless it's an intrinsic - they don't
          // read or write global variables of the kind considered here.
          KnowNothing = !F->isIntrinsic();
        }
        continue;
      }

      for (CallGraphNode::iterator CI = SCC[i]->begin(), E = SCC[i]->end();
           CI != E && !KnowNothing; ++CI)
        if (Function *Callee = CI->second->getFunction()) {
          if (FunctionRecord *CalleeFR = getFunctionInfo(Callee)) {
            // Propagate function effect up.
            FunctionEffect |= CalleeFR->FunctionEffect;

            // Incorporate callee's effects on globals into our info.
            for (const auto &G : CalleeFR->GlobalInfo)
              FR.GlobalInfo[G.first] |= G.second;
            FR.MayReadAnyGlobal |= CalleeFR->MayReadAnyGlobal;
          } else {
            // Can't say anything about it.  However, if it is inside our SCC,
            // then nothing needs to be done.
            CallGraphNode *CalleeNode = CG[Callee];
            if (std::find(SCC.begin(), SCC.end(), CalleeNode) == SCC.end())
              KnowNothing = true;
          }
        } else {
          KnowNothing = true;
        }
    }

    // If we can't say anything useful about this SCC, remove all SCC functions
    // from the FunctionInfo map.
    if (KnowNothing) {
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        FunctionInfo.erase(SCC[i]->getFunction());
      continue;
    }

    // Scan the function bodies for explicit loads or stores.
    for (auto *Node : SCC) {
      if (FunctionEffect == ModRef)
        break; // The mod/ref lattice saturates here.
      for (Instruction &I : inst_range(Node->getFunction())) {
        if (FunctionEffect == ModRef)
          break; // The mod/ref lattice saturates here.

        // We handle calls specially because the graph-relevant aspects are
        // handled above.
        if (auto CS = CallSite(&I)) {
          if (isAllocationFn(&I, TLI) || isFreeCall(&I, TLI)) {
            // FIXME: It is completely unclear why this is necessary and not
            // handled by the above graph code.
            FunctionEffect |= ModRef;
          } else if (Function *Callee = CS.getCalledFunction()) {
            // The callgraph doesn't include intrinsic calls.
            if (Callee->isIntrinsic()) {
              ModRefBehavior Behaviour =
                  AliasAnalysis::getModRefBehavior(Callee);
              FunctionEffect |= (Behaviour & ModRef);
            }
          }
          continue;
        }

        // All non-call instructions we use the primary predicates for whether
        // thay read or write memory.
        if (I.mayReadFromMemory())
          FunctionEffect |= Ref;
        if (I.mayWriteToMemory())
          FunctionEffect |= Mod;
      }
    }

    if ((FunctionEffect & Mod) == 0)
      ++NumReadMemFunctions;
    if (FunctionEffect == 0)
      ++NumNoMemFunctions;
    FR.FunctionEffect = FunctionEffect;

    // Finally, now that we know the full effect on this SCC, clone the
    // information to each function in the SCC.
    for (unsigned i = 1, e = SCC.size(); i != e; ++i)
      FunctionInfo[SCC[i]->getFunction()] = FR;
  }
}

/// alias - If one of the pointers is to a global that we are tracking, and the
/// other is some random pointer, we know there cannot be an alias, because the
/// address of the global isn't taken.
AliasResult GlobalsModRef::alias(const MemoryLocation &LocA,
                                 const MemoryLocation &LocB) {
  // Get the base object these pointers point to.
  const Value *UV1 = GetUnderlyingObject(LocA.Ptr, *DL);
  const Value *UV2 = GetUnderlyingObject(LocB.Ptr, *DL);

  // If either of the underlying values is a global, they may be non-addr-taken
  // globals, which we can answer queries about.
  const GlobalValue *GV1 = dyn_cast<GlobalValue>(UV1);
  const GlobalValue *GV2 = dyn_cast<GlobalValue>(UV2);
  if (GV1 || GV2) {
    // If the global's address is taken, pretend we don't know it's a pointer to
    // the global.
    if (GV1 && !NonAddressTakenGlobals.count(GV1))
      GV1 = nullptr;
    if (GV2 && !NonAddressTakenGlobals.count(GV2))
      GV2 = nullptr;

    // If the two pointers are derived from two different non-addr-taken
    // globals, or if one is and the other isn't, we know these can't alias.
    if ((GV1 || GV2) && GV1 != GV2)
      return NoAlias;

    // Otherwise if they are both derived from the same addr-taken global, we
    // can't know the two accesses don't overlap.
  }

  // These pointers may be based on the memory owned by an indirect global.  If
  // so, we may be able to handle this.  First check to see if the base pointer
  // is a direct load from an indirect global.
  GV1 = GV2 = nullptr;
  if (const LoadInst *LI = dyn_cast<LoadInst>(UV1))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getOperand(0)))
      if (IndirectGlobals.count(GV))
        GV1 = GV;
  if (const LoadInst *LI = dyn_cast<LoadInst>(UV2))
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getOperand(0)))
      if (IndirectGlobals.count(GV))
        GV2 = GV;

  // These pointers may also be from an allocation for the indirect global.  If
  // so, also handle them.
  if (AllocsForIndirectGlobals.count(UV1))
    GV1 = AllocsForIndirectGlobals[UV1];
  if (AllocsForIndirectGlobals.count(UV2))
    GV2 = AllocsForIndirectGlobals[UV2];

  // Now that we know whether the two pointers are related to indirect globals,
  // use this to disambiguate the pointers.  If either pointer is based on an
  // indirect global and if they are not both based on the same indirect global,
  // they cannot alias.
  if ((GV1 || GV2) && GV1 != GV2)
    return NoAlias;

  return AliasAnalysis::alias(LocA, LocB);
}

AliasAnalysis::ModRefResult
GlobalsModRef::getModRefInfo(ImmutableCallSite CS, const MemoryLocation &Loc) {
  unsigned Known = ModRef;

  // If we are asking for mod/ref info of a direct call with a pointer to a
  // global we are tracking, return information if we have it.
  const DataLayout &DL = CS.getCaller()->getParent()->getDataLayout();
  if (const GlobalValue *GV =
          dyn_cast<GlobalValue>(GetUnderlyingObject(Loc.Ptr, DL)))
    if (GV->hasLocalLinkage())
      if (const Function *F = CS.getCalledFunction())
        if (NonAddressTakenGlobals.count(GV))
          if (const FunctionRecord *FR = getFunctionInfo(F))
            Known = FR->getInfoForGlobal(GV);

  if (Known == NoModRef)
    return NoModRef; // No need to query other mod/ref analyses
  return ModRefResult(Known & AliasAnalysis::getModRefInfo(CS, Loc));
}

//===----------------------------------------------------------------------===//
// Methods to update the analysis as a result of the client transformation.
//
void GlobalsModRef::deleteValue(Value *V) {
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (NonAddressTakenGlobals.erase(GV)) {
      // This global might be an indirect global.  If so, remove it and remove
      // any AllocRelatedValues for it.
      if (IndirectGlobals.erase(GV)) {
        // Remove any entries in AllocsForIndirectGlobals for this global.
        for (std::map<const Value *, const GlobalValue *>::iterator
                 I = AllocsForIndirectGlobals.begin(),
                 E = AllocsForIndirectGlobals.end();
             I != E;) {
          if (I->second == GV) {
            AllocsForIndirectGlobals.erase(I++);
          } else {
            ++I;
          }
        }
      }
    }
  }

  // Otherwise, if this is an allocation related to an indirect global, remove
  // it.
  AllocsForIndirectGlobals.erase(V);

  AliasAnalysis::deleteValue(V);
}

void GlobalsModRef::addEscapingUse(Use &U) {
  // For the purposes of this analysis, it is conservatively correct to treat
  // a newly escaping value equivalently to a deleted one.  We could perhaps
  // be more precise by processing the new use and attempting to update our
  // saved analysis results to accommodate it.
  deleteValue(U);

  AliasAnalysis::addEscapingUse(U);
}
