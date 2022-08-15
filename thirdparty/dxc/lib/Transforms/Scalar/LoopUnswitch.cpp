//===-- LoopUnswitch.cpp - Hoist loop-invariant conditionals in loop ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops that contain branches on loop-invariant conditions
// to have multiple loops.  For example, it turns the left into the right code:
//
//  for (...)                  if (lic)
//    A                          for (...)
//    if (lic)                     A; B; C
//      B                      else
//    C                          for (...)
//                                 A; C
//
// This can increase the size of the code exponentially (doubling it every time
// a loop is unswitched) so we only unswitch if the resultant code will be
// smaller than a threshold.
//
// This pass expects LICM to be run before it to hoist invariant conditions out
// of the loop, to make the unswitching opportunity obvious.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <map>
#include <set>
using namespace llvm;

#define DEBUG_TYPE "loop-unswitch"

STATISTIC(NumBranches, "Number of branches unswitched");
STATISTIC(NumSwitches, "Number of switches unswitched");
STATISTIC(NumSelects , "Number of selects unswitched");
STATISTIC(NumTrivial , "Number of unswitches that are trivial");
STATISTIC(NumSimplify, "Number of simplifications of unswitched code");
STATISTIC(TotalInsts,  "Total number of instructions analyzed");

// The specific value of 100 here was chosen based only on intuition and a
// few specific examples.
#if 0 // HLSL Change Starts - option pending
static cl::opt<unsigned>
Threshold("loop-unswitch-threshold", cl::desc("Max loop size to unswitch"),
          cl::init(100), cl::Hidden);
#else
static const unsigned Threshold = 100;
#endif

namespace {

  class LUAnalysisCache {

    typedef DenseMap<const SwitchInst*, SmallPtrSet<const Value *, 8> >
      UnswitchedValsMap;

    typedef UnswitchedValsMap::iterator UnswitchedValsIt;

    struct LoopProperties {
      unsigned CanBeUnswitchedCount;
      unsigned WasUnswitchedCount;
      unsigned SizeEstimation;
      UnswitchedValsMap UnswitchedVals;
    };

    // Here we use std::map instead of DenseMap, since we need to keep valid
    // LoopProperties pointer for current loop for better performance.
    typedef std::map<const Loop*, LoopProperties> LoopPropsMap;
    typedef LoopPropsMap::iterator LoopPropsMapIt;

    LoopPropsMap LoopsProperties;
    UnswitchedValsMap *CurLoopInstructions;
    LoopProperties *CurrentLoopProperties;

    // A loop unswitching with an estimated cost above this threshold
    // is not performed. MaxSize is turned into unswitching quota for
    // the current loop, and reduced correspondingly, though note that
    // the quota is returned by releaseMemory() when the loop has been
    // processed, so that MaxSize will return to its previous
    // value. So in most cases MaxSize will equal the Threshold flag
    // when a new loop is processed. An exception to that is that
    // MaxSize will have a smaller value while processing nested loops
    // that were introduced due to loop unswitching of an outer loop.
    //
    // FIXME: The way that MaxSize works is subtle and depends on the
    // pass manager processing loops and calling releaseMemory() in a
    // specific order. It would be good to find a more straightforward
    // way of doing what MaxSize does.
    unsigned MaxSize;

  public:
    LUAnalysisCache()
        : CurLoopInstructions(nullptr), CurrentLoopProperties(nullptr),
          MaxSize(Threshold) {}

    // Analyze loop. Check its size, calculate is it possible to unswitch
    // it. Returns true if we can unswitch this loop.
    bool countLoop(const Loop *L, const TargetTransformInfo &TTI,
                   AssumptionCache *AC);

    // Clean all data related to given loop.
    void forgetLoop(const Loop *L);

    // Mark case value as unswitched.
    // Since SI instruction can be partly unswitched, in order to avoid
    // extra unswitching in cloned loops keep track all unswitched values.
    void setUnswitched(const SwitchInst *SI, const Value *V);

    // Check was this case value unswitched before or not.
    bool isUnswitched(const SwitchInst *SI, const Value *V);

    // Returns true if another unswitching could be done within the cost
    // threshold.
    bool CostAllowsUnswitching();

    // Clone all loop-unswitch related loop properties.
    // Redistribute unswitching quotas.
    // Note, that new loop data is stored inside the VMap.
    void cloneData(const Loop *NewLoop, const Loop *OldLoop,
                   const ValueToValueMapTy &VMap);
  };

  class LoopUnswitch : public LoopPass {
    LoopInfo *LI;  // Loop information
    LPPassManager *LPM;
    AssumptionCache *AC;

    // LoopProcessWorklist - Used to check if second loop needs processing
    // after RewriteLoopBodyWithConditionConstant rewrites first loop.
    std::vector<Loop*> LoopProcessWorklist;

    LUAnalysisCache BranchesInfo;

    bool OptimizeForSize;
    bool redoLoop;

    Loop *currentLoop;
    DominatorTree *DT;
    BasicBlock *loopHeader;
    BasicBlock *loopPreheader;

    // LoopBlocks contains all of the basic blocks of the loop, including the
    // preheader of the loop, the body of the loop, and the exit blocks of the
    // loop, in that order.
    std::vector<BasicBlock*> LoopBlocks;
    // NewBlocks contained cloned copy of basic blocks from LoopBlocks.
    std::vector<BasicBlock*> NewBlocks;

  public:
    static char ID; // Pass ID, replacement for typeid
    explicit LoopUnswitch(bool Os = false) :
      LoopPass(ID), OptimizeForSize(Os), redoLoop(false),
      currentLoop(nullptr), DT(nullptr), loopHeader(nullptr),
      loopPreheader(nullptr) {
        initializeLoopUnswitchPass(*PassRegistry::getPassRegistry());
      }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;
    bool processCurrentLoop();

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
    }

  private:

    void releaseMemory() override {
      BranchesInfo.forgetLoop(currentLoop);
    }

    void initLoopData() {
      loopHeader = currentLoop->getHeader();
      loopPreheader = currentLoop->getLoopPreheader();
    }

    /// Split all of the edges from inside the loop to their exit blocks.
    /// Update the appropriate Phi nodes as we do so.
    void SplitExitEdges(Loop *L, const SmallVectorImpl<BasicBlock *> &ExitBlocks);

    bool UnswitchIfProfitable(Value *LoopCond, Constant *Val,
                              TerminatorInst *TI = nullptr);
    void UnswitchTrivialCondition(Loop *L, Value *Cond, Constant *Val,
                                  BasicBlock *ExitBlock, TerminatorInst *TI);
    void UnswitchNontrivialCondition(Value *LIC, Constant *OnVal, Loop *L,
                                     TerminatorInst *TI);

    void RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC,
                                              Constant *Val, bool isEqual);

    void EmitPreheaderBranchOnCondition(Value *LIC, Constant *Val,
                                        BasicBlock *TrueDest,
                                        BasicBlock *FalseDest,
                                        Instruction *InsertPt,
                                        TerminatorInst *TI);

    void SimplifyCode(std::vector<Instruction*> &Worklist, Loop *L);
    bool IsTrivialUnswitchCondition(Value *Cond, Constant **Val = nullptr,
                                    BasicBlock **LoopExit = nullptr);

  };
}

// Analyze loop. Check its size, calculate is it possible to unswitch
// it. Returns true if we can unswitch this loop.
bool LUAnalysisCache::countLoop(const Loop *L, const TargetTransformInfo &TTI,
                                AssumptionCache *AC) {

  LoopPropsMapIt PropsIt;
  bool Inserted;
  std::tie(PropsIt, Inserted) =
      LoopsProperties.insert(std::make_pair(L, LoopProperties()));

  LoopProperties &Props = PropsIt->second;

  if (Inserted) {
    // New loop.

    // Limit the number of instructions to avoid causing significant code
    // expansion, and the number of basic blocks, to avoid loops with
    // large numbers of branches which cause loop unswitching to go crazy.
    // This is a very ad-hoc heuristic.

    SmallPtrSet<const Value *, 32> EphValues;
    CodeMetrics::collectEphemeralValues(L, AC, EphValues);

    // FIXME: This is overly conservative because it does not take into
    // consideration code simplification opportunities and code that can
    // be shared by the resultant unswitched loops.
    CodeMetrics Metrics;
    for (Loop::block_iterator I = L->block_begin(), E = L->block_end(); I != E;
         ++I)
      Metrics.analyzeBasicBlock(*I, TTI, EphValues);

    Props.SizeEstimation = Metrics.NumInsts;
    Props.CanBeUnswitchedCount = MaxSize / (Props.SizeEstimation);
    Props.WasUnswitchedCount = 0;
    MaxSize -= Props.SizeEstimation * Props.CanBeUnswitchedCount;

    if (Metrics.notDuplicatable) {
      DEBUG(dbgs() << "NOT unswitching loop %"
                   << L->getHeader()->getName() << ", contents cannot be "
                   << "duplicated!\n");
      return false;
    }
  }

  // Be careful. This links are good only before new loop addition.
  CurrentLoopProperties = &Props;
  CurLoopInstructions = &Props.UnswitchedVals;

  return true;
}

// Clean all data related to given loop.
void LUAnalysisCache::forgetLoop(const Loop *L) {

  LoopPropsMapIt LIt = LoopsProperties.find(L);

  if (LIt != LoopsProperties.end()) {
    LoopProperties &Props = LIt->second;
    MaxSize += (Props.CanBeUnswitchedCount + Props.WasUnswitchedCount) *
               Props.SizeEstimation;
    LoopsProperties.erase(LIt);
  }

  CurrentLoopProperties = nullptr;
  CurLoopInstructions = nullptr;
}

// Mark case value as unswitched.
// Since SI instruction can be partly unswitched, in order to avoid
// extra unswitching in cloned loops keep track all unswitched values.
void LUAnalysisCache::setUnswitched(const SwitchInst *SI, const Value *V) {
  (*CurLoopInstructions)[SI].insert(V);
}

// Check was this case value unswitched before or not.
bool LUAnalysisCache::isUnswitched(const SwitchInst *SI, const Value *V) {
  return (*CurLoopInstructions)[SI].count(V);
}

bool LUAnalysisCache::CostAllowsUnswitching() {
  return CurrentLoopProperties->CanBeUnswitchedCount > 0;
}

// Clone all loop-unswitch related loop properties.
// Redistribute unswitching quotas.
// Note, that new loop data is stored inside the VMap.
void LUAnalysisCache::cloneData(const Loop *NewLoop, const Loop *OldLoop,
                                const ValueToValueMapTy &VMap) {

  LoopProperties &NewLoopProps = LoopsProperties[NewLoop];
  LoopProperties &OldLoopProps = *CurrentLoopProperties;
  UnswitchedValsMap &Insts = OldLoopProps.UnswitchedVals;

  // Reallocate "can-be-unswitched quota"

  --OldLoopProps.CanBeUnswitchedCount;
  ++OldLoopProps.WasUnswitchedCount;
  NewLoopProps.WasUnswitchedCount = 0;
  unsigned Quota = OldLoopProps.CanBeUnswitchedCount;
  NewLoopProps.CanBeUnswitchedCount = Quota / 2;
  OldLoopProps.CanBeUnswitchedCount = Quota - Quota / 2;

  NewLoopProps.SizeEstimation = OldLoopProps.SizeEstimation;

  // Clone unswitched values info:
  // for new loop switches we clone info about values that was
  // already unswitched and has redundant successors.
  for (UnswitchedValsIt I = Insts.begin(); I != Insts.end(); ++I) {
    const SwitchInst *OldInst = I->first;
    Value *NewI = VMap.lookup(OldInst);
    const SwitchInst *NewInst = cast_or_null<SwitchInst>(NewI);
    assert(NewInst && "All instructions that are in SrcBB must be in VMap.");

    NewLoopProps.UnswitchedVals[NewInst] = OldLoopProps.UnswitchedVals[OldInst];
  }
}

char LoopUnswitch::ID = 0;
INITIALIZE_PASS_BEGIN(LoopUnswitch, "loop-unswitch", "Unswitch loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopUnswitch, "loop-unswitch", "Unswitch loops",
                      false, false)

Pass *llvm::createLoopUnswitchPass(bool Os) {
  return new LoopUnswitch(Os);
}

/// FindLIVLoopCondition - Cond is a condition that occurs in L.  If it is
/// invariant in the loop, or has an invariant piece, return the invariant.
/// Otherwise, return null.
static Value *FindLIVLoopCondition(Value *Cond, Loop *L, bool &Changed) {

  // We started analyze new instruction, increment scanned instructions counter.
  ++TotalInsts;

  // We can never unswitch on vector conditions.
  if (Cond->getType()->isVectorTy())
    return nullptr;

  // Constants should be folded, not unswitched on!
  if (isa<Constant>(Cond)) return nullptr;

  // TODO: Handle: br (VARIANT|INVARIANT).

  // Hoist simple values out.
  if (L->makeLoopInvariant(Cond, Changed))
    return Cond;

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Cond))
    if (BO->getOpcode() == Instruction::And ||
        BO->getOpcode() == Instruction::Or) {
      // If either the left or right side is invariant, we can unswitch on this,
      // which will cause the branch to go away in one loop and the condition to
      // simplify in the other one.
      if (Value *LHS = FindLIVLoopCondition(BO->getOperand(0), L, Changed))
        return LHS;
      if (Value *RHS = FindLIVLoopCondition(BO->getOperand(1), L, Changed))
        return RHS;
    }

  return nullptr;
}

bool LoopUnswitch::runOnLoop(Loop *L, LPPassManager &LPM_Ref) {
  if (skipOptnoneFunction(L))
    return false;

  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(
      *L->getHeader()->getParent());
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  LPM = &LPM_Ref;
  DominatorTreeWrapperPass *DTWP =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTWP ? &DTWP->getDomTree() : nullptr;
  currentLoop = L;
  Function *F = currentLoop->getHeader()->getParent();
  bool Changed = false;
  do {
    assert(currentLoop->isLCSSAForm(*DT));
    redoLoop = false;
    Changed |= processCurrentLoop();
  } while(redoLoop);

  if (Changed) {
    // FIXME: Reconstruct dom info, because it is not preserved properly.
    if (DT)
      DT->recalculate(*F);
  }
  return Changed;
}

/// processCurrentLoop - Do actual work and unswitch loop if possible
/// and profitable.
bool LoopUnswitch::processCurrentLoop() {
  bool Changed = false;

  initLoopData();

  // If LoopSimplify was unable to form a preheader, don't do any unswitching.
  if (!loopPreheader)
    return false;

  // Loops with indirectbr cannot be cloned.
  if (!currentLoop->isSafeToClone())
    return false;

  // Without dedicated exits, splitting the exit edge may fail.
  if (!currentLoop->hasDedicatedExits())
    return false;

  LLVMContext &Context = loopHeader->getContext();

  // Probably we reach the quota of branches for this loop. If so
  // stop unswitching.
  if (!BranchesInfo.countLoop(
          currentLoop, getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
                           *currentLoop->getHeader()->getParent()),
          AC))
    return false;

  // Loop over all of the basic blocks in the loop.  If we find an interior
  // block that is branching on a loop-invariant condition, we can unswitch this
  // loop.
  for (Loop::block_iterator I = currentLoop->block_begin(),
         E = currentLoop->block_end(); I != E; ++I) {
    TerminatorInst *TI = (*I)->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      // If this isn't branching on an invariant condition, we can't unswitch
      // it.
      if (BI->isConditional()) {
        // See if this, or some part of it, is loop invariant.  If so, we can
        // unswitch on it if we desire.
        Value *LoopCond = FindLIVLoopCondition(BI->getCondition(),
                                               currentLoop, Changed);
        if (LoopCond &&
            UnswitchIfProfitable(LoopCond, ConstantInt::getTrue(Context), TI)) {
          ++NumBranches;
          return true;
        }
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      Value *LoopCond = FindLIVLoopCondition(SI->getCondition(),
                                             currentLoop, Changed);
      unsigned NumCases = SI->getNumCases();
      if (LoopCond && NumCases) {
        // Find a value to unswitch on:
        // FIXME: this should chose the most expensive case!
        // FIXME: scan for a case with a non-critical edge?
        Constant *UnswitchVal = nullptr;

        // Do not process same value again and again.
        // At this point we have some cases already unswitched and
        // some not yet unswitched. Let's find the first not yet unswitched one.
        for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
             i != e; ++i) {
          Constant *UnswitchValCandidate = i.getCaseValue();
          if (!BranchesInfo.isUnswitched(SI, UnswitchValCandidate)) {
            UnswitchVal = UnswitchValCandidate;
            break;
          }
        }

        if (!UnswitchVal)
          continue;

        if (UnswitchIfProfitable(LoopCond, UnswitchVal)) {
          ++NumSwitches;
          return true;
        }
      }
    }

    // Scan the instructions to check for unswitchable values.
    for (BasicBlock::iterator BBI = (*I)->begin(), E = (*I)->end();
         BBI != E; ++BBI)
      if (SelectInst *SI = dyn_cast<SelectInst>(BBI)) {
        Value *LoopCond = FindLIVLoopCondition(SI->getCondition(),
                                               currentLoop, Changed);
        if (LoopCond && UnswitchIfProfitable(LoopCond,
                                             ConstantInt::getTrue(Context))) {
          ++NumSelects;
          return true;
        }
      }
  }
  return Changed;
}

/// isTrivialLoopExitBlock - Check to see if all paths from BB exit the
/// loop with no side effects (including infinite loops).
///
/// If true, we return true and set ExitBB to the block we
/// exit through.
///
static bool isTrivialLoopExitBlockHelper(Loop *L, BasicBlock *BB,
                                         BasicBlock *&ExitBB,
                                         std::set<BasicBlock*> &Visited) {
  if (!Visited.insert(BB).second) {
    // Already visited. Without more analysis, this could indicate an infinite
    // loop.
    return false;
  }
  if (!L->contains(BB)) {
    // Otherwise, this is a loop exit, this is fine so long as this is the
    // first exit.
    if (ExitBB) return false;
    ExitBB = BB;
    return true;
  }

  // Otherwise, this is an unvisited intra-loop node.  Check all successors.
  for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI) {
    // Check to see if the successor is a trivial loop exit.
    if (!isTrivialLoopExitBlockHelper(L, *SI, ExitBB, Visited))
      return false;
  }

  // Okay, everything after this looks good, check to make sure that this block
  // doesn't include any side effects.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (I->mayHaveSideEffects())
      return false;

  return true;
}

/// isTrivialLoopExitBlock - Return true if the specified block unconditionally
/// leads to an exit from the specified loop, and has no side-effects in the
/// process.  If so, return the block that is exited to, otherwise return null.
static BasicBlock *isTrivialLoopExitBlock(Loop *L, BasicBlock *BB) {
  std::set<BasicBlock*> Visited;
  Visited.insert(L->getHeader());  // Branches to header make infinite loops.
  BasicBlock *ExitBB = nullptr;
  if (isTrivialLoopExitBlockHelper(L, BB, ExitBB, Visited))
    return ExitBB;
  return nullptr;
}

/// IsTrivialUnswitchCondition - Check to see if this unswitch condition is
/// trivial: that is, that the condition controls whether or not the loop does
/// anything at all.  If this is a trivial condition, unswitching produces no
/// code duplications (equivalently, it produces a simpler loop and a new empty
/// loop, which gets deleted).
///
/// If this is a trivial condition, return true, otherwise return false.  When
/// returning true, this sets Cond and Val to the condition that controls the
/// trivial condition: when Cond dynamically equals Val, the loop is known to
/// exit.  Finally, this sets LoopExit to the BB that the loop exits to when
/// Cond == Val.
///
bool LoopUnswitch::IsTrivialUnswitchCondition(Value *Cond, Constant **Val,
                                       BasicBlock **LoopExit) {
  BasicBlock *Header = currentLoop->getHeader();
  TerminatorInst *HeaderTerm = Header->getTerminator();
  LLVMContext &Context = Header->getContext();

  BasicBlock *LoopExitBB = nullptr;
  if (BranchInst *BI = dyn_cast<BranchInst>(HeaderTerm)) {
    // If the header block doesn't end with a conditional branch on Cond, we
    // can't handle it.
    if (!BI->isConditional() || BI->getCondition() != Cond)
      return false;

    // Check to see if a successor of the branch is guaranteed to
    // exit through a unique exit block without having any
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.
    if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop,
                                             BI->getSuccessor(0)))) {
      if (Val) *Val = ConstantInt::getTrue(Context);
    } else if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop,
                                                    BI->getSuccessor(1)))) {
      if (Val) *Val = ConstantInt::getFalse(Context);
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(HeaderTerm)) {
    // If this isn't a switch on Cond, we can't handle it.
    if (SI->getCondition() != Cond) return false;

    // Check to see if a successor of the switch is guaranteed to go to the
    // latch block or exit through a one exit block without having any
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.
    // Note that we can't trivially unswitch on the default case or
    // on already unswitched cases.
    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
         i != e; ++i) {
      BasicBlock *LoopExitCandidate;
      if ((LoopExitCandidate = isTrivialLoopExitBlock(currentLoop,
                                               i.getCaseSuccessor()))) {
        // Okay, we found a trivial case, remember the value that is trivial.
        ConstantInt *CaseVal = i.getCaseValue();

        // Check that it was not unswitched before, since already unswitched
        // trivial vals are looks trivial too.
        if (BranchesInfo.isUnswitched(SI, CaseVal))
          continue;
        LoopExitBB = LoopExitCandidate;
        if (Val) *Val = CaseVal;
        break;
      }
    }
  }

  // If we didn't find a single unique LoopExit block, or if the loop exit block
  // contains phi nodes, this isn't trivial.
  if (!LoopExitBB || isa<PHINode>(LoopExitBB->begin()))
    return false;   // Can't handle this.

  if (LoopExit) *LoopExit = LoopExitBB;

  // We already know that nothing uses any scalar values defined inside of this
  // loop.  As such, we just have to check to see if this loop will execute any
  // side-effecting instructions (e.g. stores, calls, volatile loads) in the
  // part of the loop that the code *would* execute.  We already checked the
  // tail, check the header now.
  for (BasicBlock::iterator I = Header->begin(), E = Header->end(); I != E; ++I)
    if (I->mayHaveSideEffects())
      return false;
  return true;
}

/// UnswitchIfProfitable - We have found that we can unswitch currentLoop when
/// LoopCond == Val to simplify the loop.  If we decide that this is profitable,
/// unswitch the loop, reprocess the pieces, then return true.
bool LoopUnswitch::UnswitchIfProfitable(Value *LoopCond, Constant *Val,
                                        TerminatorInst *TI) {
  Function *F = loopHeader->getParent();
  Constant *CondVal = nullptr;
  BasicBlock *ExitBlock = nullptr;

  if (IsTrivialUnswitchCondition(LoopCond, &CondVal, &ExitBlock)) {
    // If the condition is trivial, always unswitch. There is no code growth
    // for this case.
    UnswitchTrivialCondition(currentLoop, LoopCond, CondVal, ExitBlock, TI);
    return true;
  }

  // Check to see if it would be profitable to unswitch current loop.
  if (!BranchesInfo.CostAllowsUnswitching()) {
    DEBUG(dbgs() << "NOT unswitching loop %"
                 << currentLoop->getHeader()->getName()
                 << " at non-trivial condition '" << *Val
                 << "' == " << *LoopCond << "\n"
                 << ". Cost too high.\n");
    return false;
  }

  // Do not do non-trivial unswitch while optimizing for size.
  if (OptimizeForSize || F->hasFnAttribute(Attribute::OptimizeForSize))
    return false;

  UnswitchNontrivialCondition(LoopCond, Val, currentLoop, TI);
  return true;
}

/// CloneLoop - Recursively clone the specified loop and all of its children,
/// mapping the blocks with the specified map.
static Loop *CloneLoop(Loop *L, Loop *PL, ValueToValueMapTy &VM,
                       LoopInfo *LI, LPPassManager *LPM) {
  Loop *New = new Loop();
  LPM->insertLoop(New, PL);

  // Add all of the blocks in L to the new loop.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    if (LI->getLoopFor(*I) == L)
      New->addBasicBlockToLoop(cast<BasicBlock>(VM[*I]), *LI);

  // Add all of the subloops to the new loop.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    CloneLoop(*I, New, VM, LI, LPM);

  return New;
}

static void copyMetadata(Instruction *DstInst, const Instruction *SrcInst,
                         bool Swapped) {
  if (!SrcInst || !SrcInst->hasMetadata())
    return;

  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  SrcInst->getAllMetadata(MDs);
  for (auto &MD : MDs) {
    switch (MD.first) {
    default:
      break;
    case LLVMContext::MD_prof:
      if (Swapped && MD.second->getNumOperands() == 3 &&
          isa<MDString>(MD.second->getOperand(0))) {
        MDString *MDName = cast<MDString>(MD.second->getOperand(0));
        if (MDName->getString() == "branch_weights") {
          auto *ValT = cast_or_null<ConstantAsMetadata>(
                           MD.second->getOperand(1))->getValue();
          auto *ValF = cast_or_null<ConstantAsMetadata>(
                           MD.second->getOperand(2))->getValue();
          assert(ValT && ValF && "Invalid Operands of branch_weights");
          auto NewMD =
              MDBuilder(DstInst->getParent()->getContext())
                  .createBranchWeights(cast<ConstantInt>(ValF)->getZExtValue(),
                                       cast<ConstantInt>(ValT)->getZExtValue());
          MD.second = NewMD;
        }
      }
      // fallthrough.
    case LLVMContext::MD_dbg:
      DstInst->setMetadata(MD.first, MD.second);
    }
  }
}

/// EmitPreheaderBranchOnCondition - Emit a conditional branch on two values
/// if LIC == Val, branch to TrueDst, otherwise branch to FalseDest.  Insert the
/// code immediately before InsertPt.
void LoopUnswitch::EmitPreheaderBranchOnCondition(Value *LIC, Constant *Val,
                                                  BasicBlock *TrueDest,
                                                  BasicBlock *FalseDest,
                                                  Instruction *InsertPt,
                                                  TerminatorInst *TI) {
  // Insert a conditional branch on LIC to the two preheaders.  The original
  // code is the true version and the new code is the false version.
  Value *BranchVal = LIC;
  bool Swapped = false;
  if (!isa<ConstantInt>(Val) ||
      Val->getType() != Type::getInt1Ty(LIC->getContext()))
    BranchVal = new ICmpInst(InsertPt, ICmpInst::ICMP_EQ, LIC, Val);
  else if (Val != ConstantInt::getTrue(Val->getContext())) {
    // We want to enter the new loop when the condition is true.
    std::swap(TrueDest, FalseDest);
    Swapped = true;
  }

  // Insert the new branch.
  BranchInst *BI = BranchInst::Create(TrueDest, FalseDest, BranchVal, InsertPt);
  copyMetadata(BI, TI, Swapped);

  // If either edge is critical, split it. This helps preserve LoopSimplify
  // form for enclosing loops.
  auto Options = CriticalEdgeSplittingOptions(DT, LI).setPreserveLCSSA();
  SplitCriticalEdge(BI, 0, Options);
  SplitCriticalEdge(BI, 1, Options);
}

/// UnswitchTrivialCondition - Given a loop that has a trivial unswitchable
/// condition in it (a cond branch from its header block to its latch block,
/// where the path through the loop that doesn't execute its body has no
/// side-effects), unswitch it.  This doesn't involve any code duplication, just
/// moving the conditional branch outside of the loop and updating loop info.
void LoopUnswitch::UnswitchTrivialCondition(Loop *L, Value *Cond, Constant *Val,
                                            BasicBlock *ExitBlock,
                                            TerminatorInst *TI) {
  DEBUG(dbgs() << "loop-unswitch: Trivial-Unswitch loop %"
               << loopHeader->getName() << " [" << L->getBlocks().size()
               << " blocks] in Function "
               << L->getHeader()->getParent()->getName() << " on cond: " << *Val
               << " == " << *Cond << "\n");

  // First step, split the preheader, so that we know that there is a safe place
  // to insert the conditional branch.  We will change loopPreheader to have a
  // conditional branch on Cond.
  BasicBlock *NewPH = SplitEdge(loopPreheader, loopHeader, DT, LI);

  // Now that we have a place to insert the conditional branch, create a place
  // to branch to: this is the exit block out of the loop that we should
  // short-circuit to.

  // Split this block now, so that the loop maintains its exit block, and so
  // that the jump from the preheader can execute the contents of the exit block
  // without actually branching to it (the exit block should be dominated by the
  // loop header, not the preheader).
  assert(!L->contains(ExitBlock) && "Exit block is in the loop?");
  BasicBlock *NewExit = SplitBlock(ExitBlock, ExitBlock->begin(), DT, LI);

  // Okay, now we have a position to branch from and a position to branch to,
  // insert the new conditional branch.
  EmitPreheaderBranchOnCondition(Cond, Val, NewExit, NewPH,
                                 loopPreheader->getTerminator(), TI);
  LPM->deleteSimpleAnalysisValue(loopPreheader->getTerminator(), L);
  loopPreheader->getTerminator()->eraseFromParent();

  // We need to reprocess this loop, it could be unswitched again.
  redoLoop = true;

  // Now that we know that the loop is never entered when this condition is a
  // particular value, rewrite the loop with this info.  We know that this will
  // at least eliminate the old branch.
  RewriteLoopBodyWithConditionConstant(L, Cond, Val, false);
  ++NumTrivial;
}

/// SplitExitEdges - Split all of the edges from inside the loop to their exit
/// blocks.  Update the appropriate Phi nodes as we do so.
void LoopUnswitch::SplitExitEdges(Loop *L,
                               const SmallVectorImpl<BasicBlock *> &ExitBlocks){

  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBlock = ExitBlocks[i];
    SmallVector<BasicBlock *, 4> Preds(pred_begin(ExitBlock),
                                       pred_end(ExitBlock));

    // Although SplitBlockPredecessors doesn't preserve loop-simplify in
    // general, if we call it on all predecessors of all exits then it does.
    SplitBlockPredecessors(ExitBlock, Preds, ".us-lcssa",
                           /*AliasAnalysis*/ nullptr, DT, LI,
                           /*PreserveLCSSA*/ true);
  }
}

/// UnswitchNontrivialCondition - We determined that the loop is profitable
/// to unswitch when LIC equal Val.  Split it into loop versions and test the
/// condition outside of either loop.  Return the loops created as Out1/Out2.
void LoopUnswitch::UnswitchNontrivialCondition(Value *LIC, Constant *Val,
                                               Loop *L, TerminatorInst *TI) {
  Function *F = loopHeader->getParent();
  DEBUG(dbgs() << "loop-unswitch: Unswitching loop %"
        << loopHeader->getName() << " [" << L->getBlocks().size()
        << " blocks] in Function " << F->getName()
        << " when '" << *Val << "' == " << *LIC << "\n");

  if (ScalarEvolution *SE = getAnalysisIfAvailable<ScalarEvolution>())
    SE->forgetLoop(L);

  LoopBlocks.clear();
  NewBlocks.clear();

  // First step, split the preheader and exit blocks, and add these blocks to
  // the LoopBlocks list.
  BasicBlock *NewPreheader = SplitEdge(loopPreheader, loopHeader, DT, LI);
  LoopBlocks.push_back(NewPreheader);

  // We want the loop to come after the preheader, but before the exit blocks.
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  // Split all of the edges from inside the loop to their exit blocks.  Update
  // the appropriate Phi nodes as we do so.
  SplitExitEdges(L, ExitBlocks);

  // The exit blocks may have been changed due to edge splitting, recompute.
  ExitBlocks.clear();
  L->getUniqueExitBlocks(ExitBlocks);

  // Add exit blocks to the loop blocks.
  LoopBlocks.insert(LoopBlocks.end(), ExitBlocks.begin(), ExitBlocks.end());

  // Next step, clone all of the basic blocks that make up the loop (including
  // the loop preheader and exit blocks), keeping track of the mapping between
  // the instructions and blocks.
  NewBlocks.reserve(LoopBlocks.size());
  ValueToValueMapTy VMap;
  for (unsigned i = 0, e = LoopBlocks.size(); i != e; ++i) {
    BasicBlock *NewBB = CloneBasicBlock(LoopBlocks[i], VMap, ".us", F);

    NewBlocks.push_back(NewBB);
    VMap[LoopBlocks[i]] = NewBB;  // Keep the BB mapping.
    LPM->cloneBasicBlockSimpleAnalysis(LoopBlocks[i], NewBB, L);
  }

  // Splice the newly inserted blocks into the function right before the
  // original preheader.
  F->getBasicBlockList().splice(NewPreheader, F->getBasicBlockList(),
                                NewBlocks[0], F->end());

  // FIXME: We could register any cloned assumptions instead of clearing the
  // whole function's cache.
  AC->clear();

  // Now we create the new Loop object for the versioned loop.
  Loop *NewLoop = CloneLoop(L, L->getParentLoop(), VMap, LI, LPM);

  // Recalculate unswitching quota, inherit simplified switches info for NewBB,
  // Probably clone more loop-unswitch related loop properties.
  BranchesInfo.cloneData(NewLoop, L, VMap);

  Loop *ParentLoop = L->getParentLoop();
  if (ParentLoop) {
    // Make sure to add the cloned preheader and exit blocks to the parent loop
    // as well.
    ParentLoop->addBasicBlockToLoop(NewBlocks[0], *LI);
  }

  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *NewExit = cast<BasicBlock>(VMap[ExitBlocks[i]]);
    // The new exit block should be in the same loop as the old one.
    if (Loop *ExitBBLoop = LI->getLoopFor(ExitBlocks[i]))
      ExitBBLoop->addBasicBlockToLoop(NewExit, *LI);

    assert(NewExit->getTerminator()->getNumSuccessors() == 1 &&
           "Exit block should have been split to have one successor!");
    BasicBlock *ExitSucc = NewExit->getTerminator()->getSuccessor(0);

    // If the successor of the exit block had PHI nodes, add an entry for
    // NewExit.
    for (BasicBlock::iterator I = ExitSucc->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      Value *V = PN->getIncomingValueForBlock(ExitBlocks[i]);
      ValueToValueMapTy::iterator It = VMap.find(V);
      if (It != VMap.end()) V = It->second;
      PN->addIncoming(V, NewExit);
    }

    if (LandingPadInst *LPad = NewExit->getLandingPadInst()) {
      PHINode *PN = PHINode::Create(LPad->getType(), 0, "",
                                    ExitSucc->getFirstInsertionPt());

      for (pred_iterator I = pred_begin(ExitSucc), E = pred_end(ExitSucc);
           I != E; ++I) {
        BasicBlock *BB = *I;
        LandingPadInst *LPI = BB->getLandingPadInst();
        LPI->replaceAllUsesWith(PN);
        PN->addIncoming(LPI, BB);
      }
    }
  }

  // Rewrite the code to refer to itself.
  for (unsigned i = 0, e = NewBlocks.size(); i != e; ++i)
    for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
      RemapInstruction(I, VMap,RF_NoModuleLevelChanges|RF_IgnoreMissingEntries);

  // Rewrite the original preheader to select between versions of the loop.
  BranchInst *OldBR = cast<BranchInst>(loopPreheader->getTerminator());
  assert(OldBR->isUnconditional() && OldBR->getSuccessor(0) == LoopBlocks[0] &&
         "Preheader splitting did not work correctly!");

  // Emit the new branch that selects between the two versions of this loop.
  EmitPreheaderBranchOnCondition(LIC, Val, NewBlocks[0], LoopBlocks[0], OldBR,
                                 TI);
  LPM->deleteSimpleAnalysisValue(OldBR, L);
  OldBR->eraseFromParent();

  LoopProcessWorklist.push_back(NewLoop);
  redoLoop = true;

  // Keep a WeakVH holding onto LIC.  If the first call to RewriteLoopBody
  // deletes the instruction (for example by simplifying a PHI that feeds into
  // the condition that we're unswitching on), we don't rewrite the second
  // iteration.
  WeakVH LICHandle(LIC);

  // Now we rewrite the original code to know that the condition is true and the
  // new code to know that the condition is false.
  RewriteLoopBodyWithConditionConstant(L, LIC, Val, false);

  // It's possible that simplifying one loop could cause the other to be
  // changed to another value or a constant.  If its a constant, don't simplify
  // it.
  if (!LoopProcessWorklist.empty() && LoopProcessWorklist.back() == NewLoop &&
      LICHandle && !isa<Constant>(LICHandle))
    RewriteLoopBodyWithConditionConstant(NewLoop, LICHandle, Val, true);
}

/// RemoveFromWorklist - Remove all instances of I from the worklist vector
/// specified.
static void RemoveFromWorklist(Instruction *I,
                               std::vector<Instruction*> &Worklist) {

  Worklist.erase(std::remove(Worklist.begin(), Worklist.end(), I),
                 Worklist.end());
}

/// ReplaceUsesOfWith - When we find that I really equals V, remove I from the
/// program, replacing all uses with V and update the worklist.
static void ReplaceUsesOfWith(Instruction *I, Value *V,
                              std::vector<Instruction*> &Worklist,
                              Loop *L, LPPassManager *LPM) {
  DEBUG(dbgs() << "Replace with '" << *V << "': " << *I);

  // Add uses to the worklist, which may be dead now.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Instruction *Use = dyn_cast<Instruction>(I->getOperand(i)))
      Worklist.push_back(Use);

  // Add users to the worklist which may be simplified now.
  for (User *U : I->users())
    Worklist.push_back(cast<Instruction>(U));
  LPM->deleteSimpleAnalysisValue(I, L);
  RemoveFromWorklist(I, Worklist);
  I->replaceAllUsesWith(V);
  I->eraseFromParent();
  ++NumSimplify;
}

// RewriteLoopBodyWithConditionConstant - We know either that the value LIC has
// the value specified by Val in the specified loop, or we know it does NOT have
// that value.  Rewrite any uses of LIC or of properties correlated to it.
void LoopUnswitch::RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC,
                                                        Constant *Val,
                                                        bool IsEqual) {
  assert(!isa<Constant>(LIC) && "Why are we unswitching on a constant?");

  // FIXME: Support correlated properties, like:
  //  for (...)
  //    if (li1 < li2)
  //      ...
  //    if (li1 > li2)
  //      ...

  // FOLD boolean conditions (X|LIC), (X&LIC).  Fold conditional branches,
  // selects, switches.
  std::vector<Instruction*> Worklist;
  LLVMContext &Context = Val->getContext();

  // If we know that LIC == Val, or that LIC == NotVal, just replace uses of LIC
  // in the loop with the appropriate one directly.
  if (IsEqual || (isa<ConstantInt>(Val) &&
      Val->getType()->isIntegerTy(1))) {
    Value *Replacement;
    if (IsEqual)
      Replacement = Val;
    else
      Replacement = ConstantInt::get(Type::getInt1Ty(Val->getContext()),
                                     !cast<ConstantInt>(Val)->getZExtValue());

    for (User *U : LIC->users()) {
      Instruction *UI = dyn_cast<Instruction>(U);
      if (!UI || !L->contains(UI))
        continue;
      Worklist.push_back(UI);
    }

    for (std::vector<Instruction*>::iterator UI = Worklist.begin(),
         UE = Worklist.end(); UI != UE; ++UI)
      (*UI)->replaceUsesOfWith(LIC, Replacement);

    SimplifyCode(Worklist, L);
    return;
  }

  // Otherwise, we don't know the precise value of LIC, but we do know that it
  // is certainly NOT "Val".  As such, simplify any uses in the loop that we
  // can.  This case occurs when we unswitch switch statements.
  for (User *U : LIC->users()) {
    Instruction *UI = dyn_cast<Instruction>(U);
    if (!UI || !L->contains(UI))
      continue;

    Worklist.push_back(UI);

    // TODO: We could do other simplifications, for example, turning
    // 'icmp eq LIC, Val' -> false.

    // If we know that LIC is not Val, use this info to simplify code.
    SwitchInst *SI = dyn_cast<SwitchInst>(UI);
    if (!SI || !isa<ConstantInt>(Val)) continue;

    SwitchInst::CaseIt DeadCase = SI->findCaseValue(cast<ConstantInt>(Val));
    // Default case is live for multiple values.
    if (DeadCase == SI->case_default()) continue;

    // Found a dead case value.  Don't remove PHI nodes in the
    // successor if they become single-entry, those PHI nodes may
    // be in the Users list.

    BasicBlock *Switch = SI->getParent();
    BasicBlock *SISucc = DeadCase.getCaseSuccessor();
    BasicBlock *Latch = L->getLoopLatch();

    BranchesInfo.setUnswitched(SI, Val);

    if (!SI->findCaseDest(SISucc)) continue;  // Edge is critical.
    // If the DeadCase successor dominates the loop latch, then the
    // transformation isn't safe since it will delete the sole predecessor edge
    // to the latch.
    if (Latch && DT->dominates(SISucc, Latch))
      continue;

    // FIXME: This is a hack.  We need to keep the successor around
    // and hooked up so as to preserve the loop structure, because
    // trying to update it is complicated.  So instead we preserve the
    // loop structure and put the block on a dead code path.
    SplitEdge(Switch, SISucc, DT, LI);
    // Compute the successors instead of relying on the return value
    // of SplitEdge, since it may have split the switch successor
    // after PHI nodes.
    BasicBlock *NewSISucc = DeadCase.getCaseSuccessor();
    BasicBlock *OldSISucc = *succ_begin(NewSISucc);
    // Create an "unreachable" destination.
    BasicBlock *Abort = BasicBlock::Create(Context, "us-unreachable",
                                           Switch->getParent(),
                                           OldSISucc);
    new UnreachableInst(Context, Abort);
    // Force the new case destination to branch to the "unreachable"
    // block while maintaining a (dead) CFG edge to the old block.
    NewSISucc->getTerminator()->eraseFromParent();
    BranchInst::Create(Abort, OldSISucc,
                       ConstantInt::getTrue(Context), NewSISucc);
    // Release the PHI operands for this edge.
    for (BasicBlock::iterator II = NewSISucc->begin();
         PHINode *PN = dyn_cast<PHINode>(II); ++II)
      PN->setIncomingValue(PN->getBasicBlockIndex(Switch),
                           UndefValue::get(PN->getType()));
    // Tell the domtree about the new block. We don't fully update the
    // domtree here -- instead we force it to do a full recomputation
    // after the pass is complete -- but we do need to inform it of
    // new blocks.
    if (DT)
      DT->addNewBlock(Abort, NewSISucc);
  }

  SimplifyCode(Worklist, L);
}

/// SimplifyCode - Okay, now that we have simplified some instructions in the
/// loop, walk over it and constant prop, dce, and fold control flow where
/// possible.  Note that this is effectively a very simple loop-structure-aware
/// optimizer.  During processing of this loop, L could very well be deleted, so
/// it must not be used.
///
/// FIXME: When the loop optimizer is more mature, separate this out to a new
/// pass.
///
void LoopUnswitch::SimplifyCode(std::vector<Instruction*> &Worklist, Loop *L) {
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  while (!Worklist.empty()) {
    Instruction *I = Worklist.back();
    Worklist.pop_back();

    // Simple DCE.
    if (isInstructionTriviallyDead(I)) {
      DEBUG(dbgs() << "Remove dead instruction '" << *I);

      // Add uses to the worklist, which may be dead now.
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *Use = dyn_cast<Instruction>(I->getOperand(i)))
          Worklist.push_back(Use);
      LPM->deleteSimpleAnalysisValue(I, L);
      RemoveFromWorklist(I, Worklist);
      I->eraseFromParent();
      ++NumSimplify;
      continue;
    }

    // See if instruction simplification can hack this up.  This is common for
    // things like "select false, X, Y" after unswitching made the condition be
    // 'false'.  TODO: update the domtree properly so we can pass it here.
    if (Value *V = SimplifyInstruction(I, DL))
      if (LI->replacementPreservesLCSSAForm(I, V)) {
        ReplaceUsesOfWith(I, V, Worklist, L, LPM);
        continue;
      }

    // Special case hacks that appear commonly in unswitched code.
    if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
      if (BI->isUnconditional()) {
        // If BI's parent is the only pred of the successor, fold the two blocks
        // together.
        BasicBlock *Pred = BI->getParent();
        BasicBlock *Succ = BI->getSuccessor(0);
        BasicBlock *SinglePred = Succ->getSinglePredecessor();
        if (!SinglePred) continue;  // Nothing to do.
        assert(SinglePred == Pred && "CFG broken");

        DEBUG(dbgs() << "Merging blocks: " << Pred->getName() << " <- "
              << Succ->getName() << "\n");

        // Resolve any single entry PHI nodes in Succ.
        while (PHINode *PN = dyn_cast<PHINode>(Succ->begin()))
          ReplaceUsesOfWith(PN, PN->getIncomingValue(0), Worklist, L, LPM);

        // If Succ has any successors with PHI nodes, update them to have
        // entries coming from Pred instead of Succ.
        Succ->replaceAllUsesWith(Pred);

        // Move all of the successor contents from Succ to Pred.
        Pred->getInstList().splice(BI, Succ->getInstList(), Succ->begin(),
                                   Succ->end());
        LPM->deleteSimpleAnalysisValue(BI, L);
        BI->eraseFromParent();
        RemoveFromWorklist(BI, Worklist);

        // Remove Succ from the loop tree.
        LI->removeBlock(Succ);
        LPM->deleteSimpleAnalysisValue(Succ, L);
        Succ->eraseFromParent();
        ++NumSimplify;
        continue;
      }

      continue;
    }
  }
}
