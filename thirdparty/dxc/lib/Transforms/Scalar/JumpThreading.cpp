//===- JumpThreading.cpp - Thread control through conditional blocks ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Jump Threading pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

#define DEBUG_TYPE "jump-threading"

STATISTIC(NumThreads, "Number of jumps threaded");
STATISTIC(NumFolds,   "Number of terminators folded");
STATISTIC(NumDupes,   "Number of branch blocks duplicated to eliminate phi");

#if 0 // HLSL Change Starts - option pending
static cl::opt<unsigned>
BBDuplicateThreshold("jump-threading-threshold",
          cl::desc("Max block size to duplicate for jump threading"),
          cl::init(6), cl::Hidden);
#else
static const unsigned BBDuplicateThreshold = 6;
#endif // HLSL Change Ends

namespace {
  // These are at global scope so static functions can use them too.
  typedef SmallVectorImpl<std::pair<Constant*, BasicBlock*> > PredValueInfo;
  typedef SmallVector<std::pair<Constant*, BasicBlock*>, 8> PredValueInfoTy;

  // This is used to keep track of what kind of constant we're currently hoping
  // to find.
  enum ConstantPreference {
    WantInteger,
    WantBlockAddress
  };

  /// This pass performs 'jump threading', which looks at blocks that have
  /// multiple predecessors and multiple successors.  If one or more of the
  /// predecessors of the block can be proven to always jump to one of the
  /// successors, we forward the edge from the predecessor to the successor by
  /// duplicating the contents of this block.
  ///
  /// An example of when this can occur is code like this:
  ///
  ///   if () { ...
  ///     X = 4;
  ///   }
  ///   if (X < 3) {
  ///
  /// In this case, the unconditional branch at the end of the first if can be
  /// revectored to the false side of the second if.
  ///
  class JumpThreading : public FunctionPass {
    TargetLibraryInfo *TLI;
    LazyValueInfo *LVI;
#ifdef NDEBUG
    SmallPtrSet<BasicBlock*, 16> LoopHeaders;
#else
    SmallSet<AssertingVH<BasicBlock>, 16> LoopHeaders;
#endif
    DenseSet<std::pair<Value*, BasicBlock*> > RecursionSet;

    unsigned BBDupThreshold;

    // RAII helper for updating the recursion stack.
    struct RecursionSetRemover {
      DenseSet<std::pair<Value*, BasicBlock*> > &TheSet;
      std::pair<Value*, BasicBlock*> ThePair;

      RecursionSetRemover(DenseSet<std::pair<Value*, BasicBlock*> > &S,
                          std::pair<Value*, BasicBlock*> P)
        : TheSet(S), ThePair(P) { }

      ~RecursionSetRemover() {
        TheSet.erase(ThePair);
      }
    };
  public:
    static char ID; // Pass identification
    JumpThreading(int T = -1) : FunctionPass(ID) {
      BBDupThreshold = (T == -1) ? BBDuplicateThreshold : unsigned(T);
      initializeJumpThreadingPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LazyValueInfo>();
      AU.addPreserved<LazyValueInfo>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }

    void FindLoopHeaders(Function &F);
    bool ProcessBlock(BasicBlock *BB);
    bool ThreadEdge(BasicBlock *BB, const SmallVectorImpl<BasicBlock*> &PredBBs,
                    BasicBlock *SuccBB);
    bool DuplicateCondBranchOnPHIIntoPred(BasicBlock *BB,
                                  const SmallVectorImpl<BasicBlock *> &PredBBs);

    bool ComputeValueKnownInPredecessors(Value *V, BasicBlock *BB,
                                         PredValueInfo &Result,
                                         ConstantPreference Preference,
                                         Instruction *CxtI = nullptr);
    bool ProcessThreadableEdges(Value *Cond, BasicBlock *BB,
                                ConstantPreference Preference,
                                Instruction *CxtI = nullptr);

    bool ProcessBranchOnPHI(PHINode *PN);
    bool ProcessBranchOnXOR(BinaryOperator *BO);

    bool SimplifyPartiallyRedundantLoad(LoadInst *LI);
    bool TryToUnfoldSelect(CmpInst *CondCmp, BasicBlock *BB);
  };
}

char JumpThreading::ID = 0;
INITIALIZE_PASS_BEGIN(JumpThreading, "jump-threading",
                "Jump Threading", false, false)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfo)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(JumpThreading, "jump-threading",
                "Jump Threading", false, false)

// Public interface to the Jump Threading pass
FunctionPass *llvm::createJumpThreadingPass(int Threshold) { return new JumpThreading(Threshold); }

/// runOnFunction - Top level algorithm.
///
bool JumpThreading::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  DEBUG(dbgs() << "Jump threading on function '" << F.getName() << "'\n");
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  LVI = &getAnalysis<LazyValueInfo>();

  // Remove unreachable blocks from function as they may result in infinite
  // loop. We do threading if we found something profitable. Jump threading a
  // branch can create other opportunities. If these opportunities form a cycle
  // i.e. if any jump treading is undoing previous threading in the path, then
  // we will loop forever. We take care of this issue by not jump threading for
  // back edges. This works for normal cases but not for unreachable blocks as
  // they may have cycle with no back edge.
  removeUnreachableBlocks(F);

  FindLoopHeaders(F);

  bool Changed, EverChanged = false;
  do {
    Changed = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = I;
      // Thread all of the branches we can over this block.
      while (ProcessBlock(BB))
        Changed = true;

      ++I;

      // If the block is trivially dead, zap it.  This eliminates the successor
      // edges which simplifies the CFG.
      if ((pred_empty(BB)
          // HLSL change begin - delete self loop.
          || BB->getSinglePredecessor() == BB
          // HLSL change end.
          ) &&
          BB != &BB->getParent()->getEntryBlock()) {
        DEBUG(dbgs() << "  JT: Deleting dead block '" << BB->getName()
              << "' with terminator: " << *BB->getTerminator() << '\n');
        LoopHeaders.erase(BB);
        LVI->eraseBlock(BB);
        DeleteDeadBlock(BB);
        Changed = true;
        continue;
      }

      BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());

      // Can't thread an unconditional jump, but if the block is "almost
      // empty", we can replace uses of it with uses of the successor and make
      // this dead.
      if (BI && BI->isUnconditional() &&
          BB != &BB->getParent()->getEntryBlock() &&
          // If the terminator is the only non-phi instruction, try to nuke it.
          BB->getFirstNonPHIOrDbg()->isTerminator()) {
        // Since TryToSimplifyUncondBranchFromEmptyBlock may delete the
        // block, we have to make sure it isn't in the LoopHeaders set.  We
        // reinsert afterward if needed.
        bool ErasedFromLoopHeaders = LoopHeaders.erase(BB);
        BasicBlock *Succ = BI->getSuccessor(0);

        // FIXME: It is always conservatively correct to drop the info
        // for a block even if it doesn't get erased.  This isn't totally
        // awesome, but it allows us to use AssertingVH to prevent nasty
        // dangling pointer issues within LazyValueInfo.
        LVI->eraseBlock(BB);
        if (TryToSimplifyUncondBranchFromEmptyBlock(BB)) {
          Changed = true;
          // If we deleted BB and BB was the header of a loop, then the
          // successor is now the header of the loop.
          BB = Succ;
        }

        if (ErasedFromLoopHeaders)
          LoopHeaders.insert(BB);
      }
    }
    EverChanged |= Changed;
  } while (Changed);

  LoopHeaders.clear();
  return EverChanged;
}

/// getJumpThreadDuplicationCost - Return the cost of duplicating this block to
/// thread across it. Stop scanning the block when passing the threshold.
static unsigned getJumpThreadDuplicationCost(const BasicBlock *BB,
                                             unsigned Threshold) {
  /// Ignore PHI nodes, these will be flattened when duplication happens.
  BasicBlock::const_iterator I = BB->getFirstNonPHI();

  // FIXME: THREADING will delete values that are just used to compute the
  // branch, so they shouldn't count against the duplication cost.

  // Sum up the cost of each instruction until we get to the terminator.  Don't
  // include the terminator because the copy won't include it.
  unsigned Size = 0;
  for (; !isa<TerminatorInst>(I); ++I) {

    // Stop scanning the block if we've reached the threshold.
    if (Size > Threshold)
      return Size;

    // Debugger intrinsics don't incur code size.
    if (isa<DbgInfoIntrinsic>(I)) continue;

    // If this is a pointer->pointer bitcast, it is free.
    if (isa<BitCastInst>(I) && I->getType()->isPointerTy())
      continue;

    // All other instructions count for at least one unit.
    ++Size;

    // Calls are more expensive.  If they are non-intrinsic calls, we model them
    // as having cost of 4.  If they are a non-vector intrinsic, we model them
    // as having cost of 2 total, and if they are a vector intrinsic, we model
    // them as having cost 1.
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      if (CI->cannotDuplicate())
        // Blocks with NoDuplicate are modelled as having infinite cost, so they
        // are never duplicated.
        return ~0U;
      else if (!isa<IntrinsicInst>(CI))
        Size += 3;
      else if (!CI->getType()->isVectorTy())
        Size += 1;
    }
  }

  // Threading through a switch statement is particularly profitable.  If this
  // block ends in a switch, decrease its cost to make it more likely to happen.
  if (isa<SwitchInst>(I))
    Size = Size > 6 ? Size-6 : 0;

  // The same holds for indirect branches, but slightly more so.
  if (isa<IndirectBrInst>(I))
    Size = Size > 8 ? Size-8 : 0;

  return Size;
}

/// FindLoopHeaders - We do not want jump threading to turn proper loop
/// structures into irreducible loops.  Doing this breaks up the loop nesting
/// hierarchy and pessimizes later transformations.  To prevent this from
/// happening, we first have to find the loop headers.  Here we approximate this
/// by finding targets of backedges in the CFG.
///
/// Note that there definitely are cases when we want to allow threading of
/// edges across a loop header.  For example, threading a jump from outside the
/// loop (the preheader) to an exit block of the loop is definitely profitable.
/// It is also almost always profitable to thread backedges from within the loop
/// to exit blocks, and is often profitable to thread backedges to other blocks
/// within the loop (forming a nested loop).  This simple analysis is not rich
/// enough to track all of these properties and keep it up-to-date as the CFG
/// mutates, so we don't allow any of these transformations.
///
void JumpThreading::FindLoopHeaders(Function &F) {
  SmallVector<std::pair<const BasicBlock*,const BasicBlock*>, 32> Edges;
  FindFunctionBackedges(F, Edges);

  for (unsigned i = 0, e = Edges.size(); i != e; ++i)
    LoopHeaders.insert(const_cast<BasicBlock*>(Edges[i].second));
}

/// getKnownConstant - Helper method to determine if we can thread over a
/// terminator with the given value as its condition, and if so what value to
/// use for that. What kind of value this is depends on whether we want an
/// integer or a block address, but an undef is always accepted.
/// Returns null if Val is null or not an appropriate constant.
static Constant *getKnownConstant(Value *Val, ConstantPreference Preference) {
  if (!Val)
    return nullptr;

  // Undef is "known" enough.
  if (UndefValue *U = dyn_cast<UndefValue>(Val))
    return U;

  if (Preference == WantBlockAddress)
    return dyn_cast<BlockAddress>(Val->stripPointerCasts());

  return dyn_cast<ConstantInt>(Val);
}

/// ComputeValueKnownInPredecessors - Given a basic block BB and a value V, see
/// if we can infer that the value is a known ConstantInt/BlockAddress or undef
/// in any of our predecessors.  If so, return the known list of value and pred
/// BB in the result vector.
///
/// This returns true if there were any known values.
///
bool JumpThreading::
ComputeValueKnownInPredecessors(Value *V, BasicBlock *BB, PredValueInfo &Result,
                                ConstantPreference Preference,
                                Instruction *CxtI) {
  // This method walks up use-def chains recursively.  Because of this, we could
  // get into an infinite loop going around loops in the use-def chain.  To
  // prevent this, keep track of what (value, block) pairs we've already visited
  // and terminate the search if we loop back to them
  if (!RecursionSet.insert(std::make_pair(V, BB)).second)
    return false;

  // An RAII help to remove this pair from the recursion set once the recursion
  // stack pops back out again.
  RecursionSetRemover remover(RecursionSet, std::make_pair(V, BB));

  // If V is a constant, then it is known in all predecessors.
  if (Constant *KC = getKnownConstant(V, Preference)) {
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      Result.push_back(std::make_pair(KC, *PI));

    return true;
  }

  // If V is a non-instruction value, or an instruction in a different block,
  // then it can't be derived from a PHI.
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || I->getParent() != BB) {

    // Okay, if this is a live-in value, see if it has a known value at the end
    // of any of our predecessors.
    //
    // FIXME: This should be an edge property, not a block end property.
    /// TODO: Per PR2563, we could infer value range information about a
    /// predecessor based on its terminator.
    //
    // FIXME: change this to use the more-rich 'getPredicateOnEdge' method if
    // "I" is a non-local compare-with-a-constant instruction.  This would be
    // able to handle value inequalities better, for example if the compare is
    // "X < 4" and "X < 3" is known true but "X < 4" itself is not available.
    // Perhaps getConstantOnEdge should be smart enough to do this?

    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *P = *PI;
      // If the value is known by LazyValueInfo to be a constant in a
      // predecessor, use that information to try to thread this block.
      Constant *PredCst = LVI->getConstantOnEdge(V, P, BB, CxtI);
      if (Constant *KC = getKnownConstant(PredCst, Preference))
        Result.push_back(std::make_pair(KC, P));
    }

    return !Result.empty();
  }

  /// If I is a PHI node, then we know the incoming values for any constants.
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *InVal = PN->getIncomingValue(i);
      if (Constant *KC = getKnownConstant(InVal, Preference)) {
        Result.push_back(std::make_pair(KC, PN->getIncomingBlock(i)));
      } else {
        Constant *CI = LVI->getConstantOnEdge(InVal,
                                              PN->getIncomingBlock(i),
                                              BB, CxtI);
        if (Constant *KC = getKnownConstant(CI, Preference))
          Result.push_back(std::make_pair(KC, PN->getIncomingBlock(i)));
      }
    }

    return !Result.empty();
  }

  PredValueInfoTy LHSVals, RHSVals;

  // Handle some boolean conditions.
  if (I->getType()->getPrimitiveSizeInBits() == 1) {
    assert(Preference == WantInteger && "One-bit non-integer type?");
    // X | true -> true
    // X & false -> false
    if (I->getOpcode() == Instruction::Or ||
        I->getOpcode() == Instruction::And) {
      ComputeValueKnownInPredecessors(I->getOperand(0), BB, LHSVals,
                                      WantInteger, CxtI);
      ComputeValueKnownInPredecessors(I->getOperand(1), BB, RHSVals,
                                      WantInteger, CxtI);

      if (LHSVals.empty() && RHSVals.empty())
        return false;

      ConstantInt *InterestingVal;
      if (I->getOpcode() == Instruction::Or)
        InterestingVal = ConstantInt::getTrue(I->getContext());
      else
        InterestingVal = ConstantInt::getFalse(I->getContext());

      SmallPtrSet<BasicBlock*, 4> LHSKnownBBs;

      // Scan for the sentinel.  If we find an undef, force it to the
      // interesting value: x|undef -> true and x&undef -> false.
      for (unsigned i = 0, e = LHSVals.size(); i != e; ++i)
        if (LHSVals[i].first == InterestingVal ||
            isa<UndefValue>(LHSVals[i].first)) {
          Result.push_back(LHSVals[i]);
          Result.back().first = InterestingVal;
          LHSKnownBBs.insert(LHSVals[i].second);
        }
      for (unsigned i = 0, e = RHSVals.size(); i != e; ++i)
        if (RHSVals[i].first == InterestingVal ||
            isa<UndefValue>(RHSVals[i].first)) {
          // If we already inferred a value for this block on the LHS, don't
          // re-add it.
          if (!LHSKnownBBs.count(RHSVals[i].second)) {
            Result.push_back(RHSVals[i]);
            Result.back().first = InterestingVal;
          }
        }

      return !Result.empty();
    }

    // Handle the NOT form of XOR.
    if (I->getOpcode() == Instruction::Xor &&
        isa<ConstantInt>(I->getOperand(1)) &&
        cast<ConstantInt>(I->getOperand(1))->isOne()) {
      ComputeValueKnownInPredecessors(I->getOperand(0), BB, Result,
                                      WantInteger, CxtI);
      if (Result.empty())
        return false;

      // Invert the known values.
      for (unsigned i = 0, e = Result.size(); i != e; ++i)
        Result[i].first = ConstantExpr::getNot(Result[i].first);

      return true;
    }

  // Try to simplify some other binary operator values.
  } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
    assert(Preference != WantBlockAddress
            && "A binary operator creating a block address?");
    if (ConstantInt *CI = dyn_cast<ConstantInt>(BO->getOperand(1))) {
      PredValueInfoTy LHSVals;
      ComputeValueKnownInPredecessors(BO->getOperand(0), BB, LHSVals,
                                      WantInteger, CxtI);

      // Try to use constant folding to simplify the binary operator.
      for (unsigned i = 0, e = LHSVals.size(); i != e; ++i) {
        Constant *V = LHSVals[i].first;
        Constant *Folded = ConstantExpr::get(BO->getOpcode(), V, CI);

        if (Constant *KC = getKnownConstant(Folded, WantInteger))
          Result.push_back(std::make_pair(KC, LHSVals[i].second));
      }
    }

    return !Result.empty();
  }

  // Handle compare with phi operand, where the PHI is defined in this block.
  if (CmpInst *Cmp = dyn_cast<CmpInst>(I)) {
    assert(Preference == WantInteger && "Compares only produce integers");
    PHINode *PN = dyn_cast<PHINode>(Cmp->getOperand(0));
    if (PN && PN->getParent() == BB) {
      const DataLayout &DL = PN->getModule()->getDataLayout();
      // We can do this simplification if any comparisons fold to true or false.
      // See if any do.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        BasicBlock *PredBB = PN->getIncomingBlock(i);
        Value *LHS = PN->getIncomingValue(i);
        Value *RHS = Cmp->getOperand(1)->DoPHITranslation(BB, PredBB);

        Value *Res = SimplifyCmpInst(Cmp->getPredicate(), LHS, RHS, DL);
        if (!Res) {
          if (!isa<Constant>(RHS))
            continue;

          LazyValueInfo::Tristate
            ResT = LVI->getPredicateOnEdge(Cmp->getPredicate(), LHS,
                                           cast<Constant>(RHS), PredBB, BB,
                                           CxtI ? CxtI : Cmp);
          if (ResT == LazyValueInfo::Unknown)
            continue;
          Res = ConstantInt::get(Type::getInt1Ty(LHS->getContext()), ResT);
        }

        if (Constant *KC = getKnownConstant(Res, WantInteger))
          Result.push_back(std::make_pair(KC, PredBB));
      }

      return !Result.empty();
    }

    // If comparing a live-in value against a constant, see if we know the
    // live-in value on any predecessors.
    if (isa<Constant>(Cmp->getOperand(1)) && Cmp->getType()->isIntegerTy()) {
      if (!isa<Instruction>(Cmp->getOperand(0)) ||
          cast<Instruction>(Cmp->getOperand(0))->getParent() != BB) {
        Constant *RHSCst = cast<Constant>(Cmp->getOperand(1));

        for (pred_iterator PI = pred_begin(BB), E = pred_end(BB);PI != E; ++PI){
          BasicBlock *P = *PI;
          // If the value is known by LazyValueInfo to be a constant in a
          // predecessor, use that information to try to thread this block.
          LazyValueInfo::Tristate Res =
            LVI->getPredicateOnEdge(Cmp->getPredicate(), Cmp->getOperand(0),
                                    RHSCst, P, BB, CxtI ? CxtI : Cmp);
          if (Res == LazyValueInfo::Unknown)
            continue;

          Constant *ResC = ConstantInt::get(Cmp->getType(), Res);
          Result.push_back(std::make_pair(ResC, P));
        }

        return !Result.empty();
      }

      // Try to find a constant value for the LHS of a comparison,
      // and evaluate it statically if we can.
      if (Constant *CmpConst = dyn_cast<Constant>(Cmp->getOperand(1))) {
        PredValueInfoTy LHSVals;
        ComputeValueKnownInPredecessors(I->getOperand(0), BB, LHSVals,
                                        WantInteger, CxtI);

        for (unsigned i = 0, e = LHSVals.size(); i != e; ++i) {
          Constant *V = LHSVals[i].first;
          Constant *Folded = ConstantExpr::getCompare(Cmp->getPredicate(),
                                                      V, CmpConst);
          if (Constant *KC = getKnownConstant(Folded, WantInteger))
            Result.push_back(std::make_pair(KC, LHSVals[i].second));
        }

        return !Result.empty();
      }
    }
  }

  if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
    // Handle select instructions where at least one operand is a known constant
    // and we can figure out the condition value for any predecessor block.
    Constant *TrueVal = getKnownConstant(SI->getTrueValue(), Preference);
    Constant *FalseVal = getKnownConstant(SI->getFalseValue(), Preference);
    PredValueInfoTy Conds;
    if ((TrueVal || FalseVal) &&
        ComputeValueKnownInPredecessors(SI->getCondition(), BB, Conds,
                                        WantInteger, CxtI)) {
      for (unsigned i = 0, e = Conds.size(); i != e; ++i) {
        Constant *Cond = Conds[i].first;

        // Figure out what value to use for the condition.
        bool KnownCond;
        if (ConstantInt *CI = dyn_cast<ConstantInt>(Cond)) {
          // A known boolean.
          KnownCond = CI->isOne();
        } else {
          assert(isa<UndefValue>(Cond) && "Unexpected condition value");
          // Either operand will do, so be sure to pick the one that's a known
          // constant.
          // FIXME: Do this more cleverly if both values are known constants?
          KnownCond = (TrueVal != nullptr);
        }

        // See if the select has a known constant value for this predecessor.
        if (Constant *Val = KnownCond ? TrueVal : FalseVal)
          Result.push_back(std::make_pair(Val, Conds[i].second));
      }

      return !Result.empty();
    }
  }

  // If all else fails, see if LVI can figure out a constant value for us.
  Constant *CI = LVI->getConstant(V, BB, CxtI);
  if (Constant *KC = getKnownConstant(CI, Preference)) {
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      Result.push_back(std::make_pair(KC, *PI));
  }

  return !Result.empty();
}



/// GetBestDestForBranchOnUndef - If we determine that the specified block ends
/// in an undefined jump, decide which block is best to revector to.
///
/// Since we can pick an arbitrary destination, we pick the successor with the
/// fewest predecessors.  This should reduce the in-degree of the others.
///
static unsigned GetBestDestForJumpOnUndef(BasicBlock *BB) {
  TerminatorInst *BBTerm = BB->getTerminator();
  unsigned MinSucc = 0;
  BasicBlock *TestBB = BBTerm->getSuccessor(MinSucc);
  // Compute the successor with the minimum number of predecessors.
  unsigned MinNumPreds = std::distance(pred_begin(TestBB), pred_end(TestBB));
  for (unsigned i = 1, e = BBTerm->getNumSuccessors(); i != e; ++i) {
    TestBB = BBTerm->getSuccessor(i);
    unsigned NumPreds = std::distance(pred_begin(TestBB), pred_end(TestBB));
    if (NumPreds < MinNumPreds) {
      MinSucc = i;
      MinNumPreds = NumPreds;
    }
  }

  return MinSucc;
}

static bool hasAddressTakenAndUsed(BasicBlock *BB) {
  if (!BB->hasAddressTaken()) return false;

  // If the block has its address taken, it may be a tree of dead constants
  // hanging off of it.  These shouldn't keep the block alive.
  BlockAddress *BA = BlockAddress::get(BB);
  BA->removeDeadConstantUsers();
  return !BA->use_empty();
}

/// ProcessBlock - If there are any predecessors whose control can be threaded
/// through to a successor, transform them now.
bool JumpThreading::ProcessBlock(BasicBlock *BB) {
  // If the block is trivially dead, just return and let the caller nuke it.
  // This simplifies other transformations.
  if (pred_empty(BB) &&
      BB != &BB->getParent()->getEntryBlock())
    return false;

  // If this block has a single predecessor, and if that pred has a single
  // successor, merge the blocks.  This encourages recursive jump threading
  // because now the condition in this block can be threaded through
  // predecessors of our predecessor block.
  if (BasicBlock *SinglePred = BB->getSinglePredecessor()) {
    if (SinglePred->getTerminator()->getNumSuccessors() == 1 &&
        SinglePred != BB && !hasAddressTakenAndUsed(BB)) {
      // If SinglePred was a loop header, BB becomes one.
      if (LoopHeaders.erase(SinglePred))
        LoopHeaders.insert(BB);

      LVI->eraseBlock(SinglePred);
      MergeBasicBlockIntoOnlyPred(BB);

      return true;
    }
  }

  // What kind of constant we're looking for.
  ConstantPreference Preference = WantInteger;

  // Look to see if the terminator is a conditional branch, switch or indirect
  // branch, if not we can't thread it.
  Value *Condition;
  Instruction *Terminator = BB->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(Terminator)) {
    // Can't thread an unconditional jump.
    if (BI->isUnconditional()) return false;
    Condition = BI->getCondition();
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(Terminator)) {
    Condition = SI->getCondition();
  } else if (IndirectBrInst *IB = dyn_cast<IndirectBrInst>(Terminator)) {
    // Can't thread indirect branch with no successors.
    if (IB->getNumSuccessors() == 0) return false;
    Condition = IB->getAddress()->stripPointerCasts();
    Preference = WantBlockAddress;
  } else {
    return false; // Must be an invoke.
  }

  // Run constant folding to see if we can reduce the condition to a simple
  // constant.
  if (Instruction *I = dyn_cast<Instruction>(Condition)) {
    Value *SimpleVal =
        ConstantFoldInstruction(I, BB->getModule()->getDataLayout(), TLI);
    if (SimpleVal) {
      I->replaceAllUsesWith(SimpleVal);
      I->eraseFromParent();
      Condition = SimpleVal;
    }
  }

  // If the terminator is branching on an undef, we can pick any of the
  // successors to branch to.  Let GetBestDestForJumpOnUndef decide.
  if (isa<UndefValue>(Condition)) {
    unsigned BestSucc = GetBestDestForJumpOnUndef(BB);

    // Fold the branch/switch.
    TerminatorInst *BBTerm = BB->getTerminator();
    for (unsigned i = 0, e = BBTerm->getNumSuccessors(); i != e; ++i) {
      if (i == BestSucc) continue;
      BBTerm->getSuccessor(i)->removePredecessor(BB, true);
    }

    DEBUG(dbgs() << "  In block '" << BB->getName()
          << "' folding undef terminator: " << *BBTerm << '\n');
    BranchInst::Create(BBTerm->getSuccessor(BestSucc), BBTerm);
    BBTerm->eraseFromParent();
    return true;
  }

  // If the terminator of this block is branching on a constant, simplify the
  // terminator to an unconditional branch.  This can occur due to threading in
  // other blocks.
  if (getKnownConstant(Condition, Preference)) {
    DEBUG(dbgs() << "  In block '" << BB->getName()
          << "' folding terminator: " << *BB->getTerminator() << '\n');
    ++NumFolds;
    ConstantFoldTerminator(BB, true);
    return true;
  }

  Instruction *CondInst = dyn_cast<Instruction>(Condition);

  // All the rest of our checks depend on the condition being an instruction.
  if (!CondInst) {
    // FIXME: Unify this with code below.
    if (ProcessThreadableEdges(Condition, BB, Preference, Terminator))
      return true;
    return false;
  }


  if (CmpInst *CondCmp = dyn_cast<CmpInst>(CondInst)) {
    // If we're branching on a conditional, LVI might be able to determine
    // it's value at the branch instruction.  We only handle comparisons
    // against a constant at this time.
    // TODO: This should be extended to handle switches as well.  
    BranchInst *CondBr = dyn_cast<BranchInst>(BB->getTerminator());
    Constant *CondConst = dyn_cast<Constant>(CondCmp->getOperand(1));
    if (CondBr && CondConst && CondBr->isConditional()) {
      LazyValueInfo::Tristate Ret =
        LVI->getPredicateAt(CondCmp->getPredicate(), CondCmp->getOperand(0),
                            CondConst, CondBr);
      if (Ret != LazyValueInfo::Unknown) {
        unsigned ToRemove = Ret == LazyValueInfo::True ? 1 : 0;
        unsigned ToKeep = Ret == LazyValueInfo::True ? 0 : 1;
        CondBr->getSuccessor(ToRemove)->removePredecessor(BB, true);
        BranchInst::Create(CondBr->getSuccessor(ToKeep), CondBr);
        CondBr->eraseFromParent();
        if (CondCmp->use_empty())
          CondCmp->eraseFromParent();
        else if (CondCmp->getParent() == BB) {
          // If the fact we just learned is true for all uses of the
          // condition, replace it with a constant value
          auto *CI = Ret == LazyValueInfo::True ?
            ConstantInt::getTrue(CondCmp->getType()) :
            ConstantInt::getFalse(CondCmp->getType());
          CondCmp->replaceAllUsesWith(CI);
          CondCmp->eraseFromParent();
        }
        return true;
      }
    }

    if (CondBr && CondConst && TryToUnfoldSelect(CondCmp, BB))
      return true;
  }

  // Check for some cases that are worth simplifying.  Right now we want to look
  // for loads that are used by a switch or by the condition for the branch.  If
  // we see one, check to see if it's partially redundant.  If so, insert a PHI
  // which can then be used to thread the values.
  //
  Value *SimplifyValue = CondInst;
  if (CmpInst *CondCmp = dyn_cast<CmpInst>(SimplifyValue))
    if (isa<Constant>(CondCmp->getOperand(1)))
      SimplifyValue = CondCmp->getOperand(0);

  // TODO: There are other places where load PRE would be profitable, such as
  // more complex comparisons.
  if (LoadInst *LI = dyn_cast<LoadInst>(SimplifyValue))
    if (SimplifyPartiallyRedundantLoad(LI))
      return true;


  // Handle a variety of cases where we are branching on something derived from
  // a PHI node in the current block.  If we can prove that any predecessors
  // compute a predictable value based on a PHI node, thread those predecessors.
  //
  if (ProcessThreadableEdges(CondInst, BB, Preference, Terminator))
    return true;

  // If this is an otherwise-unfoldable branch on a phi node in the current
  // block, see if we can simplify.
  if (PHINode *PN = dyn_cast<PHINode>(CondInst))
    if (PN->getParent() == BB && isa<BranchInst>(BB->getTerminator()))
      return ProcessBranchOnPHI(PN);


  // If this is an otherwise-unfoldable branch on a XOR, see if we can simplify.
  if (CondInst->getOpcode() == Instruction::Xor &&
      CondInst->getParent() == BB && isa<BranchInst>(BB->getTerminator()))
    return ProcessBranchOnXOR(cast<BinaryOperator>(CondInst));


  // TODO: If we have: "br (X > 0)"  and we have a predecessor where we know
  // "(X == 4)", thread through this block.

  return false;
}

/// SimplifyPartiallyRedundantLoad - If LI is an obviously partially redundant
/// load instruction, eliminate it by replacing it with a PHI node.  This is an
/// important optimization that encourages jump threading, and needs to be run
/// interlaced with other jump threading tasks.
bool JumpThreading::SimplifyPartiallyRedundantLoad(LoadInst *LI) {
  // Don't hack volatile/atomic loads.
  if (!LI->isSimple()) return false;

  // If the load is defined in a block with exactly one predecessor, it can't be
  // partially redundant.
  BasicBlock *LoadBB = LI->getParent();
  if (LoadBB->getSinglePredecessor())
    return false;

  // If the load is defined in a landing pad, it can't be partially redundant,
  // because the edges between the invoke and the landing pad cannot have other
  // instructions between them.
  if (LoadBB->isLandingPad())
    return false;

  Value *LoadedPtr = LI->getOperand(0);

  // If the loaded operand is defined in the LoadBB, it can't be available.
  // TODO: Could do simple PHI translation, that would be fun :)
  if (Instruction *PtrOp = dyn_cast<Instruction>(LoadedPtr))
    if (PtrOp->getParent() == LoadBB)
      return false;

  // Scan a few instructions up from the load, to see if it is obviously live at
  // the entry to its block.
  BasicBlock::iterator BBIt = LI;

  if (Value *AvailableVal =
        FindAvailableLoadedValue(LoadedPtr, LoadBB, BBIt, 6)) {
    // If the value if the load is locally available within the block, just use
    // it.  This frequently occurs for reg2mem'd allocas.
    //cerr << "LOAD ELIMINATED:\n" << *BBIt << *LI << "\n";

    // If the returned value is the load itself, replace with an undef. This can
    // only happen in dead loops.
    if (AvailableVal == LI) AvailableVal = UndefValue::get(LI->getType());
    if (AvailableVal->getType() != LI->getType())
      AvailableVal =
          CastInst::CreateBitOrPointerCast(AvailableVal, LI->getType(), "", LI);
    LI->replaceAllUsesWith(AvailableVal);
    LI->eraseFromParent();
    return true;
  }

  // Otherwise, if we scanned the whole block and got to the top of the block,
  // we know the block is locally transparent to the load.  If not, something
  // might clobber its value.
  if (BBIt != LoadBB->begin())
    return false;

  // If all of the loads and stores that feed the value have the same AA tags,
  // then we can propagate them onto any newly inserted loads.
  AAMDNodes AATags;
  LI->getAAMetadata(AATags);

  SmallPtrSet<BasicBlock*, 8> PredsScanned;
  typedef SmallVector<std::pair<BasicBlock*, Value*>, 8> AvailablePredsTy;
  AvailablePredsTy AvailablePreds;
  BasicBlock *OneUnavailablePred = nullptr;

  // If we got here, the loaded value is transparent through to the start of the
  // block.  Check to see if it is available in any of the predecessor blocks.
  for (pred_iterator PI = pred_begin(LoadBB), PE = pred_end(LoadBB);
       PI != PE; ++PI) {
    BasicBlock *PredBB = *PI;

    // If we already scanned this predecessor, skip it.
    if (!PredsScanned.insert(PredBB).second)
      continue;

    // Scan the predecessor to see if the value is available in the pred.
    BBIt = PredBB->end();
    AAMDNodes ThisAATags;
    Value *PredAvailable = FindAvailableLoadedValue(LoadedPtr, PredBB, BBIt, 6,
                                                    nullptr, &ThisAATags);
    if (!PredAvailable) {
      OneUnavailablePred = PredBB;
      continue;
    }

    // If AA tags disagree or are not present, forget about them.
    if (AATags != ThisAATags) AATags = AAMDNodes();

    // If so, this load is partially redundant.  Remember this info so that we
    // can create a PHI node.
    AvailablePreds.push_back(std::make_pair(PredBB, PredAvailable));
  }

  // If the loaded value isn't available in any predecessor, it isn't partially
  // redundant.
  if (AvailablePreds.empty()) return false;

  // Okay, the loaded value is available in at least one (and maybe all!)
  // predecessors.  If the value is unavailable in more than one unique
  // predecessor, we want to insert a merge block for those common predecessors.
  // This ensures that we only have to insert one reload, thus not increasing
  // code size.
  BasicBlock *UnavailablePred = nullptr;

  // If there is exactly one predecessor where the value is unavailable, the
  // already computed 'OneUnavailablePred' block is it.  If it ends in an
  // unconditional branch, we know that it isn't a critical edge.
  if (PredsScanned.size() == AvailablePreds.size()+1 &&
      OneUnavailablePred->getTerminator()->getNumSuccessors() == 1) {
    UnavailablePred = OneUnavailablePred;
  } else if (PredsScanned.size() != AvailablePreds.size()) {
    // Otherwise, we had multiple unavailable predecessors or we had a critical
    // edge from the one.
    SmallVector<BasicBlock*, 8> PredsToSplit;
    SmallPtrSet<BasicBlock*, 8> AvailablePredSet;

    for (unsigned i = 0, e = AvailablePreds.size(); i != e; ++i)
      AvailablePredSet.insert(AvailablePreds[i].first);

    // Add all the unavailable predecessors to the PredsToSplit list.
    for (pred_iterator PI = pred_begin(LoadBB), PE = pred_end(LoadBB);
         PI != PE; ++PI) {
      BasicBlock *P = *PI;
      // If the predecessor is an indirect goto, we can't split the edge.
      if (isa<IndirectBrInst>(P->getTerminator()))
        return false;

      if (!AvailablePredSet.count(P))
        PredsToSplit.push_back(P);
    }

    // Split them out to their own block.
    UnavailablePred =
      SplitBlockPredecessors(LoadBB, PredsToSplit, "thread-pre-split");
  }

  // If the value isn't available in all predecessors, then there will be
  // exactly one where it isn't available.  Insert a load on that edge and add
  // it to the AvailablePreds list.
  if (UnavailablePred) {
    assert(UnavailablePred->getTerminator()->getNumSuccessors() == 1 &&
           "Can't handle critical edge here!");
    LoadInst *NewVal = new LoadInst(LoadedPtr, LI->getName()+".pr", false,
                                 LI->getAlignment(),
                                 UnavailablePred->getTerminator());
    NewVal->setDebugLoc(LI->getDebugLoc());
    if (AATags)
      NewVal->setAAMetadata(AATags);

    AvailablePreds.push_back(std::make_pair(UnavailablePred, NewVal));
  }

  // Now we know that each predecessor of this block has a value in
  // AvailablePreds, sort them for efficient access as we're walking the preds.
  array_pod_sort(AvailablePreds.begin(), AvailablePreds.end());

  // Create a PHI node at the start of the block for the PRE'd load value.
  pred_iterator PB = pred_begin(LoadBB), PE = pred_end(LoadBB);
  PHINode *PN = PHINode::Create(LI->getType(), std::distance(PB, PE), "",
                                LoadBB->begin());
  PN->takeName(LI);
  PN->setDebugLoc(LI->getDebugLoc());

  // Insert new entries into the PHI for each predecessor.  A single block may
  // have multiple entries here.
  for (pred_iterator PI = PB; PI != PE; ++PI) {
    BasicBlock *P = *PI;
    AvailablePredsTy::iterator I =
      std::lower_bound(AvailablePreds.begin(), AvailablePreds.end(),
                       std::make_pair(P, (Value*)nullptr));

    assert(I != AvailablePreds.end() && I->first == P &&
           "Didn't find entry for predecessor!");

    // If we have an available predecessor but it requires casting, insert the
    // cast in the predecessor and use the cast. Note that we have to update the
    // AvailablePreds vector as we go so that all of the PHI entries for this
    // predecessor use the same bitcast.
    Value *&PredV = I->second;
    if (PredV->getType() != LI->getType())
      PredV = CastInst::CreateBitOrPointerCast(PredV, LI->getType(), "",
                                               P->getTerminator());

    PN->addIncoming(PredV, I->first);
  }

  //cerr << "PRE: " << *LI << *PN << "\n";

  LI->replaceAllUsesWith(PN);
  LI->eraseFromParent();

  return true;
}

/// FindMostPopularDest - The specified list contains multiple possible
/// threadable destinations.  Pick the one that occurs the most frequently in
/// the list.
static BasicBlock *
FindMostPopularDest(BasicBlock *BB,
                    const SmallVectorImpl<std::pair<BasicBlock*,
                                  BasicBlock*> > &PredToDestList) {
  assert(!PredToDestList.empty());

  // Determine popularity.  If there are multiple possible destinations, we
  // explicitly choose to ignore 'undef' destinations.  We prefer to thread
  // blocks with known and real destinations to threading undef.  We'll handle
  // them later if interesting.
  DenseMap<BasicBlock*, unsigned> DestPopularity;
  for (unsigned i = 0, e = PredToDestList.size(); i != e; ++i)
    if (PredToDestList[i].second)
      DestPopularity[PredToDestList[i].second]++;

  // Find the most popular dest.
  DenseMap<BasicBlock*, unsigned>::iterator DPI = DestPopularity.begin();
  BasicBlock *MostPopularDest = DPI->first;
  unsigned Popularity = DPI->second;
  SmallVector<BasicBlock*, 4> SamePopularity;

  for (++DPI; DPI != DestPopularity.end(); ++DPI) {
    // If the popularity of this entry isn't higher than the popularity we've
    // seen so far, ignore it.
    if (DPI->second < Popularity)
      ; // ignore.
    else if (DPI->second == Popularity) {
      // If it is the same as what we've seen so far, keep track of it.
      SamePopularity.push_back(DPI->first);
    } else {
      // If it is more popular, remember it.
      SamePopularity.clear();
      MostPopularDest = DPI->first;
      Popularity = DPI->second;
    }
  }

  // Okay, now we know the most popular destination.  If there is more than one
  // destination, we need to determine one.  This is arbitrary, but we need
  // to make a deterministic decision.  Pick the first one that appears in the
  // successor list.
  if (!SamePopularity.empty()) {
    SamePopularity.push_back(MostPopularDest);
    TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0; ; ++i) {
      assert(i != TI->getNumSuccessors() && "Didn't find any successor!");

      if (std::find(SamePopularity.begin(), SamePopularity.end(),
                    TI->getSuccessor(i)) == SamePopularity.end())
        continue;

      MostPopularDest = TI->getSuccessor(i);
      break;
    }
  }

  // Okay, we have finally picked the most popular destination.
  return MostPopularDest;
}

bool JumpThreading::ProcessThreadableEdges(Value *Cond, BasicBlock *BB,
                                           ConstantPreference Preference,
                                           Instruction *CxtI) {
  // If threading this would thread across a loop header, don't even try to
  // thread the edge.
  if (LoopHeaders.count(BB))
    return false;

  PredValueInfoTy PredValues;
  if (!ComputeValueKnownInPredecessors(Cond, BB, PredValues, Preference, CxtI))
    return false;

  assert(!PredValues.empty() &&
         "ComputeValueKnownInPredecessors returned true with no values");

  DEBUG(dbgs() << "IN BB: " << *BB;
        for (unsigned i = 0, e = PredValues.size(); i != e; ++i) {
          dbgs() << "  BB '" << BB->getName() << "': FOUND condition = "
            << *PredValues[i].first
            << " for pred '" << PredValues[i].second->getName() << "'.\n";
        });

  // Decide what we want to thread through.  Convert our list of known values to
  // a list of known destinations for each pred.  This also discards duplicate
  // predecessors and keeps track of the undefined inputs (which are represented
  // as a null dest in the PredToDestList).
  SmallPtrSet<BasicBlock*, 16> SeenPreds;
  SmallVector<std::pair<BasicBlock*, BasicBlock*>, 16> PredToDestList;

  BasicBlock *OnlyDest = nullptr;
  BasicBlock *MultipleDestSentinel = (BasicBlock*)(intptr_t)~0ULL;

  for (unsigned i = 0, e = PredValues.size(); i != e; ++i) {
    BasicBlock *Pred = PredValues[i].second;
    if (!SeenPreds.insert(Pred).second)
      continue;  // Duplicate predecessor entry.

    // If the predecessor ends with an indirect goto, we can't change its
    // destination.
    if (isa<IndirectBrInst>(Pred->getTerminator()))
      continue;

    Constant *Val = PredValues[i].first;

    BasicBlock *DestBB;
    if (isa<UndefValue>(Val))
      DestBB = nullptr;
    else if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
      DestBB = BI->getSuccessor(cast<ConstantInt>(Val)->isZero());
    else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
      DestBB = SI->findCaseValue(cast<ConstantInt>(Val)).getCaseSuccessor();
    } else {
      assert(isa<IndirectBrInst>(BB->getTerminator())
              && "Unexpected terminator");
      DestBB = cast<BlockAddress>(Val)->getBasicBlock();
    }

    // If we have exactly one destination, remember it for efficiency below.
    if (PredToDestList.empty())
      OnlyDest = DestBB;
    else if (OnlyDest != DestBB)
      OnlyDest = MultipleDestSentinel;

    PredToDestList.push_back(std::make_pair(Pred, DestBB));
  }

  // If all edges were unthreadable, we fail.
  if (PredToDestList.empty())
    return false;

  // Determine which is the most common successor.  If we have many inputs and
  // this block is a switch, we want to start by threading the batch that goes
  // to the most popular destination first.  If we only know about one
  // threadable destination (the common case) we can avoid this.
  BasicBlock *MostPopularDest = OnlyDest;

  if (MostPopularDest == MultipleDestSentinel)
    MostPopularDest = FindMostPopularDest(BB, PredToDestList);

  // Now that we know what the most popular destination is, factor all
  // predecessors that will jump to it into a single predecessor.
  SmallVector<BasicBlock*, 16> PredsToFactor;
  for (unsigned i = 0, e = PredToDestList.size(); i != e; ++i)
    if (PredToDestList[i].second == MostPopularDest) {
      BasicBlock *Pred = PredToDestList[i].first;

      // This predecessor may be a switch or something else that has multiple
      // edges to the block.  Factor each of these edges by listing them
      // according to # occurrences in PredsToFactor.
      TerminatorInst *PredTI = Pred->getTerminator();
      for (unsigned i = 0, e = PredTI->getNumSuccessors(); i != e; ++i)
        if (PredTI->getSuccessor(i) == BB)
          PredsToFactor.push_back(Pred);
    }

  // If the threadable edges are branching on an undefined value, we get to pick
  // the destination that these predecessors should get to.
  if (!MostPopularDest)
    MostPopularDest = BB->getTerminator()->
                            getSuccessor(GetBestDestForJumpOnUndef(BB));

  // Ok, try to thread it!
  return ThreadEdge(BB, PredsToFactor, MostPopularDest);
}

/// ProcessBranchOnPHI - We have an otherwise unthreadable conditional branch on
/// a PHI node in the current block.  See if there are any simplifications we
/// can do based on inputs to the phi node.
///
bool JumpThreading::ProcessBranchOnPHI(PHINode *PN) {
  BasicBlock *BB = PN->getParent();

  // TODO: We could make use of this to do it once for blocks with common PHI
  // values.
  SmallVector<BasicBlock*, 1> PredBBs;
  PredBBs.resize(1);

  // If any of the predecessor blocks end in an unconditional branch, we can
  // *duplicate* the conditional branch into that block in order to further
  // encourage jump threading and to eliminate cases where we have branch on a
  // phi of an icmp (branch on icmp is much better).
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *PredBB = PN->getIncomingBlock(i);
    if (BranchInst *PredBr = dyn_cast<BranchInst>(PredBB->getTerminator()))
      if (PredBr->isUnconditional()) {
        PredBBs[0] = PredBB;
        // Try to duplicate BB into PredBB.
        if (DuplicateCondBranchOnPHIIntoPred(BB, PredBBs))
          return true;
      }
  }

  return false;
}

/// ProcessBranchOnXOR - We have an otherwise unthreadable conditional branch on
/// a xor instruction in the current block.  See if there are any
/// simplifications we can do based on inputs to the xor.
///
bool JumpThreading::ProcessBranchOnXOR(BinaryOperator *BO) {
  BasicBlock *BB = BO->getParent();

  // If either the LHS or RHS of the xor is a constant, don't do this
  // optimization.
  if (isa<ConstantInt>(BO->getOperand(0)) ||
      isa<ConstantInt>(BO->getOperand(1)))
    return false;

  // If the first instruction in BB isn't a phi, we won't be able to infer
  // anything special about any particular predecessor.
  if (!isa<PHINode>(BB->front()))
    return false;

  // If we have a xor as the branch input to this block, and we know that the
  // LHS or RHS of the xor in any predecessor is true/false, then we can clone
  // the condition into the predecessor and fix that value to true, saving some
  // logical ops on that path and encouraging other paths to simplify.
  //
  // This copies something like this:
  //
  //  BB:
  //    %X = phi i1 [1],  [%X']
  //    %Y = icmp eq i32 %A, %B
  //    %Z = xor i1 %X, %Y
  //    br i1 %Z, ...
  //
  // Into:
  //  BB':
  //    %Y = icmp ne i32 %A, %B
  //    br i1 %Z, ...

  PredValueInfoTy XorOpValues;
  bool isLHS = true;
  if (!ComputeValueKnownInPredecessors(BO->getOperand(0), BB, XorOpValues,
                                       WantInteger, BO)) {
    assert(XorOpValues.empty());
    if (!ComputeValueKnownInPredecessors(BO->getOperand(1), BB, XorOpValues,
                                         WantInteger, BO))
      return false;
    isLHS = false;
  }

  assert(!XorOpValues.empty() &&
         "ComputeValueKnownInPredecessors returned true with no values");

  // Scan the information to see which is most popular: true or false.  The
  // predecessors can be of the set true, false, or undef.
  unsigned NumTrue = 0, NumFalse = 0;
  for (unsigned i = 0, e = XorOpValues.size(); i != e; ++i) {
    if (isa<UndefValue>(XorOpValues[i].first))
      // Ignore undefs for the count.
      continue;
    if (cast<ConstantInt>(XorOpValues[i].first)->isZero())
      ++NumFalse;
    else
      ++NumTrue;
  }

  // Determine which value to split on, true, false, or undef if neither.
  ConstantInt *SplitVal = nullptr;
  if (NumTrue > NumFalse)
    SplitVal = ConstantInt::getTrue(BB->getContext());
  else if (NumTrue != 0 || NumFalse != 0)
    SplitVal = ConstantInt::getFalse(BB->getContext());

  // Collect all of the blocks that this can be folded into so that we can
  // factor this once and clone it once.
  SmallVector<BasicBlock*, 8> BlocksToFoldInto;
  for (unsigned i = 0, e = XorOpValues.size(); i != e; ++i) {
    if (XorOpValues[i].first != SplitVal &&
        !isa<UndefValue>(XorOpValues[i].first))
      continue;

    BlocksToFoldInto.push_back(XorOpValues[i].second);
  }

  // If we inferred a value for all of the predecessors, then duplication won't
  // help us.  However, we can just replace the LHS or RHS with the constant.
  if (BlocksToFoldInto.size() ==
      cast<PHINode>(BB->front()).getNumIncomingValues()) {
    if (!SplitVal) {
      // If all preds provide undef, just nuke the xor, because it is undef too.
      BO->replaceAllUsesWith(UndefValue::get(BO->getType()));
      BO->eraseFromParent();
    } else if (SplitVal->isZero()) {
      // If all preds provide 0, replace the xor with the other input.
      BO->replaceAllUsesWith(BO->getOperand(isLHS));
      BO->eraseFromParent();
    } else {
      // If all preds provide 1, set the computed value to 1.
      BO->setOperand(!isLHS, SplitVal);
    }

    return true;
  }

  // Try to duplicate BB into PredBB.
  return DuplicateCondBranchOnPHIIntoPred(BB, BlocksToFoldInto);
}


/// AddPHINodeEntriesForMappedBlock - We're adding 'NewPred' as a new
/// predecessor to the PHIBB block.  If it has PHI nodes, add entries for
/// NewPred using the entries from OldPred (suitably mapped).
static void AddPHINodeEntriesForMappedBlock(BasicBlock *PHIBB,
                                            BasicBlock *OldPred,
                                            BasicBlock *NewPred,
                                     DenseMap<Instruction*, Value*> &ValueMap) {
  for (BasicBlock::iterator PNI = PHIBB->begin();
       PHINode *PN = dyn_cast<PHINode>(PNI); ++PNI) {
    // Ok, we have a PHI node.  Figure out what the incoming value was for the
    // DestBlock.
    Value *IV = PN->getIncomingValueForBlock(OldPred);

    // Remap the value if necessary.
    if (Instruction *Inst = dyn_cast<Instruction>(IV)) {
      DenseMap<Instruction*, Value*>::iterator I = ValueMap.find(Inst);
      if (I != ValueMap.end())
        IV = I->second;
    }

    PN->addIncoming(IV, NewPred);
  }
}

/// ThreadEdge - We have decided that it is safe and profitable to factor the
/// blocks in PredBBs to one predecessor, then thread an edge from it to SuccBB
/// across BB.  Transform the IR to reflect this change.
bool JumpThreading::ThreadEdge(BasicBlock *BB,
                               const SmallVectorImpl<BasicBlock*> &PredBBs,
                               BasicBlock *SuccBB) {
  // If threading to the same block as we come from, we would infinite loop.
  if (SuccBB == BB) {
    DEBUG(dbgs() << "  Not threading across BB '" << BB->getName()
          << "' - would thread to self!\n");
    return false;
  }

  // If threading this would thread across a loop header, don't thread the edge.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(dbgs() << "  Not threading across loop header BB '" << BB->getName()
          << "' to dest BB '" << SuccBB->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }

  unsigned JumpThreadCost = getJumpThreadDuplicationCost(BB, BBDupThreshold);
  if (JumpThreadCost > BBDupThreshold) {
    DEBUG(dbgs() << "  Not threading BB '" << BB->getName()
          << "' - Cost is too high: " << JumpThreadCost << "\n");
    return false;
  }

  // And finally, do it!  Start by factoring the predecessors is needed.
  BasicBlock *PredBB;
  if (PredBBs.size() == 1)
    PredBB = PredBBs[0];
  else {
    DEBUG(dbgs() << "  Factoring out " << PredBBs.size()
          << " common predecessors.\n");
    PredBB = SplitBlockPredecessors(BB, PredBBs, ".thr_comm");
  }

  // And finally, do it!
  DEBUG(dbgs() << "  Threading edge from '" << PredBB->getName() << "' to '"
        << SuccBB->getName() << "' with cost: " << JumpThreadCost
        << ", across block:\n    "
        << *BB << "\n");

  LVI->threadEdge(PredBB, BB, SuccBB);

  // We are going to have to map operands from the original BB block to the new
  // copy of the block 'NewBB'.  If there are PHI nodes in BB, evaluate them to
  // account for entry from PredBB.
  DenseMap<Instruction*, Value*> ValueMapping;

  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(),
                                         BB->getName()+".thread",
                                         BB->getParent(), BB);
  NewBB->moveAfter(PredBB);

  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);

  // Clone the non-phi instructions of BB into NewBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; !isa<TerminatorInst>(BI); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getName());
    NewBB->getInstList().push_back(New);
    ValueMapping[BI] = New;

    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i))) {
        DenseMap<Instruction*, Value*>::iterator I = ValueMapping.find(Inst);
        if (I != ValueMapping.end())
          New->setOperand(i, I->second);
      }
  }

  // We didn't copy the terminator from BB over to NewBB, because there is now
  // an unconditional jump to SuccBB.  Insert the unconditional jump.
  BranchInst *NewBI =BranchInst::Create(SuccBB, NewBB);
  NewBI->setDebugLoc(BB->getTerminator()->getDebugLoc());

  // Check to see if SuccBB has PHI nodes. If so, we need to add entries to the
  // PHI nodes for NewBB now.
  AddPHINodeEntriesForMappedBlock(SuccBB, BB, NewBB, ValueMapping);

  // If there were values defined in BB that are used outside the block, then we
  // now have to update all uses of the value to use either the original value,
  // the cloned value, or some PHI derived value.  This can require arbitrary
  // PHI insertion, of which we are prepared to do, clean these up now.
  SSAUpdater SSAUpdate;
  SmallVector<Use*, 16> UsesToRename;
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    // Scan all uses of this instruction to see if it is used outside of its
    // block, and if so, record them in UsesToRename.
    for (Use &U : I->uses()) {
      Instruction *User = cast<Instruction>(U.getUser());
      if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
        if (UserPN->getIncomingBlock(U) == BB)
          continue;
      } else if (User->getParent() == BB)
        continue;

      UsesToRename.push_back(&U);
    }

    // If there are no uses outside the block, we're done with this instruction.
    if (UsesToRename.empty())
      continue;

    DEBUG(dbgs() << "JT: Renaming non-local uses of: " << *I << "\n");

    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I->getType(), I->getName());
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(NewBB, ValueMapping[I]);

    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(dbgs() << "\n");
  }


  // Ok, NewBB is good to go.  Update the terminator of PredBB to jump to
  // NewBB instead of BB.  This eliminates predecessors from BB, which requires
  // us to simplify any PHI nodes in BB.
  TerminatorInst *PredTerm = PredBB->getTerminator();
  for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i)
    if (PredTerm->getSuccessor(i) == BB) {
      BB->removePredecessor(PredBB, true);
      PredTerm->setSuccessor(i, NewBB);
    }

  // At this point, the IR is fully up to date and consistent.  Do a quick scan
  // over the new instructions and zap any that are constants or dead.  This
  // frequently happens because of phi translation.
  SimplifyInstructionsInBlock(NewBB, TLI);

  // Threaded an edge!
  ++NumThreads;
  return true;
}

/// DuplicateCondBranchOnPHIIntoPred - PredBB contains an unconditional branch
/// to BB which contains an i1 PHI node and a conditional branch on that PHI.
/// If we can duplicate the contents of BB up into PredBB do so now, this
/// improves the odds that the branch will be on an analyzable instruction like
/// a compare.
bool JumpThreading::DuplicateCondBranchOnPHIIntoPred(BasicBlock *BB,
                                 const SmallVectorImpl<BasicBlock *> &PredBBs) {
  assert(!PredBBs.empty() && "Can't handle an empty set");

  // If BB is a loop header, then duplicating this block outside the loop would
  // cause us to transform this into an irreducible loop, don't do this.
  // See the comments above FindLoopHeaders for justifications and caveats.
  if (LoopHeaders.count(BB)) {
    DEBUG(dbgs() << "  Not duplicating loop header '" << BB->getName()
          << "' into predecessor block '" << PredBBs[0]->getName()
          << "' - it might create an irreducible loop!\n");
    return false;
  }

  unsigned DuplicationCost = getJumpThreadDuplicationCost(BB, BBDupThreshold);
  if (DuplicationCost > BBDupThreshold) {
    DEBUG(dbgs() << "  Not duplicating BB '" << BB->getName()
          << "' - Cost is too high: " << DuplicationCost << "\n");
    return false;
  }

  // And finally, do it!  Start by factoring the predecessors is needed.
  BasicBlock *PredBB;
  if (PredBBs.size() == 1)
    PredBB = PredBBs[0];
  else {
    DEBUG(dbgs() << "  Factoring out " << PredBBs.size()
          << " common predecessors.\n");
    PredBB = SplitBlockPredecessors(BB, PredBBs, ".thr_comm");
  }

  // Okay, we decided to do this!  Clone all the instructions in BB onto the end
  // of PredBB.
  DEBUG(dbgs() << "  Duplicating block '" << BB->getName() << "' into end of '"
        << PredBB->getName() << "' to eliminate branch on phi.  Cost: "
        << DuplicationCost << " block is:" << *BB << "\n");

  // Unless PredBB ends with an unconditional branch, split the edge so that we
  // can just clone the bits from BB into the end of the new PredBB.
  BranchInst *OldPredBranch = dyn_cast<BranchInst>(PredBB->getTerminator());

  if (!OldPredBranch || !OldPredBranch->isUnconditional()) {
    PredBB = SplitEdge(PredBB, BB);
    OldPredBranch = cast<BranchInst>(PredBB->getTerminator());
  }

  // We are going to have to map operands from the original BB block into the
  // PredBB block.  Evaluate PHI nodes in BB.
  DenseMap<Instruction*, Value*> ValueMapping;

  BasicBlock::iterator BI = BB->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(PredBB);
  // Clone the non-phi instructions of BB into PredBB, keeping track of the
  // mapping and using it to remap operands in the cloned instructions.
  for (; BI != BB->end(); ++BI) {
    Instruction *New = BI->clone();

    // Remap operands to patch up intra-block references.
    for (unsigned i = 0, e = New->getNumOperands(); i != e; ++i)
      if (Instruction *Inst = dyn_cast<Instruction>(New->getOperand(i))) {
        DenseMap<Instruction*, Value*>::iterator I = ValueMapping.find(Inst);
        if (I != ValueMapping.end())
          New->setOperand(i, I->second);
      }

    // If this instruction can be simplified after the operands are updated,
    // just use the simplified value instead.  This frequently happens due to
    // phi translation.
    if (Value *IV =
            SimplifyInstruction(New, BB->getModule()->getDataLayout())) {
      delete New;
      ValueMapping[BI] = IV;
    } else {
      // Otherwise, insert the new instruction into the block.
      New->setName(BI->getName());
      PredBB->getInstList().insert(OldPredBranch, New);
      ValueMapping[BI] = New;
    }
  }

  // Check to see if the targets of the branch had PHI nodes. If so, we need to
  // add entries to the PHI nodes for branch from PredBB now.
  BranchInst *BBBranch = cast<BranchInst>(BB->getTerminator());
  AddPHINodeEntriesForMappedBlock(BBBranch->getSuccessor(0), BB, PredBB,
                                  ValueMapping);
  AddPHINodeEntriesForMappedBlock(BBBranch->getSuccessor(1), BB, PredBB,
                                  ValueMapping);

  // If there were values defined in BB that are used outside the block, then we
  // now have to update all uses of the value to use either the original value,
  // the cloned value, or some PHI derived value.  This can require arbitrary
  // PHI insertion, of which we are prepared to do, clean these up now.
  SSAUpdater SSAUpdate;
  SmallVector<Use*, 16> UsesToRename;
  for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    // Scan all uses of this instruction to see if it is used outside of its
    // block, and if so, record them in UsesToRename.
    for (Use &U : I->uses()) {
      Instruction *User = cast<Instruction>(U.getUser());
      if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
        if (UserPN->getIncomingBlock(U) == BB)
          continue;
      } else if (User->getParent() == BB)
        continue;

      UsesToRename.push_back(&U);
    }

    // If there are no uses outside the block, we're done with this instruction.
    if (UsesToRename.empty())
      continue;

    DEBUG(dbgs() << "JT: Renaming non-local uses of: " << *I << "\n");

    // We found a use of I outside of BB.  Rename all uses of I that are outside
    // its block to be uses of the appropriate PHI node etc.  See ValuesInBlocks
    // with the two values we know.
    SSAUpdate.Initialize(I->getType(), I->getName());
    SSAUpdate.AddAvailableValue(BB, I);
    SSAUpdate.AddAvailableValue(PredBB, ValueMapping[I]);

    while (!UsesToRename.empty())
      SSAUpdate.RewriteUse(*UsesToRename.pop_back_val());
    DEBUG(dbgs() << "\n");
  }

  // PredBB no longer jumps to BB, remove entries in the PHI node for the edge
  // that we nuked.
  BB->removePredecessor(PredBB, true);

  // Remove the unconditional branch at the end of the PredBB block.
  OldPredBranch->eraseFromParent();

  ++NumDupes;
  return true;
}

/// TryToUnfoldSelect - Look for blocks of the form
/// bb1:
///   %a = select
///   br bb
///
/// bb2:
///   %p = phi [%a, %bb] ...
///   %c = icmp %p
///   br i1 %c
///
/// And expand the select into a branch structure if one of its arms allows %c
/// to be folded. This later enables threading from bb1 over bb2.
bool JumpThreading::TryToUnfoldSelect(CmpInst *CondCmp, BasicBlock *BB) {
  BranchInst *CondBr = dyn_cast<BranchInst>(BB->getTerminator());
  PHINode *CondLHS = dyn_cast<PHINode>(CondCmp->getOperand(0));
  Constant *CondRHS = cast<Constant>(CondCmp->getOperand(1));

  if (!CondBr || !CondBr->isConditional() || !CondLHS ||
      CondLHS->getParent() != BB)
    return false;

  for (unsigned I = 0, E = CondLHS->getNumIncomingValues(); I != E; ++I) {
    BasicBlock *Pred = CondLHS->getIncomingBlock(I);
    SelectInst *SI = dyn_cast<SelectInst>(CondLHS->getIncomingValue(I));

    // Look if one of the incoming values is a select in the corresponding
    // predecessor.
    if (!SI || SI->getParent() != Pred || !SI->hasOneUse())
      continue;

    BranchInst *PredTerm = dyn_cast<BranchInst>(Pred->getTerminator());
    if (!PredTerm || !PredTerm->isUnconditional())
      continue;

    // Now check if one of the select values would allow us to constant fold the
    // terminator in BB. We don't do the transform if both sides fold, those
    // cases will be threaded in any case.
    LazyValueInfo::Tristate LHSFolds =
        LVI->getPredicateOnEdge(CondCmp->getPredicate(), SI->getOperand(1),
                                CondRHS, Pred, BB, CondCmp);
    LazyValueInfo::Tristate RHSFolds =
        LVI->getPredicateOnEdge(CondCmp->getPredicate(), SI->getOperand(2),
                                CondRHS, Pred, BB, CondCmp);
    if ((LHSFolds != LazyValueInfo::Unknown ||
         RHSFolds != LazyValueInfo::Unknown) &&
        LHSFolds != RHSFolds) {
      // Expand the select.
      //
      // Pred --
      //  |    v
      //  |  NewBB
      //  |    |
      //  |-----
      //  v
      // BB
      BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "select.unfold",
                                             BB->getParent(), BB);
      // Move the unconditional branch to NewBB.
      PredTerm->removeFromParent();
      NewBB->getInstList().insert(NewBB->end(), PredTerm);
      // Create a conditional branch and update PHI nodes.
      BranchInst::Create(NewBB, BB, SI->getCondition(), Pred);
      CondLHS->setIncomingValue(I, SI->getFalseValue());
      CondLHS->addIncoming(SI->getTrueValue(), NewBB);
      // The select is now dead.
      SI->eraseFromParent();

      // Update any other PHI nodes in BB.
      for (BasicBlock::iterator BI = BB->begin();
           PHINode *Phi = dyn_cast<PHINode>(BI); ++BI)
        if (Phi != CondLHS)
          Phi->addIncoming(Phi->getIncomingValueForBlock(Pred), NewBB);
      return true;
    }
  }
  return false;
}
