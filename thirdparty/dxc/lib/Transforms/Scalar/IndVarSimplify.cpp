//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into simpler forms suitable for subsequent
// analysis and transformation.
//
// If the trip count of a loop is computable, this pass also makes the following
// changes:
//   1. The exit condition for the loop is canonicalized to compare the
//      induction value against the exit value.  This turns loops like:
//        'for (i = 7; i*i < 1000; ++i)' into 'for (i = 0; i != 25; ++i)'
//   2. Any use outside of the loop of an expression derived from the indvar
//      is changed to compute the derived value outside of the loop, eliminating
//      the dependence on the exit value of the induction variable.  If the only
//      purpose of the loop is to compute the exit value of some derived
//      expression, this transformation will make the loop dead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
using namespace llvm;

#define DEBUG_TYPE "indvars"

STATISTIC(NumWidened     , "Number of indvars widened");
STATISTIC(NumReplaced    , "Number of exit values replaced");
STATISTIC(NumLFTR        , "Number of loop exit tests replaced");
STATISTIC(NumElimExt     , "Number of IV sign/zero extends eliminated");
STATISTIC(NumElimIV      , "Number of congruent IVs eliminated");

#if 0 // HLSL Change Starts - option pending

// Trip count verification can be enabled by default under NDEBUG if we
// implement a strong expression equivalence checker in SCEV. Until then, we
// use the verify-indvars flag, which may assert in some cases.
static cl::opt<bool> VerifyIndvars(
  "verify-indvars", cl::Hidden,
  cl::desc("Verify the ScalarEvolution result after running indvars"));

static cl::opt<bool> ReduceLiveIVs("liv-reduce", cl::Hidden,
  cl::desc("Reduce live induction variables."));

enum ReplaceExitVal { NeverRepl, OnlyCheapRepl, AlwaysRepl };

static cl::opt<ReplaceExitVal> ReplaceExitValue(
    "replexitval", cl::Hidden, cl::init(OnlyCheapRepl),
    cl::desc("Choose the strategy to replace exit value in IndVarSimplify"),
    cl::values(clEnumValN(NeverRepl, "never", "never replace exit value"),
               clEnumValN(OnlyCheapRepl, "cheap",
                          "only replace exit value when the cost is cheap"),
               clEnumValN(AlwaysRepl, "always",
                          "always replace exit value whenever possible"),
               clEnumValEnd));
#else
static const bool ReduceLiveIVs = false;
enum ReplaceExitVal { NeverRepl, OnlyCheapRepl, AlwaysRepl };
static const ReplaceExitVal ReplaceExitValue = OnlyCheapRepl;
#endif // HLSL Change Ends - option pending

namespace {
struct RewritePhi;
}

namespace {
  class IndVarSimplify : public LoopPass {
    LoopInfo                  *LI;
    ScalarEvolution           *SE;
    DominatorTree             *DT;
    TargetLibraryInfo         *TLI;
    const TargetTransformInfo *TTI;

    SmallVector<WeakVH, 16> DeadInsts;
    bool Changed;
  public:

    static char ID; // Pass identification, replacement for typeid
    IndVarSimplify()
        : LoopPass(ID), LI(nullptr), SE(nullptr), DT(nullptr), Changed(false) {
      initializeIndVarSimplifyPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      AU.setPreservesCFG();
    }

  private:
    void releaseMemory() override {
      DeadInsts.clear();
    }

    bool isValidRewrite(Value *FromVal, Value *ToVal);

    void HandleFloatingPointIV(Loop *L, PHINode *PH);
    void RewriteNonIntegerIVs(Loop *L);

    void SimplifyAndExtend(Loop *L, SCEVExpander &Rewriter, LPPassManager &LPM);

    bool CanLoopBeDeleted(Loop *L, SmallVector<RewritePhi, 8> &RewritePhiSet);
    void RewriteLoopExitValues(Loop *L, SCEVExpander &Rewriter);

    Value *LinearFunctionTestReplace(Loop *L, const SCEV *BackedgeTakenCount,
                                     PHINode *IndVar, SCEVExpander &Rewriter);

    void SinkUnusedInvariants(Loop *L);

    Value *ExpandSCEVIfNeeded(SCEVExpander &Rewriter, const SCEV *S, Loop *L,
                              Instruction *InsertPt, Type *Ty,
                              bool &IsHighCostExpansion);
  };
}

char IndVarSimplify::ID = 0;
INITIALIZE_PASS_BEGIN(IndVarSimplify, "indvars",
                "Induction Variable Simplification", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(IndVarSimplify, "indvars",
                "Induction Variable Simplification", false, false)

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// isValidRewrite - Return true if the SCEV expansion generated by the
/// rewriter can replace the original value. SCEV guarantees that it
/// produces the same value, but the way it is produced may be illegal IR.
/// Ideally, this function will only be called for verification.
bool IndVarSimplify::isValidRewrite(Value *FromVal, Value *ToVal) {
  // If an SCEV expression subsumed multiple pointers, its expansion could
  // reassociate the GEP changing the base pointer. This is illegal because the
  // final address produced by a GEP chain must be inbounds relative to its
  // underlying object. Otherwise basic alias analysis, among other things,
  // could fail in a dangerous way. Ultimately, SCEV will be improved to avoid
  // producing an expression involving multiple pointers. Until then, we must
  // bail out here.
  //
  // Retrieve the pointer operand of the GEP. Don't use GetUnderlyingObject
  // because it understands lcssa phis while SCEV does not.
  Value *FromPtr = FromVal;
  Value *ToPtr = ToVal;
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(FromVal)) {
    FromPtr = GEP->getPointerOperand();
  }
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(ToVal)) {
    ToPtr = GEP->getPointerOperand();
  }
  if (FromPtr != FromVal || ToPtr != ToVal) {
    // Quickly check the common case
    if (FromPtr == ToPtr)
      return true;

    // SCEV may have rewritten an expression that produces the GEP's pointer
    // operand. That's ok as long as the pointer operand has the same base
    // pointer. Unlike GetUnderlyingObject(), getPointerBase() will find the
    // base of a recurrence. This handles the case in which SCEV expansion
    // converts a pointer type recurrence into a nonrecurrent pointer base
    // indexed by an integer recurrence.

    // If the GEP base pointer is a vector of pointers, abort.
    if (!FromPtr->getType()->isPointerTy() || !ToPtr->getType()->isPointerTy())
      return false;

    const SCEV *FromBase = SE->getPointerBase(SE->getSCEV(FromPtr));
    const SCEV *ToBase = SE->getPointerBase(SE->getSCEV(ToPtr));
    if (FromBase == ToBase)
      return true;

    DEBUG(dbgs() << "INDVARS: GEP rewrite bail out "
          << *FromBase << " != " << *ToBase << "\n");

    return false;
  }
  return true;
}

/// Determine the insertion point for this user. By default, insert immediately
/// before the user. SCEVExpander or LICM will hoist loop invariants out of the
/// loop. For PHI nodes, there may be multiple uses, so compute the nearest
/// common dominator for the incoming blocks.
static Instruction *getInsertPointForUses(Instruction *User, Value *Def,
                                          DominatorTree *DT) {
  PHINode *PHI = dyn_cast<PHINode>(User);
  if (!PHI)
    return User;

  Instruction *InsertPt = nullptr;
  for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
    if (PHI->getIncomingValue(i) != Def)
      continue;

    BasicBlock *InsertBB = PHI->getIncomingBlock(i);
    if (!InsertPt) {
      InsertPt = InsertBB->getTerminator();
      continue;
    }
    InsertBB = DT->findNearestCommonDominator(InsertPt->getParent(), InsertBB);
    InsertPt = InsertBB->getTerminator();
  }
  assert(InsertPt && "Missing phi operand");
  assert((!isa<Instruction>(Def) ||
          DT->dominates(cast<Instruction>(Def), InsertPt)) &&
         "def does not dominate all uses");
  return InsertPt;
}

//===----------------------------------------------------------------------===//
// RewriteNonIntegerIVs and helpers. Prefer integer IVs.
//===----------------------------------------------------------------------===//

/// ConvertToSInt - Convert APF to an integer, if possible.
static bool ConvertToSInt(const APFloat &APF, int64_t &IntVal) {
  bool isExact = false;
  // See if we can convert this to an int64_t
  uint64_t UIntVal;
  if (APF.convertToInteger(&UIntVal, 64, true, APFloat::rmTowardZero,
                           &isExact) != APFloat::opOK || !isExact)
    return false;
  IntVal = UIntVal;
  return true;
}

/// HandleFloatingPointIV - If the loop has floating induction variable
/// then insert corresponding integer induction variable if possible.
/// For example,
/// for(double i = 0; i < 10000; ++i)
///   bar(i)
/// is converted into
/// for(int i = 0; i < 10000; ++i)
///   bar((double)i);
///
void IndVarSimplify::HandleFloatingPointIV(Loop *L, PHINode *PN) {
  unsigned IncomingEdge = L->contains(PN->getIncomingBlock(0));
  unsigned BackEdge     = IncomingEdge^1;

  // Check incoming value.
  ConstantFP *InitValueVal =
    dyn_cast<ConstantFP>(PN->getIncomingValue(IncomingEdge));

  int64_t InitValue;
  if (!InitValueVal || !ConvertToSInt(InitValueVal->getValueAPF(), InitValue))
    return;

  // Check IV increment. Reject this PN if increment operation is not
  // an add or increment value can not be represented by an integer.
  BinaryOperator *Incr =
    dyn_cast<BinaryOperator>(PN->getIncomingValue(BackEdge));
  if (Incr == nullptr || Incr->getOpcode() != Instruction::FAdd) return;

  // If this is not an add of the PHI with a constantfp, or if the constant fp
  // is not an integer, bail out.
  ConstantFP *IncValueVal = dyn_cast<ConstantFP>(Incr->getOperand(1));
  int64_t IncValue;
  if (IncValueVal == nullptr || Incr->getOperand(0) != PN ||
      !ConvertToSInt(IncValueVal->getValueAPF(), IncValue))
    return;

  // Check Incr uses. One user is PN and the other user is an exit condition
  // used by the conditional terminator.
  Value::user_iterator IncrUse = Incr->user_begin();
  Instruction *U1 = cast<Instruction>(*IncrUse++);
  if (IncrUse == Incr->user_end()) return;
  Instruction *U2 = cast<Instruction>(*IncrUse++);
  if (IncrUse != Incr->user_end()) return;

  // Find exit condition, which is an fcmp.  If it doesn't exist, or if it isn't
  // only used by a branch, we can't transform it.
  FCmpInst *Compare = dyn_cast<FCmpInst>(U1);
  if (!Compare)
    Compare = dyn_cast<FCmpInst>(U2);
  if (!Compare || !Compare->hasOneUse() ||
      !isa<BranchInst>(Compare->user_back()))
    return;

  BranchInst *TheBr = cast<BranchInst>(Compare->user_back());

  // We need to verify that the branch actually controls the iteration count
  // of the loop.  If not, the new IV can overflow and no one will notice.
  // The branch block must be in the loop and one of the successors must be out
  // of the loop.
  assert(TheBr->isConditional() && "Can't use fcmp if not conditional");
  if (!L->contains(TheBr->getParent()) ||
      (L->contains(TheBr->getSuccessor(0)) &&
       L->contains(TheBr->getSuccessor(1))))
    return;


  // If it isn't a comparison with an integer-as-fp (the exit value), we can't
  // transform it.
  ConstantFP *ExitValueVal = dyn_cast<ConstantFP>(Compare->getOperand(1));
  int64_t ExitValue;
  if (ExitValueVal == nullptr ||
      !ConvertToSInt(ExitValueVal->getValueAPF(), ExitValue))
    return;

  // Find new predicate for integer comparison.
  CmpInst::Predicate NewPred = CmpInst::BAD_ICMP_PREDICATE;
  switch (Compare->getPredicate()) {
  default: return;  // Unknown comparison.
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_UEQ: NewPred = CmpInst::ICMP_EQ; break;
  case CmpInst::FCMP_ONE:
  case CmpInst::FCMP_UNE: NewPred = CmpInst::ICMP_NE; break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT: NewPred = CmpInst::ICMP_SGT; break;
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE: NewPred = CmpInst::ICMP_SGE; break;
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_ULT: NewPred = CmpInst::ICMP_SLT; break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULE: NewPred = CmpInst::ICMP_SLE; break;
  }

  // We convert the floating point induction variable to a signed i32 value if
  // we can.  This is only safe if the comparison will not overflow in a way
  // that won't be trapped by the integer equivalent operations.  Check for this
  // now.
  // TODO: We could use i64 if it is native and the range requires it.

  // The start/stride/exit values must all fit in signed i32.
  if (!isInt<32>(InitValue) || !isInt<32>(IncValue) || !isInt<32>(ExitValue))
    return;

  // If not actually striding (add x, 0.0), avoid touching the code.
  if (IncValue == 0)
    return;

  // Positive and negative strides have different safety conditions.
  if (IncValue > 0) {
    // If we have a positive stride, we require the init to be less than the
    // exit value.
    if (InitValue >= ExitValue)
      return;

    uint32_t Range = uint32_t(ExitValue-InitValue);
    // Check for infinite loop, either:
    // while (i <= Exit) or until (i > Exit)
    if (NewPred == CmpInst::ICMP_SLE || NewPred == CmpInst::ICMP_SGT) {
      if (++Range == 0) return;  // Range overflows.
    }

    unsigned Leftover = Range % uint32_t(IncValue);

    // If this is an equality comparison, we require that the strided value
    // exactly land on the exit value, otherwise the IV condition will wrap
    // around and do things the fp IV wouldn't.
    if ((NewPred == CmpInst::ICMP_EQ || NewPred == CmpInst::ICMP_NE) &&
        Leftover != 0)
      return;

    // If the stride would wrap around the i32 before exiting, we can't
    // transform the IV.
    if (Leftover != 0 && int32_t(ExitValue+IncValue) < ExitValue)
      return;

  } else {
    // If we have a negative stride, we require the init to be greater than the
    // exit value.
    if (InitValue <= ExitValue)
      return;

    uint32_t Range = uint32_t(InitValue-ExitValue);
    // Check for infinite loop, either:
    // while (i >= Exit) or until (i < Exit)
    if (NewPred == CmpInst::ICMP_SGE || NewPred == CmpInst::ICMP_SLT) {
      if (++Range == 0) return;  // Range overflows.
    }

    unsigned Leftover = Range % uint32_t(-IncValue);

    // If this is an equality comparison, we require that the strided value
    // exactly land on the exit value, otherwise the IV condition will wrap
    // around and do things the fp IV wouldn't.
    if ((NewPred == CmpInst::ICMP_EQ || NewPred == CmpInst::ICMP_NE) &&
        Leftover != 0)
      return;

    // If the stride would wrap around the i32 before exiting, we can't
    // transform the IV.
    if (Leftover != 0 && int32_t(ExitValue+IncValue) > ExitValue)
      return;
  }

  IntegerType *Int32Ty = Type::getInt32Ty(PN->getContext());

  // Insert new integer induction variable.
  PHINode *NewPHI = PHINode::Create(Int32Ty, 2, PN->getName()+".int", PN);
  NewPHI->addIncoming(ConstantInt::get(Int32Ty, InitValue),
                      PN->getIncomingBlock(IncomingEdge));

  Value *NewAdd =
    BinaryOperator::CreateAdd(NewPHI, ConstantInt::get(Int32Ty, IncValue),
                              Incr->getName()+".int", Incr);
  NewPHI->addIncoming(NewAdd, PN->getIncomingBlock(BackEdge));

  ICmpInst *NewCompare = new ICmpInst(TheBr, NewPred, NewAdd,
                                      ConstantInt::get(Int32Ty, ExitValue),
                                      Compare->getName());

  // In the following deletions, PN may become dead and may be deleted.
  // Use a WeakVH to observe whether this happens.
  WeakVH WeakPH = PN;

  // Delete the old floating point exit comparison.  The branch starts using the
  // new comparison.
  NewCompare->takeName(Compare);
  Compare->replaceAllUsesWith(NewCompare);
  RecursivelyDeleteTriviallyDeadInstructions(Compare, TLI);

  // Delete the old floating point increment.
  Incr->replaceAllUsesWith(UndefValue::get(Incr->getType()));
  RecursivelyDeleteTriviallyDeadInstructions(Incr, TLI);

  // If the FP induction variable still has uses, this is because something else
  // in the loop uses its value.  In order to canonicalize the induction
  // variable, we chose to eliminate the IV and rewrite it in terms of an
  // int->fp cast.
  //
  // We give preference to sitofp over uitofp because it is faster on most
  // platforms.
  if (WeakPH) {
    Value *Conv = new SIToFPInst(NewPHI, PN->getType(), "indvar.conv",
                                 PN->getParent()->getFirstInsertionPt());
    PN->replaceAllUsesWith(Conv);
    RecursivelyDeleteTriviallyDeadInstructions(PN, TLI);
  }
  Changed = true;
}

void IndVarSimplify::RewriteNonIntegerIVs(Loop *L) {
  // First step.  Check to see if there are any floating-point recurrences.
  // If there are, change them into integer recurrences, permitting analysis by
  // the SCEV routines.
  //
  BasicBlock *Header = L->getHeader();

  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(&*PHIs[i]))
      HandleFloatingPointIV(L, PN);

  // If the loop previously had floating-point IV, ScalarEvolution
  // may not have been able to compute a trip count. Now that we've done some
  // re-writing, the trip count may be computable.
  if (Changed)
    SE->forgetLoop(L);
}

namespace {
// Collect information about PHI nodes which can be transformed in
// RewriteLoopExitValues.
struct RewritePhi {
  PHINode *PN;
  unsigned Ith;  // Ith incoming value.
  Value *Val;    // Exit value after expansion.
  bool HighCost; // High Cost when expansion.
  bool SafePhi;  // LCSSASafePhiForRAUW.

  RewritePhi(PHINode *P, unsigned I, Value *V, bool H, bool S)
      : PN(P), Ith(I), Val(V), HighCost(H), SafePhi(S) {}
};
}

Value *IndVarSimplify::ExpandSCEVIfNeeded(SCEVExpander &Rewriter, const SCEV *S,
                                          Loop *L, Instruction *InsertPt,
                                          Type *ResultTy,
                                          bool &IsHighCostExpansion) {
  using namespace llvm::PatternMatch;

  if (!Rewriter.isHighCostExpansion(S, L)) {
    IsHighCostExpansion = false;
    return Rewriter.expandCodeFor(S, ResultTy, InsertPt);
  }

  // Before expanding S into an expensive LLVM expression, see if we can use an
  // already existing value as the expansion for S.  There is potential to make
  // this significantly smarter, but this simple heuristic already gets some
  // interesting cases.

  SmallVector<BasicBlock *, 4> Latches;
  L->getLoopLatches(Latches);

  for (BasicBlock *BB : Latches) {
    ICmpInst::Predicate Pred;
    Instruction *LHS, *RHS;
    BasicBlock *TrueBB, *FalseBB;

    if (!match(BB->getTerminator(),
               m_Br(m_ICmp(Pred, m_Instruction(LHS), m_Instruction(RHS)),
                    TrueBB, FalseBB)))
      continue;

    if (SE->getSCEV(LHS) == S && DT->dominates(LHS, InsertPt)) {
      IsHighCostExpansion = false;
      return LHS;
    }

    if (SE->getSCEV(RHS) == S && DT->dominates(RHS, InsertPt)) {
      IsHighCostExpansion = false;
      return RHS;
    }
  }

  // We didn't find anything, fall back to using SCEVExpander.
  assert(Rewriter.isHighCostExpansion(S, L) && "this should not have changed!");
  IsHighCostExpansion = true;
  return Rewriter.expandCodeFor(S, ResultTy, InsertPt);
}

//===----------------------------------------------------------------------===//
// RewriteLoopExitValues - Optimize IV users outside the loop.
// As a side effect, reduces the amount of IV processing within the loop.
//===----------------------------------------------------------------------===//

/// RewriteLoopExitValues - Check to see if this loop has a computable
/// loop-invariant execution count.  If so, this means that we can compute the
/// final value of any expressions that are recurrent in the loop, and
/// substitute the exit values from the loop into any instructions outside of
/// the loop that use the final values of the current expressions.
///
/// This is mostly redundant with the regular IndVarSimplify activities that
/// happen later, except that it's more powerful in some cases, because it's
/// able to brute-force evaluate arbitrary instructions as long as they have
/// constant operands at the beginning of the loop.
void IndVarSimplify::RewriteLoopExitValues(Loop *L, SCEVExpander &Rewriter) {
  // Verify the input to the pass in already in LCSSA form.
  assert(L->isLCSSAForm(*DT));

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  SmallVector<RewritePhi, 8> RewritePhiSet;
  // Find all values that are computed inside the loop, but used outside of it.
  // Because of LCSSA, these values will only occur in LCSSA PHI Nodes.  Scan
  // the exit blocks of the loop to find them.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBB = ExitBlocks[i];

    // If there are no PHI nodes in this exit block, then no values defined
    // inside the loop are used on this path, skip it.
    PHINode *PN = dyn_cast<PHINode>(ExitBB->begin());
    if (!PN) continue;

    unsigned NumPreds = PN->getNumIncomingValues();

    // We would like to be able to RAUW single-incoming value PHI nodes. We
    // have to be certain this is safe even when this is an LCSSA PHI node.
    // While the computed exit value is no longer varying in *this* loop, the
    // exit block may be an exit block for an outer containing loop as well,
    // the exit value may be varying in the outer loop, and thus it may still
    // require an LCSSA PHI node. The safe case is when this is
    // single-predecessor PHI node (LCSSA) and the exit block containing it is
    // part of the enclosing loop, or this is the outer most loop of the nest.
    // In either case the exit value could (at most) be varying in the same
    // loop body as the phi node itself. Thus if it is in turn used outside of
    // an enclosing loop it will only be via a separate LCSSA node.
    bool LCSSASafePhiForRAUW =
        NumPreds == 1 &&
        (!L->getParentLoop() || L->getParentLoop() == LI->getLoopFor(ExitBB));

    // Iterate over all of the PHI nodes.
    BasicBlock::iterator BBI = ExitBB->begin();
    while ((PN = dyn_cast<PHINode>(BBI++))) {
      if (PN->use_empty())
        continue; // dead use, don't replace it

      // SCEV only supports integer expressions for now.
      if (!PN->getType()->isIntegerTy() && !PN->getType()->isPointerTy())
        continue;

      // It's necessary to tell ScalarEvolution about this explicitly so that
      // it can walk the def-use list and forget all SCEVs, as it may not be
      // watching the PHI itself. Once the new exit value is in place, there
      // may not be a def-use connection between the loop and every instruction
      // which got a SCEVAddRecExpr for that loop.
      SE->forgetValue(PN);

      // Iterate over all of the values in all the PHI nodes.
      for (unsigned i = 0; i != NumPreds; ++i) {
        // If the value being merged in is not integer or is not defined
        // in the loop, skip it.
        Value *InVal = PN->getIncomingValue(i);
        if (!isa<Instruction>(InVal))
          continue;

        // If this pred is for a subloop, not L itself, skip it.
        if (LI->getLoopFor(PN->getIncomingBlock(i)) != L)
          continue; // The Block is in a subloop, skip it.

        // Check that InVal is defined in the loop.
        Instruction *Inst = cast<Instruction>(InVal);
        if (!L->contains(Inst))
          continue;

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.
        const SCEV *ExitValue = SE->getSCEVAtScope(Inst, L->getParentLoop());
        if (!SE->isLoopInvariant(ExitValue, L) ||
            !isSafeToExpand(ExitValue, *SE))
          continue;

        // Computing the value outside of the loop brings no benefit if :
        //  - it is definitely used inside the loop in a way which can not be
        //    optimized away.
        //  - no use outside of the loop can take advantage of hoisting the
        //    computation out of the loop
        if (ExitValue->getSCEVType()>=scMulExpr) {
          unsigned NumHardInternalUses = 0;
          unsigned NumSoftExternalUses = 0;
          unsigned NumUses = 0;
          for (auto IB = Inst->user_begin(), IE = Inst->user_end();
               IB != IE && NumUses <= 6; ++IB) {
            Instruction *UseInstr = cast<Instruction>(*IB);
            unsigned Opc = UseInstr->getOpcode();
            NumUses++;
            if (L->contains(UseInstr)) {
              if (Opc == Instruction::Call || Opc == Instruction::Ret)
                NumHardInternalUses++;
            } else {
              if (Opc == Instruction::PHI) {
                // Do not count the Phi as a use. LCSSA may have inserted
                // plenty of trivial ones.
                NumUses--;
                for (auto PB = UseInstr->user_begin(),
                          PE = UseInstr->user_end();
                     PB != PE && NumUses <= 6; ++PB, ++NumUses) {
                  unsigned PhiOpc = cast<Instruction>(*PB)->getOpcode();
                  if (PhiOpc != Instruction::Call && PhiOpc != Instruction::Ret)
                    NumSoftExternalUses++;
                }
                continue;
              }
              if (Opc != Instruction::Call && Opc != Instruction::Ret)
                NumSoftExternalUses++;
            }
          }
          if (NumUses <= 6 && NumHardInternalUses && !NumSoftExternalUses)
            continue;
        }

        bool HighCost = false;
        Value *ExitVal = ExpandSCEVIfNeeded(Rewriter, ExitValue, L, Inst,
                                            PN->getType(), HighCost);

        DEBUG(dbgs() << "INDVARS: RLEV: AfterLoopVal = " << *ExitVal << '\n'
                     << "  LoopVal = " << *Inst << "\n");

        if (!isValidRewrite(Inst, ExitVal)) {
          DeadInsts.push_back(ExitVal);
          continue;
        }

        // Collect all the candidate PHINodes to be rewritten.
        RewritePhiSet.push_back(
            RewritePhi(PN, i, ExitVal, HighCost, LCSSASafePhiForRAUW));
      }
    }
  }

  bool LoopCanBeDel = CanLoopBeDeleted(L, RewritePhiSet);

  // Transformation.
  for (const RewritePhi &Phi : RewritePhiSet) {
    PHINode *PN = Phi.PN;
    Value *ExitVal = Phi.Val;

    // Only do the rewrite when the ExitValue can be expanded cheaply.
    // If LoopCanBeDel is true, rewrite exit value aggressively.
    if (ReplaceExitValue == OnlyCheapRepl && !LoopCanBeDel && Phi.HighCost) {
      DeadInsts.push_back(ExitVal);
      continue;
    }

    Changed = true;
    ++NumReplaced;
    Instruction *Inst = cast<Instruction>(PN->getIncomingValue(Phi.Ith));
    PN->setIncomingValue(Phi.Ith, ExitVal);

    // If this instruction is dead now, delete it. Don't do it now to avoid
    // invalidating iterators.
    if (isInstructionTriviallyDead(Inst, TLI))
      DeadInsts.push_back(Inst);

    // If we determined that this PHI is safe to replace even if an LCSSA
    // PHI, do so.
    if (Phi.SafePhi) {
      PN->replaceAllUsesWith(ExitVal);
      PN->eraseFromParent();
    }
  }

  // The insertion point instruction may have been deleted; clear it out
  // so that the rewriter doesn't trip over it later.
  Rewriter.clearInsertPoint();
}

/// CanLoopBeDeleted - Check whether it is possible to delete the loop after
/// rewriting exit value. If it is possible, ignore ReplaceExitValue and
/// do rewriting aggressively.
bool IndVarSimplify::CanLoopBeDeleted(
    Loop *L, SmallVector<RewritePhi, 8> &RewritePhiSet) {

  BasicBlock *Preheader = L->getLoopPreheader();
  // If there is no preheader, the loop will not be deleted.
  if (!Preheader)
    return false;

  // In LoopDeletion pass Loop can be deleted when ExitingBlocks.size() > 1.
  // We obviate multiple ExitingBlocks case for simplicity.
  // TODO: If we see testcase with multiple ExitingBlocks can be deleted
  // after exit value rewriting, we can enhance the logic here.
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  if (ExitBlocks.size() > 1 || ExitingBlocks.size() > 1)
    return false;

  BasicBlock *ExitBlock = ExitBlocks[0];
  BasicBlock::iterator BI = ExitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    Value *Incoming = P->getIncomingValueForBlock(ExitingBlocks[0]);

    // If the Incoming value of P is found in RewritePhiSet, we know it
    // could be rewritten to use a loop invariant value in transformation
    // phase later. Skip it in the loop invariant check below.
    bool found = false;
    for (const RewritePhi &Phi : RewritePhiSet) {
      unsigned i = Phi.Ith;
      if (Phi.PN == P && (Phi.PN)->getIncomingValue(i) == Incoming) {
        found = true;
        break;
      }
    }

    Instruction *I;
    if (!found && (I = dyn_cast<Instruction>(Incoming)))
      if (!L->hasLoopInvariantOperands(I))
        return false;

    ++BI;
  }

  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end(); BI != BE;
         ++BI) {
      if (BI->mayHaveSideEffects())
        return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//  IV Widening - Extend the width of an IV to cover its widest uses.
//===----------------------------------------------------------------------===//

namespace {
  // Collect information about induction variables that are used by sign/zero
  // extend operations. This information is recorded by CollectExtend and
  // provides the input to WidenIV.
  struct WideIVInfo {
    PHINode *NarrowIV;
    Type *WidestNativeType; // Widest integer type created [sz]ext
    bool IsSigned;          // Was a sext user seen before a zext?

    WideIVInfo() : NarrowIV(nullptr), WidestNativeType(nullptr),
                   IsSigned(false) {}
  };
}

/// visitCast - Update information about the induction variable that is
/// extended by this sign or zero extend operation. This is used to determine
/// the final width of the IV before actually widening it.
static void visitIVCast(CastInst *Cast, WideIVInfo &WI, ScalarEvolution *SE,
                        const TargetTransformInfo *TTI) {
  bool IsSigned = Cast->getOpcode() == Instruction::SExt;
  if (!IsSigned && Cast->getOpcode() != Instruction::ZExt)
    return;

  Type *Ty = Cast->getType();
  uint64_t Width = SE->getTypeSizeInBits(Ty);
  if (!Cast->getModule()->getDataLayout().isLegalInteger(Width))
    return;

  // Cast is either an sext or zext up to this point.
  // We should not widen an indvar if arithmetics on the wider indvar are more
  // expensive than those on the narrower indvar. We check only the cost of ADD
  // because at least an ADD is required to increment the induction variable. We
  // could compute more comprehensively the cost of all instructions on the
  // induction variable when necessary.
  if (TTI &&
      TTI->getArithmeticInstrCost(Instruction::Add, Ty) >
          TTI->getArithmeticInstrCost(Instruction::Add,
                                      Cast->getOperand(0)->getType())) {
    return;
  }

  if (!WI.WidestNativeType) {
    WI.WidestNativeType = SE->getEffectiveSCEVType(Ty);
    WI.IsSigned = IsSigned;
    return;
  }

  // We extend the IV to satisfy the sign of its first user, arbitrarily.
  if (WI.IsSigned != IsSigned)
    return;

  if (Width > SE->getTypeSizeInBits(WI.WidestNativeType))
    WI.WidestNativeType = SE->getEffectiveSCEVType(Ty);
}

namespace {

/// NarrowIVDefUse - Record a link in the Narrow IV def-use chain along with the
/// WideIV that computes the same value as the Narrow IV def.  This avoids
/// caching Use* pointers.
struct NarrowIVDefUse {
  Instruction *NarrowDef;
  Instruction *NarrowUse;
  Instruction *WideDef;

  NarrowIVDefUse(): NarrowDef(nullptr), NarrowUse(nullptr), WideDef(nullptr) {}

  NarrowIVDefUse(Instruction *ND, Instruction *NU, Instruction *WD):
    NarrowDef(ND), NarrowUse(NU), WideDef(WD) {}
};

/// WidenIV - The goal of this transform is to remove sign and zero extends
/// without creating any new induction variables. To do this, it creates a new
/// phi of the wider type and redirects all users, either removing extends or
/// inserting truncs whenever we stop propagating the type.
///
class WidenIV {
  // Parameters
  PHINode *OrigPhi;
  Type *WideType;
  bool IsSigned;

  // Context
  LoopInfo        *LI;
  Loop            *L;
  ScalarEvolution *SE;
  DominatorTree   *DT;

  // Result
  PHINode *WidePhi;
  Instruction *WideInc;
  const SCEV *WideIncExpr;
  SmallVectorImpl<WeakVH> &DeadInsts;

  SmallPtrSet<Instruction*,16> Widened;
  SmallVector<NarrowIVDefUse, 8> NarrowIVUsers;

public:
  WidenIV(const WideIVInfo &WI, LoopInfo *LInfo,
          ScalarEvolution *SEv, DominatorTree *DTree,
          SmallVectorImpl<WeakVH> &DI) :
    OrigPhi(WI.NarrowIV),
    WideType(WI.WidestNativeType),
    IsSigned(WI.IsSigned),
    LI(LInfo),
    L(LI->getLoopFor(OrigPhi->getParent())),
    SE(SEv),
    DT(DTree),
    WidePhi(nullptr),
    WideInc(nullptr),
    WideIncExpr(nullptr),
    DeadInsts(DI) {
    assert(L->getHeader() == OrigPhi->getParent() && "Phi must be an IV");
  }

  PHINode *CreateWideIV(SCEVExpander &Rewriter);

protected:
  Value *getExtend(Value *NarrowOper, Type *WideType, bool IsSigned,
                   Instruction *Use);

  Instruction *CloneIVUser(NarrowIVDefUse DU);

  const SCEVAddRecExpr *GetWideRecurrence(Instruction *NarrowUse);

  const SCEVAddRecExpr* GetExtendedOperandRecurrence(NarrowIVDefUse DU);

  const SCEV *GetSCEVByOpCode(const SCEV *LHS, const SCEV *RHS,
                              unsigned OpCode) const;

  Instruction *WidenIVUse(NarrowIVDefUse DU, SCEVExpander &Rewriter);

  bool WidenLoopCompare(NarrowIVDefUse DU);

  void pushNarrowIVUsers(Instruction *NarrowDef, Instruction *WideDef);
};
} // anonymous namespace

/// isLoopInvariant - Perform a quick domtree based check for loop invariance
/// assuming that V is used within the loop. LoopInfo::isLoopInvariant() seems
/// gratuitous for this purpose.
static bool isLoopInvariant(Value *V, const Loop *L, const DominatorTree *DT) {
  Instruction *Inst = dyn_cast<Instruction>(V);
  if (!Inst)
    return true;

  return DT->properlyDominates(Inst->getParent(), L->getHeader());
}

Value *WidenIV::getExtend(Value *NarrowOper, Type *WideType, bool IsSigned,
                          Instruction *Use) {
  // Set the debug location and conservative insertion point.
  IRBuilder<> Builder(Use);
  // Hoist the insertion point into loop preheaders as far as possible.
  for (const Loop *L = LI->getLoopFor(Use->getParent());
       L && L->getLoopPreheader() && isLoopInvariant(NarrowOper, L, DT);
       L = L->getParentLoop())
    Builder.SetInsertPoint(L->getLoopPreheader()->getTerminator());

  return IsSigned ? Builder.CreateSExt(NarrowOper, WideType) :
                    Builder.CreateZExt(NarrowOper, WideType);
}

/// CloneIVUser - Instantiate a wide operation to replace a narrow
/// operation. This only needs to handle operations that can evaluation to
/// SCEVAddRec. It can safely return 0 for any operation we decide not to clone.
Instruction *WidenIV::CloneIVUser(NarrowIVDefUse DU) {
  unsigned Opcode = DU.NarrowUse->getOpcode();
  switch (Opcode) {
  default:
    return nullptr;
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    DEBUG(dbgs() << "Cloning IVUser: " << *DU.NarrowUse << "\n");

    // Replace NarrowDef operands with WideDef. Otherwise, we don't know
    // anything about the narrow operand yet so must insert a [sz]ext. It is
    // probably loop invariant and will be folded or hoisted. If it actually
    // comes from a widened IV, it should be removed during a future call to
    // WidenIVUse.
    Value *LHS = (DU.NarrowUse->getOperand(0) == DU.NarrowDef) ? DU.WideDef :
      getExtend(DU.NarrowUse->getOperand(0), WideType, IsSigned, DU.NarrowUse);
    Value *RHS = (DU.NarrowUse->getOperand(1) == DU.NarrowDef) ? DU.WideDef :
      getExtend(DU.NarrowUse->getOperand(1), WideType, IsSigned, DU.NarrowUse);

    BinaryOperator *NarrowBO = cast<BinaryOperator>(DU.NarrowUse);
    BinaryOperator *WideBO = BinaryOperator::Create(NarrowBO->getOpcode(),
                                                    LHS, RHS,
                                                    NarrowBO->getName());
    IRBuilder<> Builder(DU.NarrowUse);
    Builder.Insert(WideBO);
    if (const OverflowingBinaryOperator *OBO =
        dyn_cast<OverflowingBinaryOperator>(NarrowBO)) {
      if (OBO->hasNoUnsignedWrap()) WideBO->setHasNoUnsignedWrap();
      if (OBO->hasNoSignedWrap()) WideBO->setHasNoSignedWrap();
    }
    return WideBO;
  }
}

const SCEV *WidenIV::GetSCEVByOpCode(const SCEV *LHS, const SCEV *RHS,
                                     unsigned OpCode) const {
  if (OpCode == Instruction::Add)
    return SE->getAddExpr(LHS, RHS);
  if (OpCode == Instruction::Sub)
    return SE->getMinusSCEV(LHS, RHS);
  if (OpCode == Instruction::Mul)
    return SE->getMulExpr(LHS, RHS);

  llvm_unreachable("Unsupported opcode.");
}

/// No-wrap operations can transfer sign extension of their result to their
/// operands. Generate the SCEV value for the widened operation without
/// actually modifying the IR yet. If the expression after extending the
/// operands is an AddRec for this loop, return it.
const SCEVAddRecExpr* WidenIV::GetExtendedOperandRecurrence(NarrowIVDefUse DU) {

  // Handle the common case of add<nsw/nuw>
  const unsigned OpCode = DU.NarrowUse->getOpcode();
  // Only Add/Sub/Mul instructions supported yet.
  if (OpCode != Instruction::Add && OpCode != Instruction::Sub &&
      OpCode != Instruction::Mul)
    return nullptr;

  // One operand (NarrowDef) has already been extended to WideDef. Now determine
  // if extending the other will lead to a recurrence.
  const unsigned ExtendOperIdx =
      DU.NarrowUse->getOperand(0) == DU.NarrowDef ? 1 : 0;
  assert(DU.NarrowUse->getOperand(1-ExtendOperIdx) == DU.NarrowDef && "bad DU");

  const SCEV *ExtendOperExpr = nullptr;
  const OverflowingBinaryOperator *OBO =
    cast<OverflowingBinaryOperator>(DU.NarrowUse);
  if (IsSigned && OBO->hasNoSignedWrap())
    ExtendOperExpr = SE->getSignExtendExpr(
      SE->getSCEV(DU.NarrowUse->getOperand(ExtendOperIdx)), WideType);
  else if(!IsSigned && OBO->hasNoUnsignedWrap())
    ExtendOperExpr = SE->getZeroExtendExpr(
      SE->getSCEV(DU.NarrowUse->getOperand(ExtendOperIdx)), WideType);
  else
    return nullptr;

  // When creating this SCEV expr, don't apply the current operations NSW or NUW
  // flags. This instruction may be guarded by control flow that the no-wrap
  // behavior depends on. Non-control-equivalent instructions can be mapped to
  // the same SCEV expression, and it would be incorrect to transfer NSW/NUW
  // semantics to those operations.
  const SCEV *lhs = SE->getSCEV(DU.WideDef);
  const SCEV *rhs = ExtendOperExpr;

  // Let's swap operands to the initial order for the case of non-commutative
  // operations, like SUB. See PR21014.
  if (ExtendOperIdx == 0)
    std::swap(lhs, rhs);
  const SCEVAddRecExpr *AddRec =
      dyn_cast<SCEVAddRecExpr>(GetSCEVByOpCode(lhs, rhs, OpCode));

  if (!AddRec || AddRec->getLoop() != L)
    return nullptr;
  return AddRec;
}

/// GetWideRecurrence - Is this instruction potentially interesting for further
/// simplification after widening it's type? In other words, can the
/// extend be safely hoisted out of the loop with SCEV reducing the value to a
/// recurrence on the same loop. If so, return the sign or zero extended
/// recurrence. Otherwise return NULL.
const SCEVAddRecExpr *WidenIV::GetWideRecurrence(Instruction *NarrowUse) {
  if (!SE->isSCEVable(NarrowUse->getType()))
    return nullptr;

  const SCEV *NarrowExpr = SE->getSCEV(NarrowUse);
  if (SE->getTypeSizeInBits(NarrowExpr->getType())
      >= SE->getTypeSizeInBits(WideType)) {
    // NarrowUse implicitly widens its operand. e.g. a gep with a narrow
    // index. So don't follow this use.
    return nullptr;
  }

  const SCEV *WideExpr = IsSigned ?
    SE->getSignExtendExpr(NarrowExpr, WideType) :
    SE->getZeroExtendExpr(NarrowExpr, WideType);
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(WideExpr);
  if (!AddRec || AddRec->getLoop() != L)
    return nullptr;
  return AddRec;
}

/// This IV user cannot be widen. Replace this use of the original narrow IV
/// with a truncation of the new wide IV to isolate and eliminate the narrow IV.
static void truncateIVUse(NarrowIVDefUse DU, DominatorTree *DT) {
  DEBUG(dbgs() << "INDVARS: Truncate IV " << *DU.WideDef
        << " for user " << *DU.NarrowUse << "\n");
  IRBuilder<> Builder(getInsertPointForUses(DU.NarrowUse, DU.NarrowDef, DT));
  Value *Trunc = Builder.CreateTrunc(DU.WideDef, DU.NarrowDef->getType());
  DU.NarrowUse->replaceUsesOfWith(DU.NarrowDef, Trunc);
}

/// If the narrow use is a compare instruction, then widen the compare
//  (and possibly the other operand).  The extend operation is hoisted into the
// loop preheader as far as possible.
bool WidenIV::WidenLoopCompare(NarrowIVDefUse DU) {
  ICmpInst *Cmp = dyn_cast<ICmpInst>(DU.NarrowUse);
  if (!Cmp)
    return false;

  // Sign of IV user and compare must match.
  if (IsSigned != CmpInst::isSigned(Cmp->getPredicate()))
    return false;

  Value *Op = Cmp->getOperand(Cmp->getOperand(0) == DU.NarrowDef ? 1 : 0);
  unsigned CastWidth = SE->getTypeSizeInBits(Op->getType());
  unsigned IVWidth = SE->getTypeSizeInBits(WideType);
  assert (CastWidth <= IVWidth && "Unexpected width while widening compare.");

  // Widen the compare instruction.
  IRBuilder<> Builder(getInsertPointForUses(DU.NarrowUse, DU.NarrowDef, DT));
  DU.NarrowUse->replaceUsesOfWith(DU.NarrowDef, DU.WideDef);

  // Widen the other operand of the compare, if necessary.
  if (CastWidth < IVWidth) {
    Value *ExtOp = getExtend(Op, WideType, IsSigned, Cmp);
    DU.NarrowUse->replaceUsesOfWith(Op, ExtOp);
  }
  return true;
}

/// WidenIVUse - Determine whether an individual user of the narrow IV can be
/// widened. If so, return the wide clone of the user.
Instruction *WidenIV::WidenIVUse(NarrowIVDefUse DU, SCEVExpander &Rewriter) {

  // Stop traversing the def-use chain at inner-loop phis or post-loop phis.
  if (PHINode *UsePhi = dyn_cast<PHINode>(DU.NarrowUse)) {
    if (LI->getLoopFor(UsePhi->getParent()) != L) {
      // For LCSSA phis, sink the truncate outside the loop.
      // After SimplifyCFG most loop exit targets have a single predecessor.
      // Otherwise fall back to a truncate within the loop.
      if (UsePhi->getNumOperands() != 1)
        truncateIVUse(DU, DT);
      else {
        PHINode *WidePhi =
          PHINode::Create(DU.WideDef->getType(), 1, UsePhi->getName() + ".wide",
                          UsePhi);
        WidePhi->addIncoming(DU.WideDef, UsePhi->getIncomingBlock(0));
        IRBuilder<> Builder(WidePhi->getParent()->getFirstInsertionPt());
        Value *Trunc = Builder.CreateTrunc(WidePhi, DU.NarrowDef->getType());
        UsePhi->replaceAllUsesWith(Trunc);
        DeadInsts.emplace_back(UsePhi);
        DEBUG(dbgs() << "INDVARS: Widen lcssa phi " << *UsePhi
              << " to " << *WidePhi << "\n");
      }
      return nullptr;
    }
  }
  // Our raison d'etre! Eliminate sign and zero extension.
  if (IsSigned ? isa<SExtInst>(DU.NarrowUse) : isa<ZExtInst>(DU.NarrowUse)) {
    Value *NewDef = DU.WideDef;
    if (DU.NarrowUse->getType() != WideType) {
      unsigned CastWidth = SE->getTypeSizeInBits(DU.NarrowUse->getType());
      unsigned IVWidth = SE->getTypeSizeInBits(WideType);
      if (CastWidth < IVWidth) {
        // The cast isn't as wide as the IV, so insert a Trunc.
        IRBuilder<> Builder(DU.NarrowUse);
        NewDef = Builder.CreateTrunc(DU.WideDef, DU.NarrowUse->getType());
      }
      else {
        // A wider extend was hidden behind a narrower one. This may induce
        // another round of IV widening in which the intermediate IV becomes
        // dead. It should be very rare.
        DEBUG(dbgs() << "INDVARS: New IV " << *WidePhi
              << " not wide enough to subsume " << *DU.NarrowUse << "\n");
        DU.NarrowUse->replaceUsesOfWith(DU.NarrowDef, DU.WideDef);
        NewDef = DU.NarrowUse;
      }
    }
    if (NewDef != DU.NarrowUse) {
      DEBUG(dbgs() << "INDVARS: eliminating " << *DU.NarrowUse
            << " replaced by " << *DU.WideDef << "\n");
      ++NumElimExt;
      DU.NarrowUse->replaceAllUsesWith(NewDef);
      DeadInsts.emplace_back(DU.NarrowUse);
    }
    // Now that the extend is gone, we want to expose it's uses for potential
    // further simplification. We don't need to directly inform SimplifyIVUsers
    // of the new users, because their parent IV will be processed later as a
    // new loop phi. If we preserved IVUsers analysis, we would also want to
    // push the uses of WideDef here.

    // No further widening is needed. The deceased [sz]ext had done it for us.
    return nullptr;
  }

  // Does this user itself evaluate to a recurrence after widening?
  const SCEVAddRecExpr *WideAddRec = GetWideRecurrence(DU.NarrowUse);
  if (!WideAddRec)
    WideAddRec = GetExtendedOperandRecurrence(DU);

  if (!WideAddRec) {
    // If use is a loop condition, try to promote the condition instead of
    // truncating the IV first.
    if (WidenLoopCompare(DU))
      return nullptr;

    // This user does not evaluate to a recurence after widening, so don't
    // follow it. Instead insert a Trunc to kill off the original use,
    // eventually isolating the original narrow IV so it can be removed.
    truncateIVUse(DU, DT);
    return nullptr;
  }
  // Assume block terminators cannot evaluate to a recurrence. We can't to
  // insert a Trunc after a terminator if there happens to be a critical edge.
  assert(DU.NarrowUse != DU.NarrowUse->getParent()->getTerminator() &&
         "SCEV is not expected to evaluate a block terminator");

  // Reuse the IV increment that SCEVExpander created as long as it dominates
  // NarrowUse.
  Instruction *WideUse = nullptr;
  if (WideAddRec == WideIncExpr
      && Rewriter.hoistIVInc(WideInc, DU.NarrowUse))
    WideUse = WideInc;
  else {
    WideUse = CloneIVUser(DU);
    if (!WideUse)
      return nullptr;
  }
  // Evaluation of WideAddRec ensured that the narrow expression could be
  // extended outside the loop without overflow. This suggests that the wide use
  // evaluates to the same expression as the extended narrow use, but doesn't
  // absolutely guarantee it. Hence the following failsafe check. In rare cases
  // where it fails, we simply throw away the newly created wide use.
  if (WideAddRec != SE->getSCEV(WideUse)) {
    DEBUG(dbgs() << "Wide use expression mismatch: " << *WideUse
          << ": " << *SE->getSCEV(WideUse) << " != " << *WideAddRec << "\n");
    DeadInsts.emplace_back(WideUse);
    return nullptr;
  }

  // Returning WideUse pushes it on the worklist.
  return WideUse;
}

/// pushNarrowIVUsers - Add eligible users of NarrowDef to NarrowIVUsers.
///
void WidenIV::pushNarrowIVUsers(Instruction *NarrowDef, Instruction *WideDef) {
  for (User *U : NarrowDef->users()) {
    Instruction *NarrowUser = cast<Instruction>(U);

    // Handle data flow merges and bizarre phi cycles.
    if (!Widened.insert(NarrowUser).second)
      continue;

    NarrowIVUsers.push_back(NarrowIVDefUse(NarrowDef, NarrowUser, WideDef));
  }
}

/// CreateWideIV - Process a single induction variable. First use the
/// SCEVExpander to create a wide induction variable that evaluates to the same
/// recurrence as the original narrow IV. Then use a worklist to forward
/// traverse the narrow IV's def-use chain. After WidenIVUse has processed all
/// interesting IV users, the narrow IV will be isolated for removal by
/// DeleteDeadPHIs.
///
/// It would be simpler to delete uses as they are processed, but we must avoid
/// invalidating SCEV expressions.
///
PHINode *WidenIV::CreateWideIV(SCEVExpander &Rewriter) {
  // Is this phi an induction variable?
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(OrigPhi));
  if (!AddRec)
    return nullptr;

  // Widen the induction variable expression.
  const SCEV *WideIVExpr = IsSigned ?
    SE->getSignExtendExpr(AddRec, WideType) :
    SE->getZeroExtendExpr(AddRec, WideType);

  assert(SE->getEffectiveSCEVType(WideIVExpr->getType()) == WideType &&
         "Expect the new IV expression to preserve its type");

  // Can the IV be extended outside the loop without overflow?
  AddRec = dyn_cast<SCEVAddRecExpr>(WideIVExpr);
  if (!AddRec || AddRec->getLoop() != L)
    return nullptr;

  // An AddRec must have loop-invariant operands. Since this AddRec is
  // materialized by a loop header phi, the expression cannot have any post-loop
  // operands, so they must dominate the loop header.
  assert(SE->properlyDominates(AddRec->getStart(), L->getHeader()) &&
         SE->properlyDominates(AddRec->getStepRecurrence(*SE), L->getHeader())
         && "Loop header phi recurrence inputs do not dominate the loop");

  // The rewriter provides a value for the desired IV expression. This may
  // either find an existing phi or materialize a new one. Either way, we
  // expect a well-formed cyclic phi-with-increments. i.e. any operand not part
  // of the phi-SCC dominates the loop entry.
  Instruction *InsertPt = L->getHeader()->begin();
  WidePhi = cast<PHINode>(Rewriter.expandCodeFor(AddRec, WideType, InsertPt));

  // Remembering the WideIV increment generated by SCEVExpander allows
  // WidenIVUse to reuse it when widening the narrow IV's increment. We don't
  // employ a general reuse mechanism because the call above is the only call to
  // SCEVExpander. Henceforth, we produce 1-to-1 narrow to wide uses.
  if (BasicBlock *LatchBlock = L->getLoopLatch()) {
    WideInc =
      cast<Instruction>(WidePhi->getIncomingValueForBlock(LatchBlock));
    WideIncExpr = SE->getSCEV(WideInc);
  }

  DEBUG(dbgs() << "Wide IV: " << *WidePhi << "\n");
  ++NumWidened;

  // Traverse the def-use chain using a worklist starting at the original IV.
  assert(Widened.empty() && NarrowIVUsers.empty() && "expect initial state" );

  Widened.insert(OrigPhi);
  pushNarrowIVUsers(OrigPhi, WidePhi);

  while (!NarrowIVUsers.empty()) {
    NarrowIVDefUse DU = NarrowIVUsers.pop_back_val();

    // Process a def-use edge. This may replace the use, so don't hold a
    // use_iterator across it.
    Instruction *WideUse = WidenIVUse(DU, Rewriter);

    // Follow all def-use edges from the previous narrow use.
    if (WideUse)
      pushNarrowIVUsers(DU.NarrowUse, WideUse);

    // WidenIVUse may have removed the def-use edge.
    if (DU.NarrowDef->use_empty())
      DeadInsts.emplace_back(DU.NarrowDef);
  }
  return WidePhi;
}

//===----------------------------------------------------------------------===//
//  Live IV Reduction - Minimize IVs live across the loop.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//  Simplification of IV users based on SCEV evaluation.
//===----------------------------------------------------------------------===//

namespace {
  class IndVarSimplifyVisitor : public IVVisitor {
    ScalarEvolution *SE;
    const TargetTransformInfo *TTI;
    PHINode *IVPhi;

  public:
    WideIVInfo WI;

    IndVarSimplifyVisitor(PHINode *IV, ScalarEvolution *SCEV,
                          const TargetTransformInfo *TTI,
                          const DominatorTree *DTree)
        : SE(SCEV), TTI(TTI), IVPhi(IV) {
      DT = DTree;
      WI.NarrowIV = IVPhi;
      if (ReduceLiveIVs)
        setSplitOverflowIntrinsics();
    }

    // Implement the interface used by simplifyUsersOfIV.
    void visitCast(CastInst *Cast) override { visitIVCast(Cast, WI, SE, TTI); }
  };
}

/// SimplifyAndExtend - Iteratively perform simplification on a worklist of IV
/// users. Each successive simplification may push more users which may
/// themselves be candidates for simplification.
///
/// Sign/Zero extend elimination is interleaved with IV simplification.
///
void IndVarSimplify::SimplifyAndExtend(Loop *L,
                                       SCEVExpander &Rewriter,
                                       LPPassManager &LPM) {
  SmallVector<WideIVInfo, 8> WideIVs;

  SmallVector<PHINode*, 8> LoopPhis;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    LoopPhis.push_back(cast<PHINode>(I));
  }
  // Each round of simplification iterates through the SimplifyIVUsers worklist
  // for all current phis, then determines whether any IVs can be
  // widened. Widening adds new phis to LoopPhis, inducing another round of
  // simplification on the wide IVs.
  while (!LoopPhis.empty()) {
    // Evaluate as many IV expressions as possible before widening any IVs. This
    // forces SCEV to set no-wrap flags before evaluating sign/zero
    // extension. The first time SCEV attempts to normalize sign/zero extension,
    // the result becomes final. So for the most predictable results, we delay
    // evaluation of sign/zero extend evaluation until needed, and avoid running
    // other SCEV based analysis prior to SimplifyAndExtend.
    do {
      PHINode *CurrIV = LoopPhis.pop_back_val();

      // Information about sign/zero extensions of CurrIV.
      IndVarSimplifyVisitor Visitor(CurrIV, SE, TTI, DT);

      Changed |= simplifyUsersOfIV(CurrIV, SE, &LPM, DeadInsts, &Visitor);

      // HLSL change begin - don't widen type for hlsl.
      //if (Visitor.WI.WidestNativeType) {
      //  WideIVs.push_back(Visitor.WI);
      //}
      // HLSL change end.
    } while(!LoopPhis.empty());

    for (; !WideIVs.empty(); WideIVs.pop_back()) {
      WidenIV Widener(WideIVs.back(), LI, SE, DT, DeadInsts);
      if (PHINode *WidePhi = Widener.CreateWideIV(Rewriter)) {
        Changed = true;
        LoopPhis.push_back(WidePhi);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//  LinearFunctionTestReplace and its kin. Rewrite the loop exit condition.
//===----------------------------------------------------------------------===//

/// canExpandBackedgeTakenCount - Return true if this loop's backedge taken
/// count expression can be safely and cheaply expanded into an instruction
/// sequence that can be used by LinearFunctionTestReplace.
///
/// TODO: This fails for pointer-type loop counters with greater than one byte
/// strides, consequently preventing LFTR from running. For the purpose of LFTR
/// we could skip this check in the case that the LFTR loop counter (chosen by
/// FindLoopCounter) is also pointer type. Instead, we could directly convert
/// the loop test to an inequality test by checking the target data's alignment
/// of element types (given that the initial pointer value originates from or is
/// used by ABI constrained operation, as opposed to inttoptr/ptrtoint).
/// However, we don't yet have a strong motivation for converting loop tests
/// into inequality tests.
static bool canExpandBackedgeTakenCount(Loop *L, ScalarEvolution *SE,
                                        SCEVExpander &Rewriter) {
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount) ||
      BackedgeTakenCount->isZero())
    return false;

  if (!L->getExitingBlock())
    return false;

  // Can't rewrite non-branch yet.
  if (!isa<BranchInst>(L->getExitingBlock()->getTerminator()))
    return false;

  if (Rewriter.isHighCostExpansion(BackedgeTakenCount, L))
    return false;

  return true;
}

/// getLoopPhiForCounter - Return the loop header phi IFF IncV adds a loop
/// invariant value to the phi.
static PHINode *getLoopPhiForCounter(Value *IncV, Loop *L, DominatorTree *DT) {
  Instruction *IncI = dyn_cast<Instruction>(IncV);
  if (!IncI)
    return nullptr;

  switch (IncI->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
    break;
  case Instruction::GetElementPtr:
    // An IV counter must preserve its type.
    if (IncI->getNumOperands() == 2)
      break;
  default:
    return nullptr;
  }

  PHINode *Phi = dyn_cast<PHINode>(IncI->getOperand(0));
  if (Phi && Phi->getParent() == L->getHeader()) {
    if (isLoopInvariant(IncI->getOperand(1), L, DT))
      return Phi;
    return nullptr;
  }
  if (IncI->getOpcode() == Instruction::GetElementPtr)
    return nullptr;

  // Allow add/sub to be commuted.
  Phi = dyn_cast<PHINode>(IncI->getOperand(1));
  if (Phi && Phi->getParent() == L->getHeader()) {
    if (isLoopInvariant(IncI->getOperand(0), L, DT))
      return Phi;
  }
  return nullptr;
}

/// Return the compare guarding the loop latch, or NULL for unrecognized tests.
static ICmpInst *getLoopTest(Loop *L) {
  assert(L->getExitingBlock() && "expected loop exit");

  BasicBlock *LatchBlock = L->getLoopLatch();
  // Don't bother with LFTR if the loop is not properly simplified.
  if (!LatchBlock)
    return nullptr;

  BranchInst *BI = dyn_cast<BranchInst>(L->getExitingBlock()->getTerminator());
  assert(BI && "expected exit branch");

  return dyn_cast<ICmpInst>(BI->getCondition());
}

/// needsLFTR - LinearFunctionTestReplace policy. Return true unless we can show
/// that the current exit test is already sufficiently canonical.
static bool needsLFTR(Loop *L, DominatorTree *DT) {
  // Do LFTR to simplify the exit condition to an ICMP.
  ICmpInst *Cond = getLoopTest(L);
  if (!Cond)
    return true;

  // Do LFTR to simplify the exit ICMP to EQ/NE
  ICmpInst::Predicate Pred = Cond->getPredicate();
  if (Pred != ICmpInst::ICMP_NE && Pred != ICmpInst::ICMP_EQ)
    return true;

  // Look for a loop invariant RHS
  Value *LHS = Cond->getOperand(0);
  Value *RHS = Cond->getOperand(1);
  if (!isLoopInvariant(RHS, L, DT)) {
    if (!isLoopInvariant(LHS, L, DT))
      return true;
    std::swap(LHS, RHS);
  }
  // Look for a simple IV counter LHS
  PHINode *Phi = dyn_cast<PHINode>(LHS);
  if (!Phi)
    Phi = getLoopPhiForCounter(LHS, L, DT);

  if (!Phi)
    return true;

  // Do LFTR if PHI node is defined in the loop, but is *not* a counter.
  int Idx = Phi->getBasicBlockIndex(L->getLoopLatch());
  if (Idx < 0)
    return true;

  // Do LFTR if the exit condition's IV is *not* a simple counter.
  Value *IncV = Phi->getIncomingValue(Idx);
  return Phi != getLoopPhiForCounter(IncV, L, DT);
}

/// Recursive helper for hasConcreteDef(). Unfortunately, this currently boils
/// down to checking that all operands are constant and listing instructions
/// that may hide undef.
static bool hasConcreteDefImpl(Value *V, SmallPtrSetImpl<Value*> &Visited,
                               unsigned Depth) {
  if (isa<Constant>(V))
    return !isa<UndefValue>(V);

  if (Depth >= 6)
    return false;

  // Conservatively handle non-constant non-instructions. For example, Arguments
  // may be undef.
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  // Load and return values may be undef.
  if(I->mayReadFromMemory() || isa<CallInst>(I) || isa<InvokeInst>(I))
    return false;

  // Optimistically handle other instructions.
  for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI) {
    if (!Visited.insert(*OI).second)
      continue;
    if (!hasConcreteDefImpl(*OI, Visited, Depth+1))
      return false;
  }
  return true;
}

/// Return true if the given value is concrete. We must prove that undef can
/// never reach it.
///
/// TODO: If we decide that this is a good approach to checking for undef, we
/// may factor it into a common location.
static bool hasConcreteDef(Value *V) {
  SmallPtrSet<Value*, 8> Visited;
  Visited.insert(V);
  return hasConcreteDefImpl(V, Visited, 0);
}

/// AlmostDeadIV - Return true if this IV has any uses other than the (soon to
/// be rewritten) loop exit test.
static bool AlmostDeadIV(PHINode *Phi, BasicBlock *LatchBlock, Value *Cond) {
  int LatchIdx = Phi->getBasicBlockIndex(LatchBlock);
  Value *IncV = Phi->getIncomingValue(LatchIdx);

  for (User *U : Phi->users())
    if (U != Cond && U != IncV) return false;

  for (User *U : IncV->users())
    if (U != Cond && U != Phi) return false;
  return true;
}

/// FindLoopCounter - Find an affine IV in canonical form.
///
/// BECount may be an i8* pointer type. The pointer difference is already
/// valid count without scaling the address stride, so it remains a pointer
/// expression as far as SCEV is concerned.
///
/// Currently only valid for LFTR. See the comments on hasConcreteDef below.
///
/// FIXME: Accept -1 stride and set IVLimit = IVInit - BECount
///
/// FIXME: Accept non-unit stride as long as SCEV can reduce BECount * Stride.
/// This is difficult in general for SCEV because of potential overflow. But we
/// could at least handle constant BECounts.
static PHINode *FindLoopCounter(Loop *L, const SCEV *BECount,
                                ScalarEvolution *SE, DominatorTree *DT) {
  uint64_t BCWidth = SE->getTypeSizeInBits(BECount->getType());

  Value *Cond =
    cast<BranchInst>(L->getExitingBlock()->getTerminator())->getCondition();

  // Loop over all of the PHI nodes, looking for a simple counter.
  PHINode *BestPhi = nullptr;
  const SCEV *BestInit = nullptr;
  BasicBlock *LatchBlock = L->getLoopLatch();
  assert(LatchBlock && "needsLFTR should guarantee a loop latch");

  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *Phi = cast<PHINode>(I);
    if (!SE->isSCEVable(Phi->getType()))
      continue;

    // Avoid comparing an integer IV against a pointer Limit.
    if (BECount->getType()->isPointerTy() && !Phi->getType()->isPointerTy())
      continue;

    const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Phi));
    if (!AR || AR->getLoop() != L || !AR->isAffine())
      continue;

    // AR may be a pointer type, while BECount is an integer type.
    // AR may be wider than BECount. With eq/ne tests overflow is immaterial.
    // AR may not be a narrower type, or we may never exit.
    uint64_t PhiWidth = SE->getTypeSizeInBits(AR->getType());
    if (PhiWidth < BCWidth ||
        !L->getHeader()->getModule()->getDataLayout().isLegalInteger(PhiWidth))
      continue;

    const SCEV *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
    if (!Step || !Step->isOne())
      continue;

    int LatchIdx = Phi->getBasicBlockIndex(LatchBlock);
    Value *IncV = Phi->getIncomingValue(LatchIdx);
    if (getLoopPhiForCounter(IncV, L, DT) != Phi)
      continue;

    // Avoid reusing a potentially undef value to compute other values that may
    // have originally had a concrete definition.
    if (!hasConcreteDef(Phi)) {
      // We explicitly allow unknown phis as long as they are already used by
      // the loop test. In this case we assume that performing LFTR could not
      // increase the number of undef users.
      if (ICmpInst *Cond = getLoopTest(L)) {
        if (Phi != getLoopPhiForCounter(Cond->getOperand(0), L, DT)
            && Phi != getLoopPhiForCounter(Cond->getOperand(1), L, DT)) {
          continue;
        }
      }
    }
    const SCEV *Init = AR->getStart();

    if (BestPhi && !AlmostDeadIV(BestPhi, LatchBlock, Cond)) {
      // Don't force a live loop counter if another IV can be used.
      if (AlmostDeadIV(Phi, LatchBlock, Cond))
        continue;

      // Prefer to count-from-zero. This is a more "canonical" counter form. It
      // also prefers integer to pointer IVs.
      if (BestInit->isZero() != Init->isZero()) {
        if (BestInit->isZero())
          continue;
      }
      // If two IVs both count from zero or both count from nonzero then the
      // narrower is likely a dead phi that has been widened. Use the wider phi
      // to allow the other to be eliminated.
      else if (PhiWidth <= SE->getTypeSizeInBits(BestPhi->getType()))
        continue;
    }
    BestPhi = Phi;
    BestInit = Init;
  }
  return BestPhi;
}

/// genLoopLimit - Help LinearFunctionTestReplace by generating a value that
/// holds the RHS of the new loop test.
static Value *genLoopLimit(PHINode *IndVar, const SCEV *IVCount, Loop *L,
                           SCEVExpander &Rewriter, ScalarEvolution *SE) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(IndVar));
  assert(AR && AR->getLoop() == L && AR->isAffine() && "bad loop counter");
  const SCEV *IVInit = AR->getStart();

  // IVInit may be a pointer while IVCount is an integer when FindLoopCounter
  // finds a valid pointer IV. Sign extend BECount in order to materialize a
  // GEP. Avoid running SCEVExpander on a new pointer value, instead reusing
  // the existing GEPs whenever possible.
  if (IndVar->getType()->isPointerTy()
      && !IVCount->getType()->isPointerTy()) {

    // IVOffset will be the new GEP offset that is interpreted by GEP as a
    // signed value. IVCount on the other hand represents the loop trip count,
    // which is an unsigned value. FindLoopCounter only allows induction
    // variables that have a positive unit stride of one. This means we don't
    // have to handle the case of negative offsets (yet) and just need to zero
    // extend IVCount.
    Type *OfsTy = SE->getEffectiveSCEVType(IVInit->getType());
    const SCEV *IVOffset = SE->getTruncateOrZeroExtend(IVCount, OfsTy);

    // Expand the code for the iteration count.
    assert(SE->isLoopInvariant(IVOffset, L) &&
           "Computed iteration count is not loop invariant!");
    BranchInst *BI = cast<BranchInst>(L->getExitingBlock()->getTerminator());
    Value *GEPOffset = Rewriter.expandCodeFor(IVOffset, OfsTy, BI);

    Value *GEPBase = IndVar->getIncomingValueForBlock(L->getLoopPreheader());
    assert(AR->getStart() == SE->getSCEV(GEPBase) && "bad loop counter");
    // We could handle pointer IVs other than i8*, but we need to compensate for
    // gep index scaling. See canExpandBackedgeTakenCount comments.
    assert(SE->getSizeOfExpr(IntegerType::getInt64Ty(IndVar->getContext()),
             cast<PointerType>(GEPBase->getType())->getElementType())->isOne()
           && "unit stride pointer IV must be i8*");

    IRBuilder<> Builder(L->getLoopPreheader()->getTerminator());
    return Builder.CreateGEP(nullptr, GEPBase, GEPOffset, "lftr.limit");
  }
  else {
    // In any other case, convert both IVInit and IVCount to integers before
    // comparing. This may result in SCEV expension of pointers, but in practice
    // SCEV will fold the pointer arithmetic away as such:
    // BECount = (IVEnd - IVInit - 1) => IVLimit = IVInit (postinc).
    //
    // Valid Cases: (1) both integers is most common; (2) both may be pointers
    // for simple memset-style loops.
    //
    // IVInit integer and IVCount pointer would only occur if a canonical IV
    // were generated on top of case #2, which is not expected.

    const SCEV *IVLimit = nullptr;
    // For unit stride, IVCount = Start + BECount with 2's complement overflow.
    // For non-zero Start, compute IVCount here.
    if (AR->getStart()->isZero())
      IVLimit = IVCount;
    else {
      assert(AR->getStepRecurrence(*SE)->isOne() && "only handles unit stride");
      const SCEV *IVInit = AR->getStart();

      // For integer IVs, truncate the IV before computing IVInit + BECount.
      if (SE->getTypeSizeInBits(IVInit->getType())
          > SE->getTypeSizeInBits(IVCount->getType()))
        IVInit = SE->getTruncateExpr(IVInit, IVCount->getType());

      IVLimit = SE->getAddExpr(IVInit, IVCount);
    }
    // Expand the code for the iteration count.
    BranchInst *BI = cast<BranchInst>(L->getExitingBlock()->getTerminator());
    IRBuilder<> Builder(BI);
    assert(SE->isLoopInvariant(IVLimit, L) &&
           "Computed iteration count is not loop invariant!");
    // Ensure that we generate the same type as IndVar, or a smaller integer
    // type. In the presence of null pointer values, we have an integer type
    // SCEV expression (IVInit) for a pointer type IV value (IndVar).
    Type *LimitTy = IVCount->getType()->isPointerTy() ?
      IndVar->getType() : IVCount->getType();
    return Rewriter.expandCodeFor(IVLimit, LimitTy, BI);
  }
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
Value *IndVarSimplify::
LinearFunctionTestReplace(Loop *L,
                          const SCEV *BackedgeTakenCount,
                          PHINode *IndVar,
                          SCEVExpander &Rewriter) {
  assert(canExpandBackedgeTakenCount(L, SE, Rewriter) && "precondition");

  // Initialize CmpIndVar and IVCount to their preincremented values.
  Value *CmpIndVar = IndVar;
  const SCEV *IVCount = BackedgeTakenCount;

  // If the exiting block is the same as the backedge block, we prefer to
  // compare against the post-incremented value, otherwise we must compare
  // against the preincremented value.
  if (L->getExitingBlock() == L->getLoopLatch()) {
    // Add one to the "backedge-taken" count to get the trip count.
    // This addition may overflow, which is valid as long as the comparison is
    // truncated to BackedgeTakenCount->getType().
    IVCount = SE->getAddExpr(BackedgeTakenCount,
                             SE->getConstant(BackedgeTakenCount->getType(), 1));
    // The BackedgeTaken expression contains the number of times that the
    // backedge branches to the loop header.  This is one less than the
    // number of times the loop executes, so use the incremented indvar.
    CmpIndVar = IndVar->getIncomingValueForBlock(L->getExitingBlock());
  }

  Value *ExitCnt = genLoopLimit(IndVar, IVCount, L, Rewriter, SE);
  assert(ExitCnt->getType()->isPointerTy() == IndVar->getType()->isPointerTy()
         && "genLoopLimit missed a cast");

  // Insert a new icmp_ne or icmp_eq instruction before the branch.
  BranchInst *BI = cast<BranchInst>(L->getExitingBlock()->getTerminator());
  ICmpInst::Predicate P;
  if (L->contains(BI->getSuccessor(0)))
    P = ICmpInst::ICMP_NE;
  else
    P = ICmpInst::ICMP_EQ;

  DEBUG(dbgs() << "INDVARS: Rewriting loop exit condition to:\n"
               << "      LHS:" << *CmpIndVar << '\n'
               << "       op:\t"
               << (P == ICmpInst::ICMP_NE ? "!=" : "==") << "\n"
               << "      RHS:\t" << *ExitCnt << "\n"
               << "  IVCount:\t" << *IVCount << "\n");

  IRBuilder<> Builder(BI);

  // LFTR can ignore IV overflow and truncate to the width of
  // BECount. This avoids materializing the add(zext(add)) expression.
  unsigned CmpIndVarSize = SE->getTypeSizeInBits(CmpIndVar->getType());
  unsigned ExitCntSize = SE->getTypeSizeInBits(ExitCnt->getType());
  if (CmpIndVarSize > ExitCntSize) {
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(SE->getSCEV(IndVar));
    const SCEV *ARStart = AR->getStart();
    const SCEV *ARStep = AR->getStepRecurrence(*SE);
    // For constant IVCount, avoid truncation.
    if (isa<SCEVConstant>(ARStart) && isa<SCEVConstant>(IVCount)) {
      const APInt &Start = cast<SCEVConstant>(ARStart)->getValue()->getValue();
      APInt Count = cast<SCEVConstant>(IVCount)->getValue()->getValue();
      // Note that the post-inc value of BackedgeTakenCount may have overflowed
      // above such that IVCount is now zero.
      if (IVCount != BackedgeTakenCount && Count == 0) {
        Count = APInt::getMaxValue(Count.getBitWidth()).zext(CmpIndVarSize);
        ++Count;
      }
      else
        Count = Count.zext(CmpIndVarSize);
      APInt NewLimit;
      if (cast<SCEVConstant>(ARStep)->getValue()->isNegative())
        NewLimit = Start - Count;
      else
        NewLimit = Start + Count;
      ExitCnt = ConstantInt::get(CmpIndVar->getType(), NewLimit);

      DEBUG(dbgs() << "  Widen RHS:\t" << *ExitCnt << "\n");
    } else {
      CmpIndVar = Builder.CreateTrunc(CmpIndVar, ExitCnt->getType(),
                                      "lftr.wideiv");
    }
  }
  Value *Cond = Builder.CreateICmp(P, CmpIndVar, ExitCnt, "exitcond");
  Value *OrigCond = BI->getCondition();
  // It's tempting to use replaceAllUsesWith here to fully replace the old
  // comparison, but that's not immediately safe, since users of the old
  // comparison may not be dominated by the new comparison. Instead, just
  // update the branch to use the new comparison; in the common case this
  // will make old comparison dead.
  BI->setCondition(Cond);
  DeadInsts.push_back(OrigCond);

  ++NumLFTR;
  Changed = true;
  return Cond;
}

//===----------------------------------------------------------------------===//
//  SinkUnusedInvariants. A late subpass to cleanup loop preheaders.
//===----------------------------------------------------------------------===//

/// If there's a single exit block, sink any loop-invariant values that
/// were defined in the preheader but not used inside the loop into the
/// exit block to reduce register pressure in the loop.
void IndVarSimplify::SinkUnusedInvariants(Loop *L) {
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock) return;

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) return;

  Instruction *InsertPt = ExitBlock->getFirstInsertionPt();
  BasicBlock::iterator I = Preheader->getTerminator();
  while (I != Preheader->begin()) {
    --I;
    // New instructions were inserted at the end of the preheader.
    if (isa<PHINode>(I))
      break;

    // Don't move instructions which might have side effects, since the side
    // effects need to complete before instructions inside the loop.  Also don't
    // move instructions which might read memory, since the loop may modify
    // memory. Note that it's okay if the instruction might have undefined
    // behavior: LoopSimplify guarantees that the preheader dominates the exit
    // block.
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      continue;

    // Skip debug info intrinsics.
    if (isa<DbgInfoIntrinsic>(I))
      continue;

    // Skip landingpad instructions.
    if (isa<LandingPadInst>(I))
      continue;

    // Don't sink alloca: we never want to sink static alloca's out of the
    // entry block, and correctly sinking dynamic alloca's requires
    // checks for stacksave/stackrestore intrinsics.
    // FIXME: Refactor this check somehow?
    if (isa<AllocaInst>(I))
      continue;

    // Determine if there is a use in or before the loop (direct or
    // otherwise).
    bool UsedInLoop = false;
    for (Use &U : I->uses()) {
      Instruction *User = cast<Instruction>(U.getUser());
      BasicBlock *UseBB = User->getParent();
      if (PHINode *P = dyn_cast<PHINode>(User)) {
        unsigned i =
          PHINode::getIncomingValueNumForOperand(U.getOperandNo());
        UseBB = P->getIncomingBlock(i);
      }
      if (UseBB == Preheader || L->contains(UseBB)) {
        UsedInLoop = true;
        break;
      }
    }

    // If there is, the def must remain in the preheader.
    if (UsedInLoop)
      continue;

    // Otherwise, sink it to the exit block.
    Instruction *ToMove = I;
    bool Done = false;

    if (I != Preheader->begin()) {
      // Skip debug info intrinsics.
      do {
        --I;
      } while (isa<DbgInfoIntrinsic>(I) && I != Preheader->begin());

      if (isa<DbgInfoIntrinsic>(I) && I == Preheader->begin())
        Done = true;
    } else {
      Done = true;
    }

    ToMove->moveBefore(InsertPt);
    if (Done) break;
    InsertPt = ToMove;
  }
}

//===----------------------------------------------------------------------===//
//  IndVarSimplify driver. Manage several subpasses of IV simplification.
//===----------------------------------------------------------------------===//

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  // If LoopSimplify form is not available, stay out of trouble. Some notes:
  //  - LSR currently only supports LoopSimplify-form loops. Indvars'
  //    canonicalization can be a pessimization without LSR to "clean up"
  //    afterwards.
  //  - We depend on having a preheader; in particular,
  //    Loop::getCanonicalInductionVariable only supports loops with preheaders,
  //    and we're in trouble if we can't find the induction variable even when
  //    we've manually inserted one.
  if (!L->isLoopSimplifyForm())
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  TLI = TLIP ? &TLIP->getTLI() : nullptr;
  auto *TTIP = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
  TTI = TTIP ? &TTIP->getTTI(*L->getHeader()->getParent()) : nullptr;
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  DeadInsts.clear();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE, DL, "indvars");
#ifndef NDEBUG
  Rewriter.setDebugType(DEBUG_TYPE);
#endif

  // Eliminate redundant IV users.
  //
  // Simplification works best when run before other consumers of SCEV. We
  // attempt to avoid evaluating SCEVs for sign/zero extend operations until
  // other expressions involving loop IVs have been evaluated. This helps SCEV
  // set no-wrap flags before normalizing sign/zero extension.
  Rewriter.disableCanonicalMode();
  SimplifyAndExtend(L, Rewriter, LPM);

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  if (ReplaceExitValue != NeverRepl &&
      !isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, Rewriter);

  // Eliminate redundant IV cycles.
  NumElimIV += Rewriter.replaceCongruentIVs(L, DT, DeadInsts);

  // If we have a trip count expression, rewrite the loop's exit condition
  // using it.  We can currently only handle loops with a single exit.
  if (canExpandBackedgeTakenCount(L, SE, Rewriter) && needsLFTR(L, DT)) {
    PHINode *IndVar = FindLoopCounter(L, BackedgeTakenCount, SE, DT);
    if (IndVar) {
      // Check preconditions for proper SCEVExpander operation. SCEV does not
      // express SCEVExpander's dependencies, such as LoopSimplify. Instead any
      // pass that uses the SCEVExpander must do it. This does not work well for
      // loop passes because SCEVExpander makes assumptions about all loops,
      // while LoopPassManager only forces the current loop to be simplified.
      //
      // FIXME: SCEV expansion has no way to bail out, so the caller must
      // explicitly check any assumptions made by SCEV. Brittle.
      const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(BackedgeTakenCount);
      if (!AR || AR->getLoop()->getLoopPreheader())
        (void)LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                        Rewriter);
    }
  }
  // Clear the rewriter cache, because values that are in the rewriter's cache
  // can be deleted in the loop below, causing the AssertingVH in the cache to
  // trigger.
  Rewriter.clear();

  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty())
    if (Instruction *Inst =
            dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val()))
      RecursivelyDeleteTriviallyDeadInstructions(Inst, TLI);

  // The Rewriter may not be used from this point on.

  // Loop-invariant instructions in the preheader that aren't used in the
  // loop may be sunk below the loop to reduce register pressure.
  SinkUnusedInvariants(L);

  // Clean up dead instructions.
  Changed |= DeleteDeadPHIs(L->getHeader(), TLI);
  // Check a post-condition.
  assert(L->isLCSSAForm(*DT) &&
         "Indvars did not leave the loop in lcssa form!");

#if 0 // HLSL Change Starts - option pending
  // Verify that LFTR, and any other change have not interfered with SCEV's
  // ability to compute trip count.
#ifndef NDEBUG
  if (VerifyIndvars && !isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    SE->forgetLoop(L);
    const SCEV *NewBECount = SE->getBackedgeTakenCount(L);
    if (SE->getTypeSizeInBits(BackedgeTakenCount->getType()) <
        SE->getTypeSizeInBits(NewBECount->getType()))
      NewBECount = SE->getTruncateOrNoop(NewBECount,
                                         BackedgeTakenCount->getType());
    else
      BackedgeTakenCount = SE->getTruncateOrNoop(BackedgeTakenCount,
                                                 NewBECount->getType());
    assert(BackedgeTakenCount == NewBECount && "indvars must preserve SCEV");
  }
#endif
#endif // HLSL Change Ends - option pending

  return Changed;
}
