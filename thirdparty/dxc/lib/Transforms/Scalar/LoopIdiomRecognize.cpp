//===-- LoopIdiomRecognize.cpp - Loop idiom recognition -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms simple loops into a
// non-loop form.  In cases that this kicks in, it can be a significant
// performance win.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
//
// Future loop memory idioms to recognize:
//   memcmp, memmove, strlen, etc.
// Future floating point idioms to recognize in -ffast-math mode:
//   fpowi
// Future integer operation idioms to recognize:
//   ctpop, ctlz, cttz
//
// Beware that isel's default lowering for ctpop is highly inefficient for
// i64 and larger types when i64 is legal and the value has few bits set.  It
// would be good to enhance isel to emit a loop for ctpop in this case.
//
// We should enhance the memset/memcpy recognition to handle multiple stores in
// the loop.  This would handle things like:
//   void foo(_Complex float *P)
//     for (i) { __real__(*P) = 0;  __imag__(*P) = 0; }
//
// We should enhance this to handle negative strides through memory.
// Alternatively (and perhaps better) we could rely on an earlier pass to force
// forward iteration through memory, which is generally better for cache
// behavior.  Negative strides *do* happen for memset/memcpy loops.
//
// This could recognize common matrix multiplies and dot product idioms and
// replace them with calls to BLAS (if linked in??).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "loop-idiom"

STATISTIC(NumMemSet, "Number of memset's formed from loop stores");
STATISTIC(NumMemCpy, "Number of memcpy's formed from loop load+stores");

namespace {

  class LoopIdiomRecognize;

  /// This class defines some utility functions for loop idiom recognization.
  class LIRUtil {
  public:
    /// Return true iff the block contains nothing but an uncondition branch
    /// (aka goto instruction).
    static bool isAlmostEmpty(BasicBlock *);

    static BranchInst *getBranch(BasicBlock *BB) {
      return dyn_cast<BranchInst>(BB->getTerminator());
    }

    /// Derive the precondition block (i.e the block that guards the loop
    /// preheader) from the given preheader.
    static BasicBlock *getPrecondBb(BasicBlock *PreHead);
  };

  /// This class is to recoginize idioms of population-count conducted in
  /// a noncountable loop. Currently it only recognizes this pattern:
  /// \code
  ///   while(x) {cnt++; ...; x &= x - 1; ...}
  /// \endcode
  class NclPopcountRecognize {
    LoopIdiomRecognize &LIR;
    Loop *CurLoop;
    BasicBlock *PreCondBB;

    typedef IRBuilder<> IRBuilderTy;

  public:
    explicit NclPopcountRecognize(LoopIdiomRecognize &TheLIR);
    bool recognize();

  private:
    /// Take a glimpse of the loop to see if we need to go ahead recoginizing
    /// the idiom.
    bool preliminaryScreen();

    /// Check if the given conditional branch is based on the comparison
    /// between a variable and zero, and if the variable is non-zero, the
    /// control yields to the loop entry. If the branch matches the behavior,
    /// the variable involved in the comparion is returned. This function will
    /// be called to see if the precondition and postcondition of the loop
    /// are in desirable form.
    Value *matchCondition(BranchInst *Br, BasicBlock *NonZeroTarget) const;

    /// Return true iff the idiom is detected in the loop. and 1) \p CntInst
    /// is set to the instruction counting the population bit. 2) \p CntPhi
    /// is set to the corresponding phi node. 3) \p Var is set to the value
    /// whose population bits are being counted.
    bool detectIdiom
      (Instruction *&CntInst, PHINode *&CntPhi, Value *&Var) const;

    /// Insert ctpop intrinsic function and some obviously dead instructions.
    void transform(Instruction *CntInst, PHINode *CntPhi, Value *Var);

    /// Create llvm.ctpop.* intrinsic function.
    CallInst *createPopcntIntrinsic(IRBuilderTy &IRB, Value *Val, DebugLoc DL);
  };

  class LoopIdiomRecognize : public LoopPass {
    Loop *CurLoop;
    DominatorTree *DT;
    ScalarEvolution *SE;
    TargetLibraryInfo *TLI;
    const TargetTransformInfo *TTI;
  public:
    static char ID;
    explicit LoopIdiomRecognize() : LoopPass(ID) {
      initializeLoopIdiomRecognizePass(*PassRegistry::getPassRegistry());
      DT = nullptr;
      SE = nullptr;
      TLI = nullptr;
      TTI = nullptr;
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;
    bool runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                        SmallVectorImpl<BasicBlock*> &ExitBlocks);

    bool processLoopStore(StoreInst *SI, const SCEV *BECount);
    bool processLoopMemSet(MemSetInst *MSI, const SCEV *BECount);

    bool processLoopStridedStore(Value *DestPtr, unsigned StoreSize,
                                 unsigned StoreAlignment,
                                 Value *SplatValue, Instruction *TheStore,
                                 const SCEVAddRecExpr *Ev,
                                 const SCEV *BECount);
    bool processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                                    const SCEVAddRecExpr *StoreEv,
                                    const SCEVAddRecExpr *LoadEv,
                                    const SCEV *BECount);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
    }

    DominatorTree *getDominatorTree() {
      return DT ? DT
                : (DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree());
    }

    ScalarEvolution *getScalarEvolution() {
      return SE ? SE : (SE = &getAnalysis<ScalarEvolution>());
    }

    TargetLibraryInfo *getTargetLibraryInfo() {
      if (!TLI)
        TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

      return TLI;
    }

    const TargetTransformInfo *getTargetTransformInfo() {
      return TTI ? TTI
                 : (TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
                        *CurLoop->getHeader()->getParent()));
    }

    Loop *getLoop() const { return CurLoop; }

  private:
    bool runOnNoncountableLoop();
    bool runOnCountableLoop();
  };
}

char LoopIdiomRecognize::ID = 0;
INITIALIZE_PASS_BEGIN(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                    false, false)

Pass *llvm::createLoopIdiomPass() { return new LoopIdiomRecognize(); }

/// deleteDeadInstruction - Delete this instruction.  Before we do, go through
/// and zero out all the operands of this instruction.  If any of them become
/// dead, delete them and the computation tree that feeds them.
///
static void deleteDeadInstruction(Instruction *I,
                                  const TargetLibraryInfo *TLI) {
  SmallVector<Value *, 16> Operands(I->value_op_begin(), I->value_op_end());
  I->replaceAllUsesWith(UndefValue::get(I->getType()));
  I->eraseFromParent();
  for (Value *Op : Operands)
    RecursivelyDeleteTriviallyDeadInstructions(Op, TLI);
}

//===----------------------------------------------------------------------===//
//
//          Implementation of LIRUtil
//
//===----------------------------------------------------------------------===//

// This function will return true iff the given block contains nothing but goto.
// A typical usage of this function is to check if the preheader function is
// "almost" empty such that generated intrinsic functions can be moved across
// the preheader and be placed at the end of the precondition block without
// the concern of breaking data dependence.
bool LIRUtil::isAlmostEmpty(BasicBlock *BB) {
  if (BranchInst *Br = getBranch(BB)) {
    return Br->isUnconditional() && Br == BB->begin();
  }
  return false;
}

BasicBlock *LIRUtil::getPrecondBb(BasicBlock *PreHead) {
  if (BasicBlock *BB = PreHead->getSinglePredecessor()) {
    BranchInst *Br = getBranch(BB);
    return Br && Br->isConditional() ? BB : nullptr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
//
//          Implementation of NclPopcountRecognize
//
//===----------------------------------------------------------------------===//

NclPopcountRecognize::NclPopcountRecognize(LoopIdiomRecognize &TheLIR):
  LIR(TheLIR), CurLoop(TheLIR.getLoop()), PreCondBB(nullptr) {
}

bool NclPopcountRecognize::preliminaryScreen() {
  const TargetTransformInfo *TTI = LIR.getTargetTransformInfo();
  if (TTI->getPopcntSupport(32) != TargetTransformInfo::PSK_FastHardware)
    return false;

  // Counting population are usually conducted by few arithmetic instructions.
  // Such instructions can be easilly "absorbed" by vacant slots in a
  // non-compact loop. Therefore, recognizing popcount idiom only makes sense
  // in a compact loop.

  // Give up if the loop has multiple blocks or multiple backedges.
  if (CurLoop->getNumBackEdges() != 1 || CurLoop->getNumBlocks() != 1)
    return false;

  BasicBlock *LoopBody = *(CurLoop->block_begin());
  if (LoopBody->size() >= 20) {
    // The loop is too big, bail out.
    return false;
  }

  // It should have a preheader containing nothing but a goto instruction.
  BasicBlock *PreHead = CurLoop->getLoopPreheader();
  if (!PreHead || !LIRUtil::isAlmostEmpty(PreHead))
    return false;

  // It should have a precondition block where the generated popcount instrinsic
  // function will be inserted.
  PreCondBB = LIRUtil::getPrecondBb(PreHead);
  if (!PreCondBB)
    return false;

  return true;
}

Value *NclPopcountRecognize::matchCondition(BranchInst *Br,
                                            BasicBlock *LoopEntry) const {
  if (!Br || !Br->isConditional())
    return nullptr;

  ICmpInst *Cond = dyn_cast<ICmpInst>(Br->getCondition());
  if (!Cond)
    return nullptr;

  ConstantInt *CmpZero = dyn_cast<ConstantInt>(Cond->getOperand(1));
  if (!CmpZero || !CmpZero->isZero())
    return nullptr;

  ICmpInst::Predicate Pred = Cond->getPredicate();
  if ((Pred == ICmpInst::ICMP_NE && Br->getSuccessor(0) == LoopEntry) ||
      (Pred == ICmpInst::ICMP_EQ && Br->getSuccessor(1) == LoopEntry))
    return Cond->getOperand(0);

  return nullptr;
}

bool NclPopcountRecognize::detectIdiom(Instruction *&CntInst,
                                       PHINode *&CntPhi,
                                       Value *&Var) const {
  // Following code tries to detect this idiom:
  //
  //    if (x0 != 0)
  //      goto loop-exit // the precondition of the loop
  //    cnt0 = init-val;
  //    do {
  //       x1 = phi (x0, x2);
  //       cnt1 = phi(cnt0, cnt2);
  //
  //       cnt2 = cnt1 + 1;
  //        ...
  //       x2 = x1 & (x1 - 1);
  //        ...
  //    } while(x != 0);
  //
  // loop-exit:
  //

  // step 1: Check to see if the look-back branch match this pattern:
  //    "if (a!=0) goto loop-entry".
  BasicBlock *LoopEntry;
  Instruction *DefX2, *CountInst;
  Value *VarX1, *VarX0;
  PHINode *PhiX, *CountPhi;

  DefX2 = CountInst = nullptr;
  VarX1 = VarX0 = nullptr;
  PhiX = CountPhi = nullptr;
  LoopEntry = *(CurLoop->block_begin());

  // step 1: Check if the loop-back branch is in desirable form.
  {
    if (Value *T = matchCondition (LIRUtil::getBranch(LoopEntry), LoopEntry))
      DefX2 = dyn_cast<Instruction>(T);
    else
      return false;
  }

  // step 2: detect instructions corresponding to "x2 = x1 & (x1 - 1)"
  {
    if (!DefX2 || DefX2->getOpcode() != Instruction::And)
      return false;

    BinaryOperator *SubOneOp;

    if ((SubOneOp = dyn_cast<BinaryOperator>(DefX2->getOperand(0))))
      VarX1 = DefX2->getOperand(1);
    else {
      VarX1 = DefX2->getOperand(0);
      SubOneOp = dyn_cast<BinaryOperator>(DefX2->getOperand(1));
    }
    if (!SubOneOp)
      return false;

    Instruction *SubInst = cast<Instruction>(SubOneOp);
    ConstantInt *Dec = dyn_cast<ConstantInt>(SubInst->getOperand(1));
    if (!Dec ||
        !((SubInst->getOpcode() == Instruction::Sub && Dec->isOne()) ||
          (SubInst->getOpcode() == Instruction::Add && Dec->isAllOnesValue()))) {
      return false;
    }
  }

  // step 3: Check the recurrence of variable X
  {
    PhiX = dyn_cast<PHINode>(VarX1);
    if (!PhiX ||
        (PhiX->getOperand(0) != DefX2 && PhiX->getOperand(1) != DefX2)) {
      return false;
    }
  }

  // step 4: Find the instruction which count the population: cnt2 = cnt1 + 1
  {
    CountInst = nullptr;
    for (BasicBlock::iterator Iter = LoopEntry->getFirstNonPHI(),
           IterE = LoopEntry->end(); Iter != IterE; Iter++) {
      Instruction *Inst = Iter;
      if (Inst->getOpcode() != Instruction::Add)
        continue;

      ConstantInt *Inc = dyn_cast<ConstantInt>(Inst->getOperand(1));
      if (!Inc || !Inc->isOne())
        continue;

      PHINode *Phi = dyn_cast<PHINode>(Inst->getOperand(0));
      if (!Phi || Phi->getParent() != LoopEntry)
        continue;

      // Check if the result of the instruction is live of the loop.
      bool LiveOutLoop = false;
      for (User *U : Inst->users()) {
        if ((cast<Instruction>(U))->getParent() != LoopEntry) {
          LiveOutLoop = true; break;
        }
      }

      if (LiveOutLoop) {
        CountInst = Inst;
        CountPhi = Phi;
        break;
      }
    }

    if (!CountInst)
      return false;
  }

  // step 5: check if the precondition is in this form:
  //   "if (x != 0) goto loop-head ; else goto somewhere-we-don't-care;"
  {
    BranchInst *PreCondBr = LIRUtil::getBranch(PreCondBB);
    Value *T = matchCondition (PreCondBr, CurLoop->getLoopPreheader());
    if (T != PhiX->getOperand(0) && T != PhiX->getOperand(1))
      return false;

    CntInst = CountInst;
    CntPhi = CountPhi;
    Var = T;
  }

  return true;
}

void NclPopcountRecognize::transform(Instruction *CntInst,
                                     PHINode *CntPhi, Value *Var) {

  ScalarEvolution *SE = LIR.getScalarEvolution();
  TargetLibraryInfo *TLI = LIR.getTargetLibraryInfo();
  BasicBlock *PreHead = CurLoop->getLoopPreheader();
  BranchInst *PreCondBr = LIRUtil::getBranch(PreCondBB);
  const DebugLoc DL = CntInst->getDebugLoc();

  // Assuming before transformation, the loop is following:
  //  if (x) // the precondition
  //     do { cnt++; x &= x - 1; } while(x);

  // Step 1: Insert the ctpop instruction at the end of the precondition block
  IRBuilderTy Builder(PreCondBr);
  Value *PopCnt, *PopCntZext, *NewCount, *TripCnt;
  {
    PopCnt = createPopcntIntrinsic(Builder, Var, DL);
    NewCount = PopCntZext =
      Builder.CreateZExtOrTrunc(PopCnt, cast<IntegerType>(CntPhi->getType()));

    if (NewCount != PopCnt)
      (cast<Instruction>(NewCount))->setDebugLoc(DL);

    // TripCnt is exactly the number of iterations the loop has
    TripCnt = NewCount;

    // If the population counter's initial value is not zero, insert Add Inst.
    Value *CntInitVal = CntPhi->getIncomingValueForBlock(PreHead);
    ConstantInt *InitConst = dyn_cast<ConstantInt>(CntInitVal);
    if (!InitConst || !InitConst->isZero()) {
      NewCount = Builder.CreateAdd(NewCount, CntInitVal);
      (cast<Instruction>(NewCount))->setDebugLoc(DL);
    }
  }

  // Step 2: Replace the precondition from "if(x == 0) goto loop-exit" to
  //   "if(NewCount == 0) loop-exit". Withtout this change, the intrinsic
  //   function would be partial dead code, and downstream passes will drag
  //   it back from the precondition block to the preheader.
  {
    ICmpInst *PreCond = cast<ICmpInst>(PreCondBr->getCondition());

    Value *Opnd0 = PopCntZext;
    Value *Opnd1 = ConstantInt::get(PopCntZext->getType(), 0);
    if (PreCond->getOperand(0) != Var)
      std::swap(Opnd0, Opnd1);

    ICmpInst *NewPreCond =
      cast<ICmpInst>(Builder.CreateICmp(PreCond->getPredicate(), Opnd0, Opnd1));
    PreCondBr->setCondition(NewPreCond);

    RecursivelyDeleteTriviallyDeadInstructions(PreCond, TLI);
  }

  // Step 3: Note that the population count is exactly the trip count of the
  // loop in question, which enble us to to convert the loop from noncountable
  // loop into a countable one. The benefit is twofold:
  //
  //  - If the loop only counts population, the entire loop become dead after
  //    the transformation. It is lots easier to prove a countable loop dead
  //    than to prove a noncountable one. (In some C dialects, a infite loop
  //    isn't dead even if it computes nothing useful. In general, DCE needs
  //    to prove a noncountable loop finite before safely delete it.)
  //
  //  - If the loop also performs something else, it remains alive.
  //    Since it is transformed to countable form, it can be aggressively
  //    optimized by some optimizations which are in general not applicable
  //    to a noncountable loop.
  //
  // After this step, this loop (conceptually) would look like following:
  //   newcnt = __builtin_ctpop(x);
  //   t = newcnt;
  //   if (x)
  //     do { cnt++; x &= x-1; t--) } while (t > 0);
  BasicBlock *Body = *(CurLoop->block_begin());
  {
    BranchInst *LbBr = LIRUtil::getBranch(Body);
    ICmpInst *LbCond = cast<ICmpInst>(LbBr->getCondition());
    Type *Ty = TripCnt->getType();

    PHINode *TcPhi = PHINode::Create(Ty, 2, "tcphi", Body->begin());

    Builder.SetInsertPoint(LbCond);
    Value *Opnd1 = cast<Value>(TcPhi);
    Value *Opnd2 = cast<Value>(ConstantInt::get(Ty, 1));
    Instruction *TcDec =
      cast<Instruction>(Builder.CreateSub(Opnd1, Opnd2, "tcdec", false, true));

    TcPhi->addIncoming(TripCnt, PreHead);
    TcPhi->addIncoming(TcDec, Body);

    CmpInst::Predicate Pred = (LbBr->getSuccessor(0) == Body) ?
      CmpInst::ICMP_UGT : CmpInst::ICMP_SLE;
    LbCond->setPredicate(Pred);
    LbCond->setOperand(0, TcDec);
    LbCond->setOperand(1, cast<Value>(ConstantInt::get(Ty, 0)));
  }

  // Step 4: All the references to the original population counter outside
  //  the loop are replaced with the NewCount -- the value returned from
  //  __builtin_ctpop().
  CntInst->replaceUsesOutsideBlock(NewCount, Body);

  // step 5: Forget the "non-computable" trip-count SCEV associated with the
  //   loop. The loop would otherwise not be deleted even if it becomes empty.
  SE->forgetLoop(CurLoop);
}

CallInst *NclPopcountRecognize::createPopcntIntrinsic(IRBuilderTy &IRBuilder,
                                                      Value *Val, DebugLoc DL) {
  Value *Ops[] = { Val };
  Type *Tys[] = { Val->getType() };

  Module *M = (*(CurLoop->block_begin()))->getParent()->getParent();
  Value *Func = Intrinsic::getDeclaration(M, Intrinsic::ctpop, Tys);
  CallInst *CI = IRBuilder.CreateCall(Func, Ops);
  CI->setDebugLoc(DL);

  return CI;
}

/// recognize - detect population count idiom in a non-countable loop. If
///   detected, transform the relevant code to popcount intrinsic function
///   call, and return true; otherwise, return false.
bool NclPopcountRecognize::recognize() {

  if (!LIR.getTargetTransformInfo())
    return false;

  LIR.getScalarEvolution();

  if (!preliminaryScreen())
    return false;

  Instruction *CntInst;
  PHINode *CntPhi;
  Value *Val;
  if (!detectIdiom(CntInst, CntPhi, Val))
    return false;

  transform(CntInst, CntPhi, Val);
  return true;
}

//===----------------------------------------------------------------------===//
//
//          Implementation of LoopIdiomRecognize
//
//===----------------------------------------------------------------------===//

bool LoopIdiomRecognize::runOnCountableLoop() {
  const SCEV *BECount = SE->getBackedgeTakenCount(CurLoop);
  assert(!isa<SCEVCouldNotCompute>(BECount) &&
    "runOnCountableLoop() called on a loop without a predictable"
    "backedge-taken count");

  // If this loop executes exactly one time, then it should be peeled, not
  // optimized by this pass.
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    if (BECst->getValue()->getValue() == 0)
      return false;

  // set DT
  (void)getDominatorTree();

  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  // set TLI
  (void)getTargetLibraryInfo();

  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);

  DEBUG(dbgs() << "loop-idiom Scanning: F["
               << CurLoop->getHeader()->getParent()->getName()
               << "] Loop %" << CurLoop->getHeader()->getName() << "\n");

  bool MadeChange = false;
  // Scan all the blocks in the loop that are not in subloops.
  for (auto *BB : CurLoop->getBlocks()) {
    // Ignore blocks in subloops.
    if (LI.getLoopFor(BB) != CurLoop)
      continue;

    MadeChange |= runOnLoopBlock(BB, BECount, ExitBlocks);
  }
  return MadeChange;
}

bool LoopIdiomRecognize::runOnNoncountableLoop() {
  NclPopcountRecognize Popcount(*this);
  if (Popcount.recognize())
    return true;

  return false;
}

bool LoopIdiomRecognize::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  CurLoop = L;

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!L->getLoopPreheader())
    return false;

  // Disable loop idiom recognition if the function's name is a common idiom.
  StringRef Name = L->getHeader()->getParent()->getName();
  if (Name == "memset" || Name == "memcpy")
    return false;

  SE = &getAnalysis<ScalarEvolution>();
  if (SE->hasLoopInvariantBackedgeTakenCount(L))
    return runOnCountableLoop();
  return runOnNoncountableLoop();
}

/// runOnLoopBlock - Process the specified block, which lives in a counted loop
/// with the specified backedge count.  This block is known to be in the current
/// loop and not in any subloops.
bool LoopIdiomRecognize::runOnLoopBlock(BasicBlock *BB, const SCEV *BECount,
                                     SmallVectorImpl<BasicBlock*> &ExitBlocks) {
  // We can only promote stores in this block if they are unconditionally
  // executed in the loop.  For a block to be unconditionally executed, it has
  // to dominate all the exit blocks of the loop.  Verify this now.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!DT->dominates(BB, ExitBlocks[i]))
      return false;

  bool MadeChange = false;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    // Look for store instructions, which may be optimized to memset/memcpy.
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst))  {
      WeakVH InstPtr(I);
      if (!processLoopStore(SI, BECount)) continue;
      MadeChange = true;

      // If processing the store invalidated our iterator, start over from the
      // top of the block.
      if (!InstPtr)
        I = BB->begin();
      continue;
    }

    // Look for memset instructions, which may be optimized to a larger memset.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(Inst))  {
      WeakVH InstPtr(I);
      if (!processLoopMemSet(MSI, BECount)) continue;
      MadeChange = true;

      // If processing the memset invalidated our iterator, start over from the
      // top of the block.
      if (!InstPtr)
        I = BB->begin();
      continue;
    }
  }

  return MadeChange;
}


/// processLoopStore - See if this store can be promoted to a memset or memcpy.
bool LoopIdiomRecognize::processLoopStore(StoreInst *SI, const SCEV *BECount) {
  if (!SI->isSimple()) return false;

  Value *StoredVal = SI->getValueOperand();
  Value *StorePtr = SI->getPointerOperand();

  // Reject stores that are so large that they overflow an unsigned.
  auto &DL = CurLoop->getHeader()->getModule()->getDataLayout();
  uint64_t SizeInBits = DL.getTypeSizeInBits(StoredVal->getType());
  if ((SizeInBits & 7) || (SizeInBits >> 32) != 0)
    return false;

  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *StoreEv =
    dyn_cast<SCEVAddRecExpr>(SE->getSCEV(StorePtr));
  if (!StoreEv || StoreEv->getLoop() != CurLoop || !StoreEv->isAffine())
    return false;

  // Check to see if the stride matches the size of the store.  If so, then we
  // know that every byte is touched in the loop.
  unsigned StoreSize = (unsigned)SizeInBits >> 3;
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(StoreEv->getOperand(1));

  if (!Stride || StoreSize != Stride->getValue()->getValue()) {
    // TODO: Could also handle negative stride here someday, that will require
    // the validity check in mayLoopAccessLocation to be updated though.
    // Enable this to print exact negative strides.
#if 0 // HLSL Change - suppress '0 &&' warning
    if (0 && Stride && StoreSize == -Stride->getValue()->getValue()) {
      dbgs() << "NEGATIVE STRIDE: " << *SI << "\n";
      dbgs() << "BB: " << *SI->getParent();
    }
#endif // HLSL Change - suppress '0 &&' warning

    return false;
  }

  // See if we can optimize just this store in isolation.
  if (processLoopStridedStore(StorePtr, StoreSize, SI->getAlignment(),
                              StoredVal, SI, StoreEv, BECount))
    return true;

  // If the stored value is a strided load in the same loop with the same stride
  // this this may be transformable into a memcpy.  This kicks in for stuff like
  //   for (i) A[i] = B[i];
  if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
    const SCEVAddRecExpr *LoadEv =
      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(LI->getOperand(0)));
    if (LoadEv && LoadEv->getLoop() == CurLoop && LoadEv->isAffine() &&
        StoreEv->getOperand(1) == LoadEv->getOperand(1) && LI->isSimple())
      if (processLoopStoreOfLoopLoad(SI, StoreSize, StoreEv, LoadEv, BECount))
        return true;
  }
  //errs() << "UNHANDLED strided store: " << *StoreEv << " - " << *SI << "\n";

  return false;
}

/// processLoopMemSet - See if this memset can be promoted to a large memset.
bool LoopIdiomRecognize::
processLoopMemSet(MemSetInst *MSI, const SCEV *BECount) {
  // We can only handle non-volatile memsets with a constant size.
  if (MSI->isVolatile() || !isa<ConstantInt>(MSI->getLength())) return false;

  // If we're not allowed to hack on memset, we fail.
  if (!TLI->has(LibFunc::memset))
    return false;

  Value *Pointer = MSI->getDest();

  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *Ev = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Pointer));
  if (!Ev || Ev->getLoop() != CurLoop || !Ev->isAffine())
    return false;

  // Reject memsets that are so large that they overflow an unsigned.
  uint64_t SizeInBytes = cast<ConstantInt>(MSI->getLength())->getZExtValue();
  if ((SizeInBytes >> 32) != 0)
    return false;

  // Check to see if the stride matches the size of the memset.  If so, then we
  // know that every byte is touched in the loop.
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(Ev->getOperand(1));

  // TODO: Could also handle negative stride here someday, that will require the
  // validity check in mayLoopAccessLocation to be updated though.
  if (!Stride || MSI->getLength() != Stride->getValue())
    return false;

  return processLoopStridedStore(Pointer, (unsigned)SizeInBytes,
                                 MSI->getAlignment(), MSI->getValue(),
                                 MSI, Ev, BECount);
}


/// mayLoopAccessLocation - Return true if the specified loop might access the
/// specified pointer location, which is a loop-strided access.  The 'Access'
/// argument specifies what the verboten forms of access are (read or write).
static bool mayLoopAccessLocation(Value *Ptr,AliasAnalysis::ModRefResult Access,
                                  Loop *L, const SCEV *BECount,
                                  unsigned StoreSize, AliasAnalysis &AA,
                                  Instruction *IgnoredStore) {
  // Get the location that may be stored across the loop.  Since the access is
  // strided positively through memory, we say that the modified location starts
  // at the pointer and has infinite size.
  uint64_t AccessSize = MemoryLocation::UnknownSize;

  // If the loop iterates a fixed number of times, we can refine the access size
  // to be exactly the size of the memset, which is (BECount+1)*StoreSize
  if (const SCEVConstant *BECst = dyn_cast<SCEVConstant>(BECount))
    AccessSize = (BECst->getValue()->getZExtValue()+1)*StoreSize;

  // TODO: For this to be really effective, we have to dive into the pointer
  // operand in the store.  Store to &A[i] of 100 will always return may alias
  // with store of &A[100], we need to StoreLoc to be "A" with size of 100,
  // which will then no-alias a store to &A[100].
  MemoryLocation StoreLoc(Ptr, AccessSize);

  for (Loop::block_iterator BI = L->block_begin(), E = L->block_end(); BI != E;
       ++BI)
    for (BasicBlock::iterator I = (*BI)->begin(), E = (*BI)->end(); I != E; ++I)
      if (&*I != IgnoredStore &&
          (AA.getModRefInfo(I, StoreLoc) & Access))
        return true;

  return false;
}

/// getMemSetPatternValue - If a strided store of the specified value is safe to
/// turn into a memset_pattern16, return a ConstantArray of 16 bytes that should
/// be passed in.  Otherwise, return null.
///
/// Note that we don't ever attempt to use memset_pattern8 or 4, because these
/// just replicate their input array and then pass on to memset_pattern16.
static Constant *getMemSetPatternValue(Value *V, const DataLayout &DL) {
  // If the value isn't a constant, we can't promote it to being in a constant
  // array.  We could theoretically do a store to an alloca or something, but
  // that doesn't seem worthwhile.
  Constant *C = dyn_cast<Constant>(V);
  if (!C) return nullptr;

  // Only handle simple values that are a power of two bytes in size.
  uint64_t Size = DL.getTypeSizeInBits(V->getType());
  if (Size == 0 || (Size & 7) || (Size & (Size-1)))
    return nullptr;

  // Don't care enough about darwin/ppc to implement this.
  if (DL.isBigEndian())
    return nullptr;

  // Convert to size in bytes.
  Size /= 8;

  // TODO: If CI is larger than 16-bytes, we can try slicing it in half to see
  // if the top and bottom are the same (e.g. for vectors and large integers).
  if (Size > 16) return nullptr;

  // If the constant is exactly 16 bytes, just use it.
  if (Size == 16) return C;

  // Otherwise, we'll use an array of the constants.
  unsigned ArraySize = 16/Size;
  ArrayType *AT = ArrayType::get(V->getType(), ArraySize);
  return ConstantArray::get(AT, std::vector<Constant*>(ArraySize, C));
}


/// processLoopStridedStore - We see a strided store of some value.  If we can
/// transform this into a memset or memset_pattern in the loop preheader, do so.
bool LoopIdiomRecognize::
processLoopStridedStore(Value *DestPtr, unsigned StoreSize,
                        unsigned StoreAlignment, Value *StoredVal,
                        Instruction *TheStore, const SCEVAddRecExpr *Ev,
                        const SCEV *BECount) {

  // If the stored value is a byte-wise value (like i32 -1), then it may be
  // turned into a memset of i8 -1, assuming that all the consecutive bytes
  // are stored.  A store of i32 0x01020304 can never be turned into a memset,
  // but it can be turned into memset_pattern if the target supports it.
  Value *SplatValue = isBytewiseValue(StoredVal);
  Constant *PatternValue = nullptr;
  auto &DL = CurLoop->getHeader()->getModule()->getDataLayout();
  unsigned DestAS = DestPtr->getType()->getPointerAddressSpace();

  // If we're allowed to form a memset, and the stored value would be acceptable
  // for memset, use it.
  if (SplatValue && TLI->has(LibFunc::memset) &&
      // Verify that the stored value is loop invariant.  If not, we can't
      // promote the memset.
      CurLoop->isLoopInvariant(SplatValue)) {
    // Keep and use SplatValue.
    PatternValue = nullptr;
  } else if (DestAS == 0 && TLI->has(LibFunc::memset_pattern16) &&
             (PatternValue = getMemSetPatternValue(StoredVal, DL))) {
    // Don't create memset_pattern16s with address spaces.
    // It looks like we can use PatternValue!
    SplatValue = nullptr;
  } else {
    // Otherwise, this isn't an idiom we can transform.  For example, we can't
    // do anything with a 3-byte store.
    return false;
  }

  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  This allows us to insert code for it in the preheader.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  IRBuilder<> Builder(Preheader->getTerminator());
  SCEVExpander Expander(*SE, DL, "loop-idiom");

  Type *DestInt8PtrTy = Builder.getInt8PtrTy(DestAS);

  // Okay, we have a strided store "p[i]" of a splattable value.  We can turn
  // this into a memset in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write to the aliased location.  Check for any overlap by generating the
  // base pointer and checking the region.
  Value *BasePtr =
    Expander.expandCodeFor(Ev->getStart(), DestInt8PtrTy,
                           Preheader->getTerminator());

  if (mayLoopAccessLocation(BasePtr, AliasAnalysis::ModRef,
                            CurLoop, BECount,
                            StoreSize, getAnalysis<AliasAnalysis>(), TheStore)) {
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    RecursivelyDeleteTriviallyDeadInstructions(BasePtr, TLI);
    return false;
  }

  // Okay, everything looks good, insert the memset.

  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  Type *IntPtr = Builder.getIntPtrTy(DL, DestAS);
  BECount = SE->getTruncateOrZeroExtend(BECount, IntPtr);

  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtr, 1),
                                         SCEV::FlagNUW);
  if (StoreSize != 1) {
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtr, StoreSize),
                               SCEV::FlagNUW);
  }

  Value *NumBytes =
    Expander.expandCodeFor(NumBytesS, IntPtr, Preheader->getTerminator());

  CallInst *NewCall;
  if (SplatValue) {
    NewCall = Builder.CreateMemSet(BasePtr,
                                   SplatValue,
                                   NumBytes,
                                   StoreAlignment);
  } else {
    // Everything is emitted in default address space
    Type *Int8PtrTy = DestInt8PtrTy;

    Module *M = TheStore->getParent()->getParent()->getParent();
    Value *MSP = M->getOrInsertFunction("memset_pattern16",
                                        Builder.getVoidTy(),
                                        Int8PtrTy,
                                        Int8PtrTy,
                                        IntPtr,
                                        (void*)nullptr);

    // Otherwise we should form a memset_pattern16.  PatternValue is known to be
    // an constant array of 16-bytes.  Plop the value into a mergable global.
    GlobalVariable *GV = new GlobalVariable(*M, PatternValue->getType(), true,
                                            GlobalValue::PrivateLinkage,
                                            PatternValue, ".memset_pattern");
    GV->setUnnamedAddr(true); // Ok to merge these.
    GV->setAlignment(16);
    Value *PatternPtr = ConstantExpr::getBitCast(GV, Int8PtrTy);
    NewCall = Builder.CreateCall(MSP, {BasePtr, PatternPtr, NumBytes});
  }

  DEBUG(dbgs() << "  Formed memset: " << *NewCall << "\n"
               << "    from store to: " << *Ev << " at: " << *TheStore << "\n");
  NewCall->setDebugLoc(TheStore->getDebugLoc());

  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  deleteDeadInstruction(TheStore, TLI);
  ++NumMemSet;
  return true;
}

/// processLoopStoreOfLoopLoad - We see a strided store whose value is a
/// same-strided load.
bool LoopIdiomRecognize::
processLoopStoreOfLoopLoad(StoreInst *SI, unsigned StoreSize,
                           const SCEVAddRecExpr *StoreEv,
                           const SCEVAddRecExpr *LoadEv,
                           const SCEV *BECount) {
  // If we're not allowed to form memcpy, we fail.
  if (!TLI->has(LibFunc::memcpy))
    return false;

  LoadInst *LI = cast<LoadInst>(SI->getValueOperand());

  // The trip count of the loop and the base pointer of the addrec SCEV is
  // guaranteed to be loop invariant, which means that it should dominate the
  // header.  This allows us to insert code for it in the preheader.
  BasicBlock *Preheader = CurLoop->getLoopPreheader();
  IRBuilder<> Builder(Preheader->getTerminator());
  const DataLayout &DL = Preheader->getModule()->getDataLayout();
  SCEVExpander Expander(*SE, DL, "loop-idiom");

  // Okay, we have a strided store "p[i]" of a loaded value.  We can turn
  // this into a memcpy in the loop preheader now if we want.  However, this
  // would be unsafe to do if there is anything else in the loop that may read
  // or write the memory region we're storing to.  This includes the load that
  // feeds the stores.  Check for an alias by generating the base address and
  // checking everything.
  Value *StoreBasePtr =
    Expander.expandCodeFor(StoreEv->getStart(),
                           Builder.getInt8PtrTy(SI->getPointerAddressSpace()),
                           Preheader->getTerminator());

  if (mayLoopAccessLocation(StoreBasePtr, AliasAnalysis::ModRef,
                            CurLoop, BECount, StoreSize,
                            getAnalysis<AliasAnalysis>(), SI)) {
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    RecursivelyDeleteTriviallyDeadInstructions(StoreBasePtr, TLI);
    return false;
  }

  // For a memcpy, we have to make sure that the input array is not being
  // mutated by the loop.
  Value *LoadBasePtr =
    Expander.expandCodeFor(LoadEv->getStart(),
                           Builder.getInt8PtrTy(LI->getPointerAddressSpace()),
                           Preheader->getTerminator());

  if (mayLoopAccessLocation(LoadBasePtr, AliasAnalysis::Mod, CurLoop, BECount,
                            StoreSize, getAnalysis<AliasAnalysis>(), SI)) {
    Expander.clear();
    // If we generated new code for the base pointer, clean up.
    RecursivelyDeleteTriviallyDeadInstructions(LoadBasePtr, TLI);
    RecursivelyDeleteTriviallyDeadInstructions(StoreBasePtr, TLI);
    return false;
  }

  // Okay, everything is safe, we can transform this!


  // The # stored bytes is (BECount+1)*Size.  Expand the trip count out to
  // pointer size if it isn't already.
  Type *IntPtrTy = Builder.getIntPtrTy(DL, SI->getPointerAddressSpace());
  BECount = SE->getTruncateOrZeroExtend(BECount, IntPtrTy);

  const SCEV *NumBytesS = SE->getAddExpr(BECount, SE->getConstant(IntPtrTy, 1),
                                         SCEV::FlagNUW);
  if (StoreSize != 1)
    NumBytesS = SE->getMulExpr(NumBytesS, SE->getConstant(IntPtrTy, StoreSize),
                               SCEV::FlagNUW);

  Value *NumBytes =
    Expander.expandCodeFor(NumBytesS, IntPtrTy, Preheader->getTerminator());

  CallInst *NewCall =
    Builder.CreateMemCpy(StoreBasePtr, LoadBasePtr, NumBytes,
                         std::min(SI->getAlignment(), LI->getAlignment()));
  NewCall->setDebugLoc(SI->getDebugLoc());

  DEBUG(dbgs() << "  Formed memcpy: " << *NewCall << "\n"
               << "    from load ptr=" << *LoadEv << " at: " << *LI << "\n"
               << "    from store ptr=" << *StoreEv << " at: " << *SI << "\n");


  // Okay, the memset has been formed.  Zap the original store and anything that
  // feeds into it.
  deleteDeadInstruction(SI, TLI);
  ++NumMemCpy;
  return true;
}
