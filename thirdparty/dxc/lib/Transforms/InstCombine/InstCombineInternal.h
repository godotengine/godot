//===- InstCombineInternal.h - InstCombine pass internals -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides internal interfaces used to implement the InstCombine.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_INSTCOMBINE_INSTCOMBINEINTERNAL_H
#define LLVM_LIB_TRANSFORMS_INSTCOMBINE_INSTCOMBINEINTERNAL_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"

#define DEBUG_TYPE "instcombine"

namespace llvm {
class CallSite;
class DataLayout;
class DominatorTree;
class TargetLibraryInfo;
class DbgDeclareInst;
class MemIntrinsic;
class MemSetInst;

/// \brief Assign a complexity or rank value to LLVM Values.
///
/// This routine maps IR values to various complexity ranks:
///   0 -> undef
///   1 -> Constants
///   2 -> Other non-instructions
///   3 -> Arguments
///   3 -> Unary operations
///   4 -> Other instructions
static inline unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) || BinaryOperator::isFNeg(V) ||
        BinaryOperator::isNot(V))
      return 3;
    return 4;
  }
  if (isa<Argument>(V))
    return 3;
  return isa<Constant>(V) ? (isa<UndefValue>(V) ? 0 : 1) : 2;
}

/// \brief Add one to a Constant
static inline Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// \brief Subtract one from a Constant
static inline Constant *SubOne(Constant *C) {
  return ConstantExpr::getSub(C, ConstantInt::get(C->getType(), 1));
}

/// \brief Return true if the specified value is free to invert (apply ~ to).
/// This happens in cases where the ~ can be eliminated.  If WillInvertAllUses
/// is true, work under the assumption that the caller intends to remove all
/// uses of V and only keep uses of ~V.
///
static inline bool IsFreeToInvert(Value *V, bool WillInvertAllUses) {
  // ~(~(X)) -> X.
  if (BinaryOperator::isNot(V))
    return true;

  // Constants can be considered to be not'ed values.
  if (isa<ConstantInt>(V))
    return true;

  // Compares can be inverted if all of their uses are being modified to use the
  // ~V.
  if (isa<CmpInst>(V))
    return WillInvertAllUses;

  // If `V` is of the form `A + Constant` then `-1 - V` can be folded into `(-1
  // - Constant) - A` if we are willing to invert all of the uses.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(V))
    if (BO->getOpcode() == Instruction::Add ||
        BO->getOpcode() == Instruction::Sub)
      if (isa<Constant>(BO->getOperand(0)) || isa<Constant>(BO->getOperand(1)))
        return WillInvertAllUses;

  return false;
}


/// \brief Specific patterns of overflow check idioms that we match.
enum OverflowCheckFlavor {
  OCF_UNSIGNED_ADD,
  OCF_SIGNED_ADD,
  OCF_UNSIGNED_SUB,
  OCF_SIGNED_SUB,
  OCF_UNSIGNED_MUL,
  OCF_SIGNED_MUL,

  OCF_INVALID
};

/// \brief Returns the OverflowCheckFlavor corresponding to a overflow_with_op
/// intrinsic.
static inline OverflowCheckFlavor
IntrinsicIDToOverflowCheckFlavor(unsigned ID) {
  switch (ID) {
  default:
    return OCF_INVALID;
  case Intrinsic::uadd_with_overflow:
    return OCF_UNSIGNED_ADD;
  case Intrinsic::sadd_with_overflow:
    return OCF_SIGNED_ADD;
  case Intrinsic::usub_with_overflow:
    return OCF_UNSIGNED_SUB;
  case Intrinsic::ssub_with_overflow:
    return OCF_SIGNED_SUB;
  case Intrinsic::umul_with_overflow:
    return OCF_UNSIGNED_MUL;
  case Intrinsic::smul_with_overflow:
    return OCF_SIGNED_MUL;
  }
}

/// \brief An IRBuilder inserter that adds new instructions to the instcombine
/// worklist.
class LLVM_LIBRARY_VISIBILITY InstCombineIRInserter
    : public IRBuilderDefaultInserter<true> {
  InstCombineWorklist &Worklist;
  AssumptionCache *AC;

public:
  InstCombineIRInserter(InstCombineWorklist &WL, AssumptionCache *AC)
      : Worklist(WL), AC(AC) {}

  void InsertHelper(Instruction *I, const Twine &Name, BasicBlock *BB,
                    BasicBlock::iterator InsertPt) const {
    IRBuilderDefaultInserter<true>::InsertHelper(I, Name, BB, InsertPt);
    Worklist.Add(I);

    using namespace llvm::PatternMatch;
    if (match(I, m_Intrinsic<Intrinsic::assume>()))
      AC->registerAssumption(cast<CallInst>(I));
  }
};

/// \brief The core instruction combiner logic.
///
/// This class provides both the logic to recursively visit instructions and
/// combine them, as well as the pass infrastructure for running this as part
/// of the LLVM pass pipeline.
class LLVM_LIBRARY_VISIBILITY InstCombiner
    : public InstVisitor<InstCombiner, Instruction *> {
  // FIXME: These members shouldn't be public.
public:
  /// \brief A worklist of the instructions that need to be simplified.
  InstCombineWorklist &Worklist;

  /// \brief An IRBuilder that automatically inserts new instructions into the
  /// worklist.
  typedef IRBuilder<true, TargetFolder, InstCombineIRInserter> BuilderTy;
  BuilderTy *Builder;

private:
  // Mode in which we are running the combiner.
  const bool MinimizeSize;

  AliasAnalysis *AA;

  // Required analyses.
  // FIXME: These can never be null and should be references.
  AssumptionCache *AC;
  TargetLibraryInfo *TLI;
  DominatorTree *DT;
  const DataLayout &DL;

  // Optional analyses. When non-null, these can both be used to do better
  // combining and will be updated to reflect any changes.
  LoopInfo *LI;

  bool MadeIRChange;

public:
  InstCombiner(InstCombineWorklist &Worklist, BuilderTy *Builder,
               bool MinimizeSize, AliasAnalysis *AA,
               AssumptionCache *AC, TargetLibraryInfo *TLI,
               DominatorTree *DT, const DataLayout &DL, LoopInfo *LI)
      : Worklist(Worklist), Builder(Builder), MinimizeSize(MinimizeSize),
        AA(AA), AC(AC), TLI(TLI), DT(DT), DL(DL), LI(LI), MadeIRChange(false) {}

  /// \brief Run the combiner over the entire worklist until it is empty.
  ///
  /// \returns true if the IR is changed.
  bool run();

  AssumptionCache *getAssumptionCache() const { return AC; }

  const DataLayout &getDataLayout() const { return DL; }

  DominatorTree *getDominatorTree() const { return DT; }

  LoopInfo *getLoopInfo() const { return LI; }

  TargetLibraryInfo *getTargetLibraryInfo() const { return TLI; }

  // Visitation implementation - Implement instruction combining for different
  // instruction types.  The semantics are as follows:
  // Return Value:
  //    null        - No change was made
  //     I          - Change was made, I is still valid, I may be dead though
  //   otherwise    - Change was made, replace I with returned instruction
  //
  Instruction *visitAdd(BinaryOperator &I);
  Instruction *visitFAdd(BinaryOperator &I);
  Value *OptimizePointerDifference(Value *LHS, Value *RHS, Type *Ty);
  Instruction *visitSub(BinaryOperator &I);
  Instruction *visitFSub(BinaryOperator &I);
  Instruction *visitMul(BinaryOperator &I);
  Value *foldFMulConst(Instruction *FMulOrDiv, Constant *C,
                       Instruction *InsertBefore);
  Instruction *visitFMul(BinaryOperator &I);
  Instruction *visitURem(BinaryOperator &I);
  Instruction *visitSRem(BinaryOperator &I);
  Instruction *visitFRem(BinaryOperator &I);
  bool SimplifyDivRemOfSelect(BinaryOperator &I);
  Instruction *commonRemTransforms(BinaryOperator &I);
  Instruction *commonIRemTransforms(BinaryOperator &I);
  Instruction *commonDivTransforms(BinaryOperator &I);
  Instruction *commonIDivTransforms(BinaryOperator &I);
  Instruction *visitUDiv(BinaryOperator &I);
  Instruction *visitSDiv(BinaryOperator &I);
  Instruction *visitFDiv(BinaryOperator &I);
  Value *simplifyRangeCheck(ICmpInst *Cmp0, ICmpInst *Cmp1, bool Inverted);
  Value *FoldAndOfICmps(ICmpInst *LHS, ICmpInst *RHS);
  Value *FoldAndOfFCmps(FCmpInst *LHS, FCmpInst *RHS);
  Instruction *visitAnd(BinaryOperator &I);
  Value *FoldOrOfICmps(ICmpInst *LHS, ICmpInst *RHS, Instruction *CxtI);
  Value *FoldOrOfFCmps(FCmpInst *LHS, FCmpInst *RHS);
  Instruction *FoldOrWithConstants(BinaryOperator &I, Value *Op, Value *A,
                                   Value *B, Value *C);
  Instruction *FoldXorWithConstants(BinaryOperator &I, Value *Op, Value *A,
                                    Value *B, Value *C);
  Instruction *visitOr(BinaryOperator &I);
  Instruction *visitXor(BinaryOperator &I);
  Instruction *visitShl(BinaryOperator &I);
  Instruction *visitAShr(BinaryOperator &I);
  Instruction *visitLShr(BinaryOperator &I);
  Instruction *commonShiftTransforms(BinaryOperator &I);
  Instruction *FoldFCmp_IntToFP_Cst(FCmpInst &I, Instruction *LHSI,
                                    Constant *RHSC);
  Instruction *FoldCmpLoadFromIndexedGlobal(GetElementPtrInst *GEP,
                                            GlobalVariable *GV, CmpInst &ICI,
                                            ConstantInt *AndCst = nullptr);
  Instruction *visitFCmpInst(FCmpInst &I);
  Instruction *visitICmpInst(ICmpInst &I);
  Instruction *visitICmpInstWithCastAndCast(ICmpInst &ICI);
  Instruction *visitICmpInstWithInstAndIntCst(ICmpInst &ICI, Instruction *LHS,
                                              ConstantInt *RHS);
  Instruction *FoldICmpDivCst(ICmpInst &ICI, BinaryOperator *DivI,
                              ConstantInt *DivRHS);
  Instruction *FoldICmpShrCst(ICmpInst &ICI, BinaryOperator *DivI,
                              ConstantInt *DivRHS);
  Instruction *FoldICmpCstShrCst(ICmpInst &I, Value *Op, Value *A,
                                 ConstantInt *CI1, ConstantInt *CI2);
  Instruction *FoldICmpCstShlCst(ICmpInst &I, Value *Op, Value *A,
                                 ConstantInt *CI1, ConstantInt *CI2);
  Instruction *FoldICmpAddOpCst(Instruction &ICI, Value *X, ConstantInt *CI,
                                ICmpInst::Predicate Pred);
  Instruction *FoldGEPICmp(GEPOperator *GEPLHS, Value *RHS,
                           ICmpInst::Predicate Cond, Instruction &I);
  Instruction *FoldShiftByConstant(Value *Op0, Constant *Op1,
                                   BinaryOperator &I);
  Instruction *commonCastTransforms(CastInst &CI);
  Instruction *commonPointerCastTransforms(CastInst &CI);
  Instruction *visitTrunc(TruncInst &CI);
  Instruction *visitZExt(ZExtInst &CI);
  Instruction *visitSExt(SExtInst &CI);
  Instruction *visitFPTrunc(FPTruncInst &CI);
  Instruction *visitFPExt(CastInst &CI);
  Instruction *visitFPToUI(FPToUIInst &FI);
  Instruction *visitFPToSI(FPToSIInst &FI);
  Instruction *visitUIToFP(CastInst &CI);
  Instruction *visitSIToFP(CastInst &CI);
  Instruction *visitPtrToInt(PtrToIntInst &CI);
  Instruction *visitIntToPtr(IntToPtrInst &CI);
  Instruction *visitBitCast(BitCastInst &CI);
  Instruction *visitAddrSpaceCast(AddrSpaceCastInst &CI);
  Instruction *FoldSelectOpOp(SelectInst &SI, Instruction *TI, Instruction *FI);
  Instruction *FoldSelectIntoOp(SelectInst &SI, Value *, Value *);
  Instruction *FoldSPFofSPF(Instruction *Inner, SelectPatternFlavor SPF1,
                            Value *A, Value *B, Instruction &Outer,
                            SelectPatternFlavor SPF2, Value *C);
  Instruction *FoldItoFPtoI(Instruction &FI);
  Instruction *visitSelectInst(SelectInst &SI);
  Instruction *visitSelectInstWithICmp(SelectInst &SI, ICmpInst *ICI);
  Instruction *visitCallInst(CallInst &CI);
  Instruction *visitInvokeInst(InvokeInst &II);

  Instruction *SliceUpIllegalIntegerPHI(PHINode &PN);
  Instruction *visitPHINode(PHINode &PN);
  Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
  Instruction *visitAllocaInst(AllocaInst &AI);
  Instruction *visitAllocSite(Instruction &FI);
  Instruction *visitFree(CallInst &FI);
  Instruction *visitLoadInst(LoadInst &LI);
  Instruction *visitStoreInst(StoreInst &SI);
  Instruction *visitBranchInst(BranchInst &BI);
  Instruction *visitSwitchInst(SwitchInst &SI);
  Instruction *visitReturnInst(ReturnInst &RI);
  Instruction *visitInsertValueInst(InsertValueInst &IV);
  Instruction *visitInsertElementInst(InsertElementInst &IE);
  Instruction *visitExtractElementInst(ExtractElementInst &EI);
  Instruction *visitShuffleVectorInst(ShuffleVectorInst &SVI);
  Instruction *visitExtractValueInst(ExtractValueInst &EV);
  Instruction *visitLandingPadInst(LandingPadInst &LI);

  // visitInstruction - Specify what to return for unhandled instructions...
  Instruction *visitInstruction(Instruction &I) { return nullptr; }

  // True when DB dominates all uses of DI execpt UI.
  // UI must be in the same block as DI.
  // The routine checks that the DI parent and DB are different.
  bool dominatesAllUses(const Instruction *DI, const Instruction *UI,
                        const BasicBlock *DB) const;

  // Replace select with select operand SIOpd in SI-ICmp sequence when possible
  bool replacedSelectWithOperand(SelectInst *SI, const ICmpInst *Icmp,
                                 const unsigned SIOpd);

private:
  bool ShouldChangeType(Type *From, Type *To) const;
  Value *dyn_castNegVal(Value *V) const;
  Value *dyn_castFNegVal(Value *V, bool NoSignedZero = false) const;
  Type *FindElementAtOffset(PointerType *PtrTy, int64_t Offset,
                            SmallVectorImpl<Value *> &NewIndices);
  Instruction *FoldOpIntoSelect(Instruction &Op, SelectInst *SI);

  /// \brief Classify whether a cast is worth optimizing.
  ///
  /// Returns true if the cast from "V to Ty" actually results in any code
  /// being generated and is interesting to optimize out. If the cast can be
  /// eliminated by some other simple transformation, we prefer to do the
  /// simplification first.
  bool ShouldOptimizeCast(Instruction::CastOps opcode, const Value *V,
                          Type *Ty);

  /// \brief Try to optimize a sequence of instructions checking if an operation
  /// on LHS and RHS overflows.
  ///
  /// If a simplification is possible, stores the simplified result of the
  /// operation in OperationResult and result of the overflow check in
  /// OverflowResult, and return true.  If no simplification is possible,
  /// returns false.
  bool OptimizeOverflowCheck(OverflowCheckFlavor OCF, Value *LHS, Value *RHS,
                             Instruction &CtxI, Value *&OperationResult,
                             Constant *&OverflowResult);

  Instruction *visitCallSite(CallSite CS);
  Instruction *tryOptimizeCall(CallInst *CI);
  bool transformConstExprCastCall(CallSite CS);
  Instruction *transformCallThroughTrampoline(CallSite CS,
                                              IntrinsicInst *Tramp);
  Instruction *transformZExtICmp(ICmpInst *ICI, Instruction &CI,
                                 bool DoXform = true);
  Instruction *transformSExtICmp(ICmpInst *ICI, Instruction &CI);
  bool WillNotOverflowSignedAdd(Value *LHS, Value *RHS, Instruction &CxtI);
  bool WillNotOverflowSignedSub(Value *LHS, Value *RHS, Instruction &CxtI);
  bool WillNotOverflowUnsignedSub(Value *LHS, Value *RHS, Instruction &CxtI);
  bool WillNotOverflowSignedMul(Value *LHS, Value *RHS, Instruction &CxtI);
  Value *EmitGEPOffset(User *GEP);
  Instruction *scalarizePHI(ExtractElementInst &EI, PHINode *PN);
  Value *EvaluateInDifferentElementOrder(Value *V, ArrayRef<int> Mask);

public:
  /// \brief Inserts an instruction \p New before instruction \p Old
  ///
  /// Also adds the new instruction to the worklist and returns \p New so that
  /// it is suitable for use as the return from the visitation patterns.
  Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
    assert(New && !New->getParent() &&
           "New instruction already inserted into a basic block!");
    BasicBlock *BB = Old.getParent();
    BB->getInstList().insert(&Old, New); // Insert inst
    Worklist.Add(New);
    return New;
  }

  /// \brief Same as InsertNewInstBefore, but also sets the debug loc.
  Instruction *InsertNewInstWith(Instruction *New, Instruction &Old) {
    New->setDebugLoc(Old.getDebugLoc());
    return InsertNewInstBefore(New, Old);
  }

  /// \brief A combiner-aware RAUW-like routine.
  ///
  /// This method is to be used when an instruction is found to be dead,
  /// replacable with another preexisting expression. Here we add all uses of
  /// I to the worklist, replace all uses of I with the new value, then return
  /// I, so that the inst combiner will know that I was modified.
  Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
    // If there are no uses to replace, then we return nullptr to indicate that
    // no changes were made to the program.
    if (I.use_empty()) return nullptr;

    Worklist.AddUsersToWorkList(I); // Add all modified instrs to worklist.

    // If we are replacing the instruction with itself, this must be in a
    // segment of unreachable code, so just clobber the instruction.
    if (&I == V)
      V = UndefValue::get(I.getType());

    DEBUG(dbgs() << "IC: Replacing " << I << "\n"
                 << "    with " << *V << '\n');

    I.replaceAllUsesWith(V);
    return &I;
  }

  /// Creates a result tuple for an overflow intrinsic \p II with a given
  /// \p Result and a constant \p Overflow value.
  Instruction *CreateOverflowTuple(IntrinsicInst *II, Value *Result,
                                   Constant *Overflow) {
    Constant *V[] = {UndefValue::get(Result->getType()), Overflow};
    StructType *ST = cast<StructType>(II->getType());
    Constant *Struct = ConstantStruct::get(ST, V);
    return InsertValueInst::Create(Struct, Result, 0);
  }

  /// \brief Combiner aware instruction erasure.
  ///
  /// When dealing with an instruction that has side effects or produces a void
  /// value, we can't rely on DCE to delete the instruction. Instead, visit
  /// methods should return the value returned by this function.
  Instruction *EraseInstFromFunction(Instruction &I) {
    DEBUG(dbgs() << "IC: ERASE " << I << '\n');

    assert(I.use_empty() && "Cannot erase instruction that is used!");
    // Make sure that we reprocess all operands now that we reduced their
    // use counts.
    if (I.getNumOperands() < 8) {
      for (User::op_iterator i = I.op_begin(), e = I.op_end(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(*i))
          Worklist.Add(Op);
    }
    Worklist.Remove(&I);
    I.eraseFromParent();
    MadeIRChange = true;
    return nullptr; // Don't do anything with FI
  }

  void computeKnownBits(Value *V, APInt &KnownZero, APInt &KnownOne,
                        unsigned Depth, Instruction *CxtI) const {
    return llvm::computeKnownBits(V, KnownZero, KnownOne, DL, Depth, AC, CxtI,
                                  DT);
  }

  bool MaskedValueIsZero(Value *V, const APInt &Mask, unsigned Depth = 0,
                         Instruction *CxtI = nullptr) const {
    return llvm::MaskedValueIsZero(V, Mask, DL, Depth, AC, CxtI, DT);
  }
  unsigned ComputeNumSignBits(Value *Op, unsigned Depth = 0,
                              Instruction *CxtI = nullptr) const {
    return llvm::ComputeNumSignBits(Op, DL, Depth, AC, CxtI, DT);
  }
  void ComputeSignBit(Value *V, bool &KnownZero, bool &KnownOne,
                      unsigned Depth = 0, Instruction *CxtI = nullptr) const {
    return llvm::ComputeSignBit(V, KnownZero, KnownOne, DL, Depth, AC, CxtI,
                                DT);
  }
  OverflowResult computeOverflowForUnsignedMul(Value *LHS, Value *RHS,
                                               const Instruction *CxtI) {
    return llvm::computeOverflowForUnsignedMul(LHS, RHS, DL, AC, CxtI, DT);
  }
  OverflowResult computeOverflowForUnsignedAdd(Value *LHS, Value *RHS,
                                               const Instruction *CxtI) {
    return llvm::computeOverflowForUnsignedAdd(LHS, RHS, DL, AC, CxtI, DT);
  }

private:
  /// \brief Performs a few simplifications for operators which are associative
  /// or commutative.
  bool SimplifyAssociativeOrCommutative(BinaryOperator &I);

  /// \brief Tries to simplify binary operations which some other binary
  /// operation distributes over.
  ///
  /// It does this by either by factorizing out common terms (eg "(A*B)+(A*C)"
  /// -> "A*(B+C)") or expanding out if this results in simplifications (eg: "A
  /// & (B | C) -> (A&B) | (A&C)" if this is a win).  Returns the simplified
  /// value, or null if it didn't simplify.
  Value *SimplifyUsingDistributiveLaws(BinaryOperator &I);

  /// \brief Attempts to replace V with a simpler value based on the demanded
  /// bits.
  Value *SimplifyDemandedUseBits(Value *V, APInt DemandedMask, APInt &KnownZero,
                                 APInt &KnownOne, unsigned Depth,
                                 Instruction *CxtI);
  bool SimplifyDemandedBits(Use &U, APInt DemandedMask, APInt &KnownZero,
                            APInt &KnownOne, unsigned Depth = 0);
  /// Helper routine of SimplifyDemandedUseBits. It tries to simplify demanded
  /// bit for "r1 = shr x, c1; r2 = shl r1, c2" instruction sequence.
  Value *SimplifyShrShlDemandedBits(Instruction *Lsr, Instruction *Sftl,
                                    APInt DemandedMask, APInt &KnownZero,
                                    APInt &KnownOne);

  /// \brief Tries to simplify operands to an integer instruction based on its
  /// demanded bits.
  bool SimplifyDemandedInstructionBits(Instruction &Inst);

  Value *SimplifyDemandedVectorElts(Value *V, APInt DemandedElts,
                                    APInt &UndefElts, unsigned Depth = 0);

  Value *SimplifyVectorOp(BinaryOperator &Inst);
  Value *SimplifyBSwap(BinaryOperator &Inst);

  // FoldOpIntoPhi - Given a binary operator, cast instruction, or select
  // which has a PHI node as operand #0, see if we can fold the instruction
  // into the PHI (which is only possible if all operands to the PHI are
  // constants).
  //
  Instruction *FoldOpIntoPhi(Instruction &I);

  /// \brief Try to rotate an operation below a PHI node, using PHI nodes for
  /// its operands.
  Instruction *FoldPHIArgOpIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgBinOpIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgGEPIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgLoadIntoPHI(PHINode &PN);

  Instruction *OptAndOp(Instruction *Op, ConstantInt *OpRHS,
                        ConstantInt *AndRHS, BinaryOperator &TheAnd);

  Value *FoldLogicalPlusAnd(Value *LHS, Value *RHS, ConstantInt *Mask,
                            bool isSub, Instruction &I);
  Value *InsertRangeTest(Value *V, Constant *Lo, Constant *Hi, bool isSigned,
                         bool Inside);
  Instruction *PromoteCastOfAllocation(BitCastInst &CI, AllocaInst &AI);
  Instruction *MatchBSwap(BinaryOperator &I);
  bool SimplifyStoreAtEndOfBlock(StoreInst &SI);
  Instruction *SimplifyMemTransfer(MemIntrinsic *MI);
  Instruction *SimplifyMemSet(MemSetInst *MI);

  Value *EvaluateInDifferentType(Value *V, Type *Ty, bool isSigned);

  /// \brief Returns a value X such that Val = X * Scale, or null if none.
  ///
  /// If the multiplication is known not to overflow then NoSignedWrap is set.
  Value *Descale(Value *Val, APInt Scale, bool &NoSignedWrap);
};

} // end namespace llvm.

#undef DEBUG_TYPE

#endif
