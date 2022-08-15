//===- InstCombineMulDivRem.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for mul, fmul, sdiv, udiv, fdiv,
// srem, urem, frem.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"


/// simplifyValueKnownNonZero - The specific integer value is used in a context
/// where it is known to be non-zero.  If this allows us to simplify the
/// computation, do so and return the new operand, otherwise return null.
static Value *simplifyValueKnownNonZero(Value *V, InstCombiner &IC,
                                        Instruction &CxtI) {
  // If V has multiple uses, then we would have to do more analysis to determine
  // if this is safe.  For example, the use could be in dynamically unreached
  // code.
  if (!V->hasOneUse()) return nullptr;

  bool MadeChange = false;

  // ((1 << A) >>u B) --> (1 << (A-B))
  // Because V cannot be zero, we know that B is less than A.
  Value *A = nullptr, *B = nullptr, *One = nullptr;
  if (match(V, m_LShr(m_OneUse(m_Shl(m_Value(One), m_Value(A))), m_Value(B))) &&
      match(One, m_One())) {
    A = IC.Builder->CreateSub(A, B);
    return IC.Builder->CreateShl(One, A);
  }

  // (PowerOfTwo >>u B) --> isExact since shifting out the result would make it
  // inexact.  Similarly for <<.
  if (BinaryOperator *I = dyn_cast<BinaryOperator>(V))
    if (I->isLogicalShift() &&
        isKnownToBeAPowerOfTwo(I->getOperand(0), IC.getDataLayout(), false, 0,
                               IC.getAssumptionCache(), &CxtI,
                               IC.getDominatorTree())) {
      // We know that this is an exact/nuw shift and that the input is a
      // non-zero context as well.
      if (Value *V2 = simplifyValueKnownNonZero(I->getOperand(0), IC, CxtI)) {
        I->setOperand(0, V2);
        MadeChange = true;
      }

      if (I->getOpcode() == Instruction::LShr && !I->isExact()) {
        I->setIsExact();
        MadeChange = true;
      }

      if (I->getOpcode() == Instruction::Shl && !I->hasNoUnsignedWrap()) {
        I->setHasNoUnsignedWrap();
        MadeChange = true;
      }
    }

  // TODO: Lots more we could do here:
  //    If V is a phi node, we can call this on each of its operands.
  //    "select cond, X, 0" can simplify to "X".

  return MadeChange ? V : nullptr;
}


/// MultiplyOverflows - True if the multiply can not be expressed in an int
/// this size.
static bool MultiplyOverflows(const APInt &C1, const APInt &C2, APInt &Product,
                              bool IsSigned) {
  bool Overflow;
  if (IsSigned)
    Product = C1.smul_ov(C2, Overflow);
  else
    Product = C1.umul_ov(C2, Overflow);

  return Overflow;
}

/// \brief True if C2 is a multiple of C1. Quotient contains C2/C1.
static bool IsMultiple(const APInt &C1, const APInt &C2, APInt &Quotient,
                       bool IsSigned) {
  assert(C1.getBitWidth() == C2.getBitWidth() &&
         "Inconsistent width of constants!");

  APInt Remainder(C1.getBitWidth(), /*Val=*/0ULL, IsSigned);
  if (IsSigned)
    APInt::sdivrem(C1, C2, Quotient, Remainder);
  else
    APInt::udivrem(C1, C2, Quotient, Remainder);

  return Remainder.isMinValue();
}

/// \brief A helper routine of InstCombiner::visitMul().
///
/// If C is a vector of known powers of 2, then this function returns
/// a new vector obtained from C replacing each element with its logBase2.
/// Return a null pointer otherwise.
static Constant *getLogBase2Vector(ConstantDataVector *CV) {
  const APInt *IVal;
  SmallVector<Constant *, 4> Elts;

  for (unsigned I = 0, E = CV->getNumElements(); I != E; ++I) {
    Constant *Elt = CV->getElementAsConstant(I);
    if (!match(Elt, m_APInt(IVal)) || !IVal->isPowerOf2())
      return nullptr;
    Elts.push_back(ConstantInt::get(Elt->getType(), IVal->logBase2()));
  }

  return ConstantVector::get(Elts);
}

/// \brief Return true if we can prove that:
///    (mul LHS, RHS)  === (mul nsw LHS, RHS)
bool InstCombiner::WillNotOverflowSignedMul(Value *LHS, Value *RHS,
                                            Instruction &CxtI) {
  // Multiplying n * m significant bits yields a result of n + m significant
  // bits. If the total number of significant bits does not exceed the
  // result bit width (minus 1), there is no overflow.
  // This means if we have enough leading sign bits in the operands
  // we can guarantee that the result does not overflow.
  // Ref: "Hacker's Delight" by Henry Warren
  unsigned BitWidth = LHS->getType()->getScalarSizeInBits();

  // Note that underestimating the number of sign bits gives a more
  // conservative answer.
  unsigned SignBits =
      ComputeNumSignBits(LHS, 0, &CxtI) + ComputeNumSignBits(RHS, 0, &CxtI);

  // First handle the easy case: if we have enough sign bits there's
  // definitely no overflow.
  if (SignBits > BitWidth + 1)
    return true;

  // There are two ambiguous cases where there can be no overflow:
  //   SignBits == BitWidth + 1    and
  //   SignBits == BitWidth
  // The second case is difficult to check, therefore we only handle the
  // first case.
  if (SignBits == BitWidth + 1) {
    // It overflows only when both arguments are negative and the true
    // product is exactly the minimum negative number.
    // E.g. mul i16 with 17 sign bits: 0xff00 * 0xff80 = 0x8000
    // For simplicity we just check if at least one side is not negative.
    bool LHSNonNegative, LHSNegative;
    bool RHSNonNegative, RHSNegative;
    ComputeSignBit(LHS, LHSNonNegative, LHSNegative, /*Depth=*/0, &CxtI);
    ComputeSignBit(RHS, RHSNonNegative, RHSNegative, /*Depth=*/0, &CxtI);
    if (LHSNonNegative || RHSNonNegative)
      return true;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyMulInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // X * -1 == 0 - X
  if (match(Op1, m_AllOnes())) {
    BinaryOperator *BO = BinaryOperator::CreateNeg(Op0, I.getName());
    if (I.hasNoSignedWrap())
      BO->setHasNoSignedWrap();
    return BO;
  }

  // Also allow combining multiply instructions on vectors.
  {
    Value *NewOp;
    Constant *C1, *C2;
    const APInt *IVal;
    if (match(&I, m_Mul(m_Shl(m_Value(NewOp), m_Constant(C2)),
                        m_Constant(C1))) &&
        match(C1, m_APInt(IVal))) {
      // ((X << C2)*C1) == (X * (C1 << C2))
      Constant *Shl = ConstantExpr::getShl(C1, C2);
      BinaryOperator *Mul = cast<BinaryOperator>(I.getOperand(0));
      BinaryOperator *BO = BinaryOperator::CreateMul(NewOp, Shl);
      if (I.hasNoUnsignedWrap() && Mul->hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && Mul->hasNoSignedWrap() &&
          Shl->isNotMinSignedValue())
        BO->setHasNoSignedWrap();
      return BO;
    }

    if (match(&I, m_Mul(m_Value(NewOp), m_Constant(C1)))) {
      Constant *NewCst = nullptr;
      if (match(C1, m_APInt(IVal)) && IVal->isPowerOf2())
        // Replace X*(2^C) with X << C, where C is either a scalar or a splat.
        NewCst = ConstantInt::get(NewOp->getType(), IVal->logBase2());
      else if (ConstantDataVector *CV = dyn_cast<ConstantDataVector>(C1))
        // Replace X*(2^C) with X << C, where C is a vector of known
        // constant powers of 2.
        NewCst = getLogBase2Vector(CV);

      if (NewCst) {
        unsigned Width = NewCst->getType()->getPrimitiveSizeInBits();
        BinaryOperator *Shl = BinaryOperator::CreateShl(NewOp, NewCst);

        if (I.hasNoUnsignedWrap())
          Shl->setHasNoUnsignedWrap();
        if (I.hasNoSignedWrap()) {
          uint64_t V;
          if (match(NewCst, m_ConstantInt(V)) && V != Width - 1)
            Shl->setHasNoSignedWrap();
        }

        return Shl;
      }
    }
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // (Y - X) * (-(2**n)) -> (X - Y) * (2**n), for positive nonzero n
    // (Y + const) * (-(2**n)) -> (-constY) * (2**n), for positive nonzero n
    // The "* (2**n)" thus becomes a potential shifting opportunity.
    {
      const APInt &   Val = CI->getValue();
      const APInt &PosVal = Val.abs();
      if (Val.isNegative() && PosVal.isPowerOf2()) {
        Value *X = nullptr, *Y = nullptr;
        if (Op0->hasOneUse()) {
          ConstantInt *C1;
          Value *Sub = nullptr;
          if (match(Op0, m_Sub(m_Value(Y), m_Value(X))))
            Sub = Builder->CreateSub(X, Y, "suba");
          else if (match(Op0, m_Add(m_Value(Y), m_ConstantInt(C1))))
            Sub = Builder->CreateSub(Builder->CreateNeg(C1), Y, "subc");
          if (Sub)
            return
              BinaryOperator::CreateMul(Sub,
                                        ConstantInt::get(Y->getType(), PosVal));
        }
      }
    }
  }

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;

    // Canonicalize (X+C1)*CI -> X*CI+C1*CI.
    {
      Value *X;
      Constant *C1;
      if (match(Op0, m_OneUse(m_Add(m_Value(X), m_Constant(C1))))) {
        Value *Mul = Builder->CreateMul(C1, Op1);
        // Only go forward with the transform if C1*CI simplifies to a tidier
        // constant.
        if (!match(Mul, m_Mul(m_Value(), m_Value())))
          return BinaryOperator::CreateAdd(Builder->CreateMul(X, Op1), Mul);
      }
    }
  }

  if (Value *Op0v = dyn_castNegVal(Op0)) {   // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(Op1)) {
      BinaryOperator *BO = BinaryOperator::CreateMul(Op0v, Op1v);
      if (I.hasNoSignedWrap() &&
          match(Op0, m_NSWSub(m_Value(), m_Value())) &&
          match(Op1, m_NSWSub(m_Value(), m_Value())))
        BO->setHasNoSignedWrap();
      return BO;
    }
  }

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Op1C = Op1;
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0);
    if (!BO ||
        (BO->getOpcode() != Instruction::UDiv &&
         BO->getOpcode() != Instruction::SDiv)) {
      Op1C = Op0;
      BO = dyn_cast<BinaryOperator>(Op1);
    }
    Value *Neg = dyn_castNegVal(Op1C);
    if (BO && BO->hasOneUse() &&
        (BO->getOperand(1) == Op1C || BO->getOperand(1) == Neg) &&
        (BO->getOpcode() == Instruction::UDiv ||
         BO->getOpcode() == Instruction::SDiv)) {
      Value *Op0BO = BO->getOperand(0), *Op1BO = BO->getOperand(1);

      // If the division is exact, X % Y is zero, so we end up with X or -X.
      if (PossiblyExactOperator *SDiv = dyn_cast<PossiblyExactOperator>(BO))
        if (SDiv->isExact()) {
          if (Op1BO == Op1C)
            return ReplaceInstUsesWith(I, Op0BO);
          return BinaryOperator::CreateNeg(Op0BO);
        }

      Value *Rem;
      if (BO->getOpcode() == Instruction::UDiv)
        Rem = Builder->CreateURem(Op0BO, Op1BO);
      else
        Rem = Builder->CreateSRem(Op0BO, Op1BO);
      Rem->takeName(BO);

      if (Op1BO == Op1C)
        return BinaryOperator::CreateSub(Op0BO, Rem);
      return BinaryOperator::CreateSub(Rem, Op0BO);
    }
  }

  /// i1 mul -> i1 and.
  if (I.getType()->getScalarType()->isIntegerTy(1))
    return BinaryOperator::CreateAnd(Op0, Op1);

  // X*(1 << Y) --> X << Y
  // (1 << Y)*X --> X << Y
  {
    Value *Y;
    BinaryOperator *BO = nullptr;
    bool ShlNSW = false;
    if (match(Op0, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op1, Y);
      ShlNSW = cast<ShlOperator>(Op0)->hasNoSignedWrap();
    } else if (match(Op1, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op0, Y);
      ShlNSW = cast<ShlOperator>(Op1)->hasNoSignedWrap();
    }
    if (BO) {
      if (I.hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && ShlNSW)
        BO->setHasNoSignedWrap();
      return BO;
    }
  }

  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  //   X * Y (where Y is 0 or 1) -> X & (0-Y)
  if (!I.getType()->isVectorTy()) {
    // -2 is "-1 << 1" so it is all bits set except the low one.
    APInt Negative2(I.getType()->getPrimitiveSizeInBits(), (uint64_t)-2, true);

    Value *BoolCast = nullptr, *OtherOp = nullptr;
    if (MaskedValueIsZero(Op0, Negative2, 0, &I))
      BoolCast = Op0, OtherOp = Op1;
    else if (MaskedValueIsZero(Op1, Negative2, 0, &I))
      BoolCast = Op1, OtherOp = Op0;

    if (BoolCast) {
      Value *V = Builder->CreateSub(Constant::getNullValue(I.getType()),
                                    BoolCast);
      return BinaryOperator::CreateAnd(V, OtherOp);
    }
  }

  if (!I.hasNoSignedWrap() && WillNotOverflowSignedMul(Op0, Op1, I)) {
    Changed = true;
    I.setHasNoSignedWrap(true);
  }

  if (!I.hasNoUnsignedWrap() &&
      computeOverflowForUnsignedMul(Op0, Op1, &I) ==
          OverflowResult::NeverOverflows) {
    Changed = true;
    I.setHasNoUnsignedWrap(true);
  }

  return Changed ? &I : nullptr;
}

/// Detect pattern log2(Y * 0.5) with corresponding fast math flags.
static void detectLog2OfHalf(Value *&Op, Value *&Y, IntrinsicInst *&Log2) {
  if (!Op->hasOneUse())
    return;

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(Op);
  if (!II)
    return;
  if (II->getIntrinsicID() != Intrinsic::log2 || !II->hasUnsafeAlgebra())
    return;
  Log2 = II;

  Value *OpLog2Of = II->getArgOperand(0);
  if (!OpLog2Of->hasOneUse())
    return;

  Instruction *I = dyn_cast<Instruction>(OpLog2Of);
  if (!I)
    return;
  if (I->getOpcode() != Instruction::FMul || !I->hasUnsafeAlgebra())
    return;

  if (match(I->getOperand(0), m_SpecificFP(0.5)))
    Y = I->getOperand(1);
  else if (match(I->getOperand(1), m_SpecificFP(0.5)))
    Y = I->getOperand(0);
}

static bool isFiniteNonZeroFp(Constant *C) {
  if (C->getType()->isVectorTy()) {
    for (unsigned I = 0, E = C->getType()->getVectorNumElements(); I != E;
         ++I) {
      ConstantFP *CFP = dyn_cast_or_null<ConstantFP>(C->getAggregateElement(I));
      if (!CFP || !CFP->getValueAPF().isFiniteNonZero())
        return false;
    }
    return true;
  }

  return isa<ConstantFP>(C) &&
         cast<ConstantFP>(C)->getValueAPF().isFiniteNonZero();
}

static bool isNormalFp(Constant *C) {
  if (C->getType()->isVectorTy()) {
    for (unsigned I = 0, E = C->getType()->getVectorNumElements(); I != E;
         ++I) {
      ConstantFP *CFP = dyn_cast_or_null<ConstantFP>(C->getAggregateElement(I));
      if (!CFP || !CFP->getValueAPF().isNormal())
        return false;
    }
    return true;
  }

  return isa<ConstantFP>(C) && cast<ConstantFP>(C)->getValueAPF().isNormal();
}

/// Helper function of InstCombiner::visitFMul(BinaryOperator(). It returns
/// true iff the given value is FMul or FDiv with one and only one operand
/// being a normal constant (i.e. not Zero/NaN/Infinity).
static bool isFMulOrFDivWithConstant(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || (I->getOpcode() != Instruction::FMul &&
             I->getOpcode() != Instruction::FDiv))
    return false;

  Constant *C0 = dyn_cast<Constant>(I->getOperand(0));
  Constant *C1 = dyn_cast<Constant>(I->getOperand(1));

  if (C0 && C1)
    return false;

  return (C0 && isFiniteNonZeroFp(C0)) || (C1 && isFiniteNonZeroFp(C1));
}

/// foldFMulConst() is a helper routine of InstCombiner::visitFMul().
/// The input \p FMulOrDiv is a FMul/FDiv with one and only one operand
/// being a constant (i.e. isFMulOrFDivWithConstant(FMulOrDiv) == true).
/// This function is to simplify "FMulOrDiv * C" and returns the
/// resulting expression. Note that this function could return NULL in
/// case the constants cannot be folded into a normal floating-point.
///
Value *InstCombiner::foldFMulConst(Instruction *FMulOrDiv, Constant *C,
                                   Instruction *InsertBefore) {
  assert(isFMulOrFDivWithConstant(FMulOrDiv) && "V is invalid");

  Value *Opnd0 = FMulOrDiv->getOperand(0);
  Value *Opnd1 = FMulOrDiv->getOperand(1);

  Constant *C0 = dyn_cast<Constant>(Opnd0);
  Constant *C1 = dyn_cast<Constant>(Opnd1);

  BinaryOperator *R = nullptr;

  // (X * C0) * C => X * (C0*C)
  if (FMulOrDiv->getOpcode() == Instruction::FMul) {
    Constant *F = ConstantExpr::getFMul(C1 ? C1 : C0, C);
    if (isNormalFp(F))
      R = BinaryOperator::CreateFMul(C1 ? Opnd0 : Opnd1, F);
  } else {
    if (C0) {
      // (C0 / X) * C => (C0 * C) / X
      if (FMulOrDiv->hasOneUse()) {
        // It would otherwise introduce another div.
        Constant *F = ConstantExpr::getFMul(C0, C);
        if (isNormalFp(F))
          R = BinaryOperator::CreateFDiv(F, Opnd1);
      }
    } else {
      // (X / C1) * C => X * (C/C1) if C/C1 is not a denormal
      Constant *F = ConstantExpr::getFDiv(C, C1);
      if (isNormalFp(F)) {
        R = BinaryOperator::CreateFMul(Opnd0, F);
      } else {
        // (X / C1) * C => X / (C1/C)
        Constant *F = ConstantExpr::getFDiv(C1, C);
        if (isNormalFp(F))
          R = BinaryOperator::CreateFDiv(Opnd0, F);
      }
    }
  }

  if (R) {
    R->setHasUnsafeAlgebra(true);
    InsertNewInstWith(R, *InsertBefore);
  }

  return R;
}

Instruction *InstCombiner::visitFMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (isa<Constant>(Op0))
    std::swap(Op0, Op1);

  if (Value *V =
          SimplifyFMulInst(Op0, Op1, I.getFastMathFlags(), DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  bool AllowReassociate = I.hasUnsafeAlgebra();

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;

    // (fmul X, -1.0) --> (fsub -0.0, X)
    if (match(Op1, m_SpecificFP(-1.0))) {
      Constant *NegZero = ConstantFP::getNegativeZero(Op1->getType());
      Instruction *RI = BinaryOperator::CreateFSub(NegZero, Op0);
      RI->copyFastMathFlags(&I);
      return RI;
    }

    Constant *C = cast<Constant>(Op1);
    if (AllowReassociate && isFiniteNonZeroFp(C)) {
      // Let MDC denote an expression in one of these forms:
      // X * C, C/X, X/C, where C is a constant.
      //
      // Try to simplify "MDC * Constant"
      if (isFMulOrFDivWithConstant(Op0))
        if (Value *V = foldFMulConst(cast<Instruction>(Op0), C, &I))
          return ReplaceInstUsesWith(I, V);

      // (MDC +/- C1) * C => (MDC * C) +/- (C1 * C)
      Instruction *FAddSub = dyn_cast<Instruction>(Op0);
      if (FAddSub &&
          (FAddSub->getOpcode() == Instruction::FAdd ||
           FAddSub->getOpcode() == Instruction::FSub)) {
        Value *Opnd0 = FAddSub->getOperand(0);
        Value *Opnd1 = FAddSub->getOperand(1);
        Constant *C0 = dyn_cast<Constant>(Opnd0);
        Constant *C1 = dyn_cast<Constant>(Opnd1);
        bool Swap = false;
        if (C0) {
          std::swap(C0, C1);
          std::swap(Opnd0, Opnd1);
          Swap = true;
        }

        if (C1 && isFiniteNonZeroFp(C1) && isFMulOrFDivWithConstant(Opnd0)) {
          Value *M1 = ConstantExpr::getFMul(C1, C);
          Value *M0 = isNormalFp(cast<Constant>(M1)) ?
                      foldFMulConst(cast<Instruction>(Opnd0), C, &I) :
                      nullptr;
          if (M0 && M1) {
            if (Swap && FAddSub->getOpcode() == Instruction::FSub)
              std::swap(M0, M1);

            Instruction *RI = (FAddSub->getOpcode() == Instruction::FAdd)
                                  ? BinaryOperator::CreateFAdd(M0, M1)
                                  : BinaryOperator::CreateFSub(M0, M1);
            RI->copyFastMathFlags(&I);
            return RI;
          }
        }
      }
    }
  }

  // sqrt(X) * sqrt(X) -> X
  if (AllowReassociate && (Op0 == Op1))
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Op0))
      if (II->getIntrinsicID() == Intrinsic::sqrt)
        return ReplaceInstUsesWith(I, II->getOperand(0));

  // Under unsafe algebra do:
  // X * log2(0.5*Y) = X*log2(Y) - X
  if (AllowReassociate) {
    Value *OpX = nullptr;
    Value *OpY = nullptr;
    IntrinsicInst *Log2;
    detectLog2OfHalf(Op0, OpY, Log2);
    if (OpY) {
      OpX = Op1;
    } else {
      detectLog2OfHalf(Op1, OpY, Log2);
      if (OpY) {
        OpX = Op0;
      }
    }
    // if pattern detected emit alternate sequence
    if (OpX && OpY) {
      BuilderTy::FastMathFlagGuard Guard(*Builder);
      Builder->SetFastMathFlags(Log2->getFastMathFlags());
      Log2->setArgOperand(0, OpY);
      Value *FMulVal = Builder->CreateFMul(OpX, Log2);
      Value *FSub = Builder->CreateFSub(FMulVal, OpX);
      FSub->takeName(&I);
      return ReplaceInstUsesWith(I, FSub);
    }
  }

  // Handle symmetric situation in a 2-iteration loop
  Value *Opnd0 = Op0;
  Value *Opnd1 = Op1;
  for (int i = 0; i < 2; i++) {
    bool IgnoreZeroSign = I.hasNoSignedZeros();
    if (BinaryOperator::isFNeg(Opnd0, IgnoreZeroSign)) {
      BuilderTy::FastMathFlagGuard Guard(*Builder);
      Builder->SetFastMathFlags(I.getFastMathFlags());

      Value *N0 = dyn_castFNegVal(Opnd0, IgnoreZeroSign);
      Value *N1 = dyn_castFNegVal(Opnd1, IgnoreZeroSign);

      // -X * -Y => X*Y
      if (N1) {
        Value *FMul = Builder->CreateFMul(N0, N1);
        FMul->takeName(&I);
        return ReplaceInstUsesWith(I, FMul);
      }

      if (Opnd0->hasOneUse()) {
        // -X * Y => -(X*Y) (Promote negation as high as possible)
        Value *T = Builder->CreateFMul(N0, Opnd1);
        Value *Neg = Builder->CreateFNeg(T);
        Neg->takeName(&I);
        return ReplaceInstUsesWith(I, Neg);
      }
    }

    // (X*Y) * X => (X*X) * Y where Y != X
    //  The purpose is two-fold:
    //   1) to form a power expression (of X).
    //   2) potentially shorten the critical path: After transformation, the
    //  latency of the instruction Y is amortized by the expression of X*X,
    //  and therefore Y is in a "less critical" position compared to what it
    //  was before the transformation.
    //
    if (AllowReassociate) {
      Value *Opnd0_0, *Opnd0_1;
      if (Opnd0->hasOneUse() &&
          match(Opnd0, m_FMul(m_Value(Opnd0_0), m_Value(Opnd0_1)))) {
        Value *Y = nullptr;
        if (Opnd0_0 == Opnd1 && Opnd0_1 != Opnd1)
          Y = Opnd0_1;
        else if (Opnd0_1 == Opnd1 && Opnd0_0 != Opnd1)
          Y = Opnd0_0;

        if (Y) {
          BuilderTy::FastMathFlagGuard Guard(*Builder);
          Builder->SetFastMathFlags(I.getFastMathFlags());
          Value *T = Builder->CreateFMul(Opnd1, Opnd1);

          Value *R = Builder->CreateFMul(T, Y);
          R->takeName(&I);
          return ReplaceInstUsesWith(I, R);
        }
      }
    }

    if (!isa<Constant>(Op1))
      std::swap(Opnd0, Opnd1);
    else
      break;
  }

  return Changed ? &I : nullptr;
}

/// SimplifyDivRemOfSelect - Try to fold a divide or remainder of a select
/// instruction.
bool InstCombiner::SimplifyDivRemOfSelect(BinaryOperator &I) {
  SelectInst *SI = cast<SelectInst>(I.getOperand(1));

  // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
  int NonNullOperand = -1;
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
    if (ST->isNullValue())
      NonNullOperand = 2;
  // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
    if (ST->isNullValue())
      NonNullOperand = 1;

  if (NonNullOperand == -1)
    return false;

  Value *SelectCond = SI->getOperand(0);

  // Change the div/rem to use 'Y' instead of the select.
  I.setOperand(1, SI->getOperand(NonNullOperand));

  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.

  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;

  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = &I, BBFront = I.getParent()->begin();

  while (BBI != BBFront) {
    --BBI;
    // If we found a call to a function, we can't assume it will return, so
    // information from below it cannot be propagated above it.
    if (isa<CallInst>(BBI) && !isa<IntrinsicInst>(BBI))
      break;

    // Replace uses of the select or its condition with the known values.
    for (Instruction::op_iterator I = BBI->op_begin(), E = BBI->op_end();
         I != E; ++I) {
      if (*I == SI) {
        *I = SI->getOperand(NonNullOperand);
        Worklist.Add(BBI);
      } else if (*I == SelectCond) {
        *I = Builder->getInt1(NonNullOperand == 1);
        Worklist.Add(BBI);
      }
    }

    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = nullptr;
    if (&*BBI == SelectCond)
      SelectCond = nullptr;

    // If we ran out of things to eliminate, break out of the loop.
    if (!SelectCond && !SI)
      break;

  }
  return true;
}


/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (Instruction *LHS = dyn_cast<Instruction>(Op0)) {
    const APInt *C2;
    if (match(Op1, m_APInt(C2))) {
      Value *X;
      const APInt *C1;
      bool IsSigned = I.getOpcode() == Instruction::SDiv;

      // (X / C1) / C2  -> X / (C1*C2)
      if ((IsSigned && match(LHS, m_SDiv(m_Value(X), m_APInt(C1)))) ||
          (!IsSigned && match(LHS, m_UDiv(m_Value(X), m_APInt(C1))))) {
        APInt Product(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);
        if (!MultiplyOverflows(*C1, *C2, Product, IsSigned))
          return BinaryOperator::Create(I.getOpcode(), X,
                                        ConstantInt::get(I.getType(), Product));
      }

      if ((IsSigned && match(LHS, m_NSWMul(m_Value(X), m_APInt(C1)))) ||
          (!IsSigned && match(LHS, m_NUWMul(m_Value(X), m_APInt(C1))))) {
        APInt Quotient(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);

        // (X * C1) / C2 -> X / (C2 / C1) if C2 is a multiple of C1.
        if (IsMultiple(*C2, *C1, Quotient, IsSigned)) {
          BinaryOperator *BO = BinaryOperator::Create(
              I.getOpcode(), X, ConstantInt::get(X->getType(), Quotient));
          BO->setIsExact(I.isExact());
          return BO;
        }

        // (X * C1) / C2 -> X * (C1 / C2) if C1 is a multiple of C2.
        if (IsMultiple(*C1, *C2, Quotient, IsSigned)) {
          BinaryOperator *BO = BinaryOperator::Create(
              Instruction::Mul, X, ConstantInt::get(X->getType(), Quotient));
          BO->setHasNoUnsignedWrap(
              !IsSigned &&
              cast<OverflowingBinaryOperator>(LHS)->hasNoUnsignedWrap());
          BO->setHasNoSignedWrap(
              cast<OverflowingBinaryOperator>(LHS)->hasNoSignedWrap());
          return BO;
        }
      }

      if ((IsSigned && match(LHS, m_NSWShl(m_Value(X), m_APInt(C1))) &&
           *C1 != C1->getBitWidth() - 1) ||
          (!IsSigned && match(LHS, m_NUWShl(m_Value(X), m_APInt(C1))))) {
        APInt Quotient(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);
        APInt C1Shifted = APInt::getOneBitSet(
            C1->getBitWidth(), static_cast<unsigned>(C1->getLimitedValue()));

        // (X << C1) / C2 -> X / (C2 >> C1) if C2 is a multiple of C1.
        if (IsMultiple(*C2, C1Shifted, Quotient, IsSigned)) {
          BinaryOperator *BO = BinaryOperator::Create(
              I.getOpcode(), X, ConstantInt::get(X->getType(), Quotient));
          BO->setIsExact(I.isExact());
          return BO;
        }

        // (X << C1) / C2 -> X * (C2 >> C1) if C1 is a multiple of C2.
        if (IsMultiple(C1Shifted, *C2, Quotient, IsSigned)) {
          BinaryOperator *BO = BinaryOperator::Create(
              Instruction::Mul, X, ConstantInt::get(X->getType(), Quotient));
          BO->setHasNoUnsignedWrap(
              !IsSigned &&
              cast<OverflowingBinaryOperator>(LHS)->hasNoUnsignedWrap());
          BO->setHasNoSignedWrap(
              cast<OverflowingBinaryOperator>(LHS)->hasNoSignedWrap());
          return BO;
        }
      }

      if (*C2 != 0) { // avoid X udiv 0
        if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
          if (Instruction *R = FoldOpIntoSelect(I, SI))
            return R;
        if (isa<PHINode>(Op0))
          if (Instruction *NV = FoldOpIntoPhi(I))
            return NV;
      }
    }
  }

  if (ConstantInt *One = dyn_cast<ConstantInt>(Op0)) {
    if (One->isOne() && !I.getType()->isIntegerTy(1)) {
      bool isSigned = I.getOpcode() == Instruction::SDiv;
      if (isSigned) {
        // If Op1 is 0 then it's undefined behaviour, if Op1 is 1 then the
        // result is one, if Op1 is -1 then the result is minus one, otherwise
        // it's zero.
        Value *Inc = Builder->CreateAdd(Op1, One);
        Value *Cmp = Builder->CreateICmpULT(
                         Inc, ConstantInt::get(I.getType(), 3));
        return SelectInst::Create(Cmp, Op1, ConstantInt::get(I.getType(), 0));
      } else {
        // If Op1 is 0 then it's undefined behaviour. If Op1 is 1 then the
        // result is one, otherwise it's zero.
        return new ZExtInst(Builder->CreateICmpEQ(Op1, One), I.getType());
      }
    }
  }

  // See if we can fold away this div instruction.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // (X - (X rem Y)) / Y -> X / Y; usually originates as ((X / Y) * Y) / Y
  Value *X = nullptr, *Z = nullptr;
  if (match(Op0, m_Sub(m_Value(X), m_Value(Z)))) { // (X - Z) / Y; Y = Op1
    bool isSigned = I.getOpcode() == Instruction::SDiv;
    if ((isSigned && match(Z, m_SRem(m_Specific(X), m_Specific(Op1)))) ||
        (!isSigned && match(Z, m_URem(m_Specific(X), m_Specific(Op1)))))
      return BinaryOperator::Create(I.getOpcode(), X, Op1);
  }

  return nullptr;
}

/// dyn_castZExtVal - Checks if V is a zext or constant that can
/// be truncated to Ty without losing bits.
static Value *dyn_castZExtVal(Value *V, Type *Ty) {
  if (ZExtInst *Z = dyn_cast<ZExtInst>(V)) {
    if (Z->getSrcTy() == Ty)
      return Z->getOperand(0);
  } else if (ConstantInt *C = dyn_cast<ConstantInt>(V)) {
    if (C->getValue().getActiveBits() <= cast<IntegerType>(Ty)->getBitWidth())
      return ConstantExpr::getTrunc(C, Ty);
  }
  return nullptr;
}

namespace {
const unsigned MaxDepth = 6;
typedef Instruction *(*FoldUDivOperandCb)(Value *Op0, Value *Op1,
                                          const BinaryOperator &I,
                                          InstCombiner &IC);

/// \brief Used to maintain state for visitUDivOperand().
struct UDivFoldAction {
  FoldUDivOperandCb FoldAction; ///< Informs visitUDiv() how to fold this
                                ///< operand.  This can be zero if this action
                                ///< joins two actions together.

  Value *OperandToFold;         ///< Which operand to fold.
  union {
    Instruction *FoldResult;    ///< The instruction returned when FoldAction is
                                ///< invoked.

    size_t SelectLHSIdx;        ///< Stores the LHS action index if this action
                                ///< joins two actions together.
  };

  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand)
      : FoldAction(FA), OperandToFold(InputOperand), FoldResult(nullptr) {}
  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand, size_t SLHS)
      : FoldAction(FA), OperandToFold(InputOperand), SelectLHSIdx(SLHS) {}
};
}

// X udiv 2^C -> X >> C
static Instruction *foldUDivPow2Cst(Value *Op0, Value *Op1,
                                    const BinaryOperator &I, InstCombiner &IC) {
  const APInt &C = cast<Constant>(Op1)->getUniqueInteger();
  BinaryOperator *LShr = BinaryOperator::CreateLShr(
      Op0, ConstantInt::get(Op0->getType(), C.logBase2()));
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// X udiv C, where C >= signbit
static Instruction *foldUDivNegCst(Value *Op0, Value *Op1,
                                   const BinaryOperator &I, InstCombiner &IC) {
  Value *ICI = IC.Builder->CreateICmpULT(Op0, cast<ConstantInt>(Op1));

  return SelectInst::Create(ICI, Constant::getNullValue(I.getType()),
                            ConstantInt::get(I.getType(), 1));
}

// X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
static Instruction *foldUDivShl(Value *Op0, Value *Op1, const BinaryOperator &I,
                                InstCombiner &IC) {
  Instruction *ShiftLeft = cast<Instruction>(Op1);
  if (isa<ZExtInst>(ShiftLeft))
    ShiftLeft = cast<Instruction>(ShiftLeft->getOperand(0));

  const APInt &CI =
      cast<Constant>(ShiftLeft->getOperand(0))->getUniqueInteger();
  Value *N = ShiftLeft->getOperand(1);
  if (CI != 1)
    N = IC.Builder->CreateAdd(N, ConstantInt::get(N->getType(), CI.logBase2()));
  if (ZExtInst *Z = dyn_cast<ZExtInst>(Op1))
    N = IC.Builder->CreateZExt(N, Z->getDestTy());
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, N);
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// \brief Recursively visits the possible right hand operands of a udiv
// instruction, seeing through select instructions, to determine if we can
// replace the udiv with something simpler.  If we find that an operand is not
// able to simplify the udiv, we abort the entire transformation.
static size_t visitUDivOperand(Value *Op0, Value *Op1, const BinaryOperator &I,
                               SmallVectorImpl<UDivFoldAction> &Actions,
                               unsigned Depth = 0) {
  // Check to see if this is an unsigned division with an exact power of 2,
  // if so, convert to a right shift.
  if (match(Op1, m_Power2())) {
    Actions.push_back(UDivFoldAction(foldUDivPow2Cst, Op1));
    return Actions.size();
  }

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1))
    // X udiv C, where C >= signbit
    if (C->getValue().isNegative()) {
      Actions.push_back(UDivFoldAction(foldUDivNegCst, C));
      return Actions.size();
    }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (match(Op1, m_Shl(m_Power2(), m_Value())) ||
      match(Op1, m_ZExt(m_Shl(m_Power2(), m_Value())))) {
    Actions.push_back(UDivFoldAction(foldUDivShl, Op1));
    return Actions.size();
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxDepth)
    return 0;

  if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
    if (size_t LHSIdx =
            visitUDivOperand(Op0, SI->getOperand(1), I, Actions, Depth))
      if (visitUDivOperand(Op0, SI->getOperand(2), I, Actions, Depth)) {
        Actions.push_back(UDivFoldAction(nullptr, Op1, LHSIdx - 1));
        return Actions.size();
      }

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyUDivInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  // (x lshr C1) udiv C2 --> x udiv (C2 << C1)
  {
    Value *X;
    const APInt *C1, *C2;
    if (match(Op0, m_LShr(m_Value(X), m_APInt(C1))) &&
        match(Op1, m_APInt(C2))) {
      bool Overflow;
      APInt C2ShlC1 = C2->ushl_ov(*C1, Overflow);
      if (!Overflow) {
        bool IsExact = I.isExact() && match(Op0, m_Exact(m_Value()));
        BinaryOperator *BO = BinaryOperator::CreateUDiv(
            X, ConstantInt::get(X->getType(), C2ShlC1));
        if (IsExact)
          BO->setIsExact();
        return BO;
      }
    }
  }

  // (zext A) udiv (zext B) --> zext (A udiv B)
  if (ZExtInst *ZOp0 = dyn_cast<ZExtInst>(Op0))
    if (Value *ZOp1 = dyn_castZExtVal(Op1, ZOp0->getSrcTy()))
      return new ZExtInst(
          Builder->CreateUDiv(ZOp0->getOperand(0), ZOp1, "div", I.isExact()),
          I.getType());

  // (LHS udiv (select (select (...)))) -> (LHS >> (select (select (...))))
  SmallVector<UDivFoldAction, 6> UDivActions;
  if (visitUDivOperand(Op0, Op1, I, UDivActions))
    for (unsigned i = 0, e = UDivActions.size(); i != e; ++i) {
      FoldUDivOperandCb Action = UDivActions[i].FoldAction;
      Value *ActionOp1 = UDivActions[i].OperandToFold;
      Instruction *Inst;
      if (Action)
        Inst = Action(Op0, ActionOp1, I, *this);
      else {
        // This action joins two actions together.  The RHS of this action is
        // simply the last action we processed, we saved the LHS action index in
        // the joining action.
        size_t SelectRHSIdx = i - 1;
        Value *SelectRHS = UDivActions[SelectRHSIdx].FoldResult;
        size_t SelectLHSIdx = UDivActions[i].SelectLHSIdx;
        Value *SelectLHS = UDivActions[SelectLHSIdx].FoldResult;
        Inst = SelectInst::Create(cast<SelectInst>(ActionOp1)->getCondition(),
                                  SelectLHS, SelectRHS);
      }

      // If this is the last action to process, return it to the InstCombiner.
      // Otherwise, we insert it before the UDiv and record it so that we may
      // use it as part of a joining action (i.e., a SelectInst).
      if (e - i != 1) {
        Inst->insertBefore(&I);
        UDivActions[i].FoldResult = Inst;
      } else
        return Inst;
    }

  return nullptr;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifySDivInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  // sdiv X, -1 == -X
  if (match(Op1, m_AllOnes()))
    return BinaryOperator::CreateNeg(Op0);

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // sdiv X, C  -->  ashr exact X, log2(C)
    if (I.isExact() && RHS->getValue().isNonNegative() &&
        RHS->getValue().isPowerOf2()) {
      Value *ShAmt = llvm::ConstantInt::get(RHS->getType(),
                                            RHS->getValue().exactLogBase2());
      return BinaryOperator::CreateExactAShr(Op0, ShAmt, I.getName());
    }
  }

  if (Constant *RHS = dyn_cast<Constant>(Op1)) {
    // X/INT_MIN -> X == INT_MIN
    if (RHS->isMinSignedValue())
      return new ZExtInst(Builder->CreateICmpEQ(Op0, Op1), I.getType());

    // -X/C  -->  X/-C  provided the negation doesn't overflow.
    Value *X;
    if (match(Op0, m_NSWSub(m_Zero(), m_Value(X)))) {
      auto *BO = BinaryOperator::CreateSDiv(X, ConstantExpr::getNeg(RHS));
      BO->setIsExact(I.isExact());
      return BO;
    }
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isIntegerTy()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op0, Mask, 0, &I)) {
      if (MaskedValueIsZero(Op1, Mask, 0, &I)) {
        // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
        auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
        BO->setIsExact(I.isExact());
        return BO;
      }

      if (isKnownToBeAPowerOfTwo(Op1, DL, /*OrZero*/ true, 0, AC, &I, DT)) {
        // X sdiv (1 << Y) -> X udiv (1 << Y) ( -> X u>> Y)
        // Safe because the only negative value (1 << Y) can take on is
        // INT_MIN, and X sdiv INT_MIN == X udiv INT_MIN == 0 if X doesn't have
        // the sign bit set.
        auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
        BO->setIsExact(I.isExact());
        return BO;
      }
    }
  }

  return nullptr;
}

/// CvtFDivConstToReciprocal tries to convert X/C into X*1/C if C not a special
/// FP value and:
///    1) 1/C is exact, or
///    2) reciprocal is allowed.
/// If the conversion was successful, the simplified expression "X * 1/C" is
/// returned; otherwise, NULL is returned.
///
static Instruction *CvtFDivConstToReciprocal(Value *Dividend, Constant *Divisor,
                                             bool AllowReciprocal) {
  if (!isa<ConstantFP>(Divisor)) // TODO: handle vectors.
    return nullptr;

  const APFloat &FpVal = cast<ConstantFP>(Divisor)->getValueAPF();
  APFloat Reciprocal(FpVal.getSemantics());
  bool Cvt = FpVal.getExactInverse(&Reciprocal);

  if (!Cvt && AllowReciprocal && FpVal.isFiniteNonZero()) {
    Reciprocal = APFloat(FpVal.getSemantics(), 1.0f);
    (void)Reciprocal.divide(FpVal, APFloat::rmNearestTiesToEven);
    Cvt = !Reciprocal.isDenormal();
  }

  if (!Cvt)
    return nullptr;

  ConstantFP *R;
  R = ConstantFP::get(Dividend->getType()->getContext(), Reciprocal);
  return BinaryOperator::CreateFMul(Dividend, R);
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyFDivInst(Op0, Op1, I.getFastMathFlags(),
                                  DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  bool AllowReassociate = I.hasUnsafeAlgebra();
  bool AllowReciprocal = I.hasAllowReciprocal();

  if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (AllowReassociate) {
      Constant *C1 = nullptr;
      Constant *C2 = Op1C;
      Value *X;
      Instruction *Res = nullptr;

      if (match(Op0, m_FMul(m_Value(X), m_Constant(C1)))) {
        // (X*C1)/C2 => X * (C1/C2)
        //
        Constant *C = ConstantExpr::getFDiv(C1, C2);
        if (isNormalFp(C))
          Res = BinaryOperator::CreateFMul(X, C);
      } else if (match(Op0, m_FDiv(m_Value(X), m_Constant(C1)))) {
        // (X/C1)/C2 => X /(C2*C1) [=> X * 1/(C2*C1) if reciprocal is allowed]
        //
        Constant *C = ConstantExpr::getFMul(C1, C2);
        if (isNormalFp(C)) {
          Res = CvtFDivConstToReciprocal(X, C, AllowReciprocal);
          if (!Res)
            Res = BinaryOperator::CreateFDiv(X, C);
        }
      }

      if (Res) {
        Res->setFastMathFlags(I.getFastMathFlags());
        return Res;
      }
    }

    // X / C => X * 1/C
    if (Instruction *T = CvtFDivConstToReciprocal(Op0, Op1C, AllowReciprocal)) {
      T->copyFastMathFlags(&I);
      return T;
    }

    return nullptr;
  }

  if (AllowReassociate && isa<Constant>(Op0)) {
    Constant *C1 = cast<Constant>(Op0), *C2;
    Constant *Fold = nullptr;
    Value *X;
    bool CreateDiv = true;

    // C1 / (X*C2) => (C1/C2) / X
    if (match(Op1, m_FMul(m_Value(X), m_Constant(C2))))
      Fold = ConstantExpr::getFDiv(C1, C2);
    else if (match(Op1, m_FDiv(m_Value(X), m_Constant(C2)))) {
      // C1 / (X/C2) => (C1*C2) / X
      Fold = ConstantExpr::getFMul(C1, C2);
    } else if (match(Op1, m_FDiv(m_Constant(C2), m_Value(X)))) {
      // C1 / (C2/X) => (C1/C2) * X
      Fold = ConstantExpr::getFDiv(C1, C2);
      CreateDiv = false;
    }

    if (Fold && isNormalFp(Fold)) {
      Instruction *R = CreateDiv ? BinaryOperator::CreateFDiv(Fold, X)
                                 : BinaryOperator::CreateFMul(X, Fold);
      R->setFastMathFlags(I.getFastMathFlags());
      return R;
    }
    return nullptr;
  }

  if (AllowReassociate) {
    Value *X, *Y;
    Value *NewInst = nullptr;
    Instruction *SimpR = nullptr;

    if (Op0->hasOneUse() && match(Op0, m_FDiv(m_Value(X), m_Value(Y)))) {
      // (X/Y) / Z => X / (Y*Z)
      //
      if (!isa<Constant>(Y) || !isa<Constant>(Op1)) {
        NewInst = Builder->CreateFMul(Y, Op1);
        if (Instruction *RI = dyn_cast<Instruction>(NewInst)) {
          FastMathFlags Flags = I.getFastMathFlags();
          Flags &= cast<Instruction>(Op0)->getFastMathFlags();
          RI->setFastMathFlags(Flags);
        }
        SimpR = BinaryOperator::CreateFDiv(X, NewInst);
      }
    } else if (Op1->hasOneUse() && match(Op1, m_FDiv(m_Value(X), m_Value(Y)))) {
      // Z / (X/Y) => Z*Y / X
      //
      if (!isa<Constant>(Y) || !isa<Constant>(Op0)) {
        NewInst = Builder->CreateFMul(Op0, Y);
        if (Instruction *RI = dyn_cast<Instruction>(NewInst)) {
          FastMathFlags Flags = I.getFastMathFlags();
          Flags &= cast<Instruction>(Op1)->getFastMathFlags();
          RI->setFastMathFlags(Flags);
        }
        SimpR = BinaryOperator::CreateFDiv(NewInst, X);
      }
    }

    if (NewInst) {
      if (Instruction *T = dyn_cast<Instruction>(NewInst))
        T->setDebugLoc(I.getDebugLoc());
      SimpR->setFastMathFlags(I.getFastMathFlags());
      return SimpR;
    }
  }

  return nullptr;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (isa<Constant>(Op1)) {
    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      } else if (isa<PHINode>(Op0I)) {
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
      }

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitURem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyURemInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  if (Instruction *common = commonIRemTransforms(I))
    return common;

  // (zext A) urem (zext B) --> zext (A urem B)
  if (ZExtInst *ZOp0 = dyn_cast<ZExtInst>(Op0))
    if (Value *ZOp1 = dyn_castZExtVal(Op1, ZOp0->getSrcTy()))
      return new ZExtInst(Builder->CreateURem(ZOp0->getOperand(0), ZOp1),
                          I.getType());

  // X urem Y -> X and Y-1, where Y is a power of 2,
  if (isKnownToBeAPowerOfTwo(Op1, DL, /*OrZero*/ true, 0, AC, &I, DT)) {
    Constant *N1 = Constant::getAllOnesValue(I.getType());
    Value *Add = Builder->CreateAdd(Op1, N1);
    return BinaryOperator::CreateAnd(Op0, Add);
  }

  // 1 urem X -> zext(X != 1)
  if (match(Op0, m_One())) {
    Value *Cmp = Builder->CreateICmpNE(Op1, Op0);
    Value *Ext = Builder->CreateZExt(Cmp, I.getType());
    return ReplaceInstUsesWith(I, Ext);
  }

  return nullptr;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifySRemInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // Handle the integer rem common cases
  if (Instruction *Common = commonIRemTransforms(I))
    return Common;

  {
    const APInt *Y;
    // X % -Y -> X % Y
    if (match(Op1, m_APInt(Y)) && Y->isNegative() && !Y->isMinSignedValue()) {
      Worklist.AddValue(I.getOperand(1));
      I.setOperand(1, ConstantInt::get(I.getType(), -*Y));
      return &I;
    }
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  if (I.getType()->isIntegerTy()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask, 0, &I) &&
        MaskedValueIsZero(Op0, Mask, 0, &I)) {
      // X srem Y -> X urem Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateURem(Op0, Op1, I.getName());
    }
  }

  // If it's a constant vector, flip any negative values positive.
  if (isa<ConstantVector>(Op1) || isa<ConstantDataVector>(Op1)) {
    Constant *C = cast<Constant>(Op1);
    unsigned VWidth = C->getType()->getVectorNumElements();

    bool hasNegative = false;
    bool hasMissing = false;
    for (unsigned i = 0; i != VWidth; ++i) {
      Constant *Elt = C->getAggregateElement(i);
      if (!Elt) {
        hasMissing = true;
        break;
      }

      if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elt))
        if (RHS->isNegative())
          hasNegative = true;
    }

    if (hasNegative && !hasMissing) {
      SmallVector<Constant *, 16> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        Elts[i] = C->getAggregateElement(i);  // Handle undef, etc.
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elts[i])) {
          if (RHS->isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != C) {  // Don't loop on -MININT
        Worklist.AddValue(I.getOperand(1));
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyFRemInst(Op0, Op1, I.getFastMathFlags(),
                                  DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  return nullptr;
}
