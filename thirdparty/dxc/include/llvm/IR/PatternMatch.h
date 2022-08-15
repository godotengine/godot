//===- PatternMatch.h - Match on the LLVM IR --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the LLVM IR.  The power of these routines is
// that it allows you to write concise patterns that are expressive and easy to
// understand.  The other major advantage of this is that it allows you to
// trivially capture/bind elements in the pattern to variables.  For example,
// you can do something like this:
//
//  Value *Exp = ...
//  Value *X, *Y;  ConstantInt *C1, *C2;      // (X & C1) | (Y & C2)
//  if (match(Exp, m_Or(m_And(m_Value(X), m_ConstantInt(C1)),
//                      m_And(m_Value(Y), m_ConstantInt(C2))))) {
//    ... Pattern is matched and variables are bound ...
//  }
//
// This is primarily useful to things like the instruction combiner, but can
// also be useful for static analysis tools or code generators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PATTERNMATCH_H
#define LLVM_IR_PATTERNMATCH_H

#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"

namespace llvm {
namespace PatternMatch {

template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern &>(P).match(V);
}

template <typename SubPattern_t> struct OneUse_match {
  SubPattern_t SubPattern;

  OneUse_match(const SubPattern_t &SP) : SubPattern(SP) {}

  template <typename OpTy> bool match(OpTy *V) {
    return V->hasOneUse() && SubPattern.match(V);
  }
};

template <typename T> inline OneUse_match<T> m_OneUse(const T &SubPattern) {
  return SubPattern;
}

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) { return isa<Class>(V); }
};

/// \brief Match an arbitrary value and ignore it.
inline class_match<Value> m_Value() { return class_match<Value>(); }

/// \brief Match an arbitrary binary operation and ignore it.
inline class_match<BinaryOperator> m_BinOp() {
  return class_match<BinaryOperator>();
}

/// \brief Matches any compare instruction and ignore it.
inline class_match<CmpInst> m_Cmp() { return class_match<CmpInst>(); }

/// \brief Match an arbitrary ConstantInt and ignore it.
inline class_match<ConstantInt> m_ConstantInt() {
  return class_match<ConstantInt>();
}

/// \brief Match an arbitrary undef constant.
inline class_match<UndefValue> m_Undef() { return class_match<UndefValue>(); }

/// \brief Match an arbitrary Constant and ignore it.
inline class_match<Constant> m_Constant() { return class_match<Constant>(); }

/// Matching combinators
template <typename LTy, typename RTy> struct match_combine_or {
  LTy L;
  RTy R;

  match_combine_or(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      return true;
    if (R.match(V))
      return true;
    return false;
  }
};

template <typename LTy, typename RTy> struct match_combine_and {
  LTy L;
  RTy R;

  match_combine_and(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      if (R.match(V))
        return true;
    return false;
  }
};

/// Combine two pattern matchers matching L || R
template <typename LTy, typename RTy>
inline match_combine_or<LTy, RTy> m_CombineOr(const LTy &L, const RTy &R) {
  return match_combine_or<LTy, RTy>(L, R);
}

/// Combine two pattern matchers matching L && R
template <typename LTy, typename RTy>
inline match_combine_and<LTy, RTy> m_CombineAnd(const LTy &L, const RTy &R) {
  return match_combine_and<LTy, RTy>(L, R);
}

struct match_zero {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *C = dyn_cast<Constant>(V))
      return C->isNullValue();
    return false;
  }
};

/// \brief Match an arbitrary zero/null constant.  This includes
/// zero_initializer for vectors and ConstantPointerNull for pointers.
inline match_zero m_Zero() { return match_zero(); }

struct match_neg_zero {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *C = dyn_cast<Constant>(V))
      return C->isNegativeZeroValue();
    return false;
  }
};

/// \brief Match an arbitrary zero/null constant.  This includes
/// zero_initializer for vectors and ConstantPointerNull for pointers. For
/// floating point constants, this will match negative zero but not positive
/// zero
inline match_neg_zero m_NegZero() { return match_neg_zero(); }

/// \brief - Match an arbitrary zero/null constant.  This includes
/// zero_initializer for vectors and ConstantPointerNull for pointers. For
/// floating point constants, this will match negative zero and positive zero
inline match_combine_or<match_zero, match_neg_zero> m_AnyZero() {
  return m_CombineOr(m_Zero(), m_NegZero());
}

struct apint_match {
  const APInt *&Res;
  apint_match(const APInt *&R) : Res(R) {}
  template <typename ITy> bool match(ITy *V) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      Res = &CI->getValue();
      return true;
    }
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue())) {
          Res = &CI->getValue();
          return true;
        }
    return false;
  }
};

/// \brief Match a ConstantInt or splatted ConstantVector, binding the
/// specified pointer to the contained APInt.
inline apint_match m_APInt(const APInt *&Res) { return Res; }

template <int64_t Val> struct constantint_match {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *CI = dyn_cast<ConstantInt>(V)) {
      const APInt &CIV = CI->getValue();
      if (Val >= 0)
        return CIV == static_cast<uint64_t>(Val);
      // If Val is negative, and CI is shorter than it, truncate to the right
      // number of bits.  If it is larger, then we have to sign extend.  Just
      // compare their negated values.
      return -CIV == -Val;
    }
    return false;
  }
};

/// \brief Match a ConstantInt with a specific value.
template <int64_t Val> inline constantint_match<Val> m_ConstantInt() {
  return constantint_match<Val>();
}

/// \brief This helper class is used to match scalar and vector constants that
/// satisfy a specified predicate.
template <typename Predicate> struct cst_pred_ty : public Predicate {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *CI = dyn_cast<ConstantInt>(V))
      return this->isValue(CI->getValue());
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (const auto *CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue()))
          return this->isValue(CI->getValue());
    return false;
  }
};

/// \brief This helper class is used to match scalar and vector constants that
/// satisfy a specified predicate, and bind them to an APInt.
template <typename Predicate> struct api_pred_ty : public Predicate {
  const APInt *&Res;
  api_pred_ty(const APInt *&R) : Res(R) {}
  template <typename ITy> bool match(ITy *V) {
    if (const auto *CI = dyn_cast<ConstantInt>(V))
      if (this->isValue(CI->getValue())) {
        Res = &CI->getValue();
        return true;
      }
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue()))
          if (this->isValue(CI->getValue())) {
            Res = &CI->getValue();
            return true;
          }

    return false;
  }
};

struct is_one {
  bool isValue(const APInt &C) { return C == 1; }
};

/// \brief Match an integer 1 or a vector with all elements equal to 1.
inline cst_pred_ty<is_one> m_One() { return cst_pred_ty<is_one>(); }
inline api_pred_ty<is_one> m_One(const APInt *&V) { return V; }

struct is_all_ones {
  bool isValue(const APInt &C) { return C.isAllOnesValue(); }
};

/// \brief Match an integer or vector with all bits set to true.
inline cst_pred_ty<is_all_ones> m_AllOnes() {
  return cst_pred_ty<is_all_ones>();
}
inline api_pred_ty<is_all_ones> m_AllOnes(const APInt *&V) { return V; }

struct is_sign_bit {
  bool isValue(const APInt &C) { return C.isSignBit(); }
};

/// \brief Match an integer or vector with only the sign bit(s) set.
inline cst_pred_ty<is_sign_bit> m_SignBit() {
  return cst_pred_ty<is_sign_bit>();
}
inline api_pred_ty<is_sign_bit> m_SignBit(const APInt *&V) { return V; }

struct is_power2 {
  bool isValue(const APInt &C) { return C.isPowerOf2(); }
};

/// \brief Match an integer or vector power of 2.
inline cst_pred_ty<is_power2> m_Power2() { return cst_pred_ty<is_power2>(); }
inline api_pred_ty<is_power2> m_Power2(const APInt *&V) { return V; }

struct is_maxsignedvalue {
  bool isValue(const APInt &C) { return C.isMaxSignedValue(); }
};

inline cst_pred_ty<is_maxsignedvalue> m_MaxSignedValue() { return cst_pred_ty<is_maxsignedvalue>(); }
inline api_pred_ty<is_maxsignedvalue> m_MaxSignedValue(const APInt *&V) { return V; }

template <typename Class> struct bind_ty {
  Class *&VR;
  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// \brief Match a value, capturing it if we match.
inline bind_ty<Value> m_Value(Value *&V) { return V; }

/// \brief Match an instruction, capturing it if we match.
inline bind_ty<Instruction> m_Instruction(Instruction *&I) { return I; }

/// \brief Match a binary operator, capturing it if we match.
inline bind_ty<BinaryOperator> m_BinOp(BinaryOperator *&I) { return I; }

/// \brief Match a ConstantInt, capturing the value if we match.
inline bind_ty<ConstantInt> m_ConstantInt(ConstantInt *&CI) { return CI; }

/// \brief Match a Constant, capturing the value if we match.
inline bind_ty<Constant> m_Constant(Constant *&C) { return C; }

/// \brief Match a ConstantFP, capturing the value if we match.
inline bind_ty<ConstantFP> m_ConstantFP(ConstantFP *&C) { return C; }

/// \brief Match a specified Value*.
struct specificval_ty {
  const Value *Val;
  specificval_ty(const Value *V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) { return V == Val; }
};

/// \brief Match if we have a specific specified value.
inline specificval_ty m_Specific(const Value *V) { return V; }

/// \brief Match a specified floating point value or vector of all elements of
/// that value.
struct specific_fpval {
  double Val;
  specific_fpval(double V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (const auto *CFP = dyn_cast<ConstantFP>(V))
      return CFP->isExactlyValue(Val);
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CFP = dyn_cast_or_null<ConstantFP>(C->getSplatValue()))
          return CFP->isExactlyValue(Val);
    return false;
  }
};

/// \brief Match a specific floating point value or vector with all elements
/// equal to the value.
inline specific_fpval m_SpecificFP(double V) { return specific_fpval(V); }

/// \brief Match a float 1.0 or vector with all elements equal to 1.0.
inline specific_fpval m_FPOne() { return m_SpecificFP(1.0); }

struct bind_const_intval_ty {
  uint64_t &VR;
  bind_const_intval_ty(uint64_t &V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (const auto *CV = dyn_cast<ConstantInt>(V))
      if (CV->getBitWidth() <= 64) {
        VR = CV->getZExtValue();
        return true;
      }
    return false;
  }
};

/// \brief Match a specified integer value or vector of all elements of that
// value.
struct specific_intval {
  uint64_t Val;
  specific_intval(uint64_t V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) {
    const auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI && V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue());

    if (CI && CI->getBitWidth() <= 64)
      return CI->getZExtValue() == Val;

    return false;
  }
};

/// \brief Match a specific integer value or vector with all elements equal to
/// the value.
inline specific_intval m_SpecificInt(uint64_t V) { return specific_intval(V); }

/// \brief Match a ConstantInt and bind to its value.  This does not match
/// ConstantInts wider than 64-bits.
inline bind_const_intval_ty m_ConstantInt(uint64_t &V) { return V; }

//===----------------------------------------------------------------------===//
// Matcher for any binary operator.
//
template <typename LHS_t, typename RHS_t> struct AnyBinaryOp_match {
  LHS_t L;
  RHS_t R;

  AnyBinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<BinaryOperator>(V))
      return L.match(I->getOperand(0)) && R.match(I->getOperand(1));
    return false;
  }
};

template <typename LHS, typename RHS>
inline AnyBinaryOp_match<LHS, RHS> m_BinOp(const LHS &L, const RHS &R) {
  return AnyBinaryOp_match<LHS, RHS>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for specific binary operators.
//

template <typename LHS_t, typename RHS_t, unsigned Opcode>
struct BinaryOp_match {
  LHS_t L;
  RHS_t R;

  BinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<BinaryOperator>(V);
      return L.match(I->getOperand(0)) && R.match(I->getOperand(1));
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add> m_Add(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FAdd> m_FAdd(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FAdd>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Sub> m_Sub(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Sub>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FSub> m_FSub(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FSub>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul> m_Mul(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FMul> m_FMul(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FMul>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::UDiv> m_UDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::UDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SDiv> m_SDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FDiv> m_FDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::URem> m_URem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::URem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SRem> m_SRem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SRem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FRem> m_FRem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FRem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And> m_And(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Or> m_Or(const LHS &L,
                                                      const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor> m_Xor(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shl> m_Shl(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shl>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::LShr> m_LShr(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::LShr>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::AShr> m_AShr(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::AShr>(L, R);
}

template <typename LHS_t, typename RHS_t, unsigned Opcode,
          unsigned WrapFlags = 0>
struct OverflowingBinaryOp_match {
  LHS_t L;
  RHS_t R;

  OverflowingBinaryOp_match(const LHS_t &LHS, const RHS_t &RHS)
      : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *Op = dyn_cast<OverflowingBinaryOperator>(V)) {
      if (Op->getOpcode() != Opcode)
        return false;
      if (WrapFlags & OverflowingBinaryOperator::NoUnsignedWrap &&
          !Op->hasNoUnsignedWrap())
        return false;
      if (WrapFlags & OverflowingBinaryOperator::NoSignedWrap &&
          !Op->hasNoSignedWrap())
        return false;
      return L.match(Op->getOperand(0)) && R.match(Op->getOperand(1));
    }
    return false;
  }
};

template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWAdd(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWSub(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWMul(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWShl(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}

template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWAdd(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWSub(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWMul(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWShl(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches two different binary ops.
//
template <typename LHS_t, typename RHS_t, unsigned Opc1, unsigned Opc2>
struct BinOp2_match {
  LHS_t L;
  RHS_t R;

  BinOp2_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opc1 ||
        V->getValueID() == Value::InstructionVal + Opc2) {
      auto *I = cast<BinaryOperator>(V);
      return L.match(I->getOperand(0)) && R.match(I->getOperand(1));
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      return (CE->getOpcode() == Opc1 || CE->getOpcode() == Opc2) &&
             L.match(CE->getOperand(0)) && R.match(CE->getOperand(1));
    return false;
  }
};

/// \brief Matches LShr or AShr.
template <typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::AShr>
m_Shr(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::AShr>(L, R);
}

/// \brief Matches LShr or Shl.
template <typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::Shl>
m_LogicalShift(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::LShr, Instruction::Shl>(L, R);
}

/// \brief Matches UDiv and SDiv.
template <typename LHS, typename RHS>
inline BinOp2_match<LHS, RHS, Instruction::SDiv, Instruction::UDiv>
m_IDiv(const LHS &L, const RHS &R) {
  return BinOp2_match<LHS, RHS, Instruction::SDiv, Instruction::UDiv>(L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches exact binary ops.
//
template <typename SubPattern_t> struct Exact_match {
  SubPattern_t SubPattern;

  Exact_match(const SubPattern_t &SP) : SubPattern(SP) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (PossiblyExactOperator *PEO = dyn_cast<PossiblyExactOperator>(V))
      return PEO->isExact() && SubPattern.match(V);
    return false;
  }
};

template <typename T> inline Exact_match<T> m_Exact(const T &SubPattern) {
  return SubPattern;
}

//===----------------------------------------------------------------------===//
// Matchers for CmpInst classes
//

template <typename LHS_t, typename RHS_t, typename Class, typename PredicateTy>
struct CmpClass_match {
  PredicateTy &Predicate;
  LHS_t L;
  RHS_t R;

  CmpClass_match(PredicateTy &Pred, const LHS_t &LHS, const RHS_t &RHS)
      : Predicate(Pred), L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (Class *I = dyn_cast<Class>(V))
      if (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) {
        Predicate = I->getPredicate();
        return true;
      }
    return false;
  }
};

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, CmpInst, CmpInst::Predicate>
m_Cmp(CmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, CmpInst, CmpInst::Predicate>(Pred, L, R);
}

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate>
m_ICmp(ICmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate>(Pred, L, R);
}

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, FCmpInst, FCmpInst::Predicate>
m_FCmp(FCmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, FCmpInst, FCmpInst::Predicate>(Pred, L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for SelectInst classes
//

template <typename Cond_t, typename LHS_t, typename RHS_t>
struct SelectClass_match {
  Cond_t C;
  LHS_t L;
  RHS_t R;

  SelectClass_match(const Cond_t &Cond, const LHS_t &LHS, const RHS_t &RHS)
      : C(Cond), L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<SelectInst>(V))
      return C.match(I->getOperand(0)) && L.match(I->getOperand(1)) &&
             R.match(I->getOperand(2));
    return false;
  }
};

template <typename Cond, typename LHS, typename RHS>
inline SelectClass_match<Cond, LHS, RHS> m_Select(const Cond &C, const LHS &L,
                                                  const RHS &R) {
  return SelectClass_match<Cond, LHS, RHS>(C, L, R);
}

/// \brief This matches a select of two constants, e.g.:
/// m_SelectCst<-1, 0>(m_Value(V))
template <int64_t L, int64_t R, typename Cond>
inline SelectClass_match<Cond, constantint_match<L>, constantint_match<R>>
m_SelectCst(const Cond &C) {
  return m_Select(C, m_ConstantInt<L>(), m_ConstantInt<R>());
}

//===----------------------------------------------------------------------===//
// Matchers for CastInst classes
//

template <typename Op_t, unsigned Opcode> struct CastClass_match {
  Op_t Op;

  CastClass_match(const Op_t &OpMatch) : Op(OpMatch) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Operator>(V))
      return O->getOpcode() == Opcode && Op.match(O->getOperand(0));
    return false;
  }
};

/// \brief Matches BitCast.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::BitCast> m_BitCast(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::BitCast>(Op);
}

/// \brief Matches PtrToInt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::PtrToInt> m_PtrToInt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::PtrToInt>(Op);
}

/// \brief Matches Trunc.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::Trunc> m_Trunc(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::Trunc>(Op);
}

/// \brief Matches SExt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::SExt> m_SExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::SExt>(Op);
}

/// \brief Matches ZExt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::ZExt> m_ZExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::ZExt>(Op);
}

/// \brief Matches UIToFP.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::UIToFP> m_UIToFP(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::UIToFP>(Op);
}

/// \brief Matches SIToFP.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::SIToFP> m_SIToFP(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::SIToFP>(Op);
}

//===----------------------------------------------------------------------===//
// Matchers for unary operators
//

template <typename LHS_t> struct not_match {
  LHS_t L;

  not_match(const LHS_t &LHS) : L(LHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::Xor)
        return matchIfNot(O->getOperand(0), O->getOperand(1));
    return false;
  }

private:
  bool matchIfNot(Value *LHS, Value *RHS) {
    return (isa<ConstantInt>(RHS) || isa<ConstantDataVector>(RHS) ||
            // FIXME: Remove CV.
            isa<ConstantVector>(RHS)) &&
           cast<Constant>(RHS)->isAllOnesValue() && L.match(LHS);
  }
};

template <typename LHS> inline not_match<LHS> m_Not(const LHS &L) { return L; }

template <typename LHS_t> struct neg_match {
  LHS_t L;

  neg_match(const LHS_t &LHS) : L(LHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::Sub)
        return matchIfNeg(O->getOperand(0), O->getOperand(1));
    return false;
  }

private:
  bool matchIfNeg(Value *LHS, Value *RHS) {
    return ((isa<ConstantInt>(LHS) && cast<ConstantInt>(LHS)->isZero()) ||
            isa<ConstantAggregateZero>(LHS)) &&
           L.match(RHS);
  }
};

/// \brief Match an integer negate.
template <typename LHS> inline neg_match<LHS> m_Neg(const LHS &L) { return L; }

template <typename LHS_t> struct fneg_match {
  LHS_t L;

  fneg_match(const LHS_t &LHS) : L(LHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Operator>(V))
      if (O->getOpcode() == Instruction::FSub)
        return matchIfFNeg(O->getOperand(0), O->getOperand(1));
    return false;
  }

private:
  bool matchIfFNeg(Value *LHS, Value *RHS) {
    if (const auto *C = dyn_cast<ConstantFP>(LHS))
      return C->isNegativeZeroValue() && L.match(RHS);
    return false;
  }
};

/// \brief Match a floating point negate.
template <typename LHS> inline fneg_match<LHS> m_FNeg(const LHS &L) {
  return L;
}

//===----------------------------------------------------------------------===//
// Matchers for control flow.
//

struct br_match {
  BasicBlock *&Succ;
  br_match(BasicBlock *&Succ) : Succ(Succ) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *BI = dyn_cast<BranchInst>(V))
      if (BI->isUnconditional()) {
        Succ = BI->getSuccessor(0);
        return true;
      }
    return false;
  }
};

inline br_match m_UnconditionalBr(BasicBlock *&Succ) { return br_match(Succ); }

template <typename Cond_t> struct brc_match {
  Cond_t Cond;
  BasicBlock *&T, *&F;
  brc_match(const Cond_t &C, BasicBlock *&t, BasicBlock *&f)
      : Cond(C), T(t), F(f) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *BI = dyn_cast<BranchInst>(V))
      if (BI->isConditional() && Cond.match(BI->getCondition())) {
        T = BI->getSuccessor(0);
        F = BI->getSuccessor(1);
        return true;
      }
    return false;
  }
};

template <typename Cond_t>
inline brc_match<Cond_t> m_Br(const Cond_t &C, BasicBlock *&T, BasicBlock *&F) {
  return brc_match<Cond_t>(C, T, F);
}

//===----------------------------------------------------------------------===//
// Matchers for max/min idioms, eg: "select (sgt x, y), x, y" -> smax(x,y).
//

template <typename CmpInst_t, typename LHS_t, typename RHS_t, typename Pred_t>
struct MaxMin_match {
  LHS_t L;
  RHS_t R;

  MaxMin_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    // Look for "(x pred y) ? x : y" or "(x pred y) ? y : x".
    auto *SI = dyn_cast<SelectInst>(V);
    if (!SI)
      return false;
    auto *Cmp = dyn_cast<CmpInst_t>(SI->getCondition());
    if (!Cmp)
      return false;
    // At this point we have a select conditioned on a comparison.  Check that
    // it is the values returned by the select that are being compared.
    Value *TrueVal = SI->getTrueValue();
    Value *FalseVal = SI->getFalseValue();
    Value *LHS = Cmp->getOperand(0);
    Value *RHS = Cmp->getOperand(1);
    if ((TrueVal != LHS || FalseVal != RHS) &&
        (TrueVal != RHS || FalseVal != LHS))
      return false;
    typename CmpInst_t::Predicate Pred =
        LHS == TrueVal ? Cmp->getPredicate() : Cmp->getSwappedPredicate();
    // Does "(x pred y) ? x : y" represent the desired max/min operation?
    if (!Pred_t::match(Pred))
      return false;
    // It does!  Bind the operands.
    return L.match(LHS) && R.match(RHS);
  }
};

/// \brief Helper class for identifying signed max predicates.
struct smax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SGT || Pred == CmpInst::ICMP_SGE;
  }
};

/// \brief Helper class for identifying signed min predicates.
struct smin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SLT || Pred == CmpInst::ICMP_SLE;
  }
};

/// \brief Helper class for identifying unsigned max predicates.
struct umax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_UGT || Pred == CmpInst::ICMP_UGE;
  }
};

/// \brief Helper class for identifying unsigned min predicates.
struct umin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_ULT || Pred == CmpInst::ICMP_ULE;
  }
};

/// \brief Helper class for identifying ordered max predicates.
struct ofmax_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_OGT || Pred == CmpInst::FCMP_OGE;
  }
};

/// \brief Helper class for identifying ordered min predicates.
struct ofmin_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_OLT || Pred == CmpInst::FCMP_OLE;
  }
};

/// \brief Helper class for identifying unordered max predicates.
struct ufmax_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_UGT || Pred == CmpInst::FCMP_UGE;
  }
};

/// \brief Helper class for identifying unordered min predicates.
struct ufmin_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_ULT || Pred == CmpInst::FCMP_ULE;
  }
};

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty> m_SMax(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty> m_SMin(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty> m_UMax(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty> m_UMin(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty>(L, R);
}

/// \brief Match an 'ordered' floating point maximum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'maximum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ogt/ge, L, R), L, R) semantics matched by this predicate.
///
///                         max(L, R)  iff L and R are not NaN
///  m_OrdFMax(L, R) =      R          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ofmax_pred_ty> m_OrdFMax(const LHS &L,
                                                                 const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ofmax_pred_ty>(L, R);
}

/// \brief Match an 'ordered' floating point minimum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'minimum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(olt/le, L, R), L, R) semantics matched by this predicate.
///
///                         max(L, R)  iff L and R are not NaN
///  m_OrdFMin(L, R) =      R          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ofmin_pred_ty> m_OrdFMin(const LHS &L,
                                                                 const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ofmin_pred_ty>(L, R);
}

/// \brief Match an 'unordered' floating point maximum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'maximum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ugt/ge, L, R), L, R) semantics matched by this predicate.
///
///                         max(L, R)  iff L and R are not NaN
///  m_UnordFMin(L, R) =    L          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ufmax_pred_ty>
m_UnordFMax(const LHS &L, const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ufmax_pred_ty>(L, R);
}
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// Matchers for overflow check patterns: e.g. (a + b) u< a
//

template <typename LHS_t, typename RHS_t, typename Sum_t>
struct UAddWithOverflow_match {
  LHS_t L;
  RHS_t R;
  Sum_t S;

  UAddWithOverflow_match(const LHS_t &L, const RHS_t &R, const Sum_t &S)
      : L(L), R(R), S(S) {}

  template <typename OpTy> bool match(OpTy *V) {
    Value *ICmpLHS, *ICmpRHS;
    ICmpInst::Predicate Pred;
    if (!m_ICmp(Pred, m_Value(ICmpLHS), m_Value(ICmpRHS)).match(V))
      return false;

    Value *AddLHS, *AddRHS;
    auto AddExpr = m_Add(m_Value(AddLHS), m_Value(AddRHS));

    // (a + b) u< a, (a + b) u< b
    if (Pred == ICmpInst::ICMP_ULT)
      if (AddExpr.match(ICmpLHS) && (ICmpRHS == AddLHS || ICmpRHS == AddRHS))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpLHS);

    // a >u (a + b), b >u (a + b)
    if (Pred == ICmpInst::ICMP_UGT)
      if (AddExpr.match(ICmpRHS) && (ICmpLHS == AddLHS || ICmpLHS == AddRHS))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpRHS);

    return false;
  }
};

/// \brief Match an icmp instruction checking for unsigned overflow on addition.
///
/// S is matched to the addition whose result is being checked for overflow, and
/// L and R are matched to the LHS and RHS of S.
template <typename LHS_t, typename RHS_t, typename Sum_t>
UAddWithOverflow_match<LHS_t, RHS_t, Sum_t>
m_UAddWithOverflow(const LHS_t &L, const RHS_t &R, const Sum_t &S) {
  return UAddWithOverflow_match<LHS_t, RHS_t, Sum_t>(L, R, S);
}

/// \brief Match an 'unordered' floating point minimum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'minimum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ult/le, L, R), L, R) semantics matched by this predicate.
///
///                          max(L, R)  iff L and R are not NaN
///  m_UnordFMin(L, R) =     L          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ufmin_pred_ty>
m_UnordFMin(const LHS &L, const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ufmin_pred_ty>(L, R);
}

template <typename Opnd_t> struct Argument_match {
  unsigned OpI;
  Opnd_t Val;
  Argument_match(unsigned OpIdx, const Opnd_t &V) : OpI(OpIdx), Val(V) {}

  template <typename OpTy> bool match(OpTy *V) {
    CallSite CS(V);
    return CS.isCall() && Val.match(CS.getArgument(OpI));
  }
};

/// \brief Match an argument.
template <unsigned OpI, typename Opnd_t>
inline Argument_match<Opnd_t> m_Argument(const Opnd_t &Op) {
  return Argument_match<Opnd_t>(OpI, Op);
}

/// \brief Intrinsic matchers.
struct IntrinsicID_match {
  unsigned ID;
  IntrinsicID_match(Intrinsic::ID IntrID) : ID(IntrID) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (const auto *CI = dyn_cast<CallInst>(V))
      if (const auto *F = CI->getCalledFunction())
        return F->getIntrinsicID() == ID;
    return false;
  }
};

/// Intrinsic matches are combinations of ID matchers, and argument
/// matchers. Higher arity matcher are defined recursively in terms of and-ing
/// them with lower arity matchers. Here's some convenient typedefs for up to
/// several arguments, and more can be added as needed
template <typename T0 = void, typename T1 = void, typename T2 = void,
          typename T3 = void, typename T4 = void, typename T5 = void,
          typename T6 = void, typename T7 = void, typename T8 = void,
          typename T9 = void, typename T10 = void>
struct m_Intrinsic_Ty;
template <typename T0> struct m_Intrinsic_Ty<T0> {
  typedef match_combine_and<IntrinsicID_match, Argument_match<T0>> Ty;
};
template <typename T0, typename T1> struct m_Intrinsic_Ty<T0, T1> {
  typedef match_combine_and<typename m_Intrinsic_Ty<T0>::Ty, Argument_match<T1>>
      Ty;
};
template <typename T0, typename T1, typename T2>
struct m_Intrinsic_Ty<T0, T1, T2> {
  typedef match_combine_and<typename m_Intrinsic_Ty<T0, T1>::Ty,
                            Argument_match<T2>> Ty;
};
template <typename T0, typename T1, typename T2, typename T3>
struct m_Intrinsic_Ty<T0, T1, T2, T3> {
  typedef match_combine_and<typename m_Intrinsic_Ty<T0, T1, T2>::Ty,
                            Argument_match<T3>> Ty;
};

/// \brief Match intrinsic calls like this:
/// m_Intrinsic<Intrinsic::fabs>(m_Value(X))
template <Intrinsic::ID IntrID> inline IntrinsicID_match m_Intrinsic() {
  return IntrinsicID_match(IntrID);
}

template <Intrinsic::ID IntrID, typename T0>
inline typename m_Intrinsic_Ty<T0>::Ty m_Intrinsic(const T0 &Op0) {
  return m_CombineAnd(m_Intrinsic<IntrID>(), m_Argument<0>(Op0));
}

template <Intrinsic::ID IntrID, typename T0, typename T1>
inline typename m_Intrinsic_Ty<T0, T1>::Ty m_Intrinsic(const T0 &Op0,
                                                       const T1 &Op1) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0), m_Argument<1>(Op1));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2>
inline typename m_Intrinsic_Ty<T0, T1, T2>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1), m_Argument<2>(Op2));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2,
          typename T3>
inline typename m_Intrinsic_Ty<T0, T1, T2, T3>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2, const T3 &Op3) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1, Op2), m_Argument<3>(Op3));
}

// Helper intrinsic matching specializations.
template <typename Opnd0>
inline typename m_Intrinsic_Ty<Opnd0>::Ty m_BSwap(const Opnd0 &Op0) {
  return m_Intrinsic<Intrinsic::bswap>(Op0);
}

template <typename Opnd0, typename Opnd1>
inline typename m_Intrinsic_Ty<Opnd0, Opnd1>::Ty m_FMin(const Opnd0 &Op0,
                                                        const Opnd1 &Op1) {
  return m_Intrinsic<Intrinsic::minnum>(Op0, Op1);
}

template <typename Opnd0, typename Opnd1>
inline typename m_Intrinsic_Ty<Opnd0, Opnd1>::Ty m_FMax(const Opnd0 &Op0,
                                                        const Opnd1 &Op1) {
  return m_Intrinsic<Intrinsic::maxnum>(Op0, Op1);
}

} // end namespace PatternMatch
} // end namespace llvm

#endif
