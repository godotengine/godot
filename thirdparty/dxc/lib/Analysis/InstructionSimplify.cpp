//===- InstructionSimplify.cpp - Fold instruction operands ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  All operands are assumed to have already been
// simplified: This is usually true and assuming it simplifies the logic (if
// they have not been simplified then results are correct but maybe suboptimal).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include <algorithm>

#include "llvm/Analysis/DxilSimplify.h" // HLSL Change - simplify dxil call.

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "instsimplify"

enum { RecursionLimit = 3 };

STATISTIC(NumExpand,  "Number of expansions");
STATISTIC(NumReassoc, "Number of reassociations");

namespace {
struct Query {
  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  const DominatorTree *DT;
  AssumptionCache *AC;
  const Instruction *CxtI;

  Query(const DataLayout &DL, const TargetLibraryInfo *tli,
        const DominatorTree *dt, AssumptionCache *ac = nullptr,
        const Instruction *cxti = nullptr)
      : DL(DL), TLI(tli), DT(dt), AC(ac), CxtI(cxti) {}
};
} // end anonymous namespace

static Value *SimplifyAndInst(Value *, Value *, const Query &, unsigned);
static Value *SimplifyBinOp(unsigned, Value *, Value *, const Query &,
                            unsigned);
static Value *SimplifyFPBinOp(unsigned, Value *, Value *, const FastMathFlags &,
                              const Query &, unsigned);
static Value *SimplifyCmpInst(unsigned, Value *, Value *, const Query &,
                              unsigned);
static Value *SimplifyOrInst(Value *, Value *, const Query &, unsigned);
static Value *SimplifyXorInst(Value *, Value *, const Query &, unsigned);
static Value *SimplifyTruncInst(Value *, Type *, const Query &, unsigned);

/// getFalse - For a boolean type, or a vector of boolean type, return false, or
/// a vector with every element false, as appropriate for the type.
static Constant *getFalse(Type *Ty) {
  assert(Ty->getScalarType()->isIntegerTy(1) &&
         "Expected i1 type or a vector of i1!");
  return Constant::getNullValue(Ty);
}

/// getTrue - For a boolean type, or a vector of boolean type, return true, or
/// a vector with every element true, as appropriate for the type.
static Constant *getTrue(Type *Ty) {
  assert(Ty->getScalarType()->isIntegerTy(1) &&
         "Expected i1 type or a vector of i1!");
  return Constant::getAllOnesValue(Ty);
}

/// isSameCompare - Is V equivalent to the comparison "LHS Pred RHS"?
static bool isSameCompare(Value *V, CmpInst::Predicate Pred, Value *LHS,
                          Value *RHS) {
  CmpInst *Cmp = dyn_cast<CmpInst>(V);
  if (!Cmp)
    return false;
  CmpInst::Predicate CPred = Cmp->getPredicate();
  Value *CLHS = Cmp->getOperand(0), *CRHS = Cmp->getOperand(1);
  if (CPred == Pred && CLHS == LHS && CRHS == RHS)
    return true;
  return CPred == CmpInst::getSwappedPredicate(Pred) && CLHS == RHS &&
    CRHS == LHS;
}

/// ValueDominatesPHI - Does the given value dominate the specified phi node?
static bool ValueDominatesPHI(Value *V, PHINode *P, const DominatorTree *DT) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    // Arguments and constants dominate all instructions.
    return true;

  // If we are processing instructions (and/or basic blocks) that have not been
  // fully added to a function, the parent nodes may still be null. Simply
  // return the conservative answer in these cases.
  if (!I->getParent() || !P->getParent() || !I->getParent()->getParent())
    return false;

  // If we have a DominatorTree then do a precise test.
  if (DT) {
    if (!DT->isReachableFromEntry(P->getParent()))
      return true;
    if (!DT->isReachableFromEntry(I->getParent()))
      return false;
    return DT->dominates(I, P);
  }

  // Otherwise, if the instruction is in the entry block, and is not an invoke,
  // then it obviously dominates all phi nodes.
  if (I->getParent() == &I->getParent()->getParent()->getEntryBlock() &&
      !isa<InvokeInst>(I))
    return true;

  return false;
}

/// ExpandBinOp - Simplify "A op (B op' C)" by distributing op over op', turning
/// it into "(A op B) op' (A op C)".  Here "op" is given by Opcode and "op'" is
/// given by OpcodeToExpand, while "A" corresponds to LHS and "B op' C" to RHS.
/// Also performs the transform "(A op' B) op C" -> "(A op C) op' (B op C)".
/// Returns the simplified value, or null if no simplification was performed.
static Value *ExpandBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                          unsigned OpcToExpand, const Query &Q,
                          unsigned MaxRecurse) {
  Instruction::BinaryOps OpcodeToExpand = (Instruction::BinaryOps)OpcToExpand;
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  // Check whether the expression has the form "(A op' B) op C".
  if (BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS))
    if (Op0->getOpcode() == OpcodeToExpand) {
      // It does!  Try turning it into "(A op C) op' (B op C)".
      Value *A = Op0->getOperand(0), *B = Op0->getOperand(1), *C = RHS;
      // Do "A op C" and "B op C" both simplify?
      if (Value *L = SimplifyBinOp(Opcode, A, C, Q, MaxRecurse))
        if (Value *R = SimplifyBinOp(Opcode, B, C, Q, MaxRecurse)) {
          // They do! Return "L op' R" if it simplifies or is already available.
          // If "L op' R" equals "A op' B" then "L op' R" is just the LHS.
          if ((L == A && R == B) || (Instruction::isCommutative(OpcodeToExpand)
                                     && L == B && R == A)) {
            ++NumExpand;
            return LHS;
          }
          // Otherwise return "L op' R" if it simplifies.
          if (Value *V = SimplifyBinOp(OpcodeToExpand, L, R, Q, MaxRecurse)) {
            ++NumExpand;
            return V;
          }
        }
    }

  // Check whether the expression has the form "A op (B op' C)".
  if (BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS))
    if (Op1->getOpcode() == OpcodeToExpand) {
      // It does!  Try turning it into "(A op B) op' (A op C)".
      Value *A = LHS, *B = Op1->getOperand(0), *C = Op1->getOperand(1);
      // Do "A op B" and "A op C" both simplify?
      if (Value *L = SimplifyBinOp(Opcode, A, B, Q, MaxRecurse))
        if (Value *R = SimplifyBinOp(Opcode, A, C, Q, MaxRecurse)) {
          // They do! Return "L op' R" if it simplifies or is already available.
          // If "L op' R" equals "B op' C" then "L op' R" is just the RHS.
          if ((L == B && R == C) || (Instruction::isCommutative(OpcodeToExpand)
                                     && L == C && R == B)) {
            ++NumExpand;
            return RHS;
          }
          // Otherwise return "L op' R" if it simplifies.
          if (Value *V = SimplifyBinOp(OpcodeToExpand, L, R, Q, MaxRecurse)) {
            ++NumExpand;
            return V;
          }
        }
    }

  return nullptr;
}

/// SimplifyAssociativeBinOp - Generic simplifications for associative binary
/// operations.  Returns the simpler value, or null if none was found.
static Value *SimplifyAssociativeBinOp(unsigned Opc, Value *LHS, Value *RHS,
                                       const Query &Q, unsigned MaxRecurse) {
  Instruction::BinaryOps Opcode = (Instruction::BinaryOps)Opc;
  assert(Instruction::isAssociative(Opcode) && "Not an associative operation!");

  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);

  // Transform: "(A op B) op C" ==> "A op (B op C)" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "B op C" simplify?
    if (Value *V = SimplifyBinOp(Opcode, B, C, Q, MaxRecurse)) {
      // It does!  Return "A op V" if it simplifies or is already available.
      // If V equals B then "A op V" is just the LHS.
      if (V == B) return LHS;
      // Otherwise return "A op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, A, V, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "(A op B) op C" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "A op B" simplify?
    if (Value *V = SimplifyBinOp(Opcode, A, B, Q, MaxRecurse)) {
      // It does!  Return "V op C" if it simplifies or is already available.
      // If V equals B then "V op C" is just the RHS.
      if (V == B) return RHS;
      // Otherwise return "V op C" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, C, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // The remaining transforms require commutativity as well as associativity.
  if (!Instruction::isCommutative(Opcode))
    return nullptr;

  // Transform: "(A op B) op C" ==> "(C op A) op B" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, Q, MaxRecurse)) {
      // It does!  Return "V op B" if it simplifies or is already available.
      // If V equals A then "V op B" is just the LHS.
      if (V == A) return LHS;
      // Otherwise return "V op B" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, B, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "B op (C op A)" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, Q, MaxRecurse)) {
      // It does!  Return "B op V" if it simplifies or is already available.
      // If V equals C then "B op V" is just the RHS.
      if (V == C) return RHS;
      // Otherwise return "B op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, B, V, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  return nullptr;
}

/// ThreadBinOpOverSelect - In the case of a binary operation with a select
/// instruction as an operand, try to simplify the binop by seeing whether
/// evaluating it on both branches of the select results in the same value.
/// Returns the common value if so, otherwise returns null.
static Value *ThreadBinOpOverSelect(unsigned Opcode, Value *LHS, Value *RHS,
                                    const Query &Q, unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  SelectInst *SI;
  if (isa<SelectInst>(LHS)) {
    SI = cast<SelectInst>(LHS);
  } else {
    assert(isa<SelectInst>(RHS) && "No select instruction operand!");
    SI = cast<SelectInst>(RHS);
  }

  // Evaluate the BinOp on the true and false branches of the select.
  Value *TV;
  Value *FV;
  if (SI == LHS) {
    TV = SimplifyBinOp(Opcode, SI->getTrueValue(), RHS, Q, MaxRecurse);
    FV = SimplifyBinOp(Opcode, SI->getFalseValue(), RHS, Q, MaxRecurse);
  } else {
    TV = SimplifyBinOp(Opcode, LHS, SI->getTrueValue(), Q, MaxRecurse);
    FV = SimplifyBinOp(Opcode, LHS, SI->getFalseValue(), Q, MaxRecurse);
  }

  // If they simplified to the same value, then return the common value.
  // If they both failed to simplify then return null.
  if (TV == FV)
    return TV;

  // If one branch simplified to undef, return the other one.
  if (TV && isa<UndefValue>(TV))
    return FV;
  if (FV && isa<UndefValue>(FV))
    return TV;

  // If applying the operation did not change the true and false select values,
  // then the result of the binop is the select itself.
  if (TV == SI->getTrueValue() && FV == SI->getFalseValue())
    return SI;

  // If one branch simplified and the other did not, and the simplified
  // value is equal to the unsimplified one, return the simplified value.
  // For example, select (cond, X, X & Z) & Z -> X & Z.
  if ((FV && !TV) || (TV && !FV)) {
    // Check that the simplified value has the form "X op Y" where "op" is the
    // same as the original operation.
    Instruction *Simplified = dyn_cast<Instruction>(FV ? FV : TV);
    if (Simplified && Simplified->getOpcode() == Opcode) {
      // The value that didn't simplify is "UnsimplifiedLHS op UnsimplifiedRHS".
      // We already know that "op" is the same as for the simplified value.  See
      // if the operands match too.  If so, return the simplified value.
      Value *UnsimplifiedBranch = FV ? SI->getTrueValue() : SI->getFalseValue();
      Value *UnsimplifiedLHS = SI == LHS ? UnsimplifiedBranch : LHS;
      Value *UnsimplifiedRHS = SI == LHS ? RHS : UnsimplifiedBranch;
      if (Simplified->getOperand(0) == UnsimplifiedLHS &&
          Simplified->getOperand(1) == UnsimplifiedRHS)
        return Simplified;
      if (Simplified->isCommutative() &&
          Simplified->getOperand(1) == UnsimplifiedLHS &&
          Simplified->getOperand(0) == UnsimplifiedRHS)
        return Simplified;
    }
  }

  return nullptr;
}

/// ThreadCmpOverSelect - In the case of a comparison with a select instruction,
/// try to simplify the comparison by seeing whether both branches of the select
/// result in the same value.  Returns the common value if so, otherwise returns
/// null.
static Value *ThreadCmpOverSelect(CmpInst::Predicate Pred, Value *LHS,
                                  Value *RHS, const Query &Q,
                                  unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  // Make sure the select is on the LHS.
  if (!isa<SelectInst>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<SelectInst>(LHS) && "Not comparing with a select instruction!");
  SelectInst *SI = cast<SelectInst>(LHS);
  Value *Cond = SI->getCondition();
  Value *TV = SI->getTrueValue();
  Value *FV = SI->getFalseValue();

  // Now that we have "cmp select(Cond, TV, FV), RHS", analyse it.
  // Does "cmp TV, RHS" simplify?
  Value *TCmp = SimplifyCmpInst(Pred, TV, RHS, Q, MaxRecurse);
  if (TCmp == Cond) {
    // It not only simplified, it simplified to the select condition.  Replace
    // it with 'true'.
    TCmp = getTrue(Cond->getType());
  } else if (!TCmp) {
    // It didn't simplify.  However if "cmp TV, RHS" is equal to the select
    // condition then we can replace it with 'true'.  Otherwise give up.
    if (!isSameCompare(Cond, Pred, TV, RHS))
      return nullptr;
    TCmp = getTrue(Cond->getType());
  }

  // Does "cmp FV, RHS" simplify?
  Value *FCmp = SimplifyCmpInst(Pred, FV, RHS, Q, MaxRecurse);
  if (FCmp == Cond) {
    // It not only simplified, it simplified to the select condition.  Replace
    // it with 'false'.
    FCmp = getFalse(Cond->getType());
  } else if (!FCmp) {
    // It didn't simplify.  However if "cmp FV, RHS" is equal to the select
    // condition then we can replace it with 'false'.  Otherwise give up.
    if (!isSameCompare(Cond, Pred, FV, RHS))
      return nullptr;
    FCmp = getFalse(Cond->getType());
  }

  // If both sides simplified to the same value, then use it as the result of
  // the original comparison.
  if (TCmp == FCmp)
    return TCmp;

  // The remaining cases only make sense if the select condition has the same
  // type as the result of the comparison, so bail out if this is not so.
  if (Cond->getType()->isVectorTy() != RHS->getType()->isVectorTy())
    return nullptr;
  // If the false value simplified to false, then the result of the compare
  // is equal to "Cond && TCmp".  This also catches the case when the false
  // value simplified to false and the true value to true, returning "Cond".
  if (match(FCmp, m_Zero()))
    if (Value *V = SimplifyAndInst(Cond, TCmp, Q, MaxRecurse))
      return V;
  // If the true value simplified to true, then the result of the compare
  // is equal to "Cond || FCmp".
  if (match(TCmp, m_One()))
    if (Value *V = SimplifyOrInst(Cond, FCmp, Q, MaxRecurse))
      return V;
  // Finally, if the false value simplified to true and the true value to
  // false, then the result of the compare is equal to "!Cond".
  if (match(FCmp, m_One()) && match(TCmp, m_Zero()))
    if (Value *V =
        SimplifyXorInst(Cond, Constant::getAllOnesValue(Cond->getType()),
                        Q, MaxRecurse))
      return V;

  return nullptr;
}

/// ThreadBinOpOverPHI - In the case of a binary operation with an operand that
/// is a PHI instruction, try to simplify the binop by seeing whether evaluating
/// it on the incoming phi values yields the same result for every value.  If so
/// returns the common value, otherwise returns null.
static Value *ThreadBinOpOverPHI(unsigned Opcode, Value *LHS, Value *RHS,
                                 const Query &Q, unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  PHINode *PI;
  if (isa<PHINode>(LHS)) {
    PI = cast<PHINode>(LHS);
    // Bail out if RHS and the phi may be mutually interdependent due to a loop.
    if (!ValueDominatesPHI(RHS, PI, Q.DT))
      return nullptr;
  } else {
    assert(isa<PHINode>(RHS) && "No PHI instruction operand!");
    PI = cast<PHINode>(RHS);
    // Bail out if LHS and the phi may be mutually interdependent due to a loop.
    if (!ValueDominatesPHI(LHS, PI, Q.DT))
      return nullptr;
  }

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = nullptr;
  for (Value *Incoming : PI->incoming_values()) {
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    Value *V = PI == LHS ?
      SimplifyBinOp(Opcode, Incoming, RHS, Q, MaxRecurse) :
      SimplifyBinOp(Opcode, LHS, Incoming, Q, MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return nullptr;
    CommonValue = V;
  }

  return CommonValue;
}

/// ThreadCmpOverPHI - In the case of a comparison with a PHI instruction, try
/// try to simplify the comparison by seeing whether comparing with all of the
/// incoming phi values yields the same result every time.  If so returns the
/// common result, otherwise returns null.
static Value *ThreadCmpOverPHI(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                               const Query &Q, unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  // Make sure the phi is on the LHS.
  if (!isa<PHINode>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<PHINode>(LHS) && "Not comparing with a phi instruction!");
  PHINode *PI = cast<PHINode>(LHS);

  // Bail out if RHS and the phi may be mutually interdependent due to a loop.
  if (!ValueDominatesPHI(RHS, PI, Q.DT))
    return nullptr;

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = nullptr;
  for (Value *Incoming : PI->incoming_values()) {
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    Value *V = SimplifyCmpInst(Pred, Incoming, RHS, Q, MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return nullptr;
    CommonValue = V;
  }

  return CommonValue;
}

/// SimplifyAddInst - Given operands for an Add, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAddInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const Query &Q, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Add, CLHS->getType(), Ops,
                                      Q.DL, Q.TLI);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X + undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // X + 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X + (Y - X) -> Y
  // (Y - X) + X -> Y
  // Eg: X + -X -> 0
  Value *Y = nullptr;
  if (match(Op1, m_Sub(m_Value(Y), m_Specific(Op0))) ||
      match(Op0, m_Sub(m_Value(Y), m_Specific(Op1))))
    return Y;

  // X + ~X -> -1   since   ~X = -X-1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  /// i1 add -> xor.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Add, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Threading Add over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A + select(cond, B, C)" means evaluating
  // "A+B" and "A+C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A+B" and "A+C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifyAddInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const DataLayout &DL, const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyAddInst(Op0, Op1, isNSW, isNUW, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

/// \brief Compute the base pointer and cumulative constant offsets for V.
///
/// This strips all constant offsets off of V, leaving it the base pointer, and
/// accumulates the total constant offset applied in the returned constant. It
/// returns 0 if V is not a pointer, and returns the constant '0' if there are
/// no constant offsets applied.
///
/// This is very similar to GetPointerBaseWithConstantOffset except it doesn't
/// follow non-inbounds geps. This allows it to remain usable for icmp ult/etc.
/// folding.
static Constant *stripAndComputeConstantOffsets(const DataLayout &DL, Value *&V,
                                                bool AllowNonInbounds = false) {
  assert(V->getType()->getScalarType()->isPointerTy());

  Type *IntPtrTy = DL.getIntPtrType(V->getType())->getScalarType();
  APInt Offset = APInt::getNullValue(IntPtrTy->getIntegerBitWidth());

  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;
  Visited.insert(V);
  do {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      if ((!AllowNonInbounds && !GEP->isInBounds()) ||
          !GEP->accumulateConstantOffset(DL, Offset))
        break;
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->mayBeOverridden())
        break;
      V = GA->getAliasee();
    } else {
      break;
    }
    assert(V->getType()->getScalarType()->isPointerTy() &&
           "Unexpected operand type!");
  } while (Visited.insert(V).second);

  Constant *OffsetIntPtr = ConstantInt::get(IntPtrTy, Offset);
  if (V->getType()->isVectorTy())
    return ConstantVector::getSplat(V->getType()->getVectorNumElements(),
                                    OffsetIntPtr);
  return OffsetIntPtr;
}

/// \brief Compute the constant difference between two pointer values.
/// If the difference is not a constant, returns zero.
static Constant *computePointerDifference(const DataLayout &DL, Value *LHS,
                                          Value *RHS) {
  Constant *LHSOffset = stripAndComputeConstantOffsets(DL, LHS);
  Constant *RHSOffset = stripAndComputeConstantOffsets(DL, RHS);

  // If LHS and RHS are not related via constant offsets to the same base
  // value, there is nothing we can do here.
  if (LHS != RHS)
    return nullptr;

  // Otherwise, the difference of LHS - RHS can be computed as:
  //    LHS - RHS
  //  = (LHSOffset + Base) - (RHSOffset + Base)
  //  = LHSOffset - RHSOffset
  return ConstantExpr::getSub(LHSOffset, RHSOffset);
}

/// SimplifySubInst - Given operands for a Sub, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const Query &Q, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0))
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Sub, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }

  // X - undef -> undef
  // undef - X -> undef
  if (match(Op0, m_Undef()) || match(Op1, m_Undef()))
    return UndefValue::get(Op0->getType());

  // X - 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X - X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // 0 - X -> 0 if the sub is NUW.
  if (isNUW && match(Op0, m_Zero()))
    return Op0;

  // (X + Y) - Z -> X + (Y - Z) or Y + (X - Z) if everything simplifies.
  // For example, (X + Y) - Y -> X; (Y + X) - Y -> X
  Value *X = nullptr, *Y = nullptr, *Z = Op1;
  if (MaxRecurse && match(Op0, m_Add(m_Value(X), m_Value(Y)))) { // (X + Y) - Z
    // See if "V === Y - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Y, Z, Q, MaxRecurse-1))
      // It does!  Now see if "X + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, X, V, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, Q, MaxRecurse-1))
      // It does!  Now see if "Y + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, Y, V, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // X - (Y + Z) -> (X - Y) - Z or (X - Z) - Y if everything simplifies.
  // For example, X - (X + 1) -> -1
  X = Op0;
  if (MaxRecurse && match(Op1, m_Add(m_Value(Y), m_Value(Z)))) { // X - (Y + Z)
    // See if "V === X - Y" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Y, Q, MaxRecurse-1))
      // It does!  Now see if "V - Z" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Z, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, Q, MaxRecurse-1))
      // It does!  Now see if "V - Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Y, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // Z - (X - Y) -> (Z - X) + Y if everything simplifies.
  // For example, X - (X - Y) -> Y.
  Z = Op0;
  if (MaxRecurse && match(Op1, m_Sub(m_Value(X), m_Value(Y)))) // Z - (X - Y)
    // See if "V === Z - X" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Z, X, Q, MaxRecurse-1))
      // It does!  Now see if "V + Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, V, Y, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }

  // trunc(X) - trunc(Y) -> trunc(X - Y) if everything simplifies.
  if (MaxRecurse && match(Op0, m_Trunc(m_Value(X))) &&
      match(Op1, m_Trunc(m_Value(Y))))
    if (X->getType() == Y->getType())
      // See if "V === X - Y" simplifies.
      if (Value *V = SimplifyBinOp(Instruction::Sub, X, Y, Q, MaxRecurse-1))
        // It does!  Now see if "trunc V" simplifies.
        if (Value *W = SimplifyTruncInst(V, Op0->getType(), Q, MaxRecurse-1))
          // It does, return the simplified "trunc V".
          return W;

  // Variations on GEP(base, I, ...) - GEP(base, i, ...) -> GEP(null, I-i, ...).
  if (match(Op0, m_PtrToInt(m_Value(X))) &&
      match(Op1, m_PtrToInt(m_Value(Y))))
    if (Constant *Result = computePointerDifference(Q.DL, X, Y))
      return ConstantExpr::getIntegerCast(Result, Op0->getType(), true);

  // i1 sub -> xor.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Threading Sub over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A - select(cond, B, C)" means evaluating
  // "A-B" and "A-C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A-B" and "A-C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const DataLayout &DL, const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifySubInst(Op0, Op1, isNSW, isNUW, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

/// Given operands for an FAdd, see if we can fold the result.  If not, this
/// returns null.
static Value *SimplifyFAddInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const Query &Q, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::FAdd, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }
    // HLSL Change Begins.
    if (ConstantFP *FP = dyn_cast<ConstantFP>(Op0))
      if (FP->getValueAPF().isNaN())
        return Op0;
    // HLSL Change Ends.
    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }
  // HLSL Change Begins.
  if (ConstantFP *FP = dyn_cast<ConstantFP>(Op0))
    if (FP->getValueAPF().isNaN())
      return Op0;
  // HLSL Change Ends.

  // fadd X, -0 ==> X
  if (match(Op1, m_NegZero()))
    return Op0;

  // fadd X, 0 ==> X, when we know X is not -0
  if (match(Op1, m_Zero()) &&
      (FMF.noSignedZeros() || CannotBeNegativeZero(Op0)))
    return Op0;

  // fadd [nnan ninf] X, (fsub [nnan ninf] 0, X) ==> 0
  //   where nnan and ninf have to occur at least once somewhere in this
  //   expression
  Value *SubOp = nullptr;
  if (match(Op1, m_FSub(m_AnyZero(), m_Specific(Op0))))
    SubOp = Op1;
  else if (match(Op0, m_FSub(m_AnyZero(), m_Specific(Op1))))
    SubOp = Op0;
  if (SubOp) {
    Instruction *FSub = cast<Instruction>(SubOp);
    if ((FMF.noNaNs() || FSub->hasNoNaNs()) &&
        (FMF.noInfs() || FSub->hasNoInfs()))
      return Constant::getNullValue(Op0->getType());
  }

  return nullptr;
}

/// Given operands for an FSub, see if we can fold the result.  If not, this
/// returns null.
static Value *SimplifyFSubInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const Query &Q, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::FSub, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }
    // HLSL Change Begins.
    if (ConstantFP *FP = dyn_cast<ConstantFP>(Op0))
      if (FP->getValueAPF().isNaN())
        return Op0;
    // HLSL Change Ends.
  }
  // HLSL Change Begins.
  if (ConstantFP *FP = dyn_cast<ConstantFP>(Op1))
    if (FP->getValueAPF().isNaN())
      return Op1;
  // HLSL Change Ends.

  // fsub X, 0 ==> X
  if (match(Op1, m_Zero()))
    return Op0;

  // fsub X, -0 ==> X, when we know X is not -0
  if (match(Op1, m_NegZero()) &&
      (FMF.noSignedZeros() || CannotBeNegativeZero(Op0)))
    return Op0;

  // fsub 0, (fsub -0.0, X) ==> X
  Value *X;
  if (match(Op0, m_AnyZero())) {
    if (match(Op1, m_FSub(m_NegZero(), m_Value(X))))
      return X;
    if (FMF.noSignedZeros() && match(Op1, m_FSub(m_AnyZero(), m_Value(X))))
      return X;
  }

  // fsub nnan x, x ==> 0.0
  if (FMF.noNaNs() && Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  return nullptr;
}

/// Given the operands for an FMul, see if we can fold the result
static Value *SimplifyFMulInst(Value *Op0, Value *Op1,
                               FastMathFlags FMF,
                               const Query &Q,
                               unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = {CLHS, CRHS};
      return ConstantFoldInstOperands(Instruction::FMul, CLHS->getType(), Ops,
                                      Q.DL, Q.TLI);
    }
    // HLSL Change Begins.
    if (ConstantFP *FP = dyn_cast<ConstantFP>(Op0))
      if (FP->getValueAPF().isNaN())
        return Op0;
    // HLSL Change Ends.
    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }
  // HLSL Change Begins.
  if (ConstantFP *FP = dyn_cast<ConstantFP>(Op0))
    if (FP->getValueAPF().isNaN())
      return Op0;
  // HLSL Change Ends.
  // fmul X, 1.0 ==> X
  if (match(Op1, m_FPOne()))
    return Op0;

  // fmul nnan nsz X, 0 ==> 0
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op1, m_AnyZero()))
    return Op1;

  return nullptr;
}

/// SimplifyMulInst - Given operands for a Mul, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyMulInst(Value *Op0, Value *Op1, const Query &Q,
                              unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Mul, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X * undef -> 0
  if (match(Op1, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // X * 0 -> 0
  if (match(Op1, m_Zero()))
    return Op1;

  // X * 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  // (X / Y) * Y -> X if the division is exact.
  Value *X = nullptr;
  if (match(Op0, m_Exact(m_IDiv(m_Value(X), m_Specific(Op1)))) || // (X / Y) * Y
      match(Op1, m_Exact(m_IDiv(m_Value(X), m_Specific(Op0)))))   // Y * (X / Y)
    return X;

  // i1 mul -> and.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyAndInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Mul, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Mul distributes over Add.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::Mul, Op0, Op1, Instruction::Add,
                             Q, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::Mul, Op0, Op1, Q,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Mul, Op0, Op1, Q,
                                      MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyFAddInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFAddInst(Op0, Op1, FMF, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

Value *llvm::SimplifyFSubInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFSubInst(Op0, Op1, FMF, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

Value *llvm::SimplifyFMulInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFMulInst(Op0, Op1, FMF, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

Value *llvm::SimplifyMulInst(Value *Op0, Value *Op1, const DataLayout &DL,
                             const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyMulInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

/// SimplifyDiv - Given operands for an SDiv or UDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyDiv(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const Query &Q, unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, Q.DL, Q.TLI);
    }
  }

  bool isSigned = Opcode == Instruction::SDiv;

  // X / undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // X / 0 -> undef, we don't need to preserve faults!
  if (match(Op1, m_Zero()))
    return UndefValue::get(Op1->getType());

  // undef / X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // 0 / X -> 0, we don't need to preserve faults!
  if (match(Op0, m_Zero()))
    return Op0;

  // X / 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  if (Op0->getType()->isIntegerTy(1))
    // It can't be division by zero, hence it must be division by one.
    return Op0;

  // X / X -> 1
  if (Op0 == Op1)
    return ConstantInt::get(Op0->getType(), 1);

  // (X * Y) / Y -> X if the multiplication does not overflow.
  Value *X = nullptr, *Y = nullptr;
  if (match(Op0, m_Mul(m_Value(X), m_Value(Y))) && (X == Op1 || Y == Op1)) {
    if (Y != Op1) std::swap(X, Y); // Ensure expression is (X * Y) / Y, Y = Op1
    OverflowingBinaryOperator *Mul = cast<OverflowingBinaryOperator>(Op0);
    // If the Mul knows it does not overflow, then we are good to go.
    if ((isSigned && Mul->hasNoSignedWrap()) ||
        (!isSigned && Mul->hasNoUnsignedWrap()))
      return X;
    // If X has the form X = A / Y then X * Y cannot overflow.
    if (BinaryOperator *Div = dyn_cast<BinaryOperator>(X))
      if (Div->getOpcode() == Opcode && Div->getOperand(1) == Y)
        return X;
  }

  // (X rem Y) / Y -> 0
  if ((isSigned && match(Op0, m_SRem(m_Value(), m_Specific(Op1)))) ||
      (!isSigned && match(Op0, m_URem(m_Value(), m_Specific(Op1)))))
    return Constant::getNullValue(Op0->getType());

  // (X /u C1) /u C2 -> 0 if C1 * C2 overflow
  ConstantInt *C1, *C2;
  if (!isSigned && match(Op0, m_UDiv(m_Value(X), m_ConstantInt(C1))) &&
      match(Op1, m_ConstantInt(C2))) {
    bool Overflow;
    C1->getValue().umul_ov(C2->getValue(), Overflow);
    if (Overflow)
      return Constant::getNullValue(Op0->getType());
  }

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  return nullptr;
}

/// SimplifySDivInst - Given operands for an SDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySDivInst(Value *Op0, Value *Op1, const Query &Q,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyDiv(Instruction::SDiv, Op0, Op1, Q, MaxRecurse))
    return V;

  return nullptr;
}

Value *llvm::SimplifySDivInst(Value *Op0, Value *Op1, const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifySDivInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// SimplifyUDivInst - Given operands for a UDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyUDivInst(Value *Op0, Value *Op1, const Query &Q,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyDiv(Instruction::UDiv, Op0, Op1, Q, MaxRecurse))
    return V;

  return nullptr;
}

Value *llvm::SimplifyUDivInst(Value *Op0, Value *Op1, const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyUDivInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

static Value *SimplifyFDivInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                               const Query &Q, unsigned) {
  // HLSL Change Begins.
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = {C0, C1};
      return ConstantFoldInstOperands(Instruction::FDiv, C0->getType(), Ops,
                                      Q.DL, Q.TLI);
    }
    if (ConstantFP *FP = dyn_cast<ConstantFP>(C0))
      if (FP->getValueAPF().isNaN())
        return Op0;
  }
  if (ConstantFP *FP = dyn_cast<ConstantFP>(Op1))
    if (FP->getValueAPF().isNaN())
      return Op1;
  // HLSL Change Ends.

  // undef / X -> undef    (the undef could be a snan).
  if (match(Op0, m_Undef()))
    return Op0;

  // X / undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // 0 / X -> 0
  // Requires that NaNs are off (X could be zero) and signed zeroes are
  // ignored (X could be positive or negative, so the output sign is unknown).
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op0, m_AnyZero()))
    return Op0;

  if (FMF.noNaNs()) {
    // X / X -> 1.0 is legal when NaNs are ignored.
    if (Op0 == Op1)
      return ConstantFP::get(Op0->getType(), 1.0);

    // -X /  X -> -1.0 and
    //  X / -X -> -1.0 are legal when NaNs are ignored.
    // We can ignore signed zeros because +-0.0/+-0.0 is NaN and ignored.
    if ((BinaryOperator::isFNeg(Op0, /*IgnoreZeroSign=*/true) &&
         BinaryOperator::getFNegArgument(Op0) == Op1) ||
        (BinaryOperator::isFNeg(Op1, /*IgnoreZeroSign=*/true) &&
         BinaryOperator::getFNegArgument(Op1) == Op0))
      return ConstantFP::get(Op0->getType(), -1.0);
  }

  return nullptr;
}

Value *llvm::SimplifyFDivInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFDivInst(Op0, Op1, FMF, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// SimplifyRem - Given operands for an SRem or URem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyRem(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const Query &Q, unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, Q.DL, Q.TLI);
    }
  }

  // X % undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // undef % X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // 0 % X -> 0, we don't need to preserve faults!
  if (match(Op0, m_Zero()))
    return Op0;

  // X % 0 -> undef, we don't need to preserve faults!
  if (match(Op1, m_Zero()))
    return UndefValue::get(Op0->getType());

  // X % 1 -> 0
  if (match(Op1, m_One()))
    return Constant::getNullValue(Op0->getType());

  if (Op0->getType()->isIntegerTy(1))
    // It can't be remainder by zero, hence it must be remainder by one.
    return Constant::getNullValue(Op0->getType());

  // X % X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // (X % Y) % Y -> X % Y
  if ((Opcode == Instruction::SRem &&
       match(Op0, m_SRem(m_Value(), m_Specific(Op1)))) ||
      (Opcode == Instruction::URem &&
       match(Op0, m_URem(m_Value(), m_Specific(Op1)))))
    return Op0;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  return nullptr;
}

/// SimplifySRemInst - Given operands for an SRem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySRemInst(Value *Op0, Value *Op1, const Query &Q,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyRem(Instruction::SRem, Op0, Op1, Q, MaxRecurse))
    return V;

  return nullptr;
}

Value *llvm::SimplifySRemInst(Value *Op0, Value *Op1, const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifySRemInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// SimplifyURemInst - Given operands for a URem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyURemInst(Value *Op0, Value *Op1, const Query &Q,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyRem(Instruction::URem, Op0, Op1, Q, MaxRecurse))
    return V;

  return nullptr;
}

Value *llvm::SimplifyURemInst(Value *Op0, Value *Op1, const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyURemInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

static Value *SimplifyFRemInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                               const Query &Q, unsigned) {
  // HLSL Change Begins.
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = {C0, C1};
      return ConstantFoldInstOperands(Instruction::FRem, C0->getType(), Ops,
                                      Q.DL, Q.TLI);
    }
    if (ConstantFP *FP = dyn_cast<ConstantFP>(C0))
      if (FP->getValueAPF().isNaN())
        return Op0;
  }
  if (ConstantFP *FP = dyn_cast<ConstantFP>(Op1))
    if (FP->getValueAPF().isNaN())
      return Op1;
  // HLSL Change Ends.
  // undef % X -> undef    (the undef could be a snan).
  if (match(Op0, m_Undef()))
    return Op0;

  // X % undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // 0 % X -> 0
  // Requires that NaNs are off (X could be zero) and signed zeroes are
  // ignored (X could be positive or negative, so the output sign is unknown).
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op0, m_AnyZero()))
    return Op0;

  return nullptr;
}

Value *llvm::SimplifyFRemInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFRemInst(Op0, Op1, FMF, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// isUndefShift - Returns true if a shift by \c Amount always yields undef.
static bool isUndefShift(Value *Amount) {
  Constant *C = dyn_cast<Constant>(Amount);
  if (!C)
    return false;

  // X shift by undef -> undef because it may shift by the bitwidth.
  if (isa<UndefValue>(C))
    return true;

  // Shifting by the bitwidth or more is undefined.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
    if (CI->getValue().getLimitedValue() >=
        CI->getType()->getScalarSizeInBits())
      return true;

  // If all lanes of a vector shift are undefined the whole shift is.
  if (isa<ConstantVector>(C) || isa<ConstantDataVector>(C)) {
    for (unsigned I = 0, E = C->getType()->getVectorNumElements(); I != E; ++I)
      if (!isUndefShift(C->getAggregateElement(I)))
        return false;
    return true;
  }

  return false;
}

/// SimplifyShift - Given operands for an Shl, LShr or AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyShift(unsigned Opcode, Value *Op0, Value *Op1,
                            const Query &Q, unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, Q.DL, Q.TLI);
    }
  }

  // 0 shift by X -> 0
  if (match(Op0, m_Zero()))
    return Op0;

  // X shift by 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // Fold undefined shifts.
  if (isUndefShift(Op1))
    return UndefValue::get(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  return nullptr;
}

/// \brief Given operands for an Shl, LShr or AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyRightShift(unsigned Opcode, Value *Op0, Value *Op1,
                                 bool isExact, const Query &Q,
                                 unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Opcode, Op0, Op1, Q, MaxRecurse))
    return V;

  // X >> X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // undef >> X -> 0
  // undef >> X -> undef (if it's exact)
  if (match(Op0, m_Undef()))
    return isExact ? Op0 : Constant::getNullValue(Op0->getType());

  // The low bit cannot be shifted out of an exact shift if it is set.
  if (isExact) {
    unsigned BitWidth = Op0->getType()->getScalarSizeInBits();
    APInt Op0KnownZero(BitWidth, 0);
    APInt Op0KnownOne(BitWidth, 0);
    computeKnownBits(Op0, Op0KnownZero, Op0KnownOne, Q.DL, /*Depth=*/0, Q.AC,
                     Q.CxtI, Q.DT);
    if (Op0KnownOne[0])
      return Op0;
  }

  return nullptr;
}

/// SimplifyShlInst - Given operands for an Shl, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const Query &Q, unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::Shl, Op0, Op1, Q, MaxRecurse))
    return V;

  // undef << X -> 0
  // undef << X -> undef if (if it's NSW/NUW)
  if (match(Op0, m_Undef()))
    return isNSW || isNUW ? Op0 : Constant::getNullValue(Op0->getType());

  // (X >> A) << A -> X
  Value *X;
  if (match(Op0, m_Exact(m_Shr(m_Value(X), m_Specific(Op1)))))
    return X;
  return nullptr;
}

Value *llvm::SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const DataLayout &DL, const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyShlInst(Op0, Op1, isNSW, isNUW, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

/// SimplifyLShrInst - Given operands for an LShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                               const Query &Q, unsigned MaxRecurse) {
  if (Value *V = SimplifyRightShift(Instruction::LShr, Op0, Op1, isExact, Q,
                                    MaxRecurse))
      return V;

  // (X << A) >> A -> X
  Value *X;
  if (match(Op0, m_NUWShl(m_Value(X), m_Specific(Op1))))
    return X;

  return nullptr;
}

Value *llvm::SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyLShrInst(Op0, Op1, isExact, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// SimplifyAShrInst - Given operands for an AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                               const Query &Q, unsigned MaxRecurse) {
  if (Value *V = SimplifyRightShift(Instruction::AShr, Op0, Op1, isExact, Q,
                                    MaxRecurse))
    return V;

  // all ones >>a X -> all ones
  if (match(Op0, m_AllOnes()))
    return Op0;

  // (X << A) >> A -> X
  Value *X;
  if (match(Op0, m_NSWShl(m_Value(X), m_Specific(Op1))))
    return X;

  // Arithmetic shifting an all-sign-bit value is a no-op.
  unsigned NumSignBits = ComputeNumSignBits(Op0, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
  if (NumSignBits == Op0->getType()->getScalarSizeInBits())
    return Op0;

  return nullptr;
}

Value *llvm::SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyAShrInst(Op0, Op1, isExact, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

static Value *simplifyUnsignedRangeCheck(ICmpInst *ZeroICmp,
                                         ICmpInst *UnsignedICmp, bool IsAnd) {
  Value *X, *Y;

  ICmpInst::Predicate EqPred;
  if (!match(ZeroICmp, m_ICmp(EqPred, m_Value(Y), m_Zero())) ||
      !ICmpInst::isEquality(EqPred))
    return nullptr;

  ICmpInst::Predicate UnsignedPred;
  if (match(UnsignedICmp, m_ICmp(UnsignedPred, m_Value(X), m_Specific(Y))) &&
      ICmpInst::isUnsigned(UnsignedPred))
    ;
  else if (match(UnsignedICmp,
                 m_ICmp(UnsignedPred, m_Value(Y), m_Specific(X))) &&
           ICmpInst::isUnsigned(UnsignedPred))
    UnsignedPred = ICmpInst::getSwappedPredicate(UnsignedPred);
  else
    return nullptr;

  // X < Y && Y != 0  -->  X < Y
  // X < Y || Y != 0  -->  Y != 0
  if (UnsignedPred == ICmpInst::ICMP_ULT && EqPred == ICmpInst::ICMP_NE)
    return IsAnd ? UnsignedICmp : ZeroICmp;

  // X >= Y || Y != 0  -->  true
  // X >= Y || Y == 0  -->  X >= Y
  if (UnsignedPred == ICmpInst::ICMP_UGE && !IsAnd) {
    if (EqPred == ICmpInst::ICMP_NE)
      return getTrue(UnsignedICmp->getType());
    return UnsignedICmp;
  }

  // X < Y && Y == 0  -->  false
  if (UnsignedPred == ICmpInst::ICMP_ULT && EqPred == ICmpInst::ICMP_EQ &&
      IsAnd)
    return getFalse(UnsignedICmp->getType());

  return nullptr;
}

// Simplify (and (icmp ...) (icmp ...)) to true when we can tell that the range
// of possible values cannot be satisfied.
static Value *SimplifyAndOfICmps(ICmpInst *Op0, ICmpInst *Op1) {
  ICmpInst::Predicate Pred0, Pred1;
  ConstantInt *CI1, *CI2;
  Value *V;

  if (Value *X = simplifyUnsignedRangeCheck(Op0, Op1, /*IsAnd=*/true))
    return X;

  if (!match(Op0, m_ICmp(Pred0, m_Add(m_Value(V), m_ConstantInt(CI1)),
                         m_ConstantInt(CI2))))
   return nullptr;

  if (!match(Op1, m_ICmp(Pred1, m_Specific(V), m_Specific(CI1))))
    return nullptr;

  Type *ITy = Op0->getType();

  auto *AddInst = cast<BinaryOperator>(Op0->getOperand(0));
  bool isNSW = AddInst->hasNoSignedWrap();
  bool isNUW = AddInst->hasNoUnsignedWrap();

  const APInt &CI1V = CI1->getValue();
  const APInt &CI2V = CI2->getValue();
  const APInt Delta = CI2V - CI1V;
  if (CI1V.isStrictlyPositive()) {
    if (Delta == 2) {
      if (Pred0 == ICmpInst::ICMP_ULT && Pred1 == ICmpInst::ICMP_SGT)
        return getFalse(ITy);
      if (Pred0 == ICmpInst::ICMP_SLT && Pred1 == ICmpInst::ICMP_SGT && isNSW)
        return getFalse(ITy);
    }
    if (Delta == 1) {
      if (Pred0 == ICmpInst::ICMP_ULE && Pred1 == ICmpInst::ICMP_SGT)
        return getFalse(ITy);
      if (Pred0 == ICmpInst::ICMP_SLE && Pred1 == ICmpInst::ICMP_SGT && isNSW)
        return getFalse(ITy);
    }
  }
  if (CI1V.getBoolValue() && isNUW) {
    if (Delta == 2)
      if (Pred0 == ICmpInst::ICMP_ULT && Pred1 == ICmpInst::ICMP_UGT)
        return getFalse(ITy);
    if (Delta == 1)
      if (Pred0 == ICmpInst::ICMP_ULE && Pred1 == ICmpInst::ICMP_UGT)
        return getFalse(ITy);
  }

  return nullptr;
}

/// SimplifyAndInst - Given operands for an And, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAndInst(Value *Op0, Value *Op1, const Query &Q,
                              unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::And, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X & undef -> 0
  if (match(Op1, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // X & X = X
  if (Op0 == Op1)
    return Op0;

  // X & 0 = 0
  if (match(Op1, m_Zero()))
    return Op1;

  // X & -1 = X
  if (match(Op1, m_AllOnes()))
    return Op0;

  // A & ~A  =  ~A & A  =  0
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getNullValue(Op0->getType());

  // (A | ?) & A = A
  Value *A = nullptr, *B = nullptr;
  if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
      (A == Op1 || B == Op1))
    return Op1;

  // A & (A | ?) = A
  if (match(Op1, m_Or(m_Value(A), m_Value(B))) &&
      (A == Op0 || B == Op0))
    return Op0;

  // A & (-A) = A if A is a power of two or zero.
  if (match(Op0, m_Neg(m_Specific(Op1))) ||
      match(Op1, m_Neg(m_Specific(Op0)))) {
    if (isKnownToBeAPowerOfTwo(Op0, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI,
                               Q.DT))
      return Op0;
    if (isKnownToBeAPowerOfTwo(Op1, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI,
                               Q.DT))
      return Op1;
  }

  if (auto *ICILHS = dyn_cast<ICmpInst>(Op0)) {
    if (auto *ICIRHS = dyn_cast<ICmpInst>(Op1)) {
      if (Value *V = SimplifyAndOfICmps(ICILHS, ICIRHS))
        return V;
      if (Value *V = SimplifyAndOfICmps(ICIRHS, ICILHS))
        return V;
    }
  }

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::And, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // And distributes over Or.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::And, Op0, Op1, Instruction::Or,
                             Q, MaxRecurse))
    return V;

  // And distributes over Xor.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::And, Op0, Op1, Instruction::Xor,
                             Q, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::And, Op0, Op1, Q,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::And, Op0, Op1, Q,
                                      MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyAndInst(Value *Op0, Value *Op1, const DataLayout &DL,
                             const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyAndInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

// Simplify (or (icmp ...) (icmp ...)) to true when we can tell that the union
// contains all possible values.
static Value *SimplifyOrOfICmps(ICmpInst *Op0, ICmpInst *Op1) {
  ICmpInst::Predicate Pred0, Pred1;
  ConstantInt *CI1, *CI2;
  Value *V;

  if (Value *X = simplifyUnsignedRangeCheck(Op0, Op1, /*IsAnd=*/false))
    return X;

  if (!match(Op0, m_ICmp(Pred0, m_Add(m_Value(V), m_ConstantInt(CI1)),
                         m_ConstantInt(CI2))))
   return nullptr;

  if (!match(Op1, m_ICmp(Pred1, m_Specific(V), m_Specific(CI1))))
    return nullptr;

  Type *ITy = Op0->getType();

  auto *AddInst = cast<BinaryOperator>(Op0->getOperand(0));
  bool isNSW = AddInst->hasNoSignedWrap();
  bool isNUW = AddInst->hasNoUnsignedWrap();

  const APInt &CI1V = CI1->getValue();
  const APInt &CI2V = CI2->getValue();
  const APInt Delta = CI2V - CI1V;
  if (CI1V.isStrictlyPositive()) {
    if (Delta == 2) {
      if (Pred0 == ICmpInst::ICMP_UGE && Pred1 == ICmpInst::ICMP_SLE)
        return getTrue(ITy);
      if (Pred0 == ICmpInst::ICMP_SGE && Pred1 == ICmpInst::ICMP_SLE && isNSW)
        return getTrue(ITy);
    }
    if (Delta == 1) {
      if (Pred0 == ICmpInst::ICMP_UGT && Pred1 == ICmpInst::ICMP_SLE)
        return getTrue(ITy);
      if (Pred0 == ICmpInst::ICMP_SGT && Pred1 == ICmpInst::ICMP_SLE && isNSW)
        return getTrue(ITy);
    }
  }
  if (CI1V.getBoolValue() && isNUW) {
    if (Delta == 2)
      if (Pred0 == ICmpInst::ICMP_UGE && Pred1 == ICmpInst::ICMP_ULE)
        return getTrue(ITy);
    if (Delta == 1)
      if (Pred0 == ICmpInst::ICMP_UGT && Pred1 == ICmpInst::ICMP_ULE)
        return getTrue(ITy);
  }

  return nullptr;
}

/// SimplifyOrInst - Given operands for an Or, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyOrInst(Value *Op0, Value *Op1, const Query &Q,
                             unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Or, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X | undef -> -1
  if (match(Op1, m_Undef()))
    return Constant::getAllOnesValue(Op0->getType());

  // X | X = X
  if (Op0 == Op1)
    return Op0;

  // X | 0 = X
  if (match(Op1, m_Zero()))
    return Op0;

  // X | -1 = -1
  if (match(Op1, m_AllOnes()))
    return Op1;

  // A | ~A  =  ~A | A  =  -1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  // (A & ?) | A = A
  Value *A = nullptr, *B = nullptr;
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      (A == Op1 || B == Op1))
    return Op1;

  // A | (A & ?) = A
  if (match(Op1, m_And(m_Value(A), m_Value(B))) &&
      (A == Op0 || B == Op0))
    return Op0;

  // ~(A & ?) | A = -1
  if (match(Op0, m_Not(m_And(m_Value(A), m_Value(B)))) &&
      (A == Op1 || B == Op1))
    return Constant::getAllOnesValue(Op1->getType());

  // A | ~(A & ?) = -1
  if (match(Op1, m_Not(m_And(m_Value(A), m_Value(B)))) &&
      (A == Op0 || B == Op0))
    return Constant::getAllOnesValue(Op0->getType());

  if (auto *ICILHS = dyn_cast<ICmpInst>(Op0)) {
    if (auto *ICIRHS = dyn_cast<ICmpInst>(Op1)) {
      if (Value *V = SimplifyOrOfICmps(ICILHS, ICIRHS))
        return V;
      if (Value *V = SimplifyOrOfICmps(ICIRHS, ICILHS))
        return V;
    }
  }

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Or, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Or distributes over And.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::Or, Op0, Op1, Instruction::And, Q,
                             MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::Or, Op0, Op1, Q,
                                         MaxRecurse))
      return V;

  // (A & C)|(B & D)
  Value *C = nullptr, *D = nullptr;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    ConstantInt *C1 = dyn_cast<ConstantInt>(C);
    ConstantInt *C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2 && (C1->getValue() == ~C2->getValue())) {
      // (A & C1)|(B & C2)
      // If we have: ((V + N) & C1) | (V & C2)
      // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
      // replace with V+N.
      Value *V1, *V2;
      if ((C2->getValue() & (C2->getValue() + 1)) == 0 && // C2 == 0+1+
          match(A, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == B &&
            MaskedValueIsZero(V2, C2->getValue(), Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return A;
        if (V2 == B &&
            MaskedValueIsZero(V1, C2->getValue(), Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return A;
      }
      // Or commutes, try both ways.
      if ((C1->getValue() & (C1->getValue() + 1)) == 0 &&
          match(B, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == A &&
            MaskedValueIsZero(V2, C1->getValue(), Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return B;
        if (V2 == A &&
            MaskedValueIsZero(V1, C1->getValue(), Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return B;
      }
    }
  }

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Or, Op0, Op1, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyOrInst(Value *Op0, Value *Op1, const DataLayout &DL,
                            const TargetLibraryInfo *TLI,
                            const DominatorTree *DT, AssumptionCache *AC,
                            const Instruction *CxtI) {
  return ::SimplifyOrInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                          RecursionLimit);
}

/// SimplifyXorInst - Given operands for a Xor, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyXorInst(Value *Op0, Value *Op1, const Query &Q,
                              unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Xor, CLHS->getType(),
                                      Ops, Q.DL, Q.TLI);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // A ^ undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // A ^ 0 = A
  if (match(Op1, m_Zero()))
    return Op0;

  // A ^ A = 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // A ^ ~A  =  ~A ^ A  =  -1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Xor, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Threading Xor over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A ^ select(cond, B, C)" means evaluating
  // "A^B" and "A^C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A^B" and "A^C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifyXorInst(Value *Op0, Value *Op1, const DataLayout &DL,
                             const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyXorInst(Op0, Op1, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

static Type *GetCompareTy(Value *Op) {
  return CmpInst::makeCmpResultType(Op->getType());
}

/// ExtractEquivalentCondition - Rummage around inside V looking for something
/// equivalent to the comparison "LHS Pred RHS".  Return such a value if found,
/// otherwise return null.  Helper function for analyzing max/min idioms.
static Value *ExtractEquivalentCondition(Value *V, CmpInst::Predicate Pred,
                                         Value *LHS, Value *RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI)
    return nullptr;
  CmpInst *Cmp = dyn_cast<CmpInst>(SI->getCondition());
  if (!Cmp)
    return nullptr;
  Value *CmpLHS = Cmp->getOperand(0), *CmpRHS = Cmp->getOperand(1);
  if (Pred == Cmp->getPredicate() && LHS == CmpLHS && RHS == CmpRHS)
    return Cmp;
  if (Pred == CmpInst::getSwappedPredicate(Cmp->getPredicate()) &&
      LHS == CmpRHS && RHS == CmpLHS)
    return Cmp;
  return nullptr;
}

// A significant optimization not implemented here is assuming that alloca
// addresses are not equal to incoming argument values. They don't *alias*,
// as we say, but that doesn't mean they aren't equal, so we take a
// conservative approach.
//
// This is inspired in part by C++11 5.10p1:
//   "Two pointers of the same type compare equal if and only if they are both
//    null, both point to the same function, or both represent the same
//    address."
//
// This is pretty permissive.
//
// It's also partly due to C11 6.5.9p6:
//   "Two pointers compare equal if and only if both are null pointers, both are
//    pointers to the same object (including a pointer to an object and a
//    subobject at its beginning) or function, both are pointers to one past the
//    last element of the same array object, or one is a pointer to one past the
//    end of one array object and the other is a pointer to the start of a
//    different array object that happens to immediately follow the first array
//    object in the address space.)
//
// C11's version is more restrictive, however there's no reason why an argument
// couldn't be a one-past-the-end value for a stack object in the caller and be
// equal to the beginning of a stack object in the callee.
//
// If the C and C++ standards are ever made sufficiently restrictive in this
// area, it may be possible to update LLVM's semantics accordingly and reinstate
// this optimization.
static Constant *computePointerICmp(const DataLayout &DL,
                                    const TargetLibraryInfo *TLI,
                                    CmpInst::Predicate Pred, Value *LHS,
                                    Value *RHS) {
  // First, skip past any trivial no-ops.
  LHS = LHS->stripPointerCasts();
  RHS = RHS->stripPointerCasts();

  // A non-null pointer is not equal to a null pointer.
  if (llvm::isKnownNonNull(LHS, TLI) && isa<ConstantPointerNull>(RHS) &&
      (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE))
    return ConstantInt::get(GetCompareTy(LHS),
                            !CmpInst::isTrueWhenEqual(Pred));

  // We can only fold certain predicates on pointer comparisons.
  switch (Pred) {
  default:
    return nullptr;

    // Equality comaprisons are easy to fold.
  case CmpInst::ICMP_EQ:
  case CmpInst::ICMP_NE:
    break;

    // We can only handle unsigned relational comparisons because 'inbounds' on
    // a GEP only protects against unsigned wrapping.
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
    // However, we have to switch them to their signed variants to handle
    // negative indices from the base pointer.
    Pred = ICmpInst::getSignedPredicate(Pred);
    break;
  }

  // Strip off any constant offsets so that we can reason about them.
  // It's tempting to use getUnderlyingObject or even just stripInBoundsOffsets
  // here and compare base addresses like AliasAnalysis does, however there are
  // numerous hazards. AliasAnalysis and its utilities rely on special rules
  // governing loads and stores which don't apply to icmps. Also, AliasAnalysis
  // doesn't need to guarantee pointer inequality when it says NoAlias.
  Constant *LHSOffset = stripAndComputeConstantOffsets(DL, LHS);
  Constant *RHSOffset = stripAndComputeConstantOffsets(DL, RHS);

  // If LHS and RHS are related via constant offsets to the same base
  // value, we can replace it with an icmp which just compares the offsets.
  if (LHS == RHS)
    return ConstantExpr::getICmp(Pred, LHSOffset, RHSOffset);

  // Various optimizations for (in)equality comparisons.
  if (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) {
    // Different non-empty allocations that exist at the same time have
    // different addresses (if the program can tell). Global variables always
    // exist, so they always exist during the lifetime of each other and all
    // allocas. Two different allocas usually have different addresses...
    //
    // However, if there's an @llvm.stackrestore dynamically in between two
    // allocas, they may have the same address. It's tempting to reduce the
    // scope of the problem by only looking at *static* allocas here. That would
    // cover the majority of allocas while significantly reducing the likelihood
    // of having an @llvm.stackrestore pop up in the middle. However, it's not
    // actually impossible for an @llvm.stackrestore to pop up in the middle of
    // an entry block. Also, if we have a block that's not attached to a
    // function, we can't tell if it's "static" under the current definition.
    // Theoretically, this problem could be fixed by creating a new kind of
    // instruction kind specifically for static allocas. Such a new instruction
    // could be required to be at the top of the entry block, thus preventing it
    // from being subject to a @llvm.stackrestore. Instcombine could even
    // convert regular allocas into these special allocas. It'd be nifty.
    // However, until then, this problem remains open.
    //
    // So, we'll assume that two non-empty allocas have different addresses
    // for now.
    //
    // With all that, if the offsets are within the bounds of their allocations
    // (and not one-past-the-end! so we can't use inbounds!), and their
    // allocations aren't the same, the pointers are not equal.
    //
    // Note that it's not necessary to check for LHS being a global variable
    // address, due to canonicalization and constant folding.
    if (isa<AllocaInst>(LHS) &&
        (isa<AllocaInst>(RHS) || isa<GlobalVariable>(RHS))) {
      ConstantInt *LHSOffsetCI = dyn_cast<ConstantInt>(LHSOffset);
      ConstantInt *RHSOffsetCI = dyn_cast<ConstantInt>(RHSOffset);
      uint64_t LHSSize, RHSSize;
      if (LHSOffsetCI && RHSOffsetCI &&
          getObjectSize(LHS, LHSSize, DL, TLI) &&
          getObjectSize(RHS, RHSSize, DL, TLI)) {
        const APInt &LHSOffsetValue = LHSOffsetCI->getValue();
        const APInt &RHSOffsetValue = RHSOffsetCI->getValue();
        if (!LHSOffsetValue.isNegative() &&
            !RHSOffsetValue.isNegative() &&
            LHSOffsetValue.ult(LHSSize) &&
            RHSOffsetValue.ult(RHSSize)) {
          return ConstantInt::get(GetCompareTy(LHS),
                                  !CmpInst::isTrueWhenEqual(Pred));
        }
      }

      // Repeat the above check but this time without depending on DataLayout
      // or being able to compute a precise size.
      if (!cast<PointerType>(LHS->getType())->isEmptyTy() &&
          !cast<PointerType>(RHS->getType())->isEmptyTy() &&
          LHSOffset->isNullValue() &&
          RHSOffset->isNullValue())
        return ConstantInt::get(GetCompareTy(LHS),
                                !CmpInst::isTrueWhenEqual(Pred));
    }

    // Even if an non-inbounds GEP occurs along the path we can still optimize
    // equality comparisons concerning the result. We avoid walking the whole
    // chain again by starting where the last calls to
    // stripAndComputeConstantOffsets left off and accumulate the offsets.
    Constant *LHSNoBound = stripAndComputeConstantOffsets(DL, LHS, true);
    Constant *RHSNoBound = stripAndComputeConstantOffsets(DL, RHS, true);
    if (LHS == RHS)
      return ConstantExpr::getICmp(Pred,
                                   ConstantExpr::getAdd(LHSOffset, LHSNoBound),
                                   ConstantExpr::getAdd(RHSOffset, RHSNoBound));

    // If one side of the equality comparison must come from a noalias call
    // (meaning a system memory allocation function), and the other side must
    // come from a pointer that cannot overlap with dynamically-allocated
    // memory within the lifetime of the current function (allocas, byval
    // arguments, globals), then determine the comparison result here.
    SmallVector<Value *, 8> LHSUObjs, RHSUObjs;
    GetUnderlyingObjects(LHS, LHSUObjs, DL);
    GetUnderlyingObjects(RHS, RHSUObjs, DL);

    // Is the set of underlying objects all noalias calls?
    auto IsNAC = [](SmallVectorImpl<Value *> &Objects) {
      return std::all_of(Objects.begin(), Objects.end(),
                         [](Value *V){ return isNoAliasCall(V); });
    };

    // Is the set of underlying objects all things which must be disjoint from
    // noalias calls. For allocas, we consider only static ones (dynamic
    // allocas might be transformed into calls to malloc not simultaneously
    // live with the compared-to allocation). For globals, we exclude symbols
    // that might be resolve lazily to symbols in another dynamically-loaded
    // library (and, thus, could be malloc'ed by the implementation).
    auto IsAllocDisjoint = [](SmallVectorImpl<Value *> &Objects) {
      return std::all_of(Objects.begin(), Objects.end(),
                         [](Value *V){
                           if (const AllocaInst *AI = dyn_cast<AllocaInst>(V))
                             return AI->getParent() && AI->getParent()->getParent() &&
                                    AI->isStaticAlloca();
                           if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
                             return (GV->hasLocalLinkage() ||
                                     GV->hasHiddenVisibility() ||
                                     GV->hasProtectedVisibility() ||
                                     GV->hasUnnamedAddr()) &&
                                    !GV->isThreadLocal();
                           if (const Argument *A = dyn_cast<Argument>(V))
                             return A->hasByValAttr();
                           return false;
                         });
    };

    if ((IsNAC(LHSUObjs) && IsAllocDisjoint(RHSUObjs)) ||
        (IsNAC(RHSUObjs) && IsAllocDisjoint(LHSUObjs)))
        return ConstantInt::get(GetCompareTy(LHS),
                                !CmpInst::isTrueWhenEqual(Pred));
  }

  // Otherwise, fail.
  return nullptr;
}

/// SimplifyICmpInst - Given operands for an ICmpInst, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               const Query &Q, unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isIntPredicate(Pred) && "Not an integer compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, Q.DL, Q.TLI);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  Type *ITy = GetCompareTy(LHS); // The return type.
  Type *OpTy = LHS->getType();   // The operand type.

  // icmp X, X -> true/false
  // X icmp undef -> true/false.  For example, icmp ugt %X, undef -> false
  // because X could be 0.
  if (LHS == RHS || isa<UndefValue>(RHS))
    return ConstantInt::get(ITy, CmpInst::isTrueWhenEqual(Pred));

  // Special case logic when the operands have i1 type.
  if (OpTy->getScalarType()->isIntegerTy(1)) {
    switch (Pred) {
    default: break;
    case ICmpInst::ICMP_EQ:
      // X == 1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    case ICmpInst::ICMP_NE:
      // X != 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_UGT:
      // X >u 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_UGE:
      // X >=u 1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    case ICmpInst::ICMP_SLT:
      // X <s 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_SLE:
      // X <=s -1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    }
  }

  // If we are comparing with zero then try hard since this is a common case.
  if (match(RHS, m_Zero())) {
    bool LHSKnownNonNegative, LHSKnownNegative;
    switch (Pred) {
    default: llvm_unreachable("Unknown ICmp predicate!");
    case ICmpInst::ICMP_ULT:
      return getFalse(ITy);
    case ICmpInst::ICMP_UGE:
      return getTrue(ITy);
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_ULE:
      if (isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
        return getFalse(ITy);
      break;
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_UGT:
      if (isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
        return getTrue(ITy);
      break;
    case ICmpInst::ICMP_SLT:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (LHSKnownNegative)
        return getTrue(ITy);
      if (LHSKnownNonNegative)
        return getFalse(ITy);
      break;
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (LHSKnownNegative)
        return getTrue(ITy);
      if (LHSKnownNonNegative &&
          isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
        return getFalse(ITy);
      break;
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (LHSKnownNegative)
        return getFalse(ITy);
      if (LHSKnownNonNegative)
        return getTrue(ITy);
      break;
    case ICmpInst::ICMP_SGT:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (LHSKnownNegative)
        return getFalse(ITy);
      if (LHSKnownNonNegative &&
          isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
        return getTrue(ITy);
      break;
    }
  }

  // See if we are doing a comparison with a constant integer.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    // Rule out tautological comparisons (eg., ult 0 or uge 0).
    ConstantRange RHS_CR = ICmpInst::makeConstantRange(Pred, CI->getValue());
    if (RHS_CR.isEmptySet())
      return ConstantInt::getFalse(CI->getContext());
    if (RHS_CR.isFullSet())
      return ConstantInt::getTrue(CI->getContext());

    // Many binary operators with constant RHS have easy to compute constant
    // range.  Use them to check whether the comparison is a tautology.
    unsigned Width = CI->getBitWidth();
    APInt Lower = APInt(Width, 0);
    APInt Upper = APInt(Width, 0);
    ConstantInt *CI2;
    if (match(LHS, m_URem(m_Value(), m_ConstantInt(CI2)))) {
      // 'urem x, CI2' produces [0, CI2).
      Upper = CI2->getValue();
    } else if (match(LHS, m_SRem(m_Value(), m_ConstantInt(CI2)))) {
      // 'srem x, CI2' produces (-|CI2|, |CI2|).
      Upper = CI2->getValue().abs();
      Lower = (-Upper) + 1;
    } else if (match(LHS, m_UDiv(m_ConstantInt(CI2), m_Value()))) {
      // 'udiv CI2, x' produces [0, CI2].
      Upper = CI2->getValue() + 1;
    } else if (match(LHS, m_UDiv(m_Value(), m_ConstantInt(CI2)))) {
      // 'udiv x, CI2' produces [0, UINT_MAX / CI2].
      APInt NegOne = APInt::getAllOnesValue(Width);
      if (!CI2->isZero())
        Upper = NegOne.udiv(CI2->getValue()) + 1;
    } else if (match(LHS, m_SDiv(m_ConstantInt(CI2), m_Value()))) {
      if (CI2->isMinSignedValue()) {
        // 'sdiv INT_MIN, x' produces [INT_MIN, INT_MIN / -2].
        Lower = CI2->getValue();
        Upper = Lower.lshr(1) + 1;
      } else {
        // 'sdiv CI2, x' produces [-|CI2|, |CI2|].
        Upper = CI2->getValue().abs() + 1;
        Lower = (-Upper) + 1;
      }
    } else if (match(LHS, m_SDiv(m_Value(), m_ConstantInt(CI2)))) {
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      APInt Val = CI2->getValue();
      if (Val.isAllOnesValue()) {
        // 'sdiv x, -1' produces [INT_MIN + 1, INT_MAX]
        //    where CI2 != -1 and CI2 != 0 and CI2 != 1
        Lower = IntMin + 1;
        Upper = IntMax + 1;
      } else if (Val.countLeadingZeros() < Width - 1) {
        // 'sdiv x, CI2' produces [INT_MIN / CI2, INT_MAX / CI2]
        //    where CI2 != -1 and CI2 != 0 and CI2 != 1
        Lower = IntMin.sdiv(Val);
        Upper = IntMax.sdiv(Val);
        if (Lower.sgt(Upper))
          std::swap(Lower, Upper);
        Upper = Upper + 1;
        assert(Upper != Lower && "Upper part of range has wrapped!");
      }
    } else if (match(LHS, m_NUWShl(m_ConstantInt(CI2), m_Value()))) {
      // 'shl nuw CI2, x' produces [CI2, CI2 << CLZ(CI2)]
      Lower = CI2->getValue();
      Upper = Lower.shl(Lower.countLeadingZeros()) + 1;
    } else if (match(LHS, m_NSWShl(m_ConstantInt(CI2), m_Value()))) {
      if (CI2->isNegative()) {
        // 'shl nsw CI2, x' produces [CI2 << CLO(CI2)-1, CI2]
        unsigned ShiftAmount = CI2->getValue().countLeadingOnes() - 1;
        Lower = CI2->getValue().shl(ShiftAmount);
        Upper = CI2->getValue() + 1;
      } else {
        // 'shl nsw CI2, x' produces [CI2, CI2 << CLZ(CI2)-1]
        unsigned ShiftAmount = CI2->getValue().countLeadingZeros() - 1;
        Lower = CI2->getValue();
        Upper = CI2->getValue().shl(ShiftAmount) + 1;
      }
    } else if (match(LHS, m_LShr(m_Value(), m_ConstantInt(CI2)))) {
      // 'lshr x, CI2' produces [0, UINT_MAX >> CI2].
      APInt NegOne = APInt::getAllOnesValue(Width);
      if (CI2->getValue().ult(Width))
        Upper = NegOne.lshr(CI2->getValue()) + 1;
    } else if (match(LHS, m_LShr(m_ConstantInt(CI2), m_Value()))) {
      // 'lshr CI2, x' produces [CI2 >> (Width-1), CI2].
      unsigned ShiftAmount = Width - 1;
      if (!CI2->isZero() && cast<BinaryOperator>(LHS)->isExact())
        ShiftAmount = CI2->getValue().countTrailingZeros();
      Lower = CI2->getValue().lshr(ShiftAmount);
      Upper = CI2->getValue() + 1;
    } else if (match(LHS, m_AShr(m_Value(), m_ConstantInt(CI2)))) {
      // 'ashr x, CI2' produces [INT_MIN >> CI2, INT_MAX >> CI2].
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      if (CI2->getValue().ult(Width)) {
        Lower = IntMin.ashr(CI2->getValue());
        Upper = IntMax.ashr(CI2->getValue()) + 1;
      }
    } else if (match(LHS, m_AShr(m_ConstantInt(CI2), m_Value()))) {
      unsigned ShiftAmount = Width - 1;
      if (!CI2->isZero() && cast<BinaryOperator>(LHS)->isExact())
        ShiftAmount = CI2->getValue().countTrailingZeros();
      if (CI2->isNegative()) {
        // 'ashr CI2, x' produces [CI2, CI2 >> (Width-1)]
        Lower = CI2->getValue();
        Upper = CI2->getValue().ashr(ShiftAmount) + 1;
      } else {
        // 'ashr CI2, x' produces [CI2 >> (Width-1), CI2]
        Lower = CI2->getValue().ashr(ShiftAmount);
        Upper = CI2->getValue() + 1;
      }
    } else if (match(LHS, m_Or(m_Value(), m_ConstantInt(CI2)))) {
      // 'or x, CI2' produces [CI2, UINT_MAX].
      Lower = CI2->getValue();
    } else if (match(LHS, m_And(m_Value(), m_ConstantInt(CI2)))) {
      // 'and x, CI2' produces [0, CI2].
      Upper = CI2->getValue() + 1;
    }
    if (Lower != Upper) {
      ConstantRange LHS_CR = ConstantRange(Lower, Upper);
      if (RHS_CR.contains(LHS_CR))
        return ConstantInt::getTrue(RHS->getContext());
      if (RHS_CR.inverse().contains(LHS_CR))
        return ConstantInt::getFalse(RHS->getContext());
    }
  }

  // Compare of cast, for example (zext X) != 0 -> X != 0
  if (isa<CastInst>(LHS) && (isa<Constant>(RHS) || isa<CastInst>(RHS))) {
    Instruction *LI = cast<CastInst>(LHS);
    Value *SrcOp = LI->getOperand(0);
    Type *SrcTy = SrcOp->getType();
    Type *DstTy = LI->getType();

    // Turn icmp (ptrtoint x), (ptrtoint/constant) into a compare of the input
    // if the integer type is the same size as the pointer type.
    if (MaxRecurse && isa<PtrToIntInst>(LI) &&
        Q.DL.getTypeSizeInBits(SrcTy) == DstTy->getPrimitiveSizeInBits()) {
      if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
        // Transfer the cast to the constant.
        if (Value *V = SimplifyICmpInst(Pred, SrcOp,
                                        ConstantExpr::getIntToPtr(RHSC, SrcTy),
                                        Q, MaxRecurse-1))
          return V;
      } else if (PtrToIntInst *RI = dyn_cast<PtrToIntInst>(RHS)) {
        if (RI->getOperand(0)->getType() == SrcTy)
          // Compare without the cast.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          Q, MaxRecurse-1))
            return V;
      }
    }

    if (isa<ZExtInst>(LHS)) {
      // Turn icmp (zext X), (zext Y) into a compare of X and Y if they have the
      // same type.
      if (ZExtInst *RI = dyn_cast<ZExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that signed predicates become unsigned.
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                          SrcOp, RI->getOperand(0), Q,
                                          MaxRecurse-1))
            return V;
      }
      // Turn icmp (zext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::ZExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two zero-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                        SrcOp, Trunc, Q, MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are zero while RHS has a non-zero bit
        // there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default: llvm_unreachable("Unknown ICmp predicate!");
          // LHS <u RHS.
          case ICmpInst::ICMP_EQ:
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            return ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_NE:
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            return ConstantInt::getTrue(CI->getContext());

          // LHS is non-negative.  If RHS is negative then LHS >s LHS.  If RHS
          // is non-negative then LHS <s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());
          }
        }
      }
    }

    if (isa<SExtInst>(LHS)) {
      // Turn icmp (sext X), (sext Y) into a compare of X and Y if they have the
      // same type.
      if (SExtInst *RI = dyn_cast<SExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that the predicate does not change.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          Q, MaxRecurse-1))
            return V;
      }
      // Turn icmp (sext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::SExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two sign-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, Trunc, Q, MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are all equal, while RHS has varying
        // bits there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default: llvm_unreachable("Unknown ICmp predicate!");
          case ICmpInst::ICMP_EQ:
            return ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_NE:
            return ConstantInt::getTrue(CI->getContext());

          // If RHS is non-negative then LHS <s RHS.  If RHS is negative then
          // LHS >s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());

          // If LHS is non-negative then LHS <u RHS.  If LHS is negative then
          // LHS >u RHS.
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            // Comparison is true iff the LHS <s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SLT, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              Q, MaxRecurse-1))
                return V;
            break;
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            // Comparison is true iff the LHS >=s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SGE, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              Q, MaxRecurse-1))
                return V;
            break;
          }
        }
      }
    }
  }

  // Special logic for binary operators.
  BinaryOperator *LBO = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *RBO = dyn_cast<BinaryOperator>(RHS);
  if (MaxRecurse && (LBO || RBO)) {
    // Analyze the case when either LHS or RHS is an add instruction.
    Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
    // LHS = A + B (or A and B are null); RHS = C + D (or C and D are null).
    bool NoLHSWrapProblem = false, NoRHSWrapProblem = false;
    if (LBO && LBO->getOpcode() == Instruction::Add) {
      A = LBO->getOperand(0); B = LBO->getOperand(1);
      NoLHSWrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && LBO->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && LBO->hasNoSignedWrap());
    }
    if (RBO && RBO->getOpcode() == Instruction::Add) {
      C = RBO->getOperand(0); D = RBO->getOperand(1);
      NoRHSWrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && RBO->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && RBO->hasNoSignedWrap());
    }

    // icmp (X+Y), X -> icmp Y, 0 for equalities or if there is no overflow.
    if ((A == RHS || B == RHS) && NoLHSWrapProblem)
      if (Value *V = SimplifyICmpInst(Pred, A == RHS ? B : A,
                                      Constant::getNullValue(RHS->getType()),
                                      Q, MaxRecurse-1))
        return V;

    // icmp X, (X+Y) -> icmp 0, Y for equalities or if there is no overflow.
    if ((C == LHS || D == LHS) && NoRHSWrapProblem)
      if (Value *V = SimplifyICmpInst(Pred,
                                      Constant::getNullValue(LHS->getType()),
                                      C == LHS ? D : C, Q, MaxRecurse-1))
        return V;

    // icmp (X+Y), (X+Z) -> icmp Y,Z for equalities or if there is no overflow.
    if (A && C && (A == C || A == D || B == C || B == D) &&
        NoLHSWrapProblem && NoRHSWrapProblem) {
      // Determine Y and Z in the form icmp (X+Y), (X+Z).
      Value *Y, *Z;
      if (A == C) {
        // C + B == C + D  ->  B == D
        Y = B;
        Z = D;
      } else if (A == D) {
        // D + B == C + D  ->  B == C
        Y = B;
        Z = C;
      } else if (B == C) {
        // A + C == C + D  ->  A == D
        Y = A;
        Z = D;
      } else {
        assert(B == D);
        // A + D == C + D  ->  A == C
        Y = A;
        Z = C;
      }
      if (Value *V = SimplifyICmpInst(Pred, Y, Z, Q, MaxRecurse-1))
        return V;
    }
  }

  // icmp pred (or X, Y), X
  if (LBO && match(LBO, m_CombineOr(m_Or(m_Value(), m_Specific(RHS)),
                                    m_Or(m_Specific(RHS), m_Value())))) {
    if (Pred == ICmpInst::ICMP_ULT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_UGE)
      return getTrue(ITy);
  }
  // icmp pred X, (or X, Y)
  if (RBO && match(RBO, m_CombineOr(m_Or(m_Value(), m_Specific(LHS)),
                                    m_Or(m_Specific(LHS), m_Value())))) {
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
  }

  // icmp pred (and X, Y), X
  if (LBO && match(LBO, m_CombineOr(m_And(m_Value(), m_Specific(RHS)),
                                    m_And(m_Specific(RHS), m_Value())))) {
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
  }
  // icmp pred X, (and X, Y)
  if (RBO && match(RBO, m_CombineOr(m_And(m_Value(), m_Specific(LHS)),
                                    m_And(m_Specific(LHS), m_Value())))) {
    if (Pred == ICmpInst::ICMP_UGE)
      return getTrue(ITy);
    if (Pred == ICmpInst::ICMP_ULT)
      return getFalse(ITy);
  }

  // 0 - (zext X) pred C
  if (!CmpInst::isUnsigned(Pred) && match(LHS, m_Neg(m_ZExt(m_Value())))) {
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS)) {
      if (RHSC->getValue().isStrictlyPositive()) {
        if (Pred == ICmpInst::ICMP_SLT)
          return ConstantInt::getTrue(RHSC->getContext());
        if (Pred == ICmpInst::ICMP_SGE)
          return ConstantInt::getFalse(RHSC->getContext());
        if (Pred == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(RHSC->getContext());
        if (Pred == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(RHSC->getContext());
      }
      if (RHSC->getValue().isNonNegative()) {
        if (Pred == ICmpInst::ICMP_SLE)
          return ConstantInt::getTrue(RHSC->getContext());
        if (Pred == ICmpInst::ICMP_SGT)
          return ConstantInt::getFalse(RHSC->getContext());
      }
    }
  }

  // icmp pred (urem X, Y), Y
  if (LBO && match(LBO, m_URem(m_Value(), m_Specific(RHS)))) {
    bool KnownNonNegative, KnownNegative;
    switch (Pred) {
    default:
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(RHS, KnownNonNegative, KnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
      return getFalse(ITy);
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(RHS, KnownNonNegative, KnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
      return getTrue(ITy);
    }
  }

  // icmp pred X, (urem Y, X)
  if (RBO && match(RBO, m_URem(m_Value(), m_Specific(LHS)))) {
    bool KnownNonNegative, KnownNegative;
    switch (Pred) {
    default:
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(LHS, KnownNonNegative, KnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
      return getTrue(ITy);
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(LHS, KnownNonNegative, KnownNegative, Q.DL, 0, Q.AC,
                     Q.CxtI, Q.DT);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
      return getFalse(ITy);
    }
  }

  // x udiv y <=u x.
  if (LBO && match(LBO, m_UDiv(m_Specific(RHS), m_Value()))) {
    // icmp pred (X /u Y), X
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
  }

  // handle:
  //   CI2 << X == CI
  //   CI2 << X != CI
  //
  //   where CI2 is a power of 2 and CI isn't
  if (auto *CI = dyn_cast<ConstantInt>(RHS)) {
    const APInt *CI2Val, *CIVal = &CI->getValue();
    if (LBO && match(LBO, m_Shl(m_APInt(CI2Val), m_Value())) &&
        CI2Val->isPowerOf2()) {
      if (!CIVal->isPowerOf2()) {
        // CI2 << X can equal zero in some circumstances,
        // this simplification is unsafe if CI is zero.
        //
        // We know it is safe if:
        // - The shift is nsw, we can't shift out the one bit.
        // - The shift is nuw, we can't shift out the one bit.
        // - CI2 is one
        // - CI isn't zero
        if (LBO->hasNoSignedWrap() || LBO->hasNoUnsignedWrap() ||
            *CI2Val == 1 || !CI->isZero()) {
          if (Pred == ICmpInst::ICMP_EQ)
            return ConstantInt::getFalse(RHS->getContext());
          if (Pred == ICmpInst::ICMP_NE)
            return ConstantInt::getTrue(RHS->getContext());
        }
      }
      if (CIVal->isSignBit() && *CI2Val == 1) {
        if (Pred == ICmpInst::ICMP_UGT)
          return ConstantInt::getFalse(RHS->getContext());
        if (Pred == ICmpInst::ICMP_ULE)
          return ConstantInt::getTrue(RHS->getContext());
      }
    }
  }

  if (MaxRecurse && LBO && RBO && LBO->getOpcode() == RBO->getOpcode() &&
      LBO->getOperand(1) == RBO->getOperand(1)) {
    switch (LBO->getOpcode()) {
    default: break;
    case Instruction::UDiv:
    case Instruction::LShr:
      if (ICmpInst::isSigned(Pred))
        break;
      // fall-through
    case Instruction::SDiv:
    case Instruction::AShr:
      if (!LBO->isExact() || !RBO->isExact())
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse-1))
        return V;
      break;
    case Instruction::Shl: {
      bool NUW = LBO->hasNoUnsignedWrap() && RBO->hasNoUnsignedWrap();
      bool NSW = LBO->hasNoSignedWrap() && RBO->hasNoSignedWrap();
      if (!NUW && !NSW)
        break;
      if (!NSW && ICmpInst::isSigned(Pred))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse-1))
        return V;
      break;
    }
    }
  }

  // Simplify comparisons involving max/min.
  Value *A, *B;
  CmpInst::Predicate P = CmpInst::BAD_ICMP_PREDICATE;
  CmpInst::Predicate EqP; // Chosen so that "A == max/min(A,B)" iff "A EqP B".

  // Signed variants on "max(a,b)>=a -> true".
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // smax(A, B) pred A.
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_SMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred smax(A, B).
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // smin(A, B) pred A.
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred smin(A, B).
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_SLE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, Q, MaxRecurse-1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_SGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, Q, MaxRecurse-1))
          return V;
      break;
    }
    case CmpInst::ICMP_SGE:
      // Always true.
      return getTrue(ITy);
    case CmpInst::ICMP_SLT:
      // Always false.
      return getFalse(ITy);
    }
  }

  // Unsigned variants on "max(a,b)>=a -> true".
  P = CmpInst::BAD_ICMP_PREDICATE;
  if (match(LHS, m_UMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // umax(A, B) pred A.
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_UMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred umax(A, B).
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // umin(A, B) pred A.
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred umin(A, B).
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_ULE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, Q, MaxRecurse-1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_UGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, Q, MaxRecurse-1))
          return V;
      break;
    }
    case CmpInst::ICMP_UGE:
      // Always true.
      return getTrue(ITy);
    case CmpInst::ICMP_ULT:
      // Always false.
      return getFalse(ITy);
    }
  }

  // Variants on "max(x,y) >= min(x,z)".
  Value *C, *D;
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) &&
      match(RHS, m_SMin(m_Value(C), m_Value(D))) &&
      (A == C || A == D || B == C || B == D)) {
    // max(x, ?) pred min(x, ?).
    if (Pred == CmpInst::ICMP_SGE)
      // Always true.
      return getTrue(ITy);
    if (Pred == CmpInst::ICMP_SLT)
      // Always false.
      return getFalse(ITy);
  } else if (match(LHS, m_SMin(m_Value(A), m_Value(B))) &&
             match(RHS, m_SMax(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // min(x, ?) pred max(x, ?).
    if (Pred == CmpInst::ICMP_SLE)
      // Always true.
      return getTrue(ITy);
    if (Pred == CmpInst::ICMP_SGT)
      // Always false.
      return getFalse(ITy);
  } else if (match(LHS, m_UMax(m_Value(A), m_Value(B))) &&
             match(RHS, m_UMin(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // max(x, ?) pred min(x, ?).
    if (Pred == CmpInst::ICMP_UGE)
      // Always true.
      return getTrue(ITy);
    if (Pred == CmpInst::ICMP_ULT)
      // Always false.
      return getFalse(ITy);
  } else if (match(LHS, m_UMin(m_Value(A), m_Value(B))) &&
             match(RHS, m_UMax(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // min(x, ?) pred max(x, ?).
    if (Pred == CmpInst::ICMP_ULE)
      // Always true.
      return getTrue(ITy);
    if (Pred == CmpInst::ICMP_UGT)
      // Always false.
      return getFalse(ITy);
  }

  // Simplify comparisons of related pointers using a powerful, recursive
  // GEP-walk when we have target data available..
  if (LHS->getType()->isPointerTy())
    if (Constant *C = computePointerICmp(Q.DL, Q.TLI, Pred, LHS, RHS))
      return C;

  if (GetElementPtrInst *GLHS = dyn_cast<GetElementPtrInst>(LHS)) {
    if (GEPOperator *GRHS = dyn_cast<GEPOperator>(RHS)) {
      if (GLHS->getPointerOperand() == GRHS->getPointerOperand() &&
          GLHS->hasAllConstantIndices() && GRHS->hasAllConstantIndices() &&
          (ICmpInst::isEquality(Pred) ||
           (GLHS->isInBounds() && GRHS->isInBounds() &&
            Pred == ICmpInst::getSignedPredicate(Pred)))) {
        // The bases are equal and the indices are constant.  Build a constant
        // expression GEP with the same indices and a null base pointer to see
        // what constant folding can make out of it.
        Constant *Null = Constant::getNullValue(GLHS->getPointerOperandType());
        SmallVector<Value *, 4> IndicesLHS(GLHS->idx_begin(), GLHS->idx_end());
        Constant *NewLHS = ConstantExpr::getGetElementPtr(
            GLHS->getSourceElementType(), Null, IndicesLHS);

        SmallVector<Value *, 4> IndicesRHS(GRHS->idx_begin(), GRHS->idx_end());
        Constant *NewRHS = ConstantExpr::getGetElementPtr(
            GLHS->getSourceElementType(), Null, IndicesRHS);
        return ConstantExpr::getICmp(Pred, NewLHS, NewRHS);
      }
    }
  }

  // If a bit is known to be zero for A and known to be one for B,
  // then A and B cannot be equal.
  if (ICmpInst::isEquality(Pred)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
      uint32_t BitWidth = CI->getBitWidth();
      APInt LHSKnownZero(BitWidth, 0);
      APInt LHSKnownOne(BitWidth, 0);
      computeKnownBits(LHS, LHSKnownZero, LHSKnownOne, Q.DL, /*Depth=*/0, Q.AC,
                       Q.CxtI, Q.DT);
      const APInt &RHSVal = CI->getValue();
      if (((LHSKnownZero & RHSVal) != 0) || ((LHSKnownOne & ~RHSVal) != 0))
        return Pred == ICmpInst::ICMP_EQ
                   ? ConstantInt::getFalse(CI->getContext())
                   : ConstantInt::getTrue(CI->getContext());
    }
  }

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              Instruction *CxtI) {
  return ::SimplifyICmpInst(Predicate, LHS, RHS, Query(DL, TLI, DT, AC, CxtI),
                            RecursionLimit);
}

/// SimplifyFCmpInst - Given operands for an FCmpInst, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               FastMathFlags FMF, const Query &Q,
                               unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isFPPredicate(Pred) && "Not an FP compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, Q.DL, Q.TLI);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  // Fold trivial predicates.
  if (Pred == FCmpInst::FCMP_FALSE)
    return ConstantInt::get(GetCompareTy(LHS), 0);
  if (Pred == FCmpInst::FCMP_TRUE)
    return ConstantInt::get(GetCompareTy(LHS), 1);

  // UNO/ORD predicates can be trivially folded if NaNs are ignored.
  if (FMF.noNaNs()) {
    if (Pred == FCmpInst::FCMP_UNO)
      return ConstantInt::get(GetCompareTy(LHS), 0);
    if (Pred == FCmpInst::FCMP_ORD)
      return ConstantInt::get(GetCompareTy(LHS), 1);
  }

  // fcmp pred x, undef  and  fcmp pred undef, x
  // fold to true if unordered, false if ordered
  if (isa<UndefValue>(LHS) || isa<UndefValue>(RHS)) {
    // Choosing NaN for the undef will always make unordered comparison succeed
    // and ordered comparison fail.
    return ConstantInt::get(GetCompareTy(LHS), CmpInst::isUnordered(Pred));
  }

  // fcmp x,x -> true/false.  Not all compares are foldable.
  if (LHS == RHS) {
    if (CmpInst::isTrueWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 1);
    if (CmpInst::isFalseWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 0);
  }

  // Handle fcmp with constant RHS
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS)) {
    // If the constant is a nan, see if we can fold the comparison based on it.
    if (CFP->getValueAPF().isNaN()) {
      if (FCmpInst::isOrdered(Pred)) // True "if ordered and foo"
        return ConstantInt::getFalse(CFP->getContext());
      assert(FCmpInst::isUnordered(Pred) &&
             "Comparison must be either ordered or unordered!");
      // True if unordered.
      return ConstantInt::getTrue(CFP->getContext());
    }
    // Check whether the constant is an infinity.
    if (CFP->getValueAPF().isInfinity()) {
      if (CFP->getValueAPF().isNegative()) {
        switch (Pred) {
        case FCmpInst::FCMP_OLT:
          // No value is ordered and less than negative infinity.
          return ConstantInt::getFalse(CFP->getContext());
        case FCmpInst::FCMP_UGE:
          // All values are unordered with or at least negative infinity.
          return ConstantInt::getTrue(CFP->getContext());
        default:
          break;
        }
      } else {
        switch (Pred) {
        case FCmpInst::FCMP_OGT:
          // No value is ordered and greater than infinity.
          return ConstantInt::getFalse(CFP->getContext());
        case FCmpInst::FCMP_ULE:
          // All values are unordered with and at most infinity.
          return ConstantInt::getTrue(CFP->getContext());
        default:
          break;
        }
      }
    }
    if (CFP->getValueAPF().isZero()) {
      switch (Pred) {
      case FCmpInst::FCMP_UGE:
        if (CannotBeOrderedLessThanZero(LHS))
          return ConstantInt::getTrue(CFP->getContext());
        break;
      case FCmpInst::FCMP_OLT:
        // X < 0
        if (CannotBeOrderedLessThanZero(LHS))
          return ConstantInt::getFalse(CFP->getContext());
        break;
      default:
        break;
      }
    }
  }

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              FastMathFlags FMF, const DataLayout &DL,
                              const TargetLibraryInfo *TLI,
                              const DominatorTree *DT, AssumptionCache *AC,
                              const Instruction *CxtI) {
  return ::SimplifyFCmpInst(Predicate, LHS, RHS, FMF,
                            Query(DL, TLI, DT, AC, CxtI), RecursionLimit);
}

/// SimplifyWithOpReplaced - See if V simplifies when its operand Op is
/// replaced with RepOp.
static const Value *SimplifyWithOpReplaced(Value *V, Value *Op, Value *RepOp,
                                           const Query &Q,
                                           unsigned MaxRecurse) {
  // Trivial replacement.
  if (V == Op)
    return RepOp;

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return nullptr;

  // If this is a binary operator, try to simplify it with the replaced op.
  if (auto *B = dyn_cast<BinaryOperator>(I)) {
    // Consider:
    //   %cmp = icmp eq i32 %x, 2147483647
    //   %add = add nsw i32 %x, 1
    //   %sel = select i1 %cmp, i32 -2147483648, i32 %add
    //
    // We can't replace %sel with %add unless we strip away the flags.
    if (isa<OverflowingBinaryOperator>(B))
      if (B->hasNoSignedWrap() || B->hasNoUnsignedWrap())
        return nullptr;
    if (isa<PossiblyExactOperator>(B))
      if (B->isExact())
        return nullptr;

    if (MaxRecurse) {
      if (B->getOperand(0) == Op)
        return SimplifyBinOp(B->getOpcode(), RepOp, B->getOperand(1), Q,
                             MaxRecurse - 1);
      if (B->getOperand(1) == Op)
        return SimplifyBinOp(B->getOpcode(), B->getOperand(0), RepOp, Q,
                             MaxRecurse - 1);
    }
  }

  // Same for CmpInsts.
  if (CmpInst *C = dyn_cast<CmpInst>(I)) {
    if (MaxRecurse) {
      if (C->getOperand(0) == Op)
        return SimplifyCmpInst(C->getPredicate(), RepOp, C->getOperand(1), Q,
                               MaxRecurse - 1);
      if (C->getOperand(1) == Op)
        return SimplifyCmpInst(C->getPredicate(), C->getOperand(0), RepOp, Q,
                               MaxRecurse - 1);
    }
  }

  // TODO: We could hand off more cases to instsimplify here.

  // If all operands are constant after substituting Op for RepOp then we can
  // constant fold the instruction.
  if (Constant *CRepOp = dyn_cast<Constant>(RepOp)) {
    // Build a list of all constant operands.
    SmallVector<Constant *, 8> ConstOps;
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      if (I->getOperand(i) == Op)
        ConstOps.push_back(CRepOp);
      else if (Constant *COp = dyn_cast<Constant>(I->getOperand(i)))
        ConstOps.push_back(COp);
      else
        break;
    }

    // All operands were constants, fold it.
    if (ConstOps.size() == I->getNumOperands()) {
      if (CmpInst *C = dyn_cast<CmpInst>(I))
        return ConstantFoldCompareInstOperands(C->getPredicate(), ConstOps[0],
                                               ConstOps[1], Q.DL, Q.TLI);

      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        if (!LI->isVolatile())
          return ConstantFoldLoadFromConstPtr(ConstOps[0], Q.DL);

      return ConstantFoldInstOperands(I->getOpcode(), I->getType(), ConstOps,
                                      Q.DL, Q.TLI);
    }
  }

  return nullptr;
}

/// SimplifySelectInst - Given operands for a SelectInst, see if we can fold
/// the result.  If not, this returns null.
static Value *SimplifySelectInst(Value *CondVal, Value *TrueVal,
                                 Value *FalseVal, const Query &Q,
                                 unsigned MaxRecurse) {
  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (Constant *CB = dyn_cast<Constant>(CondVal)) {
    if (CB->isAllOnesValue())
      return TrueVal;
    if (CB->isNullValue())
      return FalseVal;
  }

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return TrueVal;

  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return TrueVal;
    return FalseVal;
  }
  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return FalseVal;
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return TrueVal;

  if (const auto *ICI = dyn_cast<ICmpInst>(CondVal)) {
    unsigned BitWidth = Q.DL.getTypeSizeInBits(TrueVal->getType());
    ICmpInst::Predicate Pred = ICI->getPredicate();
    Value *CmpLHS = ICI->getOperand(0);
    Value *CmpRHS = ICI->getOperand(1);
    APInt MinSignedValue = APInt::getSignBit(BitWidth);
    Value *X;
    const APInt *Y;
    bool TrueWhenUnset;
    bool IsBitTest = false;
    if (ICmpInst::isEquality(Pred) &&
        match(CmpLHS, m_And(m_Value(X), m_APInt(Y))) &&
        match(CmpRHS, m_Zero())) {
      IsBitTest = true;
      TrueWhenUnset = Pred == ICmpInst::ICMP_EQ;
    } else if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, m_Zero())) {
      X = CmpLHS;
      Y = &MinSignedValue;
      IsBitTest = true;
      TrueWhenUnset = false;
    } else if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, m_AllOnes())) {
      X = CmpLHS;
      Y = &MinSignedValue;
      IsBitTest = true;
      TrueWhenUnset = true;
    }
    if (IsBitTest) {
      const APInt *C;
      // (X & Y) == 0 ? X & ~Y : X  --> X
      // (X & Y) != 0 ? X & ~Y : X  --> X & ~Y
      if (FalseVal == X && match(TrueVal, m_And(m_Specific(X), m_APInt(C))) &&
          *Y == ~*C)
        return TrueWhenUnset ? FalseVal : TrueVal;
      // (X & Y) == 0 ? X : X & ~Y  --> X & ~Y
      // (X & Y) != 0 ? X : X & ~Y  --> X
      if (TrueVal == X && match(FalseVal, m_And(m_Specific(X), m_APInt(C))) &&
          *Y == ~*C)
        return TrueWhenUnset ? FalseVal : TrueVal;

      if (Y->isPowerOf2()) {
        // (X & Y) == 0 ? X | Y : X  --> X | Y
        // (X & Y) != 0 ? X | Y : X  --> X
        if (FalseVal == X && match(TrueVal, m_Or(m_Specific(X), m_APInt(C))) &&
            *Y == *C)
          return TrueWhenUnset ? TrueVal : FalseVal;
        // (X & Y) == 0 ? X : X | Y  --> X
        // (X & Y) != 0 ? X : X | Y  --> X | Y
        if (TrueVal == X && match(FalseVal, m_Or(m_Specific(X), m_APInt(C))) &&
            *Y == *C)
          return TrueWhenUnset ? TrueVal : FalseVal;
      }
    }
    if (ICI->hasOneUse()) {
      const APInt *C;
      if (match(CmpRHS, m_APInt(C))) {
        // X < MIN ? T : F  -->  F
        if (Pred == ICmpInst::ICMP_SLT && C->isMinSignedValue())
          return FalseVal;
        // X < MIN ? T : F  -->  F
        if (Pred == ICmpInst::ICMP_ULT && C->isMinValue())
          return FalseVal;
        // X > MAX ? T : F  -->  F
        if (Pred == ICmpInst::ICMP_SGT && C->isMaxSignedValue())
          return FalseVal;
        // X > MAX ? T : F  -->  F
        if (Pred == ICmpInst::ICMP_UGT && C->isMaxValue())
          return FalseVal;
      }
    }

    // If we have an equality comparison then we know the value in one of the
    // arms of the select. See if substituting this value into the arm and
    // simplifying the result yields the same value as the other arm.
    if (Pred == ICmpInst::ICMP_EQ) {
      if (SimplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, Q, MaxRecurse) ==
              TrueVal ||
          SimplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, Q, MaxRecurse) ==
              TrueVal)
        return FalseVal;
      if (SimplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, Q, MaxRecurse) ==
              FalseVal ||
          SimplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, Q, MaxRecurse) ==
              FalseVal)
        return FalseVal;
    } else if (Pred == ICmpInst::ICMP_NE) {
      if (SimplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, Q, MaxRecurse) ==
              FalseVal ||
          SimplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, Q, MaxRecurse) ==
              FalseVal)
        return TrueVal;
      if (SimplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, Q, MaxRecurse) ==
              TrueVal ||
          SimplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, Q, MaxRecurse) ==
              TrueVal)
        return TrueVal;
    }
  }

  return nullptr;
}

Value *llvm::SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                                const DataLayout &DL,
                                const TargetLibraryInfo *TLI,
                                const DominatorTree *DT, AssumptionCache *AC,
                                const Instruction *CxtI) {
  return ::SimplifySelectInst(Cond, TrueVal, FalseVal,
                              Query(DL, TLI, DT, AC, CxtI), RecursionLimit);
}

/// SimplifyGEPInst - Given operands for an GetElementPtrInst, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyGEPInst(Type *SrcTy, ArrayRef<Value *> Ops,
                              const Query &Q, unsigned) {
  // The type of the GEP pointer operand.
  unsigned AS =
      cast<PointerType>(Ops[0]->getType()->getScalarType())->getAddressSpace();

  // getelementptr P -> P.
  if (Ops.size() == 1)
    return Ops[0];

  // Compute the (pointer) type returned by the GEP instruction.
  Type *LastType = GetElementPtrInst::getIndexedType(SrcTy, Ops.slice(1));
  Type *GEPTy = PointerType::get(LastType, AS);
  if (VectorType *VT = dyn_cast<VectorType>(Ops[0]->getType()))
    GEPTy = VectorType::get(GEPTy, VT->getNumElements());

  if (isa<UndefValue>(Ops[0]))
    return UndefValue::get(GEPTy);

  if (Ops.size() == 2) {
    // getelementptr P, 0 -> P.
    if (match(Ops[1], m_Zero()))
      return Ops[0];

    Type *Ty = SrcTy;
    if (Ty->isSized()) {
      Value *P;
      uint64_t C;
      uint64_t TyAllocSize = Q.DL.getTypeAllocSize(Ty);
      // getelementptr P, N -> P if P points to a type of zero size.
      if (TyAllocSize == 0)
        return Ops[0];

      // The following transforms are only safe if the ptrtoint cast
      // doesn't truncate the pointers.
      if (Ops[1]->getType()->getScalarSizeInBits() ==
          Q.DL.getPointerSizeInBits(AS)) {
        auto PtrToIntOrZero = [GEPTy](Value *P) -> Value * {
          if (match(P, m_Zero()))
            return Constant::getNullValue(GEPTy);
          Value *Temp;
          if (match(P, m_PtrToInt(m_Value(Temp))))
            if (Temp->getType() == GEPTy)
              return Temp;
          return nullptr;
        };

        // getelementptr V, (sub P, V) -> P if P points to a type of size 1.
        if (TyAllocSize == 1 &&
            match(Ops[1], m_Sub(m_Value(P), m_PtrToInt(m_Specific(Ops[0])))))
          if (Value *R = PtrToIntOrZero(P))
            return R;

        // getelementptr V, (ashr (sub P, V), C) -> Q
        // if P points to a type of size 1 << C.
        if (match(Ops[1],
                  m_AShr(m_Sub(m_Value(P), m_PtrToInt(m_Specific(Ops[0]))),
                         m_ConstantInt(C))) &&
            TyAllocSize == 1ULL << C)
          if (Value *R = PtrToIntOrZero(P))
            return R;

        // getelementptr V, (sdiv (sub P, V), C) -> Q
        // if P points to a type of size C.
        if (match(Ops[1],
                  m_SDiv(m_Sub(m_Value(P), m_PtrToInt(m_Specific(Ops[0]))),
                         m_SpecificInt(TyAllocSize))))
          if (Value *R = PtrToIntOrZero(P))
            return R;
      }
    }
  }

  // Check to see if this is constant foldable.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (!isa<Constant>(Ops[i]))
      return nullptr;

  return ConstantExpr::getGetElementPtr(SrcTy, cast<Constant>(Ops[0]),
                                        Ops.slice(1));
}

Value *llvm::SimplifyGEPInst(ArrayRef<Value *> Ops, const DataLayout &DL,
                             const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyGEPInst(
      cast<PointerType>(Ops[0]->getType()->getScalarType())->getElementType(),
      Ops, Query(DL, TLI, DT, AC, CxtI), RecursionLimit);
}

/// SimplifyInsertValueInst - Given operands for an InsertValueInst, see if we
/// can fold the result.  If not, this returns null.
static Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                      ArrayRef<unsigned> Idxs, const Query &Q,
                                      unsigned) {
  if (Constant *CAgg = dyn_cast<Constant>(Agg))
    if (Constant *CVal = dyn_cast<Constant>(Val))
      return ConstantFoldInsertValueInstruction(CAgg, CVal, Idxs);

  // insertvalue x, undef, n -> x
  if (match(Val, m_Undef()))
    return Agg;

  // insertvalue x, (extractvalue y, n), n
  if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(Val))
    if (EV->getAggregateOperand()->getType() == Agg->getType() &&
        EV->getIndices() == Idxs) {
      // insertvalue undef, (extractvalue y, n), n -> y
      if (match(Agg, m_Undef()))
        return EV->getAggregateOperand();

      // insertvalue y, (extractvalue y, n), n -> y
      if (Agg == EV->getAggregateOperand())
        return Agg;
    }

  return nullptr;
}

Value *llvm::SimplifyInsertValueInst(
    Value *Agg, Value *Val, ArrayRef<unsigned> Idxs, const DataLayout &DL,
    const TargetLibraryInfo *TLI, const DominatorTree *DT, AssumptionCache *AC,
    const Instruction *CxtI) {
  return ::SimplifyInsertValueInst(Agg, Val, Idxs, Query(DL, TLI, DT, AC, CxtI),
                                   RecursionLimit);
}

/// SimplifyExtractValueInst - Given operands for an ExtractValueInst, see if we
/// can fold the result.  If not, this returns null.
static Value *SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                       const Query &, unsigned) {
  if (auto *CAgg = dyn_cast<Constant>(Agg))
    return ConstantFoldExtractValueInstruction(CAgg, Idxs);

  // extractvalue x, (insertvalue y, elt, n), n -> elt
  unsigned NumIdxs = Idxs.size();
  for (auto *IVI = dyn_cast<InsertValueInst>(Agg); IVI != nullptr;
       IVI = dyn_cast<InsertValueInst>(IVI->getAggregateOperand())) {
    ArrayRef<unsigned> InsertValueIdxs = IVI->getIndices();
    unsigned NumInsertValueIdxs = InsertValueIdxs.size();
    unsigned NumCommonIdxs = std::min(NumInsertValueIdxs, NumIdxs);
    if (InsertValueIdxs.slice(0, NumCommonIdxs) ==
        Idxs.slice(0, NumCommonIdxs)) {
      if (NumIdxs == NumInsertValueIdxs)
        return IVI->getInsertedValueOperand();
      break;
    }
  }

  return nullptr;
}

Value *llvm::SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                      const DataLayout &DL,
                                      const TargetLibraryInfo *TLI,
                                      const DominatorTree *DT,
                                      AssumptionCache *AC,
                                      const Instruction *CxtI) {
  return ::SimplifyExtractValueInst(Agg, Idxs, Query(DL, TLI, DT, AC, CxtI),
                                    RecursionLimit);
}

/// SimplifyExtractElementInst - Given operands for an ExtractElementInst, see if we
/// can fold the result.  If not, this returns null.
static Value *SimplifyExtractElementInst(Value *Vec, Value *Idx, const Query &,
                                         unsigned) {
  if (auto *CVec = dyn_cast<Constant>(Vec)) {
    if (auto *CIdx = dyn_cast<Constant>(Idx))
      return ConstantFoldExtractElementInstruction(CVec, CIdx);

    // The index is not relevant if our vector is a splat.
    if (auto *Splat = CVec->getSplatValue())
      return Splat;

    if (isa<UndefValue>(Vec))
      return UndefValue::get(Vec->getType()->getVectorElementType());
  }

  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (auto *IdxC = dyn_cast<ConstantInt>(Idx))
    if (Value *Elt = findScalarElement(Vec, IdxC->getZExtValue()))
      return Elt;

  return nullptr;
}

Value *llvm::SimplifyExtractElementInst(
    Value *Vec, Value *Idx, const DataLayout &DL, const TargetLibraryInfo *TLI,
    const DominatorTree *DT, AssumptionCache *AC, const Instruction *CxtI) {
  return ::SimplifyExtractElementInst(Vec, Idx, Query(DL, TLI, DT, AC, CxtI),
                                      RecursionLimit);
}

/// SimplifyPHINode - See if we can fold the given phi.  If not, returns null.
static Value *SimplifyPHINode(PHINode *PN, const Query &Q) {
  // If all of the PHI's incoming values are the same then replace the PHI node
  // with the common value.
  Value *CommonValue = nullptr;
  bool HasUndefInput = false;
  for (Value *Incoming : PN->incoming_values()) {
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PN) continue;
    if (isa<UndefValue>(Incoming)) {
      // Remember that we saw an undef value, but otherwise ignore them.
      HasUndefInput = true;
      continue;
    }
    if (CommonValue && Incoming != CommonValue)
      return nullptr;  // Not the same, bail out.
    CommonValue = Incoming;
  }

  // If CommonValue is null then all of the incoming values were either undef or
  // equal to the phi node itself.
  if (!CommonValue)
    return UndefValue::get(PN->getType());

  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot return X as the result of the PHI node unless it
  // dominates the PHI block.
  if (HasUndefInput)
    return ValueDominatesPHI(CommonValue, PN, Q.DT) ? CommonValue : nullptr;

  return CommonValue;
}

static Value *SimplifyTruncInst(Value *Op, Type *Ty, const Query &Q, unsigned) {
  if (Constant *C = dyn_cast<Constant>(Op))
    return ConstantFoldInstOperands(Instruction::Trunc, Ty, C, Q.DL, Q.TLI);

  return nullptr;
}

Value *llvm::SimplifyTruncInst(Value *Op, Type *Ty, const DataLayout &DL,
                               const TargetLibraryInfo *TLI,
                               const DominatorTree *DT, AssumptionCache *AC,
                               const Instruction *CxtI) {
  return ::SimplifyTruncInst(Op, Ty, Query(DL, TLI, DT, AC, CxtI),
                             RecursionLimit);
}

//=== Helper functions for higher up the class hierarchy.

/// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                            const Query &Q, unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::Add:
    return SimplifyAddInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           Q, MaxRecurse);
  case Instruction::FAdd:
    return SimplifyFAddInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);

  case Instruction::Sub:
    return SimplifySubInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           Q, MaxRecurse);
  case Instruction::FSub:
    return SimplifyFSubInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);

  case Instruction::Mul:  return SimplifyMulInst (LHS, RHS, Q, MaxRecurse);
  case Instruction::FMul:
    return SimplifyFMulInst (LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::SDiv: return SimplifySDivInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::UDiv: return SimplifyUDivInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::FDiv:
      return SimplifyFDivInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::SRem: return SimplifySRemInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::URem: return SimplifyURemInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::FRem:
      return SimplifyFRemInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::Shl:
    return SimplifyShlInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           Q, MaxRecurse);
  case Instruction::LShr:
    return SimplifyLShrInst(LHS, RHS, /*isExact*/false, Q, MaxRecurse);
  case Instruction::AShr:
    return SimplifyAShrInst(LHS, RHS, /*isExact*/false, Q, MaxRecurse);
  case Instruction::And: return SimplifyAndInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::Or:  return SimplifyOrInst (LHS, RHS, Q, MaxRecurse);
  case Instruction::Xor: return SimplifyXorInst(LHS, RHS, Q, MaxRecurse);
  default:
    if (Constant *CLHS = dyn_cast<Constant>(LHS))
      if (Constant *CRHS = dyn_cast<Constant>(RHS)) {
        Constant *COps[] = {CLHS, CRHS};
        return ConstantFoldInstOperands(Opcode, LHS->getType(), COps, Q.DL,
                                        Q.TLI);
      }

    // If the operation is associative, try some generic simplifications.
    if (Instruction::isAssociative(Opcode))
      if (Value *V = SimplifyAssociativeBinOp(Opcode, LHS, RHS, Q, MaxRecurse))
        return V;

    // If the operation is with the result of a select instruction check whether
    // operating on either branch of the select always yields the same value.
    if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
      if (Value *V = ThreadBinOpOverSelect(Opcode, LHS, RHS, Q, MaxRecurse))
        return V;

    // If the operation is with the result of a phi instruction, check whether
    // operating on all incoming values of the phi always yields the same value.
    if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
      if (Value *V = ThreadBinOpOverPHI(Opcode, LHS, RHS, Q, MaxRecurse))
        return V;

    return nullptr;
  }
}

/// SimplifyFPBinOp - Given operands for a BinaryOperator, see if we can
/// fold the result.  If not, this returns null.
/// In contrast to SimplifyBinOp, try to use FastMathFlag when folding the
/// result. In case we don't need FastMathFlags, simply fall to SimplifyBinOp.
static Value *SimplifyFPBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                              const FastMathFlags &FMF, const Query &Q,
                              unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::FAdd:
    return SimplifyFAddInst(LHS, RHS, FMF, Q, MaxRecurse);
  case Instruction::FSub:
    return SimplifyFSubInst(LHS, RHS, FMF, Q, MaxRecurse);
  case Instruction::FMul:
    return SimplifyFMulInst(LHS, RHS, FMF, Q, MaxRecurse);
  default:
    return SimplifyBinOp(Opcode, LHS, RHS, Q, MaxRecurse);
  }
}

Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                           const DataLayout &DL, const TargetLibraryInfo *TLI,
                           const DominatorTree *DT, AssumptionCache *AC,
                           const Instruction *CxtI) {
  return ::SimplifyBinOp(Opcode, LHS, RHS, Query(DL, TLI, DT, AC, CxtI),
                         RecursionLimit);
}

Value *llvm::SimplifyFPBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                             const FastMathFlags &FMF, const DataLayout &DL,
                             const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyFPBinOp(Opcode, LHS, RHS, FMF, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

/// SimplifyCmpInst - Given operands for a CmpInst, see if we can
/// fold the result.
static Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const Query &Q, unsigned MaxRecurse) {
  if (CmpInst::isIntPredicate((CmpInst::Predicate)Predicate))
    return SimplifyICmpInst(Predicate, LHS, RHS, Q, MaxRecurse);
  return SimplifyFCmpInst(Predicate, LHS, RHS, FastMathFlags(), Q, MaxRecurse);
}

Value *llvm::SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                             const DataLayout &DL, const TargetLibraryInfo *TLI,
                             const DominatorTree *DT, AssumptionCache *AC,
                             const Instruction *CxtI) {
  return ::SimplifyCmpInst(Predicate, LHS, RHS, Query(DL, TLI, DT, AC, CxtI),
                           RecursionLimit);
}

static bool IsIdempotent(Intrinsic::ID ID) {
  switch (ID) {
  default: return false;

  // Unary idempotent: f(f(x)) = f(x)
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
    return true;
  }
}

template <typename IterTy>
static Value *SimplifyIntrinsic(Function *F, IterTy ArgBegin, IterTy ArgEnd,
                                const Query &Q, unsigned MaxRecurse) {
  Intrinsic::ID IID = F->getIntrinsicID();
  unsigned NumOperands = std::distance(ArgBegin, ArgEnd);
  Type *ReturnType = F->getReturnType();

  // Binary Ops
  if (NumOperands == 2) {
    Value *LHS = *ArgBegin;
    Value *RHS = *(ArgBegin + 1);
    if (IID == Intrinsic::usub_with_overflow ||
        IID == Intrinsic::ssub_with_overflow) {
      // X - X -> { 0, false }
      if (LHS == RHS)
        return Constant::getNullValue(ReturnType);

      // X - undef -> undef
      // undef - X -> undef
      if (isa<UndefValue>(LHS) || isa<UndefValue>(RHS))
        return UndefValue::get(ReturnType);
    }

    if (IID == Intrinsic::uadd_with_overflow ||
        IID == Intrinsic::sadd_with_overflow) {
      // X + undef -> undef
      if (isa<UndefValue>(RHS))
        return UndefValue::get(ReturnType);
    }

    if (IID == Intrinsic::umul_with_overflow ||
        IID == Intrinsic::smul_with_overflow) {
      // X * 0 -> { 0, false }
      if (match(RHS, m_Zero()))
        return Constant::getNullValue(ReturnType);

      // X * undef -> { 0, false }
      if (match(RHS, m_Undef()))
        return Constant::getNullValue(ReturnType);
    }
  }

  // Perform idempotent optimizations
  if (!IsIdempotent(IID))
    return nullptr;

  // Unary Ops
  if (NumOperands == 1)
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(*ArgBegin))
      if (II->getIntrinsicID() == IID)
        return II;

  return nullptr;
}

template <typename IterTy>
static Value *SimplifyCall(Value *V, IterTy ArgBegin, IterTy ArgEnd,
                           const Query &Q, unsigned MaxRecurse) {
  Type *Ty = V->getType();
  if (PointerType *PTy = dyn_cast<PointerType>(Ty))
    Ty = PTy->getElementType();
  FunctionType *FTy = cast<FunctionType>(Ty);

  // call undef -> undef
  if (isa<UndefValue>(V))
    return UndefValue::get(FTy->getReturnType());

  Function *F = dyn_cast<Function>(V);
  if (!F)
    return nullptr;

  if (F->isIntrinsic())
    if (Value *Ret = SimplifyIntrinsic(F, ArgBegin, ArgEnd, Q, MaxRecurse))
      return Ret;

  if (!canConstantFoldCallTo(F))
    return nullptr;

  SmallVector<Constant *, 4> ConstantArgs;
  ConstantArgs.reserve(ArgEnd - ArgBegin);
  for (IterTy I = ArgBegin, E = ArgEnd; I != E; ++I) {
    Constant *C = dyn_cast<Constant>(*I);
    if (!C)
      return nullptr;
    ConstantArgs.push_back(C);
  }

  return ConstantFoldCall(F, ConstantArgs, Q.TLI);
}

Value *llvm::SimplifyCall(Value *V, User::op_iterator ArgBegin,
                          User::op_iterator ArgEnd, const DataLayout &DL,
                          const TargetLibraryInfo *TLI, const DominatorTree *DT,
                          AssumptionCache *AC, const Instruction *CxtI) {
  return ::SimplifyCall(V, ArgBegin, ArgEnd, Query(DL, TLI, DT, AC, CxtI),
                        RecursionLimit);
}

Value *llvm::SimplifyCall(Value *V, ArrayRef<Value *> Args,
                          const DataLayout &DL, const TargetLibraryInfo *TLI,
                          const DominatorTree *DT, AssumptionCache *AC,
                          const Instruction *CxtI) {
  return ::SimplifyCall(V, Args.begin(), Args.end(),
                        Query(DL, TLI, DT, AC, CxtI), RecursionLimit);
}

// HLSL Change - Begin
// Copied CastInst simplification from LLVM 8

static
Constant *foldConstVectorToAPInt(APInt &Result, Type *DestTy,
                                        Constant *C, Type *SrcEltTy,
                                        unsigned NumSrcElts,
                                        const DataLayout &DL) {
  // Now that we know that the input value is a vector of integers, just shift
  // and insert them into our result.
  unsigned BitShift = DL.getTypeSizeInBits(SrcEltTy);
  for (unsigned i = 0; i != NumSrcElts; ++i) {
    Constant *Element;
    if (DL.isLittleEndian())
      Element = C->getAggregateElement(NumSrcElts - i - 1);
    else
      Element = C->getAggregateElement(i);

    if (Element && isa<UndefValue>(Element)) {
      Result <<= BitShift;
      continue;
    }

    auto *ElementCI = dyn_cast_or_null<ConstantInt>(Element);
    if (!ElementCI)
      return ConstantExpr::getBitCast(C, DestTy);

    Result <<= BitShift;
    Result |= ElementCI->getValue().zextOrSelf(Result.getBitWidth());
  }

  return nullptr;
}

/// Constant fold bitcast, symbolically evaluating it with DataLayout.
/// This always returns a non-null constant, but it may be a
/// ConstantExpr if unfoldable.
static
Constant *FoldBitCast(Constant *C, Type *DestTy, const DataLayout &DL) {
  // Catch the obvious splat cases.
  if (C->isNullValue() && !DestTy->isX86_MMXTy())
    return Constant::getNullValue(DestTy);
  if (C->isAllOnesValue() && !DestTy->isX86_MMXTy() &&
      !DestTy->isPtrOrPtrVectorTy()) // Don't get ones for ptr types!
    return Constant::getAllOnesValue(DestTy);

  if (auto *VTy = dyn_cast<VectorType>(C->getType())) {
    // Handle a vector->scalar integer/fp cast.
    if (isa<IntegerType>(DestTy) || DestTy->isFloatingPointTy()) {
      unsigned NumSrcElts = VTy->getNumElements();
      Type *SrcEltTy = VTy->getElementType();

      // If the vector is a vector of floating point, convert it to vector of int
      // to simplify things.
      if (SrcEltTy->isFloatingPointTy()) {
        unsigned FPWidth = SrcEltTy->getPrimitiveSizeInBits();
        Type *SrcIVTy =
          VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumSrcElts);
        // Ask IR to do the conversion now that #elts line up.
        C = ConstantExpr::getBitCast(C, SrcIVTy);
      }

      APInt Result(DL.getTypeSizeInBits(DestTy), 0);
      if (Constant *CE = foldConstVectorToAPInt(Result, DestTy, C,
                                                SrcEltTy, NumSrcElts, DL))
        return CE;

      if (isa<IntegerType>(DestTy))
        return ConstantInt::get(DestTy, Result);

      APFloat FP(DestTy->getFltSemantics(), Result);
      return ConstantFP::get(DestTy->getContext(), FP);
    }
  }

  // The code below only handles casts to vectors currently.
  auto *DestVTy = dyn_cast<VectorType>(DestTy);
  if (!DestVTy)
    return ConstantExpr::getBitCast(C, DestTy);

  // If this is a scalar -> vector cast, convert the input into a <1 x scalar>
  // vector so the code below can handle it uniformly.
  if (isa<ConstantFP>(C) || isa<ConstantInt>(C)) {
    Constant *Ops = C; // don't take the address of C!
    return FoldBitCast(ConstantVector::get(Ops), DestTy, DL);
  }

  // If this is a bitcast from constant vector -> vector, fold it.
  if (!isa<ConstantDataVector>(C) && !isa<ConstantVector>(C))
    return ConstantExpr::getBitCast(C, DestTy);

  // If the element types match, IR can fold it.
  unsigned NumDstElt = DestVTy->getNumElements();
  unsigned NumSrcElt = C->getType()->getVectorNumElements();
  if (NumDstElt == NumSrcElt)
    return ConstantExpr::getBitCast(C, DestTy);

  Type *SrcEltTy = C->getType()->getVectorElementType();
  Type *DstEltTy = DestVTy->getElementType();

  // Otherwise, we're changing the number of elements in a vector, which
  // requires endianness information to do the right thing.  For example,
  //    bitcast (<2 x i64> <i64 0, i64 1> to <4 x i32>)
  // folds to (little endian):
  //    <4 x i32> <i32 0, i32 0, i32 1, i32 0>
  // and to (big endian):
  //    <4 x i32> <i32 0, i32 0, i32 0, i32 1>

  // First thing is first.  We only want to think about integer here, so if
  // we have something in FP form, recast it as integer.
  if (DstEltTy->isFloatingPointTy()) {
    // Fold to an vector of integers with same size as our FP type.
    unsigned FPWidth = DstEltTy->getPrimitiveSizeInBits();
    Type *DestIVTy =
      VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumDstElt);
    // Recursively handle this integer conversion, if possible.
    C = FoldBitCast(C, DestIVTy, DL);

    // Finally, IR can handle this now that #elts line up.
    return ConstantExpr::getBitCast(C, DestTy);
  }

  // Okay, we know the destination is integer, if the input is FP, convert
  // it to integer first.
  if (SrcEltTy->isFloatingPointTy()) {
    unsigned FPWidth = SrcEltTy->getPrimitiveSizeInBits();
    Type *SrcIVTy =
      VectorType::get(IntegerType::get(C->getContext(), FPWidth), NumSrcElt);
    // Ask IR to do the conversion now that #elts line up.
    C = ConstantExpr::getBitCast(C, SrcIVTy);
    // If IR wasn't able to fold it, bail out.
    if (!isa<ConstantVector>(C) &&  // FIXME: Remove ConstantVector.
        !isa<ConstantDataVector>(C))
      return C;
  }

  // Now we know that the input and output vectors are both integer vectors
  // of the same size, and that their #elements is not the same.  Do the
  // conversion here, which depends on whether the input or output has
  // more elements.
  bool isLittleEndian = DL.isLittleEndian();

  SmallVector<Constant*, 32> Result;
  if (NumDstElt < NumSrcElt) {
    // Handle: bitcast (<4 x i32> <i32 0, i32 1, i32 2, i32 3> to <2 x i64>)
    Constant *Zero = Constant::getNullValue(DstEltTy);
    unsigned Ratio = NumSrcElt/NumDstElt;
    unsigned SrcBitSize = SrcEltTy->getPrimitiveSizeInBits();
    unsigned SrcElt = 0;
    for (unsigned i = 0; i != NumDstElt; ++i) {
      // Build each element of the result.
      Constant *Elt = Zero;
      unsigned ShiftAmt = isLittleEndian ? 0 : SrcBitSize*(Ratio-1);
      for (unsigned j = 0; j != Ratio; ++j) {
        Constant *Src = C->getAggregateElement(SrcElt++);
        if (Src && isa<UndefValue>(Src))
          Src = Constant::getNullValue(C->getType()->getVectorElementType());
        else
          Src = dyn_cast_or_null<ConstantInt>(Src);
        if (!Src)  // Reject constantexpr elements.
          return ConstantExpr::getBitCast(C, DestTy);

        // Zero extend the element to the right size.
        Src = ConstantExpr::getZExt(Src, Elt->getType());

        // Shift it to the right place, depending on endianness.
        Src = ConstantExpr::getShl(Src,
                                   ConstantInt::get(Src->getType(), ShiftAmt));
        ShiftAmt += isLittleEndian ? SrcBitSize : -SrcBitSize;

        // Mix it in.
        Elt = ConstantExpr::getOr(Elt, Src);
      }
      Result.push_back(Elt);
    }
    return ConstantVector::get(Result);
  }

  // Handle: bitcast (<2 x i64> <i64 0, i64 1> to <4 x i32>)
  unsigned Ratio = NumDstElt/NumSrcElt;
  unsigned DstBitSize = DL.getTypeSizeInBits(DstEltTy);

  // Loop over each source value, expanding into multiple results.
  for (unsigned i = 0; i != NumSrcElt; ++i) {
    auto *Element = C->getAggregateElement(i);

    if (!Element) // Reject constantexpr elements.
      return ConstantExpr::getBitCast(C, DestTy);

    if (isa<UndefValue>(Element)) {
      // Correctly Propagate undef values.
      Result.append(Ratio, UndefValue::get(DstEltTy));
      continue;
    }

    auto *Src = dyn_cast<ConstantInt>(Element);
    if (!Src)
      return ConstantExpr::getBitCast(C, DestTy);

    unsigned ShiftAmt = isLittleEndian ? 0 : DstBitSize*(Ratio-1);
    for (unsigned j = 0; j != Ratio; ++j) {
      // Shift the piece of the value into the right place, depending on
      // endianness.
      Constant *Elt = ConstantExpr::getLShr(Src,
                                  ConstantInt::get(Src->getType(), ShiftAmt));
      ShiftAmt += isLittleEndian ? DstBitSize : -DstBitSize;

      // Truncate the element to an integer with the same pointer size and
      // convert the element back to a pointer using a inttoptr.
      if (DstEltTy->isPointerTy()) {
        IntegerType *DstIntTy = Type::getIntNTy(C->getContext(), DstBitSize);
        Constant *CE = ConstantExpr::getTrunc(Elt, DstIntTy);
        Result.push_back(ConstantExpr::getIntToPtr(CE, DstEltTy));
        continue;
      }

      // Truncate and remember this piece.
      Result.push_back(ConstantExpr::getTrunc(Elt, DstEltTy));
    }
  }

  return ConstantVector::get(Result);
}

static
Constant *ConstantFoldCastOperand(unsigned Opcode, Constant *C,
                                        Type *DestTy, const DataLayout &DL) {
  assert(Instruction::isCast(Opcode));
  switch (Opcode) {
  default:
    llvm_unreachable("Missing case");
  case Instruction::PtrToInt:
    // If the input is a inttoptr, eliminate the pair.  This requires knowing
    // the width of a pointer, so it can't be done in ConstantExpr::getCast.
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::IntToPtr) {
        Constant *Input = CE->getOperand(0);
        unsigned InWidth = Input->getType()->getScalarSizeInBits();
        unsigned PtrWidth = DL.getPointerTypeSizeInBits(CE->getType());
        if (PtrWidth < InWidth) {
          Constant *Mask =
            ConstantInt::get(CE->getContext(),
                             APInt::getLowBitsSet(InWidth, PtrWidth));
          Input = ConstantExpr::getAnd(Input, Mask);
        }
        // Do a zext or trunc to get to the dest size.
        return ConstantExpr::getIntegerCast(Input, DestTy, false);
      }
    }
    return ConstantExpr::getCast(Opcode, C, DestTy);
  case Instruction::IntToPtr:
    // If the input is a ptrtoint, turn the pair into a ptr to ptr bitcast if
    // the int size is >= the ptr size and the address spaces are the same.
    // This requires knowing the width of a pointer, so it can't be done in
    // ConstantExpr::getCast.
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::PtrToInt) {
        Constant *SrcPtr = CE->getOperand(0);
        unsigned SrcPtrSize = DL.getPointerTypeSizeInBits(SrcPtr->getType());
        unsigned MidIntSize = CE->getType()->getScalarSizeInBits();

        if (MidIntSize >= SrcPtrSize) {
          unsigned SrcAS = SrcPtr->getType()->getPointerAddressSpace();
          if (SrcAS == DestTy->getPointerAddressSpace())
            return FoldBitCast(CE->getOperand(0), DestTy, DL);
        }
      }
    }

    return ConstantExpr::getCast(Opcode, C, DestTy);
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::AddrSpaceCast:
      return ConstantExpr::getCast(Opcode, C, DestTy);
  case Instruction::BitCast:
    return FoldBitCast(C, DestTy, DL);
  }
}

static Value *SimplifyCastInst(unsigned CastOpc, Value *Op,
                               Type *Ty, const DataLayout &DL) {
  if (auto *C = dyn_cast<Constant>(Op))
    return ConstantFoldCastOperand(CastOpc, C, Ty, DL);

  if (auto *CI = dyn_cast<CastInst>(Op)) {
    auto *Src = CI->getOperand(0);
    Type *SrcTy = Src->getType();
    Type *MidTy = CI->getType();
    Type *DstTy = Ty;
    if (Src->getType() == Ty) {
      auto FirstOp = static_cast<Instruction::CastOps>(CI->getOpcode());
      auto SecondOp = static_cast<Instruction::CastOps>(CastOpc);
      Type *SrcIntPtrTy =
          SrcTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(SrcTy) : nullptr;
      Type *MidIntPtrTy =
          MidTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(MidTy) : nullptr;
      Type *DstIntPtrTy =
          DstTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(DstTy) : nullptr;
      if (CastInst::isEliminableCastPair(FirstOp, SecondOp, SrcTy, MidTy, DstTy,
                                         SrcIntPtrTy, MidIntPtrTy,
                                         DstIntPtrTy) == Instruction::BitCast)
        return Src;
    }
  }

  // bitcast x -> x
  if (CastOpc == Instruction::BitCast)
    if (Op->getType() == Ty)
      return Op;

  return nullptr;
}

Value *llvm::SimplifyCastInst(unsigned CastOpc, Value *Op,
                              Type *Ty, const DataLayout &DL) {
  return ::SimplifyCastInst(CastOpc, Op, Ty, DL);
}

// HLSL Change - End

/// SimplifyInstruction - See if we can compute a simplified version of this
/// instruction.  If not, this returns null.
Value *llvm::SimplifyInstruction(Instruction *I, const DataLayout &DL,
                                 const TargetLibraryInfo *TLI,
                                 const DominatorTree *DT, AssumptionCache *AC) {
  Value *Result;

  switch (I->getOpcode()) {
  default:
    Result = ConstantFoldInstruction(I, DL, TLI);
    break;
  case Instruction::FAdd:
    Result = SimplifyFAddInst(I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::Add:
    Result = SimplifyAddInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(), DL,
                             TLI, DT, AC, I);
    break;
  case Instruction::FSub:
    Result = SimplifyFSubInst(I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::Sub:
    Result = SimplifySubInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(), DL,
                             TLI, DT, AC, I);
    break;
  case Instruction::FMul:
    Result = SimplifyFMulInst(I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::Mul:
    Result =
        SimplifyMulInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT, AC, I);
    break;
  case Instruction::SDiv:
    Result = SimplifySDivInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::UDiv:
    Result = SimplifyUDivInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::FDiv:
    Result = SimplifyFDivInst(I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::SRem:
    Result = SimplifySRemInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::URem:
    Result = SimplifyURemInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::FRem:
    Result = SimplifyFRemInst(I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::Shl:
    Result = SimplifyShlInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(), DL,
                             TLI, DT, AC, I);
    break;
  case Instruction::LShr:
    Result = SimplifyLShrInst(I->getOperand(0), I->getOperand(1),
                              cast<BinaryOperator>(I)->isExact(), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::AShr:
    Result = SimplifyAShrInst(I->getOperand(0), I->getOperand(1),
                              cast<BinaryOperator>(I)->isExact(), DL, TLI, DT,
                              AC, I);
    break;
  case Instruction::And:
    Result =
        SimplifyAndInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT, AC, I);
    break;
  case Instruction::Or:
    Result =
        SimplifyOrInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT, AC, I);
    break;
  case Instruction::Xor:
    Result =
        SimplifyXorInst(I->getOperand(0), I->getOperand(1), DL, TLI, DT, AC, I);
    break;
  case Instruction::ICmp:
    Result =
        SimplifyICmpInst(cast<ICmpInst>(I)->getPredicate(), I->getOperand(0),
                         I->getOperand(1), DL, TLI, DT, AC, I);
    break;
  case Instruction::FCmp:
    Result = SimplifyFCmpInst(cast<FCmpInst>(I)->getPredicate(),
                              I->getOperand(0), I->getOperand(1),
                              I->getFastMathFlags(), DL, TLI, DT, AC, I);
    break;
  case Instruction::Select:
    Result = SimplifySelectInst(I->getOperand(0), I->getOperand(1),
                                I->getOperand(2), DL, TLI, DT, AC, I);
    break;
  case Instruction::GetElementPtr: {
    SmallVector<Value*, 8> Ops(I->op_begin(), I->op_end());
    Result = SimplifyGEPInst(Ops, DL, TLI, DT, AC, I);
    break;
  }
  case Instruction::InsertValue: {
    InsertValueInst *IV = cast<InsertValueInst>(I);
    Result = SimplifyInsertValueInst(IV->getAggregateOperand(),
                                     IV->getInsertedValueOperand(),
                                     IV->getIndices(), DL, TLI, DT, AC, I);
    break;
  }
  case Instruction::ExtractValue: {
    auto *EVI = cast<ExtractValueInst>(I);
    Result = SimplifyExtractValueInst(EVI->getAggregateOperand(),
                                      EVI->getIndices(), DL, TLI, DT, AC, I);
    break;
  }
  case Instruction::ExtractElement: {
    auto *EEI = cast<ExtractElementInst>(I);
    Result = SimplifyExtractElementInst(
        EEI->getVectorOperand(), EEI->getIndexOperand(), DL, TLI, DT, AC, I);
    break;
  }
  case Instruction::PHI:
    Result = SimplifyPHINode(cast<PHINode>(I), Query(DL, TLI, DT, AC, I));
    break;
  case Instruction::Call: {
    CallSite CS(cast<CallInst>(I));
    // HLSL Change Begin - simplify dxil calls.
    if (Function *Callee = CS.getCalledFunction()) {
      if (hlsl::CanSimplify(Callee)) {
        SmallVector<Value *, 4> Args(CS.arg_begin(), CS.arg_end());
        if (Value *DxilResult = hlsl::SimplifyDxilCall(CS.getCalledFunction(), Args, I, /* MayInsert */ true)) {
          Result = DxilResult;
          break;
        }
      }
    }
    // HLSL Change End.
    Result = SimplifyCall(CS.getCalledValue(), CS.arg_begin(), CS.arg_end(), DL,
                          TLI, DT, AC, I);
    break;
  }
  case Instruction::Trunc:
    Result =
        SimplifyTruncInst(I->getOperand(0), I->getType(), DL, TLI, DT, AC, I);
    break;
  }

  /// If called on unreachable code, the above logic may report that the
  /// instruction simplified to itself.  Make life easier for users by
  /// detecting that case here, returning a safe value instead.
  return Result == I ? UndefValue::get(I->getType()) : Result;
}

/// \brief Implementation of recursive simplification through an instructions
/// uses.
///
/// This is the common implementation of the recursive simplification routines.
/// If we have a pre-simplified value in 'SimpleV', that is forcibly used to
/// replace the instruction 'I'. Otherwise, we simply add 'I' to the list of
/// instructions to process and attempt to simplify it using
/// InstructionSimplify.
///
/// This routine returns 'true' only when *it* simplifies something. The passed
/// in simplified value does not count toward this.
static bool replaceAndRecursivelySimplifyImpl(Instruction *I, Value *SimpleV,
                                              const TargetLibraryInfo *TLI,
                                              const DominatorTree *DT,
                                              AssumptionCache *AC) {
  bool Simplified = false;
  SmallSetVector<Instruction *, 8> Worklist;
  const DataLayout &DL = I->getModule()->getDataLayout();

  // If we have an explicit value to collapse to, do that round of the
  // simplification loop by hand initially.
  if (SimpleV) {
    for (User *U : I->users())
      if (U != I)
        Worklist.insert(cast<Instruction>(U));

    // Replace the instruction with its simplified value.
    I->replaceAllUsesWith(SimpleV);

    // Gracefully handle edge cases where the instruction is not wired into any
    // parent block.
    if (I->getParent())
      I->eraseFromParent();
  } else {
    Worklist.insert(I);
  }

  // Note that we must test the size on each iteration, the worklist can grow.
  for (unsigned Idx = 0; Idx != Worklist.size(); ++Idx) {
    I = Worklist[Idx];

    // See if this instruction simplifies.
    SimpleV = SimplifyInstruction(I, DL, TLI, DT, AC);
    if (!SimpleV)
      continue;

    Simplified = true;

    // Stash away all the uses of the old instruction so we can check them for
    // recursive simplifications after a RAUW. This is cheaper than checking all
    // uses of To on the recursive step in most cases.
    for (User *U : I->users())
      Worklist.insert(cast<Instruction>(U));

    // Replace the instruction with its simplified value.
    I->replaceAllUsesWith(SimpleV);

    // Gracefully handle edge cases where the instruction is not wired into any
    // parent block.
    if (I->getParent())
      I->eraseFromParent();
  }
  return Simplified;
}

bool llvm::recursivelySimplifyInstruction(Instruction *I,
                                          const TargetLibraryInfo *TLI,
                                          const DominatorTree *DT,
                                          AssumptionCache *AC) {
  return replaceAndRecursivelySimplifyImpl(I, nullptr, TLI, DT, AC);
}

bool llvm::replaceAndRecursivelySimplify(Instruction *I, Value *SimpleV,
                                         const TargetLibraryInfo *TLI,
                                         const DominatorTree *DT,
                                         AssumptionCache *AC) {
  assert(I != SimpleV && "replaceAndRecursivelySimplify(X,X) is not valid!");
  assert(SimpleV && "Must provide a simplified value.");
  return replaceAndRecursivelySimplifyImpl(I, SimpleV, TLI, DT, AC);
}
