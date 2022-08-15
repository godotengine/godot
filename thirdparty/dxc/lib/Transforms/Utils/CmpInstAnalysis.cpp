//===- CmpInstAnalysis.cpp - Utils to help fold compares ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file holds routines to help analyse compare instructions
// and fold them into constants or other compare instructions
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CmpInstAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

/// getICmpCode - Encode a icmp predicate into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Note that this is only valid if the first and second predicates have the
/// same sign. Is illegal to do: (A u< B) | (A s> B)
///
/// Three bits are used to represent the condition, as follows:
///   0  A > B
///   1  A == B
///   2  A < B
///
/// <=>  Value  Definition
/// 000     0   Always false
/// 001     1   A >  B
/// 010     2   A == B
/// 011     3   A >= B
/// 100     4   A <  B
/// 101     5   A != B
/// 110     6   A <= B
/// 111     7   Always true
///
unsigned llvm::getICmpCode(const ICmpInst *ICI, bool InvertPred) {
  ICmpInst::Predicate Pred = InvertPred ? ICI->getInversePredicate()
                                        : ICI->getPredicate();
  switch (Pred) {
      // False -> 0
    case ICmpInst::ICMP_UGT: return 1;  // 001
    case ICmpInst::ICMP_SGT: return 1;  // 001
    case ICmpInst::ICMP_EQ:  return 2;  // 010
    case ICmpInst::ICMP_UGE: return 3;  // 011
    case ICmpInst::ICMP_SGE: return 3;  // 011
    case ICmpInst::ICMP_ULT: return 4;  // 100
    case ICmpInst::ICMP_SLT: return 4;  // 100
    case ICmpInst::ICMP_NE:  return 5;  // 101
    case ICmpInst::ICMP_ULE: return 6;  // 110
    case ICmpInst::ICMP_SLE: return 6;  // 110
      // True -> 7
    default:
      llvm_unreachable("Invalid ICmp predicate!");
  }
}

/// getICmpValue - This is the complement of getICmpCode, which turns an
/// opcode and two operands into either a constant true or false, or the
/// predicate for a new ICmp instruction. The sign is passed in to determine
/// which kind of predicate to use in the new icmp instruction.
/// Non-NULL return value will be a true or false constant.
/// NULL return means a new ICmp is needed.  The predicate for which is
/// output in NewICmpPred.
Value *llvm::getICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                          CmpInst::Predicate &NewICmpPred) {
  switch (Code) {
    default: llvm_unreachable("Illegal ICmp code!");
    case 0: // False.
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    case 1: NewICmpPred = Sign ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT; break;
    case 2: NewICmpPred = ICmpInst::ICMP_EQ; break;
    case 3: NewICmpPred = Sign ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE; break;
    case 4: NewICmpPred = Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT; break;
    case 5: NewICmpPred = ICmpInst::ICMP_NE; break;
    case 6: NewICmpPred = Sign ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE; break;
    case 7: // True.
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
  }
  return nullptr;
}

/// PredicatesFoldable - Return true if both predicates match sign or if at
/// least one of them is an equality comparison (which is signless).
bool llvm::PredicatesFoldable(ICmpInst::Predicate p1, ICmpInst::Predicate p2) {
  return (CmpInst::isSigned(p1) == CmpInst::isSigned(p2)) ||
         (CmpInst::isSigned(p1) && ICmpInst::isEquality(p2)) ||
         (CmpInst::isSigned(p2) && ICmpInst::isEquality(p1));
}
