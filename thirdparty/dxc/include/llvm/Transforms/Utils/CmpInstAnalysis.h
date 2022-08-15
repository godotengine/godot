//===-- CmpInstAnalysis.h - Utils to help fold compare insts ----*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_UTILS_CMPINSTANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_CMPINSTANALYSIS_H

#include "llvm/IR/InstrTypes.h"

namespace llvm {
  class ICmpInst;
  class Value;

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
  unsigned getICmpCode(const ICmpInst *ICI, bool InvertPred = false);

  /// getICmpValue - This is the complement of getICmpCode, which turns an
  /// opcode and two operands into either a constant true or false, or the
  /// predicate for a new ICmp instruction. The sign is passed in to determine
  /// which kind of predicate to use in the new icmp instruction.
  /// Non-NULL return value will be a true or false constant.
  /// NULL return means a new ICmp is needed.  The predicate for which is
  /// output in NewICmpPred.
  Value *getICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                      CmpInst::Predicate &NewICmpPred);

  /// PredicatesFoldable - Return true if both predicates match sign or if at
  /// least one of them is an equality comparison (which is signless).
  bool PredicatesFoldable(CmpInst::Predicate p1, CmpInst::Predicate p2);

} // end namespace llvm

#endif
