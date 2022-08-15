//===- llvm/Analysis/ScalarEvolutionNormalization.h - See below -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for working with "normalized" ScalarEvolution
// expressions.
//
// The following example illustrates post-increment uses and how normalized
// expressions help.
//
//   for (i=0; i!=n; ++i) {
//     ...
//   }
//   use(i);
//
// While the expression for most uses of i inside the loop is {0,+,1}<%L>, the
// expression for the use of i outside the loop is {1,+,1}<%L>, since i is
// incremented at the end of the loop body. This is inconveient, since it
// suggests that we need two different induction variables, one that starts
// at 0 and one that starts at 1. We'd prefer to be able to think of these as
// the same induction variable, with uses inside the loop using the
// "pre-incremented" value, and uses after the loop using the
// "post-incremented" value.
//
// Expressions for post-incremented uses are represented as an expression
// paired with a set of loops for which the expression is in "post-increment"
// mode (there may be multiple loops).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTIONNORMALIZATION_H
#define LLVM_ANALYSIS_SCALAREVOLUTIONNORMALIZATION_H

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {

class Instruction;
class DominatorTree;
class Loop;
class ScalarEvolution;
class SCEV;
class Value;

/// TransformKind - Different types of transformations that
/// TransformForPostIncUse can do.
enum TransformKind {
  /// Normalize - Normalize according to the given loops.
  Normalize,
  /// NormalizeAutodetect - Detect post-inc opportunities on new expressions,
  /// update the given loop set, and normalize.
  NormalizeAutodetect,
  /// Denormalize - Perform the inverse transform on the expression with the
  /// given loop set.
  Denormalize
};

/// PostIncLoopSet - A set of loops.
typedef SmallPtrSet<const Loop *, 2> PostIncLoopSet;

/// TransformForPostIncUse - Transform the given expression according to the
/// given transformation kind.
const SCEV *TransformForPostIncUse(TransformKind Kind,
                                   const SCEV *S,
                                   Instruction *User,
                                   Value *OperandValToReplace,
                                   PostIncLoopSet &Loops,
                                   ScalarEvolution &SE,
                                   DominatorTree &DT);

}

#endif
