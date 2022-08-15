//===- llvm/Analysis/DivergenceAnalysis.h - Divergence Analysis -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The divergence analysis is an LLVM pass which can be used to find out
// if a branch instruction in a GPU program is divergent or not. It can help
// branch optimizations such as jump threading and loop unswitching to make
// better decisions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {
class Value;
class DivergenceAnalysis : public FunctionPass {
public:
  static char ID;

  DivergenceAnalysis() : FunctionPass(ID) {
    initializeDivergenceAnalysisPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;

  // Print all divergent branches in the function.
  void print(raw_ostream &OS, const Module *) const override;

  // Returns true if V is divergent.
  bool isDivergent(const Value *V) const { return DivergentValues.count(V); }

  // Returns true if V is uniform/non-divergent.
  bool isUniform(const Value *V) const { return !isDivergent(V); }

private:
  // Stores all divergent values.
  DenseSet<const Value *> DivergentValues;
};
} // End llvm namespace
