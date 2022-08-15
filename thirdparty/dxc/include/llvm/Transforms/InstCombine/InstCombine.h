//===- InstCombine.h - InstCombine pass -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the primary interface to the instcombine pass. This pass
/// is suitable for use in the new pass manager. For a pass that works with the
/// legacy pass manager, please look for \c createInstructionCombiningPass() in
/// Scalar.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINE_H
#define LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINE_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"

namespace llvm {

class InstCombinePass {
  InstCombineWorklist Worklist;

public:
  static StringRef name() { return "InstCombinePass"; }

  // Explicitly define constructors for MSVC.
  InstCombinePass() {}
  InstCombinePass(InstCombinePass &&Arg) : Worklist(std::move(Arg.Worklist)) {}
  InstCombinePass &operator=(InstCombinePass &&RHS) {
    Worklist = std::move(RHS.Worklist);
    return *this;
  }

  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);
};

}

#endif
