//===-- DxilSimplify.h - Simplify Dxil operations ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//===----------------------------------------------------------------------===//
//
// This file declares routines for simplify dxil intrinsics when some operands
// are constants.
//
// We hook into the llvm::SimplifyInstruction so the function
// interfaces are dictated by what llvm provides.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_HLSLDXILSIMPLIFY_H
#define LLVM_ANALYSIS_HLSLDXILSIMPLIFY_H
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class Function;
class Instruction;
class Value;
} // namespace llvm

namespace hlsl {

/// \brief Given a function and set of arguments, see if we can fold the
/// result as dxil operation.
///
/// If this call could not be simplified returns null.
llvm::Value *SimplifyDxilCall(llvm::Function *F,
                              llvm::ArrayRef<llvm::Value *> Args,
                              llvm::Instruction *I,
                              bool MayInsert);

/// CanSimplify
/// Return true on dxil operation function which can be simplified.
bool CanSimplify(const llvm::Function *F);
} // namespace hlsl

#endif
