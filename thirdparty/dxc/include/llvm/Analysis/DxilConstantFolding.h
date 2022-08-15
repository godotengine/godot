//===-- DxilConstantFolding.h - Constant folding for Dxil ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding dxil intrinsics into constants when
// all operands are constants.
//
// We hook into the LLVM routines for constant folding so the function
// interfaces are dictated by what llvm provides.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_HLSLCONSTANTFOLDING_H
#define LLVM_ANALYSIS_HLSLCONSTANTFOLDING_H
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class Constant;
  class Function;
  class Type;
  template<typename T>
  class ArrayRef;
}

namespace hlsl {
  /// ConstantFoldScalarCall - Try to constant fold the call instruction.
  /// If successful, the constant result is returned, if not, null is returned.
  llvm::Constant *ConstantFoldScalarCall(llvm::StringRef Name, llvm::Type *Ty, llvm::ArrayRef<llvm::Constant *> Operands);

  /// ConstantFoldScalarCallExt
  /// Hook point for constant folding of extensions.
  llvm::Constant *ConstantFoldScalarCallExt(llvm::StringRef Name, llvm::Type *Ty, llvm::ArrayRef<llvm::Constant *> Operands);

  /// CanConstantFoldCallTo - Return true if we can potentially constant
  /// fold a call to the given function.
  bool CanConstantFoldCallTo(const llvm::Function *F);

  /// CanConstantFoldCallToExt
  /// Hook point for constant folding of extensions.
  bool CanConstantFoldCallToExt(const llvm::Function *F);
}

#endif
