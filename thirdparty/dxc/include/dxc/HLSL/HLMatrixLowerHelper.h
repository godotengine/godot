///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixLowerHelper.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides helper functions to lower high level matrix.           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "llvm/IR/IRBuilder.h"

namespace llvm {
  class Type;
  class Value;
  template<typename T>
  class ArrayRef;
}

namespace hlsl {

class DxilFieldAnnotation;
class DxilTypeSystem;

namespace HLMatrixLower {

llvm::Value *BuildVector(llvm::Type *EltTy,
                         llvm::ArrayRef<llvm::Value *> elts,
                         llvm::IRBuilder<> &Builder);

} // namespace HLMatrixLower

} // namespace hlsl