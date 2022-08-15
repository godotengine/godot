///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLLowerUDT.h                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower user defined type used directly by certain intrinsic operations.    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilTypeSystem.h"

namespace llvm {
class Constant;
class Function;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace hlsl {
class DxilTypeSystem;

llvm::StructType *GetLoweredUDT(
  llvm::StructType *structTy, hlsl::DxilTypeSystem *pTypeSys = nullptr);
llvm::Constant *TranslateInitForLoweredUDT(
    llvm::Constant *Init, llvm::Type *NewTy,
    // We need orientation for matrix fields
    hlsl::DxilTypeSystem *pTypeSys,
    hlsl::MatrixOrientation matOrientation = hlsl::MatrixOrientation::Undefined);
void ReplaceUsesForLoweredUDT(llvm::Value *V, llvm::Value *NewV);

} // namespace hlsl