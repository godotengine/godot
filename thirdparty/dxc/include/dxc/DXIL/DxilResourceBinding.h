///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceBinding.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation properties for DXIL resource binding.                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DxilConstants.h"

namespace llvm {
class Constant;
class Type;
}

namespace hlsl {

struct DxilResourceBinding {
  uint32_t rangeLowerBound;
  uint32_t rangeUpperBound;
  uint32_t spaceID;
  uint8_t resourceClass;
  uint8_t Reserved1;
  uint8_t Reserved2;
  uint8_t Reserved3;
  bool operator==(const DxilResourceBinding &);
  bool operator!=(const DxilResourceBinding &);
};

static_assert(sizeof(DxilResourceBinding) == 4 * sizeof(uint32_t),
              "update shader model and functions read/write "
              "DxilResourceBinding when size is changed");

class ShaderModel;
class DxilResourceBase;
struct DxilInst_CreateHandleFromBinding;

namespace resource_helper {
llvm::Constant *getAsConstant(const DxilResourceBinding &, llvm::Type *Ty,
                              const ShaderModel &);
DxilResourceBinding loadBindingFromConstant(const llvm::Constant &C);
DxilResourceBinding
loadBindingFromCreateHandleFromBinding(const DxilInst_CreateHandleFromBinding &createHandle, llvm::Type *Ty,
                       const ShaderModel &);
DxilResourceBinding loadBindingFromResourceBase(DxilResourceBase *);

} // namespace resource_helper

} // namespace hlsl
