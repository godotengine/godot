///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceBinding.cpp                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilResourceBinding.h"
#include "llvm/IR/Constant.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include "dxc/DXIL/DxilResourceBase.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilSampler.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilUtil.h"

using namespace llvm;

namespace hlsl {

bool DxilResourceBinding::operator==(const DxilResourceBinding &B) {
  return rangeLowerBound == B.rangeLowerBound &&
         rangeUpperBound == B.rangeUpperBound && spaceID == B.spaceID &&
         resourceClass == B.resourceClass;
}

bool DxilResourceBinding::operator!=(const DxilResourceBinding &B) {
  return !(*this == B) ;
}

namespace resource_helper {

// The constant is as struct with int fields.
// ShaderModel 6.6 has 4 fileds.
llvm::Constant *getAsConstant(const DxilResourceBinding &B, llvm::Type *Ty,
                              const ShaderModel &) {
  StructType *ST = cast<StructType>(Ty);
  switch (ST->getNumElements()) {
  case 4: {
    Constant *RawDwords[] = {
        ConstantInt::get(ST->getElementType(0), B.rangeLowerBound),
        ConstantInt::get(ST->getElementType(1), B.rangeUpperBound),
        ConstantInt::get(ST->getElementType(2), B.spaceID),
        ConstantInt::get(ST->getElementType(3), B.resourceClass)};
    return ConstantStruct::get(ST, RawDwords);
  } break;
  default:
    return nullptr;
    break;
  }
  return nullptr;
}
DxilResourceBinding loadBindingFromConstant(const llvm::Constant &C) {
  DxilResourceBinding B;

  // Ty Should match C.getType().
  Type *Ty = C.getType();
  StructType *ST = cast<StructType>(Ty);
  switch (ST->getNumElements()) {
  case 4: {
    if (isa<ConstantAggregateZero>(&C)) {
      B.rangeLowerBound = 0;
      B.rangeUpperBound = 0;
      B.spaceID = 0;
      B.resourceClass = 0;
    } else {
      const ConstantStruct *CS = cast<ConstantStruct>(&C);
      const Constant *rangeLowerBound = CS->getOperand(0);
      const Constant *rangeUpperBound = CS->getOperand(1);
      const Constant *spaceID = CS->getOperand(2);
      const Constant *resourceClass = CS->getOperand(3);
      B.rangeLowerBound = cast<ConstantInt>(rangeLowerBound)->getLimitedValue();
      B.rangeUpperBound = cast<ConstantInt>(rangeUpperBound)->getLimitedValue();
      B.spaceID = cast<ConstantInt>(spaceID)->getLimitedValue();
      B.resourceClass = cast<ConstantInt>(resourceClass)->getLimitedValue();
    }
  } break;
  default:
    B.resourceClass = (uint8_t)DXIL::ResourceClass::Invalid;
    break;
  }
  return B;
}
DxilResourceBinding loadBindingFromCreateHandleFromBinding(
    const DxilInst_CreateHandleFromBinding &createHandle, llvm::Type *Ty,
    const ShaderModel &) {
  Constant *B = cast<Constant>(createHandle.get_bind());
  return loadBindingFromConstant(*B);
}
DxilResourceBinding loadBindingFromResourceBase(DxilResourceBase *Res) {
  DxilResourceBinding B = {};
  B.resourceClass = (uint8_t)DXIL::ResourceClass::Invalid;
  if (!Res)
    return B;
  B.rangeLowerBound = Res->GetLowerBound();
  B.rangeUpperBound = Res->GetUpperBound();
  B.spaceID = Res->GetSpaceID();
  B.resourceClass = (uint8_t)Res->GetClass();
  return B;
}

} // namespace resource_helper
} // namespace hlsl
