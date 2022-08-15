///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceProperites.cpp                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilResourceProperties.h"
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

DxilResourceProperties::DxilResourceProperties() {
  RawDword0 = 0;
  RawDword1 = 0;
  Basic.ResourceKind = (uint8_t)DXIL::ResourceKind::Invalid;
}
bool DxilResourceProperties::isUAV() const { return Basic.IsUAV; }
bool DxilResourceProperties::isValid() const {
  return getResourceKind() != DXIL::ResourceKind::Invalid;
}

DXIL::ResourceClass DxilResourceProperties::getResourceClass() const {
  switch (static_cast<DXIL::ResourceKind>(Basic.ResourceKind)) {
  default:
    return Basic.IsUAV ? DXIL::ResourceClass::UAV : DXIL::ResourceClass::SRV;
  case DXIL::ResourceKind::CBuffer:
    return DXIL::ResourceClass::CBuffer;
  case DXIL::ResourceKind::Sampler:
    return DXIL::ResourceClass::Sampler;
  case DXIL::ResourceKind::Invalid:
    return DXIL::ResourceClass::Invalid;
  }
}

DXIL::ResourceKind DxilResourceProperties::getResourceKind() const {
  return static_cast<DXIL::ResourceKind>(Basic.ResourceKind);
}

void DxilResourceProperties::setResourceKind(DXIL::ResourceKind RK) {
  Basic.ResourceKind = (uint8_t)RK;
}

DXIL::ComponentType DxilResourceProperties::getCompType() const {
  return static_cast<DXIL::ComponentType>(Typed.CompType);
}

unsigned DxilResourceProperties::getElementStride() const {
  switch (getResourceKind()) {
  default:
    return CompType(getCompType()).GetSizeInBits() / 8;
  case DXIL::ResourceKind::RawBuffer:
    return 1;
  case DXIL::ResourceKind::StructuredBuffer:
    return StructStrideInBytes;
  case DXIL::ResourceKind::CBuffer:
  case DXIL::ResourceKind::Sampler:
    return 0;
  }
}

bool DxilResourceProperties::operator==(const DxilResourceProperties &RP) const {
  return RawDword0 == RP.RawDword0 &&
         RawDword1 == RP.RawDword1;
}

bool DxilResourceProperties::operator!=(const DxilResourceProperties &RP) const {
  return !(*this == RP) ;
}

namespace resource_helper {

// The constant is as struct with int32 fields.
// ShaderModel 6.6 has 2 fileds.
Constant *getAsConstant(const DxilResourceProperties &RP, Type *Ty,
                        const ShaderModel &) {
  StructType *ST = cast<StructType>(Ty);
  switch (ST->getNumElements()) {
  case 2: {
    Constant *RawDwords[] = {
        ConstantInt::get(ST->getElementType(0), RP.RawDword0),
        ConstantInt::get(ST->getElementType(1), RP.RawDword1)};
    return ConstantStruct::get(ST, RawDwords);
  } break;
  default:
    return nullptr;
    break;
  }
  return nullptr;
}

DxilResourceProperties loadPropsFromConstant(const Constant &C) {
  DxilResourceProperties RP;

  // Ty Should match C.getType().
  Type *Ty = C.getType();
  StructType *ST = cast<StructType>(Ty);
  switch (ST->getNumElements()) {
  case 2: {
    if (isa<ConstantAggregateZero>(&C)) {
      RP.RawDword0 = 0;
      RP.RawDword1 = 0;
    } else {
      const ConstantStruct *CS = cast<ConstantStruct>(&C);
      const Constant *RawDword0 = CS->getOperand(0);
      const Constant *RawDword1 = CS->getOperand(1);
      RP.RawDword0 = cast<ConstantInt>(RawDword0)->getLimitedValue();
      RP.RawDword1 = cast<ConstantInt>(RawDword1)->getLimitedValue();
    }
  } break;
  default:
    break;
  }
  return RP;
}

DxilResourceProperties
loadPropsFromAnnotateHandle(DxilInst_AnnotateHandle &annotateHandle,
                            const ShaderModel &SM) {
  Constant *ResProp = cast<Constant>(annotateHandle.get_props());
  return loadPropsFromConstant(*ResProp);
}

DxilResourceProperties loadPropsFromResourceBase(const DxilResourceBase *Res) {

  DxilResourceProperties RP;
  if (!Res) {
    return RP;
  }


  auto SetResProperties = [&RP](const DxilResource &Res) {
    switch (Res.GetKind()) {
    default:
      break;
    case DXIL::ResourceKind::FeedbackTexture2D:
    case DXIL::ResourceKind::FeedbackTexture2DArray:
      RP.SamplerFeedbackType = Res.GetSamplerFeedbackType();
      break;
    case DXIL::ResourceKind::RTAccelerationStructure:

      break;
    case DXIL::ResourceKind::StructuredBuffer:
    {
      RP.StructStrideInBytes = Res.GetElementStride();
      RP.Basic.BaseAlignLog2 = Res.GetBaseAlignLog2();
      break;
    }
    case DXIL::ResourceKind::Texture2DMS:
    case DXIL::ResourceKind::Texture2DMSArray:
    case DXIL::ResourceKind::TypedBuffer:
    case DXIL::ResourceKind::Texture1D:
    case DXIL::ResourceKind::Texture2D:
    case DXIL::ResourceKind::TextureCube:
    case DXIL::ResourceKind::Texture1DArray:
    case DXIL::ResourceKind::Texture2DArray:
    case DXIL::ResourceKind::TextureCubeArray:
    case DXIL::ResourceKind::Texture3D:
      Type *Ty = Res.GetRetType();
      RP.Typed.CompCount = dxilutil::GetResourceComponentCount(Ty);
      RP.Typed.CompType = (uint8_t)Res.GetCompType().GetKind();
      RP.Typed.SampleCount = (uint8_t)Res.GetSampleCount();
      break;
    }
  };

  switch (Res->GetClass()) { case DXIL::ResourceClass::Invalid: return RP;
  case DXIL::ResourceClass::SRV: {
    const DxilResource *SRV = (const DxilResource*)(Res);
    RP.Basic.ResourceKind = (uint8_t)Res->GetKind();
    SetResProperties(*SRV);
  } break;
  case DXIL::ResourceClass::UAV: {
    const DxilResource *UAV = (const DxilResource *)(Res);
    RP.Basic.IsUAV = true;
    RP.Basic.ResourceKind = (uint8_t)Res->GetKind();
    RP.Basic.IsGloballyCoherent = UAV->IsGloballyCoherent();
    RP.Basic.SamplerCmpOrHasCounter = UAV->HasCounter();
    RP.Basic.IsROV = UAV->IsROV();
    SetResProperties(*UAV);
  } break;
  case DXIL::ResourceClass::Sampler: {
    RP.Basic.ResourceKind = (uint8_t)Res->GetKind();
    const DxilSampler *Sampler = (const DxilSampler*)Res;
    if (Sampler->GetSamplerKind() == DXIL::SamplerKind::Comparison)
      RP.Basic.SamplerCmpOrHasCounter = true;
    else if (Sampler->GetSamplerKind() == DXIL::SamplerKind::Invalid)
      RP.Basic.ResourceKind = (uint8_t)DXIL::ResourceKind::Invalid;
  } break;
  case DXIL::ResourceClass::CBuffer: {
    RP.Basic.ResourceKind = (uint8_t)Res->GetKind();
    const DxilCBuffer *CB = (const DxilCBuffer *)Res;
    RP.CBufferSizeInBytes = CB->GetSize();
  } break;
  }
  return RP;
}

// Merge 2 props on a chain of annotateHandle.
DxilResourceProperties tryMergeProps(DxilResourceProperties curProps,
                                     DxilResourceProperties prevProps) {
  DxilResourceProperties props;
  if (curProps.Basic.ResourceKind != prevProps.Basic.ResourceKind) {
    return props;
  }

  if (curProps.Basic.IsUAV != prevProps.Basic.IsUAV)
    return props;

  if (curProps.Basic.IsUAV) {
    // Or hasCounter.
    if (curProps.Basic.SamplerCmpOrHasCounter !=
        prevProps.Basic.SamplerCmpOrHasCounter) {
      curProps.Basic.SamplerCmpOrHasCounter = true;
      prevProps.Basic.SamplerCmpOrHasCounter = true;
    }
    // curProps follow prevProps.
    if (curProps.Basic.IsGloballyCoherent !=
        prevProps.Basic.IsGloballyCoherent) {
      curProps.Basic.IsGloballyCoherent = prevProps.Basic.IsGloballyCoherent;
    }
  }

  if (curProps.Basic.ResourceKind == (uint8_t)DXIL::ResourceKind::CBuffer) {
    // use max cbuffer size.
    if (curProps.CBufferSizeInBytes != prevProps.CBufferSizeInBytes) {
      curProps.CBufferSizeInBytes =
          std::max(curProps.CBufferSizeInBytes, prevProps.CBufferSizeInBytes);
      prevProps.CBufferSizeInBytes = curProps.CBufferSizeInBytes;
    }
  }

  // If still not match after merge.
  // return null.
  if (curProps.RawDword0 != prevProps.RawDword0 ||
      curProps.RawDword1 != prevProps.RawDword1)
    return props;
  return curProps;
}
// Merge 2 props on a chain of annotateHandle.
Constant *tryMergeProps(const Constant *curPropsConst, const Constant *prevPropsConst, Type *Ty,
                        const ShaderModel &SM) {
  if (curPropsConst == prevPropsConst)
    return const_cast<Constant *>(curPropsConst);

  DxilResourceProperties curProps = loadPropsFromConstant(*curPropsConst);
  DxilResourceProperties prevProps = loadPropsFromConstant(*prevPropsConst);

  DxilResourceProperties props = tryMergeProps(curProps, prevProps);

  if (!props.isValid()) {
    return nullptr;
  }

  return getAsConstant(props, Ty, SM);
}

} // namespace resource_helper
} // namespace hlsl
