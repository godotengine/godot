///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResourceProperties.h                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation properties for DXIL handle.                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DxilConstants.h"

namespace llvm {
class Constant;
class Type;
}

namespace hlsl {

struct DxilResourceProperties {
  struct TypedProps {
    uint8_t CompType;     // TypedBuffer/Image component type.
    uint8_t CompCount;    // Number of components known to shader.
    uint8_t SampleCount;  // Number of samples for multisample texture if defined in HLSL.
    uint8_t Reserved3;
  };

  struct BasicProps {
    // BYTE 0
    uint8_t ResourceKind; // DXIL::ResourceKind

    // BYTE 1
    // Alignment of SRV/UAV base in 2^n. 0 is unknown/worst-case.
    uint8_t BaseAlignLog2 : 4;
    uint8_t IsUAV : 1;
    uint8_t IsROV : 1;
    uint8_t IsGloballyCoherent : 1;

    // Depending on ResourceKind, this indicates:
    //  Sampler: SamplerKind::Comparison
    //  StructuredBuffer: HasCounter
    //  Other: must be 0
    uint8_t SamplerCmpOrHasCounter : 1;

    // BYTE 2
    uint8_t Reserved2;

    // BYTE 3
    uint8_t Reserved3;
  };

  union {
    BasicProps  Basic;
    uint32_t RawDword0;
  };
  // DWORD
  union {
    TypedProps Typed;
    uint32_t StructStrideInBytes; // in bytes for StructuredBuffer.
    DXIL::SamplerFeedbackType SamplerFeedbackType; // FeedbackTexture2D.
    uint32_t CBufferSizeInBytes; // Cbuffer used size in bytes.
    uint32_t RawDword1;
  };
  DxilResourceProperties();
  DXIL::ResourceClass getResourceClass() const;
  DXIL::ResourceKind  getResourceKind() const;
  DXIL::ComponentType getCompType() const;
  unsigned getElementStride() const;
  void setResourceKind(DXIL::ResourceKind RK);
  bool isUAV() const;
  bool operator==(const DxilResourceProperties &) const;
  bool operator!=(const DxilResourceProperties &) const;
  bool isValid() const;
};

static_assert(sizeof(DxilResourceProperties) == 2 * sizeof(uint32_t),
              "update shader model and functions read/write "
              "DxilResourceProperties when size is changed");

class ShaderModel;
class DxilResourceBase;
struct DxilInst_AnnotateHandle;

namespace resource_helper {
llvm::Constant *getAsConstant(const DxilResourceProperties &, llvm::Type *Ty,
                              const ShaderModel &);
DxilResourceProperties loadPropsFromConstant(const llvm::Constant &C);
DxilResourceProperties
loadPropsFromAnnotateHandle(DxilInst_AnnotateHandle &annotateHandle, const ShaderModel &);
DxilResourceProperties loadPropsFromResourceBase(const DxilResourceBase *);
DxilResourceProperties tryMergeProps(DxilResourceProperties,
                                     DxilResourceProperties);

llvm::Constant *tryMergeProps(const llvm::Constant *, const llvm::Constant *,
                              llvm::Type *Ty, const ShaderModel &);
} // namespace resource_helper

} // namespace hlsl
