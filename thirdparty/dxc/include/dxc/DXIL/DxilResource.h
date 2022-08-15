///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilResource.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL SRVs and UAVs.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DxilConstants.h"
#include "dxc/DXIL/DxilResourceBase.h"
#include "dxc/DXIL/DxilCompType.h"


namespace hlsl {

/// Use this class to represent an HLSL resource (SRV/UAV).
class DxilResource : public DxilResourceBase {
public:
  /// Total number of coordinates necessary to access resource.
  static unsigned GetNumCoords(Kind ResourceKind);
  /// Total number of resource dimensions (Only width and height for cube).
  static unsigned GetNumDimensions(Kind ResourceKind);
  /// Total number of resource dimensions for CalcLOD (no array).
  static unsigned GetNumDimensionsForCalcLOD(Kind ResourceKind);
  /// Total number of offsets (in [-8,7]) necessary to access resource.
  static unsigned GetNumOffsets(Kind ResourceKind);
  /// Whether the resource kind is a texture. This does not include
  /// FeedbackTextures.
  static bool IsAnyTexture(Kind ResourceKind);
  /// Whether the resource kind is an array of textures. This does not include
  /// FeedbackTextures.
  static bool IsAnyArrayTexture(Kind ResourceKind);
  /// Whether the resource kind is a TextureCube or TextureCubeArray.
  static bool IsAnyTextureCube(Kind ResourceKind);
  /// Whether the resource kind is a FeedbackTexture.
  static bool IsFeedbackTexture(Kind ResourceKind);
  /// Whether the resource kind is a Texture or FeedbackTexture kind with array
  /// dimension.
  static bool IsArrayKind(Kind ResourceKind);

  DxilResource();

  CompType GetCompType() const;
  void SetCompType(const CompType CT);

  llvm::Type *GetRetType() const;

  unsigned GetSampleCount() const;
  void SetSampleCount(unsigned SampleCount);

  unsigned GetElementStride() const;
  void SetElementStride(unsigned ElemStride);

  unsigned GetBaseAlignLog2() const;
  void SetBaseAlignLog2(unsigned baseAlignLog2);

  DXIL::SamplerFeedbackType GetSamplerFeedbackType() const;
  void SetSamplerFeedbackType(DXIL::SamplerFeedbackType Value);

  bool IsGloballyCoherent() const;
  void SetGloballyCoherent(bool b);
  bool HasCounter() const;
  void SetHasCounter(bool b);

  bool IsRO() const;
  bool IsRW() const;
  void SetRW(bool bRW);
  bool IsROV() const;
  void SetROV(bool bROV);

  bool IsAnyTexture() const;
  bool IsStructuredBuffer() const;
  bool IsTypedBuffer() const;
  bool IsRawBuffer() const;
  bool IsTBuffer() const;
  bool IsFeedbackTexture() const;
  bool IsAnyArrayTexture() const;
  bool IsAnyTextureCube() const;
  bool IsArrayKind() const;

  bool HasAtomic64Use() const;
  void SetHasAtomic64Use(bool b);

  static bool classof(const DxilResourceBase *R) {
    return R->GetClass() == DXIL::ResourceClass::SRV || R->GetClass() == DXIL::ResourceClass::UAV;
  }

private:
  unsigned m_SampleCount;
  unsigned m_ElementStride; // in bytes
  unsigned m_baseAlignLog2 = 0; // worst-case alignment
  CompType m_CompType;
  DXIL::SamplerFeedbackType m_SamplerFeedbackType;
  bool m_bGloballyCoherent;
  bool m_bHasCounter;
  bool m_bROV;
  bool m_bHasAtomic64Use;
};

} // namespace hlsl
