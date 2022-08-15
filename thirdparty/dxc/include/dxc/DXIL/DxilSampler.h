///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSampler.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL sampler state.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilResourceBase.h"

namespace hlsl {

/// Use this class to represent HLSL sampler state.
class DxilSampler : public DxilResourceBase {
public:
  using SamplerKind = DXIL::SamplerKind;

  DxilSampler();

  SamplerKind GetSamplerKind() const;
  bool IsCompSampler() const;

  void SetSamplerKind(SamplerKind K);

private:
  SamplerKind m_SamplerKind;      // Sampler mode.
};

} // namespace hlsl
