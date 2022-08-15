///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCBuffer.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL constant buffer (cbuffer).                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilResourceBase.h"


namespace hlsl {

/// Use this class to represent HLSL cbuffer.
class DxilCBuffer : public DxilResourceBase {
public:
  DxilCBuffer();
  virtual ~DxilCBuffer();

  unsigned GetSize() const;

  void SetSize(unsigned InstanceSizeInBytes);

private:
  unsigned m_SizeInBytes;   // Cbuffer instance size in bytes.
};

} // namespace hlsl
