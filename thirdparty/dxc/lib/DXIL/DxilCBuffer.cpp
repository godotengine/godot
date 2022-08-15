///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCBuffer.cpp                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/Support/Global.h"


namespace hlsl {

//------------------------------------------------------------------------------
//
// DxilCBuffer methods.
//
DxilCBuffer::DxilCBuffer()
: DxilResourceBase(DxilResourceBase::Class::CBuffer)
, m_SizeInBytes(0) {
  SetKind(DxilResourceBase::Kind::CBuffer);
}

DxilCBuffer::~DxilCBuffer() {}

unsigned DxilCBuffer::GetSize() const { return m_SizeInBytes; }

void DxilCBuffer::SetSize(unsigned InstanceSizeInBytes) { m_SizeInBytes = InstanceSizeInBytes; }

} // namespace hlsl
