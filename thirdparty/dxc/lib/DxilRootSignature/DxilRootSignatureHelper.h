///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRootSignature.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides support for manipulating root signature structures.              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/Global.h"

namespace hlsl {

DEFINE_ENUM_FLAG_OPERATORS(DxilRootSignatureFlags)
DEFINE_ENUM_FLAG_OPERATORS(DxilRootDescriptorFlags)
DEFINE_ENUM_FLAG_OPERATORS(DxilDescriptorRangeType)
DEFINE_ENUM_FLAG_OPERATORS(DxilDescriptorRangeFlags)

// Execute (error) and throw.
#define EAT(x) { (x); throw ::hlsl::Exception(E_FAIL); }

namespace root_sig_helper {
// GetFlags/SetFlags overloads.
DxilRootDescriptorFlags GetFlags(const DxilRootDescriptor &);
void SetFlags(DxilRootDescriptor &, DxilRootDescriptorFlags);
DxilRootDescriptorFlags GetFlags(const DxilRootDescriptor1 &D);
void SetFlags(DxilRootDescriptor1 &D, DxilRootDescriptorFlags Flags);
void SetFlags(DxilContainerRootDescriptor1 &D, DxilRootDescriptorFlags Flags);
DxilDescriptorRangeFlags GetFlags(const DxilDescriptorRange &D);
void SetFlags(DxilDescriptorRange &, DxilDescriptorRangeFlags);
DxilDescriptorRangeFlags GetFlags(const DxilContainerDescriptorRange &D);
void SetFlags(DxilContainerDescriptorRange &, DxilDescriptorRangeFlags);
DxilDescriptorRangeFlags GetFlags(const DxilDescriptorRange1 &D);
void SetFlags(DxilDescriptorRange1 &D, DxilDescriptorRangeFlags Flags);
DxilDescriptorRangeFlags GetFlags(const DxilContainerDescriptorRange1 &D);
void SetFlags(DxilContainerDescriptorRange1 &D, DxilDescriptorRangeFlags Flags);
}

}
