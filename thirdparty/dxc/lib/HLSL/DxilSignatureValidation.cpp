///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignatureElement.h                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Validate HLSL signature element packing.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSigPoint.h"
#include "dxc/HLSL/DxilSignatureAllocator.h"

using namespace hlsl;
using namespace llvm;

#include <assert.h> // Needed for DxilPipelineStateValidation.h
#include "dxc/DxilContainer/DxilPipelineStateValidation.h"
#include <functional>
#include "dxc/HLSL/ViewIDPipelineValidation.inl"