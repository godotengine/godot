///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixCompilationInfo.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Retrieves compilation info such as HLSL entry point, macro defs, etc.     //
// from llvm debug metadata                                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/Support/WinIncludes.h"

#include <map>
#include <memory>
#include <vector>

namespace dxil_debug_info {

HRESULT
CreateDxilCompilationInfo(IMalloc *pMalloc, dxil_dia::Session *pSession,
                          IDxcPixCompilationInfo **ppResult);

} // namespace dxil_debug_info
