///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLResource.h                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Representation of HLSL SRVs and UAVs in high-level DX IR.                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilResource.h"


namespace hlsl {

/// Use this class to represent an HLSL resource (SRV/UAV) in HLDXIR.
class HLResource : public DxilResource {
public:
  //QQQ
  // TODO: this does not belong here. QQQ
  //static Kind KeywordToKind(const std::string &keyword);
  
  HLResource();
};

} // namespace hlsl
