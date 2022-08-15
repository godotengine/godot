///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilEntryProps.h                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Put entry signature and function props together.                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilFunctionProps.h"

namespace hlsl {

class DxilEntryProps {
public:
  DxilEntrySignature sig;
  DxilFunctionProps props;
  DxilEntryProps(DxilFunctionProps &p, bool bUseMinPrecision)
      : sig(p.shaderKind, bUseMinPrecision), props(p) {}
  DxilEntryProps(DxilEntryProps &p)
      : sig(p.sig), props(p.props) {}
};
}