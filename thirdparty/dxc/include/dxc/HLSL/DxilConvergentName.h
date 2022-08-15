///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilConvergentName.h                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
//  Expose helper function name to avoid link issue with spirv.              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#pragma once

namespace hlsl {
  static const char *kConvergentFunctionPrefix = "dxil.convergent.marker.";
}