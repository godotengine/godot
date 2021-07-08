//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The ValidateVaryingLocations function checks if there exists location conflicts on shader
// varyings.
//

#ifndef COMPILER_TRANSLATOR_VALIDATEVARYINGLOCATIONS_H_
#define COMPILER_TRANSLATOR_VALIDATEVARYINGLOCATIONS_H_

#include "GLSLANG/ShaderVars.h"

namespace sh
{

class TIntermBlock;
class TDiagnostics;

bool ValidateVaryingLocations(TIntermBlock *root, TDiagnostics *diagnostics, GLenum shaderType);

}  // namespace sh

#endif
