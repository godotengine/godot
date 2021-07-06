//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATORGLSL_H_
#define COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATORGLSL_H_

#include "GLSLANG/ShaderLang.h"

namespace sh
{
class BuiltInFunctionEmulator;

//
// This works around bug in Intel Mac drivers.
//
void InitBuiltInAbsFunctionEmulatorForGLSLWorkarounds(BuiltInFunctionEmulator *emu,
                                                      sh::GLenum shaderType);

//
// This works around isnan() bug in Intel Mac drivers
//
void InitBuiltInIsnanFunctionEmulatorForGLSLWorkarounds(BuiltInFunctionEmulator *emu,
                                                        int targetGLSLVersion);
//
// This works around atan(y, x) bug in NVIDIA drivers.
//
void InitBuiltInAtanFunctionEmulatorForGLSLWorkarounds(BuiltInFunctionEmulator *emu);

//
// This function is emulating built-in functions missing from GLSL 1.30 and higher.
//
void InitBuiltInFunctionEmulatorForGLSLMissingFunctions(BuiltInFunctionEmulator *emu,
                                                        sh::GLenum shaderType,
                                                        int targetGLSLVersion);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATORGLSL_H_
