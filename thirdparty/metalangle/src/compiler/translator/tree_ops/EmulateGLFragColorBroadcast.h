//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Emulate gl_FragColor broadcast behaviors in ES2 where
// GL_EXT_draw_buffers is explicitly enabled in a fragment shader.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_
#define COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_

#include <vector>

#include "common/angleutils.h"

namespace sh
{
struct ShaderVariable;
class TCompiler;
class TIntermBlock;
class TSymbolTable;

// Replace all gl_FragColor with gl_FragData[0], and in the end of main() function,
// assign gl_FragData[1] ... gl_FragData[maxDrawBuffers - 1] with gl_FragData[0].
// If gl_FragColor is in outputVariables, it is replaced by gl_FragData.
ANGLE_NO_DISCARD bool EmulateGLFragColorBroadcast(TCompiler *compiler,
                                                  TIntermBlock *root,
                                                  int maxDrawBuffers,
                                                  std::vector<ShaderVariable> *outputVariables,
                                                  TSymbolTable *symbolTable,
                                                  int shaderVersion);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_
