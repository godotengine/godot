//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EmulateGLDrawID is an AST traverser to convert the gl_DrawID builtin
// to a uniform int
//
// EmulateGLBaseVertexBaseInstance is an AST traverser to convert the gl_BaseVertex and
// gl_BaseInstance builtin to uniform ints
//
// EmulateGLBaseInstance is an AST traverser to convert the gl_BaseInstance builtin
// to a uniform int
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_EMULATEMULTIDRAWSHADERBUILTINS_H_
#define COMPILER_TRANSLATOR_TREEOPS_EMULATEMULTIDRAWSHADERBUILTINS_H_

#include <GLSLANG/ShaderLang.h>
#include <vector>

#include "common/angleutils.h"
#include "compiler/translator/HashNames.h"

namespace sh
{
struct ShaderVariable;
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool EmulateGLDrawID(TCompiler *compiler,
                                      TIntermBlock *root,
                                      TSymbolTable *symbolTable,
                                      std::vector<sh::ShaderVariable> *uniforms,
                                      bool shouldCollect);

ANGLE_NO_DISCARD bool EmulateGLBaseVertexBaseInstance(TCompiler *compiler,
                                                      TIntermBlock *root,
                                                      TSymbolTable *symbolTable,
                                                      std::vector<sh::ShaderVariable> *uniforms,
                                                      bool shouldCollect,
                                                      bool addBaseVertexToVertexID);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_EMULATEMULTIDRAWSHADERBUILTINS_H_
