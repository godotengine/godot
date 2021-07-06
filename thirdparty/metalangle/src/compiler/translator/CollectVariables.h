//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CollectVariables.h: Collect lists of shader interface variables based on the AST.

#ifndef COMPILER_TRANSLATOR_COLLECTVARIABLES_H_
#define COMPILER_TRANSLATOR_COLLECTVARIABLES_H_

#include <GLSLANG/ShaderLang.h>

#include "compiler/translator/ExtensionBehavior.h"

namespace sh
{

class TIntermBlock;
class TSymbolTable;

void CollectVariables(TIntermBlock *root,
                      std::vector<ShaderVariable> *attributes,
                      std::vector<ShaderVariable> *outputVariables,
                      std::vector<ShaderVariable> *uniforms,
                      std::vector<ShaderVariable> *inputVaryings,
                      std::vector<ShaderVariable> *outputVaryings,
                      std::vector<InterfaceBlock> *uniformBlocks,
                      std::vector<InterfaceBlock> *shaderStorageBlocks,
                      std::vector<InterfaceBlock> *inBlocks,
                      ShHashFunction64 hashFunction,
                      TSymbolTable *symbolTable,
                      GLenum shaderType,
                      const TExtensionBehavior &extensionBehavior);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_COLLECTVARIABLES_H_
