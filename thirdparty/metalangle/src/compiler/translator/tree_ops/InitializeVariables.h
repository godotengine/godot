//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_
#define COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_

#include <GLSLANG/ShaderLang.h>

#include "compiler/translator/ExtensionBehavior.h"
#include "compiler/translator/IntermNode.h"

namespace sh
{
class TCompiler;
class TSymbolTable;

typedef std::vector<sh::ShaderVariable> InitVariableList;

// For all of the functions below: If canUseLoopsToInitialize is set, for loops are used instead of
// a large number of initializers where it can make sense, such as for initializing large arrays.

// Return a sequence of assignment operations to initialize "initializedSymbol". initializedSymbol
// may be an array, struct or any combination of these, as long as it contains only basic types.
TIntermSequence *CreateInitCode(const TIntermTyped *initializedSymbol,
                                bool canUseLoopsToInitialize,
                                bool highPrecisionSupported,
                                TSymbolTable *symbolTable);

// Initialize all uninitialized local variables, so that undefined behavior is avoided.
ANGLE_NO_DISCARD bool InitializeUninitializedLocals(TCompiler *compiler,
                                                    TIntermBlock *root,
                                                    int shaderVersion,
                                                    bool canUseLoopsToInitialize,
                                                    bool highPrecisionSupported,
                                                    TSymbolTable *symbolTable);

// This function can initialize all the types that CreateInitCode is able to initialize. All
// variables must be globals which can be found in the symbol table. For now it is used for the
// following two scenarios:
//   1. Initializing gl_Position;
//   2. Initializing output variables referred to in the shader source.
// Note: The type of each lvalue in an initializer is retrieved from the symbol table. gl_FragData
// requires special handling because the number of indices which can be initialized is determined by
// enabled extensions.
ANGLE_NO_DISCARD bool InitializeVariables(TCompiler *compiler,
                                          TIntermBlock *root,
                                          const InitVariableList &vars,
                                          TSymbolTable *symbolTable,
                                          int shaderVersion,
                                          const TExtensionBehavior &extensionBehavior,
                                          bool canUseLoopsToInitialize,
                                          bool highPrecisionSupported);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_
