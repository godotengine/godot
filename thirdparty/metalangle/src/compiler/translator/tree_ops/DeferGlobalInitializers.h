//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DeferGlobalInitializers is an AST traverser that moves global initializers into a separate
// function that is called in the beginning of main(). This enables initialization of globals with
// uniforms or non-constant globals, as allowed by the WebGL spec. Some initializers referencing
// non-constants may need to be unfolded into if statements in HLSL - this kind of steps should be
// done after DeferGlobalInitializers is run. Note that it's important that the function definition
// is at the end of the shader, as some globals may be declared after main().
//
// It can also initialize all uninitialized globals.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_
#define COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool DeferGlobalInitializers(TCompiler *compiler,
                                              TIntermBlock *root,
                                              bool initializeUninitializedGlobals,
                                              bool canUseLoopsToInitialize,
                                              bool highPrecisionSupported,
                                              TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_
