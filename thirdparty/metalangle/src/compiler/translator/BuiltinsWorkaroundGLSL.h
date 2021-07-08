//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_BUILTINSWORKAROUNDGLSL_H_
#define COMPILER_TRANSLATOR_BUILTINSWORKAROUNDGLSL_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

#include "compiler/translator/Pragma.h"

namespace sh
{

ANGLE_NO_DISCARD bool ShaderBuiltinsWorkaround(TCompiler *compiler,
                                               TIntermBlock *root,
                                               TSymbolTable *symbolTable,
                                               ShCompileOptions compileOptions);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_BUILTINSWORKAROUNDGLSL_H_
