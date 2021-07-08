//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampFragDepth.h: Limit the value that is written to gl_FragDepth to the range [0.0, 1.0].
// The clamping is run at the very end of shader execution, and is only performed if the shader
// statically accesses gl_FragDepth.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_
#define COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool ClampFragDepth(TCompiler *compiler,
                                     TIntermBlock *root,
                                     TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_
