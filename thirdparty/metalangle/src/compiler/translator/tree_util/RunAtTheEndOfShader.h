//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RunAtTheEndOfShader.h: Add code to be run at the end of the shader.
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEENDOFSHADER_H_
#define COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEENDOFSHADER_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool RunAtTheEndOfShader(TCompiler *compiler,
                                          TIntermBlock *root,
                                          TIntermNode *codeToRun,
                                          TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEENDOFSHADER_H_
