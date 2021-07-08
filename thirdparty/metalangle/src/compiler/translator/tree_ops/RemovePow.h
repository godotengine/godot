//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemovePow is an AST traverser to convert pow(x, y) built-in calls where y is a
// constant to exp2(y * log2(x)). This works around an issue in NVIDIA 311 series
// OpenGL drivers.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEPOW_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEPOW_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool RemovePow(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEPOW_H_
