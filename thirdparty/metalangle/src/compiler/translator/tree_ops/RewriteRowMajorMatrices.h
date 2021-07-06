//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteRowMajorMatrices: Change row-major matrices to column-major in uniform and storage
// buffers.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITEROWMAJORMATRICES_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITEROWMAJORMATRICES_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteRowMajorMatrices(TCompiler *compiler,
                                              TIntermBlock *root,
                                              TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITEROWMAJORMATRICES_H_
