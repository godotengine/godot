//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SeparateExpressionsReturningArrays splits array-returning expressions that are not array names
// from more complex expressions, assigning them to a temporary variable a#.
// Examples where a, b and c are all arrays:
// (a = b) == (a = c) is split into a = b; type[n] a1 = a; a = c; type[n] a2 = a; a1 == a2;
// type d = type[n](...)[i]; is split into type[n] a1 = type[n](...); type d = a1[i];

#ifndef COMPILER_TRANSLATOR_TREEOPS_SEPARATEEXPRESSIONSRETURNINGARRAYS_H_
#define COMPILER_TRANSLATOR_TREEOPS_SEPARATEEXPRESSIONSRETURNINGARRAYS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool SeparateExpressionsReturningArrays(TCompiler *compiler,
                                                         TIntermNode *root,
                                                         TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SEPARATEEXPRESSIONSRETURNINGARRAYS_H_
