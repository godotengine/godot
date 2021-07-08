//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteAtomicFunctionExpressions rewrites the expressions that contain
// atomic function calls and cannot be directly translated into HLSL into
// several simple ones that can be easily handled in the HLSL translator.
//
// We need to rewite these expressions because:
// 1. All GLSL atomic functions have return values, which all represent the
//    original value of the shared or ssbo variable; while all HLSL atomic
//    functions don't, and the original value can be stored in the last
//    parameter of the function call.
// 2. For HLSL atomic functions, the last parameter that stores the original
//    value is optional except for InterlockedExchange and
//    InterlockedCompareExchange. Missing original_value in the call of
//    InterlockedExchange or InterlockedCompareExchange results in a compile
//    error from HLSL compiler.
//
// RewriteAtomicFunctionExpressions is a function that can modify the AST
// to ensure all the expressions that contain atomic function calls can be
// directly translated into HLSL expressions.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteAtomicFunctionExpressions(TCompiler *compiler,
                                                       TIntermNode *root,
                                                       TSymbolTable *symbolTable,
                                                       int shaderVersion);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_
