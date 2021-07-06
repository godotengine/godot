//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteExpressionsWithShaderStorageBlock rewrites the expressions that contain shader storage
// block calls into several simple ones that can be easily handled in the HLSL translator. After the
// AST pass, all ssbo related blocks will be like below:
//     ssbo_access_chain = ssbo_access_chain;
//     ssbo_access_chain = expr_no_ssbo;
//     lvalue_no_ssbo    = ssbo_access_chain;
//
// Below situations are needed to be rewritten (Details can be found in .cpp file).
//     SSBO as the operand of compound assignment operators.
//     SSBO as the operand of ++/--.
//     SSBO as the operand of repeated assignment.
//     SSBO as the operand of readonly unary/binary/ternary operators.
//     SSBO as the argument of aggregate type.
//     SSBO as the condition of if/switch/while/do-while/for

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteExpressionsWithShaderStorageBlock(TCompiler *compiler,
                                                               TIntermNode *root,
                                                               TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_
