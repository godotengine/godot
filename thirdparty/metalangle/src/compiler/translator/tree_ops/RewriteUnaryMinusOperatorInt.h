// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// This mutating tree traversal works around a bug on evaluating unary
// integer variable on Intel D3D driver. It works by rewriting -(int) to
// ~(int) + 1 when evaluating unary integer variables.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITEUNARYMINUSOPERATORINT_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITEUNARYMINUSOPERATORINT_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool RewriteUnaryMinusOperatorInt(TCompiler *compiler, TIntermNode *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITEUNARYMINUSOPERATORINT_H_
