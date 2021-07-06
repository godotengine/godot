//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The SeparateArrayInitialization function splits each array initialization into a declaration and
// an assignment.
// Example:
//     type[n] a = initializer;
// will effectively become
//     type[n] a;
//     a = initializer;
//
// Note that if the array is declared as const, the initialization may still be split, making the
// AST technically invalid. Because of that this transformation should only be used when subsequent
// stages don't care about const qualifiers. However, the initialization will not be split if the
// initializer can be written as a HLSL literal.

#ifndef COMPILER_TRANSLATOR_TREEOPS_SEPARATEARRAYINITIALIZATION_H_
#define COMPILER_TRANSLATOR_TREEOPS_SEPARATEARRAYINITIALIZATION_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool SeparateArrayInitialization(TCompiler *compiler, TIntermNode *root);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SEPARATEARRAYINITIALIZATION_H_
