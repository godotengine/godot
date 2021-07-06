//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The SeparateDeclarations function processes declarations, so that in the end each declaration
// contains only one declarator.
// This is useful as an intermediate step when initialization needs to be separated from
// declaration, or when things need to be unfolded out of the initializer.
// Example:
//     int a[1] = int[1](1), b[1] = int[1](2);
// gets transformed when run through this class into the AST equivalent of:
//     int a[1] = int[1](1);
//     int b[1] = int[1](2);

#ifndef COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool SeparateDeclarations(TCompiler *compiler, TIntermNode *root);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_
