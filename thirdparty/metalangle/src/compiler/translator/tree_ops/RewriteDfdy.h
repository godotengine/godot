//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// This mutating tree traversal flips the output of dFdy() to account for framebuffer flipping.
//
// From: dFdy(p)
// To:   (dFdy(p) * viewportYScale)
//
// See http://anglebug.com/3487

#ifndef COMPILER_TRANSLATOR_TREEOPS_FLIP_DFDY_H_
#define COMPILER_TRANSLATOR_TREEOPS_FLIP_DFDY_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TIntermBinary;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteDfdy(TCompiler *compiler,
                                  TIntermNode *root,
                                  const TSymbolTable &symbolTable,
                                  int shaderVersion,
                                  TIntermBinary *viewportYScale);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_FLIP_DFDY_H_
