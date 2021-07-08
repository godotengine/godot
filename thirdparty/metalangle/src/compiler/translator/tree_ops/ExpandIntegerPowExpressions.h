//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// This mutating tree traversal works around a bug in the HLSL compiler optimizer with "pow" that
// manifests under the following conditions:
//
// - If pow() has a literal exponent value
// - ... and this value is integer or within 10e-6 of an integer
// - ... and it is in {-4, -3, -2, 2, 3, 4, 5, 6, 7, 8}
//
// The workaround is to replace the pow with a series of multiplies.
// See http://anglebug.com/851

#ifndef COMPILER_TRANSLATOR_TREEOPS_EXPANDINTEGERPOWEXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_EXPANDINTEGERPOWEXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool ExpandIntegerPowExpressions(TCompiler *compiler,
                                                  TIntermNode *root,
                                                  TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_EXPANDINTEGERPOWEXPRESSIONS_H_
