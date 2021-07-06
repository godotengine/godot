//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SplitSequenceOperator is an AST traverser that detects sequence operator expressions that
// go through further AST transformations that generate statements, and splits them so that
// possible side effects of earlier parts of the sequence operator expression are guaranteed to be
// evaluated before the latter parts of the sequence operator expression are evaluated.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SPLITSEQUENCEOPERATOR_H_
#define COMPILER_TRANSLATOR_TREEOPS_SPLITSEQUENCEOPERATOR_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool SplitSequenceOperator(TCompiler *compiler,
                                            TIntermNode *root,
                                            int patternsToSplitMask,
                                            TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SPLITSEQUENCEOPERATOR_H_
