//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FoldExpressions.h: Fold expressions. This may fold expressions so that the qualifier of the
// folded node differs from the qualifier of the original expression, so it needs to be done after
// parsing and validation of qualifiers is complete. Expressions that are folded: 1. Ternary ops
// with a constant condition.

#ifndef COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TDiagnostics;

ANGLE_NO_DISCARD bool FoldExpressions(TCompiler *compiler,
                                      TIntermBlock *root,
                                      TDiagnostics *diagnostics);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_FOLDEXPRESSIONS_H_
