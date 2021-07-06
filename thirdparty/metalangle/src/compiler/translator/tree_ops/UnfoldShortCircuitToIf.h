//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// UnfoldShortCircuitToIf is an AST traverser to convert short-circuiting operators to if-else
// statements.
// The results are assigned to s# temporaries, which are used by the main translator instead of
// the original expression.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUIT_H_
#define COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUIT_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool UnfoldShortCircuitToIf(TCompiler *compiler,
                                             TIntermNode *root,
                                             TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUIT_H_
