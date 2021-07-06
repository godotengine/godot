//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// UnfoldShortCircuitAST is an AST traverser to replace short-circuiting
// operations with ternary operations.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUITAST_H_
#define COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUITAST_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;

ANGLE_NO_DISCARD bool UnfoldShortCircuitAST(TCompiler *compiler, TIntermBlock *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_UNFOLDSHORTCIRCUITAST_H_
