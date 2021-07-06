//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteElseBlocks.h: Prototype for tree transform to change
//   all if-else blocks to if-if blocks.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITEELSEBLOCKS_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITEELSEBLOCKS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteElseBlocks(TCompiler *compiler,
                                        TIntermNode *node,
                                        TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITEELSEBLOCKS_H_
