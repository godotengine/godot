//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PruneNoOps.h: The PruneNoOps function prunes:
//   1. Empty declarations "int;". Empty declarators will be pruned as well, so for example:
//        int , a;
//      is turned into
//        int a;
//   2. Literal statements: "1.0;". The ESSL output doesn't define a default precision for float,
//      so float literal statements would end up with no precision which is invalid ESSL.

#ifndef COMPILER_TRANSLATOR_TREEOPS_PRUNENOOPS_H_
#define COMPILER_TRANSLATOR_TREEOPS_PRUNENOOPS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool PruneNoOps(TCompiler *compiler,
                                 TIntermBlock *root,
                                 TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_PRUNENOOPS_H_
