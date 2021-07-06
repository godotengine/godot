//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// NameEmbeddedUniformStructs: Gives nameless uniform struct internal names.
//
// For example:
//   uniform struct { int a; } uni;
// becomes:
//   struct s1 { int a; };
//   uniform s1 uni;
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_NAMEEMBEDDEDUNIFORMSTRUCTS_H_
#define COMPILER_TRANSLATOR_TREEOPS_NAMEEMBEDDEDUNIFORMSTRUCTS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool NameEmbeddedStructUniforms(TCompiler *compiler,
                                                 TIntermBlock *root,
                                                 TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_NAMEEMBEDDEDUNIFORMSTRUCTS_H_
