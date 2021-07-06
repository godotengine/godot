//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// NameNamelessUniformBuffers: Gives nameless uniform buffer variables internal names.
//
// For example:
//   uniform UniformBuffer { int a; };
//   x = a;
// becomes:
//   uniform UniformBuffer { int a; } s123;
//   x = s123.a;
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_NAMENAMELESSUNIFORMBUFFERS_H_
#define COMPILER_TRANSLATOR_TREEOPS_NAMENAMELESSUNIFORMBUFFERS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool NameNamelessUniformBuffers(TCompiler *compiler,
                                                 TIntermBlock *root,
                                                 TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_NAMENAMELESSUNIFORMBUFFERS_H_
