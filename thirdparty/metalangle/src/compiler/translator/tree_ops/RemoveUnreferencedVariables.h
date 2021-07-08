//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveUnreferencedVariables.h:
//  Drop variables that are declared but never referenced in the AST. This avoids adding unnecessary
//  initialization code for them. Also removes unreferenced struct types.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEUNREFERENCEDVARIABLES_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEUNREFERENCEDVARIABLES_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool RemoveUnreferencedVariables(TCompiler *compiler,
                                                  TIntermBlock *root,
                                                  TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEUNREFERENCEDVARIABLES_H_
