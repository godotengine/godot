//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The ArrayReturnValueToOutParameter function changes return values of an array type to out
// parameters in function definitions, prototypes and call sites.

#ifndef COMPILER_TRANSLATOR_TREEOPS_ARRAYRETURNVALUETOOUTPARAMETER_H_
#define COMPILER_TRANSLATOR_TREEOPS_ARRAYRETURNVALUETOOUTPARAMETER_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TSymbolTable;

ANGLE_NO_DISCARD bool ArrayReturnValueToOutParameter(TCompiler *compiler,
                                                     TIntermNode *root,
                                                     TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_ARRAYRETURNVALUETOOUTPARAMETER_H_
