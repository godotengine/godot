//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// AddDefaultReturnStatements.h: Add default return statements to functions that do not end in a
//                               return.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_ADDDEFAULTRETURNSTATEMENTS_H_
#define COMPILER_TRANSLATOR_TREEOPS_ADDDEFAULTRETURNSTATEMENTS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;

ANGLE_NO_DISCARD bool AddDefaultReturnStatements(TCompiler *compiler, TIntermBlock *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_ADDDEFAULTRETURNSTATEMENTS_H_
