//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEINVARIANTDECLARATION_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEINVARIANTDECLARATION_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool RemoveInvariantDeclaration(TCompiler *compiler, TIntermNode *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEINVARIANTDECLARATION_H_
