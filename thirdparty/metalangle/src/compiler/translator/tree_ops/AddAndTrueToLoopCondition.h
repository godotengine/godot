//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Rewrite condition in for and while loops to work around driver bug on Intel Mac.

#ifndef COMPILER_TRANSLATOR_TREEOPS_ADDANDTRUETOLOOPCONDITION_H_
#define COMPILER_TRANSLATOR_TREEOPS_ADDANDTRUETOLOOPCONDITION_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool AddAndTrueToLoopCondition(TCompiler *compiler, TIntermNode *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_ADDANDTRUETOLOOPCONDITION_H_
