//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// BreakVariableAliasingInInnerLoops.h: To optimize simple assignments, the HLSL compiler frontend
//      may record a variable as aliasing another. Sometimes the alias information gets garbled
//      so we work around this issue by breaking the aliasing chain in inner loops.

#ifndef COMPILER_TRANSLATOR_TREEOPS_BREAKVARIABLEALIASINGININNERLOOPS_H_
#define COMPILER_TRANSLATOR_TREEOPS_BREAKVARIABLEALIASINGININNERLOOPS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

ANGLE_NO_DISCARD bool BreakVariableAliasingInInnerLoops(TCompiler *compiler, TIntermNode *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_BREAKVARIABLEALIASINGININNERLOOPS_H_
