//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteRepeatedAssignToSwizzled.h: Rewrite expressions that assign an assignment to a swizzled
// vector, like:
//     v.x = z = expression;
// to:
//     z = expression;
//     v.x = z;
//
// Note that this doesn't handle some corner cases: expressions nested inside other expressions,
// inside loop headers, or inside if conditions.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITEREPEATEDASSIGNTOSWIZZLED_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITEREPEATEDASSIGNTOSWIZZLED_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;

ANGLE_NO_DISCARD bool RewriteRepeatedAssignToSwizzled(TCompiler *compiler, TIntermBlock *root);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITEREPEATEDASSIGNTOSWIZZLED_H_
