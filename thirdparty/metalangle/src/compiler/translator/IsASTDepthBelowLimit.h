//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// IsASTDepthBelowLimit: Check whether AST depth is below a specific limit.

#ifndef COMPILER_TRANSLATOR_ISASTDEPTHBELOWLIMIT_H_
#define COMPILER_TRANSLATOR_ISASTDEPTHBELOWLIMIT_H_

namespace sh
{

class TIntermNode;

bool IsASTDepthBelowLimit(TIntermNode *root, int maxDepth);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_ISASTDEPTHBELOWLIMIT_H_
