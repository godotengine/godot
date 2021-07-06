//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/IsASTDepthBelowLimit.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// Traverse the tree and compute max depth. Takes a maximum depth limit to prevent stack overflow.
class MaxDepthTraverser : public TIntermTraverser
{
  public:
    MaxDepthTraverser(int depthLimit) : TIntermTraverser(true, false, false, nullptr)
    {
        setMaxAllowedDepth(depthLimit);
    }
};

}  // anonymous namespace

bool IsASTDepthBelowLimit(TIntermNode *root, int maxDepth)
{
    MaxDepthTraverser traverser(maxDepth + 1);
    root->traverse(&traverser);

    return traverser.getMaxDepth() <= maxDepth;
}

}  // namespace sh
