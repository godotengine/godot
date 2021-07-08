//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// NodeSearch.h: Utilities for searching translator node graphs
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_NODESEARCH_H_
#define COMPILER_TRANSLATOR_TREEUTIL_NODESEARCH_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

template <class Parent>
class NodeSearchTraverser : public TIntermTraverser
{
  public:
    NodeSearchTraverser() : TIntermTraverser(true, false, false), mFound(false) {}

    bool found() const { return mFound; }

    static bool search(TIntermNode *node)
    {
        Parent searchTraverser;
        node->traverse(&searchTraverser);
        return searchTraverser.found();
    }

  protected:
    bool mFound;
};

class FindDiscard : public NodeSearchTraverser<FindDiscard>
{
  public:
    virtual bool visitBranch(Visit visit, TIntermBranch *node)
    {
        switch (node->getFlowOp())
        {
            case EOpKill:
                mFound = true;
                break;

            default:
                break;
        }

        return !mFound;
    }
};
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_NODESEARCH_H_
