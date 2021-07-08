//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/UnfoldShortCircuitAST.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// "x || y" is equivalent to "x ? true : y".
TIntermTernary *UnfoldOR(TIntermTyped *x, TIntermTyped *y)
{
    return new TIntermTernary(x, CreateBoolNode(true), y);
}

// "x && y" is equivalent to "x ? y : false".
TIntermTernary *UnfoldAND(TIntermTyped *x, TIntermTyped *y)
{
    return new TIntermTernary(x, y, CreateBoolNode(false));
}

// This traverser identifies all the short circuit binary  nodes that need to
// be replaced, and creates the corresponding replacement nodes. However,
// the actual replacements happen after the traverse through updateTree().

class UnfoldShortCircuitASTTraverser : public TIntermTraverser
{
  public:
    UnfoldShortCircuitASTTraverser() : TIntermTraverser(true, false, false) {}

    bool visitBinary(Visit visit, TIntermBinary *) override;
};

bool UnfoldShortCircuitASTTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    TIntermTernary *replacement = nullptr;

    switch (node->getOp())
    {
        case EOpLogicalOr:
            replacement = UnfoldOR(node->getLeft(), node->getRight());
            break;
        case EOpLogicalAnd:
            replacement = UnfoldAND(node->getLeft(), node->getRight());
            break;
        default:
            break;
    }
    if (replacement)
    {
        queueReplacement(replacement, OriginalNode::IS_DROPPED);
    }
    return true;
}

}  // anonymous namespace

bool UnfoldShortCircuitAST(TCompiler *compiler, TIntermBlock *root)
{
    UnfoldShortCircuitASTTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
