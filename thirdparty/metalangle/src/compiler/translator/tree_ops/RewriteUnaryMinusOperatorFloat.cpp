//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/RewriteUnaryMinusOperatorFloat.h"

#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class Traverser : public TIntermTraverser
{
  public:
    ANGLE_NO_DISCARD static bool Apply(TCompiler *compiler, TIntermNode *root);

  private:
    Traverser();
    bool visitUnary(Visit visit, TIntermUnary *node) override;
    void nextIteration();

    bool mFound = false;
};

// static
bool Traverser::Apply(TCompiler *compiler, TIntermNode *root)
{
    Traverser traverser;
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.mFound)
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.mFound);

    return true;
}

Traverser::Traverser() : TIntermTraverser(true, false, false) {}

void Traverser::nextIteration()
{
    mFound = false;
}

bool Traverser::visitUnary(Visit visit, TIntermUnary *node)
{
    if (mFound)
    {
        return false;
    }

    // Detect if the current operator is unary minus operator.
    if (node->getOp() != EOpNegative)
    {
        return true;
    }

    // Detect if the current operand is a float variable.
    TIntermTyped *fValue = node->getOperand();
    if (!fValue->getType().isScalarFloat())
    {
        return true;
    }

    // 0.0 - float
    TIntermTyped *zero = CreateZeroNode(fValue->getType());
    zero->setLine(fValue->getLine());
    TIntermBinary *sub = new TIntermBinary(EOpSub, zero, fValue);
    sub->setLine(fValue->getLine());

    queueReplacement(sub, OriginalNode::IS_DROPPED);

    mFound = true;
    return false;
}

}  // anonymous namespace

bool RewriteUnaryMinusOperatorFloat(TCompiler *compiler, TIntermNode *root)
{
    return Traverser::Apply(compiler, root);
}

}  // namespace sh
