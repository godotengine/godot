//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of evaluating unary integer variable bug workaround.
// See header for more info.

#include "compiler/translator/tree_ops/RewriteUnaryMinusOperatorInt.h"

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

    // Decide if the current unary operator is unary minus.
    if (node->getOp() != EOpNegative)
    {
        return true;
    }

    // Decide if the current operand is an integer variable.
    TIntermTyped *opr = node->getOperand();
    if (!opr->getType().isScalarInt())
    {
        return true;
    }

    // Potential problem case detected, apply workaround: -(int) -> ~(int) + 1.
    // ~(int)
    TIntermUnary *bitwiseNot = new TIntermUnary(EOpBitwiseNot, opr, nullptr);
    bitwiseNot->setLine(opr->getLine());

    // Constant 1 (or 1u)
    TConstantUnion *one = new TConstantUnion();
    if (opr->getType().getBasicType() == EbtInt)
    {
        one->setIConst(1);
    }
    else
    {
        one->setUConst(1u);
    }
    TIntermConstantUnion *oneNode =
        new TIntermConstantUnion(one, TType(opr->getBasicType(), opr->getPrecision(), EvqConst));
    oneNode->setLine(opr->getLine());

    // ~(int) + 1
    TIntermBinary *add = new TIntermBinary(EOpAdd, bitwiseNot, oneNode);
    add->setLine(opr->getLine());

    queueReplacement(add, OriginalNode::IS_DROPPED);

    mFound = true;
    return false;
}

}  // anonymous namespace

bool RewriteUnaryMinusOperatorInt(TCompiler *compiler, TIntermNode *root)
{
    return Traverser::Apply(compiler, root);
}

}  // namespace sh
