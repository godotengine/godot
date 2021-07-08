//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveArrayLengthMethod.cpp:
//   Fold array length expressions, including cases where the "this" node has side effects.
//   Example:
//     int i = (a = b).length();
//     int j = (func()).length();
//   becomes:
//     (a = b);
//     int i = <constant array length>;
//     func();
//     int j = <constant array length>;
//
//   Must be run after SplitSequenceOperator, SimplifyLoopConditions and SeparateDeclarations steps
//   have been done to expressions containing calls of the array length method.
//
//   Does nothing to length method calls done on runtime-sized arrays.

#include "compiler/translator/tree_ops/RemoveArrayLengthMethod.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class RemoveArrayLengthTraverser : public TIntermTraverser
{
  public:
    RemoveArrayLengthTraverser() : TIntermTraverser(true, false, false), mFoundArrayLength(false) {}

    bool visitUnary(Visit visit, TIntermUnary *node) override;

    void nextIteration() { mFoundArrayLength = false; }

    bool foundArrayLength() const { return mFoundArrayLength; }

  private:
    bool mFoundArrayLength;
};

bool RemoveArrayLengthTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    // The only case where we leave array length() in place is for runtime-sized arrays.
    if (node->getOp() == EOpArrayLength && !node->getOperand()->getType().isUnsizedArray())
    {
        mFoundArrayLength = true;
        if (!node->getOperand()->hasSideEffects())
        {
            queueReplacement(node->fold(nullptr), OriginalNode::IS_DROPPED);
            return false;
        }
        insertStatementInParentBlock(node->getOperand()->deepCopy());
        TConstantUnion *constArray = new TConstantUnion[1];
        constArray->setIConst(node->getOperand()->getOutermostArraySize());
        queueReplacement(new TIntermConstantUnion(constArray, node->getType()),
                         OriginalNode::IS_DROPPED);
        return false;
    }
    return true;
}

}  // anonymous namespace

bool RemoveArrayLengthMethod(TCompiler *compiler, TIntermBlock *root)
{
    RemoveArrayLengthTraverser traverser;
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.foundArrayLength())
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.foundArrayLength());

    return true;
}

}  // namespace sh
