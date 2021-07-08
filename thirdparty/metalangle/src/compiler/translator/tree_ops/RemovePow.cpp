//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemovePow is an AST traverser to convert pow(x, y) built-in calls where y is a
// constant to exp2(y * log2(x)). This works around an issue in NVIDIA 311 series
// OpenGL drivers.
//

#include "compiler/translator/tree_ops/RemovePow.h"

#include "compiler/translator/InfoSink.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

bool IsProblematicPow(TIntermTyped *node)
{
    TIntermAggregate *agg = node->getAsAggregate();
    if (agg != nullptr && agg->getOp() == EOpPow)
    {
        ASSERT(agg->getSequence()->size() == 2);
        return agg->getSequence()->at(1)->getAsConstantUnion() != nullptr;
    }
    return false;
}

// Traverser that converts all pow operations simultaneously.
class RemovePowTraverser : public TIntermTraverser
{
  public:
    RemovePowTraverser(TSymbolTable *symbolTable);

    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

    void nextIteration() { mNeedAnotherIteration = false; }
    bool needAnotherIteration() const { return mNeedAnotherIteration; }

  protected:
    bool mNeedAnotherIteration;
};

RemovePowTraverser::RemovePowTraverser(TSymbolTable *symbolTable)
    : TIntermTraverser(true, false, false, symbolTable), mNeedAnotherIteration(false)
{}

bool RemovePowTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (IsProblematicPow(node))
    {
        TIntermTyped *x = node->getSequence()->at(0)->getAsTyped();
        TIntermTyped *y = node->getSequence()->at(1)->getAsTyped();

        TIntermSequence *logArgs = new TIntermSequence();
        logArgs->push_back(x);
        TIntermTyped *log = CreateBuiltInFunctionCallNode("log2", logArgs, *mSymbolTable, 100);
        log->setLine(node->getLine());

        TOperator op       = TIntermBinary::GetMulOpBasedOnOperands(y->getType(), log->getType());
        TIntermBinary *mul = new TIntermBinary(op, y, log);
        mul->setLine(node->getLine());

        TIntermSequence *expArgs = new TIntermSequence();
        expArgs->push_back(mul);
        TIntermTyped *exp = CreateBuiltInFunctionCallNode("exp2", expArgs, *mSymbolTable, 100);
        exp->setLine(node->getLine());

        queueReplacement(exp, OriginalNode::IS_DROPPED);

        // If the x parameter also needs to be replaced, we need to do that in another traversal,
        // since it's parent node will change in a way that's not handled correctly by updateTree().
        if (IsProblematicPow(x))
        {
            mNeedAnotherIteration = true;
            return false;
        }
    }
    return true;
}

}  // namespace

bool RemovePow(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable)
{
    RemovePowTraverser traverser(symbolTable);
    // Iterate as necessary, and reset the traverser between iterations.
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (!traverser.updateTree(compiler, root))
        {
            return false;
        }
    } while (traverser.needAnotherIteration());

    return true;
}

}  // namespace sh
