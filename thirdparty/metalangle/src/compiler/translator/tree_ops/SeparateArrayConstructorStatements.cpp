//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SeparateArrayConstructorStatements splits statements that are array constructors and drops all of
// their constant arguments. For example, a statement like:
//   int[2](0, i++);
// Will be changed to:
//   i++;

#include "compiler/translator/tree_ops/SeparateArrayConstructorStatements.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

void SplitConstructorArgs(const TIntermSequence &originalArgs, TIntermSequence *argsOut)
{
    for (TIntermNode *arg : originalArgs)
    {
        TIntermTyped *argTyped = arg->getAsTyped();
        if (argTyped->hasSideEffects())
        {
            TIntermAggregate *argAggregate = argTyped->getAsAggregate();
            if (argTyped->isArray() && argAggregate && argAggregate->isConstructor())
            {
                SplitConstructorArgs(*argAggregate->getSequence(), argsOut);
            }
            else
            {
                argsOut->push_back(argTyped);
            }
        }
    }
}

class SeparateArrayConstructorStatementsTraverser : public TIntermTraverser
{
  public:
    SeparateArrayConstructorStatementsTraverser();

    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
};

SeparateArrayConstructorStatementsTraverser::SeparateArrayConstructorStatementsTraverser()
    : TIntermTraverser(true, false, false)
{}

bool SeparateArrayConstructorStatementsTraverser::visitAggregate(Visit visit,
                                                                 TIntermAggregate *node)
{
    TIntermBlock *parentAsBlock = getParentNode()->getAsBlock();
    if (!parentAsBlock)
    {
        return false;
    }
    if (!node->isArray() || !node->isConstructor())
    {
        return false;
    }

    TIntermSequence constructorArgs;
    SplitConstructorArgs(*node->getSequence(), &constructorArgs);
    mMultiReplacements.push_back(
        NodeReplaceWithMultipleEntry(parentAsBlock, node, constructorArgs));

    return false;
}

}  // namespace

bool SeparateArrayConstructorStatements(TCompiler *compiler, TIntermBlock *root)
{
    SeparateArrayConstructorStatementsTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
