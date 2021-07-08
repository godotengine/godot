//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SplitSequenceOperator is an AST traverser that detects sequence operator expressions that
// go through further AST transformations that generate statements, and splits them so that
// possible side effects of earlier parts of the sequence operator expression are guaranteed to be
// evaluated before the latter parts of the sequence operator expression are evaluated.
//

#include "compiler/translator/tree_ops/SplitSequenceOperator.h"

#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class SplitSequenceOperatorTraverser : public TLValueTrackingTraverser
{
  public:
    SplitSequenceOperatorTraverser(unsigned int patternsToSplitMask, TSymbolTable *symbolTable);

    bool visitUnary(Visit visit, TIntermUnary *node) override;
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;

    void nextIteration();
    bool foundExpressionToSplit() const { return mFoundExpressionToSplit; }

  protected:
    // Marked to true once an operation that needs to be hoisted out of the expression has been
    // found. After that, no more AST updates are performed on that traversal.
    bool mFoundExpressionToSplit;
    int mInsideSequenceOperator;

    IntermNodePatternMatcher mPatternToSplitMatcher;
};

SplitSequenceOperatorTraverser::SplitSequenceOperatorTraverser(unsigned int patternsToSplitMask,
                                                               TSymbolTable *symbolTable)
    : TLValueTrackingTraverser(true, false, true, symbolTable),
      mFoundExpressionToSplit(false),
      mInsideSequenceOperator(0),
      mPatternToSplitMatcher(patternsToSplitMask)
{}

void SplitSequenceOperatorTraverser::nextIteration()
{
    mFoundExpressionToSplit = false;
    mInsideSequenceOperator = 0;
}

bool SplitSequenceOperatorTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (mFoundExpressionToSplit)
        return false;

    if (mInsideSequenceOperator > 0 && visit == PreVisit)
    {
        // Detect expressions that need to be simplified
        mFoundExpressionToSplit = mPatternToSplitMatcher.match(node, getParentNode());
        return !mFoundExpressionToSplit;
    }

    return true;
}

bool SplitSequenceOperatorTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    if (mFoundExpressionToSplit)
        return false;

    if (mInsideSequenceOperator > 0 && visit == PreVisit)
    {
        // Detect expressions that need to be simplified
        mFoundExpressionToSplit = mPatternToSplitMatcher.match(node);
        return !mFoundExpressionToSplit;
    }

    return true;
}

bool SplitSequenceOperatorTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (node->getOp() == EOpComma)
    {
        if (visit == PreVisit)
        {
            if (mFoundExpressionToSplit)
            {
                return false;
            }
            mInsideSequenceOperator++;
        }
        else if (visit == PostVisit)
        {
            // Split sequence operators starting from the outermost one to preserve correct
            // execution order.
            if (mFoundExpressionToSplit && mInsideSequenceOperator == 1)
            {
                // Move the left side operand into a separate statement in the parent block.
                TIntermSequence insertions;
                insertions.push_back(node->getLeft());
                insertStatementsInParentBlock(insertions);
                // Replace the comma node with its right side operand.
                queueReplacement(node->getRight(), OriginalNode::IS_DROPPED);
            }
            mInsideSequenceOperator--;
        }
        return true;
    }

    if (mFoundExpressionToSplit)
        return false;

    if (mInsideSequenceOperator > 0 && visit == PreVisit)
    {
        // Detect expressions that need to be simplified
        mFoundExpressionToSplit =
            mPatternToSplitMatcher.match(node, getParentNode(), isLValueRequiredHere());
        return !mFoundExpressionToSplit;
    }

    return true;
}

bool SplitSequenceOperatorTraverser::visitTernary(Visit visit, TIntermTernary *node)
{
    if (mFoundExpressionToSplit)
        return false;

    if (mInsideSequenceOperator > 0 && visit == PreVisit)
    {
        // Detect expressions that need to be simplified
        mFoundExpressionToSplit = mPatternToSplitMatcher.match(node);
        return !mFoundExpressionToSplit;
    }

    return true;
}

}  // namespace

bool SplitSequenceOperator(TCompiler *compiler,
                           TIntermNode *root,
                           int patternsToSplitMask,
                           TSymbolTable *symbolTable)
{
    SplitSequenceOperatorTraverser traverser(patternsToSplitMask, symbolTable);
    // Separate one expression at a time, and reset the traverser between iterations.
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.foundExpressionToSplit())
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.foundExpressionToSplit());

    return true;
}

}  // namespace sh
