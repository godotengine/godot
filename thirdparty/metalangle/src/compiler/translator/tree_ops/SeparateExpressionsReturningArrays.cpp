//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SeparateExpressionsReturningArrays splits array-returning expressions that are not array names
// from more complex expressions, assigning them to a temporary variable a#.
// Examples where a, b and c are all arrays:
// (a = b) == (a = c) is split into a = b; type[n] a1 = a; a = c; type[n] a2 = a; a1 == a2;
// type d = type[n](...)[i]; is split into type[n] a1 = type[n](...); type d = a1[i];

#include "compiler/translator/tree_ops/SeparateExpressionsReturningArrays.h"

#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// Traverser that separates one array expression into a statement at a time.
class SeparateExpressionsTraverser : public TIntermTraverser
{
  public:
    SeparateExpressionsTraverser(TSymbolTable *symbolTable);

    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

    void nextIteration();
    bool foundArrayExpression() const { return mFoundArrayExpression; }

  protected:
    // Marked to true once an operation that needs to be hoisted out of the expression has been
    // found. After that, no more AST updates are performed on that traversal.
    bool mFoundArrayExpression;

    IntermNodePatternMatcher mPatternToSeparateMatcher;
};

SeparateExpressionsTraverser::SeparateExpressionsTraverser(TSymbolTable *symbolTable)
    : TIntermTraverser(true, false, false, symbolTable),
      mFoundArrayExpression(false),
      mPatternToSeparateMatcher(IntermNodePatternMatcher::kExpressionReturningArray)
{}

// Performs a shallow copy of an assignment node.
// These shallow copies are useful when a node gets inserted into an aggregate node
// and also needs to be replaced in its original location by a different node.
TIntermBinary *CopyAssignmentNode(TIntermBinary *node)
{
    return new TIntermBinary(node->getOp(), node->getLeft(), node->getRight());
}

bool SeparateExpressionsTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (mFoundArrayExpression)
        return false;

    // Return if the expression is not an array or if we're not inside a complex expression.
    if (!mPatternToSeparateMatcher.match(node, getParentNode()))
        return true;

    ASSERT(node->getOp() == EOpAssign);

    mFoundArrayExpression = true;

    TIntermSequence insertions;
    insertions.push_back(CopyAssignmentNode(node));
    // TODO(oetuaho): In some cases it would be more optimal to not add the temporary node, but just
    // use the original target of the assignment. Care must be taken so that this doesn't happen
    // when the same array symbol is a target of assignment more than once in one expression.
    TIntermDeclaration *arrayVariableDeclaration;
    TVariable *arrayVariable =
        DeclareTempVariable(mSymbolTable, node->getLeft(), EvqTemporary, &arrayVariableDeclaration);
    insertions.push_back(arrayVariableDeclaration);
    insertStatementsInParentBlock(insertions);

    queueReplacement(CreateTempSymbolNode(arrayVariable), OriginalNode::IS_DROPPED);

    return false;
}

bool SeparateExpressionsTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (mFoundArrayExpression)
        return false;  // No need to traverse further

    if (!mPatternToSeparateMatcher.match(node, getParentNode()))
        return true;

    ASSERT(node->isConstructor() || node->getOp() == EOpCallFunctionInAST);

    mFoundArrayExpression = true;

    TIntermDeclaration *arrayVariableDeclaration;
    TVariable *arrayVariable = DeclareTempVariable(mSymbolTable, node->shallowCopy(), EvqTemporary,
                                                   &arrayVariableDeclaration);
    insertStatementInParentBlock(arrayVariableDeclaration);

    queueReplacement(CreateTempSymbolNode(arrayVariable), OriginalNode::IS_DROPPED);

    return false;
}

void SeparateExpressionsTraverser::nextIteration()
{
    mFoundArrayExpression = false;
}

}  // namespace

bool SeparateExpressionsReturningArrays(TCompiler *compiler,
                                        TIntermNode *root,
                                        TSymbolTable *symbolTable)
{
    SeparateExpressionsTraverser traverser(symbolTable);
    // Separate one expression at a time, and reset the traverser between iterations.
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.foundArrayExpression())
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.foundArrayExpression());

    return true;
}

}  // namespace sh
