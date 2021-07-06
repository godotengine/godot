//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of the function RewriteAtomicFunctionExpressions.
// See the header for more details.

#include "RewriteAtomicFunctionExpressions.h"

#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{
namespace
{
// Traverser that simplifies all the atomic function expressions into the ones that can be directly
// translated into HLSL.
//
// case 1 (only for atomicExchange and atomicCompSwap):
//  original:
//      atomicExchange(counter, newValue);
//  new:
//      tempValue = atomicExchange(counter, newValue);
//
// case 2 (atomic function, temporary variable required):
//  original:
//      value = atomicAdd(counter, 1) * otherValue;
//      someArray[atomicAdd(counter, 1)] = someOtherValue;
//  new:
//      value = ((tempValue = atomicAdd(counter, 1)), tempValue) * otherValue;
//      someArray[((tempValue = atomicAdd(counter, 1)), tempValue)] = someOtherValue;
//
// case 3 (atomic function used directly initialize a variable):
//  original:
//      int value = atomicAdd(counter, 1);
//  new:
//      tempValue = atomicAdd(counter, 1);
//      int value = tempValue;
//
class RewriteAtomicFunctionExpressionsTraverser : public TIntermTraverser
{
  public:
    RewriteAtomicFunctionExpressionsTraverser(TSymbolTable *symbolTable, int shaderVersion);

    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitBlock(Visit visit, TIntermBlock *node) override;

  private:
    static bool IsAtomicExchangeOrCompSwapNoReturnValue(TIntermAggregate *node,
                                                        TIntermNode *parentNode);
    static bool IsAtomicFunctionInsideExpression(TIntermAggregate *node, TIntermNode *parentNode);

    void rewriteAtomicFunctionCallNode(TIntermAggregate *oldAtomicFunctionNode);

    const TVariable *getTempVariable(const TType *type);

    int mShaderVersion;
    TIntermSequence mTempVariables;
};

RewriteAtomicFunctionExpressionsTraverser::RewriteAtomicFunctionExpressionsTraverser(
    TSymbolTable *symbolTable,
    int shaderVersion)
    : TIntermTraverser(false, false, true, symbolTable), mShaderVersion(shaderVersion)
{}

void RewriteAtomicFunctionExpressionsTraverser::rewriteAtomicFunctionCallNode(
    TIntermAggregate *oldAtomicFunctionNode)
{
    ASSERT(oldAtomicFunctionNode);

    const TVariable *returnVariable = getTempVariable(&oldAtomicFunctionNode->getType());

    TIntermBinary *rewrittenNode = new TIntermBinary(
        TOperator::EOpAssign, CreateTempSymbolNode(returnVariable), oldAtomicFunctionNode);

    auto *parentNode = getParentNode();

    auto *parentBinary = parentNode->getAsBinaryNode();
    if (parentBinary && parentBinary->getOp() == EOpInitialize)
    {
        insertStatementInParentBlock(rewrittenNode);
        queueReplacement(CreateTempSymbolNode(returnVariable), OriginalNode::IS_DROPPED);
    }
    else
    {
        // As all atomic function assignment will be converted to the last argument of an
        // interlocked function, if we need the return value, assignment needs to be wrapped with
        // the comma operator and the temporary variables.
        if (!parentNode->getAsBlock())
        {
            rewrittenNode = TIntermBinary::CreateComma(
                rewrittenNode, new TIntermSymbol(returnVariable), mShaderVersion);
        }

        queueReplacement(rewrittenNode, OriginalNode::IS_DROPPED);
    }
}

const TVariable *RewriteAtomicFunctionExpressionsTraverser::getTempVariable(const TType *type)
{
    TIntermDeclaration *variableDeclaration;
    TVariable *returnVariable =
        DeclareTempVariable(mSymbolTable, type, EvqTemporary, &variableDeclaration);
    mTempVariables.push_back(variableDeclaration);
    return returnVariable;
}

bool RewriteAtomicFunctionExpressionsTraverser::IsAtomicExchangeOrCompSwapNoReturnValue(
    TIntermAggregate *node,
    TIntermNode *parentNode)
{
    ASSERT(node);
    return (node->getOp() == EOpAtomicExchange || node->getOp() == EOpAtomicCompSwap) &&
           parentNode && parentNode->getAsBlock();
}

bool RewriteAtomicFunctionExpressionsTraverser::IsAtomicFunctionInsideExpression(
    TIntermAggregate *node,
    TIntermNode *parentNode)
{
    ASSERT(node);
    // We only need to handle atomic functions with a parent that it is not block nodes. If the
    // parent node is block, it means that the atomic function is not inside an expression.
    if (!IsAtomicFunction(node->getOp()) || parentNode->getAsBlock())
    {
        return false;
    }

    auto *parentAsBinary = parentNode->getAsBinaryNode();
    // Assignments are handled in OutputHLSL
    return !parentAsBinary || parentAsBinary->getOp() != EOpAssign;
}

bool RewriteAtomicFunctionExpressionsTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    ASSERT(visit == PostVisit);
    // Skip atomic memory functions for SSBO. They will be processed in the OutputHLSL traverser.
    if (IsAtomicFunction(node->getOp()) &&
        IsInShaderStorageBlock((*node->getSequence())[0]->getAsTyped()))
    {
        return false;
    }

    TIntermNode *parentNode = getParentNode();
    if (IsAtomicExchangeOrCompSwapNoReturnValue(node, parentNode) ||
        IsAtomicFunctionInsideExpression(node, parentNode))
    {
        rewriteAtomicFunctionCallNode(node);
    }

    return true;
}

bool RewriteAtomicFunctionExpressionsTraverser::visitBlock(Visit visit, TIntermBlock *node)
{
    ASSERT(visit == PostVisit);

    if (!mTempVariables.empty() && getParentNode()->getAsFunctionDefinition())
    {
        insertStatementsInBlockAtPosition(node, 0, mTempVariables, TIntermSequence());
        mTempVariables.clear();
    }

    return true;
}

}  // anonymous namespace

bool RewriteAtomicFunctionExpressions(TCompiler *compiler,
                                      TIntermNode *root,
                                      TSymbolTable *symbolTable,
                                      int shaderVersion)
{
    RewriteAtomicFunctionExpressionsTraverser traverser(symbolTable, shaderVersion);
    traverser.traverse(root);
    return traverser.updateTree(compiler, root);
}
}  // namespace sh
