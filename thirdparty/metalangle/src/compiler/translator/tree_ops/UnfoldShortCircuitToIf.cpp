//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// UnfoldShortCircuitToIf is an AST traverser to convert short-circuiting operators to if-else
// statements.
// The results are assigned to s# temporaries, which are used by the main translator instead of
// the original expression.
//

#include "compiler/translator/tree_ops/UnfoldShortCircuitToIf.h"

#include "compiler/translator/StaticType.h"
#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// Traverser that unfolds one short-circuiting operation at a time.
class UnfoldShortCircuitTraverser : public TIntermTraverser
{
  public:
    UnfoldShortCircuitTraverser(TSymbolTable *symbolTable);

    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;

    void nextIteration();
    bool foundShortCircuit() const { return mFoundShortCircuit; }

  protected:
    // Marked to true once an operation that needs to be unfolded has been found.
    // After that, no more unfolding is performed on that traversal.
    bool mFoundShortCircuit;

    IntermNodePatternMatcher mPatternToUnfoldMatcher;
};

UnfoldShortCircuitTraverser::UnfoldShortCircuitTraverser(TSymbolTable *symbolTable)
    : TIntermTraverser(true, false, true, symbolTable),
      mFoundShortCircuit(false),
      mPatternToUnfoldMatcher(IntermNodePatternMatcher::kUnfoldedShortCircuitExpression)
{}

bool UnfoldShortCircuitTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (mFoundShortCircuit)
        return false;

    if (visit != PreVisit)
        return true;

    if (!mPatternToUnfoldMatcher.match(node, getParentNode()))
        return true;

    // If our right node doesn't have side effects, we know we don't need to unfold this
    // expression: there will be no short-circuiting side effects to avoid
    // (note: unfolding doesn't depend on the left node -- it will always be evaluated)
    ASSERT(node->getRight()->hasSideEffects());

    mFoundShortCircuit = true;

    switch (node->getOp())
    {
        case EOpLogicalOr:
        {
            // "x || y" is equivalent to "x ? true : y", which unfolds to "bool s; if(x) s = true;
            // else s = y;",
            // and then further simplifies down to "bool s = x; if(!s) s = y;".

            TIntermSequence insertions;
            const TType *boolType = StaticType::Get<EbtBool, EbpUndefined, EvqTemporary, 1, 1>();
            TVariable *resultVariable = CreateTempVariable(mSymbolTable, boolType);

            ASSERT(node->getLeft()->getType() == *boolType);
            insertions.push_back(CreateTempInitDeclarationNode(resultVariable, node->getLeft()));

            TIntermBlock *assignRightBlock = new TIntermBlock();
            ASSERT(node->getRight()->getType() == *boolType);
            assignRightBlock->getSequence()->push_back(
                CreateTempAssignmentNode(resultVariable, node->getRight()));

            TIntermUnary *notTempSymbol =
                new TIntermUnary(EOpLogicalNot, CreateTempSymbolNode(resultVariable), nullptr);
            TIntermIfElse *ifNode = new TIntermIfElse(notTempSymbol, assignRightBlock, nullptr);
            insertions.push_back(ifNode);

            insertStatementsInParentBlock(insertions);

            queueReplacement(CreateTempSymbolNode(resultVariable), OriginalNode::IS_DROPPED);
            return false;
        }
        case EOpLogicalAnd:
        {
            // "x && y" is equivalent to "x ? y : false", which unfolds to "bool s; if(x) s = y;
            // else s = false;",
            // and then further simplifies down to "bool s = x; if(s) s = y;".
            TIntermSequence insertions;
            const TType *boolType = StaticType::Get<EbtBool, EbpUndefined, EvqTemporary, 1, 1>();
            TVariable *resultVariable = CreateTempVariable(mSymbolTable, boolType);

            ASSERT(node->getLeft()->getType() == *boolType);
            insertions.push_back(CreateTempInitDeclarationNode(resultVariable, node->getLeft()));

            TIntermBlock *assignRightBlock = new TIntermBlock();
            ASSERT(node->getRight()->getType() == *boolType);
            assignRightBlock->getSequence()->push_back(
                CreateTempAssignmentNode(resultVariable, node->getRight()));

            TIntermIfElse *ifNode =
                new TIntermIfElse(CreateTempSymbolNode(resultVariable), assignRightBlock, nullptr);
            insertions.push_back(ifNode);

            insertStatementsInParentBlock(insertions);

            queueReplacement(CreateTempSymbolNode(resultVariable), OriginalNode::IS_DROPPED);
            return false;
        }
        default:
            UNREACHABLE();
            return true;
    }
}

bool UnfoldShortCircuitTraverser::visitTernary(Visit visit, TIntermTernary *node)
{
    if (mFoundShortCircuit)
        return false;

    if (visit != PreVisit)
        return true;

    if (!mPatternToUnfoldMatcher.match(node))
        return true;

    mFoundShortCircuit = true;

    // Unfold "b ? x : y" into "type s; if(b) s = x; else s = y;"
    TIntermSequence insertions;
    TIntermDeclaration *tempDeclaration = nullptr;
    TVariable *resultVariable = DeclareTempVariable(mSymbolTable, new TType(node->getType()),
                                                    EvqTemporary, &tempDeclaration);
    insertions.push_back(tempDeclaration);

    TIntermBlock *trueBlock = new TIntermBlock();
    TIntermBinary *trueAssignment =
        CreateTempAssignmentNode(resultVariable, node->getTrueExpression());
    trueBlock->getSequence()->push_back(trueAssignment);

    TIntermBlock *falseBlock = new TIntermBlock();
    TIntermBinary *falseAssignment =
        CreateTempAssignmentNode(resultVariable, node->getFalseExpression());
    falseBlock->getSequence()->push_back(falseAssignment);

    TIntermIfElse *ifNode =
        new TIntermIfElse(node->getCondition()->getAsTyped(), trueBlock, falseBlock);
    insertions.push_back(ifNode);

    insertStatementsInParentBlock(insertions);

    TIntermSymbol *ternaryResult = CreateTempSymbolNode(resultVariable);
    queueReplacement(ternaryResult, OriginalNode::IS_DROPPED);

    return false;
}

void UnfoldShortCircuitTraverser::nextIteration()
{
    mFoundShortCircuit = false;
}

}  // namespace

bool UnfoldShortCircuitToIf(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable)
{
    UnfoldShortCircuitTraverser traverser(symbolTable);
    // Unfold one operator at a time, and reset the traverser between iterations.
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.foundShortCircuit())
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.foundShortCircuit());

    return true;
}

}  // namespace sh
