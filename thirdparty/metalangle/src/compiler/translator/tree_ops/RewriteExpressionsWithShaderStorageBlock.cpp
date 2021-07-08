//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteExpressionsWithShaderStorageBlock rewrites the expressions that contain shader storage
// block calls into several simple ones that can be easily handled in the HLSL translator. After the
// AST pass, all ssbo related blocks will be like below:
//     ssbo_access_chain = ssbo_access_chain;
//     ssbo_access_chain = expr_no_ssbo;
//     lvalue_no_ssbo    = ssbo_access_chain;
//

#include "compiler/translator/tree_ops/RewriteExpressionsWithShaderStorageBlock.h"

#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{
namespace
{

bool IsIncrementOrDecrementOperator(TOperator op)
{
    switch (op)
    {
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpPreIncrement:
        case EOpPreDecrement:
            return true;
        default:
            return false;
    }
}

bool IsCompoundAssignment(TOperator op)
{
    switch (op)
    {
        case EOpAddAssign:
        case EOpSubAssign:
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
        case EOpDivAssign:
        case EOpIModAssign:
        case EOpBitShiftLeftAssign:
        case EOpBitShiftRightAssign:
        case EOpBitwiseAndAssign:
        case EOpBitwiseXorAssign:
        case EOpBitwiseOrAssign:
            return true;
        default:
            return false;
    }
}

// EOpIndexDirect, EOpIndexIndirect, EOpIndexDirectStruct, EOpIndexDirectInterfaceBlock belong to
// operators in SSBO access chain.
bool IsReadonlyBinaryOperatorNotInSSBOAccessChain(TOperator op)
{
    switch (op)
    {
        case EOpComma:
        case EOpAdd:
        case EOpSub:
        case EOpMul:
        case EOpDiv:
        case EOpIMod:
        case EOpBitShiftLeft:
        case EOpBitShiftRight:
        case EOpBitwiseAnd:
        case EOpBitwiseXor:
        case EOpBitwiseOr:
        case EOpEqual:
        case EOpNotEqual:
        case EOpLessThan:
        case EOpGreaterThan:
        case EOpLessThanEqual:
        case EOpGreaterThanEqual:
        case EOpVectorTimesScalar:
        case EOpMatrixTimesScalar:
        case EOpVectorTimesMatrix:
        case EOpMatrixTimesVector:
        case EOpMatrixTimesMatrix:
        case EOpLogicalOr:
        case EOpLogicalXor:
        case EOpLogicalAnd:
            return true;
        default:
            return false;
    }
}

bool HasSSBOAsFunctionArgument(TIntermSequence *arguments)
{
    for (TIntermNode *arg : *arguments)
    {
        TIntermTyped *typedArg = arg->getAsTyped();
        if (IsInShaderStorageBlock(typedArg))
        {
            return true;
        }
    }
    return false;
}

class RewriteExpressionsWithShaderStorageBlockTraverser : public TIntermTraverser
{
  public:
    RewriteExpressionsWithShaderStorageBlockTraverser(TSymbolTable *symbolTable);
    void nextIteration();
    bool foundSSBO() const { return mFoundSSBO; }

  private:
    bool visitBinary(Visit, TIntermBinary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitUnary(Visit visit, TIntermUnary *node) override;

    TIntermSymbol *insertInitStatementAndReturnTempSymbol(TIntermTyped *node,
                                                          TIntermSequence *insertions);

    bool mFoundSSBO;
};

RewriteExpressionsWithShaderStorageBlockTraverser::
    RewriteExpressionsWithShaderStorageBlockTraverser(TSymbolTable *symbolTable)
    : TIntermTraverser(true, true, false, symbolTable), mFoundSSBO(false)
{}

TIntermSymbol *
RewriteExpressionsWithShaderStorageBlockTraverser::insertInitStatementAndReturnTempSymbol(
    TIntermTyped *node,
    TIntermSequence *insertions)
{
    TIntermDeclaration *variableDeclaration;
    TVariable *tempVariable =
        DeclareTempVariable(mSymbolTable, node, EvqTemporary, &variableDeclaration);

    insertions->push_back(variableDeclaration);
    return CreateTempSymbolNode(tempVariable);
}

bool RewriteExpressionsWithShaderStorageBlockTraverser::visitBinary(Visit visit,
                                                                    TIntermBinary *node)
{
    // Make sure that the expression is caculated from left to right.
    if (visit != InVisit)
    {
        return true;
    }

    if (mFoundSSBO)
    {
        return false;
    }

    bool rightSSBO = IsInShaderStorageBlock(node->getRight());
    bool leftSSBO  = IsInShaderStorageBlock(node->getLeft());
    if (!leftSSBO && !rightSSBO)
    {
        return true;
    }

    // case 1: Compound assigment operator
    //  original:
    //      lssbo += expr;
    //  new:
    //      var rvalue = expr;
    //      var temp = lssbo;
    //      temp += rvalue;
    //      lssbo = temp;
    //
    //  original:
    //      lvalue_no_ssbo += rssbo;
    //  new:
    //      var rvalue = rssbo;
    //      lvalue_no_ssbo += rvalue;
    if (IsCompoundAssignment(node->getOp()))
    {
        mFoundSSBO = true;
        TIntermSequence insertions;
        TIntermTyped *rightNode =
            insertInitStatementAndReturnTempSymbol(node->getRight(), &insertions);
        if (leftSSBO)
        {
            TIntermSymbol *tempSymbol =
                insertInitStatementAndReturnTempSymbol(node->getLeft()->deepCopy(), &insertions);
            TIntermBinary *tempCompoundOperate =
                new TIntermBinary(node->getOp(), tempSymbol->deepCopy(), rightNode->deepCopy());
            insertions.push_back(tempCompoundOperate);
            insertStatementsInParentBlock(insertions);

            TIntermBinary *assignTempValueToSSBO =
                new TIntermBinary(EOpAssign, node->getLeft(), tempSymbol->deepCopy());
            queueReplacement(assignTempValueToSSBO, OriginalNode::IS_DROPPED);
        }
        else
        {
            insertStatementsInParentBlock(insertions);
            TIntermBinary *compoundAssignRValueToLValue =
                new TIntermBinary(node->getOp(), node->getLeft(), rightNode->deepCopy());
            queueReplacement(compoundAssignRValueToLValue, OriginalNode::IS_DROPPED);
        }
    }
    // case 2: Readonly binary operator
    //  original:
    //      ssbo0 + ssbo1 + ssbo2;
    //  new:
    //      var temp0 = ssbo0;
    //      var temp1 = ssbo1;
    //      var temp2 = ssbo2;
    //      temp0 + temp1 + temp2;
    else if (IsReadonlyBinaryOperatorNotInSSBOAccessChain(node->getOp()) && (leftSSBO || rightSSBO))
    {
        mFoundSSBO              = true;
        TIntermTyped *rightNode = node->getRight();
        TIntermTyped *leftNode  = node->getLeft();
        TIntermSequence insertions;
        if (rightSSBO)
        {
            rightNode = insertInitStatementAndReturnTempSymbol(node->getRight(), &insertions);
        }
        if (leftSSBO)
        {
            leftNode = insertInitStatementAndReturnTempSymbol(node->getLeft(), &insertions);
        }

        insertStatementsInParentBlock(insertions);
        TIntermBinary *newExpr =
            new TIntermBinary(node->getOp(), leftNode->deepCopy(), rightNode->deepCopy());
        queueReplacement(newExpr, OriginalNode::IS_DROPPED);
    }
    return !mFoundSSBO;
}

// case 3: ssbo as the argument of aggregate type
//  original:
//      foo(ssbo);
//  new:
//      var tempArg = ssbo;
//      foo(tempArg);
//      ssbo = tempArg;  (Optional based on whether ssbo is an out|input argument)
//
//  original:
//      foo(ssbo) * expr;
//  new:
//      var tempArg = ssbo;
//      var tempReturn = foo(tempArg);
//      ssbo = tempArg;  (Optional based on whether ssbo is an out|input argument)
//      tempReturn * expr;
bool RewriteExpressionsWithShaderStorageBlockTraverser::visitAggregate(Visit visit,
                                                                       TIntermAggregate *node)
{
    // Make sure that visitAggregate is only executed once for same node.
    if (visit != PreVisit)
    {
        return true;
    }

    if (mFoundSSBO)
    {
        return false;
    }

    // We still need to process the ssbo as the non-first argument of atomic memory functions.
    if (IsAtomicFunction(node->getOp()) &&
        IsInShaderStorageBlock((*node->getSequence())[0]->getAsTyped()))
    {
        return true;
    }

    if (!HasSSBOAsFunctionArgument(node->getSequence()))
    {
        return true;
    }

    mFoundSSBO = true;
    TIntermSequence insertions;
    TIntermSequence readBackToSSBOs;
    TIntermSequence *originalArguments = node->getSequence();
    for (size_t i = 0; i < node->getChildCount(); ++i)
    {
        TIntermTyped *ssboArgument = (*originalArguments)[i]->getAsTyped();
        if (IsInShaderStorageBlock(ssboArgument))
        {
            TIntermSymbol *argumentCopy =
                insertInitStatementAndReturnTempSymbol(ssboArgument, &insertions);
            if (node->getFunction() != nullptr)
            {
                TQualifier qual = node->getFunction()->getParam(i)->getType().getQualifier();
                if (qual == EvqInOut || qual == EvqOut)
                {
                    TIntermBinary *readBackToSSBO = new TIntermBinary(
                        EOpAssign, ssboArgument->deepCopy(), argumentCopy->deepCopy());
                    readBackToSSBOs.push_back(readBackToSSBO);
                }
            }
            node->replaceChildNode(ssboArgument, argumentCopy);
        }
    }

    TIntermBlock *parentBlock = getParentNode()->getAsBlock();
    if (parentBlock)
    {
        // Aggregate node is as a single sentence.
        insertions.push_back(node);
        if (!readBackToSSBOs.empty())
        {
            insertions.insert(insertions.end(), readBackToSSBOs.begin(), readBackToSSBOs.end());
        }
        mMultiReplacements.push_back(NodeReplaceWithMultipleEntry(parentBlock, node, insertions));
    }
    else
    {
        // Aggregate node is inside an expression.
        TIntermSymbol *tempSymbol = insertInitStatementAndReturnTempSymbol(node, &insertions);
        if (!readBackToSSBOs.empty())
        {
            insertions.insert(insertions.end(), readBackToSSBOs.begin(), readBackToSSBOs.end());
        }
        insertStatementsInParentBlock(insertions);
        queueReplacement(tempSymbol->deepCopy(), OriginalNode::IS_DROPPED);
    }

    return false;
}

bool RewriteExpressionsWithShaderStorageBlockTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    if (mFoundSSBO)
    {
        return false;
    }

    if (!IsInShaderStorageBlock(node->getOperand()))
    {
        return true;
    }

    // .length() is processed in OutputHLSL.
    if (node->getOp() == EOpArrayLength)
    {
        return true;
    }

    mFoundSSBO = true;

    // case 4: ssbo as the operand of ++/--
    //  original:
    //      ++ssbo * expr;
    //  new:
    //      var temp1 = ssbo;
    //      var temp2 = ++temp1;
    //      ssbo = temp1;
    //      temp2 * expr;
    if (IsIncrementOrDecrementOperator(node->getOp()))
    {
        TIntermSequence insertions;
        TIntermSymbol *temp1 =
            insertInitStatementAndReturnTempSymbol(node->getOperand(), &insertions);
        TIntermUnary *newUnary = new TIntermUnary(node->getOp(), temp1->deepCopy(), nullptr);
        TIntermSymbol *temp2   = insertInitStatementAndReturnTempSymbol(newUnary, &insertions);
        TIntermBinary *readBackToSSBO =
            new TIntermBinary(EOpAssign, node->getOperand()->deepCopy(), temp1->deepCopy());
        insertions.push_back(readBackToSSBO);
        insertStatementsInParentBlock(insertions);
        queueReplacement(temp2->deepCopy(), OriginalNode::IS_DROPPED);
    }
    // case 5: ssbo as the operand of readonly unary operator
    //  original:
    //      ~ssbo * expr;
    //  new:
    //      var temp = ssbo;
    //      ~temp * expr;
    else
    {
        TIntermSequence insertions;
        TIntermSymbol *temp =
            insertInitStatementAndReturnTempSymbol(node->getOperand(), &insertions);
        insertStatementsInParentBlock(insertions);
        node->replaceChildNode(node->getOperand(), temp->deepCopy());
    }
    return false;
}

void RewriteExpressionsWithShaderStorageBlockTraverser::nextIteration()
{
    mFoundSSBO = false;
}

}  // anonymous namespace

bool RewriteExpressionsWithShaderStorageBlock(TCompiler *compiler,
                                              TIntermNode *root,
                                              TSymbolTable *symbolTable)
{
    RewriteExpressionsWithShaderStorageBlockTraverser traverser(symbolTable);
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.foundSSBO())
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.foundSSBO());

    return true;
}
}  // namespace sh
