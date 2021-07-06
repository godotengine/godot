//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of the integer pow expressions HLSL bug workaround.
// See header for more info.

#include "compiler/translator/tree_ops/ExpandIntegerPowExpressions.h"

#include <cmath>
#include <cstdlib>

#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class Traverser : public TIntermTraverser
{
  public:
    ANGLE_NO_DISCARD static bool Apply(TCompiler *compiler,
                                       TIntermNode *root,
                                       TSymbolTable *symbolTable);

  private:
    Traverser(TSymbolTable *symbolTable);
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    void nextIteration();

    bool mFound = false;
};

// static
bool Traverser::Apply(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable)
{
    Traverser traverser(symbolTable);
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

Traverser::Traverser(TSymbolTable *symbolTable) : TIntermTraverser(true, false, false, symbolTable)
{}

void Traverser::nextIteration()
{
    mFound = false;
}

bool Traverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (mFound)
    {
        return false;
    }

    // Test 0: skip non-pow operators.
    if (node->getOp() != EOpPow)
    {
        return true;
    }

    const TIntermSequence *sequence = node->getSequence();
    ASSERT(sequence->size() == 2u);
    const TIntermConstantUnion *constantExponent = sequence->at(1)->getAsConstantUnion();

    // Test 1: check for a single constant.
    if (!constantExponent || constantExponent->getNominalSize() != 1)
    {
        return true;
    }

    float exponentValue = constantExponent->getConstantValue()->getFConst();

    // Test 2: exponentValue is in the problematic range.
    if (exponentValue < -5.0f || exponentValue > 9.0f)
    {
        return true;
    }

    // Test 3: exponentValue is integer or pretty close to an integer.
    if (std::abs(exponentValue - std::round(exponentValue)) > 0.0001f)
    {
        return true;
    }

    // Test 4: skip -1, 0, and 1
    int exponent = static_cast<int>(std::round(exponentValue));
    int n        = std::abs(exponent);
    if (n < 2)
    {
        return true;
    }

    // Potential problem case detected, apply workaround.

    TIntermTyped *lhs = sequence->at(0)->getAsTyped();
    ASSERT(lhs);

    TIntermDeclaration *lhsVariableDeclaration = nullptr;
    TVariable *lhsVariable =
        DeclareTempVariable(mSymbolTable, lhs, EvqTemporary, &lhsVariableDeclaration);
    insertStatementInParentBlock(lhsVariableDeclaration);

    // Create a chain of n-1 multiples.
    TIntermTyped *current = CreateTempSymbolNode(lhsVariable);
    for (int i = 1; i < n; ++i)
    {
        TIntermBinary *mul = new TIntermBinary(EOpMul, current, CreateTempSymbolNode(lhsVariable));
        mul->setLine(node->getLine());
        current = mul;
    }

    // For negative pow, compute the reciprocal of the positive pow.
    if (exponent < 0)
    {
        TConstantUnion *oneVal = new TConstantUnion();
        oneVal->setFConst(1.0f);
        TIntermConstantUnion *oneNode = new TIntermConstantUnion(oneVal, node->getType());
        TIntermBinary *div            = new TIntermBinary(EOpDiv, oneNode, current);
        current                       = div;
    }

    queueReplacement(current, OriginalNode::IS_DROPPED);
    mFound = true;
    return false;
}

}  // anonymous namespace

bool ExpandIntegerPowExpressions(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable)
{
    return Traverser::Apply(compiler, root, symbolTable);
}

}  // namespace sh
