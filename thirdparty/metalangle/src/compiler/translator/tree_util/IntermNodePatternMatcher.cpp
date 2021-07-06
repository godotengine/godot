//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// IntermNodePatternMatcher is a helper class for matching node trees to given patterns.
// It can be used whenever the same checks for certain node structures are common to multiple AST
// traversers.
//

#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

bool ContainsMatrixNode(const TIntermSequence &sequence)
{
    for (size_t ii = 0; ii < sequence.size(); ++ii)
    {
        TIntermTyped *node = sequence[ii]->getAsTyped();
        if (node && node->isMatrix())
            return true;
    }
    return false;
}

bool ContainsVectorNode(const TIntermSequence &sequence)
{
    for (size_t ii = 0; ii < sequence.size(); ++ii)
    {
        TIntermTyped *node = sequence[ii]->getAsTyped();
        if (node && node->isVector())
            return true;
    }
    return false;
}

}  // anonymous namespace

IntermNodePatternMatcher::IntermNodePatternMatcher(const unsigned int mask) : mMask(mask) {}

// static
bool IntermNodePatternMatcher::IsDynamicIndexingOfNonSSBOVectorOrMatrix(TIntermBinary *node)
{
    return IsDynamicIndexingOfVectorOrMatrix(node) && !IsInShaderStorageBlock(node->getLeft());
}

// static
bool IntermNodePatternMatcher::IsDynamicIndexingOfVectorOrMatrix(TIntermBinary *node)
{
    return node->getOp() == EOpIndexIndirect && !node->getLeft()->isArray() &&
           node->getLeft()->getBasicType() != EbtStruct;
}

bool IntermNodePatternMatcher::matchInternal(TIntermBinary *node, TIntermNode *parentNode)
{
    if ((mMask & kExpressionReturningArray) != 0)
    {
        if (node->isArray() && node->getOp() == EOpAssign && parentNode != nullptr &&
            !parentNode->getAsBlock())
        {
            return true;
        }
    }

    if ((mMask & kUnfoldedShortCircuitExpression) != 0)
    {
        if (node->getRight()->hasSideEffects() &&
            (node->getOp() == EOpLogicalOr || node->getOp() == EOpLogicalAnd))
        {
            return true;
        }
    }
    return false;
}

bool IntermNodePatternMatcher::match(TIntermUnary *node)
{
    if ((mMask & kArrayLengthMethod) != 0)
    {
        if (node->getOp() == EOpArrayLength)
        {
            return true;
        }
    }
    return false;
}

bool IntermNodePatternMatcher::match(TIntermBinary *node, TIntermNode *parentNode)
{
    // L-value tracking information is needed to check for dynamic indexing in L-value.
    // Traversers that don't track l-values can still use this class and match binary nodes with
    // this variation of this method if they don't need to check for dynamic indexing in l-values.
    ASSERT((mMask & kDynamicIndexingOfVectorOrMatrixInLValue) == 0);
    return matchInternal(node, parentNode);
}

bool IntermNodePatternMatcher::match(TIntermBinary *node,
                                     TIntermNode *parentNode,
                                     bool isLValueRequiredHere)
{
    if (matchInternal(node, parentNode))
    {
        return true;
    }
    if ((mMask & kDynamicIndexingOfVectorOrMatrixInLValue) != 0)
    {
        if (isLValueRequiredHere && IsDynamicIndexingOfVectorOrMatrix(node))
        {
            return true;
        }
    }
    return false;
}

bool IntermNodePatternMatcher::match(TIntermAggregate *node, TIntermNode *parentNode)
{
    if ((mMask & kExpressionReturningArray) != 0)
    {
        if (parentNode != nullptr)
        {
            TIntermBinary *parentBinary = parentNode->getAsBinaryNode();
            bool parentIsAssignment =
                (parentBinary != nullptr &&
                 (parentBinary->getOp() == EOpAssign || parentBinary->getOp() == EOpInitialize));

            if (node->getType().isArray() && !parentIsAssignment &&
                (node->isConstructor() || node->isFunctionCall()) && !parentNode->getAsBlock())
            {
                return true;
            }
        }
    }
    if ((mMask & kScalarizedVecOrMatConstructor) != 0)
    {
        if (node->getOp() == EOpConstruct)
        {
            if (node->getType().isVector() && ContainsMatrixNode(*(node->getSequence())))
            {
                return true;
            }
            else if (node->getType().isMatrix() && ContainsVectorNode(*(node->getSequence())))
            {
                return true;
            }
        }
    }
    return false;
}

bool IntermNodePatternMatcher::match(TIntermTernary *node)
{
    if ((mMask & kUnfoldedShortCircuitExpression) != 0)
    {
        return true;
    }
    return false;
}

bool IntermNodePatternMatcher::match(TIntermDeclaration *node)
{
    if ((mMask & kMultiDeclaration) != 0)
    {
        if (node->getSequence()->size() > 1)
        {
            return true;
        }
    }
    if ((mMask & kArrayDeclaration) != 0)
    {
        if (node->getSequence()->front()->getAsTyped()->getType().isStructureContainingArrays())
        {
            return true;
        }
        // Need to check from all declarators whether they are arrays since that may vary between
        // declarators.
        for (TIntermNode *declarator : *node->getSequence())
        {
            if (declarator->getAsTyped()->isArray())
            {
                return true;
            }
        }
    }
    if ((mMask & kNamelessStructDeclaration) != 0)
    {
        TIntermTyped *declarator = node->getSequence()->front()->getAsTyped();
        if (declarator->getBasicType() == EbtStruct &&
            declarator->getType().getStruct()->symbolType() == SymbolType::Empty)
        {
            return true;
        }
    }
    return false;
}

}  // namespace sh
