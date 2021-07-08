// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VectorizeVectorScalarArithmetic.cpp: Turn some arithmetic operations that operate on a float
// vector-scalar pair into vector-vector operations. This is done recursively. Some scalar binary
// operations inside vector constructors are also turned into vector operations.
//
// This is targeted to work around a bug in NVIDIA OpenGL drivers that was reproducible on NVIDIA
// driver version 387.92. It works around the most common occurrences of the bug.

#include "compiler/translator/tree_ops/VectorizeVectorScalarArithmetic.h"

#include <set>

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class VectorizeVectorScalarArithmeticTraverser : public TIntermTraverser
{
  public:
    VectorizeVectorScalarArithmeticTraverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, false, false, symbolTable), mReplaced(false)
    {}

    bool didReplaceScalarsWithVectors() { return mReplaced; }
    void nextIteration()
    {
        mReplaced = false;
        mModifiedBlocks.clear();
    }

  protected:
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

  private:
    // These helpers should only be called from visitAggregate when visiting a constructor.
    // argBinary is the only argument of the constructor.
    void replaceMathInsideConstructor(TIntermAggregate *node, TIntermBinary *argBinary);
    void replaceAssignInsideConstructor(const TIntermAggregate *node,
                                        const TIntermBinary *argBinary);

    static TIntermTyped *Vectorize(TIntermTyped *node,
                                   TType vectorType,
                                   TIntermTraverser::OriginalNode *originalNodeFate);

    bool mReplaced;
    std::set<const TIntermBlock *> mModifiedBlocks;
};

TIntermTyped *VectorizeVectorScalarArithmeticTraverser::Vectorize(
    TIntermTyped *node,
    TType vectorType,
    TIntermTraverser::OriginalNode *originalNodeFate)
{
    ASSERT(node->isScalar());
    vectorType.setQualifier(EvqTemporary);
    TIntermSequence vectorConstructorArgs;
    vectorConstructorArgs.push_back(node);
    TIntermAggregate *vectorized =
        TIntermAggregate::CreateConstructor(vectorType, &vectorConstructorArgs);
    TIntermTyped *vectorizedFolded = vectorized->fold(nullptr);
    if (originalNodeFate != nullptr)
    {
        if (vectorizedFolded != vectorized)
        {
            *originalNodeFate = OriginalNode::IS_DROPPED;
        }
        else
        {
            *originalNodeFate = OriginalNode::BECOMES_CHILD;
        }
    }
    return vectorizedFolded;
}

bool VectorizeVectorScalarArithmeticTraverser::visitBinary(Visit /*visit*/, TIntermBinary *node)
{
    TIntermTyped *left  = node->getLeft();
    TIntermTyped *right = node->getRight();
    ASSERT(left);
    ASSERT(right);
    switch (node->getOp())
    {
        case EOpAdd:
        case EOpAddAssign:
            // Only these specific ops are necessary to turn into vector ops.
            break;
        default:
            return true;
    }
    if (node->getBasicType() != EbtFloat)
    {
        // Only float ops have reproduced the bug.
        return true;
    }
    if (left->isScalar() && right->isVector())
    {
        ASSERT(!node->isAssignment());
        ASSERT(!right->isArray());
        OriginalNode originalNodeFate;
        TIntermTyped *leftVectorized = Vectorize(left, right->getType(), &originalNodeFate);
        queueReplacementWithParent(node, left, leftVectorized, originalNodeFate);
        mReplaced = true;
        // Don't replace more nodes in the same subtree on this traversal. However, nodes elsewhere
        // in the tree may still be replaced.
        return false;
    }
    else if (left->isVector() && right->isScalar())
    {
        OriginalNode originalNodeFate;
        TIntermTyped *rightVectorized = Vectorize(right, left->getType(), &originalNodeFate);
        queueReplacementWithParent(node, right, rightVectorized, originalNodeFate);
        mReplaced = true;
        // Don't replace more nodes in the same subtree on this traversal. However, nodes elsewhere
        // in the tree may still be replaced.
        return false;
    }
    return true;
}

void VectorizeVectorScalarArithmeticTraverser::replaceMathInsideConstructor(
    TIntermAggregate *node,
    TIntermBinary *argBinary)
{
    // Turn:
    //   a * b
    // into:
    //   gvec(a) * gvec(b)

    TIntermTyped *left  = argBinary->getLeft();
    TIntermTyped *right = argBinary->getRight();
    ASSERT(left->isScalar() && right->isScalar());

    TType leftVectorizedType = left->getType();
    leftVectorizedType.setPrimarySize(static_cast<unsigned char>(node->getType().getNominalSize()));
    TIntermTyped *leftVectorized = Vectorize(left, leftVectorizedType, nullptr);
    TType rightVectorizedType    = right->getType();
    rightVectorizedType.setPrimarySize(
        static_cast<unsigned char>(node->getType().getNominalSize()));
    TIntermTyped *rightVectorized = Vectorize(right, rightVectorizedType, nullptr);

    TIntermBinary *newArg = new TIntermBinary(argBinary->getOp(), leftVectorized, rightVectorized);
    queueReplacementWithParent(node, argBinary, newArg, OriginalNode::IS_DROPPED);
}

void VectorizeVectorScalarArithmeticTraverser::replaceAssignInsideConstructor(
    const TIntermAggregate *node,
    const TIntermBinary *argBinary)
{
    // Turn:
    //   gvec(a *= b);
    // into:
    //   // This is inserted into the parent block:
    //   gvec s0 = gvec(a);
    //
    //   // This goes where the gvec constructor used to be:
    //   ((s0 *= b, a = s0.x), s0);

    TIntermTyped *left  = argBinary->getLeft();
    TIntermTyped *right = argBinary->getRight();
    ASSERT(left->isScalar() && right->isScalar());
    ASSERT(!left->hasSideEffects());

    TType vecType = node->getType();
    vecType.setQualifier(EvqTemporary);

    // gvec s0 = gvec(a);
    // s0 is called "tempAssignmentTarget" below.
    TIntermTyped *tempAssignmentTargetInitializer = Vectorize(left->deepCopy(), vecType, nullptr);
    TIntermDeclaration *tempAssignmentTargetDeclaration = nullptr;
    TVariable *tempAssignmentTarget =
        DeclareTempVariable(mSymbolTable, tempAssignmentTargetInitializer, EvqTemporary,
                            &tempAssignmentTargetDeclaration);

    // s0 *= b
    TOperator compoundAssignmentOp = argBinary->getOp();
    if (compoundAssignmentOp == EOpMulAssign)
    {
        compoundAssignmentOp = EOpVectorTimesScalarAssign;
    }
    TIntermBinary *replacementCompoundAssignment = new TIntermBinary(
        compoundAssignmentOp, CreateTempSymbolNode(tempAssignmentTarget), right->deepCopy());

    // s0.x
    TVector<int> swizzleXOffset;
    swizzleXOffset.push_back(0);
    TIntermSwizzle *tempAssignmentTargetX =
        new TIntermSwizzle(CreateTempSymbolNode(tempAssignmentTarget), swizzleXOffset);
    // a = s0.x
    TIntermBinary *replacementAssignBackToTarget =
        new TIntermBinary(EOpAssign, left->deepCopy(), tempAssignmentTargetX);

    // s0 *= b, a = s0.x
    TIntermBinary *replacementSequenceLeft =
        new TIntermBinary(EOpComma, replacementCompoundAssignment, replacementAssignBackToTarget);
    // (s0 *= b, a = s0.x), s0
    // Note that the created comma node is not const qualified in any case, so we can always pass
    // shader version 300 here.
    TIntermBinary *replacementSequence = TIntermBinary::CreateComma(
        replacementSequenceLeft, CreateTempSymbolNode(tempAssignmentTarget), 300);

    insertStatementInParentBlock(tempAssignmentTargetDeclaration);
    queueReplacement(replacementSequence, OriginalNode::IS_DROPPED);
}

bool VectorizeVectorScalarArithmeticTraverser::visitAggregate(Visit /*visit*/,
                                                              TIntermAggregate *node)
{
    // Transform scalar binary expressions inside vector constructors.
    if (!node->isConstructor() || !node->isVector() || node->getSequence()->size() != 1)
    {
        return true;
    }
    TIntermTyped *argument = node->getSequence()->back()->getAsTyped();
    ASSERT(argument);
    if (!argument->isScalar() || argument->getBasicType() != EbtFloat)
    {
        return true;
    }
    TIntermBinary *argBinary = argument->getAsBinaryNode();
    if (!argBinary)
    {
        return true;
    }

    // Only specific ops are necessary to change.
    switch (argBinary->getOp())
    {
        case EOpMul:
        case EOpDiv:
        {
            replaceMathInsideConstructor(node, argBinary);
            mReplaced = true;
            // Don't replace more nodes in the same subtree on this traversal. However, nodes
            // elsewhere in the tree may still be replaced.
            return false;
        }
        case EOpMulAssign:
        case EOpDivAssign:
        {
            // The case where the left side has side effects is too complicated to deal with, so we
            // leave that be.
            if (!argBinary->getLeft()->hasSideEffects())
            {
                const TIntermBlock *parentBlock = getParentBlock();
                // We can't do more than one insertion to the same block on the same traversal.
                if (mModifiedBlocks.find(parentBlock) == mModifiedBlocks.end())
                {
                    replaceAssignInsideConstructor(node, argBinary);
                    mModifiedBlocks.insert(parentBlock);
                    mReplaced = true;
                    // Don't replace more nodes in the same subtree on this traversal.
                    // However, nodes elsewhere in the tree may still be replaced.
                    return false;
                }
            }
            break;
        }
        default:
            return true;
    }
    return true;
}

}  // anonymous namespace

bool VectorizeVectorScalarArithmetic(TCompiler *compiler,
                                     TIntermBlock *root,
                                     TSymbolTable *symbolTable)
{
    VectorizeVectorScalarArithmeticTraverser traverser(symbolTable);
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (!traverser.updateTree(compiler, root))
        {
            return false;
        }
    } while (traverser.didReplaceScalarsWithVectors());

    return true;
}

}  // namespace sh
