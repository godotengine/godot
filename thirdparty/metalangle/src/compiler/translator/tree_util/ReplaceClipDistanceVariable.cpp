//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ReplaceClipDistanceVariable.cpp: Find any references to gl_ClipDistance and
// replace it with ANGLEClipDistance.
//

#include "compiler/translator/tree_util/ReplaceClipDistanceVariable.h"

#include "common/bitset_utils.h"
#include "common/debug.h"
#include "common/utilities.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

namespace sh
{
namespace
{

using ClipDistanceIdxSet = angle::BitSet<32>;

// Traverse the tree and collect the redeclaration and all constant index references of
// gl_ClipDistance
class GLClipDistanceReferenceTraverser : public TIntermTraverser
{
  public:
    GLClipDistanceReferenceTraverser(const TIntermSymbol **redeclaredSymOut,
                                     bool *nonConstIdxUsedOut,
                                     unsigned int *maxConstIdxOut,
                                     ClipDistanceIdxSet *constIndicesOut)
        : TIntermTraverser(true, false, false),
          mRedeclaredSym(redeclaredSymOut),
          mUseNonConstClipDistanceIndex(nonConstIdxUsedOut),
          mMaxConstClipDistanceIndex(maxConstIdxOut),
          mConstClipDistanceIndices(constIndicesOut)
    {
        *mRedeclaredSym                = nullptr;
        *mUseNonConstClipDistanceIndex = false;
        *mMaxConstClipDistanceIndex    = 0;
        mConstClipDistanceIndices->reset();
    }

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        // If gl_ClipDistance is redeclared, we need to collect its information
        const TIntermSequence &sequence = *(node->getSequence());

        if (sequence.size() != 1)
        {
            return true;
        }

        TIntermTyped *variable = sequence.front()->getAsTyped();
        if (!variable->getAsSymbolNode() ||
            variable->getAsSymbolNode()->getName() != "gl_ClipDistance")
        {
            return true;
        }

        *mRedeclaredSym = variable->getAsSymbolNode();

        return true;
    }

    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        TOperator op = node->getOp();
        if (op != EOpIndexDirect && op != EOpIndexIndirect)
        {
            return true;
        }
        TIntermSymbol *left = node->getLeft()->getAsSymbolNode();
        if (!left)
        {
            return true;
        }
        if (left->getName() != "gl_ClipDistance")
        {
            return true;
        }
        const TConstantUnion *constIdx = node->getRight()->getConstantValue();
        if (!constIdx)
        {
            *mUseNonConstClipDistanceIndex = true;
        }
        else
        {
            unsigned int idx = 0;
            switch (constIdx->getType())
            {
                case EbtInt:
                    idx = constIdx->getIConst();
                    break;
                case EbtUInt:
                    idx = constIdx->getUConst();
                    break;
                case EbtFloat:
                    idx = static_cast<unsigned int>(constIdx->getFConst());
                    break;
                case EbtBool:
                    idx = constIdx->getBConst() ? 1 : 0;
                    break;
                default:
                    UNREACHABLE();
                    break;
            }
            ASSERT(idx < mConstClipDistanceIndices->size());
            mConstClipDistanceIndices->set(idx);

            *mMaxConstClipDistanceIndex = std::max(*mMaxConstClipDistanceIndex, idx);
        }

        return true;
    }

  private:
    const TIntermSymbol **mRedeclaredSym;
    // Flag indicating whether there is at least one reference of gl_ClipDistance with non-constant
    // index
    bool *mUseNonConstClipDistanceIndex;
    // Max constant index that is used to reference gl_ClipDistance
    unsigned int *mMaxConstClipDistanceIndex;
    // List of constant index reference of gl_ClipDistance
    ClipDistanceIdxSet *mConstClipDistanceIndices;
};

// Replace all symbolic occurrences of given variables except one symbol.
class ReplaceVariableExceptOneTraverser : public TIntermTraverser
{
  public:
    ReplaceVariableExceptOneTraverser(const TVariable *toBeReplaced,
                                      const TIntermTyped *replacement,
                                      const TIntermSymbol *exception)
        : TIntermTraverser(true, false, false),
          mToBeReplaced(toBeReplaced),
          mException(exception),
          mReplacement(replacement)
    {}

    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == mToBeReplaced && node != mException)
        {
            queueReplacement(mReplacement->deepCopy(), OriginalNode::IS_DROPPED);
        }
    }

  private:
    const TVariable *const mToBeReplaced;
    const TIntermSymbol *const mException;
    const TIntermTyped *const mReplacement;
};

}  // anonymous namespace

ANGLE_NO_DISCARD bool ReplaceClipDistanceAssignments(TCompiler *compiler,
                                                     TIntermBlock *root,
                                                     TSymbolTable *symbolTable,
                                                     const TIntermTyped *clipDistanceEnableFlags)
{
    // Collect all constant index references of gl_ClipDistance
    ClipDistanceIdxSet constIndices;
    bool useNonConstIndex                         = false;
    const TIntermSymbol *redeclaredGLClipDistance = nullptr;
    unsigned int maxConstIndex                    = 0;
    GLClipDistanceReferenceTraverser indexTraverser(&redeclaredGLClipDistance, &useNonConstIndex,
                                                    &maxConstIndex, &constIndices);
    root->traverse(&indexTraverser);
    if (!useNonConstIndex && constIndices.none())
    {
        // No references of gl_ClipDistance
        return true;
    }

    // Retrieve gl_ClipDistance variable reference
    // Search user redeclared gl_ClipDistance first
    const TVariable *glClipDistanceVar = nullptr;
    if (redeclaredGLClipDistance)
    {
        glClipDistanceVar = &redeclaredGLClipDistance->variable();
    }
    else
    {
        ImmutableString glClipDistanceName("gl_ClipDistance");
        // User defined not found, find in built-in table
        glClipDistanceVar =
            static_cast<const TVariable *>(symbolTable->findBuiltIn(glClipDistanceName, 0));
    }
    if (!glClipDistanceVar)
    {
        return false;
    }

    // Declare a global variable substituting gl_ClipDistance
    TType *clipDistanceType = new TType(EbtFloat, EbpMedium, EvqGlobal, 1);
    if (redeclaredGLClipDistance)
    {
        // If array is redeclared by user, use that redeclared size.
        clipDistanceType->makeArray(redeclaredGLClipDistance->getType().getOutermostArraySize());
    }
    else if (!useNonConstIndex)
    {
        ASSERT(maxConstIndex < glClipDistanceVar->getType().getOutermostArraySize());
        // Only use constant index, then use max array index used.
        clipDistanceType->makeArray(maxConstIndex + 1);
    }
    else
    {
        clipDistanceType->makeArray(glClipDistanceVar->getType().getOutermostArraySize());
    }

    clipDistanceType->realize();
    TVariable *clipDistanceVar = new TVariable(symbolTable, ImmutableString("ANGLEClipDistance"),
                                               clipDistanceType, SymbolType::AngleInternal);

    TIntermSymbol *clipDistanceDeclarator = new TIntermSymbol(clipDistanceVar);
    TIntermDeclaration *clipDistanceDecl  = new TIntermDeclaration;
    clipDistanceDecl->appendDeclarator(clipDistanceDeclarator);

    // Must declare ANGLEClipDistance before any function, since gl_ClipDistance might be accessed
    // within a function declared before main.
    root->insertStatement(0, clipDistanceDecl);

    // Replace gl_ClipDistance reference with ANGLEClipDistance, except the declaration
    ReplaceVariableExceptOneTraverser replaceTraverser(glClipDistanceVar,
                                                       new TIntermSymbol(clipDistanceVar),
                                                       /** exception */ redeclaredGLClipDistance);
    root->traverse(&replaceTraverser);
    if (!replaceTraverser.updateTree(compiler, root))
    {
        return false;
    }

    TIntermBlock *reassignBlock         = new TIntermBlock;
    TIntermSymbol *glClipDistanceSymbol = new TIntermSymbol(glClipDistanceVar);
    TIntermSymbol *clipDistanceSymbol   = new TIntermSymbol(clipDistanceVar);

    // Reassign ANGLEClipDistance to gl_ClipDistance but ignore those that are disabled

    auto assignFunc = [=](unsigned int index) {
        //  if (ANGLEUniforms.clipDistancesEnabled & (0x1 << index))
        //      gl_ClipDistance[index] = ANGLEClipDistance[index];
        //  else
        //      gl_ClipDistance[index] = 0;
        TIntermConstantUnion *bitMask = CreateUIntNode(0x1 << index);
        TIntermBinary *bitwiseAnd =
            new TIntermBinary(EOpBitwiseAnd, clipDistanceEnableFlags->deepCopy(), bitMask);
        TIntermBinary *nonZero = new TIntermBinary(EOpNotEqual, bitwiseAnd, CreateUIntNode(0));

        TIntermBinary *left  = new TIntermBinary(EOpIndexDirect, glClipDistanceSymbol->deepCopy(),
                                                CreateIndexNode(index));
        TIntermBinary *right = new TIntermBinary(EOpIndexDirect, clipDistanceSymbol->deepCopy(),
                                                 CreateIndexNode(index));
        TIntermBinary *assignment = new TIntermBinary(EOpAssign, left, right);
        TIntermBlock *trueBlock   = new TIntermBlock();
        trueBlock->appendStatement(assignment);

        TIntermBinary *zeroAssignment =
            new TIntermBinary(EOpAssign, left->deepCopy(), CreateFloatNode(0));
        TIntermBlock *falseBlock = new TIntermBlock();
        falseBlock->appendStatement(zeroAssignment);

        return new TIntermIfElse(nonZero, trueBlock, falseBlock);
    };

    if (useNonConstIndex)
    {
        // If there is at least one non constant index reference,
        // Then we need to loop through the whole declared size of gl_ClipDistance.
        // Since we don't know exactly the index at compile time.
        // As mentioned in
        // https://www.khronos.org/registry/OpenGL/extensions/APPLE/APPLE_clip_distance.txt
        // Non constant index can only be used if gl_ClipDistance is redeclared with an explicit
        // size.
        for (unsigned int i = 0; i < clipDistanceType->getOutermostArraySize(); ++i)
        {
            reassignBlock->appendStatement(assignFunc(i));
        }
    }
    else
    {
        // Assign ANGLEClipDistance[i]'s value to gl_ClipDistance[i] if i is in the constant
        // indices list.
        // Those elements whose index is not in the constant index list will be zeroise.
        for (unsigned int i = 0; i < clipDistanceType->getOutermostArraySize(); ++i)
        {
            if (constIndices.test(i))
            {
                reassignBlock->appendStatement(assignFunc(i));
            }
            else
            {
                // gl_ClipDistance[i] = 0;
                TIntermBinary *left = new TIntermBinary(
                    EOpIndexDirect, glClipDistanceSymbol->deepCopy(), CreateIndexNode(i));
                TIntermBinary *zeroAssignment =
                    new TIntermBinary(EOpAssign, left, CreateFloatNode(0));
                reassignBlock->appendStatement(zeroAssignment);
            }
        }
    }

    return RunAtTheEndOfShader(compiler, root, reassignBlock, symbolTable);
}

}  // namespace sh
