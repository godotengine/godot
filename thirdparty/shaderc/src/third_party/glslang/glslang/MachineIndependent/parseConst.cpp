//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//
// Traverse a tree of constants to create a single folded constant.
// It should only be used when the whole tree is known to be constant.
//

#include "ParseHelper.h"

namespace glslang {

class TConstTraverser : public TIntermTraverser {
public:
    TConstTraverser(const TConstUnionArray& cUnion, bool singleConstParam, TOperator constructType, const TType& t)
      : unionArray(cUnion), type(t),
        constructorType(constructType), singleConstantParam(singleConstParam), error(false), isMatrix(false),
        matrixCols(0), matrixRows(0) {  index = 0; tOp = EOpNull; }

    virtual void visitConstantUnion(TIntermConstantUnion* node);
    virtual bool visitAggregate(TVisit, TIntermAggregate* node);

    int index;
    TConstUnionArray unionArray;
    TOperator tOp;
    const TType& type;
    TOperator constructorType;
    bool singleConstantParam;
    bool error;
    int size; // size of the constructor ( 4 for vec4)
    bool isMatrix;
    int matrixCols;
    int matrixRows;

protected:
    TConstTraverser(TConstTraverser&);
    TConstTraverser& operator=(TConstTraverser&);
};

bool TConstTraverser::visitAggregate(TVisit /* visit */, TIntermAggregate* node)
{
    if (! node->isConstructor() && node->getOp() != EOpComma) {
        error = true;

        return false;
    }

    bool flag = node->getSequence().size() == 1 && node->getSequence()[0]->getAsTyped()->getAsConstantUnion();
    if (flag) {
        singleConstantParam = true;
        constructorType = node->getOp();
        size = node->getType().computeNumComponents();

        if (node->getType().isMatrix()) {
            isMatrix = true;
            matrixCols = node->getType().getMatrixCols();
            matrixRows = node->getType().getMatrixRows();
        }
    }

    for (TIntermSequence::iterator p = node->getSequence().begin();
                                   p != node->getSequence().end(); p++) {

        if (node->getOp() == EOpComma)
            index = 0;

        (*p)->traverse(this);
    }
    if (flag)
    {
        singleConstantParam = false;
        constructorType = EOpNull;
        size = 0;
        isMatrix = false;
        matrixCols = 0;
        matrixRows = 0;
    }

    return false;
}

void TConstTraverser::visitConstantUnion(TIntermConstantUnion* node)
{
    TConstUnionArray leftUnionArray(unionArray);
    int instanceSize = type.computeNumComponents();

    if (index >= instanceSize)
        return;

    if (! singleConstantParam) {
        int rightUnionSize = node->getType().computeNumComponents();

        const TConstUnionArray& rightUnionArray = node->getConstArray();
        for (int i = 0; i < rightUnionSize; i++) {
            if (index >= instanceSize)
                return;
            leftUnionArray[index] = rightUnionArray[i];

            index++;
        }
    } else {
        int endIndex = index + size;
        const TConstUnionArray& rightUnionArray = node->getConstArray();
        if (! isMatrix) {
            int count = 0;
            int nodeComps = node->getType().computeNumComponents();
            for (int i = index; i < endIndex; i++) {
                if (i >= instanceSize)
                    return;

                leftUnionArray[i] = rightUnionArray[count];

                (index)++;

                if (nodeComps > 1)
                    count++;
            }
        } else {
            // constructing a matrix, but from what?
            if (node->isMatrix()) {
                // Matrix from a matrix; this has the outer matrix, node is the argument matrix.
                // Traverse the outer, potentially bigger matrix, fill in missing pieces with the
                // identity matrix.
                for (int c = 0; c < matrixCols; ++c) {
                    for (int r = 0; r < matrixRows; ++r) {
                        int targetOffset = index + c * matrixRows + r;
                        if (r < node->getType().getMatrixRows() && c < node->getType().getMatrixCols()) {
                            int srcOffset = c * node->getType().getMatrixRows() + r;
                            leftUnionArray[targetOffset] = rightUnionArray[srcOffset];
                        } else if (r == c)
                            leftUnionArray[targetOffset].setDConst(1.0);
                        else
                            leftUnionArray[targetOffset].setDConst(0.0);
                    }
                }
            } else {
                // matrix from vector
                int count = 0;
                const int startIndex = index;
                int nodeComps = node->getType().computeNumComponents();
                for (int i = startIndex; i < endIndex; i++) {
                    if (i >= instanceSize)
                        return;
                    if (i == startIndex || (i - startIndex) % (matrixRows + 1) == 0 )
                        leftUnionArray[i] = rightUnionArray[count];
                    else
                        leftUnionArray[i].setDConst(0.0);

                    index++;

                    if (nodeComps > 1)
                        count++;
                }
            }
        }
    }
}

bool TIntermediate::parseConstTree(TIntermNode* root, TConstUnionArray unionArray, TOperator constructorType, const TType& t, bool singleConstantParam)
{
    if (root == 0)
        return false;

    TConstTraverser it(unionArray, singleConstantParam, constructorType, t);

    root->traverse(&it);
    if (it.error)
        return true;
    else
        return false;
}

} // end namespace glslang
