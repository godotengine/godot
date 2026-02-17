//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2018-2020 Google, Inc.
//
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

#include "localintermediate.h"
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <climits>

namespace {

using namespace glslang;

const double pi = 3.1415926535897932384626433832795;

} // end anonymous namespace


namespace glslang {

//
// The fold functions see if an operation on a constant can be done in place,
// without generating run-time code.
//
// Returns the node to keep using, which may or may not be the node passed in.
//
// Note: As of version 1.2, all constant operations must be folded.  It is
// not opportunistic, but rather a semantic requirement.
//

//
// Do folding between a pair of nodes.
// 'this' is the left-hand operand and 'rightConstantNode' is the right-hand operand.
//
// Returns a new node representing the result.
//
TIntermTyped* TIntermConstantUnion::fold(TOperator op, const TIntermTyped* rightConstantNode) const
{
    // For most cases, the return type matches the argument type, so set that
    // up and just code to exceptions below.
    TType returnType;
    returnType.shallowCopy(getType());

    //
    // A pair of nodes is to be folded together
    //

    const TIntermConstantUnion *rightNode = rightConstantNode->getAsConstantUnion();
    TConstUnionArray leftUnionArray = getConstArray();
    TConstUnionArray rightUnionArray = rightNode->getConstArray();

    // Figure out the size of the result
    int newComps;
    int constComps;
    switch(op) {
    case EOpMatrixTimesMatrix:
        newComps = rightNode->getMatrixCols() * getMatrixRows();
        break;
    case EOpMatrixTimesVector:
        newComps = getMatrixRows();
        break;
    case EOpVectorTimesMatrix:
        newComps = rightNode->getMatrixCols();
        break;
    default:
        newComps = getType().computeNumComponents();
        constComps = rightConstantNode->getType().computeNumComponents();
        if (constComps == 1 && newComps > 1) {
            // for a case like vec4 f = vec4(2,3,4,5) + 1.2;
            TConstUnionArray smearedArray(newComps, rightNode->getConstArray()[0]);
            rightUnionArray = smearedArray;
        } else if (constComps > 1 && newComps == 1) {
            // for a case like vec4 f = 1.2 + vec4(2,3,4,5);
            newComps = constComps;
            rightUnionArray = rightNode->getConstArray();
            TConstUnionArray smearedArray(newComps, getConstArray()[0]);
            leftUnionArray = smearedArray;
            returnType.shallowCopy(rightNode->getType());
        }
        break;
    }

    TConstUnionArray newConstArray(newComps);
    TType constBool(EbtBool, EvqConst);

    switch(op) {
    case EOpAdd:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] + rightUnionArray[i];
        break;
    case EOpSub:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] - rightUnionArray[i];
        break;

    case EOpMul:
    case EOpVectorTimesScalar:
    case EOpMatrixTimesScalar:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] * rightUnionArray[i];
        break;
    case EOpMatrixTimesMatrix:
        for (int row = 0; row < getMatrixRows(); row++) {
            for (int column = 0; column < rightNode->getMatrixCols(); column++) {
                double sum = 0.0f;
                for (int i = 0; i < rightNode->getMatrixRows(); i++)
                    sum += leftUnionArray[i * getMatrixRows() + row].getDConst() * rightUnionArray[column * rightNode->getMatrixRows() + i].getDConst();
                newConstArray[column * getMatrixRows() + row].setDConst(sum);
            }
        }
        returnType.shallowCopy(TType(getType().getBasicType(), EvqConst, 0, rightNode->getMatrixCols(), getMatrixRows()));
        break;
    case EOpDiv:
        for (int i = 0; i < newComps; i++) {
            switch (getType().getBasicType()) {
            case EbtDouble:
            case EbtFloat:
            case EbtFloat16:
            case EbtBFloat16:
            case EbtFloatE5M2:
            case EbtFloatE4M3:
                if (rightUnionArray[i].getDConst() != 0.0)
                    newConstArray[i].setDConst(leftUnionArray[i].getDConst() / rightUnionArray[i].getDConst());
                else if (leftUnionArray[i].getDConst() > 0.0)
                    newConstArray[i].setDConst((double)INFINITY);
                else if (leftUnionArray[i].getDConst() < 0.0)
                    newConstArray[i].setDConst(-(double)INFINITY);
                else
                    newConstArray[i].setDConst((double)NAN);
                break;

            case EbtInt:
                if (rightUnionArray[i] == 0)
                    newConstArray[i].setIConst(0x7FFFFFFF);
                else if (rightUnionArray[i].getIConst() == -1 && leftUnionArray[i].getIConst() == (int)-0x80000000ll)
                    newConstArray[i].setIConst((int)-0x80000000ll);
                else
                    newConstArray[i].setIConst(leftUnionArray[i].getIConst() / rightUnionArray[i].getIConst());
                break;

            case EbtUint:
                if (rightUnionArray[i] == 0u)
                    newConstArray[i].setUConst(0xFFFFFFFFu);
                else
                    newConstArray[i].setUConst(leftUnionArray[i].getUConst() / rightUnionArray[i].getUConst());
                break;

            case EbtInt8:
                if (rightUnionArray[i] == (signed char)0)
                    newConstArray[i].setI8Const((signed char)0x7F);
                else if (rightUnionArray[i].getI8Const() == (signed char)-1 && leftUnionArray[i].getI8Const() == (signed char)-0x80)
                    newConstArray[i].setI8Const((signed char)-0x80);
                else
                    newConstArray[i].setI8Const(leftUnionArray[i].getI8Const() / rightUnionArray[i].getI8Const());
                break;

            case EbtUint8:
                if (rightUnionArray[i] == (unsigned char)0u)
                    newConstArray[i].setU8Const((unsigned char)0xFFu);
                else
                    newConstArray[i].setU8Const(leftUnionArray[i].getU8Const() / rightUnionArray[i].getU8Const());
                break;

           case EbtInt16:
                if (rightUnionArray[i] == (signed short)0)
                    newConstArray[i].setI16Const((signed short)0x7FFF);
                else if (rightUnionArray[i].getI16Const() == (signed short)-1 && leftUnionArray[i].getI16Const() == (signed short)-0x8000)
                    newConstArray[i].setI16Const((signed short)-0x8000);
                else
                    newConstArray[i].setI16Const(leftUnionArray[i].getI16Const() / rightUnionArray[i].getI16Const());
                break;

            case EbtUint16:
                if (rightUnionArray[i] == (unsigned short)0u)
                    newConstArray[i].setU16Const((unsigned short)0xFFFFu);
                else
                    newConstArray[i].setU16Const(leftUnionArray[i].getU16Const() / rightUnionArray[i].getU16Const());
                break;

            case EbtInt64:
                if (rightUnionArray[i] == 0ll)
                    newConstArray[i].setI64Const(LLONG_MAX);
                else if (rightUnionArray[i].getI64Const() == -1 && leftUnionArray[i].getI64Const() == LLONG_MIN)
                    newConstArray[i].setI64Const(LLONG_MIN);
                else
                    newConstArray[i].setI64Const(leftUnionArray[i].getI64Const() / rightUnionArray[i].getI64Const());
                break;

            case EbtUint64:
                if (rightUnionArray[i] == 0ull)
                    newConstArray[i].setU64Const(0xFFFFFFFFFFFFFFFFull);
                else
                    newConstArray[i].setU64Const(leftUnionArray[i].getU64Const() / rightUnionArray[i].getU64Const());
                break;
            default:
                return nullptr;
            }
        }
        break;

    case EOpMatrixTimesVector:
        for (int i = 0; i < getMatrixRows(); i++) {
            double sum = 0.0f;
            for (int j = 0; j < rightNode->getVectorSize(); j++) {
                sum += leftUnionArray[j*getMatrixRows() + i].getDConst() * rightUnionArray[j].getDConst();
            }
            newConstArray[i].setDConst(sum);
        }

        returnType.shallowCopy(TType(getBasicType(), EvqConst, getMatrixRows()));
        break;

    case EOpVectorTimesMatrix:
        for (int i = 0; i < rightNode->getMatrixCols(); i++) {
            double sum = 0.0f;
            for (int j = 0; j < getVectorSize(); j++)
                sum += leftUnionArray[j].getDConst() * rightUnionArray[i*rightNode->getMatrixRows() + j].getDConst();
            newConstArray[i].setDConst(sum);
        }

        returnType.shallowCopy(TType(getBasicType(), EvqConst, rightNode->getMatrixCols()));
        break;

    case EOpMod:
        for (int i = 0; i < newComps; i++) {
            if (rightUnionArray[i] == 0)
                newConstArray[i] = leftUnionArray[i];
            else {
                switch (getType().getBasicType()) {
                case EbtInt:
                    if (rightUnionArray[i].getIConst() == -1 && leftUnionArray[i].getIConst() == INT_MIN) {
                        newConstArray[i].setIConst(0);
                        break;
                    } else goto modulo_default;
                case EbtInt64:
                    if (rightUnionArray[i].getI64Const() == -1 && leftUnionArray[i].getI64Const() == LLONG_MIN) {
                        newConstArray[i].setI64Const(0);
                        break;
                    } else goto modulo_default;
                case EbtInt16:
                    if (rightUnionArray[i].getIConst() == -1 && leftUnionArray[i].getIConst() == SHRT_MIN) {
                        newConstArray[i].setIConst(0);
                        break;
                    } else goto modulo_default;
                default:
                modulo_default:
                    newConstArray[i] = leftUnionArray[i] % rightUnionArray[i];
                }
            }
        }
        break;

    case EOpRightShift:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] >> rightUnionArray[i];
        break;

    case EOpLeftShift:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] << rightUnionArray[i];
        break;

    case EOpAnd:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] & rightUnionArray[i];
        break;
    case EOpInclusiveOr:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] | rightUnionArray[i];
        break;
    case EOpExclusiveOr:
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] ^ rightUnionArray[i];
        break;

    case EOpLogicalAnd: // this code is written for possible future use, will not get executed currently
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] && rightUnionArray[i];
        break;

    case EOpLogicalOr: // this code is written for possible future use, will not get executed currently
        for (int i = 0; i < newComps; i++)
            newConstArray[i] = leftUnionArray[i] || rightUnionArray[i];
        break;

    case EOpLogicalXor:
        for (int i = 0; i < newComps; i++) {
            switch (getType().getBasicType()) {
            case EbtBool: newConstArray[i].setBConst((leftUnionArray[i] == rightUnionArray[i]) ? false : true); break;
            default: assert(false && "Default missing");
            }
        }
        break;

    case EOpLessThan:
        newConstArray[0].setBConst(leftUnionArray[0] < rightUnionArray[0]);
        returnType.shallowCopy(constBool);
        break;
    case EOpGreaterThan:
        newConstArray[0].setBConst(leftUnionArray[0] > rightUnionArray[0]);
        returnType.shallowCopy(constBool);
        break;
    case EOpLessThanEqual:
        newConstArray[0].setBConst(! (leftUnionArray[0] > rightUnionArray[0]));
        returnType.shallowCopy(constBool);
        break;
    case EOpGreaterThanEqual:
        newConstArray[0].setBConst(! (leftUnionArray[0] < rightUnionArray[0]));
        returnType.shallowCopy(constBool);
        break;
    case EOpEqual:
        newConstArray[0].setBConst(rightNode->getConstArray() == leftUnionArray);
        returnType.shallowCopy(constBool);
        break;
    case EOpNotEqual:
        newConstArray[0].setBConst(rightNode->getConstArray() != leftUnionArray);
        returnType.shallowCopy(constBool);
        break;

    default:
        return nullptr;
    }

    TIntermConstantUnion *newNode = new TIntermConstantUnion(newConstArray, returnType);
    newNode->setLoc(getLoc());

    return newNode;
}

//
// Do single unary node folding
//
// Returns a new node representing the result.
//
TIntermTyped* TIntermConstantUnion::fold(TOperator op, const TType& returnType) const
{
    // First, size the result, which is mostly the same as the argument's size,
    // but not always, and classify what is componentwise.
    // Also, eliminate cases that can't be compile-time constant.
    int resultSize;
    bool componentWise = true;

    int objectSize = getType().computeNumComponents();
    switch (op) {
    case EOpDeterminant:
    case EOpAny:
    case EOpAll:
    case EOpLength:
        componentWise = false;
        resultSize = 1;
        break;

    case EOpEmitStreamVertex:
    case EOpEndStreamPrimitive:
        // These don't fold
        return nullptr;

    case EOpPackSnorm2x16:
    case EOpPackUnorm2x16:
    case EOpPackHalf2x16:
        componentWise = false;
        resultSize = 1;
        break;

    case EOpUnpackSnorm2x16:
    case EOpUnpackUnorm2x16:
    case EOpUnpackHalf2x16:
        componentWise = false;
        resultSize = 2;
        break;

    case EOpPack16:
    case EOpPack32:
    case EOpPack64:
    case EOpUnpack32:
    case EOpUnpack16:
    case EOpUnpack8:
    case EOpNormalize:
        componentWise = false;
        resultSize = objectSize;
        break;

    default:
        resultSize = objectSize;
        break;
    }

    // Set up for processing
    TConstUnionArray newConstArray(resultSize);
    const TConstUnionArray& unionArray = getConstArray();

    // Process non-component-wise operations
    switch (op) {
    case EOpLength:
    case EOpNormalize:
    {
        double sum = 0;
        for (int i = 0; i < objectSize; i++)
            sum += unionArray[i].getDConst() * unionArray[i].getDConst();
        double length = sqrt(sum);
        if (op == EOpLength)
            newConstArray[0].setDConst(length);
        else {
            for (int i = 0; i < objectSize; i++)
                newConstArray[i].setDConst(unionArray[i].getDConst() / length);
        }
        break;
    }

    case EOpAny:
    {
        bool result = false;
        for (int i = 0; i < objectSize; i++) {
            if (unionArray[i].getBConst())
                result = true;
        }
        newConstArray[0].setBConst(result);
        break;
    }
    case EOpAll:
    {
        bool result = true;
        for (int i = 0; i < objectSize; i++) {
            if (! unionArray[i].getBConst())
                result = false;
        }
        newConstArray[0].setBConst(result);
        break;
    }

    case EOpPackSnorm2x16:
    case EOpPackUnorm2x16:
    case EOpPackHalf2x16:
    case EOpPack16:
    case EOpPack32:
    case EOpPack64:
    case EOpUnpack32:
    case EOpUnpack16:
    case EOpUnpack8:

    case EOpUnpackSnorm2x16:
    case EOpUnpackUnorm2x16:
    case EOpUnpackHalf2x16:

    case EOpDeterminant:
    case EOpMatrixInverse:
    case EOpTranspose:
        return nullptr;

    default:
        assert(componentWise);
        break;
    }

    // Turn off the componentwise loop
    if (! componentWise)
        objectSize = 0;

    // Process component-wise operations
    for (int i = 0; i < objectSize; i++) {
        // First read the value and convert to i64/u64/f64/bool, then convert
        // to the destination type (still 64b), then convert down to the
        // destination size.
        if (IsOpNumericConv(op)) {
            enum ConvType { CONV_FLOAT, CONV_INT, CONV_UINT, CONV_BOOL };
            ConvType srcType = CONV_UINT, dstType = CONV_UINT;
            double valf = 0.0;
            uint64_t valu = 0;
            int64_t vali = 0;
            bool valb = false;
            switch (getType().getBasicType()) {
            case EbtDouble:
            case EbtFloat16:
            case EbtBFloat16:
            case EbtFloatE5M2:
            case EbtFloatE4M3:
            case EbtFloat:
                valf = unionArray[i].getDConst();
                srcType = CONV_FLOAT;
                break;
            case EbtInt8:
                vali = unionArray[i].getI8Const();
                srcType = CONV_INT;
                break;
            case EbtInt16:
                vali = unionArray[i].getI16Const();
                srcType = CONV_INT;
                break;
            case EbtInt:
                vali = unionArray[i].getIConst();
                srcType = CONV_INT;
                break;
            case EbtInt64:
                vali = unionArray[i].getI64Const();
                srcType = CONV_INT;
                break;
            case EbtUint8:
                valu = unionArray[i].getU8Const();
                srcType = CONV_UINT;
                break;
            case EbtUint16:
                valu = unionArray[i].getU16Const();
                srcType = CONV_UINT;
                break;
            case EbtUint:
                valu = unionArray[i].getUConst();
                srcType = CONV_UINT;
                break;
            case EbtUint64:
                valu = unionArray[i].getU64Const();
                srcType = CONV_UINT;
                break;
            case EbtBool:
                valb = unionArray[i].getBConst();
                srcType = CONV_BOOL;
                break;
            default:
                assert(0);
                break;
            }

            switch (returnType.getBasicType()) {
            case EbtDouble:
            case EbtFloat16:
            case EbtBFloat16:
            case EbtFloatE5M2:
            case EbtFloatE4M3:
            case EbtFloat:
                dstType = CONV_FLOAT;
                break;
            case EbtInt8:
            case EbtInt16:
            case EbtInt:
            case EbtInt64:
                dstType = CONV_INT;
                break;
            case EbtUint8:
            case EbtUint16:
            case EbtUint:
            case EbtUint64:
                dstType = CONV_UINT;
                break;
            case EbtBool:
                dstType = CONV_BOOL;
                break;
            default:
                assert(0);
                break;
            }
            if (dstType == CONV_BOOL) {
                switch (srcType) {
                case CONV_FLOAT:
                    valb = (valf != 0.0); break;
                case CONV_INT:
                    valb = (vali != 0.0); break;
                case CONV_UINT:
                    valb = (valu != 0.0); break;
                default:
                    break;
                }
            } else if (dstType == CONV_FLOAT) {
                switch (srcType) {
                case CONV_BOOL:
                    valf = (double)valb; break;
                case CONV_INT:
                    valf = (double)vali; break;
                case CONV_UINT:
                    valf = (double)valu; break;
                default:
                    break;
                }
            } else if (dstType == CONV_INT) {
                switch (srcType) {
                case CONV_BOOL:
                    vali = (int64_t)valb; break;
                case CONV_FLOAT:
                    vali = (int64_t)valf; break;
                case CONV_UINT:
                    vali = (int64_t)valu; break;
                default:
                    break;
                }
            } else if (dstType == CONV_UINT) {
                switch (srcType) {
                case CONV_BOOL:
                    valu = (uint64_t)valb; break;
                case CONV_FLOAT:
                    valu = (uint64_t)valf; break;
                case CONV_INT:
                    valu = (uint64_t)vali; break;
                default:
                    break;
                }
            }
            switch (returnType.getBasicType()) {
            case EbtDouble:
            case EbtFloat16:
            case EbtBFloat16:
            case EbtFloatE5M2:
            case EbtFloatE4M3:
            case EbtFloat:
                newConstArray[i].setDConst(valf); break;
            case EbtInt8:
                newConstArray[i].setI8Const(static_cast<int8_t>(vali)); break;
            case EbtInt16:
                newConstArray[i].setI16Const(static_cast<int16_t>(vali)); break;
            case EbtInt:
                newConstArray[i].setIConst(static_cast<int32_t>(vali)); break;
            case EbtInt64:
                newConstArray[i].setI64Const(vali); break;
            case EbtUint8:
                newConstArray[i].setU8Const(static_cast<uint8_t>(valu)); break;
            case EbtUint16:
                newConstArray[i].setU16Const(static_cast<uint16_t>(valu)); break;
            case EbtUint:
                newConstArray[i].setUConst(static_cast<uint32_t>(valu)); break;
            case EbtUint64:
                newConstArray[i].setU64Const(valu); break;
            case EbtBool:
                newConstArray[i].setBConst(valb); break;
            default:
                assert(0);
                break;
            }
            continue;
        }
        switch (op) {
        case EOpNegative:
            switch (getType().getBasicType()) {
            case EbtDouble:
            case EbtFloat16:
            case EbtBFloat16:
            case EbtFloatE5M2:
            case EbtFloatE4M3:
            case EbtFloat: newConstArray[i].setDConst(-unionArray[i].getDConst()); break;
            // Note: avoid UBSAN error regarding negating 0x80000000
            case EbtInt:   newConstArray[i].setIConst(
                                static_cast<unsigned int>(unionArray[i].getIConst()) == 0x80000000
                                    ? -0x7FFFFFFF - 1
                                    : -unionArray[i].getIConst());
                           break;
            case EbtUint:  newConstArray[i].setUConst(static_cast<unsigned int>(-static_cast<int>(unionArray[i].getUConst())));  break;
            case EbtInt8:  newConstArray[i].setI8Const(-unionArray[i].getI8Const()); break;
            case EbtUint8: newConstArray[i].setU8Const(static_cast<unsigned int>(-static_cast<signed int>(unionArray[i].getU8Const())));  break;
            case EbtInt16: newConstArray[i].setI16Const(-unionArray[i].getI16Const()); break;
            case EbtUint16:newConstArray[i].setU16Const(static_cast<unsigned int>(-static_cast<signed int>(unionArray[i].getU16Const())));  break;
            case EbtInt64: {
                int64_t i64val = unionArray[i].getI64Const();
                newConstArray[i].setI64Const(i64val == INT64_MIN ? INT64_MIN : -i64val);
                break;
            }
            case EbtUint64: newConstArray[i].setU64Const(static_cast<unsigned long long>(-static_cast<long long>(unionArray[i].getU64Const())));  break;
            default:
                return nullptr;
            }
            break;
        case EOpLogicalNot:
        case EOpVectorLogicalNot:
            switch (getType().getBasicType()) {
            case EbtBool:  newConstArray[i].setBConst(!unionArray[i].getBConst()); break;
            default:
                return nullptr;
            }
            break;
        case EOpBitwiseNot:
            newConstArray[i] = ~unionArray[i];
            break;
        case EOpRadians:
            newConstArray[i].setDConst(unionArray[i].getDConst() * pi / 180.0);
            break;
        case EOpDegrees:
            newConstArray[i].setDConst(unionArray[i].getDConst() * 180.0 / pi);
            break;
        case EOpSin:
            newConstArray[i].setDConst(sin(unionArray[i].getDConst()));
            break;
        case EOpCos:
            newConstArray[i].setDConst(cos(unionArray[i].getDConst()));
            break;
        case EOpTan:
            newConstArray[i].setDConst(tan(unionArray[i].getDConst()));
            break;
        case EOpAsin:
            newConstArray[i].setDConst(asin(unionArray[i].getDConst()));
            break;
        case EOpAcos:
            newConstArray[i].setDConst(acos(unionArray[i].getDConst()));
            break;
        case EOpAtan:
            newConstArray[i].setDConst(atan(unionArray[i].getDConst()));
            break;

        case EOpDPdx:
        case EOpDPdy:
        case EOpFwidth:
        case EOpDPdxFine:
        case EOpDPdyFine:
        case EOpFwidthFine:
        case EOpDPdxCoarse:
        case EOpDPdyCoarse:
        case EOpFwidthCoarse:
            // The derivatives are all mandated to create a constant 0.
            newConstArray[i].setDConst(0.0);
            break;

        case EOpExp:
            newConstArray[i].setDConst(exp(unionArray[i].getDConst()));
            break;
        case EOpLog:
            newConstArray[i].setDConst(log(unionArray[i].getDConst()));
            break;
        case EOpExp2:
            newConstArray[i].setDConst(exp2(unionArray[i].getDConst()));
            break;
        case EOpLog2:
            newConstArray[i].setDConst(log2(unionArray[i].getDConst()));
            break;
        case EOpSqrt:
            newConstArray[i].setDConst(sqrt(unionArray[i].getDConst()));
            break;
        case EOpInverseSqrt:
            newConstArray[i].setDConst(1.0 / sqrt(unionArray[i].getDConst()));
            break;

        case EOpAbs:
            if (unionArray[i].getType() == EbtDouble)
                newConstArray[i].setDConst(fabs(unionArray[i].getDConst()));
            else if (unionArray[i].getType() == EbtInt)
                newConstArray[i].setIConst(abs(unionArray[i].getIConst()));
            else
                newConstArray[i] = unionArray[i];
            break;
        case EOpSign:
            #define SIGN(X) (X == 0 ? 0 : (X < 0 ? -1 : 1))
            if (unionArray[i].getType() == EbtDouble)
                newConstArray[i].setDConst(SIGN(unionArray[i].getDConst()));
            else
                newConstArray[i].setIConst(SIGN(unionArray[i].getIConst()));
            break;
        case EOpFloor:
            newConstArray[i].setDConst(floor(unionArray[i].getDConst()));
            break;
        case EOpTrunc:
            if (unionArray[i].getDConst() > 0)
                newConstArray[i].setDConst(floor(unionArray[i].getDConst()));
            else
                newConstArray[i].setDConst(ceil(unionArray[i].getDConst()));
            break;
        case EOpRound:
            newConstArray[i].setDConst(floor(0.5 + unionArray[i].getDConst()));
            break;
        case EOpRoundEven:
        {
            double flr = floor(unionArray[i].getDConst());
            bool even = flr / 2.0 == floor(flr / 2.0);
            double rounded = even ? ceil(unionArray[i].getDConst() - 0.5) : floor(unionArray[i].getDConst() + 0.5);
            newConstArray[i].setDConst(rounded);
            break;
        }
        case EOpCeil:
            newConstArray[i].setDConst(ceil(unionArray[i].getDConst()));
            break;
        case EOpFract:
        {
            double x = unionArray[i].getDConst();
            newConstArray[i].setDConst(x - floor(x));
            break;
        }

        case EOpIsNan:
        {
            newConstArray[i].setBConst(std::isnan(unionArray[i].getDConst()));
            break;
        }
        case EOpIsInf:
        {
            newConstArray[i].setBConst(std::isinf(unionArray[i].getDConst()));
            break;
        }

        case EOpConvPtrToUint64:
        case EOpConvUint64ToPtr:
        case EOpConstructReference:
            newConstArray[i].setU64Const(unionArray[i].getU64Const()); break;

        // TODO: 3.0 Functionality: unary constant folding: the rest of the ops have to be fleshed out

        case EOpSinh:
        case EOpCosh:
        case EOpTanh:
        case EOpAsinh:
        case EOpAcosh:
        case EOpAtanh:

        case EOpFloatBitsToInt:
        case EOpFloatBitsToUint:
        case EOpIntBitsToFloat:
        case EOpUintBitsToFloat:
        case EOpDoubleBitsToInt64:
        case EOpDoubleBitsToUint64:
        case EOpInt64BitsToDouble:
        case EOpUint64BitsToDouble:
        case EOpFloat16BitsToInt16:
        case EOpFloat16BitsToUint16:
        case EOpInt16BitsToFloat16:
        case EOpUint16BitsToFloat16:
        default:
            return nullptr;
        }
    }

    TIntermConstantUnion *newNode = new TIntermConstantUnion(newConstArray, returnType);
    newNode->getWritableType().getQualifier().storage = EvqConst;
    newNode->setLoc(getLoc());

    return newNode;
}

//
// Do constant folding for an aggregate node that has all its children
// as constants and an operator that requires constant folding.
//
TIntermTyped* TIntermediate::fold(TIntermAggregate* aggrNode)
{
    if (aggrNode == nullptr)
        return aggrNode;

    if (! areAllChildConst(aggrNode))
        return aggrNode;

    if (aggrNode->isConstructor())
        return foldConstructor(aggrNode);

    TIntermSequence& children = aggrNode->getSequence();

    // First, see if this is an operation to constant fold, kick out if not,
    // see what size the result is if so.

    bool componentwise = false;  // will also say componentwise if a scalar argument gets repeated to make per-component results
    int objectSize;
    switch (aggrNode->getOp()) {
    case EOpAtan:
    case EOpPow:
    case EOpMin:
    case EOpMax:
    case EOpMix:
    case EOpMod:
    case EOpClamp:
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
    case EOpVectorEqual:
    case EOpVectorNotEqual:
        componentwise = true;
        objectSize = children[0]->getAsConstantUnion()->getType().computeNumComponents();
        break;
    case EOpCross:
    case EOpReflect:
    case EOpRefract:
    case EOpFaceForward:
        objectSize = children[0]->getAsConstantUnion()->getType().computeNumComponents();
        break;
    case EOpDistance:
    case EOpDot:
        objectSize = 1;
        break;
    case EOpOuterProduct:
        objectSize = children[0]->getAsTyped()->getType().getVectorSize() *
                     children[1]->getAsTyped()->getType().getVectorSize();
        break;
    case EOpStep:
        componentwise = true;
        objectSize = std::max(children[0]->getAsTyped()->getType().getVectorSize(),
                              children[1]->getAsTyped()->getType().getVectorSize());
        break;
    case EOpSmoothStep:
        componentwise = true;
        objectSize = std::max(children[0]->getAsTyped()->getType().getVectorSize(),
                              children[2]->getAsTyped()->getType().getVectorSize());
        break;
    case EOpMul:
        {
        TIntermConstantUnion* left = children[0]->getAsConstantUnion();
        TIntermConstantUnion* right = children[1]->getAsConstantUnion();
        return left->fold(EOpMul, right);
        }
    default:
        return aggrNode;
    }
    TConstUnionArray newConstArray(objectSize);

    TVector<TConstUnionArray> childConstUnions;
    for (unsigned int arg = 0; arg < children.size(); ++arg)
        childConstUnions.push_back(children[arg]->getAsConstantUnion()->getConstArray());

    if (componentwise) {
        for (int comp = 0; comp < objectSize; comp++) {

            // some arguments are scalars instead of matching vectors; simulate a smear
            int arg0comp = std::min(comp, children[0]->getAsTyped()->getType().getVectorSize() - 1);
            int arg1comp = 0;
            if (children.size() > 1)
                arg1comp = std::min(comp, children[1]->getAsTyped()->getType().getVectorSize() - 1);
            int arg2comp = 0;
            if (children.size() > 2)
                arg2comp = std::min(comp, children[2]->getAsTyped()->getType().getVectorSize() - 1);

            switch (aggrNode->getOp()) {
            case EOpAtan:
                newConstArray[comp].setDConst(atan2(childConstUnions[0][arg0comp].getDConst(), childConstUnions[1][arg1comp].getDConst()));
                break;
            case EOpPow:
                newConstArray[comp].setDConst(pow(childConstUnions[0][arg0comp].getDConst(), childConstUnions[1][arg1comp].getDConst()));
                break;
            case EOpMod:
            {
                double arg0 = childConstUnions[0][arg0comp].getDConst();
                double arg1 = childConstUnions[1][arg1comp].getDConst();
                double result = arg0 - arg1 * floor(arg0 / arg1);
                newConstArray[comp].setDConst(result);
                break;
            }
            case EOpMin:
                switch(children[0]->getAsTyped()->getBasicType()) {
                case EbtFloat16:
                case EbtBFloat16:
                case EbtFloatE5M2:
                case EbtFloatE4M3:
                case EbtFloat:
                case EbtDouble:
                    newConstArray[comp].setDConst(std::min(childConstUnions[0][arg0comp].getDConst(), childConstUnions[1][arg1comp].getDConst()));
                    break;
                case EbtInt:
                    newConstArray[comp].setIConst(std::min(childConstUnions[0][arg0comp].getIConst(), childConstUnions[1][arg1comp].getIConst()));
                    break;
                case EbtUint:
                    newConstArray[comp].setUConst(std::min(childConstUnions[0][arg0comp].getUConst(), childConstUnions[1][arg1comp].getUConst()));
                    break;
                case EbtInt8:
                    newConstArray[comp].setI8Const(std::min(childConstUnions[0][arg0comp].getI8Const(), childConstUnions[1][arg1comp].getI8Const()));
                    break;
                case EbtUint8:
                    newConstArray[comp].setU8Const(std::min(childConstUnions[0][arg0comp].getU8Const(), childConstUnions[1][arg1comp].getU8Const()));
                    break;
                case EbtInt16:
                    newConstArray[comp].setI16Const(std::min(childConstUnions[0][arg0comp].getI16Const(), childConstUnions[1][arg1comp].getI16Const()));
                    break;
                case EbtUint16:
                    newConstArray[comp].setU16Const(std::min(childConstUnions[0][arg0comp].getU16Const(), childConstUnions[1][arg1comp].getU16Const()));
                    break;
                case EbtInt64:
                    newConstArray[comp].setI64Const(std::min(childConstUnions[0][arg0comp].getI64Const(), childConstUnions[1][arg1comp].getI64Const()));
                    break;
                case EbtUint64:
                    newConstArray[comp].setU64Const(std::min(childConstUnions[0][arg0comp].getU64Const(), childConstUnions[1][arg1comp].getU64Const()));
                    break;
                default: assert(false && "Default missing");
                }
                break;
            case EOpMax:
                switch(children[0]->getAsTyped()->getBasicType()) {
                case EbtFloat16:
                case EbtBFloat16:
                case EbtFloatE5M2:
                case EbtFloatE4M3:
                case EbtFloat:
                case EbtDouble:
                    newConstArray[comp].setDConst(std::max(childConstUnions[0][arg0comp].getDConst(), childConstUnions[1][arg1comp].getDConst()));
                    break;
                case EbtInt:
                    newConstArray[comp].setIConst(std::max(childConstUnions[0][arg0comp].getIConst(), childConstUnions[1][arg1comp].getIConst()));
                    break;
                case EbtUint:
                    newConstArray[comp].setUConst(std::max(childConstUnions[0][arg0comp].getUConst(), childConstUnions[1][arg1comp].getUConst()));
                    break;
                case EbtInt8:
                    newConstArray[comp].setI8Const(std::max(childConstUnions[0][arg0comp].getI8Const(), childConstUnions[1][arg1comp].getI8Const()));
                    break;
                case EbtUint8:
                    newConstArray[comp].setU8Const(std::max(childConstUnions[0][arg0comp].getU8Const(), childConstUnions[1][arg1comp].getU8Const()));
                    break;
                case EbtInt16:
                    newConstArray[comp].setI16Const(std::max(childConstUnions[0][arg0comp].getI16Const(), childConstUnions[1][arg1comp].getI16Const()));
                    break;
                case EbtUint16:
                    newConstArray[comp].setU16Const(std::max(childConstUnions[0][arg0comp].getU16Const(), childConstUnions[1][arg1comp].getU16Const()));
                    break;
                case EbtInt64:
                    newConstArray[comp].setI64Const(std::max(childConstUnions[0][arg0comp].getI64Const(), childConstUnions[1][arg1comp].getI64Const()));
                    break;
                case EbtUint64:
                    newConstArray[comp].setU64Const(std::max(childConstUnions[0][arg0comp].getU64Const(), childConstUnions[1][arg1comp].getU64Const()));
                    break;
                default: assert(false && "Default missing");
                }
                break;
            case EOpClamp:
                switch(children[0]->getAsTyped()->getBasicType()) {
                case EbtFloat16:
                case EbtBFloat16:
                case EbtFloatE5M2:
                case EbtFloatE4M3:
                case EbtFloat:
                case EbtDouble:
                    newConstArray[comp].setDConst(std::min(std::max(childConstUnions[0][arg0comp].getDConst(), childConstUnions[1][arg1comp].getDConst()),
                                                                                                               childConstUnions[2][arg2comp].getDConst()));
                    break;
                case EbtUint:
                    newConstArray[comp].setUConst(std::min(std::max(childConstUnions[0][arg0comp].getUConst(), childConstUnions[1][arg1comp].getUConst()),
                                                                                                                   childConstUnions[2][arg2comp].getUConst()));
                    break;
                case EbtInt8:
                    newConstArray[comp].setI8Const(std::min(std::max(childConstUnions[0][arg0comp].getI8Const(), childConstUnions[1][arg1comp].getI8Const()),
                                                                                                                   childConstUnions[2][arg2comp].getI8Const()));
                    break;
                case EbtUint8:
                     newConstArray[comp].setU8Const(std::min(std::max(childConstUnions[0][arg0comp].getU8Const(), childConstUnions[1][arg1comp].getU8Const()),
                                                                                                                   childConstUnions[2][arg2comp].getU8Const()));
                    break;
                case EbtInt16:
                    newConstArray[comp].setI16Const(std::min(std::max(childConstUnions[0][arg0comp].getI16Const(), childConstUnions[1][arg1comp].getI16Const()),
                                                                                                                   childConstUnions[2][arg2comp].getI16Const()));
                    break;
                case EbtUint16:
                    newConstArray[comp].setU16Const(std::min(std::max(childConstUnions[0][arg0comp].getU16Const(), childConstUnions[1][arg1comp].getU16Const()),
                                                                                                                   childConstUnions[2][arg2comp].getU16Const()));
                    break;
                case EbtInt:
                    newConstArray[comp].setIConst(std::min(std::max(childConstUnions[0][arg0comp].getIConst(), childConstUnions[1][arg1comp].getIConst()),
                                                                                                                   childConstUnions[2][arg2comp].getIConst()));
                    break;
                case EbtInt64:
                    newConstArray[comp].setI64Const(std::min(std::max(childConstUnions[0][arg0comp].getI64Const(), childConstUnions[1][arg1comp].getI64Const()),
                                                                                                                       childConstUnions[2][arg2comp].getI64Const()));
                    break;
                case EbtUint64:
                    newConstArray[comp].setU64Const(std::min(std::max(childConstUnions[0][arg0comp].getU64Const(), childConstUnions[1][arg1comp].getU64Const()),
                                                                                                                       childConstUnions[2][arg2comp].getU64Const()));
                    break;
                default: assert(false && "Default missing");
                }
                break;
            case EOpLessThan:
                newConstArray[comp].setBConst(childConstUnions[0][arg0comp] < childConstUnions[1][arg1comp]);
                break;
            case EOpGreaterThan:
                newConstArray[comp].setBConst(childConstUnions[0][arg0comp] > childConstUnions[1][arg1comp]);
                break;
            case EOpLessThanEqual:
                newConstArray[comp].setBConst(! (childConstUnions[0][arg0comp] > childConstUnions[1][arg1comp]));
                break;
            case EOpGreaterThanEqual:
                newConstArray[comp].setBConst(! (childConstUnions[0][arg0comp] < childConstUnions[1][arg1comp]));
                break;
            case EOpVectorEqual:
                newConstArray[comp].setBConst(childConstUnions[0][arg0comp] == childConstUnions[1][arg1comp]);
                break;
            case EOpVectorNotEqual:
                newConstArray[comp].setBConst(childConstUnions[0][arg0comp] != childConstUnions[1][arg1comp]);
                break;
            case EOpMix:
                if (!children[0]->getAsTyped()->isFloatingDomain())
                    return aggrNode;
                if (children[2]->getAsTyped()->getBasicType() == EbtBool) {
                    newConstArray[comp].setDConst(childConstUnions[2][arg2comp].getBConst()
                        ? childConstUnions[1][arg1comp].getDConst()
                        : childConstUnions[0][arg0comp].getDConst());
                } else {
                    newConstArray[comp].setDConst(
                        childConstUnions[0][arg0comp].getDConst() * (1.0 - childConstUnions[2][arg2comp].getDConst()) +
                        childConstUnions[1][arg1comp].getDConst() *        childConstUnions[2][arg2comp].getDConst());
                }
                break;
            case EOpStep:
                newConstArray[comp].setDConst(childConstUnions[1][arg1comp].getDConst() < childConstUnions[0][arg0comp].getDConst() ? 0.0 : 1.0);
                break;
            case EOpSmoothStep:
            {
                double t = (childConstUnions[2][arg2comp].getDConst() - childConstUnions[0][arg0comp].getDConst()) /
                           (childConstUnions[1][arg1comp].getDConst() - childConstUnions[0][arg0comp].getDConst());
                if (t < 0.0)
                    t = 0.0;
                if (t > 1.0)
                    t = 1.0;
                newConstArray[comp].setDConst(t * t * (3.0 - 2.0 * t));
                break;
            }
            default:
                return aggrNode;
            }
        }
    } else {
        // Non-componentwise...

        int numComps = children[0]->getAsConstantUnion()->getType().computeNumComponents();
        double dot;

        switch (aggrNode->getOp()) {
        case EOpDistance:
        {
            double sum = 0.0;
            for (int comp = 0; comp < numComps; ++comp) {
                double diff = childConstUnions[1][comp].getDConst() - childConstUnions[0][comp].getDConst();
                sum += diff * diff;
            }
            newConstArray[0].setDConst(sqrt(sum));
            break;
        }
        case EOpDot:
            if (!children[0]->getAsTyped()->isFloatingDomain()) {
                return aggrNode;
            }
            newConstArray[0].setDConst(childConstUnions[0].dot(childConstUnions[1]));
            break;
        case EOpCross:
            newConstArray[0] = childConstUnions[0][1] * childConstUnions[1][2] - childConstUnions[0][2] * childConstUnions[1][1];
            newConstArray[1] = childConstUnions[0][2] * childConstUnions[1][0] - childConstUnions[0][0] * childConstUnions[1][2];
            newConstArray[2] = childConstUnions[0][0] * childConstUnions[1][1] - childConstUnions[0][1] * childConstUnions[1][0];
            break;
        case EOpFaceForward:
            // If dot(Nref, I) < 0 return N, otherwise return -N:  Arguments are (N, I, Nref).
            dot = childConstUnions[1].dot(childConstUnions[2]);
            for (int comp = 0; comp < numComps; ++comp) {
                if (dot < 0.0)
                    newConstArray[comp] = childConstUnions[0][comp];
                else
                    newConstArray[comp].setDConst(-childConstUnions[0][comp].getDConst());
            }
            break;
        case EOpReflect:
            // I - 2 * dot(N, I) * N:  Arguments are (I, N).
            dot = childConstUnions[0].dot(childConstUnions[1]);
            dot *= 2.0;
            for (int comp = 0; comp < numComps; ++comp)
                newConstArray[comp].setDConst(childConstUnions[0][comp].getDConst() - dot * childConstUnions[1][comp].getDConst());
            break;
        case EOpRefract:
        {
            // Arguments are (I, N, eta).
            // k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I))
            // if (k < 0.0)
            //     return dvec(0.0)
            // else
            //     return eta * I - (eta * dot(N, I) + sqrt(k)) * N
            dot = childConstUnions[0].dot(childConstUnions[1]);
            double eta = childConstUnions[2][0].getDConst();
            double k = 1.0 - eta * eta * (1.0 - dot * dot);
            if (k < 0.0) {
                for (int comp = 0; comp < numComps; ++comp)
                    newConstArray[comp].setDConst(0.0);
            } else {
                for (int comp = 0; comp < numComps; ++comp)
                    newConstArray[comp].setDConst(eta * childConstUnions[0][comp].getDConst() - (eta * dot + sqrt(k)) * childConstUnions[1][comp].getDConst());
            }
            break;
        }
        case EOpOuterProduct:
        {
            int numRows = numComps;
            int numCols = children[1]->getAsConstantUnion()->getType().computeNumComponents();
            for (int row = 0; row < numRows; ++row)
                for (int col = 0; col < numCols; ++col)
                    newConstArray[col * numRows + row] = childConstUnions[0][row] * childConstUnions[1][col];
            break;
        }
        default:
            return aggrNode;
        }
    }

    TIntermConstantUnion *newNode = new TIntermConstantUnion(newConstArray, aggrNode->getType());
    newNode->getWritableType().getQualifier().storage = EvqConst;
    newNode->setLoc(aggrNode->getLoc());

    return newNode;
}

bool TIntermediate::areAllChildConst(TIntermAggregate* aggrNode)
{
    bool allConstant = true;

    // check if all the child nodes are constants so that they can be inserted into
    // the parent node
    if (aggrNode) {
        TIntermSequence& childSequenceVector = aggrNode->getSequence();
        for (TIntermSequence::iterator p  = childSequenceVector.begin();
                                       p != childSequenceVector.end(); p++) {
            if (!(*p)->getAsTyped()->getAsConstantUnion())
                return false;
        }
    }

    return allConstant;
}

TIntermTyped* TIntermediate::foldConstructor(TIntermAggregate* aggrNode)
{
    bool error = false;

    TConstUnionArray unionArray(aggrNode->getType().computeNumComponents());
    if (aggrNode->getSequence().size() == 1)
        error = parseConstTree(aggrNode, unionArray, aggrNode->getOp(), aggrNode->getType(), true);
    else
        error = parseConstTree(aggrNode, unionArray, aggrNode->getOp(), aggrNode->getType());

    if (error)
        return aggrNode;

    return addConstantUnion(unionArray, aggrNode->getType(), aggrNode->getLoc());
}

//
// Constant folding of a bracket (array-style) dereference or struct-like dot
// dereference.  Can handle anything except a multi-character swizzle, though
// all swizzles may go to foldSwizzle().
//
TIntermTyped* TIntermediate::foldDereference(TIntermTyped* node, int index, const TSourceLoc& loc)
{
    TType dereferencedType(node->getType(), index);
    dereferencedType.getQualifier().storage = EvqConst;
    TIntermTyped* result = nullptr;
    int size = dereferencedType.computeNumComponents();

    // arrays, vectors, matrices, all use simple multiplicative math
    // while structures need to add up heterogeneous members
    int start;
    if (node->getType().isCoopMat())
        start = 0;
    else if (node->isArray() || ! node->isStruct())
        start = size * index;
    else {
        // it is a structure
        assert(node->isStruct());
        start = 0;
        for (int i = 0; i < index; ++i)
            start += (*node->getType().getStruct())[i].type->computeNumComponents();
    }

    result = addConstantUnion(TConstUnionArray(node->getAsConstantUnion()->getConstArray(), start, size), node->getType(), loc);

    if (result == nullptr)
        result = node;
    else
        result->setType(dereferencedType);

    return result;
}

//
// Make a constant vector node or constant scalar node, representing a given
// constant vector and constant swizzle into it.
//
TIntermTyped* TIntermediate::foldSwizzle(TIntermTyped* node, TSwizzleSelectors<TVectorSelector>& selectors, const TSourceLoc& loc)
{
    const TConstUnionArray& unionArray = node->getAsConstantUnion()->getConstArray();
    TConstUnionArray constArray(selectors.size());

    for (int i = 0; i < selectors.size(); i++)
        constArray[i] = unionArray[selectors[i]];

    TIntermTyped* result = addConstantUnion(constArray, node->getType(), loc);

    if (result == nullptr)
        result = node;
    else
        result->setType(TType(node->getBasicType(), EvqConst, selectors.size()));

    return result;
}

} // end namespace glslang
