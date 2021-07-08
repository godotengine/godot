//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/Operator.h"

const char *GetOperatorString(TOperator op)
{
    switch (op)
    {
            // Note: EOpNull and EOpCall* can't be handled here.

        case EOpNegative:
            return "-";
        case EOpPositive:
            return "+";
        case EOpLogicalNot:
            return "!";
        case EOpBitwiseNot:
            return "~";

        case EOpPostIncrement:
            return "++";
        case EOpPostDecrement:
            return "--";
        case EOpPreIncrement:
            return "++";
        case EOpPreDecrement:
            return "--";

        case EOpArrayLength:
            return ".length()";

        case EOpAdd:
            return "+";
        case EOpSub:
            return "-";
        case EOpMul:
            return "*";
        case EOpDiv:
            return "/";
        case EOpIMod:
            return "%";

        case EOpEqual:
            return "==";
        case EOpNotEqual:
            return "!=";
        case EOpLessThan:
            return "<";
        case EOpGreaterThan:
            return ">";
        case EOpLessThanEqual:
            return "<=";
        case EOpGreaterThanEqual:
            return ">=";

        case EOpEqualComponentWise:
            return "equal";
        case EOpNotEqualComponentWise:
            return "notEqual";
        case EOpLessThanComponentWise:
            return "lessThan";
        case EOpGreaterThanComponentWise:
            return "greaterThan";
        case EOpLessThanEqualComponentWise:
            return "lessThanEqual";
        case EOpGreaterThanEqualComponentWise:
            return "greaterThanEqual";

        case EOpComma:
            return ",";

        // Fall-through.
        case EOpVectorTimesScalar:
        case EOpVectorTimesMatrix:
        case EOpMatrixTimesVector:
        case EOpMatrixTimesScalar:
        case EOpMatrixTimesMatrix:
            return "*";

        case EOpLogicalOr:
            return "||";
        case EOpLogicalXor:
            return "^^";
        case EOpLogicalAnd:
            return "&&";

        case EOpBitShiftLeft:
            return "<<";
        case EOpBitShiftRight:
            return ">>";

        case EOpBitwiseAnd:
            return "&";
        case EOpBitwiseXor:
            return "^";
        case EOpBitwiseOr:
            return "|";

        // Fall-through.
        case EOpIndexDirect:
        case EOpIndexIndirect:
            return "[]";

        case EOpIndexDirectStruct:
        case EOpIndexDirectInterfaceBlock:
            return ".";

        case EOpRadians:
            return "radians";
        case EOpDegrees:
            return "degrees";
        case EOpSin:
            return "sin";
        case EOpCos:
            return "cos";
        case EOpTan:
            return "tan";
        case EOpAsin:
            return "asin";
        case EOpAcos:
            return "acos";
        case EOpAtan:
            return "atan";

        case EOpSinh:
            return "sinh";
        case EOpCosh:
            return "cosh";
        case EOpTanh:
            return "tanh";
        case EOpAsinh:
            return "asinh";
        case EOpAcosh:
            return "acosh";
        case EOpAtanh:
            return "atanh";

        case EOpPow:
            return "pow";
        case EOpExp:
            return "exp";
        case EOpLog:
            return "log";
        case EOpExp2:
            return "exp2";
        case EOpLog2:
            return "log2";
        case EOpSqrt:
            return "sqrt";
        case EOpInversesqrt:
            return "inversesqrt";

        case EOpAbs:
            return "abs";
        case EOpSign:
            return "sign";
        case EOpFloor:
            return "floor";
        case EOpTrunc:
            return "trunc";
        case EOpRound:
            return "round";
        case EOpRoundEven:
            return "roundEven";
        case EOpCeil:
            return "ceil";
        case EOpFract:
            return "fract";
        case EOpMod:
            return "mod";
        case EOpModf:
            return "modf";
        case EOpMin:
            return "min";
        case EOpMax:
            return "max";
        case EOpClamp:
            return "clamp";
        case EOpMix:
            return "mix";
        case EOpStep:
            return "step";
        case EOpSmoothstep:
            return "smoothstep";
        case EOpIsnan:
            return "isnan";
        case EOpIsinf:
            return "isinf";

        case EOpFloatBitsToInt:
            return "floatBitsToInt";
        case EOpFloatBitsToUint:
            return "floatBitsToUint";
        case EOpIntBitsToFloat:
            return "intBitsToFloat";
        case EOpUintBitsToFloat:
            return "uintBitsToFloat";

        case EOpFrexp:
            return "frexp";
        case EOpLdexp:
            return "ldexp";

        case EOpPackSnorm2x16:
            return "packSnorm2x16";
        case EOpPackUnorm2x16:
            return "packUnorm2x16";
        case EOpPackHalf2x16:
            return "packHalf2x16";
        case EOpUnpackSnorm2x16:
            return "unpackSnorm2x16";
        case EOpUnpackUnorm2x16:
            return "unpackUnorm2x16";
        case EOpUnpackHalf2x16:
            return "unpackHalf2x16";

        case EOpPackUnorm4x8:
            return "packUnorm4x8";
        case EOpPackSnorm4x8:
            return "packSnorm4x8";
        case EOpUnpackUnorm4x8:
            return "unpackUnorm4x8";
        case EOpUnpackSnorm4x8:
            return "unpackSnorm4x8";

        case EOpLength:
            return "length";
        case EOpDistance:
            return "distance";
        case EOpDot:
            return "dot";
        case EOpCross:
            return "cross";
        case EOpNormalize:
            return "normalize";
        case EOpFaceforward:
            return "faceforward";
        case EOpReflect:
            return "reflect";
        case EOpRefract:
            return "refract";

        case EOpDFdx:
            return "dFdx";
        case EOpDFdy:
            return "dFdy";
        case EOpFwidth:
            return "fwidth";

        case EOpMulMatrixComponentWise:
            return "matrixCompMult";
        case EOpOuterProduct:
            return "outerProduct";
        case EOpTranspose:
            return "transpose";
        case EOpDeterminant:
            return "determinant";
        case EOpInverse:
            return "inverse";

        case EOpAny:
            return "any";
        case EOpAll:
            return "all";
        case EOpLogicalNotComponentWise:
            return "not";

        case EOpBitfieldExtract:
            return "bitfieldExtract";
        case EOpBitfieldInsert:
            return "bitfieldInsert";
        case EOpBitfieldReverse:
            return "bitfieldReverse";
        case EOpBitCount:
            return "bitCount";
        case EOpFindLSB:
            return "findLSB";
        case EOpFindMSB:
            return "findMSB";
        case EOpUaddCarry:
            return "uaddCarry";
        case EOpUsubBorrow:
            return "usubBorrow";
        case EOpUmulExtended:
            return "umulExtended";
        case EOpImulExtended:
            return "imulExtended";

        case EOpKill:
            return "kill";
        case EOpReturn:
            return "return";
        case EOpBreak:
            return "break";
        case EOpContinue:
            return "continue";

        case EOpAssign:
            return "=";
        case EOpInitialize:
            return "=";
        case EOpAddAssign:
            return "+=";
        case EOpSubAssign:
            return "-=";

        // Fall-through.
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
            return "*=";

        case EOpDivAssign:
            return "/=";
        case EOpIModAssign:
            return "%=";
        case EOpBitShiftLeftAssign:
            return "<<=";
        case EOpBitShiftRightAssign:
            return ">>=";
        case EOpBitwiseAndAssign:
            return "&=";
        case EOpBitwiseXorAssign:
            return "^=";
        case EOpBitwiseOrAssign:
            return "|=";
        case EOpBarrier:
            return "barrier";
        case EOpMemoryBarrier:
            return "memoryBarrier";
        case EOpMemoryBarrierAtomicCounter:
            return "memoryBarrierAtomicCounter";
        case EOpMemoryBarrierBuffer:
            return "memoryBarrierBuffer";
        case EOpMemoryBarrierImage:
            return "memoryBarrierImage";
        case EOpMemoryBarrierShared:
            return "memoryBarrierShared";
        case EOpGroupMemoryBarrier:
            return "groupMemoryBarrier";

        case EOpAtomicAdd:
            return "atomicAdd";
        case EOpAtomicMin:
            return "atomicMin";
        case EOpAtomicMax:
            return "atomicMax";
        case EOpAtomicAnd:
            return "atomicAnd";
        case EOpAtomicOr:
            return "atomicOr";
        case EOpAtomicXor:
            return "atomicXor";
        case EOpAtomicExchange:
            return "atomicExchange";
        case EOpAtomicCompSwap:
            return "atomicCompSwap";

        case EOpEmitVertex:
            return "EmitVertex";
        case EOpEndPrimitive:
            return "EndPrimitive";
        default:
            break;
    }
    return "";
}

bool IsAssignment(TOperator op)
{
    switch (op)
    {
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpPreIncrement:
        case EOpPreDecrement:
        case EOpAssign:
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

bool IsAtomicFunction(TOperator op)
{
    switch (op)
    {
        case EOpAtomicAdd:
        case EOpAtomicMin:
        case EOpAtomicMax:
        case EOpAtomicAnd:
        case EOpAtomicOr:
        case EOpAtomicXor:
        case EOpAtomicExchange:
        case EOpAtomicCompSwap:
            return true;
        default:
            return false;
    }
}
