//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_OPERATOR_H_
#define COMPILER_TRANSLATOR_OPERATOR_H_

//
// Operators used by the high-level (parse tree) representation.
//
enum TOperator
{
    EOpNull,  // if in a node, should only mean a node is still being built

    // Call a function defined in the AST. This might be a user-defined function or a function
    // inserted by an AST transformation.
    EOpCallFunctionInAST,

    // Call an internal helper function with a raw implementation - the implementation can't be
    // subject to AST transformations. Raw functions have a few constraints to keep them compatible
    // with AST traversers:
    // * They should not return arrays.
    // * They should not have out parameters.
    EOpCallInternalRawFunction,

    // Call a built-in function like a texture or image function.
    EOpCallBuiltInFunction,

    //
    // Unary operators
    //

    EOpNegative,
    EOpPositive,
    EOpLogicalNot,
    EOpBitwiseNot,

    EOpPostIncrement,
    EOpPostDecrement,
    EOpPreIncrement,
    EOpPreDecrement,

    EOpArrayLength,

    //
    // binary operations (ones with special GLSL syntax are used in TIntermBinary nodes, others in
    // TIntermAggregate nodes)
    //

    EOpAdd,
    EOpSub,
    EOpMul,
    EOpDiv,
    EOpIMod,

    EOpEqual,
    EOpNotEqual,
    EOpLessThan,
    EOpGreaterThan,
    EOpLessThanEqual,
    EOpGreaterThanEqual,

    EOpEqualComponentWise,
    EOpNotEqualComponentWise,
    EOpLessThanComponentWise,
    EOpLessThanEqualComponentWise,
    EOpGreaterThanComponentWise,
    EOpGreaterThanEqualComponentWise,

    EOpComma,

    EOpVectorTimesScalar,
    EOpVectorTimesMatrix,
    EOpMatrixTimesVector,
    EOpMatrixTimesScalar,
    EOpMatrixTimesMatrix,

    EOpLogicalOr,
    EOpLogicalXor,
    EOpLogicalAnd,

    EOpBitShiftLeft,
    EOpBitShiftRight,

    EOpBitwiseAnd,
    EOpBitwiseXor,
    EOpBitwiseOr,

    EOpIndexDirect,
    EOpIndexIndirect,
    EOpIndexDirectStruct,
    EOpIndexDirectInterfaceBlock,

    //
    // Built-in functions mapped to operators (either unary or with multiple parameters)
    //

    EOpRadians,
    EOpDegrees,
    EOpSin,
    EOpCos,
    EOpTan,
    EOpAsin,
    EOpAcos,
    EOpAtan,

    EOpSinh,
    EOpCosh,
    EOpTanh,
    EOpAsinh,
    EOpAcosh,
    EOpAtanh,

    EOpPow,
    EOpExp,
    EOpLog,
    EOpExp2,
    EOpLog2,
    EOpSqrt,
    EOpInversesqrt,

    EOpAbs,
    EOpSign,
    EOpFloor,
    EOpTrunc,
    EOpRound,
    EOpRoundEven,
    EOpCeil,
    EOpFract,
    EOpMod,
    EOpModf,
    EOpMin,
    EOpMax,
    EOpClamp,
    EOpMix,
    EOpStep,
    EOpSmoothstep,
    EOpIsnan,
    EOpIsinf,

    EOpFloatBitsToInt,
    EOpFloatBitsToUint,
    EOpIntBitsToFloat,
    EOpUintBitsToFloat,

    EOpFrexp,
    EOpLdexp,

    EOpPackSnorm2x16,
    EOpPackUnorm2x16,
    EOpPackHalf2x16,
    EOpUnpackSnorm2x16,
    EOpUnpackUnorm2x16,
    EOpUnpackHalf2x16,

    EOpPackUnorm4x8,
    EOpPackSnorm4x8,
    EOpUnpackUnorm4x8,
    EOpUnpackSnorm4x8,

    EOpLength,
    EOpDistance,
    EOpDot,
    EOpCross,
    EOpNormalize,
    EOpFaceforward,
    EOpReflect,
    EOpRefract,

    EOpDFdx,    // Fragment only, OES_standard_derivatives extension
    EOpDFdy,    // Fragment only, OES_standard_derivatives extension
    EOpFwidth,  // Fragment only, OES_standard_derivatives extension

    EOpMulMatrixComponentWise,
    EOpOuterProduct,
    EOpTranspose,
    EOpDeterminant,
    EOpInverse,

    EOpAny,
    EOpAll,
    EOpLogicalNotComponentWise,

    EOpBitfieldExtract,
    EOpBitfieldInsert,
    EOpBitfieldReverse,
    EOpBitCount,
    EOpFindLSB,
    EOpFindMSB,
    EOpUaddCarry,
    EOpUsubBorrow,
    EOpUmulExtended,
    EOpImulExtended,

    //
    // Branch
    //

    EOpKill,  // Fragment only
    EOpReturn,
    EOpBreak,
    EOpContinue,

    //
    // Constructor
    //

    EOpConstruct,

    //
    // moves
    //

    EOpAssign,
    EOpInitialize,
    EOpAddAssign,
    EOpSubAssign,

    EOpMulAssign,
    EOpVectorTimesMatrixAssign,
    EOpVectorTimesScalarAssign,
    EOpMatrixTimesScalarAssign,
    EOpMatrixTimesMatrixAssign,

    EOpDivAssign,
    EOpIModAssign,
    EOpBitShiftLeftAssign,
    EOpBitShiftRightAssign,
    EOpBitwiseAndAssign,
    EOpBitwiseXorAssign,
    EOpBitwiseOrAssign,

    // barriers
    EOpBarrier,
    EOpMemoryBarrier,
    EOpMemoryBarrierAtomicCounter,
    EOpMemoryBarrierBuffer,
    EOpMemoryBarrierImage,
    EOpMemoryBarrierShared,
    EOpGroupMemoryBarrier,

    // Atomic functions
    EOpAtomicAdd,
    EOpAtomicMin,
    EOpAtomicMax,
    EOpAtomicAnd,
    EOpAtomicOr,
    EOpAtomicXor,
    EOpAtomicExchange,
    EOpAtomicCompSwap,

    // Geometry only
    EOpEmitVertex,
    EOpEndPrimitive,

    // Desktop GLSL functions
    EOpFTransform,
    EOpFma,
    EOpPackDouble2x32,
    EOpUnpackDouble2x32,
};

// Returns the string corresponding to the operator in GLSL
const char *GetOperatorString(TOperator op);

// Say whether or not a binary or unary operation changes the value of a variable.
bool IsAssignment(TOperator op);

// Say whether or not an operator represents an atomic function.
bool IsAtomicFunction(TOperator op);

#endif  // COMPILER_TRANSLATOR_OPERATOR_H_
