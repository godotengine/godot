//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2016 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

//
// Definition of the in-memory high-level intermediate representation
// of shaders.  This is a tree that parser creates.
//
// Nodes in the tree are defined as a hierarchy of classes derived from
// TIntermNode. Each is a node in a tree.  There is no preset branching factor;
// each node can have it's own type of list of children.
//

#ifndef __INTERMEDIATE_H
#define __INTERMEDIATE_H

#if defined(_MSC_VER) && _MSC_VER >= 1900
    #pragma warning(disable : 4464) // relative include path contains '..'
    #pragma warning(disable : 5026) // 'glslang::TIntermUnary': move constructor was implicitly defined as deleted
#endif

#include "../Include/Common.h"
#include "../Include/Types.h"
#include "../Include/ConstantUnion.h"

namespace glslang {

class TIntermediate;

//
// Operators used by the high-level (parse tree) representation.
//
enum TOperator {
    EOpNull,            // if in a node, should only mean a node is still being built
    EOpSequence,        // denotes a list of statements, or parameters, etc.
    EOpLinkerObjects,   // for aggregate node of objects the linker may need, if not reference by the rest of the AST
    EOpFunctionCall,
    EOpFunction,        // For function definition
    EOpParameters,      // an aggregate listing the parameters to a function
#ifndef GLSLANG_WEB
    EOpSpirvInst,
#endif

    //
    // Unary operators
    //

    EOpNegative,
    EOpLogicalNot,
    EOpVectorLogicalNot,
    EOpBitwiseNot,

    EOpPostIncrement,
    EOpPostDecrement,
    EOpPreIncrement,
    EOpPreDecrement,

    EOpCopyObject,

    // (u)int* -> bool
    EOpConvInt8ToBool,
    EOpConvUint8ToBool,
    EOpConvInt16ToBool,
    EOpConvUint16ToBool,
    EOpConvIntToBool,
    EOpConvUintToBool,
    EOpConvInt64ToBool,
    EOpConvUint64ToBool,

    // float* -> bool
    EOpConvFloat16ToBool,
    EOpConvFloatToBool,
    EOpConvDoubleToBool,

    // bool -> (u)int*
    EOpConvBoolToInt8,
    EOpConvBoolToUint8,
    EOpConvBoolToInt16,
    EOpConvBoolToUint16,
    EOpConvBoolToInt,
    EOpConvBoolToUint,
    EOpConvBoolToInt64,
    EOpConvBoolToUint64,

    // bool -> float*
    EOpConvBoolToFloat16,
    EOpConvBoolToFloat,
    EOpConvBoolToDouble,

    // int8_t -> (u)int*
    EOpConvInt8ToInt16,
    EOpConvInt8ToInt,
    EOpConvInt8ToInt64,
    EOpConvInt8ToUint8,
    EOpConvInt8ToUint16,
    EOpConvInt8ToUint,
    EOpConvInt8ToUint64,

    // uint8_t -> (u)int*
    EOpConvUint8ToInt8,
    EOpConvUint8ToInt16,
    EOpConvUint8ToInt,
    EOpConvUint8ToInt64,
    EOpConvUint8ToUint16,
    EOpConvUint8ToUint,
    EOpConvUint8ToUint64,

    // int8_t -> float*
    EOpConvInt8ToFloat16,
    EOpConvInt8ToFloat,
    EOpConvInt8ToDouble,

    // uint8_t -> float*
    EOpConvUint8ToFloat16,
    EOpConvUint8ToFloat,
    EOpConvUint8ToDouble,

    // int16_t -> (u)int*
    EOpConvInt16ToInt8,
    EOpConvInt16ToInt,
    EOpConvInt16ToInt64,
    EOpConvInt16ToUint8,
    EOpConvInt16ToUint16,
    EOpConvInt16ToUint,
    EOpConvInt16ToUint64,

    // uint16_t -> (u)int*
    EOpConvUint16ToInt8,
    EOpConvUint16ToInt16,
    EOpConvUint16ToInt,
    EOpConvUint16ToInt64,
    EOpConvUint16ToUint8,
    EOpConvUint16ToUint,
    EOpConvUint16ToUint64,

    // int16_t -> float*
    EOpConvInt16ToFloat16,
    EOpConvInt16ToFloat,
    EOpConvInt16ToDouble,

    // uint16_t -> float*
    EOpConvUint16ToFloat16,
    EOpConvUint16ToFloat,
    EOpConvUint16ToDouble,

    // int32_t -> (u)int*
    EOpConvIntToInt8,
    EOpConvIntToInt16,
    EOpConvIntToInt64,
    EOpConvIntToUint8,
    EOpConvIntToUint16,
    EOpConvIntToUint,
    EOpConvIntToUint64,

    // uint32_t -> (u)int*
    EOpConvUintToInt8,
    EOpConvUintToInt16,
    EOpConvUintToInt,
    EOpConvUintToInt64,
    EOpConvUintToUint8,
    EOpConvUintToUint16,
    EOpConvUintToUint64,

    // int32_t -> float*
    EOpConvIntToFloat16,
    EOpConvIntToFloat,
    EOpConvIntToDouble,

    // uint32_t -> float*
    EOpConvUintToFloat16,
    EOpConvUintToFloat,
    EOpConvUintToDouble,

    // int64_t -> (u)int*
    EOpConvInt64ToInt8,
    EOpConvInt64ToInt16,
    EOpConvInt64ToInt,
    EOpConvInt64ToUint8,
    EOpConvInt64ToUint16,
    EOpConvInt64ToUint,
    EOpConvInt64ToUint64,

    // uint64_t -> (u)int*
    EOpConvUint64ToInt8,
    EOpConvUint64ToInt16,
    EOpConvUint64ToInt,
    EOpConvUint64ToInt64,
    EOpConvUint64ToUint8,
    EOpConvUint64ToUint16,
    EOpConvUint64ToUint,

    // int64_t -> float*
    EOpConvInt64ToFloat16,
    EOpConvInt64ToFloat,
    EOpConvInt64ToDouble,

    // uint64_t -> float*
    EOpConvUint64ToFloat16,
    EOpConvUint64ToFloat,
    EOpConvUint64ToDouble,

    // float16_t -> (u)int*
    EOpConvFloat16ToInt8,
    EOpConvFloat16ToInt16,
    EOpConvFloat16ToInt,
    EOpConvFloat16ToInt64,
    EOpConvFloat16ToUint8,
    EOpConvFloat16ToUint16,
    EOpConvFloat16ToUint,
    EOpConvFloat16ToUint64,

    // float16_t -> float*
    EOpConvFloat16ToFloat,
    EOpConvFloat16ToDouble,

    // float -> (u)int*
    EOpConvFloatToInt8,
    EOpConvFloatToInt16,
    EOpConvFloatToInt,
    EOpConvFloatToInt64,
    EOpConvFloatToUint8,
    EOpConvFloatToUint16,
    EOpConvFloatToUint,
    EOpConvFloatToUint64,

    // float -> float*
    EOpConvFloatToFloat16,
    EOpConvFloatToDouble,

    // float64 _t-> (u)int*
    EOpConvDoubleToInt8,
    EOpConvDoubleToInt16,
    EOpConvDoubleToInt,
    EOpConvDoubleToInt64,
    EOpConvDoubleToUint8,
    EOpConvDoubleToUint16,
    EOpConvDoubleToUint,
    EOpConvDoubleToUint64,

    // float64_t -> float*
    EOpConvDoubleToFloat16,
    EOpConvDoubleToFloat,

    // uint64_t <-> pointer
    EOpConvUint64ToPtr,
    EOpConvPtrToUint64,

    // uvec2 <-> pointer
    EOpConvUvec2ToPtr,
    EOpConvPtrToUvec2,

    // uint64_t -> accelerationStructureEXT
    EOpConvUint64ToAccStruct,

    // uvec2 -> accelerationStructureEXT
    EOpConvUvec2ToAccStruct,

    //
    // binary operations
    //

    EOpAdd,
    EOpSub,
    EOpMul,
    EOpDiv,
    EOpMod,
    EOpRightShift,
    EOpLeftShift,
    EOpAnd,
    EOpInclusiveOr,
    EOpExclusiveOr,
    EOpEqual,
    EOpNotEqual,
    EOpVectorEqual,
    EOpVectorNotEqual,
    EOpLessThan,
    EOpGreaterThan,
    EOpLessThanEqual,
    EOpGreaterThanEqual,
    EOpComma,

    EOpVectorTimesScalar,
    EOpVectorTimesMatrix,
    EOpMatrixTimesVector,
    EOpMatrixTimesScalar,

    EOpLogicalOr,
    EOpLogicalXor,
    EOpLogicalAnd,

    EOpIndexDirect,
    EOpIndexIndirect,
    EOpIndexDirectStruct,

    EOpVectorSwizzle,

    EOpMethod,
    EOpScoping,

    //
    // Built-in functions mapped to operators
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
    EOpInverseSqrt,

    EOpAbs,
    EOpSign,
    EOpFloor,
    EOpTrunc,
    EOpRound,
    EOpRoundEven,
    EOpCeil,
    EOpFract,
    EOpModf,
    EOpMin,
    EOpMax,
    EOpClamp,
    EOpMix,
    EOpStep,
    EOpSmoothStep,

    EOpIsNan,
    EOpIsInf,

    EOpFma,

    EOpFrexp,
    EOpLdexp,

    EOpFloatBitsToInt,
    EOpFloatBitsToUint,
    EOpIntBitsToFloat,
    EOpUintBitsToFloat,
    EOpDoubleBitsToInt64,
    EOpDoubleBitsToUint64,
    EOpInt64BitsToDouble,
    EOpUint64BitsToDouble,
    EOpFloat16BitsToInt16,
    EOpFloat16BitsToUint16,
    EOpInt16BitsToFloat16,
    EOpUint16BitsToFloat16,
    EOpPackSnorm2x16,
    EOpUnpackSnorm2x16,
    EOpPackUnorm2x16,
    EOpUnpackUnorm2x16,
    EOpPackSnorm4x8,
    EOpUnpackSnorm4x8,
    EOpPackUnorm4x8,
    EOpUnpackUnorm4x8,
    EOpPackHalf2x16,
    EOpUnpackHalf2x16,
    EOpPackDouble2x32,
    EOpUnpackDouble2x32,
    EOpPackInt2x32,
    EOpUnpackInt2x32,
    EOpPackUint2x32,
    EOpUnpackUint2x32,
    EOpPackFloat2x16,
    EOpUnpackFloat2x16,
    EOpPackInt2x16,
    EOpUnpackInt2x16,
    EOpPackUint2x16,
    EOpUnpackUint2x16,
    EOpPackInt4x16,
    EOpUnpackInt4x16,
    EOpPackUint4x16,
    EOpUnpackUint4x16,
    EOpPack16,
    EOpPack32,
    EOpPack64,
    EOpUnpack32,
    EOpUnpack16,
    EOpUnpack8,

    EOpLength,
    EOpDistance,
    EOpDot,
    EOpCross,
    EOpNormalize,
    EOpFaceForward,
    EOpReflect,
    EOpRefract,

    EOpMin3,
    EOpMax3,
    EOpMid3,

    EOpDPdx,            // Fragment only
    EOpDPdy,            // Fragment only
    EOpFwidth,          // Fragment only
    EOpDPdxFine,        // Fragment only
    EOpDPdyFine,        // Fragment only
    EOpFwidthFine,      // Fragment only
    EOpDPdxCoarse,      // Fragment only
    EOpDPdyCoarse,      // Fragment only
    EOpFwidthCoarse,    // Fragment only

    EOpInterpolateAtCentroid, // Fragment only
    EOpInterpolateAtSample,   // Fragment only
    EOpInterpolateAtOffset,   // Fragment only
    EOpInterpolateAtVertex,

    EOpMatrixTimesMatrix,
    EOpOuterProduct,
    EOpDeterminant,
    EOpMatrixInverse,
    EOpTranspose,

    EOpFtransform,

    EOpNoise,

    EOpEmitVertex,           // geometry only
    EOpEndPrimitive,         // geometry only
    EOpEmitStreamVertex,     // geometry only
    EOpEndStreamPrimitive,   // geometry only

    EOpBarrier,
    EOpMemoryBarrier,
    EOpMemoryBarrierAtomicCounter,
    EOpMemoryBarrierBuffer,
    EOpMemoryBarrierImage,
    EOpMemoryBarrierShared,  // compute only
    EOpGroupMemoryBarrier,   // compute only

    EOpBallot,
    EOpReadInvocation,
    EOpReadFirstInvocation,

    EOpAnyInvocation,
    EOpAllInvocations,
    EOpAllInvocationsEqual,

    EOpSubgroupGuardStart,
    EOpSubgroupBarrier,
    EOpSubgroupMemoryBarrier,
    EOpSubgroupMemoryBarrierBuffer,
    EOpSubgroupMemoryBarrierImage,
    EOpSubgroupMemoryBarrierShared, // compute only
    EOpSubgroupElect,
    EOpSubgroupAll,
    EOpSubgroupAny,
    EOpSubgroupAllEqual,
    EOpSubgroupBroadcast,
    EOpSubgroupBroadcastFirst,
    EOpSubgroupBallot,
    EOpSubgroupInverseBallot,
    EOpSubgroupBallotBitExtract,
    EOpSubgroupBallotBitCount,
    EOpSubgroupBallotInclusiveBitCount,
    EOpSubgroupBallotExclusiveBitCount,
    EOpSubgroupBallotFindLSB,
    EOpSubgroupBallotFindMSB,
    EOpSubgroupShuffle,
    EOpSubgroupShuffleXor,
    EOpSubgroupShuffleUp,
    EOpSubgroupShuffleDown,
    EOpSubgroupAdd,
    EOpSubgroupMul,
    EOpSubgroupMin,
    EOpSubgroupMax,
    EOpSubgroupAnd,
    EOpSubgroupOr,
    EOpSubgroupXor,
    EOpSubgroupInclusiveAdd,
    EOpSubgroupInclusiveMul,
    EOpSubgroupInclusiveMin,
    EOpSubgroupInclusiveMax,
    EOpSubgroupInclusiveAnd,
    EOpSubgroupInclusiveOr,
    EOpSubgroupInclusiveXor,
    EOpSubgroupExclusiveAdd,
    EOpSubgroupExclusiveMul,
    EOpSubgroupExclusiveMin,
    EOpSubgroupExclusiveMax,
    EOpSubgroupExclusiveAnd,
    EOpSubgroupExclusiveOr,
    EOpSubgroupExclusiveXor,
    EOpSubgroupClusteredAdd,
    EOpSubgroupClusteredMul,
    EOpSubgroupClusteredMin,
    EOpSubgroupClusteredMax,
    EOpSubgroupClusteredAnd,
    EOpSubgroupClusteredOr,
    EOpSubgroupClusteredXor,
    EOpSubgroupQuadBroadcast,
    EOpSubgroupQuadSwapHorizontal,
    EOpSubgroupQuadSwapVertical,
    EOpSubgroupQuadSwapDiagonal,

    EOpSubgroupPartition,
    EOpSubgroupPartitionedAdd,
    EOpSubgroupPartitionedMul,
    EOpSubgroupPartitionedMin,
    EOpSubgroupPartitionedMax,
    EOpSubgroupPartitionedAnd,
    EOpSubgroupPartitionedOr,
    EOpSubgroupPartitionedXor,
    EOpSubgroupPartitionedInclusiveAdd,
    EOpSubgroupPartitionedInclusiveMul,
    EOpSubgroupPartitionedInclusiveMin,
    EOpSubgroupPartitionedInclusiveMax,
    EOpSubgroupPartitionedInclusiveAnd,
    EOpSubgroupPartitionedInclusiveOr,
    EOpSubgroupPartitionedInclusiveXor,
    EOpSubgroupPartitionedExclusiveAdd,
    EOpSubgroupPartitionedExclusiveMul,
    EOpSubgroupPartitionedExclusiveMin,
    EOpSubgroupPartitionedExclusiveMax,
    EOpSubgroupPartitionedExclusiveAnd,
    EOpSubgroupPartitionedExclusiveOr,
    EOpSubgroupPartitionedExclusiveXor,

    EOpSubgroupGuardStop,

    EOpMinInvocations,
    EOpMaxInvocations,
    EOpAddInvocations,
    EOpMinInvocationsNonUniform,
    EOpMaxInvocationsNonUniform,
    EOpAddInvocationsNonUniform,
    EOpMinInvocationsInclusiveScan,
    EOpMaxInvocationsInclusiveScan,
    EOpAddInvocationsInclusiveScan,
    EOpMinInvocationsInclusiveScanNonUniform,
    EOpMaxInvocationsInclusiveScanNonUniform,
    EOpAddInvocationsInclusiveScanNonUniform,
    EOpMinInvocationsExclusiveScan,
    EOpMaxInvocationsExclusiveScan,
    EOpAddInvocationsExclusiveScan,
    EOpMinInvocationsExclusiveScanNonUniform,
    EOpMaxInvocationsExclusiveScanNonUniform,
    EOpAddInvocationsExclusiveScanNonUniform,
    EOpSwizzleInvocations,
    EOpSwizzleInvocationsMasked,
    EOpWriteInvocation,
    EOpMbcnt,

    EOpCubeFaceIndex,
    EOpCubeFaceCoord,
    EOpTime,

    EOpAtomicAdd,
    EOpAtomicSubtract,
    EOpAtomicMin,
    EOpAtomicMax,
    EOpAtomicAnd,
    EOpAtomicOr,
    EOpAtomicXor,
    EOpAtomicExchange,
    EOpAtomicCompSwap,
    EOpAtomicLoad,
    EOpAtomicStore,

    EOpAtomicCounterIncrement, // results in pre-increment value
    EOpAtomicCounterDecrement, // results in post-decrement value
    EOpAtomicCounter,
    EOpAtomicCounterAdd,
    EOpAtomicCounterSubtract,
    EOpAtomicCounterMin,
    EOpAtomicCounterMax,
    EOpAtomicCounterAnd,
    EOpAtomicCounterOr,
    EOpAtomicCounterXor,
    EOpAtomicCounterExchange,
    EOpAtomicCounterCompSwap,

    EOpAny,
    EOpAll,

    EOpCooperativeMatrixLoad,
    EOpCooperativeMatrixStore,
    EOpCooperativeMatrixMulAdd,

    EOpBeginInvocationInterlock, // Fragment only
    EOpEndInvocationInterlock, // Fragment only

    EOpIsHelperInvocation,

    EOpDebugPrintf,

    //
    // Branch
    //

    EOpKill,                // Fragment only
    EOpTerminateInvocation, // Fragment only
    EOpDemote,              // Fragment only
    EOpTerminateRayKHR,         // Any-hit only
    EOpIgnoreIntersectionKHR,   // Any-hit only
    EOpReturn,
    EOpBreak,
    EOpContinue,
    EOpCase,
    EOpDefault,

    //
    // Constructors
    //

    EOpConstructGuardStart,
    EOpConstructInt,          // these first scalar forms also identify what implicit conversion is needed
    EOpConstructUint,
    EOpConstructInt8,
    EOpConstructUint8,
    EOpConstructInt16,
    EOpConstructUint16,
    EOpConstructInt64,
    EOpConstructUint64,
    EOpConstructBool,
    EOpConstructFloat,
    EOpConstructDouble,
    // Keep vector and matrix constructors in a consistent relative order for
    // TParseContext::constructBuiltIn, which converts between 8/16/32 bit
    // vector constructors
    EOpConstructVec2,
    EOpConstructVec3,
    EOpConstructVec4,
    EOpConstructMat2x2,
    EOpConstructMat2x3,
    EOpConstructMat2x4,
    EOpConstructMat3x2,
    EOpConstructMat3x3,
    EOpConstructMat3x4,
    EOpConstructMat4x2,
    EOpConstructMat4x3,
    EOpConstructMat4x4,
    EOpConstructDVec2,
    EOpConstructDVec3,
    EOpConstructDVec4,
    EOpConstructBVec2,
    EOpConstructBVec3,
    EOpConstructBVec4,
    EOpConstructI8Vec2,
    EOpConstructI8Vec3,
    EOpConstructI8Vec4,
    EOpConstructU8Vec2,
    EOpConstructU8Vec3,
    EOpConstructU8Vec4,
    EOpConstructI16Vec2,
    EOpConstructI16Vec3,
    EOpConstructI16Vec4,
    EOpConstructU16Vec2,
    EOpConstructU16Vec3,
    EOpConstructU16Vec4,
    EOpConstructIVec2,
    EOpConstructIVec3,
    EOpConstructIVec4,
    EOpConstructUVec2,
    EOpConstructUVec3,
    EOpConstructUVec4,
    EOpConstructI64Vec2,
    EOpConstructI64Vec3,
    EOpConstructI64Vec4,
    EOpConstructU64Vec2,
    EOpConstructU64Vec3,
    EOpConstructU64Vec4,
    EOpConstructDMat2x2,
    EOpConstructDMat2x3,
    EOpConstructDMat2x4,
    EOpConstructDMat3x2,
    EOpConstructDMat3x3,
    EOpConstructDMat3x4,
    EOpConstructDMat4x2,
    EOpConstructDMat4x3,
    EOpConstructDMat4x4,
    EOpConstructIMat2x2,
    EOpConstructIMat2x3,
    EOpConstructIMat2x4,
    EOpConstructIMat3x2,
    EOpConstructIMat3x3,
    EOpConstructIMat3x4,
    EOpConstructIMat4x2,
    EOpConstructIMat4x3,
    EOpConstructIMat4x4,
    EOpConstructUMat2x2,
    EOpConstructUMat2x3,
    EOpConstructUMat2x4,
    EOpConstructUMat3x2,
    EOpConstructUMat3x3,
    EOpConstructUMat3x4,
    EOpConstructUMat4x2,
    EOpConstructUMat4x3,
    EOpConstructUMat4x4,
    EOpConstructBMat2x2,
    EOpConstructBMat2x3,
    EOpConstructBMat2x4,
    EOpConstructBMat3x2,
    EOpConstructBMat3x3,
    EOpConstructBMat3x4,
    EOpConstructBMat4x2,
    EOpConstructBMat4x3,
    EOpConstructBMat4x4,
    EOpConstructFloat16,
    EOpConstructF16Vec2,
    EOpConstructF16Vec3,
    EOpConstructF16Vec4,
    EOpConstructF16Mat2x2,
    EOpConstructF16Mat2x3,
    EOpConstructF16Mat2x4,
    EOpConstructF16Mat3x2,
    EOpConstructF16Mat3x3,
    EOpConstructF16Mat3x4,
    EOpConstructF16Mat4x2,
    EOpConstructF16Mat4x3,
    EOpConstructF16Mat4x4,
    EOpConstructStruct,
    EOpConstructTextureSampler,
    EOpConstructNonuniform,     // expected to be transformed away, not present in final AST
    EOpConstructReference,
    EOpConstructCooperativeMatrix,
    EOpConstructAccStruct,
    EOpConstructGuardEnd,

    //
    // moves
    //

    EOpAssign,
    EOpAddAssign,
    EOpSubAssign,
    EOpMulAssign,
    EOpVectorTimesMatrixAssign,
    EOpVectorTimesScalarAssign,
    EOpMatrixTimesScalarAssign,
    EOpMatrixTimesMatrixAssign,
    EOpDivAssign,
    EOpModAssign,
    EOpAndAssign,
    EOpInclusiveOrAssign,
    EOpExclusiveOrAssign,
    EOpLeftShiftAssign,
    EOpRightShiftAssign,

    //
    // Array operators
    //

    // Can apply to arrays, vectors, or matrices.
    // Can be decomposed to a constant at compile time, but this does not always happen,
    // due to link-time effects. So, consumer can expect either a link-time sized or
    // run-time sized array.
    EOpArrayLength,

    //
    // Image operations
    //

    EOpImageGuardBegin,

    EOpImageQuerySize,
    EOpImageQuerySamples,
    EOpImageLoad,
    EOpImageStore,
    EOpImageLoadLod,
    EOpImageStoreLod,
    EOpImageAtomicAdd,
    EOpImageAtomicMin,
    EOpImageAtomicMax,
    EOpImageAtomicAnd,
    EOpImageAtomicOr,
    EOpImageAtomicXor,
    EOpImageAtomicExchange,
    EOpImageAtomicCompSwap,
    EOpImageAtomicLoad,
    EOpImageAtomicStore,

    EOpSubpassLoad,
    EOpSubpassLoadMS,
    EOpSparseImageLoad,
    EOpSparseImageLoadLod,

    EOpImageGuardEnd,

    //
    // Texture operations
    //

    EOpTextureGuardBegin,

    EOpTextureQuerySize,
    EOpTextureQueryLod,
    EOpTextureQueryLevels,
    EOpTextureQuerySamples,

    EOpSamplingGuardBegin,

    EOpTexture,
    EOpTextureProj,
    EOpTextureLod,
    EOpTextureOffset,
    EOpTextureFetch,
    EOpTextureFetchOffset,
    EOpTextureProjOffset,
    EOpTextureLodOffset,
    EOpTextureProjLod,
    EOpTextureProjLodOffset,
    EOpTextureGrad,
    EOpTextureGradOffset,
    EOpTextureProjGrad,
    EOpTextureProjGradOffset,
    EOpTextureGather,
    EOpTextureGatherOffset,
    EOpTextureGatherOffsets,
    EOpTextureClamp,
    EOpTextureOffsetClamp,
    EOpTextureGradClamp,
    EOpTextureGradOffsetClamp,
    EOpTextureGatherLod,
    EOpTextureGatherLodOffset,
    EOpTextureGatherLodOffsets,
    EOpFragmentMaskFetch,
    EOpFragmentFetch,

    EOpSparseTextureGuardBegin,

    EOpSparseTexture,
    EOpSparseTextureLod,
    EOpSparseTextureOffset,
    EOpSparseTextureFetch,
    EOpSparseTextureFetchOffset,
    EOpSparseTextureLodOffset,
    EOpSparseTextureGrad,
    EOpSparseTextureGradOffset,
    EOpSparseTextureGather,
    EOpSparseTextureGatherOffset,
    EOpSparseTextureGatherOffsets,
    EOpSparseTexelsResident,
    EOpSparseTextureClamp,
    EOpSparseTextureOffsetClamp,
    EOpSparseTextureGradClamp,
    EOpSparseTextureGradOffsetClamp,
    EOpSparseTextureGatherLod,
    EOpSparseTextureGatherLodOffset,
    EOpSparseTextureGatherLodOffsets,

    EOpSparseTextureGuardEnd,

    EOpImageFootprintGuardBegin,
    EOpImageSampleFootprintNV,
    EOpImageSampleFootprintClampNV,
    EOpImageSampleFootprintLodNV,
    EOpImageSampleFootprintGradNV,
    EOpImageSampleFootprintGradClampNV,
    EOpImageFootprintGuardEnd,
    EOpSamplingGuardEnd,
    EOpTextureGuardEnd,

    //
    // Integer operations
    //

    EOpAddCarry,
    EOpSubBorrow,
    EOpUMulExtended,
    EOpIMulExtended,
    EOpBitfieldExtract,
    EOpBitfieldInsert,
    EOpBitFieldReverse,
    EOpBitCount,
    EOpFindLSB,
    EOpFindMSB,

    EOpCountLeadingZeros,
    EOpCountTrailingZeros,
    EOpAbsDifference,
    EOpAddSaturate,
    EOpSubSaturate,
    EOpAverage,
    EOpAverageRounded,
    EOpMul32x16,

    EOpTraceNV,
    EOpTraceRayMotionNV,
    EOpTraceKHR,
    EOpReportIntersection,
    EOpIgnoreIntersectionNV,
    EOpTerminateRayNV,
    EOpExecuteCallableNV,
    EOpExecuteCallableKHR,
    EOpWritePackedPrimitiveIndices4x8NV,

    //
    // GL_EXT_ray_query operations
    //

    EOpRayQueryInitialize,
    EOpRayQueryTerminate,
    EOpRayQueryGenerateIntersection,
    EOpRayQueryConfirmIntersection,
    EOpRayQueryProceed,
    EOpRayQueryGetIntersectionType,
    EOpRayQueryGetRayTMin,
    EOpRayQueryGetRayFlags,
    EOpRayQueryGetIntersectionT,
    EOpRayQueryGetIntersectionInstanceCustomIndex,
    EOpRayQueryGetIntersectionInstanceId,
    EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset,
    EOpRayQueryGetIntersectionGeometryIndex,
    EOpRayQueryGetIntersectionPrimitiveIndex,
    EOpRayQueryGetIntersectionBarycentrics,
    EOpRayQueryGetIntersectionFrontFace,
    EOpRayQueryGetIntersectionCandidateAABBOpaque,
    EOpRayQueryGetIntersectionObjectRayDirection,
    EOpRayQueryGetIntersectionObjectRayOrigin,
    EOpRayQueryGetWorldRayDirection,
    EOpRayQueryGetWorldRayOrigin,
    EOpRayQueryGetIntersectionObjectToWorld,
    EOpRayQueryGetIntersectionWorldToObject,

    //
    // HLSL operations
    //

    EOpClip,                // discard if input value < 0
    EOpIsFinite,
    EOpLog10,               // base 10 log
    EOpRcp,                 // 1/x
    EOpSaturate,            // clamp from 0 to 1
    EOpSinCos,              // sin and cos in out parameters
    EOpGenMul,              // mul(x,y) on any of mat/vec/scalars
    EOpDst,                 // x = 1, y=src0.y * src1.y, z=src0.z, w=src1.w
    EOpInterlockedAdd,      // atomic ops, but uses [optional] out arg instead of return
    EOpInterlockedAnd,      // ...
    EOpInterlockedCompareExchange, // ...
    EOpInterlockedCompareStore,    // ...
    EOpInterlockedExchange, // ...
    EOpInterlockedMax,      // ...
    EOpInterlockedMin,      // ...
    EOpInterlockedOr,       // ...
    EOpInterlockedXor,      // ...
    EOpAllMemoryBarrierWithGroupSync,    // memory barriers without non-hlsl AST equivalents
    EOpDeviceMemoryBarrier,              // ...
    EOpDeviceMemoryBarrierWithGroupSync, // ...
    EOpWorkgroupMemoryBarrier,           // ...
    EOpWorkgroupMemoryBarrierWithGroupSync, // ...
    EOpEvaluateAttributeSnapped,         // InterpolateAtOffset with int position on 16x16 grid
    EOpF32tof16,                         // HLSL conversion: half of a PackHalf2x16
    EOpF16tof32,                         // HLSL conversion: half of an UnpackHalf2x16
    EOpLit,                              // HLSL lighting coefficient vector
    EOpTextureBias,                      // HLSL texture bias: will be lowered to EOpTexture
    EOpAsDouble,                         // slightly different from EOpUint64BitsToDouble
    EOpD3DCOLORtoUBYTE4,                 // convert and swizzle 4-component color to UBYTE4 range

    EOpMethodSample,                     // Texture object methods.  These are translated to existing
    EOpMethodSampleBias,                 // AST methods, and exist to represent HLSL semantics until that
    EOpMethodSampleCmp,                  // translation is performed.  See HlslParseContext::decomposeSampleMethods().
    EOpMethodSampleCmpLevelZero,         // ...
    EOpMethodSampleGrad,                 // ...
    EOpMethodSampleLevel,                // ...
    EOpMethodLoad,                       // ...
    EOpMethodGetDimensions,              // ...
    EOpMethodGetSamplePosition,          // ...
    EOpMethodGather,                     // ...
    EOpMethodCalculateLevelOfDetail,     // ...
    EOpMethodCalculateLevelOfDetailUnclamped,     // ...

    // Load already defined above for textures
    EOpMethodLoad2,                      // Structure buffer object methods.  These are translated to existing
    EOpMethodLoad3,                      // AST methods, and exist to represent HLSL semantics until that
    EOpMethodLoad4,                      // translation is performed.  See HlslParseContext::decomposeSampleMethods().
    EOpMethodStore,                      // ...
    EOpMethodStore2,                     // ...
    EOpMethodStore3,                     // ...
    EOpMethodStore4,                     // ...
    EOpMethodIncrementCounter,           // ...
    EOpMethodDecrementCounter,           // ...
    // EOpMethodAppend is defined for geo shaders below
    EOpMethodConsume,

    // SM5 texture methods
    EOpMethodGatherRed,                  // These are covered under the above EOpMethodSample comment about
    EOpMethodGatherGreen,                // translation to existing AST opcodes.  They exist temporarily
    EOpMethodGatherBlue,                 // because HLSL arguments are slightly different.
    EOpMethodGatherAlpha,                // ...
    EOpMethodGatherCmp,                  // ...
    EOpMethodGatherCmpRed,               // ...
    EOpMethodGatherCmpGreen,             // ...
    EOpMethodGatherCmpBlue,              // ...
    EOpMethodGatherCmpAlpha,             // ...

    // geometry methods
    EOpMethodAppend,                     // Geometry shader methods
    EOpMethodRestartStrip,               // ...

    // matrix
    EOpMatrixSwizzle,                    // select multiple matrix components (non-column)

    // SM6 wave ops
    EOpWaveGetLaneCount,                 // Will decompose to gl_SubgroupSize.
    EOpWaveGetLaneIndex,                 // Will decompose to gl_SubgroupInvocationID.
    EOpWaveActiveCountBits,              // Will decompose to subgroupBallotBitCount(subgroupBallot()).
    EOpWavePrefixCountBits,              // Will decompose to subgroupBallotInclusiveBitCount(subgroupBallot()).

    // Shader Clock Ops
    EOpReadClockSubgroupKHR,
    EOpReadClockDeviceKHR,
};

class TIntermTraverser;
class TIntermOperator;
class TIntermAggregate;
class TIntermUnary;
class TIntermBinary;
class TIntermConstantUnion;
class TIntermSelection;
class TIntermSwitch;
class TIntermBranch;
class TIntermTyped;
class TIntermMethod;
class TIntermSymbol;
class TIntermLoop;

} // end namespace glslang

//
// Base class for the tree nodes
//
// (Put outside the glslang namespace, as it's used as part of the external interface.)
//
class TIntermNode {
public:
    POOL_ALLOCATOR_NEW_DELETE(glslang::GetThreadPoolAllocator())

    TIntermNode() { loc.init(); }
    virtual const glslang::TSourceLoc& getLoc() const { return loc; }
    virtual void setLoc(const glslang::TSourceLoc& l) { loc = l; }
    virtual void traverse(glslang::TIntermTraverser*) = 0;
    virtual       glslang::TIntermTyped*         getAsTyped()               { return 0; }
    virtual       glslang::TIntermOperator*      getAsOperator()            { return 0; }
    virtual       glslang::TIntermConstantUnion* getAsConstantUnion()       { return 0; }
    virtual       glslang::TIntermAggregate*     getAsAggregate()           { return 0; }
    virtual       glslang::TIntermUnary*         getAsUnaryNode()           { return 0; }
    virtual       glslang::TIntermBinary*        getAsBinaryNode()          { return 0; }
    virtual       glslang::TIntermSelection*     getAsSelectionNode()       { return 0; }
    virtual       glslang::TIntermSwitch*        getAsSwitchNode()          { return 0; }
    virtual       glslang::TIntermMethod*        getAsMethodNode()          { return 0; }
    virtual       glslang::TIntermSymbol*        getAsSymbolNode()          { return 0; }
    virtual       glslang::TIntermBranch*        getAsBranchNode()          { return 0; }
    virtual       glslang::TIntermLoop*          getAsLoopNode()            { return 0; }

    virtual const glslang::TIntermTyped*         getAsTyped()         const { return 0; }
    virtual const glslang::TIntermOperator*      getAsOperator()      const { return 0; }
    virtual const glslang::TIntermConstantUnion* getAsConstantUnion() const { return 0; }
    virtual const glslang::TIntermAggregate*     getAsAggregate()     const { return 0; }
    virtual const glslang::TIntermUnary*         getAsUnaryNode()     const { return 0; }
    virtual const glslang::TIntermBinary*        getAsBinaryNode()    const { return 0; }
    virtual const glslang::TIntermSelection*     getAsSelectionNode() const { return 0; }
    virtual const glslang::TIntermSwitch*        getAsSwitchNode()    const { return 0; }
    virtual const glslang::TIntermMethod*        getAsMethodNode()    const { return 0; }
    virtual const glslang::TIntermSymbol*        getAsSymbolNode()    const { return 0; }
    virtual const glslang::TIntermBranch*        getAsBranchNode()    const { return 0; }
    virtual const glslang::TIntermLoop*          getAsLoopNode()      const { return 0; }
    virtual ~TIntermNode() { }

protected:
    TIntermNode(const TIntermNode&);
    TIntermNode& operator=(const TIntermNode&);
    glslang::TSourceLoc loc;
};

namespace glslang {

//
// This is just to help yacc.
//
struct TIntermNodePair {
    TIntermNode* node1;
    TIntermNode* node2;
};

//
// Intermediate class for nodes that have a type.
//
class TIntermTyped : public TIntermNode {
public:
    TIntermTyped(const TType& t) { type.shallowCopy(t); }
    TIntermTyped(TBasicType basicType) { TType bt(basicType); type.shallowCopy(bt); }
    virtual       TIntermTyped* getAsTyped()       { return this; }
    virtual const TIntermTyped* getAsTyped() const { return this; }
    virtual void setType(const TType& t) { type.shallowCopy(t); }
    virtual const TType& getType() const { return type; }
    virtual TType& getWritableType() { return type; }

    virtual TBasicType getBasicType() const { return type.getBasicType(); }
    virtual TQualifier& getQualifier() { return type.getQualifier(); }
    virtual const TQualifier& getQualifier() const { return type.getQualifier(); }
    virtual TArraySizes* getArraySizes() { return type.getArraySizes(); }
    virtual const TArraySizes* getArraySizes() const { return type.getArraySizes(); }
    virtual void propagatePrecision(TPrecisionQualifier);
    virtual int getVectorSize() const { return type.getVectorSize(); }
    virtual int getMatrixCols() const { return type.getMatrixCols(); }
    virtual int getMatrixRows() const { return type.getMatrixRows(); }
    virtual bool isMatrix() const { return type.isMatrix(); }
    virtual bool isArray()  const { return type.isArray(); }
    virtual bool isVector() const { return type.isVector(); }
    virtual bool isScalar() const { return type.isScalar(); }
    virtual bool isStruct() const { return type.isStruct(); }
    virtual bool isFloatingDomain() const { return type.isFloatingDomain(); }
    virtual bool isIntegerDomain() const { return type.isIntegerDomain(); }
    bool isAtomic() const { return type.isAtomic(); }
    bool isReference() const { return type.isReference(); }
    TString getCompleteString() const { return type.getCompleteString(); }

protected:
    TIntermTyped& operator=(const TIntermTyped&);
    TType type;
};

//
// Handle for, do-while, and while loops.
//
class TIntermLoop : public TIntermNode {
public:
    TIntermLoop(TIntermNode* aBody, TIntermTyped* aTest, TIntermTyped* aTerminal, bool testFirst) :
        body(aBody),
        test(aTest),
        terminal(aTerminal),
        first(testFirst),
        unroll(false),
        dontUnroll(false),
        dependency(0),
        minIterations(0),
        maxIterations(iterationsInfinite),
        iterationMultiple(1),
        peelCount(0),
        partialCount(0)
    { }

    virtual       TIntermLoop* getAsLoopNode() { return this; }
    virtual const TIntermLoop* getAsLoopNode() const { return this; }
    virtual void traverse(TIntermTraverser*);
    TIntermNode*  getBody() const { return body; }
    TIntermTyped* getTest() const { return test; }
    TIntermTyped* getTerminal() const { return terminal; }
    bool testFirst() const { return first; }

    void setUnroll()     { unroll = true; }
    void setDontUnroll() {
        dontUnroll = true;
        peelCount = 0;
        partialCount = 0;
    }
    bool getUnroll()     const { return unroll; }
    bool getDontUnroll() const { return dontUnroll; }

    static const unsigned int dependencyInfinite = 0xFFFFFFFF;
    static const unsigned int iterationsInfinite = 0xFFFFFFFF;
    void setLoopDependency(int d) { dependency = d; }
    int getLoopDependency() const { return dependency; }

    void setMinIterations(unsigned int v) { minIterations = v; }
    unsigned int getMinIterations() const { return minIterations; }
    void setMaxIterations(unsigned int v) { maxIterations = v; }
    unsigned int getMaxIterations() const { return maxIterations; }
    void setIterationMultiple(unsigned int v) { iterationMultiple = v; }
    unsigned int getIterationMultiple() const { return iterationMultiple; }
    void setPeelCount(unsigned int v) {
        peelCount = v;
        dontUnroll = false;
    }
    unsigned int getPeelCount() const { return peelCount; }
    void setPartialCount(unsigned int v) {
        partialCount = v;
        dontUnroll = false;
    }
    unsigned int getPartialCount() const { return partialCount; }

protected:
    TIntermNode* body;       // code to loop over
    TIntermTyped* test;      // exit condition associated with loop, could be 0 for 'for' loops
    TIntermTyped* terminal;  // exists for for-loops
    bool first;              // true for while and for, not for do-while
    bool unroll;             // true if unroll requested
    bool dontUnroll;         // true if request to not unroll
    unsigned int dependency; // loop dependency hint; 0 means not set or unknown
    unsigned int minIterations;      // as per the SPIR-V specification
    unsigned int maxIterations;      // as per the SPIR-V specification
    unsigned int iterationMultiple;  // as per the SPIR-V specification
    unsigned int peelCount;          // as per the SPIR-V specification
    unsigned int partialCount;       // as per the SPIR-V specification
};

//
// Handle case, break, continue, return, and kill.
//
class TIntermBranch : public TIntermNode {
public:
    TIntermBranch(TOperator op, TIntermTyped* e) :
        flowOp(op),
        expression(e) { }
    virtual       TIntermBranch* getAsBranchNode()       { return this; }
    virtual const TIntermBranch* getAsBranchNode() const { return this; }
    virtual void traverse(TIntermTraverser*);
    TOperator getFlowOp() const { return flowOp; }
    TIntermTyped* getExpression() const { return expression; }
    void setExpression(TIntermTyped* pExpression) { expression = pExpression; }
    void updatePrecision(TPrecisionQualifier parentPrecision);
protected:
    TOperator flowOp;
    TIntermTyped* expression;
};

//
// Represent method names before seeing their calling signature
// or resolving them to operations.  Just an expression as the base object
// and a textural name.
//
class TIntermMethod : public TIntermTyped {
public:
    TIntermMethod(TIntermTyped* o, const TType& t, const TString& m) : TIntermTyped(t), object(o), method(m) { }
    virtual       TIntermMethod* getAsMethodNode()       { return this; }
    virtual const TIntermMethod* getAsMethodNode() const { return this; }
    virtual const TString& getMethodName() const { return method; }
    virtual TIntermTyped* getObject() const { return object; }
    virtual void traverse(TIntermTraverser*);
protected:
    TIntermTyped* object;
    TString method;
};

//
// Nodes that correspond to symbols or constants in the source code.
//
class TIntermSymbol : public TIntermTyped {
public:
    // if symbol is initialized as symbol(sym), the memory comes from the pool allocator of sym. If sym comes from
    // per process threadPoolAllocator, then it causes increased memory usage per compile
    // it is essential to use "symbol = sym" to assign to symbol
    TIntermSymbol(long long i, const TString& n, const TType& t)
        : TIntermTyped(t), id(i),
#ifndef GLSLANG_WEB
        flattenSubset(-1),
#endif
        constSubtree(nullptr)
          { name = n; }
    virtual long long getId() const { return id; }
    virtual void changeId(long long i) { id = i; }
    virtual const TString& getName() const { return name; }
    virtual void traverse(TIntermTraverser*);
    virtual       TIntermSymbol* getAsSymbolNode()       { return this; }
    virtual const TIntermSymbol* getAsSymbolNode() const { return this; }
    void setConstArray(const TConstUnionArray& c) { constArray = c; }
    const TConstUnionArray& getConstArray() const { return constArray; }
    void setConstSubtree(TIntermTyped* subtree) { constSubtree = subtree; }
    TIntermTyped* getConstSubtree() const { return constSubtree; }
#ifndef GLSLANG_WEB
    void setFlattenSubset(int subset) { flattenSubset = subset; }
    virtual const TString& getAccessName() const;

    int getFlattenSubset() const { return flattenSubset; } // -1 means full object
#endif

    // This is meant for cases where a node has already been constructed, and
    // later on, it becomes necessary to switch to a different symbol.
    virtual void switchId(long long newId) { id = newId; }

protected:
    long long id;                // the unique id of the symbol this node represents
#ifndef GLSLANG_WEB
    int flattenSubset;           // how deeply the flattened object rooted at id has been dereferenced
#endif
    TString name;                // the name of the symbol this node represents
    TConstUnionArray constArray; // if the symbol is a front-end compile-time constant, this is its value
    TIntermTyped* constSubtree;
};

class TIntermConstantUnion : public TIntermTyped {
public:
    TIntermConstantUnion(const TConstUnionArray& ua, const TType& t) : TIntermTyped(t), constArray(ua), literal(false) { }
    const TConstUnionArray& getConstArray() const { return constArray; }
    virtual       TIntermConstantUnion* getAsConstantUnion()       { return this; }
    virtual const TIntermConstantUnion* getAsConstantUnion() const { return this; }
    virtual void traverse(TIntermTraverser*);
    virtual TIntermTyped* fold(TOperator, const TIntermTyped*) const;
    virtual TIntermTyped* fold(TOperator, const TType&) const;
    void setLiteral() { literal = true; }
    void setExpression() { literal = false; }
    bool isLiteral() const { return literal; }

protected:
    TIntermConstantUnion& operator=(const TIntermConstantUnion&);

    const TConstUnionArray constArray;
    bool literal;  // true if node represents a literal in the source code
};

// Represent the independent aspects of a texturing TOperator
struct TCrackedTextureOp {
    bool query;
    bool proj;
    bool lod;
    bool fetch;
    bool offset;
    bool offsets;
    bool gather;
    bool grad;
    bool subpass;
    bool lodClamp;
    bool fragMask;
};

//
// Intermediate class for node types that hold operators.
//
class TIntermOperator : public TIntermTyped {
public:
    virtual       TIntermOperator* getAsOperator()       { return this; }
    virtual const TIntermOperator* getAsOperator() const { return this; }
    TOperator getOp() const { return op; }
    void setOp(TOperator newOp) { op = newOp; }
    bool modifiesState() const;
    bool isConstructor() const;
    bool isTexture()  const { return op > EOpTextureGuardBegin  && op < EOpTextureGuardEnd; }
    bool isSampling() const { return op > EOpSamplingGuardBegin && op < EOpSamplingGuardEnd; }
#ifdef GLSLANG_WEB
    bool isImage()          const { return false; }
    bool isSparseTexture()  const { return false; }
    bool isImageFootprint() const { return false; }
    bool isSparseImage()    const { return false; }
    bool isSubgroup()       const { return false; }
#else
    bool isImage()    const { return op > EOpImageGuardBegin    && op < EOpImageGuardEnd; }
    bool isSparseTexture() const { return op > EOpSparseTextureGuardBegin && op < EOpSparseTextureGuardEnd; }
    bool isImageFootprint() const { return op > EOpImageFootprintGuardBegin && op < EOpImageFootprintGuardEnd; }
    bool isSparseImage()   const { return op == EOpSparseImageLoad; }
    bool isSubgroup() const { return op > EOpSubgroupGuardStart && op < EOpSubgroupGuardStop; }
#endif

    void setOperationPrecision(TPrecisionQualifier p) { operationPrecision = p; }
    TPrecisionQualifier getOperationPrecision() const { return operationPrecision != EpqNone ?
                                                                                     operationPrecision :
                                                                                     type.getQualifier().precision; }
    TString getCompleteString() const
    {
        TString cs = type.getCompleteString();
        if (getOperationPrecision() != type.getQualifier().precision) {
            cs += ", operation at ";
            cs += GetPrecisionQualifierString(getOperationPrecision());
        }

        return cs;
    }

    // Crack the op into the individual dimensions of texturing operation.
    void crackTexture(TSampler sampler, TCrackedTextureOp& cracked) const
    {
        cracked.query = false;
        cracked.proj = false;
        cracked.lod = false;
        cracked.fetch = false;
        cracked.offset = false;
        cracked.offsets = false;
        cracked.gather = false;
        cracked.grad = false;
        cracked.subpass = false;
        cracked.lodClamp = false;
        cracked.fragMask = false;

        switch (op) {
        case EOpImageQuerySize:
        case EOpImageQuerySamples:
        case EOpTextureQuerySize:
        case EOpTextureQueryLod:
        case EOpTextureQueryLevels:
        case EOpTextureQuerySamples:
        case EOpSparseTexelsResident:
            cracked.query = true;
            break;
        case EOpTexture:
        case EOpSparseTexture:
            break;
        case EOpTextureProj:
            cracked.proj = true;
            break;
        case EOpTextureLod:
        case EOpSparseTextureLod:
            cracked.lod = true;
            break;
        case EOpTextureOffset:
        case EOpSparseTextureOffset:
            cracked.offset = true;
            break;
        case EOpTextureFetch:
        case EOpSparseTextureFetch:
            cracked.fetch = true;
            if (sampler.is1D() || (sampler.dim == Esd2D && ! sampler.isMultiSample()) || sampler.dim == Esd3D)
                cracked.lod = true;
            break;
        case EOpTextureFetchOffset:
        case EOpSparseTextureFetchOffset:
            cracked.fetch = true;
            cracked.offset = true;
            if (sampler.is1D() || (sampler.dim == Esd2D && ! sampler.isMultiSample()) || sampler.dim == Esd3D)
                cracked.lod = true;
            break;
        case EOpTextureProjOffset:
            cracked.offset = true;
            cracked.proj = true;
            break;
        case EOpTextureLodOffset:
        case EOpSparseTextureLodOffset:
            cracked.offset = true;
            cracked.lod = true;
            break;
        case EOpTextureProjLod:
            cracked.lod = true;
            cracked.proj = true;
            break;
        case EOpTextureProjLodOffset:
            cracked.offset = true;
            cracked.lod = true;
            cracked.proj = true;
            break;
        case EOpTextureGrad:
        case EOpSparseTextureGrad:
            cracked.grad = true;
            break;
        case EOpTextureGradOffset:
        case EOpSparseTextureGradOffset:
            cracked.grad = true;
            cracked.offset = true;
            break;
        case EOpTextureProjGrad:
            cracked.grad = true;
            cracked.proj = true;
            break;
        case EOpTextureProjGradOffset:
            cracked.grad = true;
            cracked.offset = true;
            cracked.proj = true;
            break;
#ifndef GLSLANG_WEB
        case EOpTextureClamp:
        case EOpSparseTextureClamp:
            cracked.lodClamp = true;
            break;
        case EOpTextureOffsetClamp:
        case EOpSparseTextureOffsetClamp:
            cracked.offset = true;
            cracked.lodClamp = true;
            break;
        case EOpTextureGradClamp:
        case EOpSparseTextureGradClamp:
            cracked.grad = true;
            cracked.lodClamp = true;
            break;
        case EOpTextureGradOffsetClamp:
        case EOpSparseTextureGradOffsetClamp:
            cracked.grad = true;
            cracked.offset = true;
            cracked.lodClamp = true;
            break;
        case EOpTextureGather:
        case EOpSparseTextureGather:
            cracked.gather = true;
            break;
        case EOpTextureGatherOffset:
        case EOpSparseTextureGatherOffset:
            cracked.gather = true;
            cracked.offset = true;
            break;
        case EOpTextureGatherOffsets:
        case EOpSparseTextureGatherOffsets:
            cracked.gather = true;
            cracked.offsets = true;
            break;
        case EOpTextureGatherLod:
        case EOpSparseTextureGatherLod:
            cracked.gather = true;
            cracked.lod    = true;
            break;
        case EOpTextureGatherLodOffset:
        case EOpSparseTextureGatherLodOffset:
            cracked.gather = true;
            cracked.offset = true;
            cracked.lod    = true;
            break;
        case EOpTextureGatherLodOffsets:
        case EOpSparseTextureGatherLodOffsets:
            cracked.gather  = true;
            cracked.offsets = true;
            cracked.lod     = true;
            break;
        case EOpImageLoadLod:
        case EOpImageStoreLod:
        case EOpSparseImageLoadLod:
            cracked.lod = true;
            break;
        case EOpFragmentMaskFetch:
            cracked.subpass = sampler.dim == EsdSubpass;
            cracked.fragMask = true;
            break;
        case EOpFragmentFetch:
            cracked.subpass = sampler.dim == EsdSubpass;
            cracked.fragMask = true;
            break;
        case EOpImageSampleFootprintNV:
            break;
        case EOpImageSampleFootprintClampNV:
            cracked.lodClamp = true;
            break;
        case EOpImageSampleFootprintLodNV:
            cracked.lod = true;
            break;
        case EOpImageSampleFootprintGradNV:
            cracked.grad = true;
            break;
        case EOpImageSampleFootprintGradClampNV:
            cracked.lodClamp = true;
            cracked.grad = true;
            break;
        case EOpSubpassLoad:
        case EOpSubpassLoadMS:
            cracked.subpass = true;
            break;
#endif
        default:
            break;
        }
    }

protected:
    TIntermOperator(TOperator o) : TIntermTyped(EbtFloat), op(o), operationPrecision(EpqNone) {}
    TIntermOperator(TOperator o, TType& t) : TIntermTyped(t), op(o), operationPrecision(EpqNone) {}
    TOperator op;
    // The result precision is in the inherited TType, and is usually meant to be both
    // the operation precision and the result precision. However, some more complex things,
    // like built-in function calls, distinguish between the two, in which case non-EqpNone
    // 'operationPrecision' overrides the result precision as far as operation precision
    // is concerned.
    TPrecisionQualifier operationPrecision;
};

//
// Nodes for all the basic binary math operators.
//
class TIntermBinary : public TIntermOperator {
public:
    TIntermBinary(TOperator o) : TIntermOperator(o) {}
    virtual void traverse(TIntermTraverser*);
    virtual void setLeft(TIntermTyped* n) { left = n; }
    virtual void setRight(TIntermTyped* n) { right = n; }
    virtual TIntermTyped* getLeft() const { return left; }
    virtual TIntermTyped* getRight() const { return right; }
    virtual       TIntermBinary* getAsBinaryNode()       { return this; }
    virtual const TIntermBinary* getAsBinaryNode() const { return this; }
    virtual void updatePrecision();
protected:
    TIntermTyped* left;
    TIntermTyped* right;
};

//
// Nodes for unary math operators.
//
class TIntermUnary : public TIntermOperator {
public:
    TIntermUnary(TOperator o, TType& t) : TIntermOperator(o, t), operand(0) {}
    TIntermUnary(TOperator o) : TIntermOperator(o), operand(0) {}
    virtual void traverse(TIntermTraverser*);
    virtual void setOperand(TIntermTyped* o) { operand = o; }
    virtual       TIntermTyped* getOperand() { return operand; }
    virtual const TIntermTyped* getOperand() const { return operand; }
    virtual       TIntermUnary* getAsUnaryNode()       { return this; }
    virtual const TIntermUnary* getAsUnaryNode() const { return this; }
    virtual void updatePrecision();
#ifndef GLSLANG_WEB
    void setSpirvInstruction(const TSpirvInstruction& inst) { spirvInst = inst; }
    const TSpirvInstruction& getSpirvInstruction() const { return spirvInst; }
#endif
protected:
    TIntermTyped* operand;
#ifndef GLSLANG_WEB
    TSpirvInstruction spirvInst;
#endif
};

typedef TVector<TIntermNode*> TIntermSequence;
typedef TVector<TStorageQualifier> TQualifierList;
//
// Nodes that operate on an arbitrary sized set of children.
//
class TIntermAggregate : public TIntermOperator {
public:
    TIntermAggregate() : TIntermOperator(EOpNull), userDefined(false), pragmaTable(nullptr) { }
    TIntermAggregate(TOperator o) : TIntermOperator(o), pragmaTable(nullptr) { }
    ~TIntermAggregate() { delete pragmaTable; }
    virtual       TIntermAggregate* getAsAggregate()       { return this; }
    virtual const TIntermAggregate* getAsAggregate() const { return this; }
    virtual void setOperator(TOperator o) { op = o; }
    virtual       TIntermSequence& getSequence()       { return sequence; }
    virtual const TIntermSequence& getSequence() const { return sequence; }
    virtual void setName(const TString& n) { name = n; }
    virtual const TString& getName() const { return name; }
    virtual void traverse(TIntermTraverser*);
    virtual void setUserDefined() { userDefined = true; }
    virtual bool isUserDefined() { return userDefined; }
    virtual TQualifierList& getQualifierList() { return qualifier; }
    virtual const TQualifierList& getQualifierList() const { return qualifier; }
    void setOptimize(bool o) { optimize = o; }
    void setDebug(bool d) { debug = d; }
    bool getOptimize() const { return optimize; }
    bool getDebug() const { return debug; }
    void setPragmaTable(const TPragmaTable& pTable);
    const TPragmaTable& getPragmaTable() const { return *pragmaTable; }
#ifndef GLSLANG_WEB
    void setSpirvInstruction(const TSpirvInstruction& inst) { spirvInst = inst; }
    const TSpirvInstruction& getSpirvInstruction() const { return spirvInst; }
#endif
protected:
    TIntermAggregate(const TIntermAggregate&); // disallow copy constructor
    TIntermAggregate& operator=(const TIntermAggregate&); // disallow assignment operator
    TIntermSequence sequence;
    TQualifierList qualifier;
    TString name;
    bool userDefined; // used for user defined function names
    bool optimize;
    bool debug;
    TPragmaTable* pragmaTable;
#ifndef GLSLANG_WEB
    TSpirvInstruction spirvInst;
#endif
};

//
// For if tests.
//
class TIntermSelection : public TIntermTyped {
public:
    TIntermSelection(TIntermTyped* cond, TIntermNode* trueB, TIntermNode* falseB) :
        TIntermTyped(EbtVoid), condition(cond), trueBlock(trueB), falseBlock(falseB),
        shortCircuit(true),
        flatten(false), dontFlatten(false) {}
    TIntermSelection(TIntermTyped* cond, TIntermNode* trueB, TIntermNode* falseB, const TType& type) :
        TIntermTyped(type), condition(cond), trueBlock(trueB), falseBlock(falseB),
        shortCircuit(true),
        flatten(false), dontFlatten(false) {}
    virtual void traverse(TIntermTraverser*);
    virtual TIntermTyped* getCondition() const { return condition; }
    virtual void setCondition(TIntermTyped* c) { condition = c; }
    virtual TIntermNode* getTrueBlock() const { return trueBlock; }
    virtual void setTrueBlock(TIntermTyped* tb) { trueBlock = tb; }
    virtual TIntermNode* getFalseBlock() const { return falseBlock; }
    virtual void setFalseBlock(TIntermTyped* fb) { falseBlock = fb; }
    virtual       TIntermSelection* getAsSelectionNode()       { return this; }
    virtual const TIntermSelection* getAsSelectionNode() const { return this; }

    void setNoShortCircuit() { shortCircuit = false; }
    bool getShortCircuit() const { return shortCircuit; }

    void setFlatten()     { flatten = true; }
    void setDontFlatten() { dontFlatten = true; }
    bool getFlatten()     const { return flatten; }
    bool getDontFlatten() const { return dontFlatten; }

protected:
    TIntermTyped* condition;
    TIntermNode* trueBlock;
    TIntermNode* falseBlock;
    bool shortCircuit; // normally all if-then-else and all GLSL ?: short-circuit, but HLSL ?: does not
    bool flatten;      // true if flatten requested
    bool dontFlatten;  // true if requested to not flatten
};

//
// For switch statements.  Designed use is that a switch will have sequence of nodes
// that are either case/default nodes or a *single* node that represents all the code
// in between (if any) consecutive case/defaults.  So, a traversal need only deal with
// 0 or 1 nodes per case/default statement.
//
class TIntermSwitch : public TIntermNode {
public:
    TIntermSwitch(TIntermTyped* cond, TIntermAggregate* b) : condition(cond), body(b),
        flatten(false), dontFlatten(false) {}
    virtual void traverse(TIntermTraverser*);
    virtual TIntermNode* getCondition() const { return condition; }
    virtual TIntermAggregate* getBody() const { return body; }
    virtual       TIntermSwitch* getAsSwitchNode()       { return this; }
    virtual const TIntermSwitch* getAsSwitchNode() const { return this; }

    void setFlatten()     { flatten = true; }
    void setDontFlatten() { dontFlatten = true; }
    bool getFlatten()     const { return flatten; }
    bool getDontFlatten() const { return dontFlatten; }

protected:
    TIntermTyped* condition;
    TIntermAggregate* body;
    bool flatten;     // true if flatten requested
    bool dontFlatten; // true if requested to not flatten
};

enum TVisit
{
    EvPreVisit,
    EvInVisit,
    EvPostVisit
};

//
// For traversing the tree.  User should derive from this,
// put their traversal specific data in it, and then pass
// it to a Traverse method.
//
// When using this, just fill in the methods for nodes you want visited.
// Return false from a pre-visit to skip visiting that node's subtree.
//
// Explicitly set postVisit to true if you want post visiting, otherwise,
// filled in methods will only be called at pre-visit time (before processing
// the subtree).  Similarly for inVisit for in-order visiting of nodes with
// multiple children.
//
// If you only want post-visits, explicitly turn off preVisit (and inVisit)
// and turn on postVisit.
//
// In general, for the visit*() methods, return true from interior nodes
// to have the traversal continue on to children.
//
// If you process children yourself, or don't want them processed, return false.
//
class TIntermTraverser {
public:
    POOL_ALLOCATOR_NEW_DELETE(glslang::GetThreadPoolAllocator())
    TIntermTraverser(bool preVisit = true, bool inVisit = false, bool postVisit = false, bool rightToLeft = false) :
            preVisit(preVisit),
            inVisit(inVisit),
            postVisit(postVisit),
            rightToLeft(rightToLeft),
            depth(0),
            maxDepth(0) { }
    virtual ~TIntermTraverser() { }

    virtual void visitSymbol(TIntermSymbol*)               { }
    virtual void visitConstantUnion(TIntermConstantUnion*) { }
    virtual bool visitBinary(TVisit, TIntermBinary*)       { return true; }
    virtual bool visitUnary(TVisit, TIntermUnary*)         { return true; }
    virtual bool visitSelection(TVisit, TIntermSelection*) { return true; }
    virtual bool visitAggregate(TVisit, TIntermAggregate*) { return true; }
    virtual bool visitLoop(TVisit, TIntermLoop*)           { return true; }
    virtual bool visitBranch(TVisit, TIntermBranch*)       { return true; }
    virtual bool visitSwitch(TVisit, TIntermSwitch*)       { return true; }

    int getMaxDepth() const { return maxDepth; }

    void incrementDepth(TIntermNode *current)
    {
        depth++;
        maxDepth = (std::max)(maxDepth, depth);
        path.push_back(current);
    }

    void decrementDepth()
    {
        depth--;
        path.pop_back();
    }

    TIntermNode *getParentNode()
    {
        return path.size() == 0 ? NULL : path.back();
    }

    const bool preVisit;
    const bool inVisit;
    const bool postVisit;
    const bool rightToLeft;

protected:
    TIntermTraverser& operator=(TIntermTraverser&);

    int depth;
    int maxDepth;

    // All the nodes from root to the current node's parent during traversing.
    TVector<TIntermNode *> path;
};

// KHR_vulkan_glsl says "Two arrays sized with specialization constants are the same type only if
// sized with the same symbol, involving no operations"
inline bool SameSpecializationConstants(TIntermTyped* node1, TIntermTyped* node2)
{
    return node1->getAsSymbolNode() && node2->getAsSymbolNode() &&
           node1->getAsSymbolNode()->getId() == node2->getAsSymbolNode()->getId();
}

} // end namespace glslang

#endif // __INTERMEDIATE_H
