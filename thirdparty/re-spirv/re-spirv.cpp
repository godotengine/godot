//
// re-spirv
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

#include "re-spirv.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#define SPV_ENABLE_UTILITY_CODE

#include "spirv/unified1/spirv.h"

// Enables more extensive output on errors.
#define RESPV_VERBOSE_ERRORS 0

namespace respv {
    // Common.
    
    static bool SpvIsSupported(SpvOp pOpCode) {
        switch (pOpCode) {
        case SpvOpUndef:
        case SpvOpSource:
        case SpvOpName:
        case SpvOpMemberName:
        case SpvOpExtension:
        case SpvOpExtInstImport:
        case SpvOpExtInst:
        case SpvOpMemoryModel:
        case SpvOpEntryPoint:
        case SpvOpExecutionMode:
        case SpvOpCapability:
        case SpvOpTypeVoid:
        case SpvOpTypeBool:
        case SpvOpTypeInt:
        case SpvOpTypeFloat:
        case SpvOpTypeVector:
        case SpvOpTypeMatrix:
        case SpvOpTypeImage:
        case SpvOpTypeSampler:
        case SpvOpTypeSampledImage:
        case SpvOpTypeArray:
        case SpvOpTypeRuntimeArray:
        case SpvOpTypeStruct:
        case SpvOpTypePointer:
        case SpvOpTypeFunction:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpConstant:
        case SpvOpConstantComposite:
        case SpvOpConstantNull:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpSpecConstantOp: 
        case SpvOpFunction:
        case SpvOpFunctionParameter:
        case SpvOpFunctionEnd:
        case SpvOpFunctionCall:
        case SpvOpVariable:
        case SpvOpImageTexelPointer:
        case SpvOpLoad:
        case SpvOpStore:
        case SpvOpAccessChain:
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpVectorShuffle:
        case SpvOpCompositeConstruct:
        case SpvOpCompositeExtract:
        case SpvOpCompositeInsert:
        case SpvOpCopyObject:
        case SpvOpTranspose:
        case SpvOpSampledImage:
        case SpvOpImageSampleImplicitLod:
        case SpvOpImageSampleExplicitLod:
        case SpvOpImageSampleDrefImplicitLod:
        case SpvOpImageSampleDrefExplicitLod:
        case SpvOpImageSampleProjImplicitLod:
        case SpvOpImageSampleProjExplicitLod:
        case SpvOpImageSampleProjDrefImplicitLod:
        case SpvOpImageSampleProjDrefExplicitLod:
        case SpvOpImageFetch:
        case SpvOpImageGather:
        case SpvOpImageDrefGather:
        case SpvOpImageRead:
        case SpvOpImageWrite:
        case SpvOpImage:
        case SpvOpImageQueryFormat:
        case SpvOpImageQueryOrder:
        case SpvOpImageQuerySizeLod:
        case SpvOpImageQuerySize:
        case SpvOpImageQueryLod:
        case SpvOpImageQueryLevels:
        case SpvOpImageQuerySamples:
        case SpvOpConvertFToU:
        case SpvOpConvertFToS:
        case SpvOpConvertSToF:
        case SpvOpConvertUToF:
        case SpvOpUConvert:
        case SpvOpSConvert:
        case SpvOpFConvert:
        case SpvOpBitcast:
        case SpvOpSNegate:
        case SpvOpFNegate:
        case SpvOpIAdd:
        case SpvOpFAdd:
        case SpvOpISub:
        case SpvOpFSub:
        case SpvOpIMul:
        case SpvOpFMul:
        case SpvOpUDiv:
        case SpvOpSDiv:
        case SpvOpFDiv:
        case SpvOpUMod:
        case SpvOpSRem:
        case SpvOpSMod:
        case SpvOpFRem:
        case SpvOpFMod:
        case SpvOpVectorTimesScalar:
        case SpvOpMatrixTimesScalar:
        case SpvOpVectorTimesMatrix:
        case SpvOpMatrixTimesVector:
        case SpvOpMatrixTimesMatrix:
        case SpvOpOuterProduct:
        case SpvOpDot:
        case SpvOpIAddCarry:
        case SpvOpISubBorrow:
        case SpvOpUMulExtended:
        case SpvOpSMulExtended:
        case SpvOpAny:
        case SpvOpAll:
        case SpvOpIsNan:
        case SpvOpIsInf:
        case SpvOpIsFinite:
        case SpvOpIsNormal:
        case SpvOpLogicalEqual:
        case SpvOpLogicalNotEqual:
        case SpvOpLogicalOr:
        case SpvOpLogicalAnd:
        case SpvOpLogicalNot:
        case SpvOpSelect:
        case SpvOpIEqual:
        case SpvOpINotEqual:
        case SpvOpUGreaterThan:
        case SpvOpSGreaterThan:
        case SpvOpUGreaterThanEqual:
        case SpvOpSGreaterThanEqual:
        case SpvOpULessThan:
        case SpvOpSLessThan:
        case SpvOpULessThanEqual:
        case SpvOpSLessThanEqual:
        case SpvOpFOrdEqual:
        case SpvOpFUnordEqual:
        case SpvOpFOrdNotEqual:
        case SpvOpFUnordNotEqual:
        case SpvOpFOrdLessThan:
        case SpvOpFUnordLessThan:
        case SpvOpFOrdGreaterThan:
        case SpvOpFUnordGreaterThan:
        case SpvOpFOrdLessThanEqual:
        case SpvOpFUnordLessThanEqual:
        case SpvOpFOrdGreaterThanEqual:
        case SpvOpFUnordGreaterThanEqual:
        case SpvOpShiftRightLogical:
        case SpvOpShiftRightArithmetic:
        case SpvOpShiftLeftLogical:
        case SpvOpBitwiseOr:
        case SpvOpBitwiseXor:
        case SpvOpBitwiseAnd:
        case SpvOpNot:
        case SpvOpBitFieldInsert:
        case SpvOpBitFieldSExtract:
        case SpvOpBitFieldUExtract:
        case SpvOpBitReverse:
        case SpvOpBitCount:
        case SpvOpDPdx:
        case SpvOpDPdy:
        case SpvOpFwidth:
        case SpvOpControlBarrier:
        case SpvOpMemoryBarrier:
        case SpvOpAtomicLoad:
        case SpvOpAtomicStore:
        case SpvOpAtomicExchange:
        case SpvOpAtomicCompareExchange:
        case SpvOpAtomicCompareExchangeWeak:
        case SpvOpAtomicIIncrement:
        case SpvOpAtomicIDecrement:
        case SpvOpAtomicIAdd:
        case SpvOpAtomicISub:
        case SpvOpAtomicSMin:
        case SpvOpAtomicUMin:
        case SpvOpAtomicSMax:
        case SpvOpAtomicUMax:
        case SpvOpAtomicAnd:
        case SpvOpAtomicOr:
        case SpvOpAtomicXor:
        case SpvOpPhi:
        case SpvOpLoopMerge:
        case SpvOpSelectionMerge:
        case SpvOpLabel:
        case SpvOpBranch:
        case SpvOpBranchConditional:
        case SpvOpSwitch:
        case SpvOpKill:
        case SpvOpReturn:
        case SpvOpReturnValue:
        case SpvOpUnreachable:
        case SpvOpGroupNonUniformElect:
        case SpvOpGroupNonUniformAll:
        case SpvOpGroupNonUniformAny:
        case SpvOpGroupNonUniformAllEqual:
        case SpvOpGroupNonUniformBroadcast:
        case SpvOpGroupNonUniformBroadcastFirst:
        case SpvOpGroupNonUniformBallot:
        case SpvOpGroupNonUniformInverseBallot:
        case SpvOpGroupNonUniformBallotBitExtract:
        case SpvOpGroupNonUniformBallotBitCount:
        case SpvOpGroupNonUniformBallotFindLSB:
        case SpvOpGroupNonUniformBallotFindMSB:
        case SpvOpGroupNonUniformShuffle:
        case SpvOpGroupNonUniformShuffleXor:
        case SpvOpGroupNonUniformShuffleUp:
        case SpvOpGroupNonUniformShuffleDown:
        case SpvOpGroupNonUniformIAdd:
        case SpvOpGroupNonUniformFAdd:
        case SpvOpGroupNonUniformIMul:
        case SpvOpGroupNonUniformFMul:
        case SpvOpGroupNonUniformSMin:
        case SpvOpGroupNonUniformUMin:
        case SpvOpGroupNonUniformFMin:
        case SpvOpGroupNonUniformSMax:
        case SpvOpGroupNonUniformUMax:
        case SpvOpGroupNonUniformFMax:
        case SpvOpGroupNonUniformBitwiseAnd:
        case SpvOpGroupNonUniformBitwiseOr:
        case SpvOpGroupNonUniformBitwiseXor:
        case SpvOpGroupNonUniformLogicalAnd:
        case SpvOpGroupNonUniformLogicalOr:
        case SpvOpGroupNonUniformLogicalXor:
        case SpvOpGroupNonUniformQuadBroadcast:
        case SpvOpGroupNonUniformQuadSwap:
        case SpvOpCopyLogical:
            return true;
        default:
            return false;
        }
    }

    static bool SpvIsIgnored(SpvOp pOpCode) {
        switch (pOpCode) {
        case SpvOpSource:
        case SpvOpName:
        case SpvOpMemberName:
            return true;
        default:
            return false;
        }
    }

    static bool SpvHasOperands(SpvOp pOpCode, uint32_t &rOperandWordStart, uint32_t &rOperandWordCount, uint32_t &rOperandWordStride, uint32_t &rOperandWordSkip, bool &rOperandWordSkipString, bool pIncludePhi) {
        switch (pOpCode) {
        case SpvOpExecutionMode:
        case SpvOpBranchConditional:
        case SpvOpSwitch:
        case SpvOpReturnValue:
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
            rOperandWordStart = 1;
            rOperandWordCount = 1;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpStore:
        case SpvOpMemoryBarrier:
            rOperandWordStart = 1;
            rOperandWordCount = 2;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpControlBarrier:
            rOperandWordStart = 1;
            rOperandWordCount = 3;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpTypeVector:
        case SpvOpTypeMatrix:
        case SpvOpTypeImage:
        case SpvOpTypeSampledImage:
        case SpvOpTypeRuntimeArray:
            rOperandWordStart = 2;
            rOperandWordCount = 1;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpTypeArray:
            rOperandWordStart = 2;
            rOperandWordCount = 2;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpTypeStruct:
        case SpvOpTypeFunction:
            rOperandWordStart = 2;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpEntryPoint:
            rOperandWordStart = 2;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 1;
            rOperandWordSkipString = true;
            return true;
        case SpvOpTypePointer:
        case SpvOpLoad:
        case SpvOpCompositeExtract:
        case SpvOpCopyObject:
        case SpvOpTranspose:
        case SpvOpImage:
        case SpvOpImageQueryFormat:
        case SpvOpImageQueryOrder:
        case SpvOpImageQuerySize:
        case SpvOpImageQueryLevels:
        case SpvOpImageQuerySamples:
        case SpvOpConvertFToU:
        case SpvOpConvertFToS:
        case SpvOpConvertSToF:
        case SpvOpConvertUToF:
        case SpvOpUConvert:
        case SpvOpSConvert:
        case SpvOpFConvert:
        case SpvOpBitcast:
        case SpvOpSNegate:
        case SpvOpFNegate:
        case SpvOpAny:
        case SpvOpAll:
        case SpvOpIsNan:
        case SpvOpIsInf:
        case SpvOpIsFinite:
        case SpvOpIsNormal:
        case SpvOpLogicalNot:
        case SpvOpNot:
        case SpvOpBitReverse:
        case SpvOpBitCount:
        case SpvOpDPdx:
        case SpvOpDPdy:
        case SpvOpFwidth:
        case SpvOpGroupNonUniformElect:
        case SpvOpCopyLogical:
            rOperandWordStart = 3;
            rOperandWordCount = 1;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpVectorShuffle:
        case SpvOpCompositeInsert:
        case SpvOpSampledImage:
        case SpvOpImageQuerySizeLod:
        case SpvOpImageQueryLod:
        case SpvOpIAdd:
        case SpvOpFAdd:
        case SpvOpISub:
        case SpvOpFSub:
        case SpvOpIMul:
        case SpvOpFMul:
        case SpvOpUDiv:
        case SpvOpSDiv:
        case SpvOpFDiv:
        case SpvOpUMod:
        case SpvOpSRem:
        case SpvOpSMod:
        case SpvOpFRem:
        case SpvOpFMod:
        case SpvOpVectorTimesScalar:
        case SpvOpMatrixTimesScalar:
        case SpvOpVectorTimesMatrix:
        case SpvOpMatrixTimesVector:
        case SpvOpMatrixTimesMatrix:
        case SpvOpOuterProduct:
        case SpvOpDot:
        case SpvOpIAddCarry:
        case SpvOpISubBorrow:
        case SpvOpUMulExtended:
        case SpvOpSMulExtended:
        case SpvOpLogicalEqual:
        case SpvOpLogicalNotEqual:
        case SpvOpLogicalOr:
        case SpvOpLogicalAnd:
        case SpvOpIEqual:
        case SpvOpINotEqual:
        case SpvOpUGreaterThan:
        case SpvOpSGreaterThan:
        case SpvOpUGreaterThanEqual:
        case SpvOpSGreaterThanEqual:
        case SpvOpULessThan:
        case SpvOpSLessThan:
        case SpvOpULessThanEqual:
        case SpvOpSLessThanEqual:
        case SpvOpFOrdEqual:
        case SpvOpFUnordEqual:
        case SpvOpFOrdNotEqual:
        case SpvOpFUnordNotEqual:
        case SpvOpFOrdLessThan:
        case SpvOpFUnordLessThan:
        case SpvOpFOrdGreaterThan:
        case SpvOpFUnordGreaterThan:
        case SpvOpFOrdLessThanEqual:
        case SpvOpFUnordLessThanEqual:
        case SpvOpFOrdGreaterThanEqual:
        case SpvOpFUnordGreaterThanEqual:
        case SpvOpShiftRightLogical:
        case SpvOpShiftRightArithmetic:
        case SpvOpShiftLeftLogical:
        case SpvOpBitwiseOr:
        case SpvOpBitwiseAnd:
        case SpvOpBitwiseXor:
        case SpvOpGroupNonUniformAll:
        case SpvOpGroupNonUniformAny:
        case SpvOpGroupNonUniformAllEqual:
        case SpvOpGroupNonUniformBroadcastFirst:
        case SpvOpGroupNonUniformBallot:
        case SpvOpGroupNonUniformInverseBallot:
        case SpvOpGroupNonUniformBallotFindLSB:
        case SpvOpGroupNonUniformBallotFindMSB:
            rOperandWordStart = 3;
            rOperandWordCount = 2;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpImageTexelPointer:
        case SpvOpSelect:
        case SpvOpBitFieldSExtract:
        case SpvOpBitFieldUExtract:
        case SpvOpAtomicLoad:
        case SpvOpAtomicIIncrement:
        case SpvOpAtomicIDecrement:
        case SpvOpGroupNonUniformBroadcast:
        case SpvOpGroupNonUniformBallotBitExtract:
        case SpvOpGroupNonUniformShuffle:
        case SpvOpGroupNonUniformShuffleXor:
        case SpvOpGroupNonUniformShuffleUp:
        case SpvOpGroupNonUniformShuffleDown:
        case SpvOpGroupNonUniformQuadBroadcast:
        case SpvOpGroupNonUniformQuadSwap:
            rOperandWordStart = 3;
            rOperandWordCount = 3;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpGroupNonUniformBallotBitCount:
            rOperandWordStart = 3;
            rOperandWordCount = 3;
            rOperandWordStride = 1;
            rOperandWordSkip = 1;
            rOperandWordSkipString = false;
            return true;
        case SpvOpAtomicStore:
            rOperandWordStart = 1;
            rOperandWordCount = 4;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpBitFieldInsert:
        case SpvOpAtomicExchange:
        case SpvOpAtomicIAdd:
        case SpvOpAtomicISub:
        case SpvOpAtomicSMin:
        case SpvOpAtomicUMin:
        case SpvOpAtomicSMax:
        case SpvOpAtomicUMax:
        case SpvOpAtomicAnd:
        case SpvOpAtomicOr:
        case SpvOpAtomicXor:
            rOperandWordStart = 3;
            rOperandWordCount = 4;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpAtomicCompareExchange:
        case SpvOpAtomicCompareExchangeWeak:
            rOperandWordStart = 3;
            rOperandWordCount = 6;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpConstantComposite:
        case SpvOpFunctionCall:
        case SpvOpAccessChain:
        case SpvOpCompositeConstruct:
            rOperandWordStart = 3;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpSpecConstantOp:
            rOperandWordStart = 3;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 0;
            rOperandWordSkipString = false;
            return true;
        case SpvOpExtInst:
        case SpvOpGroupNonUniformIAdd:
        case SpvOpGroupNonUniformFAdd:
        case SpvOpGroupNonUniformIMul:
        case SpvOpGroupNonUniformFMul:
        case SpvOpGroupNonUniformSMin:
        case SpvOpGroupNonUniformUMin:
        case SpvOpGroupNonUniformFMin:
        case SpvOpGroupNonUniformSMax:
        case SpvOpGroupNonUniformUMax:
        case SpvOpGroupNonUniformFMax:
        case SpvOpGroupNonUniformBitwiseAnd:
        case SpvOpGroupNonUniformBitwiseOr:
        case SpvOpGroupNonUniformBitwiseXor:
        case SpvOpGroupNonUniformLogicalAnd:
        case SpvOpGroupNonUniformLogicalOr:
        case SpvOpGroupNonUniformLogicalXor:
            rOperandWordStart = 3;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 1;
            rOperandWordSkipString = false;
            return true;
        case SpvOpImageWrite:
            rOperandWordStart = 1;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 3;
            rOperandWordSkipString = false;
            return true;
        case SpvOpImageSampleImplicitLod:
        case SpvOpImageSampleExplicitLod:
        case SpvOpImageSampleProjImplicitLod:
        case SpvOpImageSampleProjExplicitLod:
        case SpvOpImageFetch:
        case SpvOpImageRead:
            rOperandWordStart = 3;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 2;
            rOperandWordSkipString = false;
            return true;
        case SpvOpImageSampleDrefImplicitLod:
        case SpvOpImageSampleDrefExplicitLod:
        case SpvOpImageSampleProjDrefImplicitLod:
        case SpvOpImageSampleProjDrefExplicitLod:
        case SpvOpImageGather:
        case SpvOpImageDrefGather:
            rOperandWordStart = 3;
            rOperandWordCount = UINT32_MAX;
            rOperandWordStride = 1;
            rOperandWordSkip = 3;
            rOperandWordSkipString = false;
            return true;
        case SpvOpPhi:
            if (pIncludePhi) {
                rOperandWordStart = 3;
                rOperandWordCount = UINT32_MAX;
                rOperandWordStride = 2;
                rOperandWordSkip = UINT32_MAX;
                rOperandWordSkipString = false;
                return true;
            }
            else {
                rOperandWordStart = 0;
                rOperandWordCount = 0;
                rOperandWordStride = 0;
                rOperandWordSkip = 0;
                rOperandWordSkipString = false;
                return true;
            }
        case SpvOpFunction:
        case SpvOpVariable:
            rOperandWordStart = 4;
            rOperandWordCount = 1;
            rOperandWordStride = 1;
            rOperandWordSkip = UINT32_MAX;
            rOperandWordSkipString = false;
            return true;
        case SpvOpLabel:
        case SpvOpBranch:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpConstant:
        case SpvOpConstantSampler:
        case SpvOpConstantNull:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpCapability:
        case SpvOpExtInstImport:
        case SpvOpMemoryModel:
        case SpvOpTypeVoid:
        case SpvOpTypeBool:
        case SpvOpTypeInt:
        case SpvOpTypeFloat:
        case SpvOpTypeSampler:
        case SpvOpLoopMerge:
        case SpvOpSelectionMerge:
        case SpvOpKill:
        case SpvOpReturn:
        case SpvOpUnreachable:
        case SpvOpFunctionParameter:
        case SpvOpFunctionEnd:
        case SpvOpExtension:
        case SpvOpUndef:
            rOperandWordStart = 0;
            rOperandWordCount = 0;
            rOperandWordStride = 0;
            rOperandWordSkip = 0;
            rOperandWordSkipString = false;
            return true;
        default:
            return false;
        }
    }

    static bool SpvHasLabels(SpvOp pOpCode, uint32_t &rLabelWordStart, uint32_t &rLabelWordCount, uint32_t &rLabelWordStride, bool pIncludePhi) {
        switch (pOpCode) {
        case SpvOpSelectionMerge:
        case SpvOpBranch:
            rLabelWordStart = 1;
            rLabelWordCount = 1;
            rLabelWordStride = 1;
            return true;
        case SpvOpLoopMerge:
            rLabelWordStart = 1;
            rLabelWordCount = 2;
            rLabelWordStride = 1;
            return true;
        case SpvOpBranchConditional:
            rLabelWordStart = 2;
            rLabelWordCount = 2;
            rLabelWordStride = 1;
            return true;
        case SpvOpSwitch:
            rLabelWordStart = 2;
            rLabelWordCount = UINT32_MAX;
            rLabelWordStride = 2;
            return true;
        case SpvOpPhi:
            if (pIncludePhi) {
                rLabelWordStart = 4;
                rLabelWordCount = UINT32_MAX;
                rLabelWordStride = 2;
                return true;
            }
            else {
                return false;
            }
        default:
            return false;
        }
    }

    // Used to indicate which operations have side effects and can't be discarded if their result is not used.
    static bool SpvHasSideEffects(SpvOp pOpCode) {
        switch (pOpCode) {
        case SpvOpFunctionCall:
        case SpvOpAtomicExchange:
        case SpvOpAtomicCompareExchange:
        case SpvOpAtomicCompareExchangeWeak:
        case SpvOpAtomicIIncrement:
        case SpvOpAtomicIDecrement:
        case SpvOpAtomicIAdd:
        case SpvOpAtomicISub:
        case SpvOpAtomicSMin:
        case SpvOpAtomicUMin:
        case SpvOpAtomicSMax:
        case SpvOpAtomicUMax:
        case SpvOpAtomicAnd:
        case SpvOpAtomicOr:
        case SpvOpAtomicXor:
        case SpvOpAtomicFlagTestAndSet:
        case SpvOpAtomicFlagClear:
            return true;
        default:
            return false;
        }
    }

    static bool SpvOpIsTerminator(SpvOp pOpCode) {
        switch (pOpCode) {
        case SpvOpBranch:
        case SpvOpBranchConditional:
        case SpvOpSwitch:
        case SpvOpReturn:
        case SpvOpReturnValue:
        case SpvOpKill:
        case SpvOpUnreachable:
            return true;
        default:
            return false;
        }
    }

    static bool checkOperandWordSkip(uint32_t pWordIndex, const uint32_t *pSpirvWords, uint32_t pRelativeWordIndex, uint32_t pOperandWordSkip, bool pOperandWordSkipString, uint32_t &rOperandWordIndex) {
        if (pRelativeWordIndex == pOperandWordSkip) {
            if (pOperandWordSkipString) {
                const char *operandString = reinterpret_cast<const char *>(&pSpirvWords[pWordIndex + rOperandWordIndex]);
                uint32_t stringLengthInWords = (strlen(operandString) + sizeof(uint32_t)) / sizeof(uint32_t);
                rOperandWordIndex += stringLengthInWords;
            }
            else {
                rOperandWordIndex++;
            }

            return true;
        }
        else {
            return false;
        }
    }

    static uint32_t addToList(uint32_t pInstructionIndex, uint32_t pListIndex, std::vector<ListNode> &rListNodes) {
        rListNodes.emplace_back(pInstructionIndex, pListIndex);
        return uint32_t(rListNodes.size() - 1);
    }

    // Shader

    Shader::Shader() {
        // Empty.
    }

    Shader::Shader(const void *pData, size_t pSize, bool pInlineFunctions) {
        parse(pData, pSize, pInlineFunctions);
    }

    void Shader::clear() {
        extSpirvWords = nullptr;
        extSpirvWordCount = 0;
        inlinedSpirvWords.clear();
        instructions.clear();
        instructionAdjacentListIndices.clear();
        instructionInDegrees.clear();
        instructionOutDegrees.clear();
        instructionOrder.clear();
        blocks.clear();
        blockPreOrderIndices.clear();
        blockPostOrderIndices.clear();
        functions.clear();
        variableOrder.clear();
        results.clear();
        specializations.clear();
        decorations.clear();
        phis.clear();
        loopHeaders.clear();
        listNodes.clear();
        defaultSwitchOpConstantInt = UINT32_MAX;
    }

    constexpr uint32_t SpvStartWordIndex = 5;

    bool Shader::checkData(const void *pData, size_t pSize) {
        const uint32_t *words = reinterpret_cast<const uint32_t *>(pData);
        const size_t wordCount = pSize / sizeof(uint32_t);
        if (wordCount < SpvStartWordIndex) {
            fprintf(stderr, "Not enough words in SPIR-V.\n");
            return false;
        }

        if (words[0] != SpvMagicNumber) {
            fprintf(stderr, "Invalid SPIR-V Magic Number on header.\n");
            return false;
        }

        if (words[1] > SpvVersion) {
            fprintf(stderr, "SPIR-V Version is too new for the library. Max version for the library is 0x%X.\n", SpvVersion);
            return false;
        }

        return true;
    }

    bool Shader::inlineData(const void *pData, size_t pSize) {
        assert(pData != nullptr);
        assert(pSize > 0);

        struct CallItem {
            uint32_t wordIndex = 0;
            uint32_t functionId = UINT32_MAX;
            uint32_t blockId = UINT32_MAX;
            uint32_t startBlockId = UINT32_MAX;
            uint32_t loopBlockId = UINT32_MAX;
            uint32_t continueBlockId = UINT32_MAX;
            uint32_t returnBlockId = UINT32_MAX;
            uint32_t resultType = UINT32_MAX;
            uint32_t resultId = UINT32_MAX;
            uint32_t parameterIndex = 0;
            uint32_t remapsPendingCount = 0;
            uint32_t returnParametersCount = 0;
            uint32_t sameBlockOperationsCount = 0;
            bool startBlockIdAssigned = false;
            bool functionInlined = false;

            CallItem(uint32_t wordIndex, uint32_t functionId = UINT32_MAX, bool functionInlined = false, uint32_t startBlockId = UINT32_MAX, uint32_t loopBlockId = UINT32_MAX, uint32_t continueBlockId = UINT32_MAX, uint32_t returnBlockId = UINT32_MAX, uint32_t resultType = UINT32_MAX, uint32_t resultId = UINT32_MAX)
                : wordIndex(wordIndex), functionId(functionId), functionInlined(functionInlined), startBlockId(startBlockId), loopBlockId(loopBlockId), continueBlockId(continueBlockId), returnBlockId(returnBlockId), resultType(resultType), resultId(resultId)
            {
                // Regular constructor.
            }
        };

        struct FunctionDefinition {
            uint32_t wordIndex = 0;
            uint32_t wordCount = 0;
            uint32_t resultId = UINT32_MAX;
            uint32_t functionWordCount = 0;
            uint32_t codeWordCount = 0;
            uint32_t variableWordCount = 0;
            uint32_t decorationWordCount = 0;
            uint32_t inlineWordCount = 0;
            uint32_t returnValueCount = 0;
            uint32_t callIndex = 0;
            uint32_t callCount = 0;
            uint32_t parameterIndex = 0;
            uint32_t parameterCount = 0;
            uint32_t inlinedVariableWordCount = 0;
            bool canInline = true;
            
            FunctionDefinition() {
                // Default empty constructor.
            }

            FunctionDefinition(uint32_t resultId) : resultId(resultId) {
                // Constructor for sorting.
            }

            bool operator<(const FunctionDefinition &other) const {
                return resultId < other.resultId;
            }
        };

        struct FunctionParameter {
            uint32_t resultId = 0;

            FunctionParameter(uint32_t resultId) : resultId(resultId) {
                // Regular constructor.
            }
        };

        struct FunctionCall {
            uint32_t wordIndex = 0;
            uint32_t functionId = 0;
            uint32_t sameBlockWordCount = 0;

            FunctionCall(uint32_t wordIndex, uint32_t functionId, uint32_t sameBlockWordCount) : wordIndex(wordIndex), functionId(functionId), sameBlockWordCount(sameBlockWordCount) {
                // Regular constructor.
            }
        };

        struct FunctionResult {
            uint32_t wordIndex = UINT32_MAX;
            uint32_t decorationIndex = UINT32_MAX;
        };

        typedef std::vector<FunctionDefinition>::iterator FunctionDefinitionIterator;

        struct FunctionItem {
            FunctionDefinitionIterator function = {};
            FunctionDefinitionIterator rootFunction = {};
            uint32_t callIndex = 0;

            FunctionItem(FunctionDefinitionIterator function, FunctionDefinitionIterator rootFunction, uint32_t callIndex) : function(function), rootFunction(rootFunction), callIndex(callIndex) {
                // Regular constructor.
            }
        };

        struct ResultDecoration {
            uint32_t wordIndex = 0;
            uint32_t nextDecorationIndex = 0;

            ResultDecoration(uint32_t wordIndex, uint32_t nextDecorationIndex) : wordIndex(wordIndex), nextDecorationIndex(nextDecorationIndex) {
                // Regular constructor.
            }
        };

        thread_local std::vector<FunctionResult> functionResultMap;
        thread_local std::vector<ResultDecoration> resultDecorations;
        thread_local std::vector<uint32_t> loopMergeIdStack;
        thread_local std::vector<FunctionDefinition> functionDefinitions;
        thread_local std::vector<FunctionParameter> functionParameters;
        thread_local std::vector<FunctionCall> functionCalls;
        thread_local std::vector<FunctionItem> functionStack;
        thread_local std::vector<CallItem> callStack;
        thread_local std::vector<uint32_t> shaderResultMap;
        thread_local std::vector<uint32_t> storeMap;
        thread_local std::vector<uint32_t> storeMapChanges;
        thread_local std::vector<uint32_t> loadMap;
        thread_local std::vector<uint32_t> loadMapChanges;
        thread_local std::vector<uint32_t> phiMap;
        thread_local std::vector<uint32_t> opPhis;
        thread_local std::vector<uint32_t> remapsPending;
        thread_local std::vector<uint32_t> returnParameters;
        thread_local std::vector<uint32_t> sameBlockOperations;
        functionResultMap.clear();
        resultDecorations.clear();
        loopMergeIdStack.clear();
        functionDefinitions.clear();
        functionParameters.clear();
        functionCalls.clear();
        callStack.clear();
        shaderResultMap.clear();
        storeMap.clear();
        storeMapChanges.clear();
        loadMap.clear();
        loadMapChanges.clear();
        phiMap.clear();
        opPhis.clear();
        remapsPending.clear();
        returnParameters.clear();
        sameBlockOperations.clear();
        
        // Parse all instructions in the shader first.
        const uint32_t *dataWords = reinterpret_cast<const uint32_t *>(pData);
        const size_t dataWordCount = pSize / sizeof(uint32_t);
        const uint32_t dataIdBound = dataWords[3];
        functionResultMap.resize(dataIdBound);

        FunctionDefinition currentFunction;
        uint32_t parseWordIndex = SpvStartWordIndex;
        uint32_t entryPointFunctionId = UINT32_MAX;
        uint32_t globalWordCount = 0;
        uint32_t sameBlockWordCount = 0;
        while (parseWordIndex < dataWordCount) {
            SpvOp opCode = SpvOp(dataWords[parseWordIndex] & 0xFFFFU);
            uint32_t wordCount = (dataWords[parseWordIndex] >> 16U) & 0xFFFFU;
            if (wordCount == 0) {
                fprintf(stderr, "Invalid word count found at %d.\n", parseWordIndex);
                return false;
            }

            switch (opCode) {
            case SpvOpFunction:
                if (currentFunction.resultId != UINT32_MAX) {
                    fprintf(stderr, "Found function start without the previous function ending.\n");
                    return false;
                }

                currentFunction.resultId = dataWords[parseWordIndex + 2];
                currentFunction.wordIndex = parseWordIndex;
                currentFunction.functionWordCount = wordCount;
                break;
            case SpvOpFunctionEnd:
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found function end without a function start.\n");
                    return false;
                }

                currentFunction.wordCount = parseWordIndex + wordCount - currentFunction.wordIndex;
                currentFunction.functionWordCount += wordCount;
                functionDefinitions.emplace_back(currentFunction);

                // Reset the current function to being empty again.
                currentFunction = FunctionDefinition();
                break;
            case SpvOpFunctionParameter:
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found function parameter without a function start.\n");
                    return false;
                }

                currentFunction.functionWordCount += wordCount;

                if (currentFunction.parameterCount == 0) {
                    currentFunction.parameterIndex = uint32_t(functionParameters.size());
                }

                functionParameters.emplace_back(dataWords[parseWordIndex + 2]);
                currentFunction.parameterCount++;
                break;
            case SpvOpFunctionCall:
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found function call without a function start.\n");
                    return false;
                }

                currentFunction.codeWordCount += wordCount;

                if (currentFunction.callCount == 0) {
                    currentFunction.callIndex = uint32_t(functionCalls.size());
                }

                functionCalls.emplace_back(parseWordIndex, dataWords[parseWordIndex + 3], sameBlockWordCount);
                currentFunction.callCount++;
                break;
            case SpvOpDecorate: {
                uint32_t resultId = dataWords[parseWordIndex + 1];
                if (resultId >= dataIdBound) {
                    fprintf(stderr, "Found decoration with invalid result %u.\n", resultId);
                    return false;
                }

                uint32_t nextDecorationIndex = functionResultMap[resultId].decorationIndex;
                functionResultMap[resultId].decorationIndex = uint32_t(resultDecorations.size());
                resultDecorations.emplace_back(parseWordIndex, nextDecorationIndex);
                globalWordCount += wordCount;
                break;
            }
            case SpvOpVariable:
                if (currentFunction.resultId != UINT32_MAX) {
                    // Identify the variable as a local function variable.
                    uint32_t resultId = dataWords[parseWordIndex + 2];
                    if (resultId >= dataIdBound) {
                        fprintf(stderr, "Found variable with invalid result %u.\n", resultId);
                        return false;
                    }

                    currentFunction.variableWordCount += wordCount;
                }
                else {
                    globalWordCount += wordCount;
                }

                break;
            case SpvOpReturn:
                // Functions that use early returns while on a loop can't be inlined.
                if (!loopMergeIdStack.empty()) {
                    currentFunction.canInline = false;
                }

                // If inlined, an OpBranch is required to replace the return.
                currentFunction.inlineWordCount += 2;
                currentFunction.functionWordCount += wordCount;
                break;
            case SpvOpReturnValue:
                // Functions that use early returns while on a loop can't be inlined.
                if (!loopMergeIdStack.empty()) {
                    currentFunction.canInline = false;
                }

                // If inlined, an OpPhi with at least one argument is required to handle return values.
                if (currentFunction.returnValueCount == 1) {
                    currentFunction.inlineWordCount += 5;
                }

                currentFunction.returnValueCount++;

                // An OpBranch is required to replace the return.
                currentFunction.inlineWordCount += 2;

                // An argument in OpPhi is required if there's more than one return value.
                if (currentFunction.returnValueCount > 1) {
                    currentFunction.inlineWordCount += 2;
                }

                currentFunction.functionWordCount += wordCount;
                break;
            case SpvOpEntryPoint:
                if (entryPointFunctionId != UINT32_MAX) {
                    fprintf(stderr, "Found more than one entry point, which is not yet supported.\n");
                    return false;
                }

                entryPointFunctionId = dataWords[parseWordIndex + 2];
                globalWordCount += wordCount;
                break;
            case SpvOpStore: {
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found store outside of a function.\n");
                    return false;
                }

                currentFunction.codeWordCount += wordCount;
                break;
            }
            case SpvOpLabel: {
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found label outside of a function.\n");
                    return false;
                }

                uint32_t labelId = dataWords[parseWordIndex + 1];
                if (!loopMergeIdStack.empty() && (loopMergeIdStack.back() == labelId)) {
                    loopMergeIdStack.pop_back();
                }

                currentFunction.codeWordCount += wordCount;
                sameBlockWordCount = 0;
                break;
            }
            case SpvOpLoopMerge: {
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found loop outside of a function.\n");
                    return false;
                }

                uint32_t mergeId = dataWords[parseWordIndex + 1];
                loopMergeIdStack.emplace_back(mergeId);
                currentFunction.codeWordCount += wordCount;
                break;
            }
            case SpvOpImage:
            case SpvOpSampledImage: {
                if (currentFunction.resultId == UINT32_MAX) {
                    fprintf(stderr, "Found loop outside of a function.\n");
                    return false;
                }

                sameBlockWordCount += wordCount;
                currentFunction.codeWordCount += wordCount;
                break;
            }
            default:
                if (currentFunction.resultId != UINT32_MAX) {
                    currentFunction.codeWordCount += wordCount;
                }
                else {
                    globalWordCount += wordCount;
                }

                break;
            }

            if (currentFunction.resultId != UINT32_MAX) {
                bool hasResult, hasType;
                SpvHasResultAndType(opCode, &hasResult, &hasType);

                if (hasResult) {
                    // Indicate the result is associated to a function.
                    uint32_t resultId = dataWords[parseWordIndex + (hasType ? 2 : 1)];
                    functionResultMap[resultId].wordIndex = parseWordIndex;

                    // Look for all decorations associated to this result. These will be skipped when rewriting
                    // the shader and written back when the result is parsed again.
                    uint32_t decorationIndex = functionResultMap[resultId].decorationIndex;
                    while (decorationIndex != UINT32_MAX) {
                        const ResultDecoration &decoration = resultDecorations[decorationIndex];
                        uint32_t decorationWordCount = (dataWords[decoration.wordIndex] >> 16U) & 0xFFFFU;
                        currentFunction.decorationWordCount += decorationWordCount;
                        globalWordCount -= decorationWordCount;
                        decorationIndex = decoration.nextDecorationIndex;
                    }
                }
            }

            parseWordIndex += wordCount;
        }

        if (entryPointFunctionId == UINT32_MAX) {
            fprintf(stderr, "Unable to find function entry point.\n");
            return false;
        }

        // Make sure function array is sorted to make lower bound searches possible.
        std::sort(functionDefinitions.begin(), functionDefinitions.end());

        // Find the entry point function and mark that it shouldn't be inlined.
        FunctionDefinitionIterator entryFunctionIt = std::lower_bound(functionDefinitions.begin(), functionDefinitions.end(), entryPointFunctionId);
        if (entryFunctionIt == functionDefinitions.end()) {
            fprintf(stderr, "Unable to find entry point function %d.\n", entryPointFunctionId);
            return false;
        }

        entryFunctionIt->canInline = false;

        // Do a first iteration pass with the functions that can't be inlined as the starting points of the stack.
        // This pass will figure out the total size required for the final inlined shader.
        FunctionDefinitionIterator startFunctionIt = functionDefinitions.begin();
        while (startFunctionIt != functionDefinitions.end()) {
            if (!startFunctionIt->canInline) {
                functionStack.emplace_back(startFunctionIt, startFunctionIt, 0);
            }

            startFunctionIt++;
        }
        
        uint32_t codeWordCount = 0;
        uint32_t functionDecorationWordCount = 0;
        while (!functionStack.empty()) {
            FunctionItem &functionItem = functionStack.back();
            if (functionItem.callIndex == functionItem.function->callCount) {
                // Add this function's code and variables.
                codeWordCount += functionItem.function->codeWordCount;
                codeWordCount += functionItem.function->variableWordCount;
                functionDecorationWordCount += functionItem.function->decorationWordCount;

                // This function will be inlined so its variables should be reserved on the parent function instead.
                if (functionItem.function->canInline) {
                    codeWordCount += functionItem.function->inlineWordCount;
                    functionItem.rootFunction->inlinedVariableWordCount += functionItem.function->variableWordCount;
                }
                // Only add the function's word counts if can't be inlined.
                else {
                    codeWordCount += functionItem.function->functionWordCount;
                }

                functionStack.pop_back();
            }
            else {
                // Traverse the function calls to be inlined
                const FunctionCall &functionCall = functionCalls[functionItem.function->callIndex + functionItem.callIndex];
                functionItem.callIndex++;

                uint32_t callFunctionId = dataWords[functionCall.wordIndex + 3];
                FunctionDefinitionIterator callFunctionIt = std::lower_bound(functionDefinitions.begin(), functionDefinitions.end(), callFunctionId);
                if (callFunctionIt == functionDefinitions.end()) {
                    fprintf(stderr, "Unable to find function %d.\n", callFunctionId);
                    return false;
                }
                
                if (callFunctionIt->canInline) {
                    // Function call will be replaced by one OpLoopMerge, three OpLabel and three OpBranch.
                    // All words required for preserving same block operations will also be added.
                    // Substract the word count for the function call as it'll not be copied.
                    uint32_t callWordCount = (dataWords[functionCall.wordIndex] >> 16U) & 0xFFFFU;
                    codeWordCount += 4 + 2 * 3 + 2 * 3;
                    codeWordCount += functionCall.sameBlockWordCount;
                    codeWordCount -= callWordCount;
                    functionStack.emplace_back(callFunctionIt, functionItem.rootFunction, 0);
                }
            }
        }

        // Figure out the total size of the shader and copy the header.
        size_t totalWordCount = SpvStartWordIndex + globalWordCount + codeWordCount + functionDecorationWordCount;
        inlinedSpirvWords.resize(totalWordCount);
        memcpy(inlinedSpirvWords.data(), pData, SpvStartWordIndex * sizeof(uint32_t));

        // To avoid reallocation of these unless the shader really warrants it, we reserve some memory for these vectors.
        uint32_t &inlinedIdBound = inlinedSpirvWords[3];
        uint32_t dstWordIndex = SpvStartWordIndex;
        shaderResultMap.resize(dataIdBound, UINT32_MAX);
        storeMap.resize(dataIdBound, UINT32_MAX);
        loadMap.resize(dataIdBound, UINT32_MAX);
        phiMap.resize(dataIdBound, UINT32_MAX);

        auto copyInstruction = [&](uint32_t dataWordIndex, bool renameResult, uint32_t &copyWordIndex, uint32_t &copyDecorationIndex) {
            copyDecorationIndex = UINT32_MAX;

            SpvOp opCode = SpvOp(dataWords[dataWordIndex] & 0xFFFFU);
            uint32_t wordCount = (dataWords[dataWordIndex] >> 16U) & 0xFFFFU;
            for (uint32_t i = 0; i < wordCount; i++) {
                inlinedSpirvWords[copyWordIndex + i] = dataWords[dataWordIndex + i];
            }

            bool hasResult, hasType;
            SpvHasResultAndType(opCode, &hasResult, &hasType);

            if (hasResult) {
                // Any inlined functions must remap all their results and operands.
                uint32_t &resultId = inlinedSpirvWords[copyWordIndex + (hasType ? 2 : 1)];
                if ((resultId < dataIdBound) && (functionResultMap[resultId].wordIndex != UINT32_MAX)) {
                    copyDecorationIndex = functionResultMap[resultId].decorationIndex;
                }

                if (renameResult) {
                    // First labels in a function will be replaced by the assigned label if present.
                    uint32_t newResultId;
                    if ((opCode == SpvOpLabel) && (callStack.back().startBlockId != UINT32_MAX) && !callStack.back().startBlockIdAssigned) {
                        newResultId = callStack.back().startBlockId;
                        callStack.back().startBlockIdAssigned = true;
                    }
                    else {
                        newResultId = inlinedIdBound++;
                    }

                    // Remap and replace the result ID in the instruction.
                    shaderResultMap[resultId] = newResultId;
                    resultId = newResultId;

                    // Store the current block's remapped label.
                    if (opCode == SpvOpLabel) {
                        callStack.back().blockId = resultId;
                    }
                }
            }

            // Remap any operands or labels present in the instructions.
            uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
            bool operandWordSkipString;
            if (SpvHasOperands(opCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, true)) {
                uint32_t operandWordIndex = operandWordStart;
                for (uint32_t j = 0; j < operandWordCount; j++) {
                    if (checkOperandWordSkip(callStack.back().wordIndex, dataWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                        continue;
                    }

                    if (operandWordIndex >= wordCount) {
                        break;
                    }

                    uint32_t shaderWordIndex = copyWordIndex + operandWordIndex;
                    uint32_t &operandId = inlinedSpirvWords[shaderWordIndex];

                    // Discard any known stores for variables that are used in operations that the effect is not explicitly considered yet.
                    if ((opCode != SpvOpStore) && (opCode != SpvOpLoad)) {
                        storeMap[operandId] = dataIdBound;
                    }
                    
                    // Rename the operand if it originates from a load.
                    if (loadMap[operandId] < dataIdBound) {
                        operandId = loadMap[operandId];
                    }

                    // Apply the result remapping.
                    if (shaderResultMap[operandId] != UINT32_MAX) {
                        operandId = shaderResultMap[operandId];
                    }

                    operandWordIndex += operandWordStride;
                }
            }

            uint32_t labelWordStart, labelWordCount, labelWordStride;
            if (SpvHasLabels(opCode, labelWordStart, labelWordCount, labelWordStride, true)) {
                for (uint32_t j = 0; (j < labelWordCount) && ((labelWordStart + j * labelWordStride) < wordCount); j++) {
                    uint32_t labelWordIndex = labelWordStart + j * labelWordStride;
                    remapsPending.emplace_back(copyWordIndex + labelWordIndex);
                    callStack.back().remapsPendingCount++;
                }
            }

            copyWordIndex += wordCount;
        };

        auto copyDecorations = [&](uint32_t copyDecorationIndex, uint32_t &copyWordIndex) {
            uint32_t placeholderWordIndex;
            while (copyDecorationIndex != UINT32_MAX) {
                copyInstruction(resultDecorations[copyDecorationIndex].wordIndex, false, copyWordIndex, placeholderWordIndex);
                copyDecorationIndex = resultDecorations[copyDecorationIndex].nextDecorationIndex;
            }
        };

        // Perform the final pass for inlining all functions.
        uint32_t copyDecorationIndex;
        uint32_t dstInlinedDecorationWordIndex = UINT32_MAX;
        uint32_t dstInlinedDecorationWordIndexMax = UINT32_MAX;
        uint32_t dstInlinedVariableWordIndex = UINT32_MAX;
        uint32_t dstInlinedVariableWordIndexMax = UINT32_MAX;
        callStack.emplace_back(SpvStartWordIndex);
        while (!callStack.empty()) {
            uint32_t callWordIndex = callStack.back().wordIndex;
            if (callWordIndex >= dataWordCount) {
                break;
            }

            bool copyWords = true;
            bool copyWordsToVariables = false;
            SpvOp opCode = SpvOp(dataWords[callWordIndex] & 0xFFFFU);
            uint32_t wordCount = (dataWords[callWordIndex] >> 16U) & 0xFFFFU;
            if (wordCount == 0) {
                fprintf(stderr, "Function iteration landed in an invalid instruction due to an implementation error.\n");
                return false;
            }

            switch (opCode) {
            case SpvOpLabel:
                while (!storeMapChanges.empty()) {
                    storeMap[storeMapChanges.back()] = UINT32_MAX;
                    storeMapChanges.pop_back();
                }

                while (!loadMapChanges.empty()) {
                    loadMap[loadMapChanges.back()] = UINT32_MAX;
                    loadMapChanges.pop_back();
                }

                sameBlockOperations.resize(sameBlockOperations.size() - callStack.back().sameBlockOperationsCount);
                callStack.back().blockId = dataWords[callWordIndex + 1];
                callStack.back().sameBlockOperationsCount = 0;
                break;
            case SpvOpFunction: {
                uint32_t functionId = dataWords[callWordIndex + 2];
                FunctionDefinitionIterator functionIt = std::lower_bound(functionDefinitions.begin(), functionDefinitions.end(), functionId);
                if (functionIt == functionDefinitions.end()) {
                    fprintf(stderr, "Unable to find function %d.\n", functionId);
                    return false;
                }
                
                // If we're iterating on the top of the shader, we skip over the function.
                // Only copy the function's words if it's not inlined and we're iterating on it.
                if (callStack.back().functionId == UINT32_MAX) {
                    // Skip parsing the entire function on this stack level.
                    callStack.back().wordIndex += functionIt->wordCount;

                    // Insert a new stack level if we found function that isn't inlined.
                    if (!functionIt->canInline) {
                        callStack.emplace_back(callWordIndex - wordCount, functionId);
                    }
                    else {
                        callStack.back().wordIndex -= wordCount;
                    }

                    copyWords = false;
                }
                else {
                    copyWords = !functionIt->canInline;
                }

                break;
            }
            case SpvOpFunctionParameter:
                // Only copy the function's parameters if it's not inlined.
                copyWords = !callStack.back().functionInlined;
                break;
            case SpvOpFunctionEnd: {
                // Apply any pending remappings from instructions with labels.
                for (size_t i = remapsPending.size() - callStack.back().remapsPendingCount; i < remapsPending.size(); i++) {
                    uint32_t &resultId = inlinedSpirvWords[remapsPending[i]];
                    if (shaderResultMap[resultId] != UINT32_MAX) {
                        resultId = shaderResultMap[resultId];
                    }
                }

                // Only copy the function's end if it's not inlined.
                if (!callStack.back().functionInlined) {
                    copyWords = true;

                    if (dstInlinedVariableWordIndex != dstInlinedVariableWordIndexMax) {
                        fprintf(stderr, "Failed to fill all available variable space due to an implementation error.\n");
                        return false;
                    }

                    dstInlinedVariableWordIndex = UINT32_MAX;
                    dstInlinedVariableWordIndexMax = UINT32_MAX;
                }
                else {
                    // Insert a label for the continue block that connects back to the start along with a branch.
                    inlinedSpirvWords[dstWordIndex++] = SpvOpLabel | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = callStack.back().continueBlockId;

                    inlinedSpirvWords[dstWordIndex++] = SpvOpBranch | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = callStack.back().loopBlockId;

                    // Insert a label for the return block.
                    inlinedSpirvWords[dstWordIndex++] = SpvOpLabel | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = callStack.back().returnBlockId;

                    // If the function only returns one possible value, the caller instead will just remap the result to this one.
                    if (callStack.back().returnParametersCount == 2) {
                        uint32_t functionResultId = callStack.back().resultId;
                        shaderResultMap[functionResultId] = returnParameters[returnParameters.size() - callStack.back().returnParametersCount];
                    }
                    // Insert an OpPhi for selecting the result from a function call that called a function that returns multiple values.
                    else if (callStack.back().returnParametersCount > 2) {
                        // Remap the function result if necessary.
                        const CallItem &previousCallStack = callStack[callStack.size() - 2];
                        uint32_t functionResultId = callStack.back().resultId;
                        if ((previousCallStack.functionId != UINT32_MAX) && previousCallStack.functionInlined) {
                            uint32_t newFunctionResultId = inlinedIdBound++;
                            shaderResultMap[functionResultId] = newFunctionResultId;
                            functionResultId = newFunctionResultId;
                        }

                        opPhis.emplace_back(dstWordIndex);
                        inlinedSpirvWords[dstWordIndex++] = SpvOpPhi | ((3 + callStack.back().returnParametersCount) << 16U);
                        inlinedSpirvWords[dstWordIndex++] = callStack.back().resultType;
                        inlinedSpirvWords[dstWordIndex++] = functionResultId;

                        // Copy the OpPhi arguments directly.
                        for (size_t i = returnParameters.size() - callStack.back().returnParametersCount; i < returnParameters.size(); i++) {
                            inlinedSpirvWords[dstWordIndex++] = returnParameters[i];
                        }
                    }

                    copyWords = false;
                }

                // Pop this stack level and return to iterating on the previous one.
                remapsPending.resize(remapsPending.size() - callStack.back().remapsPendingCount);
                returnParameters.resize(returnParameters.size() - callStack.back().returnParametersCount);
                sameBlockOperations.resize(sameBlockOperations.size() - callStack.back().sameBlockOperationsCount);
                callStack.pop_back();

                if (!callStack.empty()) {
                    // Copy the same block operations and rename the results even if the function wasn't inlined.
                    for (size_t i = sameBlockOperations.size() - callStack.back().sameBlockOperationsCount; i < sameBlockOperations.size(); i++) {
                        copyInstruction(sameBlockOperations[i], true, dstWordIndex, copyDecorationIndex);
                        copyDecorations(copyDecorationIndex, dstInlinedDecorationWordIndex);
                    }

                    callStack.back().wordIndex -= wordCount;
                }

                break;
            }
            case SpvOpFunctionCall: {
                // Inline the function by inserting two labels and a branch.
                uint32_t functionId = dataWords[callWordIndex + 3];
                FunctionDefinitionIterator functionIt = std::lower_bound(functionDefinitions.begin(), functionDefinitions.end(), functionId);
                if (functionIt == functionDefinitions.end()) {
                    fprintf(stderr, "Unable to find function %d.\n", functionId);
                    return false;
                }

                if (functionIt->canInline) {
                    // Generate the ID that will be used to indicate the function's start and the return block.
                    uint32_t loopLabelId = inlinedIdBound++;
                    uint32_t startLabelId = inlinedIdBound++;
                    uint32_t continueLabelId = inlinedIdBound++;
                    uint32_t returnLabelId = inlinedIdBound++;

                    // In any future Phi operations, rename the current label to the return label.
                    if (callStack.back().blockId >= phiMap.size()) {
                        phiMap.resize(callStack.back().blockId + 1, UINT32_MAX);
                    }

                    phiMap[callStack.back().blockId] = returnLabelId;

                    // Branch into a new block. The new block will contain a single iteration loop.
                    inlinedSpirvWords[dstWordIndex++] = SpvOpBranch | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = loopLabelId;

                    inlinedSpirvWords[dstWordIndex++] = SpvOpLabel | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = loopLabelId;

                    inlinedSpirvWords[dstWordIndex++] = SpvOpLoopMerge | (4 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = returnLabelId;
                    inlinedSpirvWords[dstWordIndex++] = continueLabelId;
                    inlinedSpirvWords[dstWordIndex++] = SpvLoopControlMaskNone;

                    inlinedSpirvWords[dstWordIndex++] = SpvOpBranch | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = startLabelId;

                    // Pass the result Id unmodified. The function evaluation will determine how it should be remapped.
                    uint32_t functionResultId = dataWords[callWordIndex + 2];
                    callStack.back().wordIndex += wordCount;

                    // Word count should be substracted as the loop's end will add it.
                    callStack.emplace_back(functionIt->wordIndex - wordCount, functionIt->resultId, true, startLabelId, loopLabelId, continueLabelId, returnLabelId, dataWords[callWordIndex + 1], functionResultId);

                    for (uint32_t i = 0; i < functionIt->parameterCount; i++) {
                        if (wordCount <= (4 + i)) {
                            fprintf(stderr, "Not enough words for argument %d in function call.\n", i);
                            return false;
                        }

                        uint32_t functionParameterId = functionParameters[functionIt->parameterIndex + i].resultId;
                        uint32_t localParameterId = dataWords[callWordIndex + 4 + i];
                        if (shaderResultMap[localParameterId] != UINT32_MAX) {
                            localParameterId = shaderResultMap[localParameterId];
                        }

                        shaderResultMap[functionParameterId] = localParameterId;
                    }

                    copyWords = false;
                }
                else {
                    copyWords = true;
                }

                break;
            }
            case SpvOpDecorate: {
                if (dstInlinedDecorationWordIndex == UINT32_MAX) {
                    // Upon encountering the first decoration in the shader, reserve space to write out any decorations
                    // that are found to be linked to function results.
                    dstInlinedDecorationWordIndex = dstWordIndex;
                    dstWordIndex += functionDecorationWordCount;
                    dstInlinedDecorationWordIndexMax = dstWordIndex;
                }

                // Only copy the decoration as-is if it doesn't belong to a result in a function.
                uint32_t resultId = dataWords[callWordIndex + 1];
                copyWords = (functionResultMap[resultId].wordIndex == UINT32_MAX);
                break;
            }
            case SpvOpVariable:
                if ((callStack.back().functionId < UINT32_MAX) && !callStack.back().functionInlined) {
                    // As soon as we find a variable local to the function, reserve the space to insert all
                    // inlined function variables that we encounter.
                    if (dstInlinedVariableWordIndex == UINT32_MAX) {
                        FunctionDefinitionIterator functionIt = std::lower_bound(functionDefinitions.begin(), functionDefinitions.end(), callStack.back().functionId);
                        if (functionIt == functionDefinitions.end()) {
                            fprintf(stderr, "Unable to find function %d.\n", callStack.back().functionId);
                            return false;
                        }

                        dstInlinedVariableWordIndex = dstWordIndex;
                        dstWordIndex += functionIt->inlinedVariableWordCount;
                        dstInlinedVariableWordIndexMax = dstWordIndex;
                    }
                }
                else {
                    // Copy the variables into the entry point function's variables.
                    copyWordsToVariables = (callStack.back().functionId != UINT32_MAX);
                }

                copyWords = true;
                break;
            case SpvOpReturn:
                if (callStack.back().functionInlined) {
                    // Replace return with a branch to the return label.
                    inlinedSpirvWords[dstWordIndex++] = SpvOpBranch | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = callStack.back().returnBlockId;
                    copyWords = false;
                }
                else {
                    // Copy as is.
                }

                break;
            case SpvOpReturnValue: {
                if (callStack.back().functionInlined) {
                    // Replace return with a branch to the return label.
                    inlinedSpirvWords[dstWordIndex++] = SpvOpBranch | (2 << 16U);
                    inlinedSpirvWords[dstWordIndex++] = callStack.back().returnBlockId;
                    copyWords = false;

                    // Store parameters for Phi operator.
                    uint32_t operandId = dataWords[callStack.back().wordIndex + 1];
                    if (shaderResultMap[operandId] != UINT32_MAX) {
                        operandId = shaderResultMap[operandId];
                    }

                    returnParameters.emplace_back(operandId);
                    returnParameters.emplace_back(callStack.back().blockId);
                    callStack.back().returnParametersCount += 2;
                }
                else {
                    // Copy as is.
                }

                break;
            }
            case SpvOpLoad: {
                // If the pointer being loaded was modified this block, store its result to rename the
                // operands that use the result of this load operation. This load operation will go
                // unused and be deleted in the optimization pass.
                // Ignore load operations with memory operands.
                if (wordCount == 4) {
                    uint32_t pointerId = dataWords[callStack.back().wordIndex + 3];
                    if (pointerId >= dataIdBound) {
                        fprintf(stderr, "Found load operation with invalid pointer %u.\n", pointerId);
                        return false;
                    }

                    uint32_t pointerWordIndex = functionResultMap[pointerId].wordIndex;
                    if ((pointerWordIndex != UINT32_MAX) && (SpvOp(dataWords[pointerWordIndex] & 0xFFFFU) == SpvOpVariable) && (storeMap[pointerId] < dataIdBound)) {
                        uint32_t resultId = dataWords[callStack.back().wordIndex + 2];
                        if (loadMap[resultId] != storeMap[pointerId]) {
                            loadMap[resultId] = storeMap[pointerId];
                            loadMapChanges.emplace_back(resultId);
                        }
                    }
                }

                break;
            }
            case SpvOpStore: {
                // Keep track of the result last stored to the pointer on this block.
                // Ignore store operations with memory operands.
                if (wordCount == 3) {
                    uint32_t pointerId = dataWords[callStack.back().wordIndex + 1];
                    if (pointerId >= dataIdBound) {
                        fprintf(stderr, "Found store operation with invalid pointer %u.\n", pointerId);
                        return false;
                    }

                    uint32_t resultId = dataWords[callStack.back().wordIndex + 2];
                    if (resultId >= dataIdBound) {
                        fprintf(stderr, "Found store operation with invalid result %u.\n", resultId);
                        return false;
                    }

                    if (storeMap[pointerId] != resultId) {
                        storeMap[pointerId] = resultId;
                        storeMapChanges.emplace_back(pointerId);
                    }
                }

                break;
            }
            case SpvOpPhi:
                opPhis.emplace_back(dstWordIndex);
                break;
            case SpvOpImage:
            case SpvOpSampledImage: {
                sameBlockOperations.emplace_back(callStack.back().wordIndex);
                callStack.back().sameBlockOperationsCount++;
                break;
            }
            default:
                break;
            }

            if (copyWords) {
                uint32_t &copyWordIndex = copyWordsToVariables ? dstInlinedVariableWordIndex : dstWordIndex;
                copyInstruction(callWordIndex, callStack.back().functionInlined, copyWordIndex, copyDecorationIndex);
                copyDecorations(copyDecorationIndex, dstInlinedDecorationWordIndex);
            }

            if (!callStack.empty()) {
                callStack.back().wordIndex += wordCount;
            }

            assert(dstWordIndex <= totalWordCount && "Not enough words were reserved for the shader.");
            assert(dstInlinedVariableWordIndex <= dstInlinedVariableWordIndexMax && "Not enough words were reserved for inlined variables.");
            assert(dstInlinedDecorationWordIndex <= dstInlinedDecorationWordIndexMax && "Not enough words were reserved for function decorations.");
        }

        if (dstWordIndex != totalWordCount) {
            fprintf(stderr, "Failed to fill all shader data due to an implementation error.\n");
            return false;
        }

        // Fix any OpPhi operators with the labels for the blocks that were split.
        for (uint32_t wordIndex : opPhis) {
            uint32_t wordCount = (inlinedSpirvWords[wordIndex] >> 16U) & 0xFFFFU;
            for (uint32_t j = 3; j < wordCount; j += 2) {
                uint32_t &labelId = inlinedSpirvWords[wordIndex + j + 1];
                while ((phiMap.size() > labelId) && (phiMap[labelId] != UINT32_MAX)) {
                    labelId = phiMap[labelId];
                }
            }
        }

        return true;
    }

    bool Shader::parseData(const void *pData, size_t pSize) {
        assert(pData != nullptr);
        assert(pSize > 0);

        const uint32_t *dataWords = reinterpret_cast<const uint32_t *>(pData);
        const size_t dataWordCount = pSize / sizeof(uint32_t);
        const uint32_t idBound = dataWords[3];
        instructions.reserve(idBound);
        results.resize(idBound, Result());
        results.shrink_to_fit();

        // Parse all instructions.
        uint32_t blockIndex = UINT32_MAX;
        uint32_t functionInstructionIndex = UINT32_MAX;
        uint32_t functionLabelIndex = UINT32_MAX;
        uint32_t blockInstructionIndex = UINT32_MAX;
        uint32_t wordIndex = SpvStartWordIndex;
        while (wordIndex < dataWordCount) {
            SpvOp opCode = SpvOp(dataWords[wordIndex] & 0xFFFFU);
            uint32_t wordCount = (dataWords[wordIndex] >> 16U) & 0xFFFFU;
            if (wordCount == 0) {
                fprintf(stderr, "SPIR-V Parsing error. Invalid instruction word count at word %d.\n", wordIndex);
                return false;
            }

            bool hasResult, hasType;
            SpvHasResultAndType(opCode, &hasResult, &hasType);

            uint32_t instructionIndex = uint32_t(instructions.size());
            if (hasResult) {
                uint32_t resultId = dataWords[wordIndex + (hasType ? 2 : 1)];
                if (resultId >= idBound) {
                    fprintf(stderr, "SPIR-V Parsing error. Invalid Result ID: %u.\n", resultId);
                    return false;
                }

                results[resultId].instructionIndex = instructionIndex;
            }

            // Handle specific instructions.
            switch (opCode) {
            case SpvOpFunction:
                functionInstructionIndex = instructionIndex;
                break;
            case SpvOpFunctionEnd:
                functions.emplace_back(functionInstructionIndex, functionLabelIndex);
                functionInstructionIndex = functionLabelIndex = UINT32_MAX;
                break;
            case SpvOpDecorate:
            case SpvOpMemberDecorate:
                decorations.emplace_back(instructionIndex);
                break;
            case SpvOpPhi:
                phis.emplace_back(instructionIndex);
                break;
            case SpvOpLoopMerge:
                loopHeaders.emplace_back(instructionIndex, blockInstructionIndex);
                break;
            case SpvOpLabel:
                blockIndex = uint32_t(blocks.size());
                blockInstructionIndex = instructionIndex;

                if (functionLabelIndex == UINT32_MAX) {
                    functionLabelIndex = blockInstructionIndex;
                }

                break;
            default:
                break;
            }

            instructions.emplace_back(wordIndex, blockIndex);

            if (SpvOpIsTerminator(opCode)) {
                blocks.emplace_back(blockInstructionIndex, instructionIndex);
                blockIndex = UINT32_MAX;
                blockInstructionIndex = UINT32_MAX;
            }

            wordIndex += wordCount;
        }

        // Initialize all adjacent indices for the lists.
        instructionAdjacentListIndices.resize(instructions.size(), UINT32_MAX);

        return true;
    }

    bool Shader::process(const void *pData, size_t pSize) {
        // Greatly decreases the costs of adding nodes to the linked list.
        listNodes.reserve(instructions.size() * 2);

        thread_local std::vector<uint32_t> loopMergeBlockStack;
        thread_local std::vector<uint32_t> loopMergeInstructionStack;
        thread_local std::vector<bool> preOrderVisitedBlocks;
        thread_local std::vector<bool> postOrderVisitedBlocks;
        loopMergeBlockStack.clear();
        loopMergeInstructionStack.clear();
        preOrderVisitedBlocks.clear();
        postOrderVisitedBlocks.clear();

        bool foundOpSwitch = false;
        const uint32_t *dataWords = reinterpret_cast<const uint32_t *>(pData);
        const size_t dataWordCount = pSize / sizeof(uint32_t);
        uint32_t currentBlockId = 0;
        uint32_t currentLoopHeaderIndex = 0;
        for (uint32_t i = 0; i < uint32_t(instructions.size()); i++) {
            uint32_t wordIndex = instructions[i].wordIndex;
            SpvOp opCode = SpvOp(dataWords[wordIndex] & 0xFFFFU);
            uint32_t wordCount = (dataWords[wordIndex] >> 16U) & 0xFFFFU;
            if (!SpvIsSupported(opCode)) {
                fprintf(stderr, "%s is not supported yet.\n", SpvOpToString(opCode));
                return false;
            }

            bool hasResult, hasType;
            SpvHasResultAndType(opCode, &hasResult, &hasType);

            if (hasType) {
                uint32_t typeId = dataWords[wordIndex + 1];
                if (typeId >= results.size()) {
                    fprintf(stderr, "SPIR-V Parsing error. Invalid Type ID: %u.\n", typeId);
                    return false;
                }

                if (results[typeId].instructionIndex == UINT32_MAX) {
                    fprintf(stderr, "SPIR-V Parsing error. Result %u is not valid.\n", typeId);
                    return false;
                }

                uint32_t typeInstructionIndex = results[typeId].instructionIndex;
                instructionAdjacentListIndices[typeInstructionIndex] = addToList(i, instructionAdjacentListIndices[typeInstructionIndex], listNodes);

                // Check if it's an OpConstant of Int type so it can be reused on switches.
                if ((opCode == SpvOpConstant) && (defaultSwitchOpConstantInt == UINT32_MAX)) {
                    uint32_t typeWordIndex = instructions[typeInstructionIndex].wordIndex;
                    SpvOp typeOpCode = SpvOp(dataWords[typeWordIndex] & 0xFFFFU);
                    if (typeOpCode == SpvOpTypeInt) {
                        defaultSwitchOpConstantInt = dataWords[wordIndex + 2];
                    }
                }
            }
            
            // Every operand should be adjacent to this instruction.
            uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
            bool operandWordSkipString;
            if (SpvHasOperands(opCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, false)) {
                uint32_t operandWordIndex = operandWordStart;
                for (uint32_t j = 0; j < operandWordCount; j++) {
                    if (checkOperandWordSkip(wordIndex, dataWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                        continue;
                    }

                    if (operandWordIndex >= wordCount) {
                        break;
                    }

                    uint32_t operandId = dataWords[wordIndex + operandWordIndex];
                    if (operandId >= results.size()) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", operandId);
                        return false;
                    }

                    if (results[operandId].instructionIndex == UINT32_MAX) {
                        fprintf(stderr, "SPIR-V Parsing error. Result %u is not valid.\n", operandId);
                        return false;
                    }

                    uint32_t resultIndex = results[operandId].instructionIndex;
                    instructionAdjacentListIndices[resultIndex] = addToList(i, instructionAdjacentListIndices[resultIndex], listNodes);
                    operandWordIndex += operandWordStride;
                }
            }
            else {
                fprintf(stderr, "SPIR-V Parsing error. Operands for %s are not implemented yet.\n", SpvOpToString(opCode));
                return false;
            }

            // This instruction should be adjacent to every label referenced.
            uint32_t labelWordStart, labelWordCount, labelWordStride;
            if (SpvHasLabels(opCode, labelWordStart, labelWordCount, labelWordStride, false)) {
                for (uint32_t j = 0; (j < labelWordCount) && ((labelWordStart + j * labelWordStride) < wordCount); j++) {
                    uint32_t labelId = dataWords[wordIndex + labelWordStart + j * labelWordStride];
                    if (labelId >= results.size()) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", labelId);
                        return false;
                    }

                    if (results[labelId].instructionIndex == UINT32_MAX) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", labelId);
                        return false;
                    }

                    // Make sure this label not pointing back to the loop header while on a loop merge.
                    if (!loopMergeBlockStack.empty() && (labelId == loopMergeBlockStack.back())) {
                        continue;
                    }

                    uint32_t labelIndex = results[labelId].instructionIndex;
                    instructionAdjacentListIndices[i] = addToList(labelIndex, instructionAdjacentListIndices[i], listNodes);
                }
            }

            // Parse parented blocks of OpPhi to indicate the dependency.
            if (opCode == SpvOpPhi) {
                uint32_t continueLabelId = UINT32_MAX;
                if (!loopMergeInstructionStack.empty()) {
                    uint32_t loopMergeWordIndex = instructions[loopMergeInstructionStack.back()].wordIndex;
                    continueLabelId = dataWords[loopMergeWordIndex + 2];
                }

                for (uint32_t j = 3; j < wordCount; j += 2) {
                    uint32_t labelId = dataWords[wordIndex + j + 1];
                    if (labelId >= results.size()) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Parent ID: %u.\n", labelId);
                        return false;
                    }

                    if (results[labelId].instructionIndex == UINT32_MAX) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Parent ID: %u.\n", labelId);
                        return false;
                    }
                    
                    // Make sure this label doesn't come from the loop continue.
                    if (labelId == continueLabelId) {
                        continue;
                    }

                    uint32_t operandId = dataWords[wordIndex + j];
                    if (operandId >= results.size()) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", operandId);
                        return false;
                    }

                    if (results[operandId].instructionIndex == UINT32_MAX) {
                        fprintf(stderr, "SPIR-V Parsing error. Result %u is not valid.\n", operandId);
                        return false;
                    }

                    uint32_t labelIndex = results[labelId].instructionIndex;
                    uint32_t resultIndex = results[operandId].instructionIndex;
                    instructionAdjacentListIndices[labelIndex] = addToList(i, instructionAdjacentListIndices[labelIndex], listNodes);
                    instructionAdjacentListIndices[resultIndex] = addToList(i, instructionAdjacentListIndices[resultIndex], listNodes);
                }
            }
            // Parse decorations.
            else if (opCode == SpvOpDecorate) {
                uint32_t decoration = dataWords[wordIndex + 2];
                if (decoration == SpvDecorationSpecId) {
                    uint32_t resultId = dataWords[wordIndex + 1];
                    uint32_t constantId = dataWords[wordIndex + 3];
                    if (resultId >= results.size()) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", resultId);
                        return false;
                    }

                    uint32_t resultInstructionIndex = results[resultId].instructionIndex;
                    if (resultInstructionIndex == UINT32_MAX) {
                        fprintf(stderr, "SPIR-V Parsing error. Invalid Operand ID: %u.\n", resultId);
                        return false;
                    }

                    specializations.resize(std::max(specializations.size(), size_t(constantId + 1)));
                    specializations[constantId].constantInstructionIndex = resultInstructionIndex;
                    specializations[constantId].decorationInstructionIndex = i;
                }
            }
            // Check if a switch is used in the shader.
            else if (opCode == SpvOpSwitch) {
                foundOpSwitch = true;
            }
            // If a loop merge stack is active, pop it if it corresponds to the merge block.
            else if (opCode == SpvOpLabel) {
                currentBlockId = dataWords[wordIndex + 1];

                if ((currentLoopHeaderIndex < loopHeaders.size()) && (i == loopHeaders[currentLoopHeaderIndex].blockInstructionIndex)) {
                    loopMergeBlockStack.emplace_back(currentBlockId);
                    loopMergeInstructionStack.emplace_back(loopHeaders[currentLoopHeaderIndex].instructionIndex);
                    currentLoopHeaderIndex++;
                }

                if (!loopMergeBlockStack.empty() && !loopMergeInstructionStack.empty()) {
                    uint32_t loopMergeWordIndex = instructions[loopMergeInstructionStack.back()].wordIndex;
                    uint32_t mergeBlockId = dataWords[loopMergeWordIndex + 1];
                    if (currentBlockId == mergeBlockId) {
                        loopMergeBlockStack.pop_back();
                        loopMergeInstructionStack.pop_back();
                    }
                }
            }
        }
        
        // Do a pre-order and post-order traversal of the tree starting from each function. These indices are
        // later used to figure out whether instructions dominate other instructions when doing optimizations.
        thread_local std::vector<uint32_t> blockIndexStack;
        thread_local std::vector<uint32_t> blockAdjacentStack;
        uint32_t preOrderIndex = 0;
        uint32_t postOrderIndex = 0;
        blockPreOrderIndices.resize(blocks.size(), 0);
        blockPostOrderIndices.resize(blocks.size(), 0);
        preOrderVisitedBlocks.resize(blocks.size(), false);
        postOrderVisitedBlocks.resize(blocks.size(), false);
        for (uint32_t i = 0; i < uint32_t(functions.size()); i++) {
            const Function &function = functions[i];
            const Instruction &functionLabelInstruction = instructions[function.labelInstructionIndex];
            blockIndexStack.clear();
            blockAdjacentStack.clear();
            blockIndexStack.emplace_back(functionLabelInstruction.blockIndex);
            blockAdjacentStack.emplace_back(UINT32_MAX);
            while (!blockIndexStack.empty()) {
                uint32_t blockIndex = blockIndexStack.back();
                uint32_t blockAdjacentIndex = blockAdjacentStack.back();
                blockIndexStack.pop_back();
                blockAdjacentStack.pop_back();

                uint32_t terminatorInstructorIndex = blocks[blockIndex].terminatorInstructionIndex;
                if (!preOrderVisitedBlocks[blockIndex]) {
                    blockPreOrderIndices[blockIndex] = preOrderIndex++;
                    blockAdjacentIndex = instructionAdjacentListIndices[terminatorInstructorIndex];
                    preOrderVisitedBlocks[blockIndex] = true;
                }

                if ((blockAdjacentIndex == UINT32_MAX) && !postOrderVisitedBlocks[blockIndex]) {
                    blockPostOrderIndices[blockIndex] = postOrderIndex++;
                    postOrderVisitedBlocks[blockIndex] = true;
                }
                
                while (blockAdjacentIndex != UINT32_MAX) {
                    const ListNode &adjacentListNode = listNodes[blockAdjacentIndex];
                    const Instruction &adjacentInstruction = instructions[adjacentListNode.instructionIndex];
                    SpvOp adjacentOpCode = SpvOp(dataWords[adjacentInstruction.wordIndex] & 0xFFFFU);
                    if (adjacentOpCode == SpvOpLabel) {
                        blockIndexStack.emplace_back(blockIndex);
                        blockAdjacentStack.emplace_back(adjacentListNode.nextListIndex);
                        blockIndexStack.emplace_back(adjacentInstruction.blockIndex);
                        blockAdjacentStack.emplace_back(UINT32_MAX);
                        blockAdjacentIndex = UINT32_MAX;
                    }
                    else {
                        blockAdjacentIndex = adjacentListNode.nextListIndex;
                    }
                }
            }
        }

        if (foundOpSwitch && (defaultSwitchOpConstantInt == UINT32_MAX)) {
            fprintf(stderr, "Unable to find an OpConstantInt to use as replacement for switches. Adding this instruction automatically is not supported yet.\n");
            return false;
        }

        return true;
    }

    struct InstructionSort {
        union {
            struct {
                uint64_t instructionIndex : 32;
                uint64_t instructionLevel : 32;
            };

            uint64_t instructionValue = 0;
        };

        InstructionSort() {
            // Empty.
        }

        bool operator<(const InstructionSort &i) const {
            return instructionValue < i.instructionValue;
        }
    };

    bool Shader::sort(const void *pData, size_t pSize) {
        const uint32_t *dataWords = reinterpret_cast<const uint32_t *>(pData);
        const size_t dataWordCount = pSize / sizeof(uint32_t);

        // Count the in and out degrees for all instructions.
        instructionInDegrees.clear();
        instructionOutDegrees.clear();
        instructionInDegrees.resize(instructions.size(), 0);
        instructionOutDegrees.resize(instructions.size(), 0);
        for (uint32_t i = 0; i < uint32_t(instructions.size()); i++) {
            uint32_t listIndex = instructionAdjacentListIndices[i];
            while (listIndex != UINT32_MAX) {
                const ListNode &listNode = listNodes[listIndex];
                instructionInDegrees[listNode.instructionIndex]++;
                instructionOutDegrees[i]++;
                listIndex = listNode.nextListIndex;
            }
        }

        // Sort degrees doesn't need to be cleared as its contents will be copied over.
        thread_local std::vector<uint32_t> sortDegrees;
        thread_local std::vector<uint32_t> instructionStack;
        thread_local std::vector<InstructionSort> instructionSortVector;
        instructionStack.clear();
        instructionSortVector.clear();

        // Make a copy of the degrees as they'll be used to perform a topological sort.
        sortDegrees.resize(instructionInDegrees.size());
        memcpy(sortDegrees.data(), instructionInDegrees.data(), sizeof(uint32_t) * sortDegrees.size());

        // The first nodes to be processed should be the ones with no incoming connections.
        for (uint32_t i = 0; i < uint32_t(instructions.size()); i++) {
            if (sortDegrees[i] == 0) {
                instructionStack.emplace_back(i);
            }
        }

        instructionOrder.reserve(instructions.size());
        instructionOrder.clear();
        while (!instructionStack.empty()) {
            uint32_t i = instructionStack.back();
            instructionStack.pop_back();
            instructionOrder.emplace_back(i);

            // Look for the adjacents and reduce their degree. Push it to the stack if their degree reaches zero.
            uint32_t listIndex = instructionAdjacentListIndices[i];
            while (listIndex != UINT32_MAX) {
                const ListNode &listNode = listNodes[listIndex];
                uint32_t &sortDegree = sortDegrees[listNode.instructionIndex];
                assert(sortDegree > 0);
                sortDegree--;
                if (sortDegree == 0) {
                    instructionStack.emplace_back(listNode.instructionIndex);
                }

                listIndex = listNode.nextListIndex;
            }
        }
        
        if (instructionOrder.size() < instructions.size()) {
            fprintf(stderr, "Sorting shader failed. Not all instructions could be reached.\n");
#if RESPV_VERBOSE_ERRORS
            for (uint32_t i = 0; i < uint32_t(instructions.size()); i++) {
                if (sortDegrees[i] != 0) {
                    fprintf(stderr, "[%d] Remaining Degrees %d\n", i, sortDegrees[i]);
                }
            }
#endif
            return false;
        }

        instructionSortVector.resize(instructionOrder.size(), InstructionSort());
        for (uint32_t instructionIndex : instructionOrder) {
            uint64_t nextLevel = instructionSortVector[instructionIndex].instructionLevel + 1;
            uint32_t listIndex = instructionAdjacentListIndices[instructionIndex];
            while (listIndex != UINT32_MAX) {
                const ListNode &listNode = listNodes[listIndex];
                instructionSortVector[listNode.instructionIndex].instructionLevel = std::max(instructionSortVector[listNode.instructionIndex].instructionLevel, nextLevel);
                listIndex = listNode.nextListIndex;
            }

            instructionSortVector[instructionIndex].instructionIndex = instructionIndex;
        }

        std::sort(instructionSortVector.begin(), instructionSortVector.end());
        
        // Rebuild the instruction order vector with the sorted indices. If any of the instructions are pointers, store 
        // them in a separate vector that will be used for another optimization pass.
        instructionOrder.clear();
        variableOrder.clear();
        for (InstructionSort &instructionSort : instructionSortVector) {
            instructionOrder.emplace_back(uint32_t(instructionSort.instructionIndex));

            uint32_t wordIndex = instructions[instructionSort.instructionIndex].wordIndex;
            SpvOp opCode = SpvOp(dataWords[wordIndex] & 0xFFFFU);
            if (opCode == SpvOpVariable) {
                variableOrder.emplace_back(uint32_t(instructionSort.instructionIndex));
            }
        }

        return true;
    }

    bool Shader::parse(const void *pData, size_t pSize, bool pInlineFunctions) {
        assert(pData != nullptr);
        assert((pSize % sizeof(uint32_t) == 0) && "Size of data must be aligned to the word size.");

        clear();

        if (!checkData(pData, pSize)) {
            return false;
        }

        extSpirvWords = reinterpret_cast<const uint32_t *>(pData);
        extSpirvWordCount = pSize / sizeof(uint32_t);

        if (pInlineFunctions && !inlineData(pData, pSize)) {
            clear();
            return false;
        }

        const void *data = pInlineFunctions ? inlinedSpirvWords.data() : pData;
        const size_t size = pInlineFunctions ? (inlinedSpirvWords.size() * sizeof(uint32_t)) : pSize;
        if (!parseData(data, size)) {
            clear();
            return false;
        }

        if (!process(data, size)) {
            clear();
            return false;
        }

        if (!sort(data, size)) {
            clear();
            return false;
        }

        return true;
    }

    bool Shader::empty() const {
        return inlinedSpirvWords.empty() && ((extSpirvWords == nullptr) || (extSpirvWordCount == 0));
    }

    // Optimizer

    struct Resolution {
        enum Type {
            Unknown,
            Constant,
            Variable
        };

        Type type = Type::Unknown;

        struct {
            union {
                int32_t i32;
                uint32_t u32;
            };
        } value = {};

        static Resolution fromBool(bool pValue) {
            Resolution r;
            r.type = Type::Constant;
            r.value.u32 = pValue ? 1 : 0;
            return r;
        }

        static Resolution fromInt32(int32_t pValue) {
            Resolution r;
            r.type = Type::Constant;
            r.value.i32 = pValue;
            return r;
        }

        static Resolution fromUint32(uint32_t pValue) {
            Resolution r;
            r.type = Type::Constant;
            r.value.u32 = pValue;
            return r;
        }
    };

    struct OptimizerContext {
        const Shader &shader;
        std::vector<uint32_t> &instructionAdjacentListIndices;
        std::vector<uint32_t> &instructionInDegrees;
        std::vector<uint32_t> &instructionOutDegrees;
        std::vector<ListNode> &listNodes;
        std::vector<Resolution> &resolutions;
        std::vector<uint8_t> &optimizedData;
        Options options;

        OptimizerContext(const Shader &shader, std::vector<uint32_t> &instructionAdjacentListIndices, std::vector<uint32_t> &instructionInDegrees, std::vector<uint32_t> &instructionOutDegrees, std::vector<ListNode> &listNodes, std::vector<Resolution> &resolutions, std::vector<uint8_t> &optimizedData, Options options) :
            shader(shader), instructionAdjacentListIndices(instructionAdjacentListIndices), instructionInDegrees(instructionInDegrees), instructionOutDegrees(instructionOutDegrees), listNodes(listNodes), resolutions(resolutions), optimizedData(optimizedData), options(options)
        {
                // Regular constructor.
        }
    };

    static void optimizerEliminateInstruction(uint32_t pInstructionIndex, OptimizerContext &rContext) {
        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        uint32_t wordIndex = rContext.shader.instructions[pInstructionIndex].wordIndex;
        uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
        for (uint32_t j = 0; j < wordCount; j++) {
            optimizedWords[wordIndex + j] = UINT32_MAX;
        }
    }

    static void optimizerReduceResultDegrees(OptimizerContext &rContext, std::vector<uint32_t> &rResultStack) {
        const uint32_t *optimizedWords = reinterpret_cast<const uint32_t *>(rContext.optimizedData.data());
        auto optimizerCheckOperands = [&](SpvOp opCode, uint32_t wordIndex, uint32_t wordCount) {
            uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
            bool operandWordSkipString;
            if (SpvHasOperands(opCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, true)) {
                uint32_t operandWordIndex = operandWordStart;
                for (uint32_t j = 0; j < operandWordCount; j++) {
                    if (checkOperandWordSkip(wordIndex, optimizedWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                        continue;
                    }

                    if (operandWordIndex >= wordCount) {
                        break;
                    }

                    uint32_t operandId = optimizedWords[wordIndex + operandWordIndex];
                    rResultStack.emplace_back(operandId);
                    operandWordIndex += operandWordStride;
                }
            }
        };

        while (!rResultStack.empty()) {
            uint32_t resultId = rResultStack.back();
            rResultStack.pop_back();

            uint32_t instructionIndex = rContext.shader.results[resultId].instructionIndex;
            uint32_t wordIndex = rContext.shader.instructions[instructionIndex].wordIndex;

            // Instruction's been deleted.
            if (optimizedWords[wordIndex] == UINT32_MAX) {
                continue;
            }

            // Consider it's possible for a result to have no outgoing connections on an unoptimized shader.
            if (rContext.instructionOutDegrees[instructionIndex] > 0) {
                rContext.instructionOutDegrees[instructionIndex]--;
            }

            // When nothing uses the result from this instruction anymore, we can delete it. Push any operands it uses into the stack as well to reduce their out degrees.
            // Function calls are excluded from this as it's not easy to evaluate whether the function has side effects or not.
            SpvOp opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
            if ((rContext.instructionOutDegrees[instructionIndex] == 0) && !SpvHasSideEffects(opCode)) {
                uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
                optimizerCheckOperands(opCode, wordIndex, wordCount);

                // Function parameters are excluded from being deleted as they'd break the function type definitions.
                // For being able to delete them, the original function type would have to be modified and only as long as no other functions are reusing the same type definition.
                if (opCode != SpvOpFunctionParameter) {
                    optimizerEliminateInstruction(instructionIndex, rContext);
                }

                // When a function is deleted, we just delete any instructions we can find until finding the function end.
                if (opCode == SpvOpFunction) {
                    bool foundFunctionEnd = false;
                    uint32_t instructionCount = rContext.shader.instructions.size();
                    for (uint32_t i = instructionIndex; (i < instructionCount) && !foundFunctionEnd; i++) {
                        wordIndex = rContext.shader.instructions[i].wordIndex;
                        if (optimizedWords[wordIndex] == UINT32_MAX) {
                            continue;
                        }

                        opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
                        wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
                        foundFunctionEnd = opCode == SpvOpFunctionEnd;

                        optimizerCheckOperands(opCode, wordIndex, wordCount);
                        optimizerEliminateInstruction(i, rContext);
                    }
                }
            }
        }
    }

    static bool optimizerPrepareData(OptimizerContext &rContext) {
        OptimizerContext &c = rContext;
        c.resolutions.clear();
        c.resolutions.resize(c.shader.results.size(), Resolution());
        c.instructionAdjacentListIndices.resize(c.shader.instructionAdjacentListIndices.size());
        c.instructionInDegrees.resize(c.shader.instructionInDegrees.size());
        c.instructionOutDegrees.resize(c.shader.instructionOutDegrees.size());
        c.listNodes.resize(c.shader.listNodes.size());
        memcpy(c.instructionAdjacentListIndices.data(), c.shader.instructionAdjacentListIndices.data(), sizeof(uint32_t) * c.shader.instructionAdjacentListIndices.size());
        memcpy(c.instructionInDegrees.data(), c.shader.instructionInDegrees.data(), sizeof(uint32_t) * c.shader.instructionInDegrees.size());
        memcpy(c.instructionOutDegrees.data(), c.shader.instructionOutDegrees.data(), sizeof(uint32_t) * c.shader.instructionOutDegrees.size());
        memcpy(c.listNodes.data(), c.shader.listNodes.data(), sizeof(ListNode) * c.shader.listNodes.size());

        if (c.shader.inlinedSpirvWords.empty()) {
            c.optimizedData.resize(c.shader.extSpirvWordCount * sizeof(uint32_t));
            memcpy(c.optimizedData.data(), c.shader.extSpirvWords, c.optimizedData.size());
        }
        else {
            c.optimizedData.resize(c.shader.inlinedSpirvWords.size() * sizeof(uint32_t));
            memcpy(c.optimizedData.data(), c.shader.inlinedSpirvWords.data(), c.optimizedData.size());
        }

        return true;
    }

    static bool optimizerPatchSpecializationConstants(const SpecConstant *pNewSpecConstants, uint32_t pNewSpecConstantCount, OptimizerContext &rContext) {
        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        for (uint32_t i = 0; i < pNewSpecConstantCount; i++) {
            const SpecConstant &newSpecConstant = pNewSpecConstants[i];
            if (newSpecConstant.specId >= rContext.shader.specializations.size()) {
                continue;
            }

            const Specialization &specialization = rContext.shader.specializations[newSpecConstant.specId];
            if (specialization.constantInstructionIndex == UINT32_MAX) {
                continue;
            }

            uint32_t constantWordIndex = rContext.shader.instructions[specialization.constantInstructionIndex].wordIndex;
            SpvOp constantOpCode = SpvOp(optimizedWords[constantWordIndex] & 0xFFFFU);
            uint32_t constantWordCount = (optimizedWords[constantWordIndex] >> 16U) & 0xFFFFU;
            switch (constantOpCode) {
            case SpvOpSpecConstantTrue:
            case SpvOpSpecConstantFalse:
                optimizedWords[constantWordIndex] = (newSpecConstant.values[0] ? SpvOpConstantTrue : SpvOpConstantFalse) | (constantWordCount << 16U);
                break;
            case SpvOpSpecConstant:
                if (constantWordCount <= 3) {
                    fprintf(stderr, "Optimization error. Specialization constant has less words than expected.\n");
                    return false;
                }

                if (newSpecConstant.values.size() != (constantWordCount - 3)) {
                    fprintf(stderr, "Optimization error. Value count for specialization constant %u differs from the expected size.\n", newSpecConstant.specId);
                    return false;
                }

                optimizedWords[constantWordIndex] = SpvOpConstant | (constantWordCount << 16U);
                memcpy(&optimizedWords[constantWordIndex + 3], newSpecConstant.values.data(), sizeof(uint32_t) * (constantWordCount - 3));
                break;
            default:
                fprintf(stderr, "Optimization error. Can't patch opCode %u.\n", constantOpCode);
                return false;
            }

            // Eliminate the decorator instruction as well.
            optimizerEliminateInstruction(specialization.decorationInstructionIndex, rContext);
        }

        return true;
    }

    static void optimizerEvaluateResult(uint32_t pResultId, OptimizerContext &rContext) {
        const uint32_t *optimizedWords = reinterpret_cast<const uint32_t *>(rContext.optimizedData.data());
        const Result &result = rContext.shader.results[pResultId];
        Resolution &resolution = rContext.resolutions[pResultId];
        uint32_t resultWordIndex = rContext.shader.instructions[result.instructionIndex].wordIndex;
        SpvOp opCode = SpvOp(optimizedWords[resultWordIndex] & 0xFFFFU);
        uint32_t wordCount = (optimizedWords[resultWordIndex] >> 16U) & 0xFFFFU;
        switch (opCode) {
        case SpvOpConstant: {
            // Parse the known type of constants. Any other types will be considered as variable.
            const Result &typeResult = rContext.shader.results[optimizedWords[resultWordIndex + 1]];
            uint32_t typeWordIndex = rContext.shader.instructions[typeResult.instructionIndex].wordIndex;
            SpvOp typeOpCode = SpvOp(optimizedWords[typeWordIndex] & 0xFFFFU);
            uint32_t typeWidthInBits = optimizedWords[typeWordIndex + 2];
            uint32_t typeSigned = optimizedWords[typeWordIndex + 3];
            if ((typeOpCode == SpvOpTypeInt) && (typeWidthInBits == 32)) {
                if (typeSigned) {
                    resolution = Resolution::fromInt32(int32_t(optimizedWords[resultWordIndex + 3]));
                }
                else {
                    resolution = Resolution::fromUint32(optimizedWords[resultWordIndex + 3]);
                }
            }
            else {
                resolution.type = Resolution::Type::Variable;
            }

            break;
        }
        case SpvOpConstantTrue:
            resolution = Resolution::fromBool(true);
            break;
        case SpvOpConstantFalse:
            resolution = Resolution::fromBool(false);
            break;
        case SpvOpBitcast: {
            const Resolution &operandResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            resolution = Resolution::fromUint32(operandResolution.value.u32);
            break;
        }
        case SpvOpIAdd: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 + secondResolution.value.u32);
            break;
        }
        case SpvOpISub: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 - secondResolution.value.u32);
            break;
        }
        case SpvOpIMul: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 * secondResolution.value.u32);
            break;
        }
        case SpvOpUDiv: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 / secondResolution.value.u32);
            break;
        }
        case SpvOpSDiv: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.i32 / secondResolution.value.i32);
            break;
        }
        case SpvOpLogicalEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool((firstResolution.value.u32 != 0) == (secondResolution.value.u32 != 0));
            break;
        }
        case SpvOpLogicalNotEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool((firstResolution.value.u32 != 0) != (secondResolution.value.u32 != 0));
            break;
        }
        case SpvOpLogicalOr: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool((firstResolution.value.u32 != 0) || (secondResolution.value.u32 != 0));
            break;
        }
        case SpvOpLogicalAnd: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool((firstResolution.value.u32 != 0) && (secondResolution.value.u32 != 0));
            break;
        }
        case SpvOpLogicalNot: {
            const Resolution &operandResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            resolution = Resolution::fromBool(operandResolution.value.u32 == 0);
            break;
        }
        case SpvOpSelect: {
            const Resolution &conditionResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 5]];
            resolution = (conditionResolution.value.u32 != 0) ? firstResolution : secondResolution;
            break;
        }
        case SpvOpIEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 == secondResolution.value.u32);
            break;
        }
        case SpvOpINotEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 != secondResolution.value.u32);
            break;
        }
        case SpvOpUGreaterThan: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 > secondResolution.value.u32);
            break;
        }
        case SpvOpSGreaterThan: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.i32 > secondResolution.value.i32);
            break;
        }
        case SpvOpUGreaterThanEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 >= secondResolution.value.u32);
            break;
        }
        case SpvOpSGreaterThanEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.i32 >= secondResolution.value.i32);
            break;
        }
        case SpvOpULessThan: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 < secondResolution.value.u32);
            break;
        }
        case SpvOpSLessThan: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.i32 < secondResolution.value.i32);
            break;
        }
        case SpvOpULessThanEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.u32 <= secondResolution.value.u32);
            break;
        }
        case SpvOpSLessThanEqual: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromBool(firstResolution.value.i32 <= secondResolution.value.i32);
            break;
        }
        case SpvOpShiftRightLogical: {
            const Resolution &baseResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &shiftResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(baseResolution.value.u32 >> shiftResolution.value.u32);
            break;
        }
        case SpvOpShiftRightArithmetic: {
            const Resolution &baseResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &shiftResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromInt32(baseResolution.value.i32 >> shiftResolution.value.i32);
            break;
        }
        case SpvOpShiftLeftLogical: {
            const Resolution &baseResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &shiftResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(baseResolution.value.u32 << shiftResolution.value.u32);
            break;
        }
        case SpvOpBitwiseOr: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 | secondResolution.value.u32);
            break;
        }
        case SpvOpBitwiseAnd: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 & secondResolution.value.u32);
            break;
        }
        case SpvOpBitwiseXor: {
            const Resolution &firstResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            const Resolution &secondResolution = rContext.resolutions[optimizedWords[resultWordIndex + 4]];
            resolution = Resolution::fromUint32(firstResolution.value.u32 ^ secondResolution.value.u32);
            break;
        }
        case SpvOpNot: {
            const Resolution &operandResolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            resolution = Resolution::fromUint32(~operandResolution.value.u32);
            break;
        }
        case SpvOpPhi: {
            // Resolve as constant if Phi operator was compacted to only one option.
            if (wordCount == 5) {
                resolution = rContext.resolutions[optimizedWords[resultWordIndex + 3]];
            }
            else {
                resolution.type = Resolution::Type::Variable;
            }

            break;
        }
        default:
            // It's not known how to evaluate the instruction, consider the result a variable.
            resolution.type = Resolution::Type::Variable;
            break;
        }
    }

    static void optimizerReduceLabelDegree(uint32_t pFirstLabelId, OptimizerContext &rContext) {
        thread_local std::vector<uint32_t> labelStack;
        thread_local std::vector<uint32_t> resultStack;
        thread_local std::vector<uint32_t> degreeReductions;
        labelStack.emplace_back(pFirstLabelId);
        resultStack.clear();
        degreeReductions.clear();

        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        while (!labelStack.empty()) {
            uint32_t labelId = labelStack.back();
            labelStack.pop_back();

            uint32_t instructionIndex = rContext.shader.results[labelId].instructionIndex;
            if (rContext.instructionInDegrees[instructionIndex] == 0) {
                continue;
            }

            rContext.instructionInDegrees[instructionIndex]--;

            // If a label's degree becomes 0, eliminate all the instructions of the block.
            // Eliminate as many instructions as possible until finding the terminator of the block.
            // When finding the terminator, look at the labels it has and push them to the stack to
            // reduce their degrees as well.
            if (rContext.instructionInDegrees[instructionIndex] == 0) {
                bool foundTerminator = false;
                uint32_t instructionCount = rContext.shader.instructions.size();
                for (uint32_t i = instructionIndex; (i < instructionCount) && !foundTerminator; i++) {
                    uint32_t wordIndex = rContext.shader.instructions[i].wordIndex;
                    if (optimizedWords[wordIndex] == UINT32_MAX) {
                        continue;
                    }

                    // If the instruction has labels it can reference, we push the labels to reduce their degrees as well.
                    SpvOp opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
                    uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
                    uint32_t labelWordStart, labelWordCount, labelWordStride;
                    if (SpvHasLabels(opCode, labelWordStart, labelWordCount, labelWordStride, false)) {
                        for (uint32_t j = 0; (j < labelWordCount) && ((labelWordStart + j * labelWordStride) < wordCount); j++) {
                            uint32_t terminatorLabelId = optimizedWords[wordIndex + labelWordStart + j * labelWordStride];
                            labelStack.emplace_back(terminatorLabelId);
                        }
                    }

                    // If the instruction has operands, decrease their degree.
                    uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
                    bool operandWordSkipString;
                    if (SpvHasOperands(opCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, true)) {
                        uint32_t operandWordIndex = operandWordStart;
                        for (uint32_t j = 0; j < operandWordCount; j++) {
                            if (checkOperandWordSkip(wordIndex, optimizedWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                                continue;
                            }

                            if (operandWordIndex >= wordCount) {
                                break;
                            }

                            uint32_t operandId = optimizedWords[wordIndex + operandWordIndex];
                            resultStack.emplace_back(operandId);
                            operandWordIndex += operandWordStride;
                        }
                    }

                    foundTerminator = SpvOpIsTerminator(opCode);
                    optimizerEliminateInstruction(i, rContext);
                }
            }
        }

        optimizerReduceResultDegrees(rContext, resultStack);
    }

    static void optimizerEvaluateTerminator(uint32_t pInstructionIndex, OptimizerContext &rContext) {
        // For each type of supported terminator, check if the operands can be resolved into constants.
        // If they can be resolved, eliminate any other branches that don't pass the condition.
        uint32_t wordIndex = rContext.shader.instructions[pInstructionIndex].wordIndex;
        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        SpvOp opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
        uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
        uint32_t defaultLabelId = UINT32_MAX;

        // Both instructions share that the second word is the operator they must use to resolve the condition.
        // Operator can't be anything but a constant to be able to resolve a terminator.
        const uint32_t operatorId = optimizedWords[wordIndex + 1];
        const Resolution &operatorResolution = rContext.resolutions[operatorId];
        if (operatorResolution.type != Resolution::Type::Constant) {
            return;
        }
        
        if (opCode == SpvOpBranchConditional) {
            // Branch conditional only needs to choose either label depending on whether the result is true or false.
            if (operatorResolution.value.u32) {
                defaultLabelId = optimizedWords[wordIndex + 2];
                optimizerReduceLabelDegree(optimizedWords[wordIndex + 3], rContext);
            }
            else {
                defaultLabelId = optimizedWords[wordIndex + 3];
                optimizerReduceLabelDegree(optimizedWords[wordIndex + 2], rContext);
            }

            // If there's a selection merge before this branch, we place the unconditional branch in its place.
            const uint32_t mergeWordCount = 3;
            uint32_t mergeWordIndex = wordIndex - mergeWordCount;
            SpvOp mergeOpCode = SpvOp(optimizedWords[mergeWordIndex] & 0xFFFFU);

            uint32_t patchWordIndex;
            if (mergeOpCode == SpvOpSelectionMerge) {
                optimizerReduceLabelDegree(optimizedWords[mergeWordIndex + 1], rContext);
                patchWordIndex = mergeWordIndex;
            }
            else {
                patchWordIndex = wordIndex;
            }

            // Make the final label the new default case and reduce the word count.
            optimizedWords[patchWordIndex] = SpvOpBranch | (2U << 16U);
            optimizedWords[patchWordIndex + 1] = defaultLabelId;

            // Eliminate any remaining words on the block.
            for (uint32_t i = patchWordIndex + 2; i < (wordIndex + wordCount); i++) {
                optimizedWords[i] = UINT32_MAX;
            }
        }
        else if (opCode == SpvOpSwitch) {
            // Switch must compare the integer result of the operator to all the possible labels.
            // If the label is not as possible result, then reduce its block's degree.
            for (uint32_t i = 3; i < wordCount; i += 2) {
                if (operatorResolution.value.u32 == optimizedWords[wordIndex + i]) {
                    defaultLabelId = optimizedWords[wordIndex + i + 1];
                }
                else {
                    optimizerReduceLabelDegree(optimizedWords[wordIndex + i + 1], rContext);
                }
            }

            // If none are chosen, the default label is selected. Otherwise, reduce the block's degree
            // for the default label.
            if (defaultLabelId == UINT32_MAX) {
                defaultLabelId = optimizedWords[wordIndex + 2];
            }
            else {
                optimizerReduceLabelDegree(optimizedWords[wordIndex + 2], rContext);
            }

            // Make the final label the new default case and reduce the word count.
            optimizedWords[wordIndex] = SpvOpSwitch | (3U << 16U);
            optimizedWords[wordIndex + 1] = rContext.shader.defaultSwitchOpConstantInt;
            optimizedWords[wordIndex + 2] = defaultLabelId;

            // Increase the degree of the default constant that was chosen so it's not considered as dead code.
            uint32_t defaultConstantInstructionIndex = rContext.shader.results[rContext.shader.defaultSwitchOpConstantInt].instructionIndex;
            rContext.instructionOutDegrees[defaultConstantInstructionIndex]++;

            // Eliminate any remaining words on the block.
            for (uint32_t i = wordIndex + 3; i < (wordIndex + wordCount); i++) {
                optimizedWords[i] = UINT32_MAX;
            }
        }

        // The condition operator can be discarded.
        thread_local std::vector<uint32_t> resultStack;
        resultStack.clear();
        resultStack.emplace_back(operatorId);
        optimizerReduceResultDegrees(rContext, resultStack);
    }

    static bool optimizerCompactPhi(uint32_t pInstructionIndex, OptimizerContext &rContext) {
        // Do a backwards search first to find out what label this instruction belongs to.
        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        uint32_t searchInstructionIndex = pInstructionIndex;
        uint32_t instructionLabelId = UINT32_MAX;
        while (searchInstructionIndex > 0) {
            uint32_t searchWordIndex = rContext.shader.instructions[searchInstructionIndex].wordIndex;
            SpvOp searchOpCode = SpvOp(optimizedWords[searchWordIndex] & 0xFFFFU);
            if (searchOpCode == SpvOpLabel) {
                instructionLabelId = optimizedWords[searchWordIndex + 1];
                break;
            }

            searchInstructionIndex--;
        }

        if (instructionLabelId == UINT32_MAX) {
            fprintf(stderr, "Unable to find a label before OpPhi.\n");
            return false;
        }

        thread_local std::vector<uint32_t> resultStack;
        resultStack.clear();

        uint32_t wordIndex = rContext.shader.instructions[pInstructionIndex].wordIndex;
        uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
        uint32_t newWordCount = 3;
        uint32_t instructionCount = rContext.shader.instructions.size();
        for (uint32_t i = 3; i < wordCount; i += 2) {
            uint32_t labelId = optimizedWords[wordIndex + i + 1];
            uint32_t labelInstructionIndex = rContext.shader.results[labelId].instructionIndex;
            uint32_t labelWordIndex = rContext.shader.instructions[labelInstructionIndex].wordIndex;

            // Label's been eliminated. Skip it.
            if (optimizedWords[labelWordIndex] == UINT32_MAX) {
                resultStack.emplace_back(optimizedWords[wordIndex + i]);
                continue;
            }

            // While the label may not have been eliminated, verify its terminator is still pointing to this block.
            bool foundBranchToThisBlock = false;
            for (uint32_t j = labelInstructionIndex; j < instructionCount; j++) {
                uint32_t searchWordIndex = rContext.shader.instructions[j].wordIndex;
                SpvOp searchOpCode = SpvOp(optimizedWords[searchWordIndex] & 0xFFFFU);
                uint32_t searchWordCount = (optimizedWords[searchWordIndex] >> 16U) & 0xFFFFU;
                if (SpvOpIsTerminator(searchOpCode)) {
                    uint32_t labelWordStart, labelWordCount, labelWordStride;
                    if (SpvHasLabels(searchOpCode, labelWordStart, labelWordCount, labelWordStride, false)) {
                        for (uint32_t j = 0; (j < labelWordCount) && ((labelWordStart + j * labelWordStride) < searchWordCount); j++) {
                            uint32_t searchLabelId = optimizedWords[searchWordIndex + labelWordStart + j * labelWordStride];
                            if (searchLabelId == instructionLabelId) {
                                foundBranchToThisBlock = true;
                                break;
                            }
                        }
                    }

                    break;
                }
            }

            // The preceding block did not have any reference to this block. Skip it.
            if (!foundBranchToThisBlock) {
                resultStack.emplace_back(optimizedWords[wordIndex + i]);
                continue;
            }

            // Copy the words.
            optimizedWords[wordIndex + newWordCount + 0] = optimizedWords[wordIndex + i + 0];
            optimizedWords[wordIndex + newWordCount + 1] = optimizedWords[wordIndex + i + 1];
            newWordCount += 2;
        }

        // Patch in the new word count.
        assert((optimizedWords[wordIndex] != UINT32_MAX) && "The instruction shouldn't be getting deleted from reducing the degree of the operands.");
        optimizedWords[wordIndex] = SpvOpPhi | (newWordCount << 16U);

        // Delete any of the remaining words.
        for (uint32_t i = newWordCount; i < wordCount; i++) {
            optimizedWords[wordIndex + i] = UINT32_MAX;
        }

        optimizerReduceResultDegrees(rContext, resultStack);

        return true;
    }

    static bool optimizerRunEvaluationPass(OptimizerContext &rContext) {
        if (!rContext.options.removeDeadCode) {
            return true;
        }

        thread_local std::vector<uint32_t> resultStack;
        resultStack.clear();

        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        uint32_t orderCount = uint32_t(rContext.shader.instructionOrder.size());
        for (uint32_t i = 0; i < orderCount; i++) {
            uint32_t instructionIndex = rContext.shader.instructionOrder[i];
            uint32_t wordIndex = rContext.shader.instructions[instructionIndex].wordIndex;

            // Instruction has been deleted.
            if (optimizedWords[wordIndex] == UINT32_MAX) {
                continue;
            }

            SpvOp opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
            uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
            uint32_t patchedWordCount = wordCount;
            bool hasResult, hasType;
            SpvHasResultAndType(opCode, &hasResult, &hasType);

            if (hasResult) {
                const uint32_t resultId = optimizedWords[wordIndex + (hasType ? 2 : 1)];
                if ((opCode != SpvOpLabel) && (opCode != SpvOpFunctionCall) && (rContext.instructionOutDegrees[instructionIndex] == 0)) {
                    resultStack.emplace_back(resultId);
                }
                else {
                    if (opCode == SpvOpPhi) {
                        if (optimizerCompactPhi(instructionIndex, rContext)) {
                            patchedWordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
                        }
                        else {
                            return false;
                        }
                    }

                    // Check if any of the operands isn't a constant.
                    bool allOperandsAreConstant = true;
                    uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
                    bool operandWordSkipString;
                    if (SpvHasOperands(opCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, true)) {
                        uint32_t operandWordIndex = operandWordStart;
                        for (uint32_t j = 0; j < operandWordCount; j++) {
                            if (checkOperandWordSkip(wordIndex, optimizedWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                                continue;
                            }

                            if (operandWordIndex >= patchedWordCount) {
                                break;
                            }

                            uint32_t operandId = optimizedWords[wordIndex + operandWordIndex];
                            assert((operandId != UINT32_MAX) && "An operand that's been deleted shouldn't be getting evaluated.");

                            // It shouldn't be possible for an operand to not be solved, but OpPhi can do so because previous blocks might've been deleted.
                            if ((opCode != SpvOpPhi) && (rContext.resolutions[operandId].type == Resolution::Type::Unknown)) {
                                fprintf(stderr, "Error in resolution of the operations. Operand %u was not solved.\n", operandId);
                                return false;
                            }

                            if (rContext.resolutions[operandId].type == Resolution::Type::Variable) {
                                allOperandsAreConstant = false;
                                break;
                            }

                            operandWordIndex += operandWordStride;
                        }
                    }

                    // The result can only be evaluated if all operands are constant.
                    if (allOperandsAreConstant) {
                        optimizerEvaluateResult(resultId, rContext);
                    }
                    else {
                        rContext.resolutions[resultId].type = Resolution::Type::Variable;
                    }
                }
            }
            else if ((opCode == SpvOpBranchConditional) || (opCode == SpvOpSwitch)) {
                optimizerEvaluateTerminator(instructionIndex, rContext);
            }
        }

        optimizerReduceResultDegrees(rContext, resultStack);

        return true;
    }

    static bool optimizerDoesInstructionDominate(const Shader &pShader, const Instruction &pInstructionA, const Instruction &pInstructionB) {
        // If on the same block, the instruction will only dominate the other one if it precedes it.
        if (pInstructionA.blockIndex == pInstructionB.blockIndex) {
            return pInstructionA.wordIndex < pInstructionB.wordIndex;
        }
        // If the blocks are different, compare the indices of the pre-order and post-order traversal
        // to determine whether it dominates the other block.
        else {
            const uint32_t aPreIndex = pShader.blockPreOrderIndices[pInstructionA.blockIndex];
            const uint32_t bPreIndex = pShader.blockPreOrderIndices[pInstructionB.blockIndex];
            const uint32_t aPostIndex = pShader.blockPostOrderIndices[pInstructionA.blockIndex];
            const uint32_t bPostIndex = pShader.blockPostOrderIndices[pInstructionB.blockIndex];
            return (aPreIndex < bPreIndex) && (aPostIndex > bPostIndex);
        }
    }

    static bool optimizerRemoveUnusedVariables(OptimizerContext &rContext) {
        if (!rContext.options.removeDeadCode) {
            return true;
        }

        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        int32_t orderCount = int32_t(rContext.shader.variableOrder.size());
        for (int32_t i = orderCount - 1; i >= 0; i--) {
            uint32_t instructionIndex = rContext.shader.variableOrder[i];
            const Instruction &instruction = rContext.shader.instructions[instructionIndex];
            uint32_t resultId = optimizedWords[instruction.wordIndex + 2];
            if (resultId == UINT32_MAX) {
                // This variable has already been deleted.
                continue;
            }
            
            SpvStorageClass storageClass = SpvStorageClass(optimizedWords[instruction.wordIndex + 3]);
            if (storageClass != SpvStorageClassFunction) {
                // Only evaluate variables local to the function.
                continue;
            }

            thread_local std::vector<uint32_t> resultStack;
            thread_local std::vector<uint32_t> accessStack;
            thread_local std::vector<uint32_t> storeInstructionIndices;
            thread_local std::vector<uint32_t> partialLoadInstructionIndices;
            thread_local std::vector<uint32_t> fullLoadInstructionIndices;
            bool storeIsFull = true;
            resultStack.clear();
            accessStack.clear();
            storeInstructionIndices.clear();
            partialLoadInstructionIndices.clear();
            fullLoadInstructionIndices.clear();
            accessStack.emplace_back(instructionIndex);
            while (!accessStack.empty()) {
                uint32_t accessInstructionIndex = accessStack.back();
                const Instruction &accessInstruction = rContext.shader.instructions[accessInstructionIndex];
                accessStack.pop_back();

                if (rContext.instructionOutDegrees[accessInstructionIndex] > 0) {
                    uint32_t listIndex = rContext.instructionAdjacentListIndices[accessInstructionIndex];
                    while (listIndex != UINT32_MAX) {
                        uint32_t adjacentInstructionIndex = rContext.listNodes[listIndex].instructionIndex;
                        uint32_t adjacentWordIndex = rContext.shader.instructions[adjacentInstructionIndex].wordIndex;
                        listIndex = rContext.listNodes[listIndex].nextListIndex;

                        // Only check the instruction if it hasn't been deleted yet.
                        if (optimizedWords[adjacentWordIndex] != UINT32_MAX) {
                            SpvOp opCode = SpvOp(optimizedWords[adjacentWordIndex] & 0xFFFFU);
                            if (opCode == SpvOpAccessChain) {
                                accessStack.emplace_back(adjacentInstructionIndex);
                            }
                            else if (opCode == SpvOpStore) {
                                storeInstructionIndices.emplace_back(adjacentInstructionIndex);
                                storeIsFull = storeIsFull && (optimizedWords[adjacentWordIndex + 1] == resultId);
                            }
                            else if (opCode == SpvOpLoad) {
                                if (optimizedWords[adjacentWordIndex + 3] == resultId) {
                                    fullLoadInstructionIndices.emplace_back(adjacentInstructionIndex);
                                }
                                else {
                                    partialLoadInstructionIndices.emplace_back(adjacentInstructionIndex);
                                }
                            }
                            else {
                                // The whole search process is stopped if anything in the access chain is not recognized.
                                accessStack.clear();
                                storeInstructionIndices.clear();
                                fullLoadInstructionIndices.clear();
                                partialLoadInstructionIndices.clear();
                                listIndex = UINT32_MAX;
                            }
                        }
                    }
                }
                else {
                    resultStack.emplace_back(resultId);
                }
            }

            // Single store load elimination. Any variables that are only stored to once can eliminate any loads
            // and remap the results of the adjacent instructions. However, a strict requirement is that the block
            // that holds the store must dominate the block that holds the load as per SPIR-V rules.
            size_t fullLoadInstructionsEliminated = 0;
            if (!fullLoadInstructionIndices.empty() && (storeInstructionIndices.size() == 1) && storeIsFull) {
                uint32_t storeInstructionIndex = storeInstructionIndices.front();
                const Instruction &storeInstruction = rContext.shader.instructions[storeInstructionIndex];
                if (optimizedWords[storeInstruction.wordIndex] != UINT32_MAX) {
                    uint32_t storeResultId = optimizedWords[storeInstruction.wordIndex + 2];
                    uint32_t storeResultInstructionIndex = rContext.shader.results[storeResultId].instructionIndex;
                    for (uint32_t loadInstructionIndex : fullLoadInstructionIndices) {
                        const Instruction &loadInstruction = rContext.shader.instructions[loadInstructionIndex];
                        uint32_t loadWordIndex = loadInstruction.wordIndex;
                        if (optimizedWords[loadWordIndex] == UINT32_MAX) {
                            // Instruction has been deleted already.
                            continue;
                        }

                        if (!optimizerDoesInstructionDominate(rContext.shader, storeInstruction, loadInstruction)) {
                            // Store's block must dominate the load's block for the elimination to be possible.
                            continue;
                        }

                        uint32_t loadResultId = optimizedWords[loadWordIndex + 2];
                        uint32_t listIndex = rContext.instructionAdjacentListIndices[loadInstructionIndex];
                        while (listIndex != UINT32_MAX) {
                            uint32_t adjacentInstructionIndex = rContext.listNodes[listIndex].instructionIndex;
                            uint32_t adjacentWordIndex = rContext.shader.instructions[adjacentInstructionIndex].wordIndex;
                            if (optimizedWords[adjacentWordIndex] != UINT32_MAX) {
                                SpvOp adjacentOpCode = SpvOp(optimizedWords[adjacentWordIndex] & 0xFFFFU);
                                uint32_t adjancentWordCount = (optimizedWords[adjacentWordIndex] >> 16U) & 0xFFFFU;
                                uint32_t operandWordStart, operandWordCount, operandWordStride, operandWordSkip;
                                bool operandWordSkipString;
                                if (SpvHasOperands(adjacentOpCode, operandWordStart, operandWordCount, operandWordStride, operandWordSkip, operandWordSkipString, true)) {
                                    uint32_t operandWordIndex = operandWordStart;
                                    for (uint32_t j = 0; j < operandWordCount; j++) {
                                        if (checkOperandWordSkip(adjacentWordIndex, optimizedWords, j, operandWordSkip, operandWordSkipString, operandWordIndex)) {
                                            continue;
                                        }

                                        if (operandWordIndex >= adjancentWordCount) {
                                            break;
                                        }

                                        uint32_t shaderWordIndex = adjacentWordIndex + operandWordIndex;
                                        uint32_t &operandId = optimizedWords[shaderWordIndex];
                                        if (operandId == loadResultId) {
                                            operandId = storeResultId;
                                            resultStack.emplace_back(loadResultId);
                                            rContext.instructionAdjacentListIndices[storeResultInstructionIndex] = addToList(adjacentInstructionIndex, rContext.instructionAdjacentListIndices[storeResultInstructionIndex], rContext.listNodes);
                                            rContext.instructionOutDegrees[storeResultInstructionIndex]++;
                                        }

                                        operandWordIndex += operandWordStride;
                                    }
                                }
                            }

                            listIndex = rContext.listNodes[listIndex].nextListIndex;
                        }

                        fullLoadInstructionsEliminated++;
                    }
                }
            }
            
            if ((fullLoadInstructionIndices.size() == fullLoadInstructionsEliminated) && partialLoadInstructionIndices.empty()) {
                // Unused store elimination. Any variables which have no loads but have stores can be eliminated.
                for (uint32_t storeInstructionIndex : storeInstructionIndices) {
                    uint32_t storeWordIndex = rContext.shader.instructions[storeInstructionIndex].wordIndex;
                    if (optimizedWords[storeWordIndex] == UINT32_MAX) {
                        // Instruction has been deleted already.
                        continue;
                    }

                    resultStack.emplace_back(optimizedWords[storeWordIndex + 1]);
                    resultStack.emplace_back(optimizedWords[storeWordIndex + 2]);
                    optimizerEliminateInstruction(storeInstructionIndex, rContext);
                }
            }

            optimizerReduceResultDegrees(rContext, resultStack);
        }

        return true;
    }

    static bool optimizerRemoveUnusedDecorations(OptimizerContext &rContext) {
        if (!rContext.options.removeDeadCode) {
            return true;
        }

        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        for (Decoration decoration : rContext.shader.decorations) {
            uint32_t wordIndex = rContext.shader.instructions[decoration.instructionIndex].wordIndex;
            uint32_t resultId = optimizedWords[wordIndex + 1];
            if (resultId == UINT32_MAX) {
                // This decoration has already been deleted.
                continue;
            }

            uint32_t resultInstructionIndex = rContext.shader.results[resultId].instructionIndex;
            uint32_t resultWordIndex = rContext.shader.instructions[resultInstructionIndex].wordIndex;

            // The result has been deleted, so we delete the decoration as well.
            if (optimizedWords[resultWordIndex] == UINT32_MAX) {
                optimizerEliminateInstruction(decoration.instructionIndex, rContext);
            }
        }

        return true;
    }

    static bool optimizerCompactPhis(OptimizerContext &rContext) {
        if (!rContext.options.removeDeadCode) {
            return true;
        }

        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        for (Phi phi : rContext.shader.phis) {
            uint32_t wordIndex = rContext.shader.instructions[phi.instructionIndex].wordIndex;
            if (optimizedWords[wordIndex] == UINT32_MAX) {
                // This operation has already been deleted.
                continue;
            }

            if (!optimizerCompactPhi(phi.instructionIndex, rContext)) {
                return false;
            }
        }

        return true;
    }

    static bool optimizerCompactData(OptimizerContext &rContext) {
        uint32_t *optimizedWords = reinterpret_cast<uint32_t *>(rContext.optimizedData.data());
        uint32_t optimizedWordCount = 0;
        uint32_t instructionCount = rContext.shader.instructions.size();

        // Copy the header.
        const uint32_t startingWordIndex = 5;
        for (uint32_t i = 0; i < startingWordIndex; i++) {
            optimizedWords[optimizedWordCount++] = optimizedWords[i];
        }

        // Write out all the words for all the instructions and skip any that were marked as deleted.
        for (uint32_t i = 0; i < instructionCount; i++) {
            uint32_t wordIndex = rContext.shader.instructions[i].wordIndex;

            // Instruction has been deleted.
            if (optimizedWords[wordIndex] == UINT32_MAX) {
                continue;
            }

            // Check if the instruction should be ignored.
            SpvOp opCode = SpvOp(optimizedWords[wordIndex] & 0xFFFFU);
            if (rContext.options.removeDeadCode && SpvIsIgnored(opCode)) {
                continue;
            }

            // Copy all the words of the instruction.
            uint32_t wordCount = (optimizedWords[wordIndex] >> 16U) & 0xFFFFU;
            for (uint32_t j = 0; j < wordCount; j++) {
                optimizedWords[optimizedWordCount++] = optimizedWords[wordIndex + j];
            }
        }

        rContext.optimizedData.resize(optimizedWordCount * sizeof(uint32_t));

        return true;
    }

    bool Optimizer::run(const Shader &pShader, const SpecConstant *pNewSpecConstants, uint32_t pNewSpecConstantCount, std::vector<uint8_t> &pOptimizedData, Options pOptions) {
        thread_local std::vector<uint32_t> instructionAdjacentListIndices;
        thread_local std::vector<uint32_t> instructionInDegrees;
        thread_local std::vector<uint32_t> instructionOutDegrees;
        thread_local std::vector<ListNode> listNodes;
        thread_local std::vector<Resolution> resolutions;
        OptimizerContext context = { pShader, instructionAdjacentListIndices, instructionInDegrees, instructionOutDegrees, listNodes, resolutions, pOptimizedData, pOptions };
        if (!optimizerPrepareData(context)) {
            return false;
        }

        if (!optimizerPatchSpecializationConstants(pNewSpecConstants, pNewSpecConstantCount, context)) {
            return false;
        }

        if (!optimizerRunEvaluationPass(context)) {
            return false;
        }

        if (!optimizerRemoveUnusedVariables(context)) {
            return false;
        }

        if (!optimizerRemoveUnusedDecorations(context)) {
            return false;
        }

        // FIXME: For some reason, it seems that based on the order of the resolution, OpPhis can be compacted
        // before all their preceding blocks have been evaluated in time whether they should be deleted or not.
        // This pass merely re-runs the compaction step as a safeguard to remove any stale references. There's
        // potential for further optimization if this is fixed properly.
        if (!optimizerCompactPhis(context)) {
            return false;
        }

        if (!optimizerCompactData(context)) {
            return false;
        }

        return true;
    }
    }; //namespace respv
