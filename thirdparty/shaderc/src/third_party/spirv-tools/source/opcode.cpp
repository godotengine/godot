// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/opcode.h"

#include <assert.h>
#include <string.h>

#include <algorithm>
#include <cstdlib>

#include "source/instruction.h"
#include "source/macro.h"
#include "source/spirv_constant.h"
#include "source/spirv_endian.h"
#include "source/spirv_target_env.h"
#include "spirv-tools/libspirv.h"

namespace {
struct OpcodeDescPtrLen {
  const spv_opcode_desc_t* ptr;
  uint32_t len;
};

#include "core.insts-unified1.inc"

static const spv_opcode_table_t kOpcodeTable = {ARRAY_SIZE(kOpcodeTableEntries),
                                                kOpcodeTableEntries};

// Represents a vendor tool entry in the SPIR-V XML Regsitry.
struct VendorTool {
  uint32_t value;
  const char* vendor;
  const char* tool;         // Might be empty string.
  const char* vendor_tool;  // Combiantion of vendor and tool.
};

const VendorTool vendor_tools[] = {
#include "generators.inc"
};

}  // anonymous namespace

// TODO(dneto): Move this to another file.  It doesn't belong with opcode
// processing.
const char* spvGeneratorStr(uint32_t generator) {
  auto where = std::find_if(
      std::begin(vendor_tools), std::end(vendor_tools),
      [generator](const VendorTool& vt) { return generator == vt.value; });
  if (where != std::end(vendor_tools)) return where->vendor_tool;
  return "Unknown";
}

uint32_t spvOpcodeMake(uint16_t wordCount, SpvOp opcode) {
  return ((uint32_t)opcode) | (((uint32_t)wordCount) << 16);
}

void spvOpcodeSplit(const uint32_t word, uint16_t* pWordCount,
                    uint16_t* pOpcode) {
  if (pWordCount) {
    *pWordCount = (uint16_t)((0xffff0000 & word) >> 16);
  }
  if (pOpcode) {
    *pOpcode = 0x0000ffff & word;
  }
}

spv_result_t spvOpcodeTableGet(spv_opcode_table* pInstTable, spv_target_env) {
  if (!pInstTable) return SPV_ERROR_INVALID_POINTER;

  // Descriptions of each opcode.  Each entry describes the format of the
  // instruction that follows a particular opcode.

  *pInstTable = &kOpcodeTable;
  return SPV_SUCCESS;
}

spv_result_t spvOpcodeTableNameLookup(spv_target_env env,
                                      const spv_opcode_table table,
                                      const char* name,
                                      spv_opcode_desc* pEntry) {
  if (!name || !pEntry) return SPV_ERROR_INVALID_POINTER;
  if (!table) return SPV_ERROR_INVALID_TABLE;

  // TODO: This lookup of the Opcode table is suboptimal! Binary sort would be
  // preferable but the table requires sorting on the Opcode name, but it's
  // static const initialized and matches the order of the spec.
  const size_t nameLength = strlen(name);
  for (uint64_t opcodeIndex = 0; opcodeIndex < table->count; ++opcodeIndex) {
    const spv_opcode_desc_t& entry = table->entries[opcodeIndex];
    // We considers the current opcode as available as long as
    // 1. The target environment satisfies the minimal requirement of the
    //    opcode; or
    // 2. There is at least one extension enabling this opcode.
    //
    // Note that the second rule assumes the extension enabling this instruction
    // is indeed requested in the SPIR-V code; checking that should be
    // validator's work.
    if ((spvVersionForTargetEnv(env) >= entry.minVersion ||
         entry.numExtensions > 0u || entry.numCapabilities > 0u) &&
        nameLength == strlen(entry.name) &&
        !strncmp(name, entry.name, nameLength)) {
      // NOTE: Found out Opcode!
      *pEntry = &entry;
      return SPV_SUCCESS;
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

spv_result_t spvOpcodeTableValueLookup(spv_target_env env,
                                       const spv_opcode_table table,
                                       const SpvOp opcode,
                                       spv_opcode_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  const auto beg = table->entries;
  const auto end = table->entries + table->count;

  spv_opcode_desc_t needle = {"",    opcode, 0, nullptr, 0,  {},
                              false, false,  0, nullptr, ~0u};

  auto comp = [](const spv_opcode_desc_t& lhs, const spv_opcode_desc_t& rhs) {
    return lhs.opcode < rhs.opcode;
  };

  // We need to loop here because there can exist multiple symbols for the same
  // opcode value, and they can be introduced in different target environments,
  // which means they can have different minimal version requirements.
  // Assumes the underlying table is already sorted ascendingly according to
  // opcode value.
  for (auto it = std::lower_bound(beg, end, needle, comp);
       it != end && it->opcode == opcode; ++it) {
    // We considers the current opcode as available as long as
    // 1. The target environment satisfies the minimal requirement of the
    //    opcode; or
    // 2. There is at least one extension enabling this opcode.
    //
    // Note that the second rule assumes the extension enabling this instruction
    // is indeed requested in the SPIR-V code; checking that should be
    // validator's work.
    if (spvVersionForTargetEnv(env) >= it->minVersion ||
        it->numExtensions > 0u || it->numCapabilities > 0u) {
      *pEntry = it;
      return SPV_SUCCESS;
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

void spvInstructionCopy(const uint32_t* words, const SpvOp opcode,
                        const uint16_t wordCount, const spv_endianness_t endian,
                        spv_instruction_t* pInst) {
  pInst->opcode = opcode;
  pInst->words.resize(wordCount);
  for (uint16_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    pInst->words[wordIndex] = spvFixWord(words[wordIndex], endian);
    if (!wordIndex) {
      uint16_t thisWordCount;
      uint16_t thisOpcode;
      spvOpcodeSplit(pInst->words[wordIndex], &thisWordCount, &thisOpcode);
      assert(opcode == static_cast<SpvOp>(thisOpcode) &&
             wordCount == thisWordCount && "Endianness failed!");
    }
  }
}

const char* spvOpcodeString(const SpvOp opcode) {
  const auto beg = kOpcodeTableEntries;
  const auto end = kOpcodeTableEntries + ARRAY_SIZE(kOpcodeTableEntries);
  spv_opcode_desc_t needle = {"",    opcode, 0, nullptr, 0,  {},
                              false, false,  0, nullptr, ~0u};
  auto comp = [](const spv_opcode_desc_t& lhs, const spv_opcode_desc_t& rhs) {
    return lhs.opcode < rhs.opcode;
  };
  auto it = std::lower_bound(beg, end, needle, comp);
  if (it != end && it->opcode == opcode) {
    return it->name;
  }

  assert(0 && "Unreachable!");
  return "unknown";
}

int32_t spvOpcodeIsScalarType(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeBool:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsSpecConstant(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpSpecConstantTrue:
    case SpvOpSpecConstantFalse:
    case SpvOpSpecConstant:
    case SpvOpSpecConstantComposite:
    case SpvOpSpecConstantOp:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsConstant(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpConstantTrue:
    case SpvOpConstantFalse:
    case SpvOpConstant:
    case SpvOpConstantComposite:
    case SpvOpConstantSampler:
    case SpvOpConstantNull:
    case SpvOpSpecConstantTrue:
    case SpvOpSpecConstantFalse:
    case SpvOpSpecConstant:
    case SpvOpSpecConstantComposite:
    case SpvOpSpecConstantOp:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsConstantOrUndef(const SpvOp opcode) {
  return opcode == SpvOpUndef || spvOpcodeIsConstant(opcode);
}

bool spvOpcodeIsScalarSpecConstant(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpSpecConstantTrue:
    case SpvOpSpecConstantFalse:
    case SpvOpSpecConstant:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsComposite(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeArray:
    case SpvOpTypeStruct:
    case SpvOpTypeCooperativeMatrixNV:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeReturnsLogicalVariablePointer(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpVariable:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpFunctionParameter:
    case SpvOpImageTexelPointer:
    case SpvOpCopyObject:
    case SpvOpSelect:
    case SpvOpPhi:
    case SpvOpFunctionCall:
    case SpvOpPtrAccessChain:
    case SpvOpLoad:
    case SpvOpConstantNull:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeReturnsLogicalPointer(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpVariable:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpFunctionParameter:
    case SpvOpImageTexelPointer:
    case SpvOpCopyObject:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeGeneratesType(SpvOp op) {
  switch (op) {
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
    case SpvOpTypeOpaque:
    case SpvOpTypePointer:
    case SpvOpTypeFunction:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipe:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
    case SpvOpTypeAccelerationStructureNV:
    case SpvOpTypeCooperativeMatrixNV:
      return true;
    default:
      // In particular, OpTypeForwardPointer does not generate a type,
      // but declares a storage class for a pointer type generated
      // by a different instruction.
      break;
  }
  return 0;
}

bool spvOpcodeIsDecoration(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpDecorate:
    case SpvOpDecorateId:
    case SpvOpMemberDecorate:
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
    case SpvOpDecorateStringGOOGLE:
    case SpvOpMemberDecorateStringGOOGLE:
      return true;
    default:
      break;
  }
  return false;
}

bool spvOpcodeIsLoad(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpLoad:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleImplicitLod:
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
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseFetch:
    case SpvOpImageSparseGather:
    case SpvOpImageSparseDrefGather:
    case SpvOpImageSparseRead:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsBranch(SpvOp opcode) {
  switch (opcode) {
    case SpvOpBranch:
    case SpvOpBranchConditional:
    case SpvOpSwitch:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsAtomicWithLoad(const SpvOp opcode) {
  switch (opcode) {
    case SpvOpAtomicLoad:
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
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsAtomicOp(const SpvOp opcode) {
  return (spvOpcodeIsAtomicWithLoad(opcode) || opcode == SpvOpAtomicStore ||
          opcode == SpvOpAtomicFlagClear);
}

bool spvOpcodeIsReturn(SpvOp opcode) {
  switch (opcode) {
    case SpvOpReturn:
    case SpvOpReturnValue:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsReturnOrAbort(SpvOp opcode) {
  return spvOpcodeIsReturn(opcode) || opcode == SpvOpKill ||
         opcode == SpvOpUnreachable;
}

bool spvOpcodeIsBlockTerminator(SpvOp opcode) {
  return spvOpcodeIsBranch(opcode) || spvOpcodeIsReturnOrAbort(opcode);
}

bool spvOpcodeIsBaseOpaqueType(SpvOp opcode) {
  switch (opcode) {
    case SpvOpTypeImage:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeOpaque:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsNonUniformGroupOperation(SpvOp opcode) {
  switch (opcode) {
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
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsScalarizable(SpvOp opcode) {
  switch (opcode) {
    case SpvOpPhi:
    case SpvOpCopyObject:
    case SpvOpConvertFToU:
    case SpvOpConvertFToS:
    case SpvOpConvertSToF:
    case SpvOpConvertUToF:
    case SpvOpUConvert:
    case SpvOpSConvert:
    case SpvOpFConvert:
    case SpvOpQuantizeToF16:
    case SpvOpVectorInsertDynamic:
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
    case SpvOpIAddCarry:
    case SpvOpISubBorrow:
    case SpvOpUMulExtended:
    case SpvOpSMulExtended:
    case SpvOpShiftRightLogical:
    case SpvOpShiftRightArithmetic:
    case SpvOpShiftLeftLogical:
    case SpvOpBitwiseOr:
    case SpvOpBitwiseAnd:
    case SpvOpNot:
    case SpvOpBitFieldInsert:
    case SpvOpBitFieldSExtract:
    case SpvOpBitFieldUExtract:
    case SpvOpBitReverse:
    case SpvOpBitCount:
    case SpvOpIsNan:
    case SpvOpIsInf:
    case SpvOpIsFinite:
    case SpvOpIsNormal:
    case SpvOpSignBitSet:
    case SpvOpLessOrGreater:
    case SpvOpOrdered:
    case SpvOpUnordered:
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
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsDebug(SpvOp opcode) {
  switch (opcode) {
    case SpvOpName:
    case SpvOpMemberName:
    case SpvOpSource:
    case SpvOpSourceContinued:
    case SpvOpSourceExtension:
    case SpvOpString:
    case SpvOpLine:
    case SpvOpNoLine:
      return true;
    default:
      return false;
  }
}

std::vector<uint32_t> spvOpcodeMemorySemanticsOperandIndices(SpvOp opcode) {
  switch (opcode) {
    case SpvOpMemoryBarrier:
      return {1};
    case SpvOpAtomicStore:
    case SpvOpControlBarrier:
    case SpvOpAtomicFlagClear:
    case SpvOpMemoryNamedBarrier:
      return {2};
    case SpvOpAtomicLoad:
    case SpvOpAtomicExchange:
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
      return {4};
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
      return {4, 5};
    default:
      return {};
  }
}
