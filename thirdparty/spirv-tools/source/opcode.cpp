// Copyright (c) 2015-2022 The Khronos Group Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights
// reserved.
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

// Represents a vendor tool entry in the SPIR-V XML Registry.
struct VendorTool {
  uint32_t value;
  const char* vendor;
  const char* tool;         // Might be empty string.
  const char* vendor_tool;  // Combination of vendor and tool.
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

uint32_t spvOpcodeMake(uint16_t wordCount, spv::Op opcode) {
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
  const auto version = spvVersionForTargetEnv(env);
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
    if (((version >= entry.minVersion && version <= entry.lastVersion) ||
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
                                       const spv::Op opcode,
                                       spv_opcode_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  const auto beg = table->entries;
  const auto end = table->entries + table->count;

  spv_opcode_desc_t needle = {"",    opcode, 0, nullptr, 0,   {},
                              false, false,  0, nullptr, ~0u, ~0u};

  auto comp = [](const spv_opcode_desc_t& lhs, const spv_opcode_desc_t& rhs) {
    return lhs.opcode < rhs.opcode;
  };

  // We need to loop here because there can exist multiple symbols for the same
  // opcode value, and they can be introduced in different target environments,
  // which means they can have different minimal version requirements.
  // Assumes the underlying table is already sorted ascendingly according to
  // opcode value.
  const auto version = spvVersionForTargetEnv(env);
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
    if ((version >= it->minVersion && version <= it->lastVersion) ||
        it->numExtensions > 0u || it->numCapabilities > 0u) {
      *pEntry = it;
      return SPV_SUCCESS;
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

void spvInstructionCopy(const uint32_t* words, const spv::Op opcode,
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
      assert(opcode == static_cast<spv::Op>(thisOpcode) &&
             wordCount == thisWordCount && "Endianness failed!");
    }
  }
}

const char* spvOpcodeString(const uint32_t opcode) {
  const auto beg = kOpcodeTableEntries;
  const auto end = kOpcodeTableEntries + ARRAY_SIZE(kOpcodeTableEntries);
  spv_opcode_desc_t needle = {"",    static_cast<spv::Op>(opcode),
                              0,     nullptr,
                              0,     {},
                              false, false,
                              0,     nullptr,
                              ~0u,   ~0u};
  auto comp = [](const spv_opcode_desc_t& lhs, const spv_opcode_desc_t& rhs) {
    return lhs.opcode < rhs.opcode;
  };
  auto it = std::lower_bound(beg, end, needle, comp);
  if (it != end && it->opcode == spv::Op(opcode)) {
    return it->name;
  }

  assert(0 && "Unreachable!");
  return "unknown";
}

const char* spvOpcodeString(const spv::Op opcode) {
  return spvOpcodeString(static_cast<uint32_t>(opcode));
}

int32_t spvOpcodeIsScalarType(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeBool:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsSpecConstant(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsConstant(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantSampler:
    case spv::Op::OpConstantNull:
    case spv::Op::OpConstantFunctionPointerINTEL:
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsConstantOrUndef(const spv::Op opcode) {
  return opcode == spv::Op::OpUndef || spvOpcodeIsConstant(opcode);
}

bool spvOpcodeIsScalarSpecConstant(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsComposite(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeCooperativeMatrixNV:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeReturnsLogicalVariablePointer(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpVariable:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpCopyObject:
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpLoad:
    case spv::Op::OpConstantNull:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeReturnsLogicalPointer(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpVariable:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpCopyObject:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeGeneratesType(spv::Op op) {
  switch (op) {
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeOpaque:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeEvent:
    case spv::Op::OpTypeDeviceEvent:
    case spv::Op::OpTypeReserveId:
    case spv::Op::OpTypeQueue:
    case spv::Op::OpTypePipe:
    case spv::Op::OpTypePipeStorage:
    case spv::Op::OpTypeNamedBarrier:
    case spv::Op::OpTypeAccelerationStructureNV:
    case spv::Op::OpTypeCooperativeMatrixNV:
    // case spv::Op::OpTypeAccelerationStructureKHR: covered by
    // spv::Op::OpTypeAccelerationStructureNV
    case spv::Op::OpTypeRayQueryKHR:
    case spv::Op::OpTypeHitObjectNV:
      return true;
    default:
      // In particular, OpTypeForwardPointer does not generate a type,
      // but declares a storage class for a pointer type generated
      // by a different instruction.
      break;
  }
  return 0;
}

bool spvOpcodeIsDecoration(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpMemberDecorate:
    case spv::Op::OpGroupDecorate:
    case spv::Op::OpGroupMemberDecorate:
    case spv::Op::OpDecorateStringGOOGLE:
    case spv::Op::OpMemberDecorateStringGOOGLE:
      return true;
    default:
      break;
  }
  return false;
}

bool spvOpcodeIsLoad(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpLoad:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageFetch:
    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImageRead:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseFetch:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
    case spv::Op::OpImageSparseRead:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsBranch(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpBranch:
    case spv::Op::OpBranchConditional:
    case spv::Op::OpSwitch:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsAtomicWithLoad(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicFAddEXT:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicFMinEXT:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicFMaxEXT:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
    case spv::Op::OpAtomicFlagTestAndSet:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsAtomicOp(const spv::Op opcode) {
  return (spvOpcodeIsAtomicWithLoad(opcode) ||
          opcode == spv::Op::OpAtomicStore ||
          opcode == spv::Op::OpAtomicFlagClear);
}

bool spvOpcodeIsReturn(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpReturn:
    case spv::Op::OpReturnValue:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsAbort(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpKill:
    case spv::Op::OpUnreachable:
    case spv::Op::OpTerminateInvocation:
    case spv::Op::OpTerminateRayKHR:
    case spv::Op::OpIgnoreIntersectionKHR:
    case spv::Op::OpEmitMeshTasksEXT:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsReturnOrAbort(spv::Op opcode) {
  return spvOpcodeIsReturn(opcode) || spvOpcodeIsAbort(opcode);
}

bool spvOpcodeIsBlockTerminator(spv::Op opcode) {
  return spvOpcodeIsBranch(opcode) || spvOpcodeIsReturnOrAbort(opcode);
}

bool spvOpcodeIsBaseOpaqueType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeOpaque:
    case spv::Op::OpTypeEvent:
    case spv::Op::OpTypeDeviceEvent:
    case spv::Op::OpTypeReserveId:
    case spv::Op::OpTypeQueue:
    case spv::Op::OpTypePipe:
    case spv::Op::OpTypeForwardPointer:
    case spv::Op::OpTypePipeStorage:
    case spv::Op::OpTypeNamedBarrier:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsNonUniformGroupOperation(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpGroupNonUniformElect:
    case spv::Op::OpGroupNonUniformAll:
    case spv::Op::OpGroupNonUniformAny:
    case spv::Op::OpGroupNonUniformAllEqual:
    case spv::Op::OpGroupNonUniformBroadcast:
    case spv::Op::OpGroupNonUniformBroadcastFirst:
    case spv::Op::OpGroupNonUniformBallot:
    case spv::Op::OpGroupNonUniformInverseBallot:
    case spv::Op::OpGroupNonUniformBallotBitExtract:
    case spv::Op::OpGroupNonUniformBallotBitCount:
    case spv::Op::OpGroupNonUniformBallotFindLSB:
    case spv::Op::OpGroupNonUniformBallotFindMSB:
    case spv::Op::OpGroupNonUniformShuffle:
    case spv::Op::OpGroupNonUniformShuffleXor:
    case spv::Op::OpGroupNonUniformShuffleUp:
    case spv::Op::OpGroupNonUniformShuffleDown:
    case spv::Op::OpGroupNonUniformIAdd:
    case spv::Op::OpGroupNonUniformFAdd:
    case spv::Op::OpGroupNonUniformIMul:
    case spv::Op::OpGroupNonUniformFMul:
    case spv::Op::OpGroupNonUniformSMin:
    case spv::Op::OpGroupNonUniformUMin:
    case spv::Op::OpGroupNonUniformFMin:
    case spv::Op::OpGroupNonUniformSMax:
    case spv::Op::OpGroupNonUniformUMax:
    case spv::Op::OpGroupNonUniformFMax:
    case spv::Op::OpGroupNonUniformBitwiseAnd:
    case spv::Op::OpGroupNonUniformBitwiseOr:
    case spv::Op::OpGroupNonUniformBitwiseXor:
    case spv::Op::OpGroupNonUniformLogicalAnd:
    case spv::Op::OpGroupNonUniformLogicalOr:
    case spv::Op::OpGroupNonUniformLogicalXor:
    case spv::Op::OpGroupNonUniformQuadBroadcast:
    case spv::Op::OpGroupNonUniformQuadSwap:
    case spv::Op::OpGroupNonUniformRotateKHR:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsScalarizable(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpPhi:
    case spv::Op::OpCopyObject:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpQuantizeToF16:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpUMod:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpFRem:
    case spv::Op::OpFMod:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpIAddCarry:
    case spv::Op::OpISubBorrow:
    case spv::Op::OpUMulExtended:
    case spv::Op::OpSMulExtended:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
    case spv::Op::OpBitFieldInsert:
    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
    case spv::Op::OpIsNan:
    case spv::Op::OpIsInf:
    case spv::Op::OpIsFinite:
    case spv::Op::OpIsNormal:
    case spv::Op::OpSignBitSet:
    case spv::Op::OpLessOrGreater:
    case spv::Op::OpOrdered:
    case spv::Op::OpUnordered:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot:
    case spv::Op::OpSelect:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFUnordNotEqual:
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFUnordLessThan:
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFUnordGreaterThan:
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFUnordLessThanEqual:
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsDebug(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpName:
    case spv::Op::OpMemberName:
    case spv::Op::OpSource:
    case spv::Op::OpSourceContinued:
    case spv::Op::OpSourceExtension:
    case spv::Op::OpString:
    case spv::Op::OpLine:
    case spv::Op::OpNoLine:
    case spv::Op::OpModuleProcessed:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsCommutativeBinaryOperator(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpDot:
    case spv::Op::OpIAddCarry:
    case spv::Op::OpUMulExtended:
    case spv::Op::OpSMulExtended:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpOrdered:
    case spv::Op::OpUnordered:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFUnordNotEqual:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsLinearAlgebra(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpTranspose:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpMatrixTimesScalar:
    case spv::Op::OpVectorTimesMatrix:
    case spv::Op::OpMatrixTimesVector:
    case spv::Op::OpMatrixTimesMatrix:
    case spv::Op::OpOuterProduct:
    case spv::Op::OpDot:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsImageSample(const spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
      return true;
    default:
      return false;
  }
}

std::vector<uint32_t> spvOpcodeMemorySemanticsOperandIndices(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpMemoryBarrier:
      return {1};
    case spv::Op::OpAtomicStore:
    case spv::Op::OpControlBarrier:
    case spv::Op::OpAtomicFlagClear:
    case spv::Op::OpMemoryNamedBarrier:
      return {2};
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicFAddEXT:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
    case spv::Op::OpAtomicFlagTestAndSet:
      return {4};
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
      return {4, 5};
    default:
      return {};
  }
}

bool spvOpcodeIsAccessChain(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
      return true;
    default:
      return false;
  }
}

bool spvOpcodeIsBit(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
      return true;
    default:
      return false;
  }
}
