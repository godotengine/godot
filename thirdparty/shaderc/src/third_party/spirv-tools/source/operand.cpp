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

#include "source/operand.h"

#include <assert.h>
#include <string.h>
#include <algorithm>

#include "source/macro.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"

// For now, assume unified1 contains up to SPIR-V 1.3 and no later
// SPIR-V version.
// TODO(dneto): Make one set of tables, but with version tags on a
// per-item basis. https://github.com/KhronosGroup/SPIRV-Tools/issues/1195

#include "operand.kinds-unified1.inc"

static const spv_operand_table_t kOperandTable = {
    ARRAY_SIZE(pygen_variable_OperandInfoTable),
    pygen_variable_OperandInfoTable};

spv_result_t spvOperandTableGet(spv_operand_table* pOperandTable,
                                spv_target_env) {
  if (!pOperandTable) return SPV_ERROR_INVALID_POINTER;

  *pOperandTable = &kOperandTable;
  return SPV_SUCCESS;
}

spv_result_t spvOperandTableNameLookup(spv_target_env env,
                                       const spv_operand_table table,
                                       const spv_operand_type_t type,
                                       const char* name,
                                       const size_t nameLength,
                                       spv_operand_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!name || !pEntry) return SPV_ERROR_INVALID_POINTER;

  for (uint64_t typeIndex = 0; typeIndex < table->count; ++typeIndex) {
    const auto& group = table->types[typeIndex];
    if (type != group.type) continue;
    for (uint64_t index = 0; index < group.count; ++index) {
      const auto& entry = group.entries[index];
      // We consider the current operand as available as long as
      // 1. The target environment satisfies the minimal requirement of the
      //    operand; or
      // 2. There is at least one extension enabling this operand; or
      // 3. There is at least one capability enabling this operand.
      //
      // Note that the second rule assumes the extension enabling this operand
      // is indeed requested in the SPIR-V code; checking that should be
      // validator's work.
      if ((spvVersionForTargetEnv(env) >= entry.minVersion ||
           entry.numExtensions > 0u || entry.numCapabilities > 0u) &&
          nameLength == strlen(entry.name) &&
          !strncmp(entry.name, name, nameLength)) {
        *pEntry = &entry;
        return SPV_SUCCESS;
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

spv_result_t spvOperandTableValueLookup(spv_target_env env,
                                        const spv_operand_table table,
                                        const spv_operand_type_t type,
                                        const uint32_t value,
                                        spv_operand_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  spv_operand_desc_t needle = {"", value, 0, nullptr, 0, nullptr, {}, ~0u};

  auto comp = [](const spv_operand_desc_t& lhs, const spv_operand_desc_t& rhs) {
    return lhs.value < rhs.value;
  };

  for (uint64_t typeIndex = 0; typeIndex < table->count; ++typeIndex) {
    const auto& group = table->types[typeIndex];
    if (type != group.type) continue;

    const auto beg = group.entries;
    const auto end = group.entries + group.count;

    // We need to loop here because there can exist multiple symbols for the
    // same operand value, and they can be introduced in different target
    // environments, which means they can have different minimal version
    // requirements. For example, SubgroupEqMaskKHR can exist in any SPIR-V
    // version as long as the SPV_KHR_shader_ballot extension is there; but
    // starting from SPIR-V 1.3, SubgroupEqMask, which has the same numeric
    // value as SubgroupEqMaskKHR, is available in core SPIR-V without extension
    // requirements.
    // Assumes the underlying table is already sorted ascendingly according to
    // opcode value.
    for (auto it = std::lower_bound(beg, end, needle, comp);
         it != end && it->value == value; ++it) {
      // We consider the current operand as available as long as
      // 1. The target environment satisfies the minimal requirement of the
      //    operand; or
      // 2. There is at least one extension enabling this operand; or
      // 3. There is at least one capability enabling this operand.
      //
      // Note that the second rule assumes the extension enabling this operand
      // is indeed requested in the SPIR-V code; checking that should be
      // validator's work.
      if (spvVersionForTargetEnv(env) >= it->minVersion ||
          it->numExtensions > 0u || it->numCapabilities > 0u) {
        *pEntry = it;
        return SPV_SUCCESS;
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

const char* spvOperandTypeStr(spv_operand_type_t type) {
  switch (type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
      return "ID";
    case SPV_OPERAND_TYPE_TYPE_ID:
      return "type ID";
    case SPV_OPERAND_TYPE_RESULT_ID:
      return "result ID";
    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_NUMBER:
      return "literal number";
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
      return "possibly multi-word literal integer";
    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
      return "possibly multi-word literal number";
    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER:
      return "extension instruction number";
    case SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER:
      return "OpSpecConstantOp opcode";
    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING:
      return "literal string";
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
      return "source language";
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
      return "execution model";
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
      return "addressing model";
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
      return "memory model";
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
      return "execution mode";
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
      return "storage class";
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
      return "dimensionality";
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
      return "sampler addressing mode";
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
      return "sampler filter mode";
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
      return "image format";
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
      return "floating-point fast math mode";
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
      return "floating-point rounding mode";
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
      return "linkage type";
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER:
      return "access qualifier";
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
      return "function parameter attribute";
    case SPV_OPERAND_TYPE_DECORATION:
      return "decoration";
    case SPV_OPERAND_TYPE_BUILT_IN:
      return "built-in";
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
      return "selection control";
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
      return "loop control";
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
      return "function control";
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      return "memory semantics ID";
    case SPV_OPERAND_TYPE_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
      return "memory access";
    case SPV_OPERAND_TYPE_SCOPE_ID:
      return "scope ID";
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
      return "group operation";
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
      return "kernel enqeue flags";
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO:
      return "kernel profiling info";
    case SPV_OPERAND_TYPE_CAPABILITY:
      return "capability";
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
      return "image";
    case SPV_OPERAND_TYPE_OPTIONAL_CIV:
      return "context-insensitive value";
    case SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS:
      return "debug info flags";
    case SPV_OPERAND_TYPE_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
      return "debug base type encoding";
    case SPV_OPERAND_TYPE_DEBUG_COMPOSITE_TYPE:
      return "debug composite type";
    case SPV_OPERAND_TYPE_DEBUG_TYPE_QUALIFIER:
      return "debug type qualifier";
    case SPV_OPERAND_TYPE_DEBUG_OPERATION:
      return "debug operation";

    // The next values are for values returned from an instruction, not actually
    // an operand.  So the specific strings don't matter.  But let's add them
    // for completeness and ease of testing.
    case SPV_OPERAND_TYPE_IMAGE_CHANNEL_ORDER:
      return "image channel order";
    case SPV_OPERAND_TYPE_IMAGE_CHANNEL_DATA_TYPE:
      return "image channel data type";

    case SPV_OPERAND_TYPE_NONE:
      return "NONE";
    default:
      assert(0 && "Unhandled operand type!");
      break;
  }
  return "unknown";
}

void spvPushOperandTypes(const spv_operand_type_t* types,
                         spv_operand_pattern_t* pattern) {
  const spv_operand_type_t* endTypes;
  for (endTypes = types; *endTypes != SPV_OPERAND_TYPE_NONE; ++endTypes) {
  }

  while (endTypes-- != types) {
    pattern->push_back(*endTypes);
  }
}

void spvPushOperandTypesForMask(spv_target_env env,
                                const spv_operand_table operandTable,
                                const spv_operand_type_t type,
                                const uint32_t mask,
                                spv_operand_pattern_t* pattern) {
  // Scan from highest bits to lowest bits because we will append in LIFO
  // fashion, and we need the operands for lower order bits to be consumed first
  for (uint32_t candidate_bit = (1u << 31u); candidate_bit;
       candidate_bit >>= 1) {
    if (candidate_bit & mask) {
      spv_operand_desc entry = nullptr;
      if (SPV_SUCCESS == spvOperandTableValueLookup(env, operandTable, type,
                                                    candidate_bit, &entry)) {
        spvPushOperandTypes(entry->operandTypes, pattern);
      }
    }
  }
}

bool spvOperandIsConcrete(spv_operand_type_t type) {
  if (spvIsIdType(type) || spvOperandIsConcreteMask(type)) {
    return true;
  }
  switch (type) {
    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER:
    case SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER:
    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
    case SPV_OPERAND_TYPE_IMAGE_CHANNEL_ORDER:
    case SPV_OPERAND_TYPE_IMAGE_CHANNEL_DATA_TYPE:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO:
    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
    case SPV_OPERAND_TYPE_DEBUG_COMPOSITE_TYPE:
    case SPV_OPERAND_TYPE_DEBUG_TYPE_QUALIFIER:
    case SPV_OPERAND_TYPE_DEBUG_OPERATION:
      return true;
    default:
      break;
  }
  return false;
}

bool spvOperandIsConcreteMask(spv_operand_type_t type) {
  switch (type) {
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS:
      return true;
    default:
      break;
  }
  return false;
}

bool spvOperandIsOptional(spv_operand_type_t type) {
  return SPV_OPERAND_TYPE_FIRST_OPTIONAL_TYPE <= type &&
         type <= SPV_OPERAND_TYPE_LAST_OPTIONAL_TYPE;
}

bool spvOperandIsVariable(spv_operand_type_t type) {
  return SPV_OPERAND_TYPE_FIRST_VARIABLE_TYPE <= type &&
         type <= SPV_OPERAND_TYPE_LAST_VARIABLE_TYPE;
}

bool spvExpandOperandSequenceOnce(spv_operand_type_t type,
                                  spv_operand_pattern_t* pattern) {
  switch (type) {
    case SPV_OPERAND_TYPE_VARIABLE_ID:
      pattern->push_back(type);
      pattern->push_back(SPV_OPERAND_TYPE_OPTIONAL_ID);
      return true;
    case SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER:
      pattern->push_back(type);
      pattern->push_back(SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER);
      return true;
    case SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER_ID:
      // Represents Zero or more (Literal number, Id) pairs,
      // where the literal number must be a scalar integer.
      pattern->push_back(type);
      pattern->push_back(SPV_OPERAND_TYPE_ID);
      pattern->push_back(SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER);
      return true;
    case SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL_INTEGER:
      // Represents Zero or more (Id, Literal number) pairs.
      pattern->push_back(type);
      pattern->push_back(SPV_OPERAND_TYPE_LITERAL_INTEGER);
      pattern->push_back(SPV_OPERAND_TYPE_OPTIONAL_ID);
      return true;
    default:
      break;
  }
  return false;
}

spv_operand_type_t spvTakeFirstMatchableOperand(
    spv_operand_pattern_t* pattern) {
  assert(!pattern->empty());
  spv_operand_type_t result;
  do {
    result = pattern->back();
    pattern->pop_back();
  } while (spvExpandOperandSequenceOnce(result, pattern));
  return result;
}

spv_operand_pattern_t spvAlternatePatternFollowingImmediate(
    const spv_operand_pattern_t& pattern) {
  auto it =
      std::find(pattern.crbegin(), pattern.crend(), SPV_OPERAND_TYPE_RESULT_ID);
  if (it != pattern.crend()) {
    spv_operand_pattern_t alternatePattern(it - pattern.crbegin() + 2,
                                           SPV_OPERAND_TYPE_OPTIONAL_CIV);
    alternatePattern[1] = SPV_OPERAND_TYPE_RESULT_ID;
    return alternatePattern;
  }

  // No result-id found, so just expect CIVs.
  return {SPV_OPERAND_TYPE_OPTIONAL_CIV};
}

bool spvIsIdType(spv_operand_type_t type) {
  switch (type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_RESULT_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
      return true;
    default:
      return false;
  }
}

bool spvIsInIdType(spv_operand_type_t type) {
  if (!spvIsIdType(type)) {
    // If it is not an ID it cannot be an input ID.
    return false;
  }
  switch (type) {
    // Blacklist non-input IDs.
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_RESULT_ID:
      return false;
    default:
      return true;
  }
}

std::function<bool(unsigned)> spvOperandCanBeForwardDeclaredFunction(
    SpvOp opcode) {
  std::function<bool(unsigned index)> out;
  switch (opcode) {
    case SpvOpExecutionMode:
    case SpvOpExecutionModeId:
    case SpvOpEntryPoint:
    case SpvOpName:
    case SpvOpMemberName:
    case SpvOpSelectionMerge:
    case SpvOpDecorate:
    case SpvOpMemberDecorate:
    case SpvOpDecorateId:
    case SpvOpDecorateStringGOOGLE:
    case SpvOpMemberDecorateStringGOOGLE:
    case SpvOpTypeStruct:
    case SpvOpBranch:
    case SpvOpLoopMerge:
      out = [](unsigned) { return true; };
      break;
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
    case SpvOpBranchConditional:
    case SpvOpSwitch:
      out = [](unsigned index) { return index != 0; };
      break;

    case SpvOpFunctionCall:
      // The Function parameter.
      out = [](unsigned index) { return index == 2; };
      break;

    case SpvOpPhi:
      out = [](unsigned index) { return index > 1; };
      break;

    case SpvOpEnqueueKernel:
      // The Invoke parameter.
      out = [](unsigned index) { return index == 8; };
      break;

    case SpvOpGetKernelNDrangeSubGroupCount:
    case SpvOpGetKernelNDrangeMaxSubGroupSize:
      // The Invoke parameter.
      out = [](unsigned index) { return index == 3; };
      break;

    case SpvOpGetKernelWorkGroupSize:
    case SpvOpGetKernelPreferredWorkGroupSizeMultiple:
      // The Invoke parameter.
      out = [](unsigned index) { return index == 2; };
      break;
    case SpvOpTypeForwardPointer:
      out = [](unsigned index) { return index == 0; };
      break;
    case SpvOpTypeArray:
      out = [](unsigned index) { return index == 1; };
      break;
    default:
      out = [](unsigned) { return false; };
      break;
  }
  return out;
}
