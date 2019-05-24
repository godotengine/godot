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

#include "source/assembly_grammar.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "source/ext_inst.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/table.h"

namespace spvtools {
namespace {

/// @brief Parses a mask expression string for the given operand type.
///
/// A mask expression is a sequence of one or more terms separated by '|',
/// where each term a named enum value for the given type.  No whitespace
/// is permitted.
///
/// On success, the value is written to pValue.
///
/// @param[in] operandTable operand lookup table
/// @param[in] type of the operand
/// @param[in] textValue word of text to be parsed
/// @param[out] pValue where the resulting value is written
///
/// @return result code
spv_result_t spvTextParseMaskOperand(spv_target_env env,
                                     const spv_operand_table operandTable,
                                     const spv_operand_type_t type,
                                     const char* textValue, uint32_t* pValue) {
  if (textValue == nullptr) return SPV_ERROR_INVALID_TEXT;
  size_t text_length = strlen(textValue);
  if (text_length == 0) return SPV_ERROR_INVALID_TEXT;
  const char* text_end = textValue + text_length;

  // We only support mask expressions in ASCII, so the separator value is a
  // char.
  const char separator = '|';

  // Accumulate the result by interpreting one word at a time, scanning
  // from left to right.
  uint32_t value = 0;
  const char* begin = textValue;  // The left end of the current word.
  const char* end = nullptr;  // One character past the end of the current word.
  do {
    end = std::find(begin, text_end, separator);

    spv_operand_desc entry = nullptr;
    if (spvOperandTableNameLookup(env, operandTable, type, begin, end - begin,
                                  &entry)) {
      return SPV_ERROR_INVALID_TEXT;
    }
    value |= entry->value;

    // Advance to the next word by skipping over the separator.
    begin = end + 1;
  } while (end != text_end);

  *pValue = value;
  return SPV_SUCCESS;
}

// Associates an opcode with its name.
struct SpecConstantOpcodeEntry {
  SpvOp opcode;
  const char* name;
};

// All the opcodes allowed as the operation for OpSpecConstantOp.
// The name does not have the usual "Op" prefix. For example opcode SpvOpIAdd
// is associated with the name "IAdd".
//
// clang-format off
#define CASE(NAME) { SpvOp##NAME, #NAME }
const SpecConstantOpcodeEntry kOpSpecConstantOpcodes[] = {
    // Conversion
    CASE(SConvert),
    CASE(FConvert),
    CASE(ConvertFToS),
    CASE(ConvertSToF),
    CASE(ConvertFToU),
    CASE(ConvertUToF),
    CASE(UConvert),
    CASE(ConvertPtrToU),
    CASE(ConvertUToPtr),
    CASE(GenericCastToPtr),
    CASE(PtrCastToGeneric),
    CASE(Bitcast),
    CASE(QuantizeToF16),
    // Arithmetic
    CASE(SNegate),
    CASE(Not),
    CASE(IAdd),
    CASE(ISub),
    CASE(IMul),
    CASE(UDiv),
    CASE(SDiv),
    CASE(UMod),
    CASE(SRem),
    CASE(SMod),
    CASE(ShiftRightLogical),
    CASE(ShiftRightArithmetic),
    CASE(ShiftLeftLogical),
    CASE(BitwiseOr),
    CASE(BitwiseAnd),
    CASE(BitwiseXor),
    CASE(FNegate),
    CASE(FAdd),
    CASE(FSub),
    CASE(FMul),
    CASE(FDiv),
    CASE(FRem),
    CASE(FMod),
    // Composite
    CASE(VectorShuffle),
    CASE(CompositeExtract),
    CASE(CompositeInsert),
    // Logical
    CASE(LogicalOr),
    CASE(LogicalAnd),
    CASE(LogicalNot),
    CASE(LogicalEqual),
    CASE(LogicalNotEqual),
    CASE(Select),
    // Comparison
    CASE(IEqual),
    CASE(INotEqual),
    CASE(ULessThan),
    CASE(SLessThan),
    CASE(UGreaterThan),
    CASE(SGreaterThan),
    CASE(ULessThanEqual),
    CASE(SLessThanEqual),
    CASE(UGreaterThanEqual),
    CASE(SGreaterThanEqual),
    // Memory
    CASE(AccessChain),
    CASE(InBoundsAccessChain),
    CASE(PtrAccessChain),
    CASE(InBoundsPtrAccessChain),
    CASE(CooperativeMatrixLengthNV)
};

// The 60 is determined by counting the opcodes listed in the spec.
static_assert(60 == sizeof(kOpSpecConstantOpcodes)/sizeof(kOpSpecConstantOpcodes[0]),
              "OpSpecConstantOp opcode table is incomplete");
#undef CASE
// clang-format on

const size_t kNumOpSpecConstantOpcodes =
    sizeof(kOpSpecConstantOpcodes) / sizeof(kOpSpecConstantOpcodes[0]);

}  // namespace

bool AssemblyGrammar::isValid() const {
  return operandTable_ && opcodeTable_ && extInstTable_;
}

CapabilitySet AssemblyGrammar::filterCapsAgainstTargetEnv(
    const SpvCapability* cap_array, uint32_t count) const {
  CapabilitySet cap_set;
  for (uint32_t i = 0; i < count; ++i) {
    spv_operand_desc cap_desc = {};
    if (SPV_SUCCESS == lookupOperand(SPV_OPERAND_TYPE_CAPABILITY,
                                     static_cast<uint32_t>(cap_array[i]),
                                     &cap_desc)) {
      // spvOperandTableValueLookup() filters capabilities internally
      // according to the current target environment by itself. So we
      // should be safe to add this capability if the lookup succeeds.
      cap_set.Add(cap_array[i]);
    }
  }
  return cap_set;
}

spv_result_t AssemblyGrammar::lookupOpcode(const char* name,
                                           spv_opcode_desc* desc) const {
  return spvOpcodeTableNameLookup(target_env_, opcodeTable_, name, desc);
}

spv_result_t AssemblyGrammar::lookupOpcode(SpvOp opcode,
                                           spv_opcode_desc* desc) const {
  return spvOpcodeTableValueLookup(target_env_, opcodeTable_, opcode, desc);
}

spv_result_t AssemblyGrammar::lookupOperand(spv_operand_type_t type,
                                            const char* name, size_t name_len,
                                            spv_operand_desc* desc) const {
  return spvOperandTableNameLookup(target_env_, operandTable_, type, name,
                                   name_len, desc);
}

spv_result_t AssemblyGrammar::lookupOperand(spv_operand_type_t type,
                                            uint32_t operand,
                                            spv_operand_desc* desc) const {
  return spvOperandTableValueLookup(target_env_, operandTable_, type, operand,
                                    desc);
}

spv_result_t AssemblyGrammar::lookupSpecConstantOpcode(const char* name,
                                                       SpvOp* opcode) const {
  const auto* last = kOpSpecConstantOpcodes + kNumOpSpecConstantOpcodes;
  const auto* found =
      std::find_if(kOpSpecConstantOpcodes, last,
                   [name](const SpecConstantOpcodeEntry& entry) {
                     return 0 == strcmp(name, entry.name);
                   });
  if (found == last) return SPV_ERROR_INVALID_LOOKUP;
  *opcode = found->opcode;
  return SPV_SUCCESS;
}

spv_result_t AssemblyGrammar::lookupSpecConstantOpcode(SpvOp opcode) const {
  const auto* last = kOpSpecConstantOpcodes + kNumOpSpecConstantOpcodes;
  const auto* found =
      std::find_if(kOpSpecConstantOpcodes, last,
                   [opcode](const SpecConstantOpcodeEntry& entry) {
                     return opcode == entry.opcode;
                   });
  if (found == last) return SPV_ERROR_INVALID_LOOKUP;
  return SPV_SUCCESS;
}

spv_result_t AssemblyGrammar::parseMaskOperand(const spv_operand_type_t type,
                                               const char* textValue,
                                               uint32_t* pValue) const {
  return spvTextParseMaskOperand(target_env_, operandTable_, type, textValue,
                                 pValue);
}
spv_result_t AssemblyGrammar::lookupExtInst(spv_ext_inst_type_t type,
                                            const char* textValue,
                                            spv_ext_inst_desc* extInst) const {
  return spvExtInstTableNameLookup(extInstTable_, type, textValue, extInst);
}

spv_result_t AssemblyGrammar::lookupExtInst(spv_ext_inst_type_t type,
                                            uint32_t firstWord,
                                            spv_ext_inst_desc* extInst) const {
  return spvExtInstTableValueLookup(extInstTable_, type, firstWord, extInst);
}

void AssemblyGrammar::pushOperandTypesForMask(
    const spv_operand_type_t type, const uint32_t mask,
    spv_operand_pattern_t* pattern) const {
  spvPushOperandTypesForMask(target_env_, operandTable_, type, mask, pattern);
}

}  // namespace spvtools
