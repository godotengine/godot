// Copyright (c) 2016 Google Inc.
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

#include "source/opt/set_spec_constant_default_value_pass.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <tuple>
#include <vector>

#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/type_manager.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"
#include "source/util/parse_number.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {
namespace {
using utils::EncodeNumberStatus;
using utils::NumberType;
using utils::ParseAndEncodeNumber;
using utils::ParseNumber;

// Given a numeric value in a null-terminated c string and the expected type of
// the value, parses the string and encodes it in a vector of words. If the
// value is a scalar integer or floating point value, encodes the value in
// SPIR-V encoding format. If the value is 'false' or 'true', returns a vector
// with single word with value 0 or 1 respectively. Returns the vector
// containing the encoded value on success. Otherwise returns an empty vector.
std::vector<uint32_t> ParseDefaultValueStr(const char* text,
                                           const analysis::Type* type) {
  std::vector<uint32_t> result;
  if (!strcmp(text, "true") && type->AsBool()) {
    result.push_back(1u);
  } else if (!strcmp(text, "false") && type->AsBool()) {
    result.push_back(0u);
  } else {
    NumberType number_type = {32, SPV_NUMBER_UNSIGNED_INT};
    if (const auto* IT = type->AsInteger()) {
      number_type.bitwidth = IT->width();
      number_type.kind =
          IT->IsSigned() ? SPV_NUMBER_SIGNED_INT : SPV_NUMBER_UNSIGNED_INT;
    } else if (const auto* FT = type->AsFloat()) {
      number_type.bitwidth = FT->width();
      number_type.kind = SPV_NUMBER_FLOATING;
    } else {
      // Does not handle types other then boolean, integer or float. Returns
      // empty vector.
      result.clear();
      return result;
    }
    EncodeNumberStatus rc = ParseAndEncodeNumber(
        text, number_type, [&result](uint32_t word) { result.push_back(word); },
        nullptr);
    // Clear the result vector on failure.
    if (rc != EncodeNumberStatus::kSuccess) {
      result.clear();
    }
  }
  return result;
}

// Given a bit pattern and a type, checks if the bit pattern is compatible
// with the type. If so, returns the bit pattern, otherwise returns an empty
// bit pattern. If the given bit pattern is empty, returns an empty bit
// pattern. If the given type represents a SPIR-V Boolean type, the bit pattern
// to be returned is determined with the following standard:
//   If any words in the input bit pattern are non zero, returns a bit pattern
//   with 0x1, which represents a 'true'.
//   If all words in the bit pattern are zero, returns a bit pattern with 0x0,
//   which represents a 'false'.
// For integer and floating point types narrower than 32 bits, the upper bits
// in the input bit pattern are ignored.  Instead the upper bits are set
// according to SPIR-V literal requirements: sign extend a signed integer, and
// otherwise set the upper bits to zero.
std::vector<uint32_t> ParseDefaultValueBitPattern(
    const std::vector<uint32_t>& input_bit_pattern,
    const analysis::Type* type) {
  std::vector<uint32_t> result;
  if (type->AsBool()) {
    if (std::any_of(input_bit_pattern.begin(), input_bit_pattern.end(),
                    [](uint32_t i) { return i != 0; })) {
      result.push_back(1u);
    } else {
      result.push_back(0u);
    }
    return result;
  } else if (const auto* IT = type->AsInteger()) {
    const auto width = IT->width();
    assert(width > 0);
    const auto adjusted_width = std::max(32u, width);
    if (adjusted_width == input_bit_pattern.size() * sizeof(uint32_t) * 8) {
      result = std::vector<uint32_t>(input_bit_pattern);
      if (width < 32) {
        const uint32_t high_active_bit = (1u << width) >> 1;
        if (IT->IsSigned() && (high_active_bit & result[0])) {
          // Sign extend.  This overwrites the sign bit again, but that's ok.
          result[0] = result[0] | ~(high_active_bit - 1);
        } else {
          // Upper bits must be zero.
          result[0] = result[0] & ((1u << width) - 1);
        }
      }
      return result;
    }
  } else if (const auto* FT = type->AsFloat()) {
    const auto width = FT->width();
    const auto adjusted_width = std::max(32u, width);
    if (adjusted_width == input_bit_pattern.size() * sizeof(uint32_t) * 8) {
      result = std::vector<uint32_t>(input_bit_pattern);
      if (width < 32) {
        // Upper bits must be zero.
        result[0] = result[0] & ((1u << width) - 1);
      }
      return result;
    }
  }
  result.clear();
  return result;
}

// Returns true if the given instruction's result id could have a SpecId
// decoration.
bool CanHaveSpecIdDecoration(const Instruction& inst) {
  switch (inst.opcode()) {
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstantTrue:
      return true;
    default:
      return false;
  }
}

// Given a decoration group defining instruction that is decorated with SpecId
// decoration, finds the spec constant defining instruction which is the real
// target of the SpecId decoration. Returns the spec constant defining
// instruction if such an instruction is found, otherwise returns a nullptr.
Instruction* GetSpecIdTargetFromDecorationGroup(
    const Instruction& decoration_group_defining_inst,
    analysis::DefUseManager* def_use_mgr) {
  // Find the OpGroupDecorate instruction which consumes the given decoration
  // group. Note that the given decoration group has SpecId decoration, which
  // is unique for different spec constants. So the decoration group cannot be
  // consumed by different OpGroupDecorate instructions. Therefore we only need
  // the first OpGroupDecoration instruction that uses the given decoration
  // group.
  Instruction* group_decorate_inst = nullptr;
  if (def_use_mgr->WhileEachUser(&decoration_group_defining_inst,
                                 [&group_decorate_inst](Instruction* user) {
                                   if (user->opcode() ==
                                       spv::Op::OpGroupDecorate) {
                                     group_decorate_inst = user;
                                     return false;
                                   }
                                   return true;
                                 }))
    return nullptr;

  // Scan through the target ids of the OpGroupDecorate instruction. There
  // should be only one spec constant target consumes the SpecId decoration.
  // If multiple target ids are presented in the OpGroupDecorate instruction,
  // they must be the same one that defined by an eligible spec constant
  // instruction. If the OpGroupDecorate instruction has different target ids
  // or a target id is not defined by an eligible spec cosntant instruction,
  // returns a nullptr.
  Instruction* target_inst = nullptr;
  for (uint32_t i = 1; i < group_decorate_inst->NumInOperands(); i++) {
    // All the operands of a OpGroupDecorate instruction should be of type
    // SPV_OPERAND_TYPE_ID.
    uint32_t candidate_id = group_decorate_inst->GetSingleWordInOperand(i);
    Instruction* candidate_inst = def_use_mgr->GetDef(candidate_id);

    if (!candidate_inst) {
      continue;
    }

    if (!target_inst) {
      // If the spec constant target has not been found yet, check if the
      // candidate instruction is the target.
      if (CanHaveSpecIdDecoration(*candidate_inst)) {
        target_inst = candidate_inst;
      } else {
        // Spec id decoration should not be applied on other instructions.
        // TODO(qining): Emit an error message in the invalid case once the
        // error handling is done.
        return nullptr;
      }
    } else {
      // If the spec constant target has been found, check if the candidate
      // instruction is the same one as the target. The module is invalid if
      // the candidate instruction is different with the found target.
      // TODO(qining): Emit an error messaage in the invalid case once the
      // error handling is done.
      if (candidate_inst != target_inst) return nullptr;
    }
  }
  return target_inst;
}
}  // namespace

Pass::Status SetSpecConstantDefaultValuePass::Process() {
  // The operand index of decoration target in an OpDecorate instruction.
  constexpr uint32_t kTargetIdOperandIndex = 0;
  // The operand index of the decoration literal in an OpDecorate instruction.
  constexpr uint32_t kDecorationOperandIndex = 1;
  // The operand index of Spec id literal value in an OpDecorate SpecId
  // instruction.
  constexpr uint32_t kSpecIdLiteralOperandIndex = 2;
  // The number of operands in an OpDecorate SpecId instruction.
  constexpr uint32_t kOpDecorateSpecIdNumOperands = 3;
  // The in-operand index of the default value in a OpSpecConstant instruction.
  constexpr uint32_t kOpSpecConstantLiteralInOperandIndex = 0;

  bool modified = false;
  // Scan through all the annotation instructions to find 'OpDecorate SpecId'
  // instructions. Then extract the decoration target of those instructions.
  // The decoration targets should be spec constant defining instructions with
  // opcode: OpSpecConstant{|True|False}. The spec id of those spec constants
  // will be used to look up their new default values in the mapping from
  // spec id to new default value strings. Once a new default value string
  // is found for a spec id, the string will be parsed according to the target
  // spec constant type. The parsed value will be used to replace the original
  // default value of the target spec constant.
  for (Instruction& inst : context()->annotations()) {
    // Only process 'OpDecorate SpecId' instructions
    if (inst.opcode() != spv::Op::OpDecorate) continue;
    if (inst.NumOperands() != kOpDecorateSpecIdNumOperands) continue;
    if (inst.GetSingleWordInOperand(kDecorationOperandIndex) !=
        uint32_t(spv::Decoration::SpecId)) {
      continue;
    }

    // 'inst' is an OpDecorate SpecId instruction.
    uint32_t spec_id = inst.GetSingleWordOperand(kSpecIdLiteralOperandIndex);
    uint32_t target_id = inst.GetSingleWordOperand(kTargetIdOperandIndex);

    // Find the spec constant defining instruction. Note that the
    // target_id might be a decoration group id.
    Instruction* spec_inst = nullptr;
    if (Instruction* target_inst = get_def_use_mgr()->GetDef(target_id)) {
      if (target_inst->opcode() == spv::Op::OpDecorationGroup) {
        spec_inst =
            GetSpecIdTargetFromDecorationGroup(*target_inst, get_def_use_mgr());
      } else {
        spec_inst = target_inst;
      }
    } else {
      continue;
    }
    if (!spec_inst) continue;

    // Get the default value bit pattern for this spec id.
    std::vector<uint32_t> bit_pattern;

    if (spec_id_to_value_str_.size() != 0) {
      // Search for the new string-form default value for this spec id.
      auto iter = spec_id_to_value_str_.find(spec_id);
      if (iter == spec_id_to_value_str_.end()) {
        continue;
      }

      // Gets the string of the default value and parses it to bit pattern
      // with the type of the spec constant.
      const std::string& default_value_str = iter->second;
      bit_pattern = ParseDefaultValueStr(
          default_value_str.c_str(),
          context()->get_type_mgr()->GetType(spec_inst->type_id()));

    } else {
      // Search for the new bit-pattern-form default value for this spec id.
      auto iter = spec_id_to_value_bit_pattern_.find(spec_id);
      if (iter == spec_id_to_value_bit_pattern_.end()) {
        continue;
      }

      // Gets the bit-pattern of the default value from the map directly.
      bit_pattern = ParseDefaultValueBitPattern(
          iter->second,
          context()->get_type_mgr()->GetType(spec_inst->type_id()));
    }

    if (bit_pattern.empty()) continue;

    // Update the operand bit patterns of the spec constant defining
    // instruction.
    switch (spec_inst->opcode()) {
      case spv::Op::OpSpecConstant:
        // If the new value is the same with the original value, no
        // need to do anything. Otherwise update the operand words.
        if (spec_inst->GetInOperand(kOpSpecConstantLiteralInOperandIndex)
                .words != bit_pattern) {
          spec_inst->SetInOperand(kOpSpecConstantLiteralInOperandIndex,
                                  std::move(bit_pattern));
          modified = true;
        }
        break;
      case spv::Op::OpSpecConstantTrue:
        // If the new value is also 'true', no need to change anything.
        // Otherwise, set the opcode to OpSpecConstantFalse;
        if (!static_cast<bool>(bit_pattern.front())) {
          spec_inst->SetOpcode(spv::Op::OpSpecConstantFalse);
          modified = true;
        }
        break;
      case spv::Op::OpSpecConstantFalse:
        // If the new value is also 'false', no need to change anything.
        // Otherwise, set the opcode to OpSpecConstantTrue;
        if (static_cast<bool>(bit_pattern.front())) {
          spec_inst->SetOpcode(spv::Op::OpSpecConstantTrue);
          modified = true;
        }
        break;
      default:
        break;
    }
    // No need to update the DefUse manager, as this pass does not change any
    // ids.
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

// Returns true if the given char is ':', '\0' or considered as blank space
// (i.e.: '\n', '\r', '\v', '\t', '\f' and ' ').
bool IsSeparator(char ch) {
  return std::strchr(":\0", ch) || std::isspace(ch) != 0;
}

std::unique_ptr<SetSpecConstantDefaultValuePass::SpecIdToValueStrMap>
SetSpecConstantDefaultValuePass::ParseDefaultValuesString(const char* str) {
  if (!str) return nullptr;

  auto spec_id_to_value = MakeUnique<SpecIdToValueStrMap>();

  // The parsing loop, break when points to the end.
  while (*str) {
    // Find the spec id.
    while (std::isspace(*str)) str++;  // skip leading spaces.
    const char* entry_begin = str;
    while (!IsSeparator(*str)) str++;
    const char* entry_end = str;
    std::string spec_id_str(entry_begin, entry_end - entry_begin);
    uint32_t spec_id = 0;
    if (!ParseNumber(spec_id_str.c_str(), &spec_id)) {
      // The spec id is not a valid uint32 number.
      return nullptr;
    }
    auto iter = spec_id_to_value->find(spec_id);
    if (iter != spec_id_to_value->end()) {
      // Same spec id has been defined before
      return nullptr;
    }
    // Find the ':', spaces between the spec id and the ':' are not allowed.
    if (*str++ != ':') {
      // ':' not found
      return nullptr;
    }
    // Find the value string
    const char* val_begin = str;
    while (!IsSeparator(*str)) str++;
    const char* val_end = str;
    if (val_end == val_begin) {
      // Value string is empty.
      return nullptr;
    }
    // Update the mapping with spec id and value string.
    (*spec_id_to_value)[spec_id] = std::string(val_begin, val_end - val_begin);

    // Skip trailing spaces.
    while (std::isspace(*str)) str++;
  }

  return spec_id_to_value;
}

}  // namespace opt
}  // namespace spvtools
