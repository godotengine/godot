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

#include "source/val/instruction.h"

#include <utility>

#include "source/binary.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace val {

Instruction::Instruction(const spv_parsed_instruction_t* inst)
    : words_(inst->words, inst->words + inst->num_words),
      operands_(inst->operands, inst->operands + inst->num_operands),
      inst_({words_.data(), inst->num_words, inst->opcode, inst->ext_inst_type,
             inst->type_id, inst->result_id, operands_.data(),
             inst->num_operands}) {}

void Instruction::RegisterUse(const Instruction* inst, uint32_t index) {
  uses_.push_back(std::make_pair(inst, index));
}

bool operator<(const Instruction& lhs, const Instruction& rhs) {
  return lhs.id() < rhs.id();
}
bool operator<(const Instruction& lhs, uint32_t rhs) { return lhs.id() < rhs; }
bool operator==(const Instruction& lhs, const Instruction& rhs) {
  return lhs.id() == rhs.id();
}
bool operator==(const Instruction& lhs, uint32_t rhs) {
  return lhs.id() == rhs;
}

template <>
std::string Instruction::GetOperandAs<std::string>(size_t index) const {
  const spv_parsed_operand_t& o = operands_.at(index);
  assert(o.offset + o.num_words <= inst_.num_words);
  return spvtools::utils::MakeString(words_.data() + o.offset, o.num_words);
}

}  // namespace val
}  // namespace spvtools
