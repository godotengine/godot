// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/instruction_message.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

protobufs::Instruction MakeInstructionMessage(
    spv::Op opcode, uint32_t result_type_id, uint32_t result_id,
    const opt::Instruction::OperandList& input_operands) {
  protobufs::Instruction result;
  result.set_opcode(uint32_t(opcode));
  result.set_result_type_id(result_type_id);
  result.set_result_id(result_id);
  for (auto& operand : input_operands) {
    auto operand_message = result.add_input_operand();
    operand_message->set_operand_type(static_cast<uint32_t>(operand.type));
    for (auto operand_word : operand.words) {
      operand_message->add_operand_data(operand_word);
    }
  }
  return result;
}

protobufs::Instruction MakeInstructionMessage(
    const opt::Instruction* instruction) {
  opt::Instruction::OperandList input_operands;
  for (uint32_t input_operand_index = 0;
       input_operand_index < instruction->NumInOperands();
       input_operand_index++) {
    input_operands.push_back(instruction->GetInOperand(input_operand_index));
  }
  return MakeInstructionMessage(instruction->opcode(), instruction->type_id(),
                                instruction->result_id(), input_operands);
}

std::unique_ptr<opt::Instruction> InstructionFromMessage(
    opt::IRContext* ir_context,
    const protobufs::Instruction& instruction_message) {
  // First, update the module's id bound with respect to the new instruction,
  // if it has a result id.
  if (instruction_message.result_id()) {
    fuzzerutil::UpdateModuleIdBound(ir_context,
                                    instruction_message.result_id());
  }
  // Now create a sequence of input operands from the input operand data in the
  // protobuf message.
  opt::Instruction::OperandList in_operands;
  for (auto& operand_message : instruction_message.input_operand()) {
    opt::Operand::OperandData operand_data;
    for (auto& word : operand_message.operand_data()) {
      operand_data.push_back(word);
    }
    in_operands.push_back(
        {static_cast<spv_operand_type_t>(operand_message.operand_type()),
         operand_data});
  }
  // Create and return the instruction.
  return MakeUnique<opt::Instruction>(
      ir_context, static_cast<spv::Op>(instruction_message.opcode()),
      instruction_message.result_type_id(), instruction_message.result_id(),
      in_operands);
}

}  // namespace fuzz
}  // namespace spvtools
