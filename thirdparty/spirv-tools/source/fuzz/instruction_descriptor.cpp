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

#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

opt::Instruction* FindInstruction(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    spvtools::opt::IRContext* context) {
  auto block = context->get_instr_block(
      instruction_descriptor.base_instruction_result_id());
  if (block == nullptr) {
    return nullptr;
  }
  bool found_base =
      block->id() == instruction_descriptor.base_instruction_result_id();
  uint32_t num_ignored = 0;
  for (auto& instruction : *block) {
    if (instruction.HasResultId() &&
        instruction.result_id() ==
            instruction_descriptor.base_instruction_result_id()) {
      assert(!found_base &&
             "It should not be possible to find the base instruction "
             "multiple times.");
      found_base = true;
      assert(num_ignored == 0 &&
             "The skipped instruction count should only be incremented "
             "after the instruction base has been found.");
    }
    if (found_base &&
        instruction.opcode() ==
            spv::Op(instruction_descriptor.target_instruction_opcode())) {
      if (num_ignored == instruction_descriptor.num_opcodes_to_ignore()) {
        return &instruction;
      }
      num_ignored++;
    }
  }
  return nullptr;
}

protobufs::InstructionDescriptor MakeInstructionDescriptor(
    uint32_t base_instruction_result_id, spv::Op target_instruction_opcode,
    uint32_t num_opcodes_to_ignore) {
  protobufs::InstructionDescriptor result;
  result.set_base_instruction_result_id(base_instruction_result_id);
  result.set_target_instruction_opcode(uint32_t(target_instruction_opcode));
  result.set_num_opcodes_to_ignore(num_opcodes_to_ignore);
  return result;
}

protobufs::InstructionDescriptor MakeInstructionDescriptor(
    const opt::BasicBlock& block,
    const opt::BasicBlock::const_iterator& inst_it) {
  const spv::Op opcode =
      inst_it->opcode();    // The opcode of the instruction being described.
  uint32_t skip_count = 0;  // The number of these opcodes we have skipped when
  // searching backwards.

  // Consider instructions in the block in reverse order, starting from
  // |inst_it|.
  for (opt::BasicBlock::const_iterator backwards_iterator = inst_it;;
       --backwards_iterator) {
    if (backwards_iterator->HasResultId()) {
      // As soon as we find an instruction with a result id, we can return a
      // descriptor for |inst_it|.
      return MakeInstructionDescriptor(backwards_iterator->result_id(), opcode,
                                       skip_count);
    }
    if (backwards_iterator != inst_it &&
        backwards_iterator->opcode() == opcode) {
      // We are skipping over an instruction with the same opcode as |inst_it|;
      // we increase our skip count to reflect this.
      skip_count++;
    }
    if (backwards_iterator == block.begin()) {
      // We exit the loop when we reach the start of the block, but only after
      // we have processed the first instruction in the block.
      break;
    }
  }
  // We did not find an instruction inside the block with a result id, so we use
  // the block's label's id.
  return MakeInstructionDescriptor(block.id(), opcode, skip_count);
}

protobufs::InstructionDescriptor MakeInstructionDescriptor(
    opt::IRContext* context, opt::Instruction* inst) {
  auto block = context->get_instr_block(inst);
  uint32_t base_instruction_result_id = block->id();
  uint32_t num_opcodes_to_ignore = 0;
  for (auto& inst_in_block : *block) {
    if (inst_in_block.HasResultId()) {
      base_instruction_result_id = inst_in_block.result_id();
      num_opcodes_to_ignore = 0;
    }
    if (&inst_in_block == inst) {
      return MakeInstructionDescriptor(base_instruction_result_id,
                                       inst->opcode(), num_opcodes_to_ignore);
    }
    if (inst_in_block.opcode() == inst->opcode()) {
      num_opcodes_to_ignore++;
    }
  }
  assert(false && "No matching instruction was found.");
  return protobufs::InstructionDescriptor();
}

}  // namespace fuzz
}  // namespace spvtools
