// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_swap_conditional_branch_operands.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationSwapConditionalBranchOperands::
    TransformationSwapConditionalBranchOperands(
        protobufs::TransformationSwapConditionalBranchOperands message)
    : message_(std::move(message)) {}

TransformationSwapConditionalBranchOperands::
    TransformationSwapConditionalBranchOperands(
        const protobufs::InstructionDescriptor& instruction_descriptor,
        uint32_t fresh_id) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
  message_.set_fresh_id(fresh_id);
}

bool TransformationSwapConditionalBranchOperands::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  const auto* inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  return fuzzerutil::IsFreshId(ir_context, message_.fresh_id()) && inst &&
         inst->opcode() == spv::Op::OpBranchConditional;
}

void TransformationSwapConditionalBranchOperands::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* branch_inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  assert(branch_inst);

  auto* block = ir_context->get_instr_block(branch_inst);
  assert(block);

  // Compute the last instruction in the |block| that allows us to insert
  // OpLogicalNot above it.
  auto iter = fuzzerutil::GetIteratorForInstruction(block, branch_inst);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLogicalNot,
                                                    iter)) {
    // There might be a merge instruction before OpBranchConditional.
    --iter;
  }

  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLogicalNot,
                                                      iter) &&
         "We should now be able to insert spv::Op::OpLogicalNot before |iter|");

  // Get the instruction whose result is used as a condition for
  // OpBranchConditional.
  const auto* condition_inst = ir_context->get_def_use_mgr()->GetDef(
      branch_inst->GetSingleWordInOperand(0));
  assert(condition_inst);

  // We are swapping the labels in OpBranchConditional. This means that we must
  // invert the guard as well. We are using OpLogicalNot for that purpose here.
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLogicalNot, condition_inst->type_id(),
      message_.fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {condition_inst->result_id()}}});
  auto new_instruction_ptr = new_instruction.get();
  iter.InsertBefore(std::move(new_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Update OpBranchConditional condition operand.
  branch_inst->GetInOperand(0).words[0] = message_.fresh_id();

  // Swap label operands.
  std::swap(branch_inst->GetInOperand(1), branch_inst->GetInOperand(2));

  // Additionally, swap branch weights if present.
  if (branch_inst->NumInOperands() > 3) {
    std::swap(branch_inst->GetInOperand(3), branch_inst->GetInOperand(4));
  }

  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr, block);
  ir_context->get_def_use_mgr()->EraseUseRecordsOfOperandIds(branch_inst);
  ir_context->get_def_use_mgr()->AnalyzeInstUse(branch_inst);

  // No analyses need to be invalidated since the transformation is local to a
  // block and the def-use and instruction-to-block mappings have been updated.
}

protobufs::Transformation
TransformationSwapConditionalBranchOperands::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_conditional_branch_operands() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationSwapConditionalBranchOperands::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
