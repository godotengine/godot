// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        protobufs::TransformationReplaceOpSelectWithConditionalBranch message)
    : message_(std::move(message)) {}

TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        uint32_t select_id, uint32_t true_block_id, uint32_t false_block_id) {
  message_.set_select_id(select_id);
  message_.set_true_block_id(true_block_id);
  message_.set_false_block_id(false_block_id);
}

bool TransformationReplaceOpSelectWithConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  assert((message_.true_block_id() || message_.false_block_id()) &&
         "At least one of the ids must be non-zero.");

  // Check that the non-zero ids are fresh.
  std::set<uint32_t> used_ids;
  for (uint32_t id : {message_.true_block_id(), message_.false_block_id()}) {
    if (id && !CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                            &used_ids)) {
      return false;
    }
  }

  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  // The instruction must exist and it must be an OpSelect instruction.
  if (!instruction || instruction->opcode() != spv::Op::OpSelect) {
    return false;
  }

  // Check that the condition is a scalar boolean.
  auto condition = ir_context->get_def_use_mgr()->GetDef(
      instruction->GetSingleWordInOperand(0));
  assert(condition && "The condition should always exist in a valid module.");

  auto condition_type =
      ir_context->get_type_mgr()->GetType(condition->type_id());
  if (!condition_type->AsBool()) {
    return false;
  }

  auto block = ir_context->get_instr_block(instruction);
  assert(block && "The block containing the instruction must be found");

  // The instruction must be the first in its block.
  if (instruction->unique_id() != block->begin()->unique_id()) {
    return false;
  }

  // The block must not be a merge block.
  if (ir_context->GetStructuredCFGAnalysis()->IsMergeBlock(block->id())) {
    return false;
  }

  // The block must have exactly one predecessor.
  auto predecessors = ir_context->cfg()->preds(block->id());
  if (predecessors.size() != 1) {
    return false;
  }

  uint32_t pred_id = predecessors[0];
  auto predecessor = ir_context->get_instr_block(pred_id);

  // The predecessor must not be the header of a construct and it must end with
  // OpBranch.
  if (predecessor->GetMergeInst() != nullptr ||
      predecessor->terminator()->opcode() != spv::Op::OpBranch) {
    return false;
  }

  return true;
}

void TransformationReplaceOpSelectWithConditionalBranch::Apply(
    opt::IRContext* ir_context, TransformationContext* /* unused */) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  auto block = ir_context->get_instr_block(instruction);

  auto predecessor =
      ir_context->get_instr_block(ir_context->cfg()->preds(block->id())[0]);

  // Create a new block for each non-zero id in {|message_.true_branch_id|,
  // |message_.false_branch_id|}. Make each newly-created block branch
  // unconditionally to the instruction block.
  for (uint32_t id : {message_.true_block_id(), message_.false_block_id()}) {
    if (id) {
      fuzzerutil::UpdateModuleIdBound(ir_context, id);

      // Create the new block.
      auto new_block = MakeUnique<opt::BasicBlock>(
          MakeUnique<opt::Instruction>(ir_context, spv::Op::OpLabel, 0, id,
                                       opt::Instruction::OperandList{}));

      // Add an unconditional branch from the new block to the instruction
      // block.
      new_block->AddInstruction(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpBranch, 0, 0,
          opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {block->id()}}}));

      // Insert the new block right after the predecessor of the instruction
      // block.
      block->GetParent()->InsertBasicBlockBefore(std::move(new_block), block);
    }
  }

  // Delete the OpBranch instruction from the predecessor.
  ir_context->KillInst(predecessor->terminator());

  // Add an OpSelectionMerge instruction to the predecessor block, where the
  // merge block is the instruction block.
  predecessor->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {block->id()}},
          {SPV_OPERAND_TYPE_SELECTION_CONTROL,
           {uint32_t(spv::SelectionControlMask::MaskNone)}}}));

  // |if_block| will be the true block, if it has been created, the instruction
  // block otherwise.
  uint32_t if_block =
      message_.true_block_id() ? message_.true_block_id() : block->id();

  // |else_block| will be the false block, if it has been created, the
  // instruction block otherwise.
  uint32_t else_block =
      message_.false_block_id() ? message_.false_block_id() : block->id();

  assert(if_block != else_block &&
         "|if_block| and |else_block| should always be different, if the "
         "transformation is applicable.");

  // Add a conditional branching instruction to the predecessor, branching to
  // |if_block| if the condition is true and to |if_false| otherwise.
  predecessor->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpBranchConditional, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {instruction->GetSingleWordInOperand(0)}},
          {SPV_OPERAND_TYPE_ID, {if_block}},
          {SPV_OPERAND_TYPE_ID, {else_block}}}));

  // |if_pred| will be the true block, if it has been created, the existing
  // predecessor otherwise.
  uint32_t if_pred =
      message_.true_block_id() ? message_.true_block_id() : predecessor->id();

  // |else_pred| will be the false block, if it has been created, the existing
  // predecessor otherwise.
  uint32_t else_pred =
      message_.false_block_id() ? message_.false_block_id() : predecessor->id();

  // Replace the OpSelect instruction in the merge block with an OpPhi.
  // This:          OpSelect %type %cond %if %else
  // will become:   OpPhi %type %if %if_pred %else %else_pred
  instruction->SetOpcode(spv::Op::OpPhi);
  std::vector<opt::Operand> operands;

  operands.emplace_back(instruction->GetInOperand(1));
  operands.emplace_back(opt::Operand{SPV_OPERAND_TYPE_ID, {if_pred}});

  operands.emplace_back(instruction->GetInOperand(2));
  operands.emplace_back(opt::Operand{SPV_OPERAND_TYPE_ID, {else_pred}});

  instruction->SetInOperands(std::move(operands));

  // Invalidate all analyses, since the structure of the module was changed.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceOpSelectWithConditionalBranch::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opselect_with_conditional_branch() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceOpSelectWithConditionalBranch::GetFreshIds() const {
  return {message_.true_block_id(), message_.false_block_id()};
}

}  // namespace fuzz
}  // namespace spvtools
