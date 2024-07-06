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

#include "source/fuzz/transformation_add_dead_block.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddDeadBlock::TransformationAddDeadBlock(
    protobufs::TransformationAddDeadBlock message)
    : message_(std::move(message)) {}

TransformationAddDeadBlock::TransformationAddDeadBlock(uint32_t fresh_id,
                                                       uint32_t existing_block,
                                                       bool condition_value) {
  message_.set_fresh_id(fresh_id);
  message_.set_existing_block(existing_block);
  message_.set_condition_value(condition_value);
}

bool TransformationAddDeadBlock::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The new block's id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // First, we check that a constant with the same value as
  // |message_.condition_value| is present.
  if (!fuzzerutil::MaybeGetBoolConstant(ir_context, transformation_context,
                                        message_.condition_value(), false)) {
    // The required constant is not present, so the transformation cannot be
    // applied.
    return false;
  }

  // The existing block must indeed exist.
  auto existing_block =
      fuzzerutil::MaybeFindBlock(ir_context, message_.existing_block());
  if (!existing_block) {
    return false;
  }

  // It must not head a loop.
  if (existing_block->IsLoopHeader()) {
    return false;
  }

  // It must end with OpBranch.
  if (existing_block->terminator()->opcode() != spv::Op::OpBranch) {
    return false;
  }

  // Its successor must not be a merge block nor continue target.
  auto successor_block_id =
      existing_block->terminator()->GetSingleWordInOperand(0);
  if (fuzzerutil::IsMergeOrContinue(ir_context, successor_block_id)) {
    return false;
  }

  // The successor must not be a loop header (i.e., |message_.existing_block|
  // must not be a back-edge block.
  if (ir_context->cfg()->block(successor_block_id)->IsLoopHeader()) {
    return false;
  }

  // |existing_block| must be reachable.
  if (!ir_context->IsReachable(*existing_block)) {
    return false;
  }

  assert(existing_block->id() != successor_block_id &&
         "|existing_block| must be different from |successor_block_id|");

  // Even though we know |successor_block_id| is not a merge block, it might
  // still have multiple predecessors because divergent control flow is allowed
  // to converge early (before the merge block). In this case, when we create
  // the selection construct, its header |existing_block| will not dominate the
  // merge block |successor_block_id|, which is invalid. Thus, |existing_block|
  // must dominate |successor_block_id|.
  opt::DominatorAnalysis* dominator_analysis =
      ir_context->GetDominatorAnalysis(existing_block->GetParent());
  if (!dominator_analysis->Dominates(existing_block->id(),
                                     successor_block_id)) {
    return false;
  }

  return true;
}

void TransformationAddDeadBlock::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Update the module id bound so that it is at least the id of the new block.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Get the existing block and its successor.
  auto existing_block = ir_context->cfg()->block(message_.existing_block());
  auto successor_block_id =
      existing_block->terminator()->GetSingleWordInOperand(0);

  // Get the id of the boolean value that will be used as the branch condition.
  auto bool_id = fuzzerutil::MaybeGetBoolConstant(
      ir_context, *transformation_context, message_.condition_value(), false);

  // Make a new block that unconditionally branches to the original successor
  // block.
  auto enclosing_function = existing_block->GetParent();
  std::unique_ptr<opt::BasicBlock> new_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLabel, 0, message_.fresh_id(),
          opt::Instruction::OperandList()));
  new_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {successor_block_id}}})));

  // Turn the original block into a selection merge, with its original successor
  // as the merge block.
  existing_block->terminator()->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSelectionMerge, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {successor_block_id}},
           {SPV_OPERAND_TYPE_SELECTION_CONTROL,
            {uint32_t(spv::SelectionControlMask::MaskNone)}}})));

  // Change the original block's terminator to be a conditional branch on the
  // given boolean, with the original successor and the new successor as branch
  // targets, and such that at runtime control will always transfer to the
  // original successor.
  existing_block->terminator()->SetOpcode(spv::Op::OpBranchConditional);
  existing_block->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID, {bool_id}},
       {SPV_OPERAND_TYPE_ID,
        {message_.condition_value() ? successor_block_id
                                    : message_.fresh_id()}},
       {SPV_OPERAND_TYPE_ID,
        {message_.condition_value() ? message_.fresh_id()
                                    : successor_block_id}}});

  // Add the new block to the enclosing function.
  enclosing_function->InsertBasicBlockAfter(std::move(new_block),
                                            existing_block);

  // Fix up OpPhi instructions in the successor block, so that the values they
  // yield when control has transferred from the new block are the same as if
  // control had transferred from |message_.existing_block|.  This is guaranteed
  // to be valid since |message_.existing_block| dominates the new block by
  // construction.  Other transformations can change these phi operands to more
  // interesting values.
  ir_context->cfg()
      ->block(successor_block_id)
      ->ForEachPhiInst([this](opt::Instruction* phi_inst) {
        // Copy the operand that provides the phi value for the first of any
        // existing predecessors.
        opt::Operand copy_of_existing_operand = phi_inst->GetInOperand(0);
        // Use this as the value associated with the new predecessor.
        phi_inst->AddOperand(std::move(copy_of_existing_operand));
        phi_inst->AddOperand({SPV_OPERAND_TYPE_ID, {message_.fresh_id()}});
      });

  // Do not rely on any existing analysis results since the control flow graph
  // of the module has changed.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // Record the fact that the new block is dead.
  transformation_context->GetFactManager()->AddFactBlockIsDead(
      message_.fresh_id());
}

protobufs::Transformation TransformationAddDeadBlock::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_dead_block() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddDeadBlock::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
