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

#include "source/fuzz/transformation_add_loop_to_create_int_constant_synonym.h"
#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
namespace {
uint32_t kMaxNumOfIterations = 32;
}

TransformationAddLoopToCreateIntConstantSynonym::
    TransformationAddLoopToCreateIntConstantSynonym(
        protobufs::TransformationAddLoopToCreateIntConstantSynonym message)
    : message_(std::move(message)) {}

TransformationAddLoopToCreateIntConstantSynonym::
    TransformationAddLoopToCreateIntConstantSynonym(
        uint32_t constant_id, uint32_t initial_val_id, uint32_t step_val_id,
        uint32_t num_iterations_id, uint32_t block_after_loop_id,
        uint32_t syn_id, uint32_t loop_id, uint32_t ctr_id, uint32_t temp_id,
        uint32_t eventual_syn_id, uint32_t incremented_ctr_id, uint32_t cond_id,
        uint32_t additional_block_id) {
  message_.set_constant_id(constant_id);
  message_.set_initial_val_id(initial_val_id);
  message_.set_step_val_id(step_val_id);
  message_.set_num_iterations_id(num_iterations_id);
  message_.set_block_after_loop_id(block_after_loop_id);
  message_.set_syn_id(syn_id);
  message_.set_loop_id(loop_id);
  message_.set_ctr_id(ctr_id);
  message_.set_temp_id(temp_id);
  message_.set_eventual_syn_id(eventual_syn_id);
  message_.set_incremented_ctr_id(incremented_ctr_id);
  message_.set_cond_id(cond_id);
  message_.set_additional_block_id(additional_block_id);
}

bool TransformationAddLoopToCreateIntConstantSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |message_.constant_id|, |message_.initial_val_id| and
  // |message_.step_val_id| are existing constants, and that their values are
  // not irrelevant.
  auto constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant_id());
  auto initial_val = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.initial_val_id());
  auto step_val = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.step_val_id());

  if (!constant || !initial_val || !step_val) {
    return false;
  }
  if (transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.constant_id()) ||
      transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.initial_val_id()) ||
      transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.step_val_id())) {
    return false;
  }

  // Check that the type of |constant| is integer scalar or vector with integer
  // components.
  if (!constant->AsIntConstant() &&
      (!constant->AsVectorConstant() ||
       !constant->type()->AsVector()->element_type()->AsInteger())) {
    return false;
  }

  // Check that the component bit width of |constant| is <= 64.
  // Consider the width of the constant if it is an integer, of a single
  // component if it is a vector.
  uint32_t bit_width =
      constant->AsIntConstant()
          ? constant->type()->AsInteger()->width()
          : constant->type()->AsVector()->element_type()->AsInteger()->width();
  if (bit_width > 64) {
    return false;
  }

  auto constant_def =
      ir_context->get_def_use_mgr()->GetDef(message_.constant_id());
  auto initial_val_def =
      ir_context->get_def_use_mgr()->GetDef(message_.initial_val_id());
  auto step_val_def =
      ir_context->get_def_use_mgr()->GetDef(message_.step_val_id());

  // Check that |constant|, |initial_val| and |step_val| have the same type,
  // with possibly different signedness.
  if (!fuzzerutil::TypesAreEqualUpToSign(ir_context, constant_def->type_id(),
                                         initial_val_def->type_id()) ||
      !fuzzerutil::TypesAreEqualUpToSign(ir_context, constant_def->type_id(),
                                         step_val_def->type_id())) {
    return false;
  }

  // |message_.num_iterations_id| must be a non-irrelevant integer constant with
  // bit width 32.
  auto num_iterations = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.num_iterations_id());

  if (!num_iterations || !num_iterations->AsIntConstant() ||
      num_iterations->type()->AsInteger()->width() != 32 ||
      transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.num_iterations_id())) {
    return false;
  }

  // Check that the number of iterations is > 0 and <= 32.
  uint32_t num_iterations_value =
      num_iterations->AsIntConstant()->GetU32BitValue();

  if (num_iterations_value == 0 || num_iterations_value > kMaxNumOfIterations) {
    return false;
  }

  // Check that the module contains 32-bit signed integer scalar constants of
  // value 0 and 1.
  if (!fuzzerutil::MaybeGetIntegerConstant(ir_context, transformation_context,
                                           {0}, 32, true, false)) {
    return false;
  }

  if (!fuzzerutil::MaybeGetIntegerConstant(ir_context, transformation_context,
                                           {1}, 32, true, false)) {
    return false;
  }

  // Check that the module contains the Bool type.
  if (!fuzzerutil::MaybeGetBoolType(ir_context)) {
    return false;
  }

  // Check that the equation C = I - S * N is satisfied.

  // Collect the components in vectors (if the constants are scalars, these
  // vectors will contain the constants themselves).
  std::vector<const opt::analysis::Constant*> c_components;
  std::vector<const opt::analysis::Constant*> i_components;
  std::vector<const opt::analysis::Constant*> s_components;
  if (constant->AsIntConstant()) {
    c_components.emplace_back(constant);
    i_components.emplace_back(initial_val);
    s_components.emplace_back(step_val);
  } else {
    // It is a vector: get all the components.
    c_components = constant->AsVectorConstant()->GetComponents();
    i_components = initial_val->AsVectorConstant()->GetComponents();
    s_components = step_val->AsVectorConstant()->GetComponents();
  }

  // Check the value of the components satisfy the equation.
  for (uint32_t i = 0; i < c_components.size(); i++) {
    // Use 64-bits integers to be able to handle constants of any width <= 64.
    uint64_t c_value = c_components[i]->AsIntConstant()->GetZeroExtendedValue();
    uint64_t i_value = i_components[i]->AsIntConstant()->GetZeroExtendedValue();
    uint64_t s_value = s_components[i]->AsIntConstant()->GetZeroExtendedValue();

    uint64_t result = i_value - s_value * num_iterations_value;

    // Use bit shifts to ignore the first bits in excess (if there are any). By
    // shifting left, we discard the first |64 - bit_width| bits. By shifting
    // right, we move the bits back to their correct position.
    result = (result << (64 - bit_width)) >> (64 - bit_width);

    if (c_value != result) {
      return false;
    }
  }

  // Check that |message_.block_after_loop_id| is the label of a block.
  auto block =
      fuzzerutil::MaybeFindBlock(ir_context, message_.block_after_loop_id());

  // Check that the block exists and has a single predecessor.
  if (!block || ir_context->cfg()->preds(block->id()).size() != 1) {
    return false;
  }

  // Check that the block is not dead.  If it is then the new loop would be
  // dead and the data it computes would be irrelevant, so we would not be able
  // to make a synonym.
  if (transformation_context.GetFactManager()->BlockIsDead(block->id())) {
    return false;
  }

  // Check that the block is not a merge block.
  if (ir_context->GetStructuredCFGAnalysis()->IsMergeBlock(block->id())) {
    return false;
  }

  // Check that the block is not a continue block.
  if (ir_context->GetStructuredCFGAnalysis()->IsContinueBlock(block->id())) {
    return false;
  }

  // Check that the block is not a loop header.
  if (block->IsLoopHeader()) {
    return false;
  }

  // Check all the fresh ids.
  std::set<uint32_t> fresh_ids_used;
  for (uint32_t id : {message_.syn_id(), message_.loop_id(), message_.ctr_id(),
                      message_.temp_id(), message_.eventual_syn_id(),
                      message_.incremented_ctr_id(), message_.cond_id()}) {
    if (!id || !CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                             &fresh_ids_used)) {
      return false;
    }
  }

  // Check the additional block id if it is non-zero.
  return !message_.additional_block_id() ||
         CheckIdIsFreshAndNotUsedByThisTransformation(
             message_.additional_block_id(), ir_context, &fresh_ids_used);
}

void TransformationAddLoopToCreateIntConstantSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Find 32-bit signed integer constants 0 and 1.
  uint32_t const_0_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {0}, 32, true, false);
  auto const_0_def = ir_context->get_def_use_mgr()->GetDef(const_0_id);
  uint32_t const_1_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {1}, 32, true, false);

  // Retrieve the instruction defining the initial value constant.
  auto initial_val_def =
      ir_context->get_def_use_mgr()->GetDef(message_.initial_val_id());

  // Retrieve the block before which we want to insert the loop.
  auto block_after_loop =
      ir_context->get_instr_block(message_.block_after_loop_id());

  // Find the predecessor of the block.
  uint32_t pred_id =
      ir_context->cfg()->preds(message_.block_after_loop_id())[0];

  // Get the id for the last block in the new loop. It will be
  // |message_.additional_block_id| if this is non_zero, |message_.loop_id|
  // otherwise.
  uint32_t last_loop_block_id = message_.additional_block_id()
                                    ? message_.additional_block_id()
                                    : message_.loop_id();

  // Create the loop header block.
  std::unique_ptr<opt::BasicBlock> loop_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLabel, 0, message_.loop_id(),
          opt::Instruction::OperandList{}));

  // Add OpPhi instructions to retrieve the current value of the counter and of
  // the temporary variable that will be decreased at each operation.
  loop_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpPhi, const_0_def->type_id(), message_.ctr_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {const_0_id}},
          {SPV_OPERAND_TYPE_ID, {pred_id}},
          {SPV_OPERAND_TYPE_ID, {message_.incremented_ctr_id()}},
          {SPV_OPERAND_TYPE_ID, {last_loop_block_id}}}));

  loop_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpPhi, initial_val_def->type_id(),
      message_.temp_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.initial_val_id()}},
          {SPV_OPERAND_TYPE_ID, {pred_id}},
          {SPV_OPERAND_TYPE_ID, {message_.eventual_syn_id()}},
          {SPV_OPERAND_TYPE_ID, {last_loop_block_id}}}));

  // Collect the other instructions in a list. These will be added to an
  // additional block if |message_.additional_block_id| is defined, to the loop
  // header otherwise.
  std::vector<std::unique_ptr<opt::Instruction>> other_instructions;

  // Add an instruction to subtract the step value from the temporary value.
  // The value of this id will converge to the constant in the last iteration.
  other_instructions.push_back(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpISub, initial_val_def->type_id(),
      message_.eventual_syn_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.temp_id()}},
          {SPV_OPERAND_TYPE_ID, {message_.step_val_id()}}}));

  // Add an instruction to increment the counter.
  other_instructions.push_back(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpIAdd, const_0_def->type_id(),
      message_.incremented_ctr_id(),
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {message_.ctr_id()}},
                                    {SPV_OPERAND_TYPE_ID, {const_1_id}}}));

  // Add an instruction to decide whether the condition holds.
  other_instructions.push_back(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSLessThan,
      fuzzerutil::MaybeGetBoolType(ir_context), message_.cond_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.incremented_ctr_id()}},
          {SPV_OPERAND_TYPE_ID, {message_.num_iterations_id()}}}));

  // Define the OpLoopMerge instruction for the loop header. The merge block is
  // the existing block, the continue block is the last block in the loop
  // (either the loop itself or the additional block).
  std::unique_ptr<opt::Instruction> merge_inst = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLoopMerge, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.block_after_loop_id()}},
          {SPV_OPERAND_TYPE_ID, {last_loop_block_id}},
          {SPV_OPERAND_TYPE_LOOP_CONTROL,
           {uint32_t(spv::LoopControlMask::MaskNone)}}});

  // Define a conditional branch instruction, branching to the loop header if
  // the condition holds, and to the existing block otherwise. This instruction
  // will be added to the last block in the loop.
  std::unique_ptr<opt::Instruction> conditional_branch =
      MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpBranchConditional, 0, 0,
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {message_.cond_id()}},
              {SPV_OPERAND_TYPE_ID, {message_.loop_id()}},
              {SPV_OPERAND_TYPE_ID, {message_.block_after_loop_id()}}});

  if (message_.additional_block_id()) {
    // If an id for the additional block is specified, create an additional
    // block, containing the instructions in the list and a branching
    // instruction.

    std::unique_ptr<opt::BasicBlock> additional_block =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpLabel, 0, message_.additional_block_id(),
            opt::Instruction::OperandList{}));

    for (auto& instruction : other_instructions) {
      additional_block->AddInstruction(std::move(instruction));
    }

    additional_block->AddInstruction(std::move(conditional_branch));

    // Add the merge instruction to the header.
    loop_block->AddInstruction(std::move(merge_inst));

    // Add an unconditional branch from the header to the additional block.
    loop_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpBranch, 0, 0,
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {message_.additional_block_id()}}}));

    // Insert the two loop blocks before the existing block.
    block_after_loop->GetParent()->InsertBasicBlockBefore(std::move(loop_block),
                                                          block_after_loop);
    block_after_loop->GetParent()->InsertBasicBlockBefore(
        std::move(additional_block), block_after_loop);
  } else {
    // If no id for an additional block is specified, the loop will only be made
    // up of one block, so we need to add all the instructions to it.

    for (auto& instruction : other_instructions) {
      loop_block->AddInstruction(std::move(instruction));
    }

    // Add the merge and conditional branch instructions.
    loop_block->AddInstruction(std::move(merge_inst));
    loop_block->AddInstruction(std::move(conditional_branch));

    // Insert the header before the existing block.
    block_after_loop->GetParent()->InsertBasicBlockBefore(std::move(loop_block),
                                                          block_after_loop);
  }

  // Update the branching instructions leading to this block.
  ir_context->get_def_use_mgr()->ForEachUse(
      message_.block_after_loop_id(),
      [this](opt::Instruction* instruction, uint32_t operand_index) {
        assert(instruction->opcode() != spv::Op::OpLoopMerge &&
               instruction->opcode() != spv::Op::OpSelectionMerge &&
               "The block should not be referenced by OpLoopMerge or "
               "OpSelectionMerge, by construction.");
        // Replace all uses of the label inside branch instructions.
        if (instruction->opcode() == spv::Op::OpBranch ||
            instruction->opcode() == spv::Op::OpBranchConditional ||
            instruction->opcode() == spv::Op::OpSwitch) {
          instruction->SetOperand(operand_index, {message_.loop_id()});
        }
      });

  // Update all the OpPhi instructions in the block after the loop: its
  // predecessor is now the last block in the loop.
  block_after_loop->ForEachPhiInst(
      [last_loop_block_id](opt::Instruction* phi_inst) {
        // Since the block only had one predecessor, the id of the predecessor
        // is input operand 1.
        phi_inst->SetInOperand(1, {last_loop_block_id});
      });

  // Add a new OpPhi instruction at the beginning of the block after the loop,
  // defining the synonym of the constant. The type id will be the same as
  // |message_.initial_value_id|, since this is the value that is decremented in
  // the loop.
  block_after_loop->begin()->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpPhi, initial_val_def->type_id(), message_.syn_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.eventual_syn_id()}},
          {SPV_OPERAND_TYPE_ID, {last_loop_block_id}}}));

  // Update the module id bound with all the fresh ids used.
  for (uint32_t id : {message_.syn_id(), message_.loop_id(), message_.ctr_id(),
                      message_.temp_id(), message_.eventual_syn_id(),
                      message_.incremented_ctr_id(), message_.cond_id(),
                      message_.cond_id(), message_.additional_block_id()}) {
    fuzzerutil::UpdateModuleIdBound(ir_context, id);
  }

  // Since we changed the structure of the module, we need to invalidate all the
  // analyses.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // Record that |message_.syn_id| is synonymous with |message_.constant_id|.
  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.syn_id(), {}),
      MakeDataDescriptor(message_.constant_id(), {}));
}

protobufs::Transformation
TransformationAddLoopToCreateIntConstantSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_loop_to_create_int_constant_synonym() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationAddLoopToCreateIntConstantSynonym::GetFreshIds() const {
  return {message_.syn_id(),          message_.loop_id(),
          message_.ctr_id(),          message_.temp_id(),
          message_.eventual_syn_id(), message_.incremented_ctr_id(),
          message_.cond_id(),         message_.additional_block_id()};
}

}  // namespace fuzz
}  // namespace spvtools
