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

#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceBranchFromDeadBlockWithExit::
    TransformationReplaceBranchFromDeadBlockWithExit(
        protobufs::TransformationReplaceBranchFromDeadBlockWithExit message)
    : message_(std::move(message)) {}

TransformationReplaceBranchFromDeadBlockWithExit::
    TransformationReplaceBranchFromDeadBlockWithExit(uint32_t block_id,
                                                     spv::Op opcode,
                                                     uint32_t return_value_id) {
  message_.set_block_id(block_id);
  message_.set_opcode(uint32_t(opcode));
  message_.set_return_value_id(return_value_id);
}

bool TransformationReplaceBranchFromDeadBlockWithExit::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The block whose terminator is to be changed must exist.
  auto block = ir_context->get_instr_block(message_.block_id());
  if (!block) {
    return false;
  }
  if (!BlockIsSuitable(ir_context, transformation_context, *block)) {
    return false;
  }
  auto function_return_type_id = block->GetParent()->type_id();
  switch (spv::Op(message_.opcode())) {
    case spv::Op::OpKill:
      for (auto& entry_point : ir_context->module()->entry_points()) {
        if (spv::ExecutionModel(entry_point.GetSingleWordInOperand(0)) !=
            spv::ExecutionModel::Fragment) {
          // OpKill is only allowed in a fragment shader.  This is a
          // conservative check: if the module contains a non-fragment entry
          // point then adding an OpKill might lead to OpKill being used in a
          // non-fragment shader.
          return false;
        }
      }
      break;
    case spv::Op::OpReturn:
      if (ir_context->get_def_use_mgr()
              ->GetDef(function_return_type_id)
              ->opcode() != spv::Op::OpTypeVoid) {
        // OpReturn is only allowed in a function with void return type.
        return false;
      }
      break;
    case spv::Op::OpReturnValue: {
      // If the terminator is to be changed to OpReturnValue, with
      // |message_.return_value_id| being the value that will be returned, then
      // |message_.return_value_id| must have a compatible type and be available
      // at the block terminator.
      auto return_value =
          ir_context->get_def_use_mgr()->GetDef(message_.return_value_id());
      if (!return_value || return_value->type_id() != function_return_type_id) {
        return false;
      }
      if (!fuzzerutil::IdIsAvailableBeforeInstruction(
              ir_context, block->terminator(), message_.return_value_id())) {
        return false;
      }
      break;
    }
    default:
      assert(spv::Op(message_.opcode()) == spv::Op::OpUnreachable &&
             "Invalid early exit opcode.");
      break;
  }
  return true;
}

void TransformationReplaceBranchFromDeadBlockWithExit::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // If the successor block has OpPhi instructions then arguments related to
  // |message_.block_id| need to be removed from these instruction.
  auto block = ir_context->get_instr_block(message_.block_id());
  assert(block->terminator()->opcode() == spv::Op::OpBranch &&
         "Precondition: the block must end with OpBranch.");
  auto successor = ir_context->get_instr_block(
      block->terminator()->GetSingleWordInOperand(0));
  successor->ForEachPhiInst([block](opt::Instruction* phi_inst) {
    opt::Instruction::OperandList new_phi_in_operands;
    for (uint32_t i = 0; i < phi_inst->NumInOperands(); i += 2) {
      if (phi_inst->GetSingleWordInOperand(i + 1) == block->id()) {
        continue;
      }
      new_phi_in_operands.emplace_back(phi_inst->GetInOperand(i));
      new_phi_in_operands.emplace_back(phi_inst->GetInOperand(i + 1));
    }
    assert(new_phi_in_operands.size() == phi_inst->NumInOperands() - 2);
    phi_inst->SetInOperands(std::move(new_phi_in_operands));
  });

  // Rewrite the terminator of |message_.block_id|.
  opt::Instruction::OperandList new_terminator_in_operands;
  if (spv::Op(message_.opcode()) == spv::Op::OpReturnValue) {
    new_terminator_in_operands.push_back(
        {SPV_OPERAND_TYPE_ID, {message_.return_value_id()}});
  }
  auto terminator = block->terminator();
  terminator->SetOpcode(static_cast<spv::Op>(message_.opcode()));
  terminator->SetInOperands(std::move(new_terminator_in_operands));
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

std::unordered_set<uint32_t>
TransformationReplaceBranchFromDeadBlockWithExit::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

protobufs::Transformation
TransformationReplaceBranchFromDeadBlockWithExit::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_branch_from_dead_block_with_exit() = message_;
  return result;
}

bool TransformationReplaceBranchFromDeadBlockWithExit::BlockIsSuitable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const opt::BasicBlock& block) {
  // The block must be dead.
  if (!transformation_context.GetFactManager()->BlockIsDead(block.id())) {
    return false;
  }
  // The block's terminator must be OpBranch.
  if (block.terminator()->opcode() != spv::Op::OpBranch) {
    return false;
  }
  if (ir_context->GetStructuredCFGAnalysis()->IsInContinueConstruct(
          block.id())) {
    // Early exits from continue constructs are not allowed as they would break
    // the SPIR-V structured control flow rules.
    return false;
  }
  // We only allow changing OpBranch to an early terminator if the target of the
  // OpBranch has at least one other predecessor.
  auto successor = ir_context->get_instr_block(
      block.terminator()->GetSingleWordInOperand(0));
  if (ir_context->cfg()->preds(successor->id()).size() < 2) {
    return false;
  }
  // Make sure that domination rules are satisfied when we remove the branch
  // from the |block| to its |successor|.
  return fuzzerutil::NewTerminatorPreservesDominationRules(
      ir_context, block.id(), {ir_context, spv::Op::OpUnreachable});
}

}  // namespace fuzz
}  // namespace spvtools
