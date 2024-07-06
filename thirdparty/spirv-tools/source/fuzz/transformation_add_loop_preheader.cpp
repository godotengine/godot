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

#include "transformation_add_loop_preheader.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {
TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    protobufs::TransformationAddLoopPreheader message)
    : message_(std::move(message)) {}

TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    uint32_t loop_header_block, uint32_t fresh_id,
    std::vector<uint32_t> phi_id) {
  message_.set_loop_header_block(loop_header_block);
  message_.set_fresh_id(fresh_id);
  for (auto id : phi_id) {
    message_.add_phi_id(id);
  }
}

bool TransformationAddLoopPreheader::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  // |message_.loop_header_block()| must be the id of a loop header block.
  opt::BasicBlock* loop_header_block =
      fuzzerutil::MaybeFindBlock(ir_context, message_.loop_header_block());
  if (!loop_header_block || !loop_header_block->IsLoopHeader()) {
    return false;
  }

  // The id for the preheader must actually be fresh.
  std::set<uint32_t> used_ids;
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(message_.fresh_id(),
                                                    ir_context, &used_ids)) {
    return false;
  }

  size_t num_predecessors =
      ir_context->cfg()->preds(message_.loop_header_block()).size();

  // The block must have at least 2 predecessors (the back-edge block and
  // another predecessor outside of the loop)
  if (num_predecessors < 2) {
    return false;
  }

  // If the block only has one predecessor outside of the loop (and thus 2 in
  // total), then no additional fresh ids are necessary.
  if (num_predecessors == 2) {
    return true;
  }

  // Count the number of OpPhi instructions.
  int32_t num_phi_insts = 0;
  loop_header_block->ForEachPhiInst(
      [&num_phi_insts](opt::Instruction* /* unused */) { num_phi_insts++; });

  // There must be enough fresh ids for the OpPhi instructions.
  if (num_phi_insts > message_.phi_id_size()) {
    return false;
  }

  // Check that the needed ids are fresh and distinct.
  for (int32_t i = 0; i < num_phi_insts; i++) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(message_.phi_id(i),
                                                      ir_context, &used_ids)) {
      return false;
    }
  }

  return true;
}

void TransformationAddLoopPreheader::Apply(
    opt::IRContext* ir_context,
    TransformationContext* /* transformation_context */) const {
  // Find the loop header.
  opt::BasicBlock* loop_header =
      fuzzerutil::MaybeFindBlock(ir_context, message_.loop_header_block());

  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(loop_header->GetParent());

  uint32_t back_edge_block_id = 0;

  // Update the branching instructions of the out-of-loop predecessors of the
  // header. Set |back_edge_block_id| to be the id of the back-edge block.
  ir_context->get_def_use_mgr()->ForEachUse(
      loop_header->id(),
      [this, &ir_context, &dominator_analysis, &loop_header,
       &back_edge_block_id](opt::Instruction* use_inst, uint32_t use_index) {
        if (dominator_analysis->Dominates(loop_header->GetLabelInst(),
                                          use_inst)) {
          // If |use_inst| is a branch instruction dominated by the header, the
          // block containing it is the back-edge block.
          if (use_inst->IsBranch()) {
            assert(back_edge_block_id == 0 &&
                   "There should only be one back-edge block");
            back_edge_block_id = ir_context->get_instr_block(use_inst)->id();
          }
          // References to the header inside the loop should not be updated
          return;
        }

        // If |use_inst| is not a branch or merge instruction, it should not be
        // changed.
        if (!use_inst->IsBranch() &&
            use_inst->opcode() != spv::Op::OpSelectionMerge &&
            use_inst->opcode() != spv::Op::OpLoopMerge) {
          return;
        }

        // Update the reference.
        use_inst->SetOperand(use_index, {message_.fresh_id()});
      });

  assert(back_edge_block_id && "The back-edge block should have been found");

  // Make a new block for the preheader.
  std::unique_ptr<opt::BasicBlock> preheader = MakeUnique<opt::BasicBlock>(
      std::unique_ptr<opt::Instruction>(new opt::Instruction(
          ir_context, spv::Op::OpLabel, 0, message_.fresh_id(), {})));

  uint32_t phi_ids_used = 0;

  // Update the OpPhi instructions and, if there is more than one out-of-loop
  // predecessor, add necessary OpPhi instructions so the preheader.
  loop_header->ForEachPhiInst([this, &ir_context, &preheader,
                               &back_edge_block_id,
                               &phi_ids_used](opt::Instruction* phi_inst) {
    // The loop header must have at least 2 incoming edges (the back edge, and
    // at least one from outside the loop).
    assert(phi_inst->NumInOperands() >= 4);

    if (phi_inst->NumInOperands() == 4) {
      // There is just one out-of-loop predecessor, so no additional
      // instructions in the preheader are necessary. The reference to the
      // original out-of-loop predecessor needs to be updated so that it refers
      // to the preheader.
      uint32_t index_of_out_of_loop_pred_id =
          phi_inst->GetInOperand(1).words[0] == back_edge_block_id ? 3 : 1;
      phi_inst->SetInOperand(index_of_out_of_loop_pred_id, {preheader->id()});
    } else {
      // There is more than one out-of-loop predecessor, so an OpPhi instruction
      // needs to be added to the preheader, and its value will depend on all
      // the current out-of-loop predecessors of the header.

      // Get the operand list and the value corresponding to the back-edge
      // block.
      std::vector<opt::Operand> preheader_in_operands;
      uint32_t back_edge_val = 0;

      for (uint32_t i = 0; i < phi_inst->NumInOperands(); i += 2) {
        // Only add operands if they don't refer to the back-edge block.
        if (phi_inst->GetInOperand(i + 1).words[0] == back_edge_block_id) {
          back_edge_val = phi_inst->GetInOperand(i).words[0];
        } else {
          preheader_in_operands.push_back(std::move(phi_inst->GetInOperand(i)));
          preheader_in_operands.push_back(
              std::move(phi_inst->GetInOperand(i + 1)));
        }
      }

      // Add the new instruction to the preheader.
      uint32_t fresh_phi_id = message_.phi_id(phi_ids_used++);

      // Update id bound.
      fuzzerutil::UpdateModuleIdBound(ir_context, fresh_phi_id);

      preheader->AddInstruction(std::unique_ptr<opt::Instruction>(
          new opt::Instruction(ir_context, spv::Op::OpPhi, phi_inst->type_id(),
                               fresh_phi_id, preheader_in_operands)));

      // Update the OpPhi instruction in the header so that it refers to the
      // back edge block and the preheader as the predecessors, and it uses the
      // newly-defined OpPhi in the preheader for the corresponding value.
      phi_inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {fresh_phi_id}},
                               {SPV_OPERAND_TYPE_ID, {preheader->id()}},
                               {SPV_OPERAND_TYPE_ID, {back_edge_val}},
                               {SPV_OPERAND_TYPE_ID, {back_edge_block_id}}});
    }
  });

  // Update id bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Add an unconditional branch from the preheader to the header.
  preheader->AddInstruction(
      std::unique_ptr<opt::Instruction>(new opt::Instruction(
          ir_context, spv::Op::OpBranch, 0, 0,
          std::initializer_list<opt::Operand>{opt::Operand(
              spv_operand_type_t::SPV_OPERAND_TYPE_ID, {loop_header->id()})})));

  // Insert the preheader in the module.
  loop_header->GetParent()->InsertBasicBlockBefore(std::move(preheader),
                                                   loop_header);

  // Invalidate analyses because the structure of the program changed.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationAddLoopPreheader::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_loop_preheader() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddLoopPreheader::GetFreshIds()
    const {
  std::unordered_set<uint32_t> result = {message_.fresh_id()};
  for (auto id : message_.phi_id()) {
    result.insert(id);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
