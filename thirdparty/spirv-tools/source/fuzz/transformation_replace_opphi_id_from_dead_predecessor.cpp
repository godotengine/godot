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

#include "source/fuzz/transformation_replace_opphi_id_from_dead_predecessor.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceOpPhiIdFromDeadPredecessor::
    TransformationReplaceOpPhiIdFromDeadPredecessor(
        protobufs::TransformationReplaceOpPhiIdFromDeadPredecessor message)
    : message_(std::move(message)) {}

TransformationReplaceOpPhiIdFromDeadPredecessor::
    TransformationReplaceOpPhiIdFromDeadPredecessor(uint32_t opphi_id,
                                                    uint32_t pred_label_id,
                                                    uint32_t replacement_id) {
  message_.set_opphi_id(opphi_id);
  message_.set_pred_label_id(pred_label_id);
  message_.set_replacement_id(replacement_id);
}

bool TransformationReplaceOpPhiIdFromDeadPredecessor::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // |opphi_id| must be the id of an OpPhi instruction.
  auto opphi_def = ir_context->get_def_use_mgr()->GetDef(message_.opphi_id());
  if (!opphi_def || opphi_def->opcode() != spv::Op::OpPhi) {
    return false;
  }

  // |pred_label_id| must be the label id of a dead block.
  auto pred_block = ir_context->get_instr_block(message_.pred_label_id());
  if (!pred_block || pred_block->id() != message_.pred_label_id() ||
      !transformation_context.GetFactManager()->BlockIsDead(pred_block->id())) {
    return false;
  }

  // |pred_label_id| must be one of the predecessors of the block containing the
  // OpPhi instruction.
  bool found = false;
  for (auto pred :
       ir_context->cfg()->preds(ir_context->get_instr_block(opphi_def)->id())) {
    if (pred == message_.pred_label_id()) {
      found = true;
      break;
    }
  }

  if (!found) {
    return false;
  }

  // |replacement_id| must have the same type id as the OpPhi instruction.
  auto replacement_def =
      ir_context->get_def_use_mgr()->GetDef(message_.replacement_id());

  if (!replacement_def || replacement_def->type_id() != opphi_def->type_id()) {
    return false;
  }

  // The replacement id must be available at the end of the predecessor.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, pred_block->terminator(), replacement_def->result_id());
}

void TransformationReplaceOpPhiIdFromDeadPredecessor::Apply(
    opt::IRContext* ir_context,
    TransformationContext* /* transformation_context */) const {
  // Get the OpPhi instruction.
  auto opphi_def = ir_context->get_def_use_mgr()->GetDef(message_.opphi_id());

  // Find the index corresponding to the operand being replaced and replace it,
  // by looping through the odd-indexed input operands and finding
  // |pred_label_id|. The index that we are interested in is the one before
  // that.
  for (uint32_t i = 1; i < opphi_def->NumInOperands(); i += 2) {
    if (opphi_def->GetSingleWordInOperand(i) == message_.pred_label_id()) {
      // The operand to be replaced is at index i-1.
      opphi_def->SetInOperand(i - 1, {message_.replacement_id()});
    }
  }

  // Invalidate the analyses because we have altered the usages of ids.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceOpPhiIdFromDeadPredecessor::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opphi_id_from_dead_predecessor() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceOpPhiIdFromDeadPredecessor::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
