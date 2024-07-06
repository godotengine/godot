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

#include "source/fuzz/transformation_wrap_region_in_selection.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationWrapRegionInSelection::TransformationWrapRegionInSelection(
    protobufs::TransformationWrapRegionInSelection message)
    : message_(std::move(message)) {}

TransformationWrapRegionInSelection::TransformationWrapRegionInSelection(
    uint32_t region_entry_block_id, uint32_t region_exit_block_id,
    bool branch_condition) {
  message_.set_region_entry_block_id(region_entry_block_id);
  message_.set_region_exit_block_id(region_exit_block_id);
  message_.set_branch_condition(branch_condition);
}

bool TransformationWrapRegionInSelection::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that it is possible to outline a region of blocks without breaking
  // domination and structured control flow rules.
  if (!IsApplicableToBlockRange(ir_context, message_.region_entry_block_id(),
                                message_.region_exit_block_id())) {
    return false;
  }

  // There must exist an irrelevant boolean constant to be used as a condition
  // in the OpBranchConditional instruction.
  return fuzzerutil::MaybeGetBoolConstant(ir_context, transformation_context,
                                          message_.branch_condition(),
                                          true) != 0;
}

void TransformationWrapRegionInSelection::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* new_header_block =
      ir_context->cfg()->block(message_.region_entry_block_id());
  assert(new_header_block->terminator()->opcode() == spv::Op::OpBranch &&
         "This condition should have been checked in the IsApplicable");

  const auto successor_id =
      new_header_block->terminator()->GetSingleWordInOperand(0);

  // Change |entry_block|'s terminator to |OpBranchConditional|.
  new_header_block->terminator()->SetOpcode(spv::Op::OpBranchConditional);
  new_header_block->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID,
        {fuzzerutil::MaybeGetBoolConstant(ir_context, *transformation_context,
                                          message_.branch_condition(), true)}},
       {SPV_OPERAND_TYPE_ID, {successor_id}},
       {SPV_OPERAND_TYPE_ID, {successor_id}}});

  // Insert OpSelectionMerge before the terminator.
  new_header_block->terminator()->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.region_exit_block_id()}},
          {SPV_OPERAND_TYPE_SELECTION_CONTROL,
           {uint32_t(spv::SelectionControlMask::MaskNone)}}}));

  // We've change the module so we must invalidate analyses.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationWrapRegionInSelection::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_wrap_region_in_selection() = message_;
  return result;
}

bool TransformationWrapRegionInSelection::IsApplicableToBlockRange(
    opt::IRContext* ir_context, uint32_t header_block_candidate_id,
    uint32_t merge_block_candidate_id) {
  // Check that |header_block_candidate_id| and |merge_block_candidate_id| are
  // valid.
  const auto* header_block_candidate =
      fuzzerutil::MaybeFindBlock(ir_context, header_block_candidate_id);
  if (!header_block_candidate) {
    return false;
  }

  const auto* merge_block_candidate =
      fuzzerutil::MaybeFindBlock(ir_context, merge_block_candidate_id);
  if (!merge_block_candidate) {
    return false;
  }

  // |header_block_candidate| and |merge_block_candidate| must be from the same
  // function.
  if (header_block_candidate->GetParent() !=
      merge_block_candidate->GetParent()) {
    return false;
  }

  const auto* dominator_analysis =
      ir_context->GetDominatorAnalysis(header_block_candidate->GetParent());
  const auto* postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(header_block_candidate->GetParent());

  if (!dominator_analysis->StrictlyDominates(header_block_candidate,
                                             merge_block_candidate) ||
      !postdominator_analysis->StrictlyDominates(merge_block_candidate,
                                                 header_block_candidate)) {
    return false;
  }

  // |header_block_candidate| can't be a header since we are about to make it
  // one.
  if (header_block_candidate->GetMergeInst()) {
    return false;
  }

  // |header_block_candidate| must have an OpBranch terminator.
  if (header_block_candidate->terminator()->opcode() != spv::Op::OpBranch) {
    return false;
  }

  // Every header block must have a unique merge block. Thus,
  // |merge_block_candidate| can't be a merge block of some other header.
  auto* structured_cfg = ir_context->GetStructuredCFGAnalysis();
  if (structured_cfg->IsMergeBlock(merge_block_candidate_id)) {
    return false;
  }

  // |header_block_candidate|'s containing construct must also contain
  // |merge_block_candidate|.
  //
  // ContainingConstruct will return the id of a loop header for a block in the
  // loop's continue construct. Thus, we must also check the case when one of
  // the candidates is in continue construct and the other one is not.
  if (structured_cfg->ContainingConstruct(header_block_candidate_id) !=
          structured_cfg->ContainingConstruct(merge_block_candidate_id) ||
      structured_cfg->IsInContinueConstruct(header_block_candidate_id) !=
          structured_cfg->IsInContinueConstruct(merge_block_candidate_id)) {
    return false;
  }

  return true;
}

std::unordered_set<uint32_t> TransformationWrapRegionInSelection::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
