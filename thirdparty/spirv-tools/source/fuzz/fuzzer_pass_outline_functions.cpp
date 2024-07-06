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

#include "source/fuzz/fuzzer_pass_outline_functions.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_outline_function.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassOutlineFunctions::FuzzerPassOutlineFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassOutlineFunctions::Apply() {
  std::vector<opt::Function*> original_functions;
  for (auto& function : *GetIRContext()->module()) {
    original_functions.push_back(&function);
  }
  for (auto& function : original_functions) {
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfOutliningFunction())) {
      continue;
    }
    std::vector<opt::BasicBlock*> blocks;
    for (auto& block : *function) {
      blocks.push_back(&block);
    }
    auto entry_block = MaybeGetEntryBlockSuitableForOutlining(
        blocks[GetFuzzerContext()->RandomIndex(blocks)]);

    if (!entry_block) {
      // The chosen block is not suitable to be the entry block of a region that
      // will be outlined.
      continue;
    }

    auto dominator_analysis = GetIRContext()->GetDominatorAnalysis(function);
    auto postdominator_analysis =
        GetIRContext()->GetPostDominatorAnalysis(function);
    std::vector<opt::BasicBlock*> candidate_exit_blocks;
    for (auto postdominates_entry_block = entry_block;
         postdominates_entry_block != nullptr;
         postdominates_entry_block = postdominator_analysis->ImmediateDominator(
             postdominates_entry_block)) {
      // Consider the block if it is dominated by the entry block, ignore it if
      // it is a continue target.
      if (dominator_analysis->Dominates(entry_block,
                                        postdominates_entry_block) &&
          !GetIRContext()->GetStructuredCFGAnalysis()->IsContinueBlock(
              postdominates_entry_block->id())) {
        candidate_exit_blocks.push_back(postdominates_entry_block);
      }
    }
    if (candidate_exit_blocks.empty()) {
      continue;
    }
    auto exit_block = MaybeGetExitBlockSuitableForOutlining(
        candidate_exit_blocks[GetFuzzerContext()->RandomIndex(
            candidate_exit_blocks)]);

    if (!exit_block) {
      // The block chosen is not suitable
      continue;
    }

    auto region_blocks = TransformationOutlineFunction::GetRegionBlocks(
        GetIRContext(), entry_block, exit_block);
    std::map<uint32_t, uint32_t> input_id_to_fresh_id;
    for (auto id : TransformationOutlineFunction::GetRegionInputIds(
             GetIRContext(), region_blocks, exit_block)) {
      input_id_to_fresh_id[id] = GetFuzzerContext()->GetFreshId();
    }
    std::map<uint32_t, uint32_t> output_id_to_fresh_id;
    for (auto id : TransformationOutlineFunction::GetRegionOutputIds(
             GetIRContext(), region_blocks, exit_block)) {
      output_id_to_fresh_id[id] = GetFuzzerContext()->GetFreshId();
    }
    TransformationOutlineFunction transformation(
        entry_block->id(), exit_block->id(),
        /*new_function_struct_return_type_id*/
        GetFuzzerContext()->GetFreshId(),
        /*new_function_type_id*/ GetFuzzerContext()->GetFreshId(),
        /*new_function_id*/ GetFuzzerContext()->GetFreshId(),
        /*new_function_region_entry_block*/
        GetFuzzerContext()->GetFreshId(),
        /*new_caller_result_id*/ GetFuzzerContext()->GetFreshId(),
        /*new_callee_result_id*/ GetFuzzerContext()->GetFreshId(),
        /*input_id_to_fresh_id*/ input_id_to_fresh_id,
        /*output_id_to_fresh_id*/ output_id_to_fresh_id);
    MaybeApplyTransformation(transformation);
  }
}

opt::BasicBlock*
FuzzerPassOutlineFunctions::MaybeGetEntryBlockSuitableForOutlining(
    opt::BasicBlock* entry_block) {
  // If the entry block is a loop header, we need to get or create its
  // preheader and make it the entry block, if possible.
  if (entry_block->IsLoopHeader()) {
    auto predecessors =
        GetIRContext()->cfg()->preds(entry_block->GetLabel()->result_id());

    if (predecessors.size() < 2) {
      // The header only has one predecessor (the back-edge block) and thus
      // it is unreachable. The block cannot be adjusted to be suitable for
      // outlining.
      return nullptr;
    }

    // Get or create a suitable preheader and make it become the entry block.
    entry_block =
        GetOrCreateSimpleLoopPreheader(entry_block->GetLabel()->result_id());
  }

  assert(!entry_block->IsLoopHeader() &&
         "The entry block cannot be a loop header at this point.");

  // If the entry block starts with OpPhi or OpVariable, try to split it.
  if (entry_block->begin()->opcode() == spv::Op::OpPhi ||
      entry_block->begin()->opcode() == spv::Op::OpVariable) {
    // Find the first non-OpPhi and non-OpVariable instruction.
    auto non_phi_or_var_inst = &*entry_block->begin();
    while (non_phi_or_var_inst->opcode() == spv::Op::OpPhi ||
           non_phi_or_var_inst->opcode() == spv::Op::OpVariable) {
      non_phi_or_var_inst = non_phi_or_var_inst->NextNode();
    }

    // Split the block.
    uint32_t new_block_id = GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationSplitBlock(
        MakeInstructionDescriptor(GetIRContext(), non_phi_or_var_inst),
        new_block_id));

    // The new entry block is the newly-created block.
    entry_block = &*entry_block->GetParent()->FindBlock(new_block_id);
  }

  return entry_block;
}

opt::BasicBlock*
FuzzerPassOutlineFunctions::MaybeGetExitBlockSuitableForOutlining(
    opt::BasicBlock* exit_block) {
  // The exit block must not be a continue target.
  assert(!GetIRContext()->GetStructuredCFGAnalysis()->IsContinueBlock(
             exit_block->id()) &&
         "A candidate exit block cannot be a continue target.");

  // If the exit block is a merge block, try to split it and return the second
  // block in the pair as the exit block.
  if (GetIRContext()->GetStructuredCFGAnalysis()->IsMergeBlock(
          exit_block->id())) {
    uint32_t new_block_id = GetFuzzerContext()->GetFreshId();

    // Find the first non-OpPhi instruction, after which to split.
    auto split_before = &*exit_block->begin();
    while (split_before->opcode() == spv::Op::OpPhi) {
      split_before = split_before->NextNode();
    }

    if (!MaybeApplyTransformation(TransformationSplitBlock(
            MakeInstructionDescriptor(GetIRContext(), split_before),
            new_block_id))) {
      return nullptr;
    }

    return &*exit_block->GetParent()->FindBlock(new_block_id);
  }

  return exit_block;
}

}  // namespace fuzz
}  // namespace spvtools
