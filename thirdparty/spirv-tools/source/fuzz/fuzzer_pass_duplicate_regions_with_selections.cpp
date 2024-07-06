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

#include "source/fuzz/fuzzer_pass_duplicate_regions_with_selections.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_duplicate_region_with_selection.h"

namespace spvtools {
namespace fuzz {

FuzzerPassDuplicateRegionsWithSelections::
    FuzzerPassDuplicateRegionsWithSelections(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassDuplicateRegionsWithSelections::Apply() {
  // Iterate over all of the functions in the module.
  for (auto& function : *GetIRContext()->module()) {
    // Randomly decide whether to apply the transformation.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfDuplicatingRegionWithSelection())) {
      continue;
    }
    std::vector<opt::BasicBlock*> candidate_entry_blocks;
    for (auto& block : function) {
      // We don't consider the first block to be the entry block, since it
      // could contain OpVariable instructions that would require additional
      // operations to be reassigned.
      // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3778):
      //     Consider extending this fuzzer pass to allow the first block to be
      //     used in duplication.
      if (&block == &*function.begin()) {
        continue;
      }
      candidate_entry_blocks.push_back(&block);
    }
    if (candidate_entry_blocks.empty()) {
      continue;
    }
    // Randomly choose the entry block.
    auto entry_block = candidate_entry_blocks[GetFuzzerContext()->RandomIndex(
        candidate_entry_blocks)];
    auto dominator_analysis = GetIRContext()->GetDominatorAnalysis(&function);
    auto postdominator_analysis =
        GetIRContext()->GetPostDominatorAnalysis(&function);
    std::vector<opt::BasicBlock*> candidate_exit_blocks;
    for (auto postdominates_entry_block = entry_block;
         postdominates_entry_block != nullptr;
         postdominates_entry_block = postdominator_analysis->ImmediateDominator(
             postdominates_entry_block)) {
      // The candidate exit block must be dominated by the entry block and the
      // entry block must be post-dominated by the candidate exit block. Ignore
      // the block if it heads a selection construct or a loop construct.
      if (dominator_analysis->Dominates(entry_block,
                                        postdominates_entry_block) &&
          !postdominates_entry_block->GetMergeInst()) {
        candidate_exit_blocks.push_back(postdominates_entry_block);
      }
    }
    if (candidate_exit_blocks.empty()) {
      continue;
    }
    // Randomly choose the exit block.
    auto exit_block = candidate_exit_blocks[GetFuzzerContext()->RandomIndex(
        candidate_exit_blocks)];

    auto region_blocks =
        TransformationDuplicateRegionWithSelection::GetRegionBlocks(
            GetIRContext(), entry_block, exit_block);

    // Construct |original_label_to_duplicate_label| by iterating over all
    // blocks in the region. Construct |original_id_to_duplicate_id| and
    // |original_id_to_phi_id| by iterating over all instructions in each block.
    std::map<uint32_t, uint32_t> original_label_to_duplicate_label;
    std::map<uint32_t, uint32_t> original_id_to_duplicate_id;
    std::map<uint32_t, uint32_t> original_id_to_phi_id;
    for (auto& block : region_blocks) {
      original_label_to_duplicate_label[block->id()] =
          GetFuzzerContext()->GetFreshId();
      for (auto& instr : *block) {
        if (instr.result_id()) {
          original_id_to_duplicate_id[instr.result_id()] =
              GetFuzzerContext()->GetFreshId();
          auto final_instruction = &*exit_block->tail();
          // &*exit_block->tail() is the final instruction of the region.
          // The instruction is available at the end of the region if and only
          // if it is available before this final instruction or it is the final
          // instruction.
          if ((&instr == final_instruction ||
               fuzzerutil::IdIsAvailableBeforeInstruction(
                   GetIRContext(), final_instruction, instr.result_id()))) {
            original_id_to_phi_id[instr.result_id()] =
                GetFuzzerContext()->GetFreshId();
          }
        }
      }
    }
    // Randomly decide between value "true" or "false" for a bool constant.
    // Make sure the transformation has access to a bool constant to be used
    // while creating conditional construct.
    auto condition_id =
        FindOrCreateBoolConstant(GetFuzzerContext()->ChooseEven(), true);

    TransformationDuplicateRegionWithSelection transformation =
        TransformationDuplicateRegionWithSelection(
            GetFuzzerContext()->GetFreshId(), condition_id,
            GetFuzzerContext()->GetFreshId(), entry_block->id(),
            exit_block->id(), original_label_to_duplicate_label,
            original_id_to_duplicate_id, original_id_to_phi_id);
    MaybeApplyTransformation(transformation);
  }
}
}  // namespace fuzz
}  // namespace spvtools
