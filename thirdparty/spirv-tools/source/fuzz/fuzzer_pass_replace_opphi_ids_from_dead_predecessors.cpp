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

#include "source/fuzz/fuzzer_pass_replace_opphi_ids_from_dead_predecessors.h"

#include "source/fuzz/transformation_replace_opphi_id_from_dead_predecessor.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceOpPhiIdsFromDeadPredecessors::
    FuzzerPassReplaceOpPhiIdsFromDeadPredecessors(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceOpPhiIdsFromDeadPredecessors::Apply() {
  // Keep a vector of the transformations to apply.
  std::vector<TransformationReplaceOpPhiIdFromDeadPredecessor> transformations;

  // Loop through the reachable blocks in the module.
  for (auto& function : *GetIRContext()->module()) {
    GetIRContext()->cfg()->ForEachBlockInPostOrder(
        &*function.begin(),
        [this, &function, &transformations](opt::BasicBlock* block) {
          // Only consider dead blocks.
          if (!GetTransformationContext()->GetFactManager()->BlockIsDead(
                  block->id())) {
            return;
          }

          // Find all the uses of the label id of the block inside OpPhi
          // instructions.
          GetIRContext()->get_def_use_mgr()->ForEachUse(
              block->id(), [this, &function, block, &transformations](
                               opt::Instruction* instruction, uint32_t) {
                // Only consider OpPhi instructions.
                if (instruction->opcode() != spv::Op::OpPhi) {
                  return;
                }

                // Randomly decide whether to consider this use.
                if (!GetFuzzerContext()->ChoosePercentage(
                        GetFuzzerContext()
                            ->GetChanceOfReplacingOpPhiIdFromDeadPredecessor())) {
                  return;
                }

                // Get the current id corresponding to the predecessor.
                uint32_t current_id = 0;
                for (uint32_t i = 1; i < instruction->NumInOperands(); i += 2) {
                  if (instruction->GetSingleWordInOperand(i) == block->id()) {
                    // The corresponding id is at the index of the block - 1.
                    current_id = instruction->GetSingleWordInOperand(i - 1);
                    break;
                  }
                }
                assert(
                    current_id != 0 &&
                    "The predecessor - and corresponding id - should always be "
                    "found.");

                uint32_t type_id = instruction->type_id();

                // Find all the suitable instructions to replace the id.
                const auto& candidates = FindAvailableInstructions(
                    &function, block, block->end(),
                    [type_id, current_id](opt::IRContext* /* unused */,
                                          opt::Instruction* candidate) -> bool {
                      // Only consider instructions with a result id different
                      // from the currently-used one, and with the right type.
                      return candidate->HasResultId() &&
                             candidate->type_id() == type_id &&
                             candidate->result_id() != current_id;
                    });

                // If there is no possible replacement, we cannot apply any
                // transformation.
                if (candidates.empty()) {
                  return;
                }

                // Choose one of the candidates.
                uint32_t replacement_id =
                    candidates[GetFuzzerContext()->RandomIndex(candidates)]
                        ->result_id();

                // Add a new transformation to the list of transformations to
                // apply.
                transformations.emplace_back(
                    TransformationReplaceOpPhiIdFromDeadPredecessor(
                        instruction->result_id(), block->id(), replacement_id));
              });
        });
  }

  // Apply all the transformations.
  for (const auto& transformation : transformations) {
    ApplyTransformation(transformation);
  }
}

}  // namespace fuzz
}  // namespace spvtools
