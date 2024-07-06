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

#include "source/fuzz/fuzzer_pass_propagate_instructions_down.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation_propagate_instruction_down.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPropagateInstructionsDown::FuzzerPassPropagateInstructionsDown(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPropagateInstructionsDown::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    std::vector<const opt::BasicBlock*> reachable_blocks;
    for (const auto& block : function) {
      if (GetIRContext()->IsReachable(block)) {
        reachable_blocks.push_back(&block);
      }
    }

    for (const auto* block : reachable_blocks) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfPropagatingInstructionsDown())) {
        continue;
      }

      if (TransformationPropagateInstructionDown::IsApplicableToBlock(
              GetIRContext(), block->id())) {
        // Record fresh ids for every successor of the |block| that we can
        // propagate an instruction into.
        std::map<uint32_t, uint32_t> fresh_ids;
        for (auto id :
             TransformationPropagateInstructionDown::GetAcceptableSuccessors(
                 GetIRContext(), block->id())) {
          fresh_ids[id] = GetFuzzerContext()->GetFreshId();
        }

        ApplyTransformation(TransformationPropagateInstructionDown(
            block->id(), GetFuzzerContext()->GetFreshId(), fresh_ids));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
