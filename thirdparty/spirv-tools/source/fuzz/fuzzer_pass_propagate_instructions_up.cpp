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

#include "source/fuzz/fuzzer_pass_propagate_instructions_up.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_propagate_instruction_up.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPropagateInstructionsUp::FuzzerPassPropagateInstructionsUp(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPropagateInstructionsUp::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    for (const auto& block : function) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfPropagatingInstructionsUp())) {
        continue;
      }

      if (TransformationPropagateInstructionUp::IsApplicableToBlock(
              GetIRContext(), block.id())) {
        std::map<uint32_t, uint32_t> fresh_ids;
        for (auto id : GetIRContext()->cfg()->preds(block.id())) {
          auto& fresh_id = fresh_ids[id];

          if (!fresh_id) {
            // Create a fresh id if it hasn't been created yet. |fresh_id| will
            // be default-initialized to 0 in this case.
            fresh_id = GetFuzzerContext()->GetFreshId();
          }
        }
        ApplyTransformation(
            TransformationPropagateInstructionUp(block.id(), fresh_ids));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
