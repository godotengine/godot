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

#include "source/fuzz/fuzzer_pass_permute_instructions.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_move_instruction_down.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteInstructions::FuzzerPassPermuteInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPermuteInstructions::Apply() {
  // We are iterating over all instructions in all basic blocks.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We need to collect all instructions in the block into a separate vector
      // since application of the transformation below might invalidate
      // iterators.
      std::vector<opt::Instruction*> instructions;
      for (auto& instruction : block) {
        instructions.push_back(&instruction);
      }

      // We consider all instructions in reverse to increase the possible number
      // of applied transformations.
      for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfPermutingInstructions())) {
          continue;
        }

        while (MaybeApplyTransformation(TransformationMoveInstructionDown(
            MakeInstructionDescriptor(GetIRContext(), *it)))) {
          // Apply the transformation as many times as possible.
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
