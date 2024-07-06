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

#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

namespace spvtools {
namespace fuzz {

namespace {
const uint32_t kArithmeticInstructionIndexLeftInOperand = 0;
}  // namespace

FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::
    FuzzerPassReplaceAddsSubsMulsWithCarryingExtended(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::Apply() {
  std::vector<opt::Instruction> instructions_for_transformation;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // Randomly decide whether to apply the transformation.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingAddSubMulWithCarryingExtended())) {
          continue;
        }

        // Check if the transformation can be applied to this instruction.
        if (!TransformationReplaceAddSubMulWithCarryingExtended::
                IsInstructionSuitable(GetIRContext(), instruction)) {
          continue;
        }
        instructions_for_transformation.push_back(instruction);
      }
    }
  }
  for (auto& instruction : instructions_for_transformation) {
    // Get the operand type id. We know that both operands have the same
    // type.
    uint32_t operand_type_id =
        GetIRContext()
            ->get_def_use_mgr()
            ->GetDef(instruction.GetSingleWordInOperand(
                kArithmeticInstructionIndexLeftInOperand))
            ->type_id();

    // Ensure the required struct type exists. The struct type is based on
    // the operand type.
    FindOrCreateStructType({operand_type_id, operand_type_id});

    ApplyTransformation(TransformationReplaceAddSubMulWithCarryingExtended(
        GetFuzzerContext()->GetFreshId(), instruction.result_id()));
  }
}
}  // namespace fuzz
}  // namespace spvtools
