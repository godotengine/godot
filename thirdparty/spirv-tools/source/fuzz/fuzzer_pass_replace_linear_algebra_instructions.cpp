// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_replace_linear_algebra_instructions.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_linear_algebra_instruction.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceLinearAlgebraInstructions::
    FuzzerPassReplaceLinearAlgebraInstructions(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceLinearAlgebraInstructions::Apply() {
  // For each instruction, checks whether it is a linear algebra instruction. In
  // this case, the transformation is randomly applied.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        if (!spvOpcodeIsLinearAlgebra(instruction.opcode())) {
          continue;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingLinearAlgebraInstructions())) {
          continue;
        }

        ApplyTransformation(TransformationReplaceLinearAlgebraInstruction(
            GetFuzzerContext()->GetFreshIds(
                TransformationReplaceLinearAlgebraInstruction::
                    GetRequiredFreshIdCount(GetIRContext(), &instruction)),
            MakeInstructionDescriptor(GetIRContext(), &instruction)));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
