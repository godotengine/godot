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

#include "source/fuzz/fuzzer_pass_add_relaxed_decorations.h"

#include "source/fuzz/transformation_add_relaxed_decoration.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddRelaxedDecorations::FuzzerPassAddRelaxedDecorations(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddRelaxedDecorations::Apply() {
  // Consider every instruction in every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        // Randomly choose whether to apply the RelaxedPrecision decoration
        // to this instruction.
        if (GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingRelaxedDecoration())) {
          TransformationAddRelaxedDecoration transformation(inst.result_id());
          // Restrict attention to numeric instructions (returning 32-bit
          // floats or ints according to SPIR-V documentation) in dead blocks.
          if (transformation.IsApplicable(GetIRContext(),
                                          *GetTransformationContext())) {
            transformation.Apply(GetIRContext(), GetTransformationContext());
            *GetTransformations()->add_transformation() =
                transformation.ToMessage();
          }
        }
      }
    }
  }
}
}  // namespace fuzz
}  // namespace spvtools
