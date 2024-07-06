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

#include "source/fuzz/fuzzer_pass_add_bit_instruction_synonyms.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_bit_instruction_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddBitInstructionSynonyms::FuzzerPassAddBitInstructionSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddBitInstructionSynonyms::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // This fuzzer pass can add a *lot* of ids.  We bail out early if we hit
        // the recommended id limit.
        if (GetIRContext()->module()->id_bound() >=
            GetFuzzerContext()->GetIdBoundLimit()) {
          return;
        }

        // Randomly decides whether the transformation will be applied.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingBitInstructionSynonym())) {
          continue;
        }

        // Make sure fuzzer never applies a transformation to a bitwise
        // instruction with differently signed operands, only integer operands
        // are supported and bitwise operations are supported only.
        if (!TransformationAddBitInstructionSynonym::IsInstructionSupported(
                GetIRContext(), &instruction)) {
          continue;
        }

        // Make sure all bit indexes are defined as 32-bit unsigned integers.
        uint32_t width = GetIRContext()
                             ->get_type_mgr()
                             ->GetType(instruction.type_id())
                             ->AsInteger()
                             ->width();
        for (uint32_t i = 0; i < width; i++) {
          FindOrCreateIntegerConstant({i}, 32, false, false);
        }

        // Applies the add bit instruction synonym transformation.
        ApplyTransformation(TransformationAddBitInstructionSynonym(
            instruction.result_id(),
            GetFuzzerContext()->GetFreshIds(
                TransformationAddBitInstructionSynonym::GetRequiredFreshIdCount(
                    GetIRContext(), &instruction))));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
