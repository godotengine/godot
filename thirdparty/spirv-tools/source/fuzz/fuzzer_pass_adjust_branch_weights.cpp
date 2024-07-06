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

#include "source/fuzz/fuzzer_pass_adjust_branch_weights.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_adjust_branch_weights.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustBranchWeights::FuzzerPassAdjustBranchWeights(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAdjustBranchWeights::Apply() {
  // For all OpBranchConditional instructions,
  // randomly applies the transformation.
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    if (instruction->opcode() == spv::Op::OpBranchConditional &&
        GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAdjustingBranchWeights())) {
      ApplyTransformation(TransformationAdjustBranchWeights(
          MakeInstructionDescriptor(GetIRContext(), instruction),
          GetFuzzerContext()->GetRandomBranchWeights()));
    }
  });
}

}  // namespace fuzz
}  // namespace spvtools
