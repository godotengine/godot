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

#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_invert_comparison_operator.h"

namespace spvtools {
namespace fuzz {

FuzzerPassInvertComparisonOperators::FuzzerPassInvertComparisonOperators(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassInvertComparisonOperators::Apply() {
  GetIRContext()->module()->ForEachInst([this](const opt::Instruction* inst) {
    if (!TransformationInvertComparisonOperator::IsInversionSupported(
            inst->opcode())) {
      return;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfInvertingComparisonOperators())) {
      return;
    }

    ApplyTransformation(TransformationInvertComparisonOperator(
        inst->result_id(), GetFuzzerContext()->GetFreshId()));
  });
}

}  // namespace fuzz
}  // namespace spvtools
