// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/fuzzer_pass_permute_function_variables.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_swap_function_variables.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionVariables::FuzzerPassPermuteFunctionVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {
}  // Here we call parent constructor.

void FuzzerPassPermuteFunctionVariables::Apply() {
  // Permuting OpVariable instructions in each function.
  for (auto& function : *GetIRContext()->module()) {
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfPermutingFunctionVariables())) {
      continue;
    }

    auto first_block = function.entry().get();

    std::vector<opt::Instruction*> variables;
    for (auto& instruction : *first_block) {
      if (instruction.opcode() == spv::Op::OpVariable) {
        variables.push_back(&instruction);
      }
    }
    if (variables.size() <= 1) {
      continue;
    }
    do {
      uint32_t instruction_1_index = GetFuzzerContext()->RandomIndex(variables);
      uint32_t instruction_2_index = GetFuzzerContext()->RandomIndex(variables);

      if (instruction_1_index != instruction_2_index) {
        ApplyTransformation(TransformationSwapFunctionVariables(
            variables[instruction_1_index]->result_id(),
            variables[instruction_2_index]->result_id()));
      }

    } while (GetFuzzerContext()->ChoosePercentage(
                 GetFuzzerContext()
                     ->GetChanceOfSwappingAnotherPairOfFunctionVariables()) &&
             variables.size() > 2);
  }
}

}  // namespace fuzz
}  // namespace spvtools
