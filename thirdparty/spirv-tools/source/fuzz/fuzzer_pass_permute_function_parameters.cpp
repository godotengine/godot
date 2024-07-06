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

#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_permute_function_parameters.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionParameters::FuzzerPassPermuteFunctionParameters(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPermuteFunctionParameters::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    uint32_t function_id = function.result_id();

    // Skip the function if it is an entry point
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(), function_id)) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfPermutingParameters())) {
      continue;
    }

    // Compute permutation for parameters
    auto* function_type =
        fuzzerutil::GetFunctionType(GetIRContext(), &function);
    assert(function_type && "Function type is null");

    // Don't take return type into account
    uint32_t arg_size = function_type->NumInOperands() - 1;

    // Create a vector, fill it with [0, n-1] values and shuffle it
    std::vector<uint32_t> permutation(arg_size);
    std::iota(permutation.begin(), permutation.end(), 0);
    GetFuzzerContext()->Shuffle(&permutation);

    // Apply our transformation
    ApplyTransformation(TransformationPermuteFunctionParameters(
        function_id, GetFuzzerContext()->GetFreshId(), permutation));
  }
}

}  // namespace fuzz
}  // namespace spvtools
