// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/fuzzer_pass_swap_functions.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation_swap_two_functions.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSwapFunctions::FuzzerPassSwapFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassSwapFunctions::Apply() {
  // Collect all function ids in a module.
  std::vector<uint32_t> function_ids;
  for (auto& function : *GetIRContext()->module()) {
    function_ids.emplace_back(function.result_id());
  }

  // Iterate through every combination of id i & j where i!=j.
  for (size_t i = 0; i < function_ids.size(); ++i) {
    for (size_t j = i + 1; j < function_ids.size(); ++j) {
      // Perform function swap randomly.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfSwappingFunctions())) {
        continue;
      }
      TransformationSwapTwoFunctions transformation(function_ids[i],
                                                    function_ids[j]);
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
