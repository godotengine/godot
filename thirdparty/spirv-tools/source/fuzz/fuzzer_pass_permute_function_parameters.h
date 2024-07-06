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

#ifndef SOURCE_FUZZ_FUZZER_PASS_PERMUTE_FUNCTION_PARAMETERS_H_
#define SOURCE_FUZZ_FUZZER_PASS_PERMUTE_FUNCTION_PARAMETERS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that, given a non-entry-point function taking n parameters
// and a permutation of the set [0, n - 1]:
//   1. Introduces a new function type that is the same as the original
//      function's type but with the order of arguments permuted
//      (only add this if it doesn't already exist)
//   2. Changes the type of the function to this type
//   3. Adjusts all calls to the function so that their arguments are permuted
class FuzzerPassPermuteFunctionParameters : public FuzzerPass {
 public:
  FuzzerPassPermuteFunctionParameters(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_PERMUTE_FUNCTION_PARAMETERS_H_
