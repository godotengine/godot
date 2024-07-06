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

#ifndef SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H
#define SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

class FuzzerPassFlattenConditionalBranches : public FuzzerPass {
 public:
  FuzzerPassFlattenConditionalBranches(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // If the SPIR-V version requires vector OpSelects to be component-wise, or
  // if |use_vector_select_if_optional| holds, |fresh_id_for_bvec_selector| is
  // populated with a fresh id if it is currently zero, and a
  // |vector_dimension|-dimensional boolean vector type is added to the module
  // if not already present.
  void PrepareForOpPhiOnVectors(uint32_t vector_dimension,
                                bool use_vector_select_if_optional,
                                uint32_t* fresh_id_for_bvec_selector);
};
}  // namespace fuzz
}  // namespace spvtools
#endif  // SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H
