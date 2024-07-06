// Copyright (c) 2020 Stefano Milizia
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_ZERO_LIKE_CONSTANTS_H_
#define SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_ZERO_LIKE_CONSTANTS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A pass that:
// - Finds all the zero-like constant definitions in the module and adds the
//   definitions of the corresponding synonym, recording the fact that they
//   are synonymous. If the synonym is already in the module, it does not
//   add a new one.
// - For each use of a zero-like constant, decides whether to change it to the
//   id of the toggled constant.
class FuzzerPassInterchangeZeroLikeConstants : public FuzzerPass {
 public:
  FuzzerPassInterchangeZeroLikeConstants(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Given the declaration of a zero-like constant, it finds or creates the
  // corresponding toggled constant (a scalar constant of value 0 becomes a
  // null constant of the same type and vice versa).
  // Returns the id of the toggled instruction if the constant is zero-like,
  // 0 otherwise.
  uint32_t FindOrCreateToggledConstant(opt::Instruction* declaration);
};

}  // namespace fuzz
}  // namespace spvtools
#endif  // SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_ZERO_LIKE_CONSTANTS_H_
