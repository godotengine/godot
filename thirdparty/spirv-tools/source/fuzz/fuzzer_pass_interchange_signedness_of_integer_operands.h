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

#ifndef SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_SIGNEDNESS_OF_INTEGER_OPERANDS_H_
#define SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_SIGNEDNESS_OF_INTEGER_OPERANDS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A pass that:
// - Finds all the integer constant (scalar and vector) definitions in the
//   module and adds the definitions of the integer with the same data words but
//   opposite signedness. If the synonym is already in the module, it does not
//   add a new one.
// - For each use of an integer constant where its signedness does not matter,
// decides whether to change it to the id of the toggled constant.
class FuzzerPassInterchangeSignednessOfIntegerOperands : public FuzzerPass {
 public:
  FuzzerPassInterchangeSignednessOfIntegerOperands(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Given the id of an integer constant (scalar or vector), it finds or creates
  // the corresponding toggled constant (the integer with the same data words
  // but opposite signedness). Returns the id of the toggled instruction if the
  // constant is an integer scalar or vector, 0 otherwise.
  uint32_t FindOrCreateToggledIntegerConstant(uint32_t id);
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_INTERCHANGE_SIGNEDNESS_OF_INTEGER_OPERANDS_H_
