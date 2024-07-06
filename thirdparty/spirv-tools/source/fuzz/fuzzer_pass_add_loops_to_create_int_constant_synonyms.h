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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_LOOPS_TO_CREATE_INT_CONSTANT_SYNONYMS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_LOOPS_TO_CREATE_INT_CONSTANT_SYNONYMS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that adds synonyms for integer, scalar or vector, constants, by
// adding loops that compute the same value by subtracting a value S from an
// initial value I, and for N times, so that C = I - S*N.
class FuzzerPassAddLoopsToCreateIntConstantSynonyms : public FuzzerPass {
 public:
  FuzzerPassAddLoopsToCreateIntConstantSynonyms(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Returns a pair (initial_val_id, step_val_id) such that both ids are
  // integer scalar constants of the same type as the scalar integer constant
  // identified by the given |constant_val|, |bit_width| and signedness, and
  // such that, if I is the value of initial_val_id, S is the value of
  // step_val_id and C is the value of the constant, the equation (C = I - S *
  // num_iterations) holds, (only considering the last |bit_width| bits of each
  // constant).
  std::pair<uint32_t, uint32_t> FindSuitableStepAndInitialValueConstants(
      uint64_t constant_val, uint32_t bit_width, bool is_signed,
      uint32_t num_iterations);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_LOOPS_TO_CREATE_INT_CONSTANT_SYNONYMS_H_
