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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {
class TransformationAddLoopToCreateIntConstantSynonym : public Transformation {
 public:
  explicit TransformationAddLoopToCreateIntConstantSynonym(
      protobufs::TransformationAddLoopToCreateIntConstantSynonym message);

  TransformationAddLoopToCreateIntConstantSynonym(
      uint32_t constant_id, uint32_t initial_val_id, uint32_t step_val_id,
      uint32_t num_iterations_id, uint32_t block_after_loop_id, uint32_t syn_id,
      uint32_t loop_id, uint32_t ctr_id, uint32_t temp_id,
      uint32_t eventual_syn_id, uint32_t incremented_ctr_id, uint32_t cond_id,
      uint32_t additional_block_id);

  // - |message_.constant_id|, |message_.initial_value_id|,
  //   |message_.step_val_id| are integer constants (scalar or vectors) with the
  //   same type (with possibly different signedness, but same bit width, which
  //   must be <= 64). Let their value be C, I, S respectively.
  // - |message_.num_iterations_id| is a 32-bit integer scalar constant, with
  //   value N > 0 and N <= 32.
  // - The module contains 32-bit signed integer scalar constants of values 0
  //   and 1.
  // - The module contains the boolean type.
  // - C = I - S * N
  // - |message_.block_after_loop_id| is the label of a block which has a single
  //   predecessor and which is not a merge block, a continue block or a loop
  //   header.
  // - |message_.block_after_loop_id| must not be a dead block.
  // - |message_.additional_block_id| is either 0 or a valid fresh id, distinct
  //   from the other fresh ids.
  // - All of the other parameters are valid fresh ids.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a loop to the module, defining a synonym of an integer (scalar or
  // vector) constant. This id is marked as synonym with the original constant.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddLoopToCreateIntConstantSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_
