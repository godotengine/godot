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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceOpSelectWithConditionalBranch
    : public Transformation {
 public:
  explicit TransformationReplaceOpSelectWithConditionalBranch(
      protobufs::TransformationReplaceOpSelectWithConditionalBranch message);

  TransformationReplaceOpSelectWithConditionalBranch(uint32_t select_id,
                                                     uint32_t true_block_id,
                                                     uint32_t false_block_id);

  // - |message_.select_id| is the result id of an OpSelect instruction.
  // - The condition of the OpSelect must be a scalar boolean.
  // - The OpSelect instruction is the first instruction in its block.
  // - The block containing the instruction is not a merge block, and it has a
  //   single predecessor, which is not a header and whose last instruction is
  //   OpBranch.
  // - Each of |message_.true_block_id| and |message_.false_block_id| is either
  //   0 or a valid fresh id, and at most one of them is 0. They must be
  //   distinct.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the OpSelect instruction with id |message_.select_id| with a
  // conditional branch and an OpPhi instruction.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceOpSelectWithConditionalBranch message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H
