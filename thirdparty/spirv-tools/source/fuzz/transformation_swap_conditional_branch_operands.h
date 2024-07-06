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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SWAP_CONDITIONAL_BRANCH_OPERANDS_H_
#define SOURCE_FUZZ_TRANSFORMATION_SWAP_CONDITIONAL_BRANCH_OPERANDS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSwapConditionalBranchOperands : public Transformation {
 public:
  explicit TransformationSwapConditionalBranchOperands(
      protobufs::TransformationSwapConditionalBranchOperands message);

  TransformationSwapConditionalBranchOperands(
      const protobufs::InstructionDescriptor& instruction_descriptor,
      uint32_t fresh_id);

  // - |message_.instruction_descriptor| must be a valid descriptor of some
  //   OpBranchConditional instruction in the module.
  // - |message_.fresh_id| must be a fresh id.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts |%fresh_id = OpLogicalNot %bool_type_id %cond_id| before
  // |OpBranchConditional %cond_id %branch_a %branch_b [%weight_a %weight_b]|.
  // Replaces %cond_id with %fresh_id and swaps %branch_* and %weight_*
  // operands.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationSwapConditionalBranchOperands message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SWAP_CONDITIONAL_BRANCH_OPERANDS_H_
