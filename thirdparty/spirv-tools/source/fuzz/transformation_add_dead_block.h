// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BLOCK_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BLOCK_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddDeadBlock : public Transformation {
 public:
  explicit TransformationAddDeadBlock(
      protobufs::TransformationAddDeadBlock message);

  TransformationAddDeadBlock(uint32_t fresh_id, uint32_t existing_block,
                             bool condition_value);

  // - |message_.fresh_id| must be a fresh id
  // - A constant with the same value as |message_.condition_value| must be
  //   available
  // - |message_.existing_block| must be a block that is not a loop header,
  //   and that ends with OpBranch to a block that is not a merge block nor
  //   continue target - this is because the successor will become the merge
  //   block of a selection construct headed at |message_.existing_block|
  // - |message_.existing_block| must not be a back-edge block, since in this
  //   case the newly-added block would lead to another back-edge to the
  //   associated loop header
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Changes the OpBranch from |message_.existing_block| to its successor 's'
  // to an OpBranchConditional to either 's' or a new block,
  // |message_.fresh_id|, which itself unconditionally branches to 's'.  The
  // conditional branch uses |message.condition_value| as its condition, and is
  // arranged so that control will pass to 's' at runtime.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddDeadBlock message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BLOCK_H_
