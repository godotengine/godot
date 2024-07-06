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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_
#define SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationMoveBlockDown : public Transformation {
 public:
  explicit TransformationMoveBlockDown(
      protobufs::TransformationMoveBlockDown message);

  explicit TransformationMoveBlockDown(uint32_t id);

  // - |message_.block_id| must be the id of a block b in the given module.
  // - b must not be the first nor last block appearing, in program order,
  //   in a function.
  // - b must not dominate the block that follows it in program order.
  // - b must be reachable.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // The block with id |message_.block_id| is moved down; i.e. the program order
  // between it and the block that follows it is swapped.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationMoveBlockDown message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_
