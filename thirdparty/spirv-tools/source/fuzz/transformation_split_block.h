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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_
#define SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSplitBlock : public Transformation {
 public:
  explicit TransformationSplitBlock(
      protobufs::TransformationSplitBlock message);

  TransformationSplitBlock(
      const protobufs::InstructionDescriptor& instruction_to_split_before,
      uint32_t fresh_id);

  // - |message_.base_instruction_id| must be the result id of an instruction
  //   'base' in some block 'blk'.
  // - 'blk' must contain an instruction 'inst' located |message_.offset|
  //   instructions after 'base' (if |message_.offset| = 0 then 'inst' =
  //   'base').
  // - Splitting 'blk' at 'inst', so that all instructions from 'inst' onwards
  //   appear in a new block that 'blk' directly jumps to must be valid.
  // - |message_.fresh_id| must not be used by the module.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - A new block with label |message_.fresh_id| is inserted right after 'blk'
  //   in program order.
  // - All instructions of 'blk' from 'inst' onwards are moved into the new
  //   block.
  // - 'blk' is made to jump unconditionally to the new block.
  // - If 'blk' was dead, the new block is also dead.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationSplitBlock message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_
