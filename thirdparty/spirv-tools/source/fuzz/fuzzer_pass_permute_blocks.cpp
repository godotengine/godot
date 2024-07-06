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

#include "source/fuzz/fuzzer_pass_permute_blocks.h"

#include "source/fuzz/transformation_move_block_down.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteBlocks::FuzzerPassPermuteBlocks(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPermuteBlocks::Apply() {
  // For now we do something very simple: we randomly decide whether to move a
  // block, and for each block that we do move, we push it down as far as we
  // legally can.
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2635): it would be
  //  nice to randomly sample from the set of legal block permutations and then
  //  encode the chosen permutation via a series of move-block-down
  //  transformations.  This should be possible but will require some thought.

  for (auto& function : *GetIRContext()->module()) {
    std::vector<uint32_t> block_ids;
    // Collect all block ids for the function before messing with block
    // ordering.
    for (auto& block : function) {
      block_ids.push_back(block.id());
    }
    // Now consider each block id.  We consider block ids in reverse, because
    // e.g. in code generated from the following:
    //
    // if (...) {
    //   A
    //   B
    // } else {
    //   C
    // }
    //
    // block A cannot be moved down, but B has freedom to move and that movement
    // would provide more freedom for A to move.
    for (auto id = block_ids.rbegin(); id != block_ids.rend(); ++id) {
      // Randomly decide whether to ignore the block id.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfMovingBlockDown())) {
        continue;
      }
      // Keep pushing the block down, until pushing down fails.
      // The loop is guaranteed to terminate because a block cannot be pushed
      // down indefinitely.
      while (true) {
        TransformationMoveBlockDown transformation(*id);
        if (!MaybeApplyTransformation(transformation)) {
          break;
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
