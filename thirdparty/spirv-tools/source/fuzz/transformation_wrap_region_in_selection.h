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

#ifndef SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapRegionInSelection : public Transformation {
 public:
  explicit TransformationWrapRegionInSelection(
      protobufs::TransformationWrapRegionInSelection message);

  TransformationWrapRegionInSelection(uint32_t region_entry_block_id,
                                      uint32_t region_exit_block_id,
                                      bool branch_condition);

  // - It should be possible to apply this transformation to a
  //   single-exit-single-entry region of blocks dominated by
  //   |region_entry_block_id| and postdominated by |region_exit_block_id|
  //   (see IsApplicableToBlockRange method for further details).
  //
  //   TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3828):
  //    Consider applying this transformation to non-single-entry-single-exit
  //    regions of blocks.
  // - There must exist an irrelevant boolean constant with value
  //   |branch_condition|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Transforms |region_entry_block_id| into a selection header with both
  //   branches pointing to the block's successor.
  // - |branch_condition| is used as a condition in the header's
  //   OpBranchConditional instruction.
  // - Transforms |region_exit_block_id| into a merge block of the selection's
  //   header.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if it's possible to apply this transformation to the
  // single-exit-single-entry region of blocks starting with
  // |header_block_candidate_id| and ending with |merge_block_candidate_id|.
  // Concretely:
  // - Both |header_block_candidate_id| and |merge_block_candidate_id| must be
  //   result ids of some blocks in the module.
  // - Both blocks must belong to the same function.
  // - |header_block_candidate_id| must strictly dominate
  //   |merge_block_candidate_id| and |merge_block_candidate_id| must strictly
  //   postdominate |header_block_candidate_id|.
  // - |header_block_candidate_id| can't be a header block of any construct.
  // - |header_block_candidate_id|'s terminator must be an OpBranch.
  // - |merge_block_candidate_id| can't be a merge block of any other construct.
  // - Both |header_block_candidate_id| and |merge_block_candidate_id| must be
  //   inside the same construct if any.
  static bool IsApplicableToBlockRange(opt::IRContext* ir_context,
                                       uint32_t header_block_candidate_id,
                                       uint32_t merge_block_candidate_id);

  std::unordered_set<uint32_t> GetFreshIds() const override;

 private:
  protobufs::TransformationWrapRegionInSelection message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_
