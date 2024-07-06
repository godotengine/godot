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

#ifndef SOURCE_FUZZ_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationDuplicateRegionWithSelection : public Transformation {
 public:
  explicit TransformationDuplicateRegionWithSelection(
      protobufs::TransformationDuplicateRegionWithSelection message);

  explicit TransformationDuplicateRegionWithSelection(
      uint32_t new_entry_fresh_id, uint32_t condition_id,
      uint32_t merge_label_fresh_id, uint32_t entry_block_id,
      uint32_t exit_block_id,
      const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
      const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id,
      const std::map<uint32_t, uint32_t>& original_id_to_phi_id);

  // - |new_entry_fresh_id|, |merge_label_fresh_id| must be fresh and distinct.
  // - |condition_id| must refer to a valid instruction of boolean type.
  // - |entry_block_id| and |exit_block_id| must refer to valid blocks and they
  //   must form a single-entry, single-exit region. Its constructs and their
  //   merge blocks must be either wholly within or wholly outside of the
  //   region.
  // - |original_label_to_duplicate_label| must contain at least a key for every
  //   block in the original region.
  // - |original_id_to_duplicate_id| must contain at least a key for every
  //   result id in the original region.
  // - |original_id_to_phi_id| must contain at least a key for every result id
  //   available at the end of the original region.
  // - In each of these three maps, each value must be a distinct, fresh id.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // A transformation that inserts a conditional statement with a boolean
  // expression of arbitrary value and duplicates a given single-entry,
  // single-exit region, so that it is present in each conditional branch and
  // will be executed regardless of which branch will be taken.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Returns the set of blocks dominated by |entry_block| and post-dominated
  // by |exit_block|.
  static std::set<opt::BasicBlock*> GetRegionBlocks(
      opt::IRContext* ir_context, opt::BasicBlock* entry_block,
      opt::BasicBlock* exit_block);

  // Returns true if and only if |instr| is available at the end of the region
  // for which |exit_block| is the final block.
  static bool AvailableAfterRegion(const opt::Instruction& instr,
                                   opt::BasicBlock* exit_block,
                                   opt::IRContext* ir_context);

  // Returns true if and only if |instr| is valid as an argument to an OpPhi
  // instruction.
  static bool ValidOpPhiArgument(const opt::Instruction& instr,
                                 opt::IRContext* ir_context);

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationDuplicateRegionWithSelection message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H_
