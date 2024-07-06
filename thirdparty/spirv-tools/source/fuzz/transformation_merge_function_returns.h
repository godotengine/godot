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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_
#define SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {
class TransformationMergeFunctionReturns : public Transformation {
 public:
  explicit TransformationMergeFunctionReturns(
      protobufs::TransformationMergeFunctionReturns message);

  TransformationMergeFunctionReturns(
      uint32_t function_id, uint32_t outer_header_id,
      uint32_t unreachable_continue_id, uint32_t outer_return_id,
      uint32_t return_val_id, uint32_t any_returnable_val_id,
      const std::vector<protobufs::ReturnMergingInfo>& returns_merging_info);

  // - |message_.function_id| is the id of a function.
  // - The entry block of |message_.function_id| branches unconditionally to
  //   another block.
  // - |message_.any_returnable_val_id| is an id whose type is the same as the
  //   return type of the function and which is available at the end of the
  //   entry block. If this id is not found in the module, the transformation
  //   will try to find a suitable one.
  //   If the function is void, or no loops in the function contain return
  //   statements, this id will be ignored.
  // - Merge blocks of reachable loops that contain return statements only
  //   consist of OpLabel, OpPhi or OpBranch instructions.
  // - The module contains OpConstantTrue and OpConstantFalse instructions.
  // - For all merge blocks of reachable loops that contain return statements,
  //   either:
  //   - a mapping is provided in |message_.return_merging_info|, all of the
  //     corresponding fresh ids are valid and, for each OpPhi instruction in
  //     the block, there is a mapping to an available id of the same type in
  //     |opphi_to_suitable_id| or a suitable id, available at the end of the
  //     entry block, can be found in the module.
  //   - there is no mapping, but overflow ids are available and, for every
  //     OpPhi instruction in the merge blocks that need to be modified, a
  //     suitable id, available at the end of the entry block, can be found.
  // - The addition of new predecessors to the relevant merge blocks does not
  //   cause any id use to be invalid (i.e. every id must dominate all its uses
  //   even after the transformation has added new branches).
  // - All of the fresh ids that are provided and needed by the transformation
  //   are valid.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Changes the function so that there is only one reachable return
  // instruction. The function is enclosed by an outer loop, whose merge block
  // is the new return block. All existing return statements are replaced by
  // branch instructions to the merge block of the loop enclosing them, and
  // OpPhi instructions are used to keep track of the return value and of
  // whether the function is returning.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Returns a map from merge block ids to the corresponding info in
  // |message_.return_merging_info|.
  std::map<uint32_t, protobufs::ReturnMergingInfo>
  GetMappingOfMergeBlocksToInfo() const;

  // Returns a map from type ids to an id with that type and which is available
  // at the end of the entry block of |message_.function_id|.
  // Assumes that the function exists.
  std::map<uint32_t, uint32_t> GetTypesToIdAvailableAfterEntryBlock(
      opt::IRContext* ir_context) const;

  // Returns true if adding new predecessors to the given loop merge blocks
  // does not render any instructions invalid (each id definition must still
  // dominate all of its uses). The loop merge blocks and corresponding new
  // predecessors to consider are given in |merge_blocks_to_new_predecessors|.
  // All of the new predecessors are assumed to be inside the loop associated
  // with the corresponding loop merge block.
  static bool CheckDefinitionsStillDominateUsesAfterAddingNewPredecessors(
      opt::IRContext* ir_context, const opt::Function* function,
      const std::map<uint32_t, std::set<uint32_t>>&
          merge_blocks_to_new_predecessors);

  // Returns true if the required ids for |merge_block| are provided in the
  // |merge_blocks_to_info| map, or if ids of the suitable type can be found.
  static bool CheckThatTheCorrectIdsAreGivenForMergeBlock(
      uint32_t merge_block,
      const std::map<uint32_t, protobufs::ReturnMergingInfo>&
          merge_blocks_to_info,
      const std::map<uint32_t, uint32_t>& types_to_available_id,
      bool function_is_void, opt::IRContext* ir_context,
      const TransformationContext& transformation_context,
      std::set<uint32_t>* used_fresh_ids);

  protobufs::TransformationMergeFunctionReturns message_;
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_
