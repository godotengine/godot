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

#ifndef SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_

#include <map>
#include <set>
#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationOutlineFunction : public Transformation {
 public:
  explicit TransformationOutlineFunction(
      protobufs::TransformationOutlineFunction message);

  TransformationOutlineFunction(
      uint32_t entry_block, uint32_t exit_block,
      uint32_t new_function_struct_return_type_id,
      uint32_t new_function_type_id, uint32_t new_function_id,
      uint32_t new_function_region_entry_block, uint32_t new_caller_result_id,
      uint32_t new_callee_result_id,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id);

  // - All the fresh ids occurring in the transformation must be distinct and
  //   fresh
  // - |message_.entry_block| and |message_.exit_block| must form a single-entry
  //   single-exit control flow graph region
  // - |message_.entry_block| must not start with OpVariable
  // - |message_.entry_block| must not be a loop header
  // - |message_.exit_block| must not be a merge block or the continue target
  //   of a loop
  // - A structured control flow construct must lie either completely within the
  //   region or completely outside it
  // - |message.entry_block| must not start with OpPhi; this is to keep the
  //   transformation simple - another transformation should be used to split
  //   a desired entry block that starts with OpPhi if needed
  // - |message_.input_id_to_fresh_id| must contain an entry for every id
  //   defined outside the region but used in the region
  // - |message_.output_id_to_fresh_id| must contain an entry for every id
  //   defined in the region but used outside the region
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - A new function with id |message_.new_function_id| is added to the module.
  // - If the region generates output ids, the return type of this function is
  //   a new struct type with one field per output id, and with type id
  //   |message_.new_function_struct_return_type|, otherwise the function return
  //   types is void and |message_.new_function_struct_return_type| is not used.
  // - If the region generates input ids, the new function has one parameter per
  //   input id.  Fresh ids for these parameters are provided by
  //   |message_.input_id_to_fresh_id|.
  // - Unless the type required for the new function is already known,
  //   |message_.new_function_type_id| is used as the type id for a new function
  //   type, and the new function uses this type.
  // - The new function starts with a placeholder block with id
  //   |message_.new_function_first_block|, which jumps straight to a successor
  //   block, to avoid violating rules on what the first block in a function may
  //   look like.
  // - The outlined region is replaced with a single block, with the same id
  //   as |message_.entry_block|, and which calls the new function, passing the
  //   region's input ids as parameters.  The result is  stored in
  //   |message_.new_caller_result_id|, which has type
  //   |message_.new_function_struct_return_type| (unless there are
  //   no output ids, in which case the return type is void).  The components
  //   of this returned struct are then copied out into the region's output ids.
  //   The block ends with the merge instruction (if any) and terminator of
  //   |message_.exit_block|.
  // - The body of the new function is identical to the outlined region, except
  //   that (a) the region's entry block has id
  //   |message_.new_function_region_entry_block|, (b) input id uses are
  //   replaced with parameter accesses, (c) and definitions of output ids are
  //   replaced with definitions of corresponding fresh ids provided by
  //   |message_.output_id_to_fresh_id|, and (d) the block of the function
  //   ends by returning a composite of type
  //   |message_.new_function_struct_return_type| comprised of all the fresh
  //   output ids (unless the return type is void, in which case no value is
  //   returned.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the set of blocks dominated by |entry_block| and post-dominated
  // by |exit_block|.
  static std::set<opt::BasicBlock*> GetRegionBlocks(
      opt::IRContext* ir_context, opt::BasicBlock* entry_block,
      opt::BasicBlock* exit_block);

  // Yields ids that are used in |region_set| and that are either parameters
  // to the function containing |region_set|, or are defined by blocks of this
  // function that are outside |region_set|.
  //
  // Special cases: OpPhi instructions in |region_entry_block| and the
  // terminator of |region_exit_block| do not get outlined, therefore
  // - id uses in OpPhi instructions in |region_entry_block| are ignored
  // - id uses in the terminator instruction of |region_exit_block| are ignored
  static std::vector<uint32_t> GetRegionInputIds(
      opt::IRContext* ir_context, const std::set<opt::BasicBlock*>& region_set,
      opt::BasicBlock* region_exit_block);

  // Yields all ids that are defined in |region_set| and used outside
  // |region_set|.
  //
  // Special cases: for similar reasons as for |GetRegionInputIds|,
  // - ids defined in the region and used in the terminator of
  //   |region_exit_block| count as output ids
  static std::vector<uint32_t> GetRegionOutputIds(
      opt::IRContext* ir_context, const std::set<opt::BasicBlock*>& region_set,
      opt::BasicBlock* region_exit_block);

 private:
  // Ensures that the module's id bound is at least the maximum of any fresh id
  // associated with the transformation.
  void UpdateModuleIdBoundForFreshIds(
      opt::IRContext* ir_context,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const;

  // Uses |input_id_to_fresh_id_map| and |output_id_to_fresh_id_map| to convert,
  // in the region to be outlined, all the input ids in |region_input_ids| and
  // the output ids in |region_output_ids| to their fresh counterparts.
  // Parameters |region_blocks| provides access to the blocks that must be
  // modified, and |original_region_exit_block| allows for some special cases
  // where ids should not be remapped.
  void RemapInputAndOutputIdsInRegion(
      opt::IRContext* ir_context,
      const opt::BasicBlock& original_region_exit_block,
      const std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const;

  // Produce a Function object that has the right function type and parameter
  // declarations.  The function argument types and parameter ids are dictated
  // by |region_input_ids| and |input_id_to_fresh_id_map|.  The function return
  // type is dictated by |region_output_ids|.
  //
  // A new struct type to represent the function return type, and a new function
  // type for the function, will be added to the module (unless suitable types
  // are already present).
  //
  // Facts about the function containing the outlined region that are relevant
  // to the new function are propagated via the vact manager in
  // |transformation_context|.
  std::unique_ptr<opt::Function> PrepareFunctionPrototype(
      const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      opt::IRContext* ir_context,
      TransformationContext* transformation_context) const;

  // Creates the body of the outlined function by cloning blocks from the
  // original region, given by |region_blocks|, adapting the cloned version
  // of |original_region_exit_block| so that it returns something appropriate,
  // and patching up branches to |original_region_entry_block| to refer to its
  // clone.  Parameters |region_output_ids| and |output_id_to_fresh_id_map| are
  // used to determine what the function should return.  Parameter
  // |output_id_to_type_id| provides the type of each output id.
  //
  // The |transformation_context| argument allow facts about blocks being
  // outlined, e.g. whether they are dead blocks, to be asserted about blocks
  // that get created during outlining.
  void PopulateOutlinedFunction(
      const opt::BasicBlock& original_region_entry_block,
      const opt::BasicBlock& original_region_exit_block,
      const std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& output_id_to_type_id,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map,
      opt::IRContext* ir_context, opt::Function* outlined_function) const;

  // Shrinks the outlined region, given by |region_blocks|, down to the single
  // block |original_region_entry_block|.  This block is itself shrunk to just
  // contain:
  // - any OpPhi instructions that were originally present
  // - a call to the outlined function, with parameters provided by
  //   |region_input_ids|
  // - instructions to route components of the call's return value into
  //   |region_output_ids|
  // - The merge instruction (if any) and terminator of the original region's
  //   exit block, given by |cloned_exit_block_merge| and
  //   |cloned_exit_block_terminator|
  // Parameters |output_id_to_type_id| and |return_type_id| provide the
  // provide types for the region's output ids, and the return type of the
  // outlined function: as the module is in an inconsistent state when this
  // function is called, this information cannot be gotten from the def-use
  // manager.
  void ShrinkOriginalRegion(
      opt::IRContext* ir_context,
      const std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& output_id_to_type_id,
      uint32_t return_type_id,
      std::unique_ptr<opt::Instruction> cloned_exit_block_merge,
      std::unique_ptr<opt::Instruction> cloned_exit_block_terminator,
      opt::BasicBlock* original_region_entry_block) const;

  protobufs::TransformationOutlineFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
