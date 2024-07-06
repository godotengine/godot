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

#ifndef SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H_
#define SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationFlattenConditionalBranch : public Transformation {
 public:
  explicit TransformationFlattenConditionalBranch(
      protobufs::TransformationFlattenConditionalBranch message);

  TransformationFlattenConditionalBranch(
      uint32_t header_block_id, bool true_branch_first,
      uint32_t fresh_id_for_bvec2_selector,
      uint32_t fresh_id_for_bvec3_selector,
      uint32_t fresh_id_for_bvec4_selector,
      const std::vector<protobufs::SideEffectWrapperInfo>&
          side_effect_wrappers_info);

  // - |message_.header_block_id| must be the label id of a reachable selection
  //   header, which ends with an OpBranchConditional instruction.
  // - The condition of the OpBranchConditional instruction must not be an
  //   irrelevant id.
  // - The header block and the merge block must describe a single-entry,
  //   single-exit region.
  // - The region must not contain barrier or OpSampledImage instructions.
  // - The region must not contain selection or loop constructs.
  // - The region must not define ids that are the base objects for existing
  //   synonym facts.
  // - For each instruction that requires additional fresh ids, then:
  //   - if the instruction is mapped to the required ids for enclosing it by
  //     |message_.side_effect_wrapper_info|, these must be valid (the
  //     fresh ids must be non-zero, fresh and distinct);
  //   - if there is no such mapping, the transformation context must have
  //     overflow ids available.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Flattens the selection construct with header |message_.header_block_id|,
  // changing any OpPhi in the block where the flow converges to OpSelect and
  // enclosing any instruction with side effects in conditionals so that
  // they are only executed when they should.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if the conditional headed by |header| can be flattened,
  // according to the conditions of the IsApplicable method, assuming that
  // enough fresh ids would be provided. In this case, it fills the
  // |instructions_that_need_ids| set with all the instructions that would
  // require fresh ids.
  // Returns false otherwise.
  // Assumes that |header| is the header of a conditional, so its last two
  // instructions are OpSelectionMerge and OpBranchConditional.
  static bool GetProblematicInstructionsIfConditionalCanBeFlattened(
      opt::IRContext* ir_context, opt::BasicBlock* header,
      const TransformationContext& transformation_context,
      std::set<opt::Instruction*>* instructions_that_need_ids);

  // Returns true iff the given instruction needs a placeholder to be enclosed
  // inside a conditional. So, it returns:
  // - true if the instruction has a non-void result id,
  // - false if the instruction does not have a result id or has a void result
  //   id.
  // Assumes that the instruction has side effects, requiring it to be enclosed
  // inside a conditional, and that it can be enclosed inside a conditional
  // keeping the module valid. Assumes that, if the instruction has a void
  // result type, its result id is not used in the module.
  static bool InstructionNeedsPlaceholder(opt::IRContext* ir_context,
                                          const opt::Instruction& instruction);

  // Returns true if and only if the SPIR-V version is such that the arguments
  // to OpSelect are restricted to only scalars, pointers (if the appropriate
  // capability is enabled) and component-wise vectors.
  static bool OpSelectArgumentsAreRestricted(opt::IRContext* ir_context);

  // Find the first block where flow converges (it is not necessarily the merge
  // block) by walking the true branch until reaching a block that post-
  // dominates the header.
  // This is necessary because a potential common set of blocks at the end of
  // the construct should not be duplicated.
  static uint32_t FindConvergenceBlock(opt::IRContext* ir_context,
                                       const opt::BasicBlock& header_block);

 private:
  // Returns an unordered_map mapping instructions to the info required to
  // enclose them inside a conditional. It maps the instructions to the
  // corresponding entry in |message_.side_effect_wrapper_info|.
  std::unordered_map<opt::Instruction*, protobufs::SideEffectWrapperInfo>
  GetInstructionsToWrapperInfo(opt::IRContext* ir_context) const;

  // Splits the given block, adding a new selection construct so that the given
  // instruction is only executed if the boolean value of |condition_id| matches
  // the value of |exec_if_cond_true|.
  // Assumes that all parameters are consistent.
  // 2 fresh ids are required if the instruction does not have a result id (the
  // first two ids in |wrapper_info| must be valid fresh ids), 5 otherwise.
  // Returns the merge block created.
  //
  // |dead_blocks| and |irrelevant_ids| are used to record the ids of blocks
  // and instructions for which dead block and irrelevant id facts should
  // ultimately be created.
  static opt::BasicBlock* EncloseInstructionInConditional(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context,
      opt::BasicBlock* block, opt::Instruction* instruction,
      const protobufs::SideEffectWrapperInfo& wrapper_info,
      uint32_t condition_id, bool exec_if_cond_true,
      std::vector<uint32_t>* dead_blocks,
      std::vector<uint32_t>* irrelevant_ids);

  // Turns every OpPhi instruction of |convergence_block| -- the convergence
  // block for |header_block| (both in |ir_context|) into an OpSelect
  // instruction.
  void RewriteOpPhiInstructionsAtConvergenceBlock(
      const opt::BasicBlock& header_block, uint32_t convergence_block_id,
      opt::IRContext* ir_context) const;

  // Adds an OpCompositeExtract instruction to the start of |block| in
  // |ir_context|, with result id given by |fresh_id|.  The instruction will
  // make a |dimension|-dimensional boolean vector with
  // |branch_condition_operand| at every component.
  static void AddBooleanVectorConstructorToBlock(
      uint32_t fresh_id, uint32_t dimension,
      const opt::Operand& branch_condition_operand, opt::IRContext* ir_context,
      opt::BasicBlock* block);

  // Returns true if the given instruction either has no side effects or it can
  // be handled by being enclosed in a conditional.
  static bool InstructionCanBeHandled(opt::IRContext* ir_context,
                                      const opt::Instruction& instruction);

  protobufs::TransformationFlattenConditionalBranch message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H_
