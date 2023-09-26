// Copyright (c) 2021 Google LLC
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

#ifndef SOURCE_OPT_REPLACE_DESC_VAR_INDEX_ACCESS_H_
#define SOURCE_OPT_REPLACE_DESC_VAR_INDEX_ACCESS_H_

#include <cstdio>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/function.h"
#include "source/opt/pass.h"
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class ReplaceDescArrayAccessUsingVarIndex : public Pass {
 public:
  ReplaceDescArrayAccessUsingVarIndex() {}

  const char* name() const override {
    return "replace-desc-array-access-using-var-index";
  }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Replaces all accesses to |var| using variable indices with constant
  // elements of the array |var|. Creates switch-case statements to determine
  // the value of the variable index for all the possible cases. Returns
  // whether replacement is done or not.
  bool ReplaceVariableAccessesWithConstantElements(Instruction* var) const;

  // Replaces the OpAccessChain or OpInBoundsAccessChain instruction |use| that
  // uses the descriptor variable |var| with the OpAccessChain or
  // OpInBoundsAccessChain instruction with a constant Indexes operand.
  void ReplaceAccessChain(Instruction* var, Instruction* use) const;

  // Updates the first Indexes operand of the OpAccessChain or
  // OpInBoundsAccessChain instruction |access_chain| to let it use a constant
  // index |const_element_idx|.
  void UseConstIndexForAccessChain(Instruction* access_chain,
                                   uint32_t const_element_idx) const;

  // Replaces users of the OpAccessChain or OpInBoundsAccessChain instruction
  // |access_chain| that accesses an array descriptor variable using variable
  // indices with constant elements. |number_of_elements| is the number
  // of array elements.
  void ReplaceUsersOfAccessChain(Instruction* access_chain,
                                 uint32_t number_of_elements) const;

  // Puts all the recursive users of |access_chain| with concrete result types
  // or the ones without result it in |final_users|.
  void CollectRecursiveUsersWithConcreteType(
      Instruction* access_chain, std::vector<Instruction*>* final_users) const;

  // Recursively collects the operands of |user| (and operands of the operands)
  // whose result types are images/samplers (or pointers/arrays/ structs of
  // them) and access chains instructions and returns them. The returned
  // collection includes |user|.
  std::deque<Instruction*> CollectRequiredImageAndAccessInsts(
      Instruction* user) const;

  // Returns whether result type of |inst| is an image/sampler/pointer of image
  // or sampler or not.
  bool HasImageOrImagePtrType(const Instruction* inst) const;

  // Returns whether |type_inst| is an image/sampler or pointer/array/struct of
  // image or sampler or not.
  bool IsImageOrImagePtrType(const Instruction* type_inst) const;

  // Returns whether the type with |type_id| is a concrete type or not.
  bool IsConcreteType(uint32_t type_id) const;

  // Replaces the non-uniform access to a descriptor variable
  // |access_chain_final_user| with OpSwitch instruction and case blocks. Each
  // case block will contain a clone of |access_chain| and clones of
  // |non_uniform_accesses_to_clone| that are recursively used by
  // |access_chain_final_user|. The clone of |access_chain| (or
  // OpInBoundsAccessChain) will have a constant index for its first index. The
  // OpSwitch instruction will have the cases for the variable index of
  // |access_chain| from 0 to |number_of_elements| - 1.
  void ReplaceNonUniformAccessWithSwitchCase(
      Instruction* access_chain_final_user, Instruction* access_chain,
      uint32_t number_of_elements,
      const std::deque<Instruction*>& non_uniform_accesses_to_clone) const;

  // Creates and returns a new basic block that contains all instructions of
  // |block| after |separation_begin_inst|. The new basic block is added to the
  // function in this method.
  BasicBlock* SeparateInstructionsIntoNewBlock(
      BasicBlock* block, Instruction* separation_begin_inst) const;

  // Creates and returns a new block.
  BasicBlock* CreateNewBlock() const;

  // Returns the first operand id of the OpAccessChain or OpInBoundsAccessChain
  // instruction |access_chain|.
  uint32_t GetFirstIndexOfAccessChain(Instruction* access_chain) const;

  // Adds a clone of the OpAccessChain or OpInBoundsAccessChain instruction
  // |access_chain| to |case_block|. The clone of |access_chain| will use
  // |const_element_idx| for its first index. |old_ids_to_new_ids| keeps the
  // mapping from the result id of |access_chain| to the result of its clone.
  void AddConstElementAccessToCaseBlock(
      BasicBlock* case_block, Instruction* access_chain,
      uint32_t const_element_idx,
      std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const;

  // Clones all instructions in |insts_to_be_cloned| and put them to |block|.
  // |old_ids_to_new_ids| keeps the mapping from the result id of each
  // instruction of |insts_to_be_cloned| to the result of their clones.
  void CloneInstsToBlock(
      BasicBlock* block, Instruction* inst_to_skip_cloning,
      const std::deque<Instruction*>& insts_to_be_cloned,
      std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const;

  // Adds OpBranch to |branch_destination| at the end of |parent_block|.
  void AddBranchToBlock(BasicBlock* parent_block,
                        uint32_t branch_destination) const;

  // Replaces in-operands of all instructions in the basic block |block| using
  // |old_ids_to_new_ids|. It conducts the replacement only if the in-operand
  // id is a key of |old_ids_to_new_ids|.
  void UseNewIdsInBlock(
      BasicBlock* block,
      const std::unordered_map<uint32_t, uint32_t>& old_ids_to_new_ids) const;

  // Creates a case block for |element_index| case. It adds clones of
  // |insts_to_be_cloned| and a clone of |access_chain| with |element_index| as
  // its first index. The termination instruction of the created case block will
  // be a branch to |branch_target_id|. Puts old ids to new ids map for the
  // cloned instructions in |old_ids_to_new_ids|.
  BasicBlock* CreateCaseBlock(
      Instruction* access_chain, uint32_t element_index,
      const std::deque<Instruction*>& insts_to_be_cloned,
      uint32_t branch_target_id,
      std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const;

  // Creates a default block for switch-case statement that has only a single
  // instruction OpBranch whose target is a basic block with |merge_block_id|.
  // If |null_const_for_phi_is_needed| is true, gets or creates a default null
  // constant value for a phi instruction whose operands are |phi_operands| and
  // puts it in |phi_operands|.
  BasicBlock* CreateDefaultBlock(bool null_const_for_phi_is_needed,
                                 std::vector<uint32_t>* phi_operands,
                                 uint32_t merge_block_id) const;

  // Creates and adds an OpSwitch used for the selection of OpAccessChain whose
  // first Indexes operand is |access_chain_index_var_id|. The OpSwitch will be
  // added at the end of |parent_block|. It will jump to |default_id| for the
  // default case and jumps to one of case blocks whose ids are |case_block_ids|
  // if |access_chain_index_var_id| matches the case number. |merge_id| is the
  // merge block id.
  void AddSwitchForAccessChain(
      BasicBlock* parent_block, uint32_t access_chain_index_var_id,
      uint32_t default_id, uint32_t merge_id,
      const std::vector<uint32_t>& case_block_ids) const;

  // Creates a phi instruction with |phi_operands| as values and
  // |case_block_ids| and |default_block_id| as incoming blocks. The size of
  // |phi_operands| must be exactly 1 larger than the size of |case_block_ids|.
  // The last element of |phi_operands| will be used for |default_block_id|. It
  // adds the phi instruction to the beginning of |parent_block|.
  uint32_t CreatePhiInstruction(BasicBlock* parent_block,
                                const std::vector<uint32_t>& phi_operands,
                                const std::vector<uint32_t>& case_block_ids,
                                uint32_t default_block_id) const;

  // Replaces the incoming block operand of OpPhi instructions with
  // |new_incoming_block_id| if the incoming block operand is
  // |old_incoming_block_id|.
  void ReplacePhiIncomingBlock(uint32_t old_incoming_block_id,
                               uint32_t new_incoming_block_id) const;

  // Create an OpConstantNull instruction whose result type id is |type_id|.
  Instruction* GetConstNull(uint32_t type_id) const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REPLACE_DESC_VAR_INDEX_ACCESS_H_
