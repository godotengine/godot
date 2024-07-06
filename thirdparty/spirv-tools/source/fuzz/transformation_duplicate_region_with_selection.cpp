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

#include "source/fuzz/transformation_duplicate_region_with_selection.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        protobufs::TransformationDuplicateRegionWithSelection message)
    : message_(std::move(message)) {}

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        uint32_t new_entry_fresh_id, uint32_t condition_id,
        uint32_t merge_label_fresh_id, uint32_t entry_block_id,
        uint32_t exit_block_id,
        const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
        const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id,
        const std::map<uint32_t, uint32_t>& original_id_to_phi_id) {
  message_.set_new_entry_fresh_id(new_entry_fresh_id);
  message_.set_condition_id(condition_id);
  message_.set_merge_label_fresh_id(merge_label_fresh_id);
  message_.set_entry_block_id(entry_block_id);
  message_.set_exit_block_id(exit_block_id);
  *message_.mutable_original_label_to_duplicate_label() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_label_to_duplicate_label);
  *message_.mutable_original_id_to_duplicate_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_duplicate_id);
  *message_.mutable_original_id_to_phi_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_phi_id);
}

bool TransformationDuplicateRegionWithSelection::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Instruction with the id |condition_id| must exist and must be of a bool
  // type.
  auto bool_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.condition_id());
  if (bool_instr == nullptr || !bool_instr->type_id()) {
    return false;
  }
  if (!ir_context->get_type_mgr()->GetType(bool_instr->type_id())->AsBool()) {
    return false;
  }

  // The |new_entry_fresh_id| must be fresh and distinct.
  std::set<uint32_t> ids_used_by_this_transformation;
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_entry_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // The |merge_label_fresh_id| must be fresh and distinct.
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.merge_label_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // The entry and exit block ids must refer to blocks.
  for (auto block_id : {message_.entry_block_id(), message_.exit_block_id()}) {
    auto block_label = ir_context->get_def_use_mgr()->GetDef(block_id);
    if (!block_label || block_label->opcode() != spv::Op::OpLabel) {
      return false;
    }
  }
  auto entry_block = ir_context->cfg()->block(message_.entry_block_id());
  auto exit_block = ir_context->cfg()->block(message_.exit_block_id());

  // The |entry_block| and the |exit_block| must be in the same function.
  if (entry_block->GetParent() != exit_block->GetParent()) {
    return false;
  }

  // The |entry_block| must dominate the |exit_block|.
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(entry_block->GetParent());
  if (!dominator_analysis->Dominates(entry_block, exit_block)) {
    return false;
  }

  // The |exit_block| must post-dominate the |entry_block|.
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(entry_block->GetParent());
  if (!postdominator_analysis->Dominates(exit_block, entry_block)) {
    return false;
  }

  auto enclosing_function = entry_block->GetParent();

  // |entry_block| cannot be the first block of the |enclosing_function|.
  if (&*enclosing_function->begin() == entry_block) {
    return false;
  }

  // To make the process of resolving OpPhi instructions easier, we require that
  // the entry block has only one predecessor.
  auto entry_block_preds = ir_context->cfg()->preds(entry_block->id());
  std::sort(entry_block_preds.begin(), entry_block_preds.end());
  entry_block_preds.erase(
      std::unique(entry_block_preds.begin(), entry_block_preds.end()),
      entry_block_preds.end());
  if (entry_block_preds.size() > 1) {
    return false;
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3785):
  //     The following code has been copied from TransformationOutlineFunction.
  //     Consider refactoring to avoid duplication.
  auto region_set = GetRegionBlocks(ir_context, entry_block, exit_block);

  // Check whether |region_set| really is a single-entry single-exit region, and
  // also check whether structured control flow constructs and their merge
  // and continue constructs are either wholly in or wholly out of the region -
  // e.g. avoid the situation where the region contains the head of a loop but
  // not the loop's continue construct.
  //
  // This is achieved by going through every block in the |enclosing_function|
  for (auto& block : *enclosing_function) {
    if (&block == exit_block) {
      // It is not OK for the exit block to head a loop construct or a
      // conditional construct.
      if (block.GetMergeInst()) {
        return false;
      }
      continue;
    }
    if (region_set.count(&block) != 0) {
      // The block is in the region and is not the region's exit block.  Let's
      // see whether all of the block's successors are in the region. If they
      // are not, the region is not single-entry single-exit.
      bool all_successors_in_region = true;
      block.WhileEachSuccessorLabel([&all_successors_in_region, ir_context,
                                     &region_set](uint32_t successor) -> bool {
        if (region_set.count(ir_context->cfg()->block(successor)) == 0) {
          all_successors_in_region = false;
          return false;
        }
        return true;
      });
      if (!all_successors_in_region) {
        return false;
      }
    }

    if (auto merge = block.GetMergeInst()) {
      // The block is a loop or selection header. The header and its
      // associated merge block must be both in the region or both be
      // outside the region.
      auto merge_block =
          ir_context->cfg()->block(merge->GetSingleWordOperand(0));
      if (region_set.count(&block) != region_set.count(merge_block)) {
        return false;
      }
    }

    if (auto loop_merge = block.GetLoopMergeInst()) {
      // The continue target of a loop must be within the region if and only if
      // the header of the loop is.
      auto continue_target =
          ir_context->cfg()->block(loop_merge->GetSingleWordOperand(1));
      // The continue target is a single-entry, single-exit region. Therefore,
      // if the continue target is the exit block, the region might not contain
      // the loop header. However, we would like to exclude this situation,
      // since it would be impossible for the modified exit block to branch to
      // the new selection merge block. In this scenario the exit block is
      // required to branch to the loop header.
      if (region_set.count(&block) != region_set.count(continue_target)) {
        return false;
      }
    }
  }

  // Get the maps from the protobuf.
  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::map<uint32_t, uint32_t> original_id_to_phi_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.original_id_to_phi_id());

  for (auto block : region_set) {
    // The label of every block in the region must be present in the map
    // |original_label_to_duplicate_label|, unless overflow ids are present.
    if (original_label_to_duplicate_label.count(block->id()) == 0) {
      if (!transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
        return false;
      }
    } else {
      auto duplicate_label = original_label_to_duplicate_label.at(block->id());
      // Each id assigned to labels in the region must be distinct and fresh.
      if (!duplicate_label ||
          !CheckIdIsFreshAndNotUsedByThisTransformation(
              duplicate_label, ir_context, &ids_used_by_this_transformation)) {
        return false;
      }
    }
    for (auto& instr : *block) {
      if (!instr.HasResultId()) {
        continue;
      }
      // Every instruction with a result id in the region must be present in the
      // map |original_id_to_duplicate_id|, unless overflow ids are present.
      if (original_id_to_duplicate_id.count(instr.result_id()) == 0) {
        if (!transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
          return false;
        }
      } else {
        auto duplicate_id = original_id_to_duplicate_id.at(instr.result_id());
        // Id assigned to this result id in the region must be distinct and
        // fresh.
        if (!duplicate_id ||
            !CheckIdIsFreshAndNotUsedByThisTransformation(
                duplicate_id, ir_context, &ids_used_by_this_transformation)) {
          return false;
        }
      }
      // If the instruction is available at the end of the region then we would
      // like to be able to add an OpPhi instruction at the merge point of the
      // duplicated region to capture the values computed by both duplicates of
      // the instruction, so that this is also available after the region.  We
      // do this not just for instructions that are already used after the
      // region, but for all instructions so that the phi is available to future
      // transformations.
      if (AvailableAfterRegion(instr, exit_block, ir_context)) {
        if (!ValidOpPhiArgument(instr, ir_context)) {
          // The instruction cannot be used as an OpPhi argument.  This is a
          // blocker if there are uses of the instruction after the region.
          // Otherwise we can simply avoid generating an OpPhi for this
          // instruction and its duplicate.
          if (!ir_context->get_def_use_mgr()->WhileEachUser(
                  &instr,
                  [ir_context,
                   &region_set](opt::Instruction* use_instr) -> bool {
                    opt::BasicBlock* use_block =
                        ir_context->get_instr_block(use_instr);
                    return use_block == nullptr ||
                           region_set.count(use_block) > 0;
                  })) {
            return false;
          }
        } else {
          // Every instruction with a result id available at the end of the
          // region must be present in the map |original_id_to_phi_id|, unless
          // overflow ids are present.
          if (original_id_to_phi_id.count(instr.result_id()) == 0) {
            if (!transformation_context.GetOverflowIdSource()
                     ->HasOverflowIds()) {
              return false;
            }
          } else {
            auto phi_id = original_id_to_phi_id.at(instr.result_id());
            // Id assigned to this result id in the region must be distinct and
            // fresh.
            if (!phi_id ||
                !CheckIdIsFreshAndNotUsedByThisTransformation(
                    phi_id, ir_context, &ids_used_by_this_transformation)) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

void TransformationDuplicateRegionWithSelection::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_entry_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.merge_label_fresh_id());

  // Create the new entry block containing the main conditional instruction. Set
  // its parent to the parent of the original entry block, since it is located
  // in the same function.
  std::unique_ptr<opt::BasicBlock> new_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLabel, 0, message_.new_entry_fresh_id(),
          opt::Instruction::OperandList()));
  auto entry_block = ir_context->cfg()->block(message_.entry_block_id());
  auto enclosing_function = entry_block->GetParent();
  auto exit_block = ir_context->cfg()->block(message_.exit_block_id());

  // Get the blocks contained in the region.
  std::set<opt::BasicBlock*> region_blocks =
      GetRegionBlocks(ir_context, entry_block, exit_block);

  // Construct the merge block.
  std::unique_ptr<opt::BasicBlock> merge_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLabel, 0, message_.merge_label_fresh_id(),
          opt::Instruction::OperandList()));

  // Get the maps from the protobuf.
  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::map<uint32_t, uint32_t> original_id_to_phi_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.original_id_to_phi_id());

  // Use overflow ids to fill in any required ids that are missing from these
  // maps.
  for (auto block : region_blocks) {
    if (original_label_to_duplicate_label.count(block->id()) == 0) {
      original_label_to_duplicate_label.insert(
          {block->id(),
           transformation_context->GetOverflowIdSource()->GetNextOverflowId()});
    }
    for (auto& instr : *block) {
      if (!instr.HasResultId()) {
        continue;
      }
      if (original_id_to_duplicate_id.count(instr.result_id()) == 0) {
        original_id_to_duplicate_id.insert(
            {instr.result_id(), transformation_context->GetOverflowIdSource()
                                    ->GetNextOverflowId()});
      }
      if (AvailableAfterRegion(instr, exit_block, ir_context) &&
          ValidOpPhiArgument(instr, ir_context)) {
        if (original_id_to_phi_id.count(instr.result_id()) == 0) {
          original_id_to_phi_id.insert(
              {instr.result_id(), transformation_context->GetOverflowIdSource()
                                      ->GetNextOverflowId()});
        }
      }
    }
  }

  // Before adding duplicate blocks, we need to update the OpPhi instructions in
  // the successors of the |exit_block|. We know that the execution of the
  // transformed region will end in |merge_block|. Hence, we need to change all
  // occurrences of the label id of the |exit_block| to the label id of the
  // |merge_block|.
  exit_block->ForEachSuccessorLabel([this, ir_context](uint32_t label_id) {
    auto block = ir_context->cfg()->block(label_id);
    for (auto& instr : *block) {
      if (instr.opcode() == spv::Op::OpPhi) {
        instr.ForEachId([this](uint32_t* id) {
          if (*id == message_.exit_block_id()) {
            *id = message_.merge_label_fresh_id();
          }
        });
      }
    }
  });

  // Get vector of predecessors id of |entry_block|. Remove any duplicate
  // values.
  auto entry_block_preds = ir_context->cfg()->preds(entry_block->id());
  std::sort(entry_block_preds.begin(), entry_block_preds.end());
  entry_block_preds.erase(
      unique(entry_block_preds.begin(), entry_block_preds.end()),
      entry_block_preds.end());
  // We know that |entry_block| has only one predecessor, since the region is
  // single-entry, single-exit and its constructs and their merge blocks must be
  // either wholly within or wholly outside of the region.
  assert(entry_block_preds.size() == 1 &&
         "The entry of the region to be duplicated can have only one "
         "predecessor.");
  uint32_t entry_block_pred_id =
      ir_context->get_instr_block(entry_block_preds[0])->id();
  // Update all the OpPhi instructions in the |entry_block|. Change every
  // occurrence of |entry_block_pred_id| to the id of |new_entry|, because we
  // will insert |new_entry| before |entry_block|.
  for (auto& instr : *entry_block) {
    if (instr.opcode() == spv::Op::OpPhi) {
      instr.ForEachId([this, entry_block_pred_id](uint32_t* id) {
        if (*id == entry_block_pred_id) {
          *id = message_.new_entry_fresh_id();
        }
      });
    }
  }

  // Duplication of blocks will invalidate iterators. Store all the blocks from
  // the enclosing function.
  std::vector<opt::BasicBlock*> blocks;
  for (auto& block : *enclosing_function) {
    blocks.push_back(&block);
  }

  opt::BasicBlock* previous_block = nullptr;
  opt::BasicBlock* duplicated_exit_block = nullptr;
  // Iterate over all blocks of the function to duplicate blocks of the original
  // region and their instructions.
  for (auto& block : blocks) {
    // The block must be contained in the region.
    if (region_blocks.count(block) == 0) {
      continue;
    }

    fuzzerutil::UpdateModuleIdBound(
        ir_context, original_label_to_duplicate_label.at(block->id()));

    std::unique_ptr<opt::BasicBlock> duplicated_block =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpLabel, 0,
            original_label_to_duplicate_label.at(block->id()),
            opt::Instruction::OperandList()));

    for (auto& instr : *block) {
      // Case where an instruction is the terminator of the exit block is
      // handled separately.
      if (block == exit_block && instr.IsBlockTerminator()) {
        switch (instr.opcode()) {
          case spv::Op::OpBranch:
          case spv::Op::OpBranchConditional:
          case spv::Op::OpReturn:
          case spv::Op::OpReturnValue:
          case spv::Op::OpUnreachable:
          case spv::Op::OpKill:
            continue;
          default:
            assert(false &&
                   "Unexpected terminator for |exit_block| of the region.");
        }
      }
      // Duplicate the instruction.
      auto cloned_instr = instr.Clone(ir_context);
      duplicated_block->AddInstruction(
          std::unique_ptr<opt::Instruction>(cloned_instr));

      if (instr.HasResultId()) {
        fuzzerutil::UpdateModuleIdBound(
            ir_context, original_id_to_duplicate_id.at(instr.result_id()));
      }

      // If an id from the original region was used in this instruction,
      // replace it with the value from |original_id_to_duplicate_id|.
      // If a label from the original region was used in this instruction,
      // replace it with the value from |original_label_to_duplicate_label|.
      cloned_instr->ForEachId(
          [original_id_to_duplicate_id,
           original_label_to_duplicate_label](uint32_t* op) {
            if (original_id_to_duplicate_id.count(*op) != 0) {
              *op = original_id_to_duplicate_id.at(*op);
            } else if (original_label_to_duplicate_label.count(*op) != 0) {
              *op = original_label_to_duplicate_label.at(*op);
            }
          });
    }

    // If the block is the first duplicated block, insert it after the exit
    // block of the original region. Otherwise, insert it after the preceding
    // one.
    auto duplicated_block_ptr = duplicated_block.get();
    if (previous_block) {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                previous_block);
    } else {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                exit_block);
    }
    previous_block = duplicated_block_ptr;
    if (block == exit_block) {
      // After execution of the loop, this variable stores a pointer to the last
      // duplicated block.
      duplicated_exit_block = duplicated_block_ptr;
    }
  }

  for (auto& block : region_blocks) {
    for (auto& instr : *block) {
      if (instr.result_id() == 0) {
        continue;
      }
      if (AvailableAfterRegion(instr, exit_block, ir_context) &&
          ValidOpPhiArgument(instr, ir_context)) {
        // Add an OpPhi instruction for every result id that is available at
        // the end of the region, as long as the result id is valid for use
        // with OpPhi.
        merge_block->AddInstruction(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpPhi, instr.type_id(),
            original_id_to_phi_id.at(instr.result_id()),
            opt::Instruction::OperandList({
                {SPV_OPERAND_TYPE_ID, {instr.result_id()}},
                {SPV_OPERAND_TYPE_ID, {exit_block->id()}},
                {SPV_OPERAND_TYPE_ID,
                 {original_id_to_duplicate_id.at(instr.result_id())}},
                {SPV_OPERAND_TYPE_ID, {duplicated_exit_block->id()}},
            })));

        fuzzerutil::UpdateModuleIdBound(
            ir_context, original_id_to_phi_id.at(instr.result_id()));

        // If the instruction has been remapped by an OpPhi, look
        // for all its uses outside of the region and outside of the
        // merge block (to not overwrite just added instructions in
        // the merge block) and replace the original instruction id
        // with the id of the corresponding OpPhi instruction.
        ir_context->get_def_use_mgr()->ForEachUse(
            &instr,
            [ir_context, &instr, region_blocks, original_id_to_phi_id,
             &merge_block](opt::Instruction* user, uint32_t operand_index) {
              auto user_block = ir_context->get_instr_block(user);
              if ((region_blocks.find(user_block) != region_blocks.end()) ||
                  user_block == merge_block.get()) {
                return;
              }
              user->SetOperand(operand_index,
                               {original_id_to_phi_id.at(instr.result_id())});
            });
      }
    }
  }

  // Construct a conditional instruction in the |new_entry_block|.
  // If the condition is true, the execution proceeds in the
  // |entry_block| of the original region. If the condition is
  // false, the execution proceeds in the first block of the
  // duplicated region.
  new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSelectionMerge, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.merge_label_fresh_id()}},
           {SPV_OPERAND_TYPE_SELECTION_CONTROL,
            {uint32_t(spv::SelectionControlMask::MaskNone)}}})));

  new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpBranchConditional, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.condition_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.entry_block_id()}},
           {SPV_OPERAND_TYPE_ID,
            {original_label_to_duplicate_label.at(
                message_.entry_block_id())}}})));

  // Move the terminator of |exit_block| to the end of
  // |merge_block|.
  auto exit_block_terminator = exit_block->terminator();
  auto cloned_instr = exit_block_terminator->Clone(ir_context);
  merge_block->AddInstruction(std::unique_ptr<opt::Instruction>(cloned_instr));
  ir_context->KillInst(exit_block_terminator);

  // Add OpBranch instruction to the merge block at the end of
  // |exit_block| and at the end of |duplicated_exit_block|, so that
  // the execution proceeds in the |merge_block|.
  opt::Instruction merge_branch_instr = opt::Instruction(
      ir_context, spv::Op::OpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.merge_label_fresh_id()}}}));
  exit_block->AddInstruction(MakeUnique<opt::Instruction>(merge_branch_instr));
  duplicated_exit_block->AddInstruction(
      std::unique_ptr<opt::Instruction>(merge_branch_instr.Clone(ir_context)));

  // Execution needs to start in the |new_entry_block|. Change all
  // the uses of |entry_block_label_instr| outside of the original
  // region to |message_.new_entry_fresh_id|.
  auto entry_block_label_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.entry_block_id());
  ir_context->get_def_use_mgr()->ForEachUse(
      entry_block_label_instr,
      [this, ir_context, region_blocks](opt::Instruction* user,
                                        uint32_t operand_index) {
        auto user_block = ir_context->get_instr_block(user);
        if ((region_blocks.count(user_block) != 0)) {
          return;
        }
        switch (user->opcode()) {
          case spv::Op::OpSwitch:
          case spv::Op::OpBranch:
          case spv::Op::OpBranchConditional:
          case spv::Op::OpLoopMerge:
          case spv::Op::OpSelectionMerge: {
            user->SetOperand(operand_index, {message_.new_entry_fresh_id()});
          } break;
          case spv::Op::OpName:
            break;
          default:
            assert(false &&
                   "The label id cannot be used by instructions "
                   "other than "
                   "OpSwitch, OpBranch, OpBranchConditional, "
                   "OpLoopMerge, "
                   "OpSelectionMerge");
        }
      });

  opt::Instruction* merge_block_terminator = merge_block->terminator();
  switch (merge_block_terminator->opcode()) {
    case spv::Op::OpReturnValue:
    case spv::Op::OpBranchConditional: {
      uint32_t operand = merge_block_terminator->GetSingleWordInOperand(0);
      if (original_id_to_phi_id.count(operand)) {
        merge_block_terminator->SetInOperand(
            0, {original_id_to_phi_id.at(operand)});
      }
      break;
    }
    default:
      break;
  }

  // Insert the merge block after the |duplicated_exit_block| (the
  // last duplicated block).
  enclosing_function->InsertBasicBlockAfter(std::move(merge_block),
                                            duplicated_exit_block);

  // Insert the |new_entry_block| before the entry block of the
  // original region.
  enclosing_function->InsertBasicBlockBefore(std::move(new_entry_block),
                                             entry_block);

  // Since we have changed the module, most of the analysis are now
  // invalid. We can invalidate analyses now after all of the blocks
  // have been registered.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3785):
//     The following method has been copied from
//     TransformationOutlineFunction. Consider refactoring to avoid
//     duplication.
std::set<opt::BasicBlock*>
TransformationDuplicateRegionWithSelection::GetRegionBlocks(
    opt::IRContext* ir_context, opt::BasicBlock* entry_block,
    opt::BasicBlock* exit_block) {
  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(enclosing_function);

  // A block belongs to a region between the entry block and the exit
  // block if and only if it is dominated by the entry block and
  // post-dominated by the exit block.
  std::set<opt::BasicBlock*> result;
  for (auto& block : *enclosing_function) {
    if (dominator_analysis->Dominates(entry_block, &block) &&
        postdominator_analysis->Dominates(exit_block, &block)) {
      result.insert(&block);
    }
  }
  return result;
}

protobufs::Transformation
TransformationDuplicateRegionWithSelection::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_duplicate_region_with_selection() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationDuplicateRegionWithSelection::GetFreshIds() const {
  std::unordered_set<uint32_t> result = {message_.new_entry_fresh_id(),
                                         message_.merge_label_fresh_id()};
  for (auto& pair : message_.original_label_to_duplicate_label()) {
    result.insert(pair.second());
  }
  for (auto& pair : message_.original_id_to_duplicate_id()) {
    result.insert(pair.second());
  }
  for (auto& pair : message_.original_id_to_phi_id()) {
    result.insert(pair.second());
  }
  return result;
}

bool TransformationDuplicateRegionWithSelection::AvailableAfterRegion(
    const opt::Instruction& instr, opt::BasicBlock* exit_block,
    opt::IRContext* ir_context) {
  opt::Instruction* final_instruction_in_region = &*exit_block->tail();
  return &instr == final_instruction_in_region ||
         fuzzerutil::IdIsAvailableBeforeInstruction(
             ir_context, final_instruction_in_region, instr.result_id());
}

bool TransformationDuplicateRegionWithSelection::ValidOpPhiArgument(
    const opt::Instruction& instr, opt::IRContext* ir_context) {
  opt::Instruction* instr_type =
      ir_context->get_def_use_mgr()->GetDef(instr.type_id());

  // It is invalid to apply OpPhi to void-typed values.
  if (instr_type->opcode() == spv::Op::OpTypeVoid) {
    return false;
  }

  // Using pointers with OpPhi requires capability VariablePointers.
  if (instr_type->opcode() == spv::Op::OpTypePointer &&
      !ir_context->get_feature_mgr()->HasCapability(
          spv::Capability::VariablePointers)) {
    return false;
  }

  // OpTypeSampledImage cannot be the result type of an OpPhi instruction.
  if (instr_type->opcode() == spv::Op::OpTypeSampledImage) {
    return false;
  }
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
