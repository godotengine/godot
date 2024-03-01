// Copyright (c) 2017 Google Inc.
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

#include "source/opt/merge_return_pass.h"

#include <list>
#include <memory>
#include <utility>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/util/bit_vector.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status MergeReturnPass::Process() {
  bool is_shader =
      context()->get_feature_mgr()->HasCapability(spv::Capability::Shader);

  bool failed = false;
  ProcessFunction pfn = [&failed, is_shader, this](Function* function) {
    std::vector<BasicBlock*> return_blocks = CollectReturnBlocks(function);
    if (return_blocks.size() <= 1) {
      if (!is_shader || return_blocks.size() == 0) {
        return false;
      }
      bool isInConstruct =
          context()->GetStructuredCFGAnalysis()->ContainingConstruct(
              return_blocks[0]->id()) != 0;
      bool endsWithReturn = return_blocks[0] == function->tail();
      if (!isInConstruct && endsWithReturn) {
        return false;
      }
    }

    function_ = function;
    return_flag_ = nullptr;
    return_value_ = nullptr;
    final_return_block_ = nullptr;

    if (is_shader) {
      if (!ProcessStructured(function, return_blocks)) {
        failed = true;
      }
    } else {
      MergeReturnBlocks(function, return_blocks);
    }
    return true;
  };

  bool modified = context()->ProcessReachableCallTree(pfn);

  if (failed) {
    return Status::Failure;
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

void MergeReturnPass::GenerateState(BasicBlock* block) {
  if (Instruction* mergeInst = block->GetMergeInst()) {
    if (mergeInst->opcode() == spv::Op::OpLoopMerge) {
      // If new loop, break to this loop merge block
      state_.emplace_back(mergeInst, mergeInst);
    } else {
      auto branchInst = mergeInst->NextNode();
      if (branchInst->opcode() == spv::Op::OpSwitch) {
        // If switch inside of loop, break to innermost loop merge block.
        // Otherwise need to break to this switch merge block.
        auto lastMergeInst = state_.back().BreakMergeInst();
        if (lastMergeInst && lastMergeInst->opcode() == spv::Op::OpLoopMerge)
          state_.emplace_back(lastMergeInst, mergeInst);
        else
          state_.emplace_back(mergeInst, mergeInst);
      } else {
        // If branch conditional inside loop, always break to innermost
        // loop merge block. If branch conditional inside switch, break to
        // innermost switch merge block.
        auto lastMergeInst = state_.back().BreakMergeInst();
        state_.emplace_back(lastMergeInst, mergeInst);
      }
    }
  }
}

bool MergeReturnPass::ProcessStructured(
    Function* function, const std::vector<BasicBlock*>& return_blocks) {
  if (HasNontrivialUnreachableBlocks(function)) {
    if (consumer()) {
      std::string message =
          "Module contains unreachable blocks during merge return.  Run dead "
          "branch elimination before merge return.";
      consumer()(SPV_MSG_ERROR, 0, {0, 0, 0}, message.c_str());
    }
    return false;
  }

  RecordImmediateDominators(function);
  if (!AddSingleCaseSwitchAroundFunction()) {
    return false;
  }

  std::list<BasicBlock*> order;
  cfg()->ComputeStructuredOrder(function, &*function->begin(), &order);

  state_.clear();
  state_.emplace_back(nullptr, nullptr);
  for (auto block : order) {
    if (cfg()->IsPseudoEntryBlock(block) || cfg()->IsPseudoExitBlock(block) ||
        block == final_return_block_) {
      continue;
    }

    auto blockId = block->GetLabelInst()->result_id();
    if (blockId == CurrentState().CurrentMergeId()) {
      // Pop the current state as we've hit the merge
      state_.pop_back();
    }

    ProcessStructuredBlock(block);

    // Generate state for next block if warranted
    GenerateState(block);
  }

  state_.clear();
  state_.emplace_back(nullptr, nullptr);
  std::unordered_set<BasicBlock*> predicated;
  for (auto block : order) {
    if (cfg()->IsPseudoEntryBlock(block) || cfg()->IsPseudoExitBlock(block)) {
      continue;
    }

    auto blockId = block->id();
    if (blockId == CurrentState().CurrentMergeId()) {
      // Pop the current state as we've hit the merge
      state_.pop_back();
    }

    // Predicate successors of the original return blocks as necessary.
    if (std::find(return_blocks.begin(), return_blocks.end(), block) !=
        return_blocks.end()) {
      if (!PredicateBlocks(block, &predicated, &order)) {
        return false;
      }
    }

    // Generate state for next block if warranted
    GenerateState(block);
  }

  // We have not kept the dominator tree up-to-date.
  // Invalidate it at this point to make sure it will be rebuilt.
  context()->RemoveDominatorAnalysis(function);
  AddNewPhiNodes();
  return true;
}

void MergeReturnPass::CreateReturnBlock() {
  // Create a label for the new return block
  std::unique_ptr<Instruction> return_label(
      new Instruction(context(), spv::Op::OpLabel, 0u, TakeNextId(), {}));

  // Create the new basic block
  std::unique_ptr<BasicBlock> return_block(
      new BasicBlock(std::move(return_label)));
  function_->AddBasicBlock(std::move(return_block));
  final_return_block_ = &*(--function_->end());
  context()->AnalyzeDefUse(final_return_block_->GetLabelInst());
  context()->set_instr_block(final_return_block_->GetLabelInst(),
                             final_return_block_);
  assert(final_return_block_->GetParent() == function_ &&
         "The function should have been set when the block was created.");
}

void MergeReturnPass::CreateReturn(BasicBlock* block) {
  AddReturnValue();

  if (return_value_) {
    // Load and return the final return value
    uint32_t loadId = TakeNextId();
    block->AddInstruction(MakeUnique<Instruction>(
        context(), spv::Op::OpLoad, function_->type_id(), loadId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_ID, {return_value_->result_id()}}}));
    Instruction* var_inst = block->terminator();
    context()->AnalyzeDefUse(var_inst);
    context()->set_instr_block(var_inst, block);
    context()->get_decoration_mgr()->CloneDecorations(
        return_value_->result_id(), loadId,
        {spv::Decoration::RelaxedPrecision});

    block->AddInstruction(MakeUnique<Instruction>(
        context(), spv::Op::OpReturnValue, 0, 0,
        std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {loadId}}}));
    context()->AnalyzeDefUse(block->terminator());
    context()->set_instr_block(block->terminator(), block);
  } else {
    block->AddInstruction(
        MakeUnique<Instruction>(context(), spv::Op::OpReturn));
    context()->AnalyzeDefUse(block->terminator());
    context()->set_instr_block(block->terminator(), block);
  }
}

void MergeReturnPass::ProcessStructuredBlock(BasicBlock* block) {
  spv::Op tail_opcode = block->tail()->opcode();
  if (tail_opcode == spv::Op::OpReturn ||
      tail_opcode == spv::Op::OpReturnValue) {
    if (!return_flag_) {
      AddReturnFlag();
    }
  }

  if (tail_opcode == spv::Op::OpReturn ||
      tail_opcode == spv::Op::OpReturnValue ||
      tail_opcode == spv::Op::OpUnreachable) {
    assert(CurrentState().InBreakable() &&
           "Should be in the placeholder construct.");
    BranchToBlock(block, CurrentState().BreakMergeId());
    return_blocks_.insert(block->id());
  }
}

void MergeReturnPass::BranchToBlock(BasicBlock* block, uint32_t target) {
  if (block->tail()->opcode() == spv::Op::OpReturn ||
      block->tail()->opcode() == spv::Op::OpReturnValue) {
    RecordReturned(block);
    RecordReturnValue(block);
  }

  BasicBlock* target_block = context()->get_instr_block(target);
  if (target_block->GetLoopMergeInst()) {
    cfg()->SplitLoopHeader(target_block);
  }
  UpdatePhiNodes(block, target_block);

  Instruction* return_inst = block->terminator();
  return_inst->SetOpcode(spv::Op::OpBranch);
  return_inst->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {target}}});
  context()->get_def_use_mgr()->AnalyzeInstDefUse(return_inst);
  new_edges_[target_block].insert(block->id());
  cfg()->AddEdge(block->id(), target);
}

void MergeReturnPass::UpdatePhiNodes(BasicBlock* new_source,
                                     BasicBlock* target) {
  target->ForEachPhiInst([this, new_source](Instruction* inst) {
    uint32_t undefId = Type2Undef(inst->type_id());
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {undefId}});
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {new_source->id()}});
    context()->UpdateDefUse(inst);
  });
}

void MergeReturnPass::CreatePhiNodesForInst(BasicBlock* merge_block,
                                            Instruction& inst) {
  DominatorAnalysis* dom_tree =
      context()->GetDominatorAnalysis(merge_block->GetParent());

  if (inst.result_id() != 0) {
    BasicBlock* inst_bb = context()->get_instr_block(&inst);
    std::vector<Instruction*> users_to_update;
    context()->get_def_use_mgr()->ForEachUser(
        &inst,
        [&users_to_update, &dom_tree, &inst, inst_bb, this](Instruction* user) {
          BasicBlock* user_bb = nullptr;
          if (user->opcode() != spv::Op::OpPhi) {
            user_bb = context()->get_instr_block(user);
          } else {
            // For OpPhi, the use should be considered to be in the predecessor.
            for (uint32_t i = 0; i < user->NumInOperands(); i += 2) {
              if (user->GetSingleWordInOperand(i) == inst.result_id()) {
                uint32_t user_bb_id = user->GetSingleWordInOperand(i + 1);
                user_bb = context()->get_instr_block(user_bb_id);
                break;
              }
            }
          }

          // If |user_bb| is nullptr, then |user| is not in the function.  It is
          // something like an OpName or decoration, which should not be
          // replaced with the result of the OpPhi.
          if (user_bb && !dom_tree->Dominates(inst_bb, user_bb)) {
            users_to_update.push_back(user);
          }
        });

    if (users_to_update.empty()) {
      return;
    }

    // There is at least one values that needs to be replaced.
    // First create the OpPhi instruction.
    uint32_t undef_id = Type2Undef(inst.type_id());
    std::vector<uint32_t> phi_operands;
    const std::set<uint32_t>& new_edges = new_edges_[merge_block];

    // Add the OpPhi operands. If the predecessor is a return block use undef,
    // otherwise use |inst|'s id.
    std::vector<uint32_t> preds = cfg()->preds(merge_block->id());
    for (uint32_t pred_id : preds) {
      if (new_edges.count(pred_id)) {
        phi_operands.push_back(undef_id);
      } else {
        phi_operands.push_back(inst.result_id());
      }
      phi_operands.push_back(pred_id);
    }

    Instruction* new_phi = nullptr;
    // If the instruction is a pointer and variable pointers are not an option,
    // then we have to regenerate the instruction instead of creating an OpPhi
    // instruction.  If not, the Spir-V will be invalid.
    Instruction* inst_type = get_def_use_mgr()->GetDef(inst.type_id());
    bool regenerateInstruction = false;
    if (inst_type->opcode() == spv::Op::OpTypePointer) {
      if (!context()->get_feature_mgr()->HasCapability(
              spv::Capability::VariablePointers)) {
        regenerateInstruction = true;
      }

      auto storage_class =
          spv::StorageClass(inst_type->GetSingleWordInOperand(0));
      if (storage_class != spv::StorageClass::Workgroup &&
          storage_class != spv::StorageClass::StorageBuffer) {
        regenerateInstruction = true;
      }
    }

    if (regenerateInstruction) {
      std::unique_ptr<Instruction> regen_inst(inst.Clone(context()));
      uint32_t new_id = TakeNextId();
      regen_inst->SetResultId(new_id);
      Instruction* insert_pos = &*merge_block->begin();
      while (insert_pos->opcode() == spv::Op::OpPhi) {
        insert_pos = insert_pos->NextNode();
      }
      new_phi = insert_pos->InsertBefore(std::move(regen_inst));
      get_def_use_mgr()->AnalyzeInstDefUse(new_phi);
      context()->set_instr_block(new_phi, merge_block);

      new_phi->ForEachInId([dom_tree, merge_block, this](uint32_t* use_id) {
        Instruction* use = get_def_use_mgr()->GetDef(*use_id);
        BasicBlock* use_bb = context()->get_instr_block(use);
        if (use_bb != nullptr && !dom_tree->Dominates(use_bb, merge_block)) {
          CreatePhiNodesForInst(merge_block, *use);
        }
      });
    } else {
      InstructionBuilder builder(
          context(), &*merge_block->begin(),
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
      new_phi = builder.AddPhi(inst.type_id(), phi_operands);
    }
    uint32_t result_of_phi = new_phi->result_id();

    // Update all of the users to use the result of the new OpPhi.
    for (Instruction* user : users_to_update) {
      user->ForEachInId([&inst, result_of_phi](uint32_t* id) {
        if (*id == inst.result_id()) {
          *id = result_of_phi;
        }
      });
      context()->AnalyzeUses(user);
    }
  }
}

bool MergeReturnPass::PredicateBlocks(
    BasicBlock* return_block, std::unordered_set<BasicBlock*>* predicated,
    std::list<BasicBlock*>* order) {
  // The CFG is being modified as the function proceeds so avoid caching
  // successors.

  if (predicated->count(return_block)) {
    return true;
  }

  BasicBlock* block = nullptr;
  const BasicBlock* const_block = const_cast<const BasicBlock*>(return_block);
  const_block->ForEachSuccessorLabel([this, &block](const uint32_t idx) {
    BasicBlock* succ_block = context()->get_instr_block(idx);
    assert(block == nullptr);
    block = succ_block;
  });
  assert(block &&
         "Return blocks should have returns already replaced by a single "
         "unconditional branch.");

  auto state = state_.rbegin();
  std::unordered_set<BasicBlock*> seen;
  if (block->id() == state->CurrentMergeId()) {
    state++;
  } else if (block->id() == state->BreakMergeId()) {
    while (state->BreakMergeId() == block->id()) {
      state++;
    }
  }

  while (block != nullptr && block != final_return_block_) {
    if (!predicated->insert(block).second) break;
    // Skip structured subgraphs.
    assert(state->InBreakable() &&
           "Should be in the placeholder construct at the very least.");
    Instruction* break_merge_inst = state->BreakMergeInst();
    uint32_t merge_block_id = break_merge_inst->GetSingleWordInOperand(0);
    while (state->BreakMergeId() == merge_block_id) {
      state++;
    }
    if (!BreakFromConstruct(block, predicated, order, break_merge_inst)) {
      return false;
    }
    block = context()->get_instr_block(merge_block_id);
  }
  return true;
}

bool MergeReturnPass::BreakFromConstruct(
    BasicBlock* block, std::unordered_set<BasicBlock*>* predicated,
    std::list<BasicBlock*>* order, Instruction* break_merge_inst) {
  // Make sure the CFG is build here.  If we don't then it becomes very hard
  // to know which new blocks need to be updated.
  context()->InvalidateAnalyses(IRContext::kAnalysisCFG);
  context()->BuildInvalidAnalyses(IRContext::kAnalysisCFG);

  // When predicating, be aware of whether this block is a header block, a
  // merge block or both.
  //
  // If this block is a merge block, ensure the appropriate header stays
  // up-to-date with any changes (i.e. points to the pre-header).
  //
  // If this block is a header block, predicate the entire structured
  // subgraph. This can act recursively.

  // If |block| is a loop header, then the back edge must jump to the original
  // code, not the new header.
  if (block->GetLoopMergeInst()) {
    if (cfg()->SplitLoopHeader(block) == nullptr) {
      return false;
    }
  }

  uint32_t merge_block_id = break_merge_inst->GetSingleWordInOperand(0);
  BasicBlock* merge_block = context()->get_instr_block(merge_block_id);
  if (merge_block->GetLoopMergeInst()) {
    cfg()->SplitLoopHeader(merge_block);
  }

  // Leave the phi instructions behind.
  auto iter = block->begin();
  while (iter->opcode() == spv::Op::OpPhi) {
    ++iter;
  }

  // Forget about the edges leaving block.  They will be removed.
  cfg()->RemoveSuccessorEdges(block);

  auto old_body_id = TakeNextId();
  BasicBlock* old_body = block->SplitBasicBlock(context(), old_body_id, iter);
  predicated->insert(old_body);

  // If a return block is being split, mark the new body block also as a return
  // block.
  if (return_blocks_.count(block->id())) {
    return_blocks_.insert(old_body_id);
  }

  // If |block| was a continue target for a loop |old_body| is now the correct
  // continue target.
  if (break_merge_inst->opcode() == spv::Op::OpLoopMerge &&
      break_merge_inst->GetSingleWordInOperand(1) == block->id()) {
    break_merge_inst->SetInOperand(1, {old_body->id()});
    context()->UpdateDefUse(break_merge_inst);
  }

  // Update |order| so old_block will be traversed.
  InsertAfterElement(block, old_body, order);

  // Within the new header we need the following:
  // 1. Load of the return status flag
  // 2. Branch to |merge_block| (true) or old body (false)
  // 3. Update OpPhi instructions in |merge_block|.
  // 4. Update the CFG.
  //
  // Since we are branching to the merge block of the current construct, there
  // is no need for an OpSelectionMerge.

  InstructionBuilder builder(
      context(), block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // 1. Load of the return status flag
  analysis::Bool bool_type;
  uint32_t bool_id = context()->get_type_mgr()->GetId(&bool_type);
  assert(bool_id != 0);
  uint32_t load_id =
      builder.AddLoad(bool_id, return_flag_->result_id())->result_id();

  // 2. Branch to |merge_block| (true) or |old_body| (false)
  builder.AddConditionalBranch(load_id, merge_block->id(), old_body->id(),
                               old_body->id());

  if (!new_edges_[merge_block].insert(block->id()).second) {
    // It is possible that we already inserted a new edge to the merge block.
    // If so, that edge now goes from |old_body| to |merge_block|.
    new_edges_[merge_block].insert(old_body->id());
  }

  // 3. Update OpPhi instructions in |merge_block|.
  UpdatePhiNodes(block, merge_block);

  // 4. Update the CFG.  We do this after updating the OpPhi instructions
  // because |UpdatePhiNodes| assumes the edge from |block| has not been added
  // to the CFG yet.
  cfg()->AddEdges(block);
  cfg()->RegisterBlock(old_body);

  assert(old_body->begin() != old_body->end());
  assert(block->begin() != block->end());
  return true;
}

void MergeReturnPass::RecordReturned(BasicBlock* block) {
  if (block->tail()->opcode() != spv::Op::OpReturn &&
      block->tail()->opcode() != spv::Op::OpReturnValue)
    return;

  assert(return_flag_ && "Did not generate the return flag variable.");

  if (!constant_true_) {
    analysis::Bool temp;
    const analysis::Bool* bool_type =
        context()->get_type_mgr()->GetRegisteredType(&temp)->AsBool();

    analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
    const analysis::Constant* true_const =
        const_mgr->GetConstant(bool_type, {true});
    constant_true_ = const_mgr->GetDefiningInstruction(true_const);
    context()->UpdateDefUse(constant_true_);
  }

  std::unique_ptr<Instruction> return_store(new Instruction(
      context(), spv::Op::OpStore, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {return_flag_->result_id()}},
          {SPV_OPERAND_TYPE_ID, {constant_true_->result_id()}}}));

  Instruction* store_inst =
      &*block->tail().InsertBefore(std::move(return_store));
  context()->set_instr_block(store_inst, block);
  context()->AnalyzeDefUse(store_inst);
}

void MergeReturnPass::RecordReturnValue(BasicBlock* block) {
  auto terminator = *block->tail();
  if (terminator.opcode() != spv::Op::OpReturnValue) {
    return;
  }

  assert(return_value_ &&
         "Did not generate the variable to hold the return value.");

  std::unique_ptr<Instruction> value_store(new Instruction(
      context(), spv::Op::OpStore, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {return_value_->result_id()}},
          {SPV_OPERAND_TYPE_ID, {terminator.GetSingleWordInOperand(0u)}}}));

  Instruction* store_inst =
      &*block->tail().InsertBefore(std::move(value_store));
  context()->set_instr_block(store_inst, block);
  context()->AnalyzeDefUse(store_inst);
}

void MergeReturnPass::AddReturnValue() {
  if (return_value_) return;

  uint32_t return_type_id = function_->type_id();
  if (get_def_use_mgr()->GetDef(return_type_id)->opcode() ==
      spv::Op::OpTypeVoid)
    return;

  uint32_t return_ptr_type = context()->get_type_mgr()->FindPointerToType(
      return_type_id, spv::StorageClass::Function);

  uint32_t var_id = TakeNextId();
  std::unique_ptr<Instruction> returnValue(
      new Instruction(context(), spv::Op::OpVariable, return_ptr_type, var_id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_STORAGE_CLASS,
                           {uint32_t(spv::StorageClass::Function)}}}));

  auto insert_iter = function_->begin()->begin();
  insert_iter.InsertBefore(std::move(returnValue));
  BasicBlock* entry_block = &*function_->begin();
  return_value_ = &*entry_block->begin();
  context()->AnalyzeDefUse(return_value_);
  context()->set_instr_block(return_value_, entry_block);

  context()->get_decoration_mgr()->CloneDecorations(
      function_->result_id(), var_id, {spv::Decoration::RelaxedPrecision});
}

void MergeReturnPass::AddReturnFlag() {
  if (return_flag_) return;

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  analysis::Bool temp;
  uint32_t bool_id = type_mgr->GetTypeInstruction(&temp);
  analysis::Bool* bool_type = type_mgr->GetType(bool_id)->AsBool();

  const analysis::Constant* false_const =
      const_mgr->GetConstant(bool_type, {false});
  uint32_t const_false_id =
      const_mgr->GetDefiningInstruction(false_const)->result_id();

  uint32_t bool_ptr_id =
      type_mgr->FindPointerToType(bool_id, spv::StorageClass::Function);

  uint32_t var_id = TakeNextId();
  std::unique_ptr<Instruction> returnFlag(new Instruction(
      context(), spv::Op::OpVariable, bool_ptr_id, var_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_STORAGE_CLASS,
                                      {uint32_t(spv::StorageClass::Function)}},
                                     {SPV_OPERAND_TYPE_ID, {const_false_id}}}));

  auto insert_iter = function_->begin()->begin();

  insert_iter.InsertBefore(std::move(returnFlag));
  BasicBlock* entry_block = &*function_->begin();
  return_flag_ = &*entry_block->begin();
  context()->AnalyzeDefUse(return_flag_);
  context()->set_instr_block(return_flag_, entry_block);
}

std::vector<BasicBlock*> MergeReturnPass::CollectReturnBlocks(
    Function* function) {
  std::vector<BasicBlock*> return_blocks;
  for (auto& block : *function) {
    Instruction& terminator = *block.tail();
    if (terminator.opcode() == spv::Op::OpReturn ||
        terminator.opcode() == spv::Op::OpReturnValue) {
      return_blocks.push_back(&block);
    }
  }
  return return_blocks;
}

void MergeReturnPass::MergeReturnBlocks(
    Function* function, const std::vector<BasicBlock*>& return_blocks) {
  if (return_blocks.size() <= 1) {
    // No work to do.
    return;
  }

  CreateReturnBlock();
  uint32_t return_id = final_return_block_->id();
  auto ret_block_iter = --function->end();
  // Create the PHI for the merged block (if necessary).
  // Create new return.
  std::vector<Operand> phi_ops;
  for (auto block : return_blocks) {
    if (block->tail()->opcode() == spv::Op::OpReturnValue) {
      phi_ops.push_back(
          {SPV_OPERAND_TYPE_ID, {block->tail()->GetSingleWordInOperand(0u)}});
      phi_ops.push_back({SPV_OPERAND_TYPE_ID, {block->id()}});
    }
  }

  if (!phi_ops.empty()) {
    // Need a PHI node to select the correct return value.
    uint32_t phi_result_id = TakeNextId();
    uint32_t phi_type_id = function->type_id();
    std::unique_ptr<Instruction> phi_inst(new Instruction(
        context(), spv::Op::OpPhi, phi_type_id, phi_result_id, phi_ops));
    ret_block_iter->AddInstruction(std::move(phi_inst));
    BasicBlock::iterator phiIter = ret_block_iter->tail();

    std::unique_ptr<Instruction> return_inst(
        new Instruction(context(), spv::Op::OpReturnValue, 0u, 0u,
                        {{SPV_OPERAND_TYPE_ID, {phi_result_id}}}));
    ret_block_iter->AddInstruction(std::move(return_inst));
    BasicBlock::iterator ret = ret_block_iter->tail();

    // Register the phi def and mark instructions for use updates.
    get_def_use_mgr()->AnalyzeInstDefUse(&*phiIter);
    get_def_use_mgr()->AnalyzeInstDef(&*ret);
  } else {
    std::unique_ptr<Instruction> return_inst(
        new Instruction(context(), spv::Op::OpReturn));
    ret_block_iter->AddInstruction(std::move(return_inst));
  }

  // Replace returns with branches
  for (auto block : return_blocks) {
    context()->ForgetUses(block->terminator());
    block->tail()->SetOpcode(spv::Op::OpBranch);
    block->tail()->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {return_id}}});
    get_def_use_mgr()->AnalyzeInstUse(block->terminator());
    get_def_use_mgr()->AnalyzeInstUse(block->GetLabelInst());
  }

  get_def_use_mgr()->AnalyzeInstDefUse(ret_block_iter->GetLabelInst());
}

void MergeReturnPass::AddNewPhiNodes() {
  std::list<BasicBlock*> order;
  cfg()->ComputeStructuredOrder(function_, &*function_->begin(), &order);

  for (BasicBlock* bb : order) {
    AddNewPhiNodes(bb);
  }
}

void MergeReturnPass::AddNewPhiNodes(BasicBlock* bb) {
  // New phi nodes are needed for any id whose definition used to dominate |bb|,
  // but no longer dominates |bb|.  These are found by walking the dominator
  // tree starting at the original immediate dominator of |bb| and ending at its
  // current dominator.

  // Because we are walking the updated dominator tree it is important that the
  // new phi nodes for the original dominators of |bb| have already been added.
  // Otherwise some ids might be missed.  Consider the case where bb1 dominates
  // bb2, and bb2 dominates bb3.  Suppose there are changes such that bb1 no
  // longer dominates bb2 and the same for bb2 and bb3.  This algorithm will not
  // look at the ids defined in bb1.  However, calling |AddNewPhiNodes(bb2)|
  // first will add a phi node in bb2 for that value.  Then a call to
  // |AddNewPhiNodes(bb3)| will process that value by processing the phi in bb2.
  DominatorAnalysis* dom_tree = context()->GetDominatorAnalysis(function_);

  BasicBlock* dominator = dom_tree->ImmediateDominator(bb);
  if (dominator == nullptr) {
    return;
  }

  BasicBlock* current_bb = context()->get_instr_block(original_dominator_[bb]);
  while (current_bb != nullptr && current_bb != dominator) {
    for (Instruction& inst : *current_bb) {
      CreatePhiNodesForInst(bb, inst);
    }
    current_bb = dom_tree->ImmediateDominator(current_bb);
  }
}

void MergeReturnPass::RecordImmediateDominators(Function* function) {
  DominatorAnalysis* dom_tree = context()->GetDominatorAnalysis(function);
  for (BasicBlock& bb : *function) {
    BasicBlock* dominator_bb = dom_tree->ImmediateDominator(&bb);
    if (dominator_bb && dominator_bb != cfg()->pseudo_entry_block()) {
      original_dominator_[&bb] = dominator_bb->terminator();
    } else {
      original_dominator_[&bb] = nullptr;
    }
  }
}

void MergeReturnPass::InsertAfterElement(BasicBlock* element,
                                         BasicBlock* new_element,
                                         std::list<BasicBlock*>* list) {
  auto pos = std::find(list->begin(), list->end(), element);
  assert(pos != list->end());
  ++pos;
  list->insert(pos, new_element);
}

bool MergeReturnPass::AddSingleCaseSwitchAroundFunction() {
  CreateReturnBlock();
  CreateReturn(final_return_block_);

  if (context()->AreAnalysesValid(IRContext::kAnalysisCFG)) {
    cfg()->RegisterBlock(final_return_block_);
  }

  if (!CreateSingleCaseSwitch(final_return_block_)) {
    return false;
  }
  return true;
}

BasicBlock* MergeReturnPass::CreateContinueTarget(uint32_t header_label_id) {
  std::unique_ptr<Instruction> label(
      new Instruction(context(), spv::Op::OpLabel, 0u, TakeNextId(), {}));

  // Create the new basic block
  std::unique_ptr<BasicBlock> block(new BasicBlock(std::move(label)));

  // Insert the new block just before the return block
  auto pos = function_->end();
  assert(pos != function_->begin());
  pos--;
  assert(pos != function_->begin());
  assert(&*pos == final_return_block_);
  auto new_block = &*pos.InsertBefore(std::move(block));
  new_block->SetParent(function_);

  context()->AnalyzeDefUse(new_block->GetLabelInst());
  context()->set_instr_block(new_block->GetLabelInst(), new_block);

  InstructionBuilder builder(
      context(), new_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  builder.AddBranch(header_label_id);

  if (context()->AreAnalysesValid(IRContext::kAnalysisCFG)) {
    cfg()->RegisterBlock(new_block);
  }

  return new_block;
}

bool MergeReturnPass::CreateSingleCaseSwitch(BasicBlock* merge_target) {
  // Insert the switch before any code is run.  We have to split the entry
  // block to make sure the OpVariable instructions remain in the entry block.
  BasicBlock* start_block = &*function_->begin();
  auto split_pos = start_block->begin();
  while (split_pos->opcode() == spv::Op::OpVariable) {
    ++split_pos;
  }

  BasicBlock* old_block =
      start_block->SplitBasicBlock(context(), TakeNextId(), split_pos);

  // Add the switch to the end of the entry block.
  InstructionBuilder builder(
      context(), start_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t const_zero_id = builder.GetUintConstantId(0u);
  if (const_zero_id == 0) {
    return false;
  }
  builder.AddSwitch(const_zero_id, old_block->id(), {}, merge_target->id());

  if (context()->AreAnalysesValid(IRContext::kAnalysisCFG)) {
    cfg()->RegisterBlock(old_block);
    cfg()->AddEdges(start_block);
  }
  return true;
}

bool MergeReturnPass::HasNontrivialUnreachableBlocks(Function* function) {
  utils::BitVector reachable_blocks;
  cfg()->ForEachBlockInPostOrder(
      function->entry().get(),
      [&reachable_blocks](BasicBlock* bb) { reachable_blocks.Set(bb->id()); });

  for (auto& bb : *function) {
    if (reachable_blocks.Get(bb.id())) {
      continue;
    }

    StructuredCFGAnalysis* struct_cfg_analysis =
        context()->GetStructuredCFGAnalysis();
    if (struct_cfg_analysis->IsContinueBlock(bb.id())) {
      // |bb| must be an empty block ending with a branch to the header.
      Instruction* inst = &*bb.begin();
      if (inst->opcode() != spv::Op::OpBranch) {
        return true;
      }

      if (inst->GetSingleWordInOperand(0) !=
          struct_cfg_analysis->ContainingLoop(bb.id())) {
        return true;
      }
    } else if (struct_cfg_analysis->IsMergeBlock(bb.id())) {
      // |bb| must be an empty block ending with OpUnreachable.
      if (bb.begin()->opcode() != spv::Op::OpUnreachable) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

}  // namespace opt
}  // namespace spvtools
