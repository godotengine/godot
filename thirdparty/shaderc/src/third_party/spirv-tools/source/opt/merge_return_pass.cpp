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
      context()->get_feature_mgr()->HasCapability(SpvCapabilityShader);

  bool failed = false;
  ProcessFunction pfn = [&failed, is_shader, this](Function* function) {
    std::vector<BasicBlock*> return_blocks = CollectReturnBlocks(function);
    if (return_blocks.size() <= 1) {
      return false;
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

  AddDummyLoopAroundFunction();

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

    // Generate state for next block
    if (Instruction* mergeInst = block->GetMergeInst()) {
      Instruction* loopMergeInst = block->GetLoopMergeInst();
      if (!loopMergeInst) loopMergeInst = state_.back().LoopMergeInst();
      state_.emplace_back(loopMergeInst, mergeInst);
    }
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

    // Generate state for next block
    if (Instruction* mergeInst = block->GetMergeInst()) {
      Instruction* loopMergeInst = block->GetLoopMergeInst();
      if (!loopMergeInst) loopMergeInst = state_.back().LoopMergeInst();
      state_.emplace_back(loopMergeInst, mergeInst);
    }
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
      new Instruction(context(), SpvOpLabel, 0u, TakeNextId(), {}));

  // Create the new basic block
  std::unique_ptr<BasicBlock> return_block(
      new BasicBlock(std::move(return_label)));
  function_->AddBasicBlock(std::move(return_block));
  final_return_block_ = &*(--function_->end());
  context()->AnalyzeDefUse(final_return_block_->GetLabelInst());
  context()->set_instr_block(final_return_block_->GetLabelInst(),
                             final_return_block_);
  final_return_block_->SetParent(function_);
}

void MergeReturnPass::CreateReturn(BasicBlock* block) {
  AddReturnValue();

  if (return_value_) {
    // Load and return the final return value
    uint32_t loadId = TakeNextId();
    block->AddInstruction(MakeUnique<Instruction>(
        context(), SpvOpLoad, function_->type_id(), loadId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_ID, {return_value_->result_id()}}}));
    Instruction* var_inst = block->terminator();
    context()->AnalyzeDefUse(var_inst);
    context()->set_instr_block(var_inst, block);
    context()->get_decoration_mgr()->CloneDecorations(
        return_value_->result_id(), loadId, {SpvDecorationRelaxedPrecision});

    block->AddInstruction(MakeUnique<Instruction>(
        context(), SpvOpReturnValue, 0, 0,
        std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {loadId}}}));
    context()->AnalyzeDefUse(block->terminator());
    context()->set_instr_block(block->terminator(), block);
  } else {
    block->AddInstruction(MakeUnique<Instruction>(context(), SpvOpReturn));
    context()->AnalyzeDefUse(block->terminator());
    context()->set_instr_block(block->terminator(), block);
  }
}

void MergeReturnPass::ProcessStructuredBlock(BasicBlock* block) {
  SpvOp tail_opcode = block->tail()->opcode();
  if (tail_opcode == SpvOpReturn || tail_opcode == SpvOpReturnValue) {
    if (!return_flag_) {
      AddReturnFlag();
    }
  }

  if (tail_opcode == SpvOpReturn || tail_opcode == SpvOpReturnValue ||
      tail_opcode == SpvOpUnreachable) {
    assert(CurrentState().InLoop() && "Should be in the dummy loop.");
    BranchToBlock(block, CurrentState().LoopMergeId());
    return_blocks_.insert(block->id());
  }
}

void MergeReturnPass::BranchToBlock(BasicBlock* block, uint32_t target) {
  if (block->tail()->opcode() == SpvOpReturn ||
      block->tail()->opcode() == SpvOpReturnValue) {
    RecordReturned(block);
    RecordReturnValue(block);
  }

  BasicBlock* target_block = context()->get_instr_block(target);
  if (target_block->GetLoopMergeInst()) {
    cfg()->SplitLoopHeader(target_block);
  }
  UpdatePhiNodes(block, target_block);

  Instruction* return_inst = block->terminator();
  return_inst->SetOpcode(SpvOpBranch);
  return_inst->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {target}}});
  context()->get_def_use_mgr()->AnalyzeInstDefUse(return_inst);
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

  const auto& target_pred = cfg()->preds(target->id());
  if (target_pred.size() == 1) {
    MarkForNewPhiNodes(target, context()->get_instr_block(target_pred[0]));
  } else {
    // If the loop contained a break and a return, OpPhi instructions may be
    // required starting from the dominator of the loop merge.
    DominatorAnalysis* dom_tree =
        context()->GetDominatorAnalysis(target->GetParent());
    auto idom = dom_tree->ImmediateDominator(target);
    if (idom) {
      MarkForNewPhiNodes(target, idom);
    }
  }
}

void MergeReturnPass::CreatePhiNodesForInst(BasicBlock* merge_block,
                                            Instruction& inst) {
  DominatorAnalysis* dom_tree =
      context()->GetDominatorAnalysis(merge_block->GetParent());
  BasicBlock* inst_bb = context()->get_instr_block(&inst);

  if (inst.result_id() != 0) {
    std::vector<Instruction*> users_to_update;
    context()->get_def_use_mgr()->ForEachUser(
        &inst,
        [&users_to_update, &dom_tree, &inst, inst_bb, this](Instruction* user) {
          BasicBlock* user_bb = nullptr;
          if (user->opcode() != SpvOpPhi) {
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
    InstructionBuilder builder(
        context(), &*merge_block->begin(),
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    uint32_t undef_id = Type2Undef(inst.type_id());
    std::vector<uint32_t> phi_operands;

    // Add the OpPhi operands. If the predecessor is a return block use undef,
    // otherwise use |inst|'s id.
    std::vector<uint32_t> preds = cfg()->preds(merge_block->id());
    for (uint32_t pred_id : preds) {
      if (return_blocks_.count(pred_id)) {
        phi_operands.push_back(undef_id);
      } else {
        phi_operands.push_back(inst.result_id());
      }
      phi_operands.push_back(pred_id);
    }

    Instruction* new_phi = builder.AddPhi(inst.type_id(), phi_operands);
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
  } else if (block->id() == state->LoopMergeId()) {
    while (state->LoopMergeId() == block->id()) {
      state++;
    }
  }

  while (block != nullptr && block != final_return_block_) {
    if (!predicated->insert(block).second) break;
    // Skip structured subgraphs.
    assert(state->InLoop() && "Should be in the dummy loop at the very least.");
    Instruction* current_loop_merge_inst = state->LoopMergeInst();
    uint32_t merge_block_id =
        current_loop_merge_inst->GetSingleWordInOperand(0);
    while (state->LoopMergeId() == merge_block_id) {
      state++;
    }
    if (!BreakFromConstruct(block, predicated, order,
                            current_loop_merge_inst)) {
      return false;
    }
    block = context()->get_instr_block(merge_block_id);
  }
  return true;
}

bool MergeReturnPass::BreakFromConstruct(
    BasicBlock* block, std::unordered_set<BasicBlock*>* predicated,
    std::list<BasicBlock*>* order, Instruction* loop_merge_inst) {
  assert(loop_merge_inst->opcode() == SpvOpLoopMerge &&
         "loop_merge_inst must be a loop merge instruction.");
  // Make sure the CFG is build here.  If we don't then it becomes very hard
  // to know which new blocks need to be updated.
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

  uint32_t merge_block_id = loop_merge_inst->GetSingleWordInOperand(0);
  BasicBlock* merge_block = context()->get_instr_block(merge_block_id);
  if (merge_block->GetLoopMergeInst()) {
    cfg()->SplitLoopHeader(merge_block);
  }

  // Leave the phi instructions behind.
  auto iter = block->begin();
  while (iter->opcode() == SpvOpPhi) {
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
  if (loop_merge_inst->GetSingleWordInOperand(1) == block->id()) {
    loop_merge_inst->SetInOperand(1, {old_body->id()});
    context()->UpdateDefUse(loop_merge_inst);
  }

  // Update |order| so old_block will be traversed.
  InsertAfterElement(block, old_body, order);

  // Within the new header we need the following:
  // 1. Load of the return status flag
  // 2. Branch to |merge_block| (true) or old body (false)
  // 3. Update OpPhi instructions in |merge_block|.
  // 4. Update the CFG.
  //
  // Sine we are branching to the merge block of the current construct, there is
  // no need for an OpSelectionMerge.

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

  // 3. Update OpPhi instructions in |merge_block|.
  BasicBlock* merge_original_pred = MarkedSinglePred(merge_block);
  if (merge_original_pred == nullptr) {
    UpdatePhiNodes(block, merge_block);
  } else if (merge_original_pred == block) {
    MarkForNewPhiNodes(merge_block, old_body);
  }

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
  if (block->tail()->opcode() != SpvOpReturn &&
      block->tail()->opcode() != SpvOpReturnValue)
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
      context(), SpvOpStore, 0, 0,
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
  if (terminator.opcode() != SpvOpReturnValue) {
    return;
  }

  assert(return_value_ &&
         "Did not generate the variable to hold the return value.");

  std::unique_ptr<Instruction> value_store(new Instruction(
      context(), SpvOpStore, 0, 0,
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
  if (get_def_use_mgr()->GetDef(return_type_id)->opcode() == SpvOpTypeVoid)
    return;

  uint32_t return_ptr_type = context()->get_type_mgr()->FindPointerToType(
      return_type_id, SpvStorageClassFunction);

  uint32_t var_id = TakeNextId();
  std::unique_ptr<Instruction> returnValue(new Instruction(
      context(), SpvOpVariable, return_ptr_type, var_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}}}));

  auto insert_iter = function_->begin()->begin();
  insert_iter.InsertBefore(std::move(returnValue));
  BasicBlock* entry_block = &*function_->begin();
  return_value_ = &*entry_block->begin();
  context()->AnalyzeDefUse(return_value_);
  context()->set_instr_block(return_value_, entry_block);

  context()->get_decoration_mgr()->CloneDecorations(
      function_->result_id(), var_id, {SpvDecorationRelaxedPrecision});
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
      type_mgr->FindPointerToType(bool_id, SpvStorageClassFunction);

  uint32_t var_id = TakeNextId();
  std::unique_ptr<Instruction> returnFlag(new Instruction(
      context(), SpvOpVariable, bool_ptr_id, var_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}},
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
    if (terminator.opcode() == SpvOpReturn ||
        terminator.opcode() == SpvOpReturnValue) {
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
    if (block->tail()->opcode() == SpvOpReturnValue) {
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
        context(), SpvOpPhi, phi_type_id, phi_result_id, phi_ops));
    ret_block_iter->AddInstruction(std::move(phi_inst));
    BasicBlock::iterator phiIter = ret_block_iter->tail();

    std::unique_ptr<Instruction> return_inst(
        new Instruction(context(), SpvOpReturnValue, 0u, 0u,
                        {{SPV_OPERAND_TYPE_ID, {phi_result_id}}}));
    ret_block_iter->AddInstruction(std::move(return_inst));
    BasicBlock::iterator ret = ret_block_iter->tail();

    // Register the phi def and mark instructions for use updates.
    get_def_use_mgr()->AnalyzeInstDefUse(&*phiIter);
    get_def_use_mgr()->AnalyzeInstDef(&*ret);
  } else {
    std::unique_ptr<Instruction> return_inst(
        new Instruction(context(), SpvOpReturn));
    ret_block_iter->AddInstruction(std::move(return_inst));
  }

  // Replace returns with branches
  for (auto block : return_blocks) {
    context()->ForgetUses(block->terminator());
    block->tail()->SetOpcode(SpvOpBranch);
    block->tail()->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {return_id}}});
    get_def_use_mgr()->AnalyzeInstUse(block->terminator());
    get_def_use_mgr()->AnalyzeInstUse(block->GetLabelInst());
  }

  get_def_use_mgr()->AnalyzeInstDefUse(ret_block_iter->GetLabelInst());
}

void MergeReturnPass::AddNewPhiNodes() {
  DominatorAnalysis* dom_tree = context()->GetDominatorAnalysis(function_);
  std::list<BasicBlock*> order;
  cfg()->ComputeStructuredOrder(function_, &*function_->begin(), &order);

  for (BasicBlock* bb : order) {
    BasicBlock* dominator = dom_tree->ImmediateDominator(bb);
    if (dominator) {
      AddNewPhiNodes(bb, new_merge_nodes_[bb], dominator->id());
    }
  }
}

void MergeReturnPass::AddNewPhiNodes(BasicBlock* bb, BasicBlock* pred,
                                     uint32_t header_id) {
  DominatorAnalysis* dom_tree = context()->GetDominatorAnalysis(function_);
  // Insert as a stopping point.  We do not have to add anything in the block
  // or above because the header dominates |bb|.

  BasicBlock* current_bb = pred;
  while (current_bb != nullptr && current_bb->id() != header_id) {
    for (Instruction& inst : *current_bb) {
      CreatePhiNodesForInst(bb, inst);
    }
    current_bb = dom_tree->ImmediateDominator(current_bb);
  }
}

void MergeReturnPass::MarkForNewPhiNodes(BasicBlock* block,
                                         BasicBlock* single_original_pred) {
  new_merge_nodes_[block] = single_original_pred;
}

void MergeReturnPass::InsertAfterElement(BasicBlock* element,
                                         BasicBlock* new_element,
                                         std::list<BasicBlock*>* list) {
  auto pos = std::find(list->begin(), list->end(), element);
  assert(pos != list->end());
  ++pos;
  list->insert(pos, new_element);
}

void MergeReturnPass::AddDummyLoopAroundFunction() {
  CreateReturnBlock();
  CreateReturn(final_return_block_);

  if (context()->AreAnalysesValid(IRContext::kAnalysisCFG)) {
    cfg()->RegisterBlock(final_return_block_);
  }

  CreateDummyLoop(final_return_block_);
}

BasicBlock* MergeReturnPass::CreateContinueTarget(uint32_t header_label_id) {
  std::unique_ptr<Instruction> label(
      new Instruction(context(), SpvOpLabel, 0u, TakeNextId(), {}));

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

void MergeReturnPass::CreateDummyLoop(BasicBlock* merge_target) {
  std::unique_ptr<Instruction> label(
      new Instruction(context(), SpvOpLabel, 0u, TakeNextId(), {}));

  // Create the new basic block
  std::unique_ptr<BasicBlock> block(new BasicBlock(std::move(label)));

  // Insert the new block before any code is run.  We have to split the entry
  // block to make sure the OpVariable instructions remain in the entry block.
  BasicBlock* start_block = &*function_->begin();
  auto split_pos = start_block->begin();
  while (split_pos->opcode() == SpvOpVariable) {
    ++split_pos;
  }

  BasicBlock* old_block =
      start_block->SplitBasicBlock(context(), TakeNextId(), split_pos);

  // The new block must be inserted after the entry block.  We cannot make the
  // entry block the header for the dummy loop because it is not valid to have a
  // branch to the entry block, and the continue target must branch back to the
  // loop header.
  auto pos = function_->begin();
  pos++;
  BasicBlock* header_block = &*pos.InsertBefore(std::move(block));
  context()->AnalyzeDefUse(header_block->GetLabelInst());
  header_block->SetParent(function_);

  // We have to create the continue block before OpLoopMerge instruction.
  // Otherwise the def-use manager will compalain that there is a use without a
  // definition.
  uint32_t continue_target = CreateContinueTarget(header_block->id())->id();

  // Add the code the the header block.
  InstructionBuilder builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  builder.AddLoopMerge(merge_target->id(), continue_target);
  builder.AddBranch(old_block->id());

  // Fix up the entry block by adding a branch to the loop header.
  InstructionBuilder builder2(
      context(), start_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  builder2.AddBranch(header_block->id());

  if (context()->AreAnalysesValid(IRContext::kAnalysisCFG)) {
    cfg()->RegisterBlock(old_block);
    cfg()->RegisterBlock(header_block);
    cfg()->AddEdges(start_block);
  }
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
    if (struct_cfg_analysis->IsMergeBlock(bb.id())) {
      // |bb| must be an empty block ending with OpUnreachable.
      if (bb.begin()->opcode() != SpvOpUnreachable) {
        return true;
      }
    } else if (struct_cfg_analysis->IsContinueBlock(bb.id())) {
      // |bb| must be an empty block ending with a branch to the header.
      Instruction* inst = &*bb.begin();
      if (inst->opcode() != SpvOpBranch) {
        return true;
      }

      if (inst->GetSingleWordInOperand(0) !=
          struct_cfg_analysis->ContainingLoop(bb.id())) {
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
