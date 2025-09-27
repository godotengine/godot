// Copyright (c) 2018 Google LLC.
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

#include "source/opt/loop_fusion.h"

#include <algorithm>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/loop_dependence.h"
#include "source/opt/loop_descriptor.h"

namespace spvtools {
namespace opt {
namespace {

// Append all the loops nested in |loop| to |loops|.
void CollectChildren(Loop* loop, std::vector<const Loop*>* loops) {
  for (auto child : *loop) {
    loops->push_back(child);
    if (child->NumImmediateChildren() != 0) {
      CollectChildren(child, loops);
    }
  }
}

// Return the set of locations accessed by |stores| and |loads|.
std::set<Instruction*> GetLocationsAccessed(
    const std::map<Instruction*, std::vector<Instruction*>>& stores,
    const std::map<Instruction*, std::vector<Instruction*>>& loads) {
  std::set<Instruction*> locations{};

  for (const auto& kv : stores) {
    locations.insert(std::get<0>(kv));
  }

  for (const auto& kv : loads) {
    locations.insert(std::get<0>(kv));
  }

  return locations;
}

// Append all dependences from |sources| to |destinations| to |dependences|.
void GetDependences(std::vector<DistanceVector>* dependences,
                    LoopDependenceAnalysis* analysis,
                    const std::vector<Instruction*>& sources,
                    const std::vector<Instruction*>& destinations,
                    size_t num_entries) {
  for (auto source : sources) {
    for (auto destination : destinations) {
      DistanceVector dist(num_entries);
      if (!analysis->GetDependence(source, destination, &dist)) {
        dependences->push_back(dist);
      }
    }
  }
}

// Apped all instructions in |block| to |instructions|.
void AddInstructionsInBlock(std::vector<Instruction*>* instructions,
                            BasicBlock* block) {
  for (auto& inst : *block) {
    instructions->push_back(&inst);
  }

  instructions->push_back(block->GetLabelInst());
}

}  // namespace

bool LoopFusion::UsedInContinueOrConditionBlock(Instruction* phi_instruction,
                                                Loop* loop) {
  auto condition_block = loop->FindConditionBlock()->id();
  auto continue_block = loop->GetContinueBlock()->id();
  auto not_used = context_->get_def_use_mgr()->WhileEachUser(
      phi_instruction,
      [this, condition_block, continue_block](Instruction* instruction) {
        auto block_id = context_->get_instr_block(instruction)->id();
        return block_id != condition_block && block_id != continue_block;
      });

  return !not_used;
}

void LoopFusion::RemoveIfNotUsedContinueOrConditionBlock(
    std::vector<Instruction*>* instructions, Loop* loop) {
  instructions->erase(
      std::remove_if(std::begin(*instructions), std::end(*instructions),
                     [this, loop](Instruction* instruction) {
                       return !UsedInContinueOrConditionBlock(instruction,
                                                              loop);
                     }),
      std::end(*instructions));
}

bool LoopFusion::AreCompatible() {
  // Check that the loops are in the same function.
  if (loop_0_->GetHeaderBlock()->GetParent() !=
      loop_1_->GetHeaderBlock()->GetParent()) {
    return false;
  }

  // Check that both loops have pre-header blocks.
  if (!loop_0_->GetPreHeaderBlock() || !loop_1_->GetPreHeaderBlock()) {
    return false;
  }

  // Check there are no breaks.
  if (context_->cfg()->preds(loop_0_->GetMergeBlock()->id()).size() != 1 ||
      context_->cfg()->preds(loop_1_->GetMergeBlock()->id()).size() != 1) {
    return false;
  }

  // Check there are no continues.
  if (context_->cfg()->preds(loop_0_->GetContinueBlock()->id()).size() != 1 ||
      context_->cfg()->preds(loop_1_->GetContinueBlock()->id()).size() != 1) {
    return false;
  }

  // |GetInductionVariables| returns all OpPhi in the header. Check that both
  // loops have exactly one that is used in the continue and condition blocks.
  std::vector<Instruction*> inductions_0{}, inductions_1{};
  loop_0_->GetInductionVariables(inductions_0);
  RemoveIfNotUsedContinueOrConditionBlock(&inductions_0, loop_0_);

  if (inductions_0.size() != 1) {
    return false;
  }

  induction_0_ = inductions_0.front();

  loop_1_->GetInductionVariables(inductions_1);
  RemoveIfNotUsedContinueOrConditionBlock(&inductions_1, loop_1_);

  if (inductions_1.size() != 1) {
    return false;
  }

  induction_1_ = inductions_1.front();

  if (!CheckInit()) {
    return false;
  }

  if (!CheckCondition()) {
    return false;
  }

  if (!CheckStep()) {
    return false;
  }

  // Check adjacency, |loop_0_| should come just before |loop_1_|.
  // There is always at least one block between loops, even if it's empty.
  // We'll check at most 2 preceding blocks.

  auto pre_header_1 = loop_1_->GetPreHeaderBlock();

  std::vector<BasicBlock*> block_to_check{};
  block_to_check.push_back(pre_header_1);

  if (loop_0_->GetMergeBlock() != loop_1_->GetPreHeaderBlock()) {
    // Follow CFG for one more block.
    auto preds = context_->cfg()->preds(pre_header_1->id());
    if (preds.size() == 1) {
      auto block = &*containing_function_->FindBlock(preds.front());
      if (block == loop_0_->GetMergeBlock()) {
        block_to_check.push_back(block);
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  // Check that the separating blocks are either empty or only contains a store
  // to a local variable that is never read (left behind by
  // '--eliminate-local-multi-store'). Also allow OpPhi, since the loop could be
  // in LCSSA form.
  for (auto block : block_to_check) {
    for (auto& inst : *block) {
      if (inst.opcode() == spv::Op::OpStore) {
        // Get the definition of the target to check it's function scope so
        // there are no observable side effects.
        auto variable =
            context_->get_def_use_mgr()->GetDef(inst.GetSingleWordInOperand(0));

        if (variable->opcode() != spv::Op::OpVariable ||
            spv::StorageClass(variable->GetSingleWordInOperand(0)) !=
                spv::StorageClass::Function) {
          return false;
        }

        // Check the target is never loaded.
        auto is_used = false;
        context_->get_def_use_mgr()->ForEachUse(
            inst.GetSingleWordInOperand(0),
            [&is_used](Instruction* use_inst, uint32_t) {
              if (use_inst->opcode() == spv::Op::OpLoad) {
                is_used = true;
              }
            });

        if (is_used) {
          return false;
        }
      } else if (inst.opcode() == spv::Op::OpPhi) {
        if (inst.NumInOperands() != 2) {
          return false;
        }
      } else if (inst.opcode() != spv::Op::OpBranch) {
        return false;
      }
    }
  }

  return true;
}  // namespace opt

bool LoopFusion::ContainsBarriersOrFunctionCalls(Loop* loop) {
  for (const auto& block : loop->GetBlocks()) {
    for (const auto& inst : *containing_function_->FindBlock(block)) {
      auto opcode = inst.opcode();
      if (opcode == spv::Op::OpFunctionCall ||
          opcode == spv::Op::OpControlBarrier ||
          opcode == spv::Op::OpMemoryBarrier ||
          opcode == spv::Op::OpTypeNamedBarrier ||
          opcode == spv::Op::OpNamedBarrierInitialize ||
          opcode == spv::Op::OpMemoryNamedBarrier) {
        return true;
      }
    }
  }

  return false;
}

bool LoopFusion::CheckInit() {
  int64_t loop_0_init;
  if (!loop_0_->GetInductionInitValue(induction_0_, &loop_0_init)) {
    return false;
  }

  int64_t loop_1_init;
  if (!loop_1_->GetInductionInitValue(induction_1_, &loop_1_init)) {
    return false;
  }

  if (loop_0_init != loop_1_init) {
    return false;
  }

  return true;
}

bool LoopFusion::CheckCondition() {
  auto condition_0 = loop_0_->GetConditionInst();
  auto condition_1 = loop_1_->GetConditionInst();

  if (!loop_0_->IsSupportedCondition(condition_0->opcode()) ||
      !loop_1_->IsSupportedCondition(condition_1->opcode())) {
    return false;
  }

  if (condition_0->opcode() != condition_1->opcode()) {
    return false;
  }

  for (uint32_t i = 0; i < condition_0->NumInOperandWords(); ++i) {
    auto arg_0 = context_->get_def_use_mgr()->GetDef(
        condition_0->GetSingleWordInOperand(i));
    auto arg_1 = context_->get_def_use_mgr()->GetDef(
        condition_1->GetSingleWordInOperand(i));

    if (arg_0 == induction_0_ && arg_1 == induction_1_) {
      continue;
    }

    if (arg_0 == induction_0_ && arg_1 != induction_1_) {
      return false;
    }

    if (arg_1 == induction_1_ && arg_0 != induction_0_) {
      return false;
    }

    if (arg_0 != arg_1) {
      return false;
    }
  }

  return true;
}

bool LoopFusion::CheckStep() {
  auto scalar_analysis = context_->GetScalarEvolutionAnalysis();
  SENode* induction_node_0 = scalar_analysis->SimplifyExpression(
      scalar_analysis->AnalyzeInstruction(induction_0_));
  if (!induction_node_0->AsSERecurrentNode()) {
    return false;
  }

  SENode* induction_step_0 =
      induction_node_0->AsSERecurrentNode()->GetCoefficient();
  if (!induction_step_0->AsSEConstantNode()) {
    return false;
  }

  SENode* induction_node_1 = scalar_analysis->SimplifyExpression(
      scalar_analysis->AnalyzeInstruction(induction_1_));
  if (!induction_node_1->AsSERecurrentNode()) {
    return false;
  }

  SENode* induction_step_1 =
      induction_node_1->AsSERecurrentNode()->GetCoefficient();
  if (!induction_step_1->AsSEConstantNode()) {
    return false;
  }

  if (*induction_step_0 != *induction_step_1) {
    return false;
  }

  return true;
}

std::map<Instruction*, std::vector<Instruction*>> LoopFusion::LocationToMemOps(
    const std::vector<Instruction*>& mem_ops) {
  std::map<Instruction*, std::vector<Instruction*>> location_map{};

  for (auto instruction : mem_ops) {
    auto access_location = context_->get_def_use_mgr()->GetDef(
        instruction->GetSingleWordInOperand(0));

    while (access_location->opcode() == spv::Op::OpAccessChain) {
      access_location = context_->get_def_use_mgr()->GetDef(
          access_location->GetSingleWordInOperand(0));
    }

    location_map[access_location].push_back(instruction);
  }

  return location_map;
}

std::pair<std::vector<Instruction*>, std::vector<Instruction*>>
LoopFusion::GetLoadsAndStoresInLoop(Loop* loop) {
  std::vector<Instruction*> loads{};
  std::vector<Instruction*> stores{};

  for (auto block_id : loop->GetBlocks()) {
    if (block_id == loop->GetContinueBlock()->id()) {
      continue;
    }

    for (auto& instruction : *containing_function_->FindBlock(block_id)) {
      if (instruction.opcode() == spv::Op::OpLoad) {
        loads.push_back(&instruction);
      } else if (instruction.opcode() == spv::Op::OpStore) {
        stores.push_back(&instruction);
      }
    }
  }

  return std::make_pair(loads, stores);
}

bool LoopFusion::IsUsedInLoop(Instruction* instruction, Loop* loop) {
  auto not_used = context_->get_def_use_mgr()->WhileEachUser(
      instruction, [this, loop](Instruction* user) {
        auto block_id = context_->get_instr_block(user)->id();
        return !loop->IsInsideLoop(block_id);
      });

  return !not_used;
}

bool LoopFusion::IsLegal() {
  assert(AreCompatible() && "Fusion can't be legal, loops are not compatible.");

  // Bail out if there are function calls as they could have side-effects that
  // cause dependencies or if there are any barriers.
  if (ContainsBarriersOrFunctionCalls(loop_0_) ||
      ContainsBarriersOrFunctionCalls(loop_1_)) {
    return false;
  }

  std::vector<Instruction*> phi_instructions{};
  loop_0_->GetInductionVariables(phi_instructions);

  // Check no OpPhi in |loop_0_| is used in |loop_1_|.
  for (auto phi_instruction : phi_instructions) {
    if (IsUsedInLoop(phi_instruction, loop_1_)) {
      return false;
    }
  }

  // Check no LCSSA OpPhi in merge block of |loop_0_| is used in |loop_1_|.
  auto phi_used = false;
  loop_0_->GetMergeBlock()->ForEachPhiInst(
      [this, &phi_used](Instruction* phi_instruction) {
        phi_used |= IsUsedInLoop(phi_instruction, loop_1_);
      });

  if (phi_used) {
    return false;
  }

  // Grab loads & stores from both loops.
  auto loads_stores_0 = GetLoadsAndStoresInLoop(loop_0_);
  auto loads_stores_1 = GetLoadsAndStoresInLoop(loop_1_);

  // Build memory location to operation maps.
  auto load_locs_0 = LocationToMemOps(std::get<0>(loads_stores_0));
  auto store_locs_0 = LocationToMemOps(std::get<1>(loads_stores_0));

  auto load_locs_1 = LocationToMemOps(std::get<0>(loads_stores_1));
  auto store_locs_1 = LocationToMemOps(std::get<1>(loads_stores_1));

  // Get the locations accessed in both loops.
  auto locations_0 = GetLocationsAccessed(store_locs_0, load_locs_0);
  auto locations_1 = GetLocationsAccessed(store_locs_1, load_locs_1);

  std::vector<Instruction*> potential_clashes{};

  std::set_intersection(std::begin(locations_0), std::end(locations_0),
                        std::begin(locations_1), std::end(locations_1),
                        std::back_inserter(potential_clashes));

  // If the loops don't access the same variables, the fusion is legal.
  if (potential_clashes.empty()) {
    return true;
  }

  // Find variables that have at least one store.
  std::vector<Instruction*> potential_clashes_with_stores{};
  for (auto location : potential_clashes) {
    if (store_locs_0.find(location) != std::end(store_locs_0) ||
        store_locs_1.find(location) != std::end(store_locs_1)) {
      potential_clashes_with_stores.push_back(location);
    }
  }

  // If there are only loads to the same variables, the fusion is legal.
  if (potential_clashes_with_stores.empty()) {
    return true;
  }

  // Else if loads and at least one store (across loops) to the same variable
  // there is a potential dependence and we need to check the dependence
  // distance.

  // Find all the loops in this loop nest for the dependency analysis.
  std::vector<const Loop*> loops{};

  // Find the parents.
  for (auto current_loop = loop_0_; current_loop != nullptr;
       current_loop = current_loop->GetParent()) {
    loops.push_back(current_loop);
  }

  auto this_loop_position = loops.size() - 1;
  std::reverse(std::begin(loops), std::end(loops));

  // Find the children.
  CollectChildren(loop_0_, &loops);
  CollectChildren(loop_1_, &loops);

  // Check that any dependes created are legal. That means the fused loops do
  // not have any dependencies with dependence distance greater than 0 that did
  // not exist in the original loops.

  LoopDependenceAnalysis analysis(context_, loops);

  analysis.GetScalarEvolution()->AddLoopsToPretendAreTheSame(
      {loop_0_, loop_1_});

  for (auto location : potential_clashes_with_stores) {
    // Analyse dependences from |loop_0_| to |loop_1_|.
    std::vector<DistanceVector> dependences;
    // Read-After-Write.
    GetDependences(&dependences, &analysis, store_locs_0[location],
                   load_locs_1[location], loops.size());
    // Write-After-Read.
    GetDependences(&dependences, &analysis, load_locs_0[location],
                   store_locs_1[location], loops.size());
    // Write-After-Write.
    GetDependences(&dependences, &analysis, store_locs_0[location],
                   store_locs_1[location], loops.size());

    // Check that the induction variables either don't appear in the subscripts
    // or the dependence distance is negative.
    for (const auto& dependence : dependences) {
      const auto& entry = dependence.GetEntries()[this_loop_position];
      if ((entry.dependence_information ==
               DistanceEntry::DependenceInformation::DISTANCE &&
           entry.distance < 1) ||
          (entry.dependence_information ==
           DistanceEntry::DependenceInformation::IRRELEVANT)) {
        continue;
      } else {
        return false;
      }
    }
  }

  return true;
}

void ReplacePhiParentWith(Instruction* inst, uint32_t orig_block,
                          uint32_t new_block) {
  if (inst->GetSingleWordInOperand(1) == orig_block) {
    inst->SetInOperand(1, {new_block});
  } else {
    inst->SetInOperand(3, {new_block});
  }
}

void LoopFusion::Fuse() {
  assert(AreCompatible() && "Can't fuse, loops aren't compatible");
  assert(IsLegal() && "Can't fuse, illegal");

  // Save the pointers/ids, won't be found in the middle of doing modifications.
  auto header_1 = loop_1_->GetHeaderBlock()->id();
  auto condition_1 = loop_1_->FindConditionBlock()->id();
  auto continue_1 = loop_1_->GetContinueBlock()->id();
  auto continue_0 = loop_0_->GetContinueBlock()->id();
  auto condition_block_of_0 = loop_0_->FindConditionBlock();

  // Find the blocks whose branches need updating.
  auto first_block_of_1 = &*(++containing_function_->FindBlock(condition_1));
  auto last_block_of_1 = &*(--containing_function_->FindBlock(continue_1));
  auto last_block_of_0 = &*(--containing_function_->FindBlock(continue_0));

  // Update the branch for |last_block_of_loop_0| to go to |first_block_of_1|.
  last_block_of_0->ForEachSuccessorLabel(
      [first_block_of_1](uint32_t* succ) { *succ = first_block_of_1->id(); });

  // Update the branch for the |last_block_of_loop_1| to go to the continue
  // block of |loop_0_|.
  last_block_of_1->ForEachSuccessorLabel(
      [this](uint32_t* succ) { *succ = loop_0_->GetContinueBlock()->id(); });

  // Update merge block id in the header of |loop_0_| to the merge block of
  // |loop_1_|.
  loop_0_->GetHeaderBlock()->ForEachInst([this](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpLoopMerge) {
      inst->SetInOperand(0, {loop_1_->GetMergeBlock()->id()});
    }
  });

  // Update condition branch target in |loop_0_| to the merge block of
  // |loop_1_|.
  condition_block_of_0->ForEachInst([this](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpBranchConditional) {
      auto loop_0_merge_block_id = loop_0_->GetMergeBlock()->id();

      if (inst->GetSingleWordInOperand(1) == loop_0_merge_block_id) {
        inst->SetInOperand(1, {loop_1_->GetMergeBlock()->id()});
      } else {
        inst->SetInOperand(2, {loop_1_->GetMergeBlock()->id()});
      }
    }
  });

  // Move OpPhi instructions not corresponding to the induction variable from
  // the header of |loop_1_| to the header of |loop_0_|.
  std::vector<Instruction*> instructions_to_move{};
  for (auto& instruction : *loop_1_->GetHeaderBlock()) {
    if (instruction.opcode() == spv::Op::OpPhi &&
        &instruction != induction_1_) {
      instructions_to_move.push_back(&instruction);
    }
  }

  for (auto& it : instructions_to_move) {
    it->RemoveFromList();
    it->InsertBefore(induction_0_);
  }

  // Update the OpPhi parents to the correct blocks in |loop_0_|.
  loop_0_->GetHeaderBlock()->ForEachPhiInst([this](Instruction* i) {
    ReplacePhiParentWith(i, loop_1_->GetPreHeaderBlock()->id(),
                         loop_0_->GetPreHeaderBlock()->id());

    ReplacePhiParentWith(i, loop_1_->GetContinueBlock()->id(),
                         loop_0_->GetContinueBlock()->id());
  });

  // Update instruction to block mapping & DefUseManager.
  for (auto& phi_instruction : instructions_to_move) {
    context_->set_instr_block(phi_instruction, loop_0_->GetHeaderBlock());
    context_->get_def_use_mgr()->AnalyzeInstUse(phi_instruction);
  }

  // Replace the uses of the induction variable of |loop_1_| with that the
  // induction variable of |loop_0_|.
  context_->ReplaceAllUsesWith(induction_1_->result_id(),
                               induction_0_->result_id());

  // Replace LCSSA OpPhi in merge block of |loop_0_|.
  loop_0_->GetMergeBlock()->ForEachPhiInst([this](Instruction* instruction) {
    context_->ReplaceAllUsesWith(instruction->result_id(),
                                 instruction->GetSingleWordInOperand(0));
  });

  // Update LCSSA OpPhi in merge block of |loop_1_|.
  loop_1_->GetMergeBlock()->ForEachPhiInst(
      [condition_block_of_0](Instruction* instruction) {
        instruction->SetInOperand(1, {condition_block_of_0->id()});
      });

  // Move the continue block of |loop_0_| after the last block of |loop_1_|.
  containing_function_->MoveBasicBlockToAfter(continue_0, last_block_of_1);

  // Gather all instructions to be killed from |loop_1_| (induction variable
  // initialisation, header, condition and continue blocks).
  std::vector<Instruction*> instr_to_delete{};
  AddInstructionsInBlock(&instr_to_delete, loop_1_->GetPreHeaderBlock());
  AddInstructionsInBlock(&instr_to_delete, loop_1_->GetHeaderBlock());
  AddInstructionsInBlock(&instr_to_delete, loop_1_->FindConditionBlock());
  AddInstructionsInBlock(&instr_to_delete, loop_1_->GetContinueBlock());

  // There was an additional empty block between the loops, kill that too.
  if (loop_0_->GetMergeBlock() != loop_1_->GetPreHeaderBlock()) {
    AddInstructionsInBlock(&instr_to_delete, loop_0_->GetMergeBlock());
  }

  // Update the CFG, so it wouldn't need invalidating.
  auto cfg = context_->cfg();

  cfg->ForgetBlock(loop_1_->GetPreHeaderBlock());
  cfg->ForgetBlock(loop_1_->GetHeaderBlock());
  cfg->ForgetBlock(loop_1_->FindConditionBlock());
  cfg->ForgetBlock(loop_1_->GetContinueBlock());

  if (loop_0_->GetMergeBlock() != loop_1_->GetPreHeaderBlock()) {
    cfg->ForgetBlock(loop_0_->GetMergeBlock());
  }

  cfg->RemoveEdge(last_block_of_0->id(), loop_0_->GetContinueBlock()->id());
  cfg->AddEdge(last_block_of_0->id(), first_block_of_1->id());

  cfg->AddEdge(last_block_of_1->id(), loop_0_->GetContinueBlock()->id());

  cfg->AddEdge(loop_0_->GetContinueBlock()->id(),
               loop_1_->GetHeaderBlock()->id());

  cfg->AddEdge(condition_block_of_0->id(), loop_1_->GetMergeBlock()->id());

  // Update DefUseManager.
  auto def_use_mgr = context_->get_def_use_mgr();

  // Uses of labels that are in updated branches need analysing.
  def_use_mgr->AnalyzeInstUse(last_block_of_0->terminator());
  def_use_mgr->AnalyzeInstUse(last_block_of_1->terminator());
  def_use_mgr->AnalyzeInstUse(loop_0_->GetHeaderBlock()->GetLoopMergeInst());
  def_use_mgr->AnalyzeInstUse(condition_block_of_0->terminator());

  // Update the LoopDescriptor, so it wouldn't need invalidating.
  auto ld = context_->GetLoopDescriptor(containing_function_);

  // Create a copy, so the iterator wouldn't be invalidated.
  std::vector<Loop*> loops_to_add_remove{};
  for (auto child_loop : *loop_1_) {
    loops_to_add_remove.push_back(child_loop);
  }

  for (auto child_loop : loops_to_add_remove) {
    loop_1_->RemoveChildLoop(child_loop);
    loop_0_->AddNestedLoop(child_loop);
  }

  auto loop_1_blocks = loop_1_->GetBlocks();

  for (auto block : loop_1_blocks) {
    loop_1_->RemoveBasicBlock(block);
    if (block != header_1 && block != condition_1 && block != continue_1) {
      loop_0_->AddBasicBlock(block);
      if ((*ld)[block] == loop_1_) {
        ld->SetBasicBlockToLoop(block, loop_0_);
      }
    }

    if ((*ld)[block] == loop_1_) {
      ld->ForgetBasicBlock(block);
    }
  }

  loop_1_->RemoveBasicBlock(loop_1_->GetPreHeaderBlock()->id());
  ld->ForgetBasicBlock(loop_1_->GetPreHeaderBlock()->id());

  if (loop_0_->GetMergeBlock() != loop_1_->GetPreHeaderBlock()) {
    loop_0_->RemoveBasicBlock(loop_0_->GetMergeBlock()->id());
    ld->ForgetBasicBlock(loop_0_->GetMergeBlock()->id());
  }

  loop_0_->SetMergeBlock(loop_1_->GetMergeBlock());

  loop_1_->ClearBlocks();

  ld->RemoveLoop(loop_1_);

  // Kill unnecessary instructions and remove all empty blocks.
  for (auto inst : instr_to_delete) {
    context_->KillInst(inst);
  }

  containing_function_->RemoveEmptyBlocks();

  // Invalidate analyses.
  context_->InvalidateAnalysesExceptFor(
      IRContext::Analysis::kAnalysisInstrToBlockMapping |
      IRContext::Analysis::kAnalysisLoopAnalysis |
      IRContext::Analysis::kAnalysisDefUse | IRContext::Analysis::kAnalysisCFG);
}

}  // namespace opt
}  // namespace spvtools
