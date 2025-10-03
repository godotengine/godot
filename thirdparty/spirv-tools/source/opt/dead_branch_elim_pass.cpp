// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
// Copyright (c) 2018 Google Inc.
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

#include "source/opt/dead_branch_elim_pass.h"

#include <list>
#include <memory>
#include <vector>

#include "source/cfa.h"
#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"
#include "source/opt/struct_cfg_analysis.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kBranchCondTrueLabIdInIdx = 1;
constexpr uint32_t kBranchCondFalseLabIdInIdx = 2;
}  // namespace

bool DeadBranchElimPass::GetConstCondition(uint32_t condId, bool* condVal) {
  bool condIsConst;
  Instruction* cInst = get_def_use_mgr()->GetDef(condId);
  switch (cInst->opcode()) {
    case spv::Op::OpConstantNull:
    case spv::Op::OpConstantFalse: {
      *condVal = false;
      condIsConst = true;
    } break;
    case spv::Op::OpConstantTrue: {
      *condVal = true;
      condIsConst = true;
    } break;
    case spv::Op::OpLogicalNot: {
      bool negVal;
      condIsConst =
          GetConstCondition(cInst->GetSingleWordInOperand(0), &negVal);
      if (condIsConst) *condVal = !negVal;
    } break;
    default: { condIsConst = false; } break;
  }
  return condIsConst;
}

bool DeadBranchElimPass::GetConstInteger(uint32_t selId, uint32_t* selVal) {
  Instruction* sInst = get_def_use_mgr()->GetDef(selId);
  uint32_t typeId = sInst->type_id();
  Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  if (!typeInst || (typeInst->opcode() != spv::Op::OpTypeInt)) return false;
  // TODO(greg-lunarg): Support non-32 bit ints
  if (typeInst->GetSingleWordInOperand(0) != 32) return false;
  if (sInst->opcode() == spv::Op::OpConstant) {
    *selVal = sInst->GetSingleWordInOperand(0);
    return true;
  } else if (sInst->opcode() == spv::Op::OpConstantNull) {
    *selVal = 0;
    return true;
  }
  return false;
}

void DeadBranchElimPass::AddBranch(uint32_t labelId, BasicBlock* bp) {
  assert(get_def_use_mgr()->GetDef(labelId) != nullptr);
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), spv::Op::OpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  context()->AnalyzeDefUse(&*newBranch);
  context()->set_instr_block(&*newBranch, bp);
  bp->AddInstruction(std::move(newBranch));
}

BasicBlock* DeadBranchElimPass::GetParentBlock(uint32_t id) {
  return context()->get_instr_block(get_def_use_mgr()->GetDef(id));
}

bool DeadBranchElimPass::MarkLiveBlocks(
    Function* func, std::unordered_set<BasicBlock*>* live_blocks) {
  std::vector<std::pair<BasicBlock*, uint32_t>> conditions_to_simplify;
  std::unordered_set<BasicBlock*> blocks_with_backedge;
  std::vector<BasicBlock*> stack;
  stack.push_back(&*func->begin());
  bool modified = false;
  while (!stack.empty()) {
    BasicBlock* block = stack.back();
    stack.pop_back();

    // Live blocks doubles as visited set.
    if (!live_blocks->insert(block).second) continue;

    uint32_t cont_id = block->ContinueBlockIdIfAny();
    if (cont_id != 0) {
      AddBlocksWithBackEdge(cont_id, block->id(), block->MergeBlockIdIfAny(),
                            &blocks_with_backedge);
    }

    Instruction* terminator = block->terminator();
    uint32_t live_lab_id = 0;
    // Check if the terminator has a single valid successor.
    if (terminator->opcode() == spv::Op::OpBranchConditional) {
      bool condVal;
      if (GetConstCondition(terminator->GetSingleWordInOperand(0u), &condVal)) {
        live_lab_id = terminator->GetSingleWordInOperand(
            condVal ? kBranchCondTrueLabIdInIdx : kBranchCondFalseLabIdInIdx);
      }
    } else if (terminator->opcode() == spv::Op::OpSwitch) {
      uint32_t sel_val;
      if (GetConstInteger(terminator->GetSingleWordInOperand(0u), &sel_val)) {
        // Search switch operands for selector value, set live_lab_id to
        // corresponding label, use default if not found.
        uint32_t icnt = 0;
        uint32_t case_val;
        terminator->WhileEachInOperand(
            [&icnt, &case_val, &sel_val, &live_lab_id](const uint32_t* idp) {
              if (icnt == 1) {
                // Start with default label.
                live_lab_id = *idp;
              } else if (icnt > 1) {
                if (icnt % 2 == 0) {
                  case_val = *idp;
                } else {
                  if (case_val == sel_val) {
                    live_lab_id = *idp;
                    return false;
                  }
                }
              }
              ++icnt;
              return true;
            });
      }
    }

    // Don't simplify back edges unless it becomes a branch to the header. Every
    // loop must have exactly one back edge to the loop header, so we cannot
    // remove it.
    bool simplify = false;
    if (live_lab_id != 0) {
      if (!blocks_with_backedge.count(block)) {
        // This is not a back edge.
        simplify = true;
      } else {
        const auto& struct_cfg_analysis = context()->GetStructuredCFGAnalysis();
        uint32_t header_id = struct_cfg_analysis->ContainingLoop(block->id());
        if (live_lab_id == header_id) {
          // The new branch will be a branch to the header.
          simplify = true;
        }
      }
    }

    if (simplify) {
      conditions_to_simplify.push_back({block, live_lab_id});
      stack.push_back(GetParentBlock(live_lab_id));
    } else {
      // All successors are live.
      const auto* const_block = block;
      const_block->ForEachSuccessorLabel([&stack, this](const uint32_t label) {
        stack.push_back(GetParentBlock(label));
      });
    }
  }

  // Traverse |conditions_to_simplify| in reverse order.  This is done so that
  // we simplify nested constructs before simplifying the constructs that
  // contain them.
  for (auto b = conditions_to_simplify.rbegin();
       b != conditions_to_simplify.rend(); ++b) {
    modified |= SimplifyBranch(b->first, b->second);
  }

  return modified;
}

bool DeadBranchElimPass::SimplifyBranch(BasicBlock* block,
                                        uint32_t live_lab_id) {
  Instruction* merge_inst = block->GetMergeInst();
  Instruction* terminator = block->terminator();
  if (merge_inst && merge_inst->opcode() == spv::Op::OpSelectionMerge) {
    if (merge_inst->NextNode()->opcode() == spv::Op::OpSwitch &&
        SwitchHasNestedBreak(block->id())) {
      if (terminator->NumInOperands() == 2) {
        // We cannot remove the branch, and it already has a single case, so no
        // work to do.
        return false;
      }
      // We have to keep the switch because it has a nest break, so we
      // remove all cases except for the live one.
      Instruction::OperandList new_operands;
      new_operands.push_back(terminator->GetInOperand(0));
      new_operands.push_back({SPV_OPERAND_TYPE_ID, {live_lab_id}});
      terminator->SetInOperands(std::move(new_operands));
      context()->UpdateDefUse(terminator);
    } else {
      // Check if the merge instruction is still needed because of a
      // non-nested break from the construct.  Move the merge instruction if
      // it is still needed.
      StructuredCFGAnalysis* cfg_analysis =
          context()->GetStructuredCFGAnalysis();
      Instruction* first_break = FindFirstExitFromSelectionMerge(
          live_lab_id, merge_inst->GetSingleWordInOperand(0),
          cfg_analysis->LoopMergeBlock(live_lab_id),
          cfg_analysis->LoopContinueBlock(live_lab_id),
          cfg_analysis->SwitchMergeBlock(live_lab_id));

      AddBranch(live_lab_id, block);
      context()->KillInst(terminator);
      if (first_break == nullptr) {
        context()->KillInst(merge_inst);
      } else {
        merge_inst->RemoveFromList();
        first_break->InsertBefore(std::unique_ptr<Instruction>(merge_inst));
        context()->set_instr_block(merge_inst,
                                   context()->get_instr_block(first_break));
      }
    }
  } else {
    AddBranch(live_lab_id, block);
    context()->KillInst(terminator);
  }
  return true;
}

void DeadBranchElimPass::MarkUnreachableStructuredTargets(
    const std::unordered_set<BasicBlock*>& live_blocks,
    std::unordered_set<BasicBlock*>* unreachable_merges,
    std::unordered_map<BasicBlock*, BasicBlock*>* unreachable_continues) {
  for (auto block : live_blocks) {
    if (auto merge_id = block->MergeBlockIdIfAny()) {
      BasicBlock* merge_block = GetParentBlock(merge_id);
      if (!live_blocks.count(merge_block)) {
        unreachable_merges->insert(merge_block);
      }
      if (auto cont_id = block->ContinueBlockIdIfAny()) {
        BasicBlock* cont_block = GetParentBlock(cont_id);
        if (!live_blocks.count(cont_block)) {
          (*unreachable_continues)[cont_block] = block;
        }
      }
    }
  }
}

bool DeadBranchElimPass::FixPhiNodesInLiveBlocks(
    Function* func, const std::unordered_set<BasicBlock*>& live_blocks,
    const std::unordered_map<BasicBlock*, BasicBlock*>& unreachable_continues) {
  bool modified = false;
  for (auto& block : *func) {
    if (live_blocks.count(&block)) {
      for (auto iter = block.begin(); iter != block.end();) {
        if (iter->opcode() != spv::Op::OpPhi) {
          break;
        }

        bool changed = false;
        bool backedge_added = false;
        Instruction* inst = &*iter;
        std::vector<Operand> operands;
        // Build a complete set of operands (not just input operands). Start
        // with type and result id operands.
        operands.push_back(inst->GetOperand(0u));
        operands.push_back(inst->GetOperand(1u));
        // Iterate through the incoming labels and determine which to keep
        // and/or modify.  If there in an unreachable continue block, there will
        // be an edge from that block to the header.  We need to keep it to
        // maintain the structured control flow.  If the header has more that 2
        // incoming edges, then the OpPhi must have an entry for that edge.
        // However, if there is only one other incoming edge, the OpPhi can be
        // eliminated.
        for (uint32_t i = 1; i < inst->NumInOperands(); i += 2) {
          BasicBlock* inc = GetParentBlock(inst->GetSingleWordInOperand(i));
          auto cont_iter = unreachable_continues.find(inc);
          if (cont_iter != unreachable_continues.end() &&
              cont_iter->second == &block && inst->NumInOperands() > 4) {
            if (get_def_use_mgr()
                    ->GetDef(inst->GetSingleWordInOperand(i - 1))
                    ->opcode() == spv::Op::OpUndef) {
              // Already undef incoming value, no change necessary.
              operands.push_back(inst->GetInOperand(i - 1));
              operands.push_back(inst->GetInOperand(i));
              backedge_added = true;
            } else {
              // Replace incoming value with undef if this phi exists in the
              // loop header. Otherwise, this edge is not live since the
              // unreachable continue block will be replaced with an
              // unconditional branch to the header only.
              operands.emplace_back(
                  SPV_OPERAND_TYPE_ID,
                  std::initializer_list<uint32_t>{Type2Undef(inst->type_id())});
              operands.push_back(inst->GetInOperand(i));
              changed = true;
              backedge_added = true;
            }
          } else if (live_blocks.count(inc) && inc->IsSuccessor(&block)) {
            // Keep live incoming edge.
            operands.push_back(inst->GetInOperand(i - 1));
            operands.push_back(inst->GetInOperand(i));
          } else {
            // Remove incoming edge.
            changed = true;
          }
        }

        if (changed) {
          modified = true;
          uint32_t continue_id = block.ContinueBlockIdIfAny();
          if (!backedge_added && continue_id != 0 &&
              unreachable_continues.count(GetParentBlock(continue_id)) &&
              operands.size() > 4) {
            // Changed the backedge to branch from the continue block instead
            // of a successor of the continue block. Add an entry to the phi to
            // provide an undef for the continue block. Since the successor of
            // the continue must also be unreachable (dominated by the continue
            // block), any entry for the original backedge has been removed
            // from the phi operands.
            operands.emplace_back(
                SPV_OPERAND_TYPE_ID,
                std::initializer_list<uint32_t>{Type2Undef(inst->type_id())});
            operands.emplace_back(SPV_OPERAND_TYPE_ID,
                                  std::initializer_list<uint32_t>{continue_id});
          }

          // Either replace the phi with a single value or rebuild the phi out
          // of |operands|.
          //
          // We always have type and result id operands. So this phi has a
          // single source if there are two more operands beyond those.
          if (operands.size() == 4) {
            // First input data operands is at index 2.
            uint32_t replId = operands[2u].words[0];
            context()->KillNamesAndDecorates(inst->result_id());
            context()->ReplaceAllUsesWith(inst->result_id(), replId);
            iter = context()->KillInst(&*inst);
          } else {
            // We've rewritten the operands, so first instruct the def/use
            // manager to forget uses in the phi before we replace them. After
            // replacing operands update the def/use manager by re-analyzing
            // the used ids in this phi.
            get_def_use_mgr()->EraseUseRecordsOfOperandIds(inst);
            inst->ReplaceOperands(operands);
            get_def_use_mgr()->AnalyzeInstUse(inst);
            ++iter;
          }
        } else {
          ++iter;
        }
      }
    }
  }

  return modified;
}

bool DeadBranchElimPass::EraseDeadBlocks(
    Function* func, const std::unordered_set<BasicBlock*>& live_blocks,
    const std::unordered_set<BasicBlock*>& unreachable_merges,
    const std::unordered_map<BasicBlock*, BasicBlock*>& unreachable_continues) {
  bool modified = false;
  for (auto ebi = func->begin(); ebi != func->end();) {
    if (unreachable_continues.count(&*ebi)) {
      uint32_t cont_id = unreachable_continues.find(&*ebi)->second->id();
      if (ebi->begin() != ebi->tail() ||
          ebi->terminator()->opcode() != spv::Op::OpBranch ||
          ebi->terminator()->GetSingleWordInOperand(0u) != cont_id) {
        // Make unreachable, but leave the label.
        KillAllInsts(&*ebi, false);
        // Add unconditional branch to header.
        assert(unreachable_continues.count(&*ebi));
        ebi->AddInstruction(MakeUnique<Instruction>(
            context(), spv::Op::OpBranch, 0, 0,
            std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {cont_id}}}));
        get_def_use_mgr()->AnalyzeInstUse(&*ebi->tail());
        context()->set_instr_block(&*ebi->tail(), &*ebi);
        modified = true;
      }
      ++ebi;
    } else if (unreachable_merges.count(&*ebi)) {
      if (ebi->begin() != ebi->tail() ||
          ebi->terminator()->opcode() != spv::Op::OpUnreachable) {
        // Make unreachable, but leave the label.
        KillAllInsts(&*ebi, false);
        // Add unreachable terminator.
        ebi->AddInstruction(
            MakeUnique<Instruction>(context(), spv::Op::OpUnreachable, 0, 0,
                                    std::initializer_list<Operand>{}));
        context()->AnalyzeUses(ebi->terminator());
        context()->set_instr_block(ebi->terminator(), &*ebi);
        modified = true;
      }
      ++ebi;
    } else if (!live_blocks.count(&*ebi)) {
      // Kill this block.
      KillAllInsts(&*ebi);
      ebi = ebi.Erase();
      modified = true;
    } else {
      ++ebi;
    }
  }

  return modified;
}

bool DeadBranchElimPass::EliminateDeadBranches(Function* func) {
  if (func->IsDeclaration()) {
    return false;
  }

  bool modified = false;
  std::unordered_set<BasicBlock*> live_blocks;
  modified |= MarkLiveBlocks(func, &live_blocks);

  std::unordered_set<BasicBlock*> unreachable_merges;
  std::unordered_map<BasicBlock*, BasicBlock*> unreachable_continues;
  MarkUnreachableStructuredTargets(live_blocks, &unreachable_merges,
                                   &unreachable_continues);
  modified |= FixPhiNodesInLiveBlocks(func, live_blocks, unreachable_continues);
  modified |= EraseDeadBlocks(func, live_blocks, unreachable_merges,
                              unreachable_continues);

  return modified;
}

void DeadBranchElimPass::FixBlockOrder() {
  context()->BuildInvalidAnalyses(IRContext::kAnalysisCFG |
                                  IRContext::kAnalysisDominatorAnalysis);
  // Reorders blocks according to DFS of dominator tree.
  ProcessFunction reorder_dominators = [this](Function* function) {
    DominatorAnalysis* dominators = context()->GetDominatorAnalysis(function);
    std::vector<BasicBlock*> blocks;
    for (auto iter = dominators->GetDomTree().begin();
         iter != dominators->GetDomTree().end(); ++iter) {
      if (iter->id() != 0) {
        blocks.push_back(iter->bb_);
      }
    }
    for (uint32_t i = 1; i < blocks.size(); ++i) {
      function->MoveBasicBlockToAfter(blocks[i]->id(), blocks[i - 1]);
    }
    return true;
  };

  // Reorders blocks according to structured order.
  ProcessFunction reorder_structured = [](Function* function) {
    function->ReorderBasicBlocksInStructuredOrder();
    return true;
  };

  // Structured order is more intuitive so use it where possible.
  if (context()->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    context()->ProcessReachableCallTree(reorder_structured);
  } else {
    context()->ProcessReachableCallTree(reorder_dominators);
  }
}

Pass::Status DeadBranchElimPass::Process() {
  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == spv::Op::OpGroupDecorate)
      return Status::SuccessWithoutChange;
  // Process all entry point functions
  ProcessFunction pfn = [this](Function* fp) {
    return EliminateDeadBranches(fp);
  };
  bool modified = context()->ProcessReachableCallTree(pfn);
  if (modified) FixBlockOrder();
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Instruction* DeadBranchElimPass::FindFirstExitFromSelectionMerge(
    uint32_t start_block_id, uint32_t merge_block_id, uint32_t loop_merge_id,
    uint32_t loop_continue_id, uint32_t switch_merge_id) {
  // To find the "first" exit, we follow branches looking for a conditional
  // branch that is not in a nested construct and is not the header of a new
  // construct.  We follow the control flow from |start_block_id| to find the
  // first one.

  while (start_block_id != merge_block_id && start_block_id != loop_merge_id &&
         start_block_id != loop_continue_id) {
    BasicBlock* start_block = context()->get_instr_block(start_block_id);
    Instruction* branch = start_block->terminator();
    uint32_t next_block_id = 0;
    switch (branch->opcode()) {
      case spv::Op::OpBranchConditional:
        next_block_id = start_block->MergeBlockIdIfAny();
        if (next_block_id == 0) {
          // If a possible target is the |loop_merge_id| or |loop_continue_id|,
          // which are not the current merge node, then we continue the search
          // with the other target.
          for (uint32_t i = 1; i < 3; i++) {
            if (branch->GetSingleWordInOperand(i) == loop_merge_id &&
                loop_merge_id != merge_block_id) {
              next_block_id = branch->GetSingleWordInOperand(3 - i);
              break;
            }
            if (branch->GetSingleWordInOperand(i) == loop_continue_id &&
                loop_continue_id != merge_block_id) {
              next_block_id = branch->GetSingleWordInOperand(3 - i);
              break;
            }
            if (branch->GetSingleWordInOperand(i) == switch_merge_id &&
                switch_merge_id != merge_block_id) {
              next_block_id = branch->GetSingleWordInOperand(3 - i);
              break;
            }
          }

          if (next_block_id == 0) {
            return branch;
          }
        }
        break;
      case spv::Op::OpSwitch:
        next_block_id = start_block->MergeBlockIdIfAny();
        if (next_block_id == 0) {
          // A switch with no merge instructions can have at most 5 targets:
          //   a. |merge_block_id|
          //   b. |loop_merge_id|
          //   c. |loop_continue_id|
          //   d. |switch_merge_id|
          //   e. 1 block inside the current region.
          //
          // Note that because this is a switch, |merge_block_id| must equal
          // |switch_merge_id|.
          //
          // This leads to a number of cases of what to do.
          //
          // 1. Does not jump to a block inside of the current construct.  In
          // this case, there is not conditional break, so we should return
          // |nullptr|.
          //
          // 2. Jumps to |merge_block_id| and a block inside the current
          // construct.  In this case, this branch conditionally break to the
          // end of the current construct, so return the current branch.
          //
          // 3.  Otherwise, this branch may break, but not to the current merge
          // block.  So we continue with the block that is inside the loop.
          bool found_break = false;
          for (uint32_t i = 1; i < branch->NumInOperands(); i += 2) {
            uint32_t target = branch->GetSingleWordInOperand(i);
            if (target == merge_block_id) {
              found_break = true;
            } else if (target != loop_merge_id && target != loop_continue_id) {
              next_block_id = branch->GetSingleWordInOperand(i);
            }
          }

          if (next_block_id == 0) {
            // Case 1.
            return nullptr;
          }

          if (found_break) {
            // Case 2.
            return branch;
          }

          // The fall through is case 3.
        }
        break;
      case spv::Op::OpBranch:
        // Need to check if this is the header of a loop nested in the
        // selection construct.
        next_block_id = start_block->MergeBlockIdIfAny();
        if (next_block_id == 0) {
          next_block_id = branch->GetSingleWordInOperand(0);
        }
        break;
      default:
        return nullptr;
    }
    start_block_id = next_block_id;
  }
  return nullptr;
}

void DeadBranchElimPass::AddBlocksWithBackEdge(
    uint32_t cont_id, uint32_t header_id, uint32_t merge_id,
    std::unordered_set<BasicBlock*>* blocks_with_back_edges) {
  std::unordered_set<uint32_t> visited;
  visited.insert(cont_id);
  visited.insert(header_id);
  visited.insert(merge_id);

  std::vector<uint32_t> work_list;
  work_list.push_back(cont_id);

  while (!work_list.empty()) {
    uint32_t bb_id = work_list.back();
    work_list.pop_back();

    BasicBlock* bb = context()->get_instr_block(bb_id);

    bool has_back_edge = false;
    bb->ForEachSuccessorLabel([header_id, &visited, &work_list,
                               &has_back_edge](uint32_t* succ_label_id) {
      if (visited.insert(*succ_label_id).second) {
        work_list.push_back(*succ_label_id);
      }
      if (*succ_label_id == header_id) {
        has_back_edge = true;
      }
    });

    if (has_back_edge) {
      blocks_with_back_edges->insert(bb);
    }
  }
}

bool DeadBranchElimPass::SwitchHasNestedBreak(uint32_t switch_header_id) {
  std::vector<BasicBlock*> block_in_construct;
  BasicBlock* start_block = context()->get_instr_block(switch_header_id);
  uint32_t merge_block_id = start_block->MergeBlockIdIfAny();

  StructuredCFGAnalysis* cfg_analysis = context()->GetStructuredCFGAnalysis();
  return !get_def_use_mgr()->WhileEachUser(
      merge_block_id,
      [this, cfg_analysis, switch_header_id](Instruction* inst) {
        if (!inst->IsBranch()) {
          return true;
        }

        BasicBlock* bb = context()->get_instr_block(inst);
        if (bb->id() == switch_header_id) {
          return true;
        }
        return (cfg_analysis->ContainingConstruct(inst) == switch_header_id &&
                bb->GetMergeInst() == nullptr);
      });
}

}  // namespace opt
}  // namespace spvtools
