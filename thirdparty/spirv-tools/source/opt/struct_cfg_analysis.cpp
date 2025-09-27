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

#include "source/opt/struct_cfg_analysis.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kMergeNodeIndex = 0;
constexpr uint32_t kContinueNodeIndex = 1;
}  // namespace

StructuredCFGAnalysis::StructuredCFGAnalysis(IRContext* ctx) : context_(ctx) {
  // If this is not a shader, there are no merge instructions, and not
  // structured CFG to analyze.
  if (!context_->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    return;
  }

  for (auto& func : *context_->module()) {
    AddBlocksInFunction(&func);
  }
}

void StructuredCFGAnalysis::AddBlocksInFunction(Function* func) {
  if (func->begin() == func->end()) return;

  std::list<BasicBlock*> order;
  context_->cfg()->ComputeStructuredOrder(func, &*func->begin(), &order);

  struct TraversalInfo {
    ConstructInfo cinfo;
    uint32_t merge_node;
    uint32_t continue_node;
  };

  // Set up a stack to keep track of currently active constructs.
  std::vector<TraversalInfo> state;
  state.emplace_back();
  state[0].cinfo.containing_construct = 0;
  state[0].cinfo.containing_loop = 0;
  state[0].cinfo.containing_switch = 0;
  state[0].cinfo.in_continue = false;
  state[0].merge_node = 0;
  state[0].continue_node = 0;

  for (BasicBlock* block : order) {
    if (context_->cfg()->IsPseudoEntryBlock(block) ||
        context_->cfg()->IsPseudoExitBlock(block)) {
      continue;
    }

    if (block->id() == state.back().merge_node) {
      state.pop_back();
    }

    // This works because the structured order is designed to keep the blocks in
    // the continue construct between the continue header and the merge node.
    if (block->id() == state.back().continue_node) {
      state.back().cinfo.in_continue = true;
    }

    bb_to_construct_.emplace(std::make_pair(block->id(), state.back().cinfo));

    if (Instruction* merge_inst = block->GetMergeInst()) {
      TraversalInfo new_state;
      new_state.merge_node =
          merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
      new_state.cinfo.containing_construct = block->id();

      if (merge_inst->opcode() == spv::Op::OpLoopMerge) {
        new_state.cinfo.containing_loop = block->id();
        new_state.cinfo.containing_switch = 0;
        new_state.continue_node =
            merge_inst->GetSingleWordInOperand(kContinueNodeIndex);
        if (block->id() == new_state.continue_node) {
          new_state.cinfo.in_continue = true;
          bb_to_construct_[block->id()].in_continue = true;
        } else {
          new_state.cinfo.in_continue = false;
        }
      } else {
        new_state.cinfo.containing_loop = state.back().cinfo.containing_loop;
        new_state.cinfo.in_continue = state.back().cinfo.in_continue;
        new_state.continue_node = state.back().continue_node;

        if (merge_inst->NextNode()->opcode() == spv::Op::OpSwitch) {
          new_state.cinfo.containing_switch = block->id();
        } else {
          new_state.cinfo.containing_switch =
              state.back().cinfo.containing_switch;
        }
      }

      state.emplace_back(new_state);
      merge_blocks_.Set(new_state.merge_node);
    }
  }
}

uint32_t StructuredCFGAnalysis::ContainingConstruct(Instruction* inst) {
  uint32_t bb = context_->get_instr_block(inst)->id();
  return ContainingConstruct(bb);
}

uint32_t StructuredCFGAnalysis::MergeBlock(uint32_t bb_id) {
  uint32_t header_id = ContainingConstruct(bb_id);
  if (header_id == 0) {
    return 0;
  }

  BasicBlock* header = context_->cfg()->block(header_id);
  Instruction* merge_inst = header->GetMergeInst();
  return merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
}

uint32_t StructuredCFGAnalysis::NestingDepth(uint32_t bb_id) {
  uint32_t result = 0;

  // Find the merge block of the current merge construct as long as the block is
  // inside a merge construct, exiting one for each iteration.
  for (uint32_t merge_block_id = MergeBlock(bb_id); merge_block_id != 0;
       merge_block_id = MergeBlock(merge_block_id)) {
    result++;
  }

  return result;
}

uint32_t StructuredCFGAnalysis::LoopMergeBlock(uint32_t bb_id) {
  uint32_t header_id = ContainingLoop(bb_id);
  if (header_id == 0) {
    return 0;
  }

  BasicBlock* header = context_->cfg()->block(header_id);
  Instruction* merge_inst = header->GetMergeInst();
  return merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
}

uint32_t StructuredCFGAnalysis::LoopContinueBlock(uint32_t bb_id) {
  uint32_t header_id = ContainingLoop(bb_id);
  if (header_id == 0) {
    return 0;
  }

  BasicBlock* header = context_->cfg()->block(header_id);
  Instruction* merge_inst = header->GetMergeInst();
  return merge_inst->GetSingleWordInOperand(kContinueNodeIndex);
}

uint32_t StructuredCFGAnalysis::LoopNestingDepth(uint32_t bb_id) {
  uint32_t result = 0;

  // Find the merge block of the current loop as long as the block is inside a
  // loop, exiting a loop for each iteration.
  for (uint32_t merge_block_id = LoopMergeBlock(bb_id); merge_block_id != 0;
       merge_block_id = LoopMergeBlock(merge_block_id)) {
    result++;
  }

  return result;
}

uint32_t StructuredCFGAnalysis::SwitchMergeBlock(uint32_t bb_id) {
  uint32_t header_id = ContainingSwitch(bb_id);
  if (header_id == 0) {
    return 0;
  }

  BasicBlock* header = context_->cfg()->block(header_id);
  Instruction* merge_inst = header->GetMergeInst();
  return merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
}

bool StructuredCFGAnalysis::IsContinueBlock(uint32_t bb_id) {
  assert(bb_id != 0);
  return LoopContinueBlock(bb_id) == bb_id;
}

bool StructuredCFGAnalysis::IsInContainingLoopsContinueConstruct(
    uint32_t bb_id) {
  auto it = bb_to_construct_.find(bb_id);
  if (it == bb_to_construct_.end()) {
    return false;
  }
  return it->second.in_continue;
}

bool StructuredCFGAnalysis::IsInContinueConstruct(uint32_t bb_id) {
  while (bb_id != 0) {
    if (IsInContainingLoopsContinueConstruct(bb_id)) {
      return true;
    }
    bb_id = ContainingLoop(bb_id);
  }
  return false;
}

bool StructuredCFGAnalysis::IsMergeBlock(uint32_t bb_id) {
  return merge_blocks_.Get(bb_id);
}

std::unordered_set<uint32_t>
StructuredCFGAnalysis::FindFuncsCalledFromContinue() {
  std::unordered_set<uint32_t> called_from_continue;
  std::queue<uint32_t> funcs_to_process;

  // First collect the functions that are called directly from a continue
  // construct.
  for (Function& func : *context_->module()) {
    for (auto& bb : func) {
      if (IsInContainingLoopsContinueConstruct(bb.id())) {
        for (const Instruction& inst : bb) {
          if (inst.opcode() == spv::Op::OpFunctionCall) {
            funcs_to_process.push(inst.GetSingleWordInOperand(0));
          }
        }
      }
    }
  }

  // Now collect all of the functions that are indirectly called as well.
  while (!funcs_to_process.empty()) {
    uint32_t func_id = funcs_to_process.front();
    funcs_to_process.pop();
    Function* func = context_->GetFunction(func_id);
    if (called_from_continue.insert(func_id).second) {
      context_->AddCalls(func, &funcs_to_process);
    }
  }
  return called_from_continue;
}

}  // namespace opt
}  // namespace spvtools
