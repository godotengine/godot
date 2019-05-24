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

namespace {
const uint32_t kMergeNodeIndex = 0;
const uint32_t kContinueNodeIndex = 1;
}  // namespace

namespace spvtools {
namespace opt {

StructuredCFGAnalysis::StructuredCFGAnalysis(IRContext* ctx) : context_(ctx) {
  // If this is not a shader, there are no merge instructions, and not
  // structured CFG to analyze.
  if (!context_->get_feature_mgr()->HasCapability(SpvCapabilityShader)) {
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
  };

  // Set up a stack to keep track of currently active constructs.
  std::vector<TraversalInfo> state;
  state.emplace_back();
  state[0].cinfo.containing_construct = 0;
  state[0].cinfo.containing_loop = 0;
  state[0].merge_node = 0;

  for (BasicBlock* block : order) {
    if (context_->cfg()->IsPseudoEntryBlock(block) ||
        context_->cfg()->IsPseudoExitBlock(block)) {
      continue;
    }

    if (block->id() == state.back().merge_node) {
      state.pop_back();
    }

    bb_to_construct_.emplace(std::make_pair(block->id(), state.back().cinfo));

    if (Instruction* merge_inst = block->GetMergeInst()) {
      TraversalInfo new_state;
      new_state.merge_node =
          merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
      new_state.cinfo.containing_construct = block->id();

      if (merge_inst->opcode() == SpvOpLoopMerge) {
        new_state.cinfo.containing_loop = block->id();
      } else {
        new_state.cinfo.containing_loop = state.back().cinfo.containing_loop;
      }

      state.emplace_back(new_state);
      merge_blocks_.Set(new_state.merge_node);
    }
  }
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

bool StructuredCFGAnalysis::IsContinueBlock(uint32_t bb_id) {
  assert(bb_id != 0);
  return LoopContinueBlock(bb_id) == bb_id;
}

bool StructuredCFGAnalysis::IsMergeBlock(uint32_t bb_id) {
  return merge_blocks_.Get(bb_id);
}

}  // namespace opt
}  // namespace spvtools
