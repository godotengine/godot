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

#include "source/opt/licm_pass.h"

#include <queue>
#include <utility>

#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

Pass::Status LICMPass::Process() { return ProcessIRContext(); }

Pass::Status LICMPass::ProcessIRContext() {
  Status status = Status::SuccessWithoutChange;
  Module* module = get_module();

  // Process each function in the module
  for (auto func = module->begin();
       func != module->end() && status != Status::Failure; ++func) {
    status = CombineStatus(status, ProcessFunction(&*func));
  }
  return status;
}

Pass::Status LICMPass::ProcessFunction(Function* f) {
  Status status = Status::SuccessWithoutChange;
  LoopDescriptor* loop_descriptor = context()->GetLoopDescriptor(f);

  // Process each loop in the function
  for (auto it = loop_descriptor->begin();
       it != loop_descriptor->end() && status != Status::Failure; ++it) {
    Loop& loop = *it;
    // Ignore nested loops, as we will process them in order in ProcessLoop
    if (loop.IsNested()) {
      continue;
    }
    status = CombineStatus(status, ProcessLoop(&loop, f));
  }
  return status;
}

Pass::Status LICMPass::ProcessLoop(Loop* loop, Function* f) {
  Status status = Status::SuccessWithoutChange;

  // Process all nested loops first
  for (auto nl = loop->begin(); nl != loop->end() && status != Status::Failure;
       ++nl) {
    Loop* nested_loop = *nl;
    status = CombineStatus(status, ProcessLoop(nested_loop, f));
  }

  std::vector<BasicBlock*> loop_bbs{};
  status = CombineStatus(
      status,
      AnalyseAndHoistFromBB(loop, f, loop->GetHeaderBlock(), &loop_bbs));

  for (size_t i = 0; i < loop_bbs.size() && status != Status::Failure; ++i) {
    BasicBlock* bb = loop_bbs[i];
    // do not delete the element
    status =
        CombineStatus(status, AnalyseAndHoistFromBB(loop, f, bb, &loop_bbs));
  }

  return status;
}

Pass::Status LICMPass::AnalyseAndHoistFromBB(
    Loop* loop, Function* f, BasicBlock* bb,
    std::vector<BasicBlock*>* loop_bbs) {
  bool modified = false;
  std::function<bool(Instruction*)> hoist_inst =
      [this, &loop, &modified](Instruction* inst) {
        if (loop->ShouldHoistInstruction(this->context(), inst)) {
          if (!HoistInstruction(loop, inst)) {
            return false;
          }
          modified = true;
        }
        return true;
      };

  if (IsImmediatelyContainedInLoop(loop, f, bb)) {
    if (!bb->WhileEachInst(hoist_inst, false)) {
      return Status::Failure;
    }
  }

  DominatorAnalysis* dom_analysis = context()->GetDominatorAnalysis(f);
  DominatorTree& dom_tree = dom_analysis->GetDomTree();

  for (DominatorTreeNode* child_dom_tree_node : *dom_tree.GetTreeNode(bb)) {
    if (loop->IsInsideLoop(child_dom_tree_node->bb_)) {
      loop_bbs->push_back(child_dom_tree_node->bb_);
    }
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool LICMPass::IsImmediatelyContainedInLoop(Loop* loop, Function* f,
                                            BasicBlock* bb) {
  LoopDescriptor* loop_descriptor = context()->GetLoopDescriptor(f);
  return loop == (*loop_descriptor)[bb->id()];
}

bool LICMPass::HoistInstruction(Loop* loop, Instruction* inst) {
  // TODO(1841): Handle failure to create pre-header.
  BasicBlock* pre_header_bb = loop->GetOrCreatePreHeaderBlock();
  if (!pre_header_bb) {
    return false;
  }
  Instruction* insertion_point = &*pre_header_bb->tail();
  Instruction* previous_node = insertion_point->PreviousNode();
  if (previous_node && (previous_node->opcode() == SpvOpLoopMerge ||
                        previous_node->opcode() == SpvOpSelectionMerge)) {
    insertion_point = previous_node;
  }

  inst->InsertBefore(insertion_point);
  context()->set_instr_block(inst, pre_header_bb);
  return true;
}

}  // namespace opt
}  // namespace spvtools
