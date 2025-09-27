// Copyright (c) 2021 Google LLC.
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

#include "source/opt/dataflow.h"

#include <algorithm>
#include <cstdint>

namespace spvtools {
namespace opt {

bool DataFlowAnalysis::Enqueue(Instruction* inst) {
  bool& is_enqueued = on_worklist_[inst];
  if (is_enqueued) return false;
  is_enqueued = true;
  worklist_.push(inst);
  return true;
}

DataFlowAnalysis::VisitResult DataFlowAnalysis::RunOnce(
    Function* function, bool is_first_iteration) {
  InitializeWorklist(function, is_first_iteration);
  VisitResult ret = VisitResult::kResultFixed;
  while (!worklist_.empty()) {
    Instruction* top = worklist_.front();
    worklist_.pop();
    on_worklist_[top] = false;
    VisitResult result = Visit(top);
    if (result == VisitResult::kResultChanged) {
      EnqueueSuccessors(top);
      ret = VisitResult::kResultChanged;
    }
  }
  return ret;
}

void DataFlowAnalysis::Run(Function* function) {
  VisitResult result = RunOnce(function, true);
  while (result == VisitResult::kResultChanged) {
    result = RunOnce(function, false);
  }
}

void ForwardDataFlowAnalysis::InitializeWorklist(Function* function,
                                                 bool /*is_first_iteration*/) {
  context().cfg()->ForEachBlockInReversePostOrder(
      function->entry().get(), [this](BasicBlock* bb) {
        if (label_position_ == LabelPosition::kLabelsOnly) {
          Enqueue(bb->GetLabelInst());
          return;
        }
        if (label_position_ == LabelPosition::kLabelsAtBeginning) {
          Enqueue(bb->GetLabelInst());
        }
        for (Instruction& inst : *bb) {
          Enqueue(&inst);
        }
        if (label_position_ == LabelPosition::kLabelsAtEnd) {
          Enqueue(bb->GetLabelInst());
        }
      });
}

void ForwardDataFlowAnalysis::EnqueueUsers(Instruction* inst) {
  context().get_def_use_mgr()->ForEachUser(
      inst, [this](Instruction* user) { Enqueue(user); });
}

void ForwardDataFlowAnalysis::EnqueueBlockSuccessors(Instruction* inst) {
  if (inst->opcode() != spv::Op::OpLabel) return;
  context()
      .cfg()
      ->block(inst->result_id())
      ->ForEachSuccessorLabel([this](uint32_t* label) {
        Enqueue(context().cfg()->block(*label)->GetLabelInst());
      });
}

}  // namespace opt
}  // namespace spvtools
