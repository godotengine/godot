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

#include "source/opt/workaround1209.h"

#include <list>
#include <memory>
#include <stack>
#include <utility>

namespace spvtools {
namespace opt {

Pass::Status Workaround1209::Process() {
  bool modified = false;
  modified = RemoveOpUnreachableInLoops();
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool Workaround1209::RemoveOpUnreachableInLoops() {
  bool modified = false;
  for (auto& func : *get_module()) {
    std::list<BasicBlock*> structured_order;
    cfg()->ComputeStructuredOrder(&func, &*func.begin(), &structured_order);

    // Keep track of the loop merges.  The top of the stack will always be the
    // loop merge for the loop that immediately contains the basic block being
    // processed.
    std::stack<uint32_t> loop_merges;
    for (BasicBlock* bb : structured_order) {
      if (!loop_merges.empty() && bb->id() == loop_merges.top()) {
        loop_merges.pop();
      }

      if (bb->tail()->opcode() == SpvOpUnreachable) {
        if (!loop_merges.empty()) {
          // We found an OpUnreachable inside a loop.
          // Replace it with an unconditional branch to the loop merge.
          context()->KillInst(&*bb->tail());
          std::unique_ptr<Instruction> new_branch(
              new Instruction(context(), SpvOpBranch, 0, 0,
                              {{spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                                {loop_merges.top()}}}));
          context()->AnalyzeDefUse(&*new_branch);
          bb->AddInstruction(std::move(new_branch));
          modified = true;
        }
      } else {
        if (bb->GetLoopMergeInst()) {
          loop_merges.push(bb->MergeBlockIdIfAny());
        }
      }
    }
  }
  return modified;
}
}  // namespace opt
}  // namespace spvtools
