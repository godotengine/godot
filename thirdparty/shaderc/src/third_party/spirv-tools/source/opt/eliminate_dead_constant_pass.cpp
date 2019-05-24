// Copyright (c) 2016 Google Inc.
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

#include "source/opt/eliminate_dead_constant_pass.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

Pass::Status EliminateDeadConstantPass::Process() {
  std::unordered_set<Instruction*> working_list;
  // Traverse all the instructions to get the initial set of dead constants as
  // working list and count number of real uses for constants. Uses in
  // annotation instructions do not count.
  std::unordered_map<Instruction*, size_t> use_counts;
  std::vector<Instruction*> constants = context()->GetConstants();
  for (auto* c : constants) {
    uint32_t const_id = c->result_id();
    size_t count = 0;
    context()->get_def_use_mgr()->ForEachUse(
        const_id, [&count](Instruction* user, uint32_t index) {
          (void)index;
          SpvOp op = user->opcode();
          if (!(IsAnnotationInst(op) || IsDebug1Inst(op) || IsDebug2Inst(op) ||
                IsDebug3Inst(op))) {
            ++count;
          }
        });
    use_counts[c] = count;
    if (!count) {
      working_list.insert(c);
    }
  }

  // Start from the constants with 0 uses, back trace through the def-use chain
  // to find all dead constants.
  std::unordered_set<Instruction*> dead_consts;
  while (!working_list.empty()) {
    Instruction* inst = *working_list.begin();
    // Back propagate if the instruction contains IDs in its operands.
    switch (inst->opcode()) {
      case SpvOp::SpvOpConstantComposite:
      case SpvOp::SpvOpSpecConstantComposite:
      case SpvOp::SpvOpSpecConstantOp:
        for (uint32_t i = 0; i < inst->NumInOperands(); i++) {
          // SpecConstantOp instruction contains 'opcode' as its operand. Need
          // to exclude such operands when decreasing uses.
          if (inst->GetInOperand(i).type != SPV_OPERAND_TYPE_ID) {
            continue;
          }
          uint32_t operand_id = inst->GetSingleWordInOperand(i);
          Instruction* def_inst =
              context()->get_def_use_mgr()->GetDef(operand_id);
          // If the use_count does not have any count for the def_inst,
          // def_inst must not be a constant, and should be ignored here.
          if (!use_counts.count(def_inst)) {
            continue;
          }
          // The number of uses should never be less then 0, so it can not be
          // less than 1 before it decreases.
          SPIRV_ASSERT(consumer(), use_counts[def_inst] > 0);
          --use_counts[def_inst];
          if (!use_counts[def_inst]) {
            working_list.insert(def_inst);
          }
        }
        break;
      default:
        break;
    }
    dead_consts.insert(inst);
    working_list.erase(inst);
  }

  // Turn all dead instructions and uses of them to nop
  for (auto* dc : dead_consts) {
    context()->KillDef(dc->result_id());
  }
  return dead_consts.empty() ? Status::SuccessWithoutChange
                             : Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools
