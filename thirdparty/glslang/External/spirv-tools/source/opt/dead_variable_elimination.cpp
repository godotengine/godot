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

#include "source/opt/dead_variable_elimination.h"

#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

// This optimization removes global variables that are not needed because they
// are definitely not accessed.
Pass::Status DeadVariableElimination::Process() {
  // The algorithm will compute the reference count for every global variable.
  // Anything with a reference count of 0 will then be deleted.  For variables
  // that might have references that are not explicit in this context, we use
  // the value kMustKeep as the reference count.
  std::vector<uint32_t> ids_to_remove;

  // Get the reference count for all of the global OpVariable instructions.
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() != spv::Op::OpVariable) {
      continue;
    }

    size_t count = 0;
    uint32_t result_id = inst.result_id();

    // Check the linkage.  If it is exported, it could be reference somewhere
    // else, so we must keep the variable around.
    get_decoration_mgr()->ForEachDecoration(
        result_id, uint32_t(spv::Decoration::LinkageAttributes),
        [&count](const Instruction& linkage_instruction) {
          uint32_t last_operand = linkage_instruction.NumOperands() - 1;
          if (spv::LinkageType(linkage_instruction.GetSingleWordOperand(
                  last_operand)) == spv::LinkageType::Export) {
            count = kMustKeep;
          }
        });

    if (count != kMustKeep) {
      // If we don't have to keep the instruction for other reasons, then look
      // at the uses and count the number of real references.
      count = 0;
      get_def_use_mgr()->ForEachUser(result_id, [&count](Instruction* user) {
        if (!IsAnnotationInst(user->opcode()) &&
            user->opcode() != spv::Op::OpName) {
          ++count;
        }
      });
    }
    reference_count_[result_id] = count;
    if (count == 0) {
      ids_to_remove.push_back(result_id);
    }
  }

  // Remove all of the variables that have a reference count of 0.
  bool modified = false;
  if (!ids_to_remove.empty()) {
    modified = true;
    for (auto result_id : ids_to_remove) {
      DeleteVariable(result_id);
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

void DeadVariableElimination::DeleteVariable(uint32_t result_id) {
  Instruction* inst = get_def_use_mgr()->GetDef(result_id);
  assert(inst->opcode() == spv::Op::OpVariable &&
         "Should not be trying to delete anything other than an OpVariable.");

  // Look for an initializer that references another variable.  We need to know
  // if that variable can be deleted after the reference is removed.
  if (inst->NumOperands() == 4) {
    Instruction* initializer =
        get_def_use_mgr()->GetDef(inst->GetSingleWordOperand(3));

    // TODO: Handle OpSpecConstantOP which might be defined in terms of other
    // variables.  Will probably require a unified dead code pass that does all
    // instruction types.  (Issue 906)
    if (initializer->opcode() == spv::Op::OpVariable) {
      uint32_t initializer_id = initializer->result_id();
      size_t& count = reference_count_[initializer_id];
      if (count != kMustKeep) {
        --count;
      }

      if (count == 0) {
        DeleteVariable(initializer_id);
      }
    }
  }
  context()->KillDef(result_id);
}
}  // namespace opt
}  // namespace spvtools
