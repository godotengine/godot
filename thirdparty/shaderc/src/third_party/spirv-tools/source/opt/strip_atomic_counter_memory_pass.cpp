// Copyright (c) 2019 Google Inc.
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

#include "source/opt/strip_atomic_counter_memory_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status StripAtomicCounterMemoryPass::Process() {
  bool changed = false;
  context()->module()->ForEachInst([this, &changed](Instruction* inst) {
    auto indices = spvOpcodeMemorySemanticsOperandIndices(inst->opcode());
    if (indices.empty()) return;

    for (auto idx : indices) {
      auto mem_sem_id = inst->GetSingleWordOperand(idx);
      const auto& mem_sem_inst =
          context()->get_def_use_mgr()->GetDef(mem_sem_id);
      // The spec explicitly says that this id must be an OpConstant
      auto mem_sem_val = mem_sem_inst->GetSingleWordOperand(2);
      if (!(mem_sem_val & SpvMemorySemanticsAtomicCounterMemoryMask)) {
        continue;
      }
      mem_sem_val &= ~SpvMemorySemanticsAtomicCounterMemoryMask;

      analysis::Integer int_type(32, false);
      const analysis::Type* uint32_type =
          context()->get_type_mgr()->GetRegisteredType(&int_type);
      auto* new_const = context()->get_constant_mgr()->GetConstant(
          uint32_type, {mem_sem_val});
      auto* new_const_inst =
          context()->get_constant_mgr()->GetDefiningInstruction(new_const);
      auto new_const_id = new_const_inst->result_id();

      inst->SetOperand(idx, {new_const_id});
      context()->UpdateDefUse(inst);
      changed = true;
    }
  });

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
