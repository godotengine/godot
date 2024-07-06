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

#include "source/opt/module.h"

#include <algorithm>
#include <cstring>
#include <ostream>

#include "source/operand.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

uint32_t Module::TakeNextIdBound() {
  if (context()) {
    if (id_bound() >= context()->max_id_bound()) {
      return 0;
    }
  } else if (id_bound() >= kDefaultMaxIdBound) {
    return 0;
  }

  return header_.bound++;
}

std::vector<Instruction*> Module::GetTypes() {
  std::vector<Instruction*> type_insts;
  for (auto& inst : types_values_) {
    if (IsTypeInst(inst.opcode())) type_insts.push_back(&inst);
  }
  return type_insts;
}

std::vector<const Instruction*> Module::GetTypes() const {
  std::vector<const Instruction*> type_insts;
  for (auto& inst : types_values_) {
    if (IsTypeInst(inst.opcode())) type_insts.push_back(&inst);
  }
  return type_insts;
}

std::vector<Instruction*> Module::GetConstants() {
  std::vector<Instruction*> const_insts;
  for (auto& inst : types_values_) {
    if (IsConstantInst(inst.opcode())) const_insts.push_back(&inst);
  }
  return const_insts;
}

std::vector<const Instruction*> Module::GetConstants() const {
  std::vector<const Instruction*> const_insts;
  for (auto& inst : types_values_) {
    if (IsConstantInst(inst.opcode())) const_insts.push_back(&inst);
  }
  return const_insts;
}

uint32_t Module::GetGlobalValue(spv::Op opcode) const {
  for (auto& inst : types_values_) {
    if (inst.opcode() == opcode) return inst.result_id();
  }
  return 0;
}

void Module::AddGlobalValue(spv::Op opcode, uint32_t result_id,
                            uint32_t type_id) {
  std::unique_ptr<Instruction> newGlobal(
      new Instruction(context(), opcode, type_id, result_id, {}));
  AddGlobalValue(std::move(newGlobal));
}

void Module::ForEachInst(const std::function<void(Instruction*)>& f,
                         bool run_on_debug_line_insts) {
#define DELEGATE(list) list.ForEachInst(f, run_on_debug_line_insts)
  DELEGATE(capabilities_);
  DELEGATE(extensions_);
  DELEGATE(ext_inst_imports_);
  if (memory_model_) memory_model_->ForEachInst(f, run_on_debug_line_insts);
  if (sampled_image_address_mode_)
    sampled_image_address_mode_->ForEachInst(f, run_on_debug_line_insts);
  DELEGATE(entry_points_);
  DELEGATE(execution_modes_);
  DELEGATE(debugs1_);
  DELEGATE(debugs2_);
  DELEGATE(debugs3_);
  DELEGATE(ext_inst_debuginfo_);
  DELEGATE(annotations_);
  DELEGATE(types_values_);
  for (auto& i : functions_) {
    i->ForEachInst(f, run_on_debug_line_insts,
                   /* run_on_non_semantic_insts = */ true);
  }
#undef DELEGATE
}

void Module::ForEachInst(const std::function<void(const Instruction*)>& f,
                         bool run_on_debug_line_insts) const {
#define DELEGATE(i) i.ForEachInst(f, run_on_debug_line_insts)
  for (auto& i : capabilities_) DELEGATE(i);
  for (auto& i : extensions_) DELEGATE(i);
  for (auto& i : ext_inst_imports_) DELEGATE(i);
  if (memory_model_)
    static_cast<const Instruction*>(memory_model_.get())
        ->ForEachInst(f, run_on_debug_line_insts);
  if (sampled_image_address_mode_)
    static_cast<const Instruction*>(sampled_image_address_mode_.get())
        ->ForEachInst(f, run_on_debug_line_insts);
  for (auto& i : entry_points_) DELEGATE(i);
  for (auto& i : execution_modes_) DELEGATE(i);
  for (auto& i : debugs1_) DELEGATE(i);
  for (auto& i : debugs2_) DELEGATE(i);
  for (auto& i : debugs3_) DELEGATE(i);
  for (auto& i : annotations_) DELEGATE(i);
  for (auto& i : types_values_) DELEGATE(i);
  for (auto& i : ext_inst_debuginfo_) DELEGATE(i);
  for (auto& i : functions_) {
    static_cast<const Function*>(i.get())->ForEachInst(
        f, run_on_debug_line_insts,
        /* run_on_non_semantic_insts = */ true);
  }
  if (run_on_debug_line_insts) {
    for (auto& i : trailing_dbg_line_info_) DELEGATE(i);
  }
#undef DELEGATE
}

void Module::ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const {
  binary->push_back(header_.magic_number);
  binary->push_back(header_.version);
  // TODO(antiagainst): should we change the generator number?
  binary->push_back(header_.generator);
  binary->push_back(header_.bound);
  binary->push_back(header_.schema);

  size_t bound_idx = binary->size() - 2;
  DebugScope last_scope(kNoDebugScope, kNoInlinedAt);
  const Instruction* last_line_inst = nullptr;
  bool between_merge_and_branch = false;
  bool between_label_and_phi_var = false;
  auto write_inst = [binary, skip_nop, &last_scope, &last_line_inst,
                     &between_merge_and_branch, &between_label_and_phi_var,
                     this](const Instruction* i) {
    // Skip emitting line instructions between merge and branch instructions.
    auto opcode = i->opcode();
    if (between_merge_and_branch && i->IsLineInst()) {
      return;
    }
    if (last_line_inst != nullptr) {
      // If the current instruction is OpLine or DebugLine and it is the same
      // as the last line instruction that is still effective (can be applied
      // to the next instruction), we skip writing the current instruction.
      if (i->IsLine()) {
        uint32_t operand_index = 0;
        if (last_line_inst->WhileEachInOperand(
                [&operand_index, i](const uint32_t* word) {
                  assert(i->NumInOperandWords() > operand_index);
                  return *word == i->GetSingleWordInOperand(operand_index++);
                })) {
          return;
        }
      } else if (!i->IsNoLine() && i->dbg_line_insts().empty()) {
        // If the current instruction does not have the line information,
        // the last line information is not effective any more. Emit OpNoLine
        // or DebugNoLine to specify it.
        uint32_t shader_set_id = context()
                                     ->get_feature_mgr()
                                     ->GetExtInstImportId_Shader100DebugInfo();
        if (shader_set_id != 0) {
          binary->push_back((5 << 16) |
                            static_cast<uint16_t>(spv::Op::OpExtInst));
          binary->push_back(context()->get_type_mgr()->GetVoidTypeId());
          binary->push_back(context()->TakeNextId());
          binary->push_back(shader_set_id);
          binary->push_back(NonSemanticShaderDebugInfo100DebugNoLine);
        } else {
          binary->push_back((1 << 16) |
                            static_cast<uint16_t>(spv::Op::OpNoLine));
        }
        last_line_inst = nullptr;
      }
    }

    if (opcode == spv::Op::OpLabel) {
      between_label_and_phi_var = true;
    } else if (opcode != spv::Op::OpVariable && opcode != spv::Op::OpPhi &&
               !spvtools::opt::IsOpLineInst(opcode)) {
      between_label_and_phi_var = false;
    }

    if (!(skip_nop && i->IsNop())) {
      const auto& scope = i->GetDebugScope();
      if (scope != last_scope && !between_merge_and_branch) {
        // Can only emit nonsemantic instructions after all phi instructions
        // in a block so don't emit scope instructions before phi instructions
        // for NonSemantic.Shader.DebugInfo.100.
        if (!between_label_and_phi_var ||
            context()
                ->get_feature_mgr()
                ->GetExtInstImportId_OpenCL100DebugInfo()) {
          // Emit DebugScope |scope| to |binary|.
          auto dbg_inst = ext_inst_debuginfo_.begin();
          scope.ToBinary(dbg_inst->type_id(), context()->TakeNextId(),
                         dbg_inst->GetSingleWordOperand(2), binary);
        }
        last_scope = scope;
      }

      i->ToBinaryWithoutAttachedDebugInsts(binary);
    }
    // Update the last line instruction.
    between_merge_and_branch = false;
    if (spvOpcodeIsBlockTerminator(opcode) || i->IsNoLine()) {
      last_line_inst = nullptr;
    } else if (opcode == spv::Op::OpLoopMerge ||
               opcode == spv::Op::OpSelectionMerge) {
      between_merge_and_branch = true;
      last_line_inst = nullptr;
    } else if (i->IsLine()) {
      last_line_inst = i;
    }
  };
  ForEachInst(write_inst, true);

  // We create new instructions for DebugScope and DebugNoLine. The bound must
  // be updated.
  binary->data()[bound_idx] = header_.bound;
}

uint32_t Module::ComputeIdBound() const {
  uint32_t highest = 0;

  ForEachInst(
      [&highest](const Instruction* inst) {
        for (const auto& operand : *inst) {
          if (spvIsIdType(operand.type)) {
            highest = std::max(highest, operand.words[0]);
          }
        }
      },
      true /* scan debug line insts as well */);

  return highest + 1;
}

bool Module::HasExplicitCapability(uint32_t cap) {
  for (auto& ci : capabilities_) {
    uint32_t tcap = ci.GetSingleWordOperand(0);
    if (tcap == cap) {
      return true;
    }
  }
  return false;
}

uint32_t Module::GetExtInstImportId(const char* extstr) {
  for (auto& ei : ext_inst_imports_)
    if (!ei.GetInOperand(0).AsString().compare(extstr)) return ei.result_id();
  return 0;
}

std::ostream& operator<<(std::ostream& str, const Module& module) {
  module.ForEachInst([&str](const Instruction* inst) {
    str << *inst;
    if (inst->opcode() != spv::Op::OpFunctionEnd) {
      str << std::endl;
    }
  });
  return str;
}

}  // namespace opt
}  // namespace spvtools
