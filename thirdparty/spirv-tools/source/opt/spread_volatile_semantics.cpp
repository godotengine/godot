// Copyright (c) 2022 Google LLC
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

#include "source/opt/spread_volatile_semantics.h"

#include "source/opt/decoration_manager.h"
#include "source/spirv_constant.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kOpDecorateInOperandBuiltinDecoration = 2u;
constexpr uint32_t kOpLoadInOperandMemoryOperands = 1u;
constexpr uint32_t kOpEntryPointInOperandEntryPoint = 1u;
constexpr uint32_t kOpEntryPointInOperandInterface = 3u;

bool HasBuiltinDecoration(analysis::DecorationManager* decoration_manager,
                          uint32_t var_id, uint32_t built_in) {
  return decoration_manager->FindDecoration(
      var_id, uint32_t(spv::Decoration::BuiltIn),
      [built_in](const Instruction& inst) {
        return built_in == inst.GetSingleWordInOperand(
                               kOpDecorateInOperandBuiltinDecoration);
      });
}

bool IsBuiltInForRayTracingVolatileSemantics(spv::BuiltIn built_in) {
  switch (built_in) {
    case spv::BuiltIn::SMIDNV:
    case spv::BuiltIn::WarpIDNV:
    case spv::BuiltIn::SubgroupSize:
    case spv::BuiltIn::SubgroupLocalInvocationId:
    case spv::BuiltIn::SubgroupEqMask:
    case spv::BuiltIn::SubgroupGeMask:
    case spv::BuiltIn::SubgroupGtMask:
    case spv::BuiltIn::SubgroupLeMask:
    case spv::BuiltIn::SubgroupLtMask:
      return true;
    default:
      return false;
  }
}

bool HasBuiltinForRayTracingVolatileSemantics(
    analysis::DecorationManager* decoration_manager, uint32_t var_id) {
  return decoration_manager->FindDecoration(
      var_id, uint32_t(spv::Decoration::BuiltIn), [](const Instruction& inst) {
        spv::BuiltIn built_in = spv::BuiltIn(
            inst.GetSingleWordInOperand(kOpDecorateInOperandBuiltinDecoration));
        return IsBuiltInForRayTracingVolatileSemantics(built_in);
      });
}

bool HasVolatileDecoration(analysis::DecorationManager* decoration_manager,
                           uint32_t var_id) {
  return decoration_manager->HasDecoration(var_id,
                                           uint32_t(spv::Decoration::Volatile));
}

}  // namespace

Pass::Status SpreadVolatileSemantics::Process() {
  if (HasNoExecutionModel()) {
    return Status::SuccessWithoutChange;
  }
  const bool is_vk_memory_model_enabled =
      context()->get_feature_mgr()->HasCapability(
          spv::Capability::VulkanMemoryModel);
  CollectTargetsForVolatileSemantics(is_vk_memory_model_enabled);

  // If VulkanMemoryModel capability is not enabled, we have to set Volatile
  // decoration for interface variables instead of setting Volatile for load
  // instructions. If an interface (or pointers to it) is used by two load
  // instructions in two entry points and one must be volatile while another
  // is not, we have to report an error for the conflict.
  if (!is_vk_memory_model_enabled &&
      HasInterfaceInConflictOfVolatileSemantics()) {
    return Status::Failure;
  }

  return SpreadVolatileSemanticsToVariables(is_vk_memory_model_enabled);
}

Pass::Status SpreadVolatileSemantics::SpreadVolatileSemanticsToVariables(
    const bool is_vk_memory_model_enabled) {
  Status status = Status::SuccessWithoutChange;
  for (Instruction& var : context()->types_values()) {
    auto entry_function_ids =
        EntryFunctionsToSpreadVolatileSemanticsForVar(var.result_id());
    if (entry_function_ids.empty()) {
      continue;
    }

    if (is_vk_memory_model_enabled) {
      SetVolatileForLoadsInEntries(&var, entry_function_ids);
    } else {
      DecorateVarWithVolatile(&var);
    }
    status = Status::SuccessWithChange;
  }
  return status;
}

bool SpreadVolatileSemantics::IsTargetUsedByNonVolatileLoadInEntryPoint(
    uint32_t var_id, Instruction* entry_point) {
  uint32_t entry_function_id =
      entry_point->GetSingleWordInOperand(kOpEntryPointInOperandEntryPoint);
  std::unordered_set<uint32_t> funcs;
  context()->CollectCallTreeFromRoots(entry_function_id, &funcs);
  return !VisitLoadsOfPointersToVariableInEntries(
      var_id,
      [](Instruction* load) {
        // If it has a load without volatile memory operand, finish traversal
        // and return false.
        if (load->NumInOperands() <= kOpLoadInOperandMemoryOperands) {
          return false;
        }
        uint32_t memory_operands =
            load->GetSingleWordInOperand(kOpLoadInOperandMemoryOperands);
        return (memory_operands & uint32_t(spv::MemoryAccessMask::Volatile)) !=
               0;
      },
      funcs);
}

bool SpreadVolatileSemantics::HasInterfaceInConflictOfVolatileSemantics() {
  for (Instruction& entry_point : get_module()->entry_points()) {
    spv::ExecutionModel execution_model =
        static_cast<spv::ExecutionModel>(entry_point.GetSingleWordInOperand(0));
    for (uint32_t operand_index = kOpEntryPointInOperandInterface;
         operand_index < entry_point.NumInOperands(); ++operand_index) {
      uint32_t var_id = entry_point.GetSingleWordInOperand(operand_index);
      if (!EntryFunctionsToSpreadVolatileSemanticsForVar(var_id).empty() &&
          !IsTargetForVolatileSemantics(var_id, execution_model) &&
          IsTargetUsedByNonVolatileLoadInEntryPoint(var_id, &entry_point)) {
        Instruction* inst = context()->get_def_use_mgr()->GetDef(var_id);
        context()->EmitErrorMessage(
            "Variable is a target for Volatile semantics for an entry point, "
            "but it is not for another entry point",
            inst);
        return true;
      }
    }
  }
  return false;
}

void SpreadVolatileSemantics::MarkVolatileSemanticsForVariable(
    uint32_t var_id, Instruction* entry_point) {
  uint32_t entry_function_id =
      entry_point->GetSingleWordInOperand(kOpEntryPointInOperandEntryPoint);
  auto itr = var_ids_to_entry_fn_for_volatile_semantics_.find(var_id);
  if (itr == var_ids_to_entry_fn_for_volatile_semantics_.end()) {
    var_ids_to_entry_fn_for_volatile_semantics_[var_id] = {entry_function_id};
    return;
  }
  itr->second.insert(entry_function_id);
}

void SpreadVolatileSemantics::CollectTargetsForVolatileSemantics(
    const bool is_vk_memory_model_enabled) {
  for (Instruction& entry_point : get_module()->entry_points()) {
    spv::ExecutionModel execution_model =
        static_cast<spv::ExecutionModel>(entry_point.GetSingleWordInOperand(0));
    for (uint32_t operand_index = kOpEntryPointInOperandInterface;
         operand_index < entry_point.NumInOperands(); ++operand_index) {
      uint32_t var_id = entry_point.GetSingleWordInOperand(operand_index);
      if (!IsTargetForVolatileSemantics(var_id, execution_model)) {
        continue;
      }
      if (is_vk_memory_model_enabled ||
          IsTargetUsedByNonVolatileLoadInEntryPoint(var_id, &entry_point)) {
        MarkVolatileSemanticsForVariable(var_id, &entry_point);
      }
    }
  }
}

void SpreadVolatileSemantics::DecorateVarWithVolatile(Instruction* var) {
  analysis::DecorationManager* decoration_manager =
      context()->get_decoration_mgr();
  uint32_t var_id = var->result_id();
  if (HasVolatileDecoration(decoration_manager, var_id)) {
    return;
  }
  get_decoration_mgr()->AddDecoration(
      spv::Op::OpDecorate,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {var_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {uint32_t(spv::Decoration::Volatile)}}});
}

bool SpreadVolatileSemantics::VisitLoadsOfPointersToVariableInEntries(
    uint32_t var_id, const std::function<bool(Instruction*)>& handle_load,
    const std::unordered_set<uint32_t>& function_ids) {
  std::vector<uint32_t> worklist({var_id});
  auto* def_use_mgr = context()->get_def_use_mgr();
  while (!worklist.empty()) {
    uint32_t ptr_id = worklist.back();
    worklist.pop_back();
    bool finish_traversal = !def_use_mgr->WhileEachUser(
        ptr_id, [this, &worklist, &ptr_id, handle_load,
                 &function_ids](Instruction* user) {
          BasicBlock* block = context()->get_instr_block(user);
          if (block == nullptr ||
              function_ids.find(block->GetParent()->result_id()) ==
                  function_ids.end()) {
            return true;
          }

          if (user->opcode() == spv::Op::OpAccessChain ||
              user->opcode() == spv::Op::OpInBoundsAccessChain ||
              user->opcode() == spv::Op::OpPtrAccessChain ||
              user->opcode() == spv::Op::OpInBoundsPtrAccessChain ||
              user->opcode() == spv::Op::OpCopyObject) {
            if (ptr_id == user->GetSingleWordInOperand(0))
              worklist.push_back(user->result_id());
            return true;
          }

          if (user->opcode() != spv::Op::OpLoad) {
            return true;
          }

          return handle_load(user);
        });
    if (finish_traversal) return false;
  }
  return true;
}

void SpreadVolatileSemantics::SetVolatileForLoadsInEntries(
    Instruction* var, const std::unordered_set<uint32_t>& entry_function_ids) {
  // Set Volatile memory operand for all load instructions if they do not have
  // it.
  for (auto entry_id : entry_function_ids) {
    std::unordered_set<uint32_t> funcs;
    context()->CollectCallTreeFromRoots(entry_id, &funcs);
    VisitLoadsOfPointersToVariableInEntries(
        var->result_id(),
        [](Instruction* load) {
          if (load->NumInOperands() <= kOpLoadInOperandMemoryOperands) {
            load->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                              {uint32_t(spv::MemoryAccessMask::Volatile)}});
            return true;
          }
          uint32_t memory_operands =
              load->GetSingleWordInOperand(kOpLoadInOperandMemoryOperands);
          memory_operands |= uint32_t(spv::MemoryAccessMask::Volatile);
          load->SetInOperand(kOpLoadInOperandMemoryOperands, {memory_operands});
          return true;
        },
        funcs);
  }
}

bool SpreadVolatileSemantics::IsTargetForVolatileSemantics(
    uint32_t var_id, spv::ExecutionModel execution_model) {
  analysis::DecorationManager* decoration_manager =
      context()->get_decoration_mgr();
  if (execution_model == spv::ExecutionModel::Fragment) {
    return get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 6) &&
           HasBuiltinDecoration(decoration_manager, var_id,
                                uint32_t(spv::BuiltIn::HelperInvocation));
  }

  if (execution_model == spv::ExecutionModel::IntersectionKHR ||
      execution_model == spv::ExecutionModel::IntersectionNV) {
    if (HasBuiltinDecoration(decoration_manager, var_id,
                             uint32_t(spv::BuiltIn::RayTmaxKHR))) {
      return true;
    }
  }

  switch (execution_model) {
    case spv::ExecutionModel::RayGenerationKHR:
    case spv::ExecutionModel::ClosestHitKHR:
    case spv::ExecutionModel::MissKHR:
    case spv::ExecutionModel::CallableKHR:
    case spv::ExecutionModel::IntersectionKHR:
      return HasBuiltinForRayTracingVolatileSemantics(decoration_manager,
                                                      var_id);
    default:
      return false;
  }
}

}  // namespace opt
}  // namespace spvtools
