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

#include "source/opt/interface_var_sroa.h"

#include <iostream>

#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/log.h"
#include "source/opt/type_manager.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kOpDecorateDecorationInOperandIndex = 1;
constexpr uint32_t kOpDecorateLiteralInOperandIndex = 2;
constexpr uint32_t kOpEntryPointInOperandInterface = 3;
constexpr uint32_t kOpVariableStorageClassInOperandIndex = 0;
constexpr uint32_t kOpTypeArrayElemTypeInOperandIndex = 0;
constexpr uint32_t kOpTypeArrayLengthInOperandIndex = 1;
constexpr uint32_t kOpTypeMatrixColCountInOperandIndex = 1;
constexpr uint32_t kOpTypeMatrixColTypeInOperandIndex = 0;
constexpr uint32_t kOpTypePtrTypeInOperandIndex = 1;
constexpr uint32_t kOpConstantValueInOperandIndex = 0;

// Get the length of the OpTypeArray |array_type|.
uint32_t GetArrayLength(analysis::DefUseManager* def_use_mgr,
                        Instruction* array_type) {
  assert(array_type->opcode() == spv::Op::OpTypeArray);
  uint32_t const_int_id =
      array_type->GetSingleWordInOperand(kOpTypeArrayLengthInOperandIndex);
  Instruction* array_length_inst = def_use_mgr->GetDef(const_int_id);
  assert(array_length_inst->opcode() == spv::Op::OpConstant);
  return array_length_inst->GetSingleWordInOperand(
      kOpConstantValueInOperandIndex);
}

// Get the element type instruction of the OpTypeArray |array_type|.
Instruction* GetArrayElementType(analysis::DefUseManager* def_use_mgr,
                                 Instruction* array_type) {
  assert(array_type->opcode() == spv::Op::OpTypeArray);
  uint32_t elem_type_id =
      array_type->GetSingleWordInOperand(kOpTypeArrayElemTypeInOperandIndex);
  return def_use_mgr->GetDef(elem_type_id);
}

// Get the column type instruction of the OpTypeMatrix |matrix_type|.
Instruction* GetMatrixColumnType(analysis::DefUseManager* def_use_mgr,
                                 Instruction* matrix_type) {
  assert(matrix_type->opcode() == spv::Op::OpTypeMatrix);
  uint32_t column_type_id =
      matrix_type->GetSingleWordInOperand(kOpTypeMatrixColTypeInOperandIndex);
  return def_use_mgr->GetDef(column_type_id);
}

// Traverses the component type of OpTypeArray or OpTypeMatrix. Repeats it
// |depth_to_component| times recursively and returns the component type.
// |type_id| is the result id of the OpTypeArray or OpTypeMatrix instruction.
uint32_t GetComponentTypeOfArrayMatrix(analysis::DefUseManager* def_use_mgr,
                                       uint32_t type_id,
                                       uint32_t depth_to_component) {
  if (depth_to_component == 0) return type_id;

  Instruction* type_inst = def_use_mgr->GetDef(type_id);
  if (type_inst->opcode() == spv::Op::OpTypeArray) {
    uint32_t elem_type_id =
        type_inst->GetSingleWordInOperand(kOpTypeArrayElemTypeInOperandIndex);
    return GetComponentTypeOfArrayMatrix(def_use_mgr, elem_type_id,
                                         depth_to_component - 1);
  }

  assert(type_inst->opcode() == spv::Op::OpTypeMatrix);
  uint32_t column_type_id =
      type_inst->GetSingleWordInOperand(kOpTypeMatrixColTypeInOperandIndex);
  return GetComponentTypeOfArrayMatrix(def_use_mgr, column_type_id,
                                       depth_to_component - 1);
}

// Creates an OpDecorate instruction whose Target is |var_id| and Decoration is
// |decoration|. Adds |literal| as an extra operand of the instruction.
void CreateDecoration(analysis::DecorationManager* decoration_mgr,
                      uint32_t var_id, spv::Decoration decoration,
                      uint32_t literal) {
  std::vector<Operand> operands({
      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {var_id}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_DECORATION,
       {static_cast<uint32_t>(decoration)}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {literal}},
  });
  decoration_mgr->AddDecoration(spv::Op::OpDecorate, std::move(operands));
}

// Replaces load instructions with composite construct instructions in all the
// users of the loads. |loads_to_composites| is the mapping from each load to
// its corresponding OpCompositeConstruct.
void ReplaceLoadWithCompositeConstruct(
    IRContext* context,
    const std::unordered_map<Instruction*, Instruction*>& loads_to_composites) {
  for (const auto& load_and_composite : loads_to_composites) {
    Instruction* load = load_and_composite.first;
    Instruction* composite_construct = load_and_composite.second;

    std::vector<Instruction*> users;
    context->get_def_use_mgr()->ForEachUse(
        load, [&users, composite_construct](Instruction* user, uint32_t index) {
          user->GetOperand(index).words[0] = composite_construct->result_id();
          users.push_back(user);
        });

    for (Instruction* user : users)
      context->get_def_use_mgr()->AnalyzeInstUse(user);
  }
}

// Returns the storage class of the instruction |var|.
spv::StorageClass GetStorageClass(Instruction* var) {
  return static_cast<spv::StorageClass>(
      var->GetSingleWordInOperand(kOpVariableStorageClassInOperandIndex));
}

}  // namespace

bool InterfaceVariableScalarReplacement::HasExtraArrayness(
    Instruction& entry_point, Instruction* var) {
  spv::ExecutionModel execution_model =
      static_cast<spv::ExecutionModel>(entry_point.GetSingleWordInOperand(0));
  if (execution_model != spv::ExecutionModel::TessellationEvaluation &&
      execution_model != spv::ExecutionModel::TessellationControl) {
    return false;
  }
  if (!context()->get_decoration_mgr()->HasDecoration(
          var->result_id(), uint32_t(spv::Decoration::Patch))) {
    if (execution_model == spv::ExecutionModel::TessellationControl)
      return true;
    return GetStorageClass(var) != spv::StorageClass::Output;
  }
  return false;
}

bool InterfaceVariableScalarReplacement::
    CheckExtraArraynessConflictBetweenEntries(Instruction* interface_var,
                                              bool has_extra_arrayness) {
  if (has_extra_arrayness) {
    return !ReportErrorIfHasNoExtraArraynessForOtherEntry(interface_var);
  }
  return !ReportErrorIfHasExtraArraynessForOtherEntry(interface_var);
}

bool InterfaceVariableScalarReplacement::GetVariableLocation(
    Instruction* var, uint32_t* location) {
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      var->result_id(), uint32_t(spv::Decoration::Location),
      [location](const Instruction& inst) {
        *location =
            inst.GetSingleWordInOperand(kOpDecorateLiteralInOperandIndex);
        return false;
      });
}

bool InterfaceVariableScalarReplacement::GetVariableComponent(
    Instruction* var, uint32_t* component) {
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      var->result_id(), uint32_t(spv::Decoration::Component),
      [component](const Instruction& inst) {
        *component =
            inst.GetSingleWordInOperand(kOpDecorateLiteralInOperandIndex);
        return false;
      });
}

std::vector<Instruction*>
InterfaceVariableScalarReplacement::CollectInterfaceVariables(
    Instruction& entry_point) {
  std::vector<Instruction*> interface_vars;
  for (uint32_t i = kOpEntryPointInOperandInterface;
       i < entry_point.NumInOperands(); ++i) {
    Instruction* interface_var = context()->get_def_use_mgr()->GetDef(
        entry_point.GetSingleWordInOperand(i));
    assert(interface_var->opcode() == spv::Op::OpVariable);

    spv::StorageClass storage_class = GetStorageClass(interface_var);
    if (storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      continue;
    }

    interface_vars.push_back(interface_var);
  }
  return interface_vars;
}

void InterfaceVariableScalarReplacement::KillInstructionAndUsers(
    Instruction* inst) {
  if (inst->opcode() == spv::Op::OpEntryPoint) {
    return;
  }
  if (inst->opcode() != spv::Op::OpAccessChain) {
    context()->KillInst(inst);
    return;
  }
  std::vector<Instruction*> users;
  context()->get_def_use_mgr()->ForEachUser(
      inst, [&users](Instruction* user) { users.push_back(user); });
  for (auto user : users) {
    context()->KillInst(user);
  }
  context()->KillInst(inst);
}

void InterfaceVariableScalarReplacement::KillInstructionsAndUsers(
    const std::vector<Instruction*>& insts) {
  for (Instruction* inst : insts) {
    KillInstructionAndUsers(inst);
  }
}

void InterfaceVariableScalarReplacement::KillLocationAndComponentDecorations(
    uint32_t var_id) {
  context()->get_decoration_mgr()->RemoveDecorationsFrom(
      var_id, [](const Instruction& inst) {
        spv::Decoration decoration = spv::Decoration(
            inst.GetSingleWordInOperand(kOpDecorateDecorationInOperandIndex));
        return decoration == spv::Decoration::Location ||
               decoration == spv::Decoration::Component;
      });
}

bool InterfaceVariableScalarReplacement::ReplaceInterfaceVariableWithScalars(
    Instruction* interface_var, Instruction* interface_var_type,
    uint32_t location, uint32_t component, uint32_t extra_array_length) {
  NestedCompositeComponents scalar_interface_vars =
      CreateScalarInterfaceVarsForReplacement(interface_var_type,
                                              GetStorageClass(interface_var),
                                              extra_array_length);

  AddLocationAndComponentDecorations(scalar_interface_vars, &location,
                                     component);
  KillLocationAndComponentDecorations(interface_var->result_id());

  if (!ReplaceInterfaceVarWith(interface_var, extra_array_length,
                               scalar_interface_vars)) {
    return false;
  }

  context()->KillInst(interface_var);
  return true;
}

bool InterfaceVariableScalarReplacement::ReplaceInterfaceVarWith(
    Instruction* interface_var, uint32_t extra_array_length,
    const NestedCompositeComponents& scalar_interface_vars) {
  std::vector<Instruction*> users;
  context()->get_def_use_mgr()->ForEachUser(
      interface_var, [&users](Instruction* user) { users.push_back(user); });

  std::vector<uint32_t> interface_var_component_indices;
  std::unordered_map<Instruction*, Instruction*> loads_to_composites;
  std::unordered_map<Instruction*, Instruction*>
      loads_for_access_chain_to_composites;
  if (extra_array_length != 0) {
    // Note that the extra arrayness is the first dimension of the array
    // interface variable.
    for (uint32_t index = 0; index < extra_array_length; ++index) {
      std::unordered_map<Instruction*, Instruction*> loads_to_component_values;
      if (!ReplaceComponentsOfInterfaceVarWith(
              interface_var, users, scalar_interface_vars,
              interface_var_component_indices, &index,
              &loads_to_component_values,
              &loads_for_access_chain_to_composites)) {
        return false;
      }
      AddComponentsToCompositesForLoads(loads_to_component_values,
                                        &loads_to_composites, 0);
    }
  } else if (!ReplaceComponentsOfInterfaceVarWith(
                 interface_var, users, scalar_interface_vars,
                 interface_var_component_indices, nullptr, &loads_to_composites,
                 &loads_for_access_chain_to_composites)) {
    return false;
  }

  ReplaceLoadWithCompositeConstruct(context(), loads_to_composites);
  ReplaceLoadWithCompositeConstruct(context(),
                                    loads_for_access_chain_to_composites);

  KillInstructionsAndUsers(users);
  return true;
}

void InterfaceVariableScalarReplacement::AddLocationAndComponentDecorations(
    const NestedCompositeComponents& vars, uint32_t* location,
    uint32_t component) {
  if (!vars.HasMultipleComponents()) {
    uint32_t var_id = vars.GetComponentVariable()->result_id();
    CreateDecoration(context()->get_decoration_mgr(), var_id,
                     spv::Decoration::Location, *location);
    CreateDecoration(context()->get_decoration_mgr(), var_id,
                     spv::Decoration::Component, component);
    ++(*location);
    return;
  }
  for (const auto& var : vars.GetComponents()) {
    AddLocationAndComponentDecorations(var, location, component);
  }
}

bool InterfaceVariableScalarReplacement::ReplaceComponentsOfInterfaceVarWith(
    Instruction* interface_var,
    const std::vector<Instruction*>& interface_var_users,
    const NestedCompositeComponents& scalar_interface_vars,
    std::vector<uint32_t>& interface_var_component_indices,
    const uint32_t* extra_array_index,
    std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
    std::unordered_map<Instruction*, Instruction*>*
        loads_for_access_chain_to_composites) {
  if (!scalar_interface_vars.HasMultipleComponents()) {
    for (Instruction* interface_var_user : interface_var_users) {
      if (!ReplaceComponentOfInterfaceVarWith(
              interface_var, interface_var_user,
              scalar_interface_vars.GetComponentVariable(),
              interface_var_component_indices, extra_array_index,
              loads_to_composites, loads_for_access_chain_to_composites)) {
        return false;
      }
    }
    return true;
  }
  return ReplaceMultipleComponentsOfInterfaceVarWith(
      interface_var, interface_var_users, scalar_interface_vars.GetComponents(),
      interface_var_component_indices, extra_array_index, loads_to_composites,
      loads_for_access_chain_to_composites);
}

bool InterfaceVariableScalarReplacement::
    ReplaceMultipleComponentsOfInterfaceVarWith(
        Instruction* interface_var,
        const std::vector<Instruction*>& interface_var_users,
        const std::vector<NestedCompositeComponents>& components,
        std::vector<uint32_t>& interface_var_component_indices,
        const uint32_t* extra_array_index,
        std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
        std::unordered_map<Instruction*, Instruction*>*
            loads_for_access_chain_to_composites) {
  for (uint32_t i = 0; i < components.size(); ++i) {
    interface_var_component_indices.push_back(i);
    std::unordered_map<Instruction*, Instruction*> loads_to_component_values;
    std::unordered_map<Instruction*, Instruction*>
        loads_for_access_chain_to_component_values;
    if (!ReplaceComponentsOfInterfaceVarWith(
            interface_var, interface_var_users, components[i],
            interface_var_component_indices, extra_array_index,
            &loads_to_component_values,
            &loads_for_access_chain_to_component_values)) {
      return false;
    }
    interface_var_component_indices.pop_back();

    uint32_t depth_to_component =
        static_cast<uint32_t>(interface_var_component_indices.size());
    AddComponentsToCompositesForLoads(
        loads_for_access_chain_to_component_values,
        loads_for_access_chain_to_composites, depth_to_component);
    if (extra_array_index) ++depth_to_component;
    AddComponentsToCompositesForLoads(loads_to_component_values,
                                      loads_to_composites, depth_to_component);
  }
  return true;
}

bool InterfaceVariableScalarReplacement::ReplaceComponentOfInterfaceVarWith(
    Instruction* interface_var, Instruction* interface_var_user,
    Instruction* scalar_var,
    const std::vector<uint32_t>& interface_var_component_indices,
    const uint32_t* extra_array_index,
    std::unordered_map<Instruction*, Instruction*>* loads_to_component_values,
    std::unordered_map<Instruction*, Instruction*>*
        loads_for_access_chain_to_component_values) {
  spv::Op opcode = interface_var_user->opcode();
  if (opcode == spv::Op::OpStore) {
    uint32_t value_id = interface_var_user->GetSingleWordInOperand(1);
    StoreComponentOfValueToScalarVar(value_id, interface_var_component_indices,
                                     scalar_var, extra_array_index,
                                     interface_var_user);
    return true;
  }
  if (opcode == spv::Op::OpLoad) {
    Instruction* scalar_load =
        LoadScalarVar(scalar_var, extra_array_index, interface_var_user);
    loads_to_component_values->insert({interface_var_user, scalar_load});
    return true;
  }

  // Copy OpName and annotation instructions only once. Therefore, we create
  // them only for the first element of the extra array.
  if (extra_array_index && *extra_array_index != 0) return true;

  if (opcode == spv::Op::OpDecorateId || opcode == spv::Op::OpDecorateString ||
      opcode == spv::Op::OpDecorate) {
    CloneAnnotationForVariable(interface_var_user, scalar_var->result_id());
    return true;
  }

  if (opcode == spv::Op::OpName) {
    std::unique_ptr<Instruction> new_inst(interface_var_user->Clone(context()));
    new_inst->SetInOperand(0, {scalar_var->result_id()});
    context()->AddDebug2Inst(std::move(new_inst));
    return true;
  }

  if (opcode == spv::Op::OpEntryPoint) {
    return ReplaceInterfaceVarInEntryPoint(interface_var, interface_var_user,
                                           scalar_var->result_id());
  }

  if (opcode == spv::Op::OpAccessChain) {
    ReplaceAccessChainWith(interface_var_user, interface_var_component_indices,
                           scalar_var,
                           loads_for_access_chain_to_component_values);
    return true;
  }

  std::string message("Unhandled instruction");
  message += "\n  " + interface_var_user->PrettyPrint(
                          SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  message +=
      "\nfor interface variable scalar replacement\n  " +
      interface_var->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
  return false;
}

void InterfaceVariableScalarReplacement::UseBaseAccessChainForAccessChain(
    Instruction* access_chain, Instruction* base_access_chain) {
  assert(base_access_chain->opcode() == spv::Op::OpAccessChain &&
         access_chain->opcode() == spv::Op::OpAccessChain &&
         access_chain->GetSingleWordInOperand(0) ==
             base_access_chain->result_id());
  Instruction::OperandList new_operands;
  for (uint32_t i = 0; i < base_access_chain->NumInOperands(); ++i) {
    new_operands.emplace_back(base_access_chain->GetInOperand(i));
  }
  for (uint32_t i = 1; i < access_chain->NumInOperands(); ++i) {
    new_operands.emplace_back(access_chain->GetInOperand(i));
  }
  access_chain->SetInOperands(std::move(new_operands));
}

Instruction* InterfaceVariableScalarReplacement::CreateAccessChainToVar(
    uint32_t var_type_id, Instruction* var,
    const std::vector<uint32_t>& index_ids, Instruction* insert_before,
    uint32_t* component_type_id) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  *component_type_id = GetComponentTypeOfArrayMatrix(
      def_use_mgr, var_type_id, static_cast<uint32_t>(index_ids.size()));

  uint32_t ptr_type_id =
      GetPointerType(*component_type_id, GetStorageClass(var));

  std::unique_ptr<Instruction> new_access_chain(new Instruction(
      context(), spv::Op::OpAccessChain, ptr_type_id, TakeNextId(),
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {var->result_id()}}}));
  for (uint32_t index_id : index_ids) {
    new_access_chain->AddOperand({SPV_OPERAND_TYPE_ID, {index_id}});
  }

  Instruction* inst = new_access_chain.get();
  def_use_mgr->AnalyzeInstDefUse(inst);
  insert_before->InsertBefore(std::move(new_access_chain));
  return inst;
}

Instruction* InterfaceVariableScalarReplacement::CreateAccessChainWithIndex(
    uint32_t component_type_id, Instruction* var, uint32_t index,
    Instruction* insert_before) {
  uint32_t ptr_type_id =
      GetPointerType(component_type_id, GetStorageClass(var));
  uint32_t index_id = context()->get_constant_mgr()->GetUIntConstId(index);
  std::unique_ptr<Instruction> new_access_chain(new Instruction(
      context(), spv::Op::OpAccessChain, ptr_type_id, TakeNextId(),
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {var->result_id()}},
          {SPV_OPERAND_TYPE_ID, {index_id}},
      }));
  Instruction* inst = new_access_chain.get();
  context()->get_def_use_mgr()->AnalyzeInstDefUse(inst);
  insert_before->InsertBefore(std::move(new_access_chain));
  return inst;
}

void InterfaceVariableScalarReplacement::ReplaceAccessChainWith(
    Instruction* access_chain,
    const std::vector<uint32_t>& interface_var_component_indices,
    Instruction* scalar_var,
    std::unordered_map<Instruction*, Instruction*>* loads_to_component_values) {
  std::vector<uint32_t> indexes;
  for (uint32_t i = 1; i < access_chain->NumInOperands(); ++i) {
    indexes.push_back(access_chain->GetSingleWordInOperand(i));
  }

  // Note that we have a strong assumption that |access_chain| has only a single
  // index that is for the extra arrayness.
  context()->get_def_use_mgr()->ForEachUser(
      access_chain,
      [this, access_chain, &indexes, &interface_var_component_indices,
       scalar_var, loads_to_component_values](Instruction* user) {
        switch (user->opcode()) {
          case spv::Op::OpAccessChain: {
            UseBaseAccessChainForAccessChain(user, access_chain);
            ReplaceAccessChainWith(user, interface_var_component_indices,
                                   scalar_var, loads_to_component_values);
            return;
          }
          case spv::Op::OpStore: {
            uint32_t value_id = user->GetSingleWordInOperand(1);
            StoreComponentOfValueToAccessChainToScalarVar(
                value_id, interface_var_component_indices, scalar_var, indexes,
                user);
            return;
          }
          case spv::Op::OpLoad: {
            Instruction* value =
                LoadAccessChainToVar(scalar_var, indexes, user);
            loads_to_component_values->insert({user, value});
            return;
          }
          default:
            break;
        }
      });
}

void InterfaceVariableScalarReplacement::CloneAnnotationForVariable(
    Instruction* annotation_inst, uint32_t var_id) {
  assert(annotation_inst->opcode() == spv::Op::OpDecorate ||
         annotation_inst->opcode() == spv::Op::OpDecorateId ||
         annotation_inst->opcode() == spv::Op::OpDecorateString);
  std::unique_ptr<Instruction> new_inst(annotation_inst->Clone(context()));
  new_inst->SetInOperand(0, {var_id});
  context()->AddAnnotationInst(std::move(new_inst));
}

bool InterfaceVariableScalarReplacement::ReplaceInterfaceVarInEntryPoint(
    Instruction* interface_var, Instruction* entry_point,
    uint32_t scalar_var_id) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t interface_var_id = interface_var->result_id();
  if (interface_vars_removed_from_entry_point_operands_.find(
          interface_var_id) !=
      interface_vars_removed_from_entry_point_operands_.end()) {
    entry_point->AddOperand({SPV_OPERAND_TYPE_ID, {scalar_var_id}});
    def_use_mgr->AnalyzeInstUse(entry_point);
    return true;
  }

  bool success = !entry_point->WhileEachInId(
      [&interface_var_id, &scalar_var_id](uint32_t* id) {
        if (*id == interface_var_id) {
          *id = scalar_var_id;
          return false;
        }
        return true;
      });
  if (!success) {
    std::string message(
        "interface variable is not an operand of the entry point");
    message += "\n  " + interface_var->PrettyPrint(
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
    message += "\n  " + entry_point->PrettyPrint(
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
    context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
    return false;
  }

  def_use_mgr->AnalyzeInstUse(entry_point);
  interface_vars_removed_from_entry_point_operands_.insert(interface_var_id);
  return true;
}

uint32_t InterfaceVariableScalarReplacement::GetPointeeTypeIdOfVar(
    Instruction* var) {
  assert(var->opcode() == spv::Op::OpVariable);

  uint32_t ptr_type_id = var->type_id();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  Instruction* ptr_type_inst = def_use_mgr->GetDef(ptr_type_id);

  assert(ptr_type_inst->opcode() == spv::Op::OpTypePointer &&
         "Variable must have a pointer type.");
  return ptr_type_inst->GetSingleWordInOperand(kOpTypePtrTypeInOperandIndex);
}

void InterfaceVariableScalarReplacement::StoreComponentOfValueToScalarVar(
    uint32_t value_id, const std::vector<uint32_t>& component_indices,
    Instruction* scalar_var, const uint32_t* extra_array_index,
    Instruction* insert_before) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(scalar_var);
  Instruction* ptr = scalar_var;
  if (extra_array_index) {
    auto* ty_mgr = context()->get_type_mgr();
    analysis::Array* array_type = ty_mgr->GetType(component_type_id)->AsArray();
    assert(array_type != nullptr);
    component_type_id = ty_mgr->GetTypeInstruction(array_type->element_type());
    ptr = CreateAccessChainWithIndex(component_type_id, scalar_var,
                                     *extra_array_index, insert_before);
  }

  StoreComponentOfValueTo(component_type_id, value_id, component_indices, ptr,
                          extra_array_index, insert_before);
}

Instruction* InterfaceVariableScalarReplacement::LoadScalarVar(
    Instruction* scalar_var, const uint32_t* extra_array_index,
    Instruction* insert_before) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(scalar_var);
  Instruction* ptr = scalar_var;
  if (extra_array_index) {
    auto* ty_mgr = context()->get_type_mgr();
    analysis::Array* array_type = ty_mgr->GetType(component_type_id)->AsArray();
    assert(array_type != nullptr);
    component_type_id = ty_mgr->GetTypeInstruction(array_type->element_type());
    ptr = CreateAccessChainWithIndex(component_type_id, scalar_var,
                                     *extra_array_index, insert_before);
  }

  return CreateLoad(component_type_id, ptr, insert_before);
}

Instruction* InterfaceVariableScalarReplacement::CreateLoad(
    uint32_t type_id, Instruction* ptr, Instruction* insert_before) {
  std::unique_ptr<Instruction> load(
      new Instruction(context(), spv::Op::OpLoad, type_id, TakeNextId(),
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_ID, {ptr->result_id()}}}));
  Instruction* load_inst = load.get();
  context()->get_def_use_mgr()->AnalyzeInstDefUse(load_inst);
  insert_before->InsertBefore(std::move(load));
  return load_inst;
}

void InterfaceVariableScalarReplacement::StoreComponentOfValueTo(
    uint32_t component_type_id, uint32_t value_id,
    const std::vector<uint32_t>& component_indices, Instruction* ptr,
    const uint32_t* extra_array_index, Instruction* insert_before) {
  std::unique_ptr<Instruction> composite_extract(CreateCompositeExtract(
      component_type_id, value_id, component_indices, extra_array_index));

  std::unique_ptr<Instruction> new_store(
      new Instruction(context(), spv::Op::OpStore));
  new_store->AddOperand({SPV_OPERAND_TYPE_ID, {ptr->result_id()}});
  new_store->AddOperand(
      {SPV_OPERAND_TYPE_ID, {composite_extract->result_id()}});

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->AnalyzeInstDefUse(composite_extract.get());
  def_use_mgr->AnalyzeInstDefUse(new_store.get());

  insert_before->InsertBefore(std::move(composite_extract));
  insert_before->InsertBefore(std::move(new_store));
}

Instruction* InterfaceVariableScalarReplacement::CreateCompositeExtract(
    uint32_t type_id, uint32_t composite_id,
    const std::vector<uint32_t>& indexes, const uint32_t* extra_first_index) {
  uint32_t component_id = TakeNextId();
  Instruction* composite_extract = new Instruction(
      context(), spv::Op::OpCompositeExtract, type_id, component_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {composite_id}}});
  if (extra_first_index) {
    composite_extract->AddOperand(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {*extra_first_index}});
  }
  for (uint32_t index : indexes) {
    composite_extract->AddOperand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}});
  }
  return composite_extract;
}

void InterfaceVariableScalarReplacement::
    StoreComponentOfValueToAccessChainToScalarVar(
        uint32_t value_id, const std::vector<uint32_t>& component_indices,
        Instruction* scalar_var,
        const std::vector<uint32_t>& access_chain_indices,
        Instruction* insert_before) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(scalar_var);
  Instruction* ptr = scalar_var;
  if (!access_chain_indices.empty()) {
    ptr = CreateAccessChainToVar(component_type_id, scalar_var,
                                 access_chain_indices, insert_before,
                                 &component_type_id);
  }

  StoreComponentOfValueTo(component_type_id, value_id, component_indices, ptr,
                          nullptr, insert_before);
}

Instruction* InterfaceVariableScalarReplacement::LoadAccessChainToVar(
    Instruction* var, const std::vector<uint32_t>& indexes,
    Instruction* insert_before) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(var);
  Instruction* ptr = var;
  if (!indexes.empty()) {
    ptr = CreateAccessChainToVar(component_type_id, var, indexes, insert_before,
                                 &component_type_id);
  }

  return CreateLoad(component_type_id, ptr, insert_before);
}

Instruction*
InterfaceVariableScalarReplacement::CreateCompositeConstructForComponentOfLoad(
    Instruction* load, uint32_t depth_to_component) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t type_id = load->type_id();
  if (depth_to_component != 0) {
    type_id = GetComponentTypeOfArrayMatrix(def_use_mgr, load->type_id(),
                                            depth_to_component);
  }
  uint32_t new_id = context()->TakeNextId();
  std::unique_ptr<Instruction> new_composite_construct(new Instruction(
      context(), spv::Op::OpCompositeConstruct, type_id, new_id, {}));
  Instruction* composite_construct = new_composite_construct.get();
  def_use_mgr->AnalyzeInstDefUse(composite_construct);

  // Insert |new_composite_construct| after |load|. When there are multiple
  // recursive composite construct instructions for a load, we have to place the
  // composite construct with a lower depth later because it constructs the
  // composite that contains other composites with lower depths.
  auto* insert_before = load->NextNode();
  while (true) {
    auto itr =
        composite_ids_to_component_depths.find(insert_before->result_id());
    if (itr == composite_ids_to_component_depths.end()) break;
    if (itr->second <= depth_to_component) break;
    insert_before = insert_before->NextNode();
  }
  insert_before->InsertBefore(std::move(new_composite_construct));
  composite_ids_to_component_depths.insert({new_id, depth_to_component});
  return composite_construct;
}

void InterfaceVariableScalarReplacement::AddComponentsToCompositesForLoads(
    const std::unordered_map<Instruction*, Instruction*>&
        loads_to_component_values,
    std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
    uint32_t depth_to_component) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  for (auto& load_and_component_vale : loads_to_component_values) {
    Instruction* load = load_and_component_vale.first;
    Instruction* component_value = load_and_component_vale.second;
    Instruction* composite_construct = nullptr;
    auto itr = loads_to_composites->find(load);
    if (itr == loads_to_composites->end()) {
      composite_construct =
          CreateCompositeConstructForComponentOfLoad(load, depth_to_component);
      loads_to_composites->insert({load, composite_construct});
    } else {
      composite_construct = itr->second;
    }
    composite_construct->AddOperand(
        {SPV_OPERAND_TYPE_ID, {component_value->result_id()}});
    def_use_mgr->AnalyzeInstDefUse(composite_construct);
  }
}

uint32_t InterfaceVariableScalarReplacement::GetArrayType(
    uint32_t elem_type_id, uint32_t array_length) {
  analysis::Type* elem_type = context()->get_type_mgr()->GetType(elem_type_id);
  uint32_t array_length_id =
      context()->get_constant_mgr()->GetUIntConstId(array_length);
  analysis::Array array_type(
      elem_type,
      analysis::Array::LengthInfo{array_length_id, {0, array_length}});
  return context()->get_type_mgr()->GetTypeInstruction(&array_type);
}

uint32_t InterfaceVariableScalarReplacement::GetPointerType(
    uint32_t type_id, spv::StorageClass storage_class) {
  analysis::Type* type = context()->get_type_mgr()->GetType(type_id);
  analysis::Pointer ptr_type(type, storage_class);
  return context()->get_type_mgr()->GetTypeInstruction(&ptr_type);
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateScalarInterfaceVarsForArray(
    Instruction* interface_var_type, spv::StorageClass storage_class,
    uint32_t extra_array_length) {
  assert(interface_var_type->opcode() == spv::Op::OpTypeArray);

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t array_length = GetArrayLength(def_use_mgr, interface_var_type);
  Instruction* elem_type = GetArrayElementType(def_use_mgr, interface_var_type);

  NestedCompositeComponents scalar_vars;
  while (array_length > 0) {
    NestedCompositeComponents scalar_vars_for_element =
        CreateScalarInterfaceVarsForReplacement(elem_type, storage_class,
                                                extra_array_length);
    scalar_vars.AddComponent(scalar_vars_for_element);
    --array_length;
  }
  return scalar_vars;
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateScalarInterfaceVarsForMatrix(
    Instruction* interface_var_type, spv::StorageClass storage_class,
    uint32_t extra_array_length) {
  assert(interface_var_type->opcode() == spv::Op::OpTypeMatrix);

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t column_count = interface_var_type->GetSingleWordInOperand(
      kOpTypeMatrixColCountInOperandIndex);
  Instruction* column_type =
      GetMatrixColumnType(def_use_mgr, interface_var_type);

  NestedCompositeComponents scalar_vars;
  while (column_count > 0) {
    NestedCompositeComponents scalar_vars_for_column =
        CreateScalarInterfaceVarsForReplacement(column_type, storage_class,
                                                extra_array_length);
    scalar_vars.AddComponent(scalar_vars_for_column);
    --column_count;
  }
  return scalar_vars;
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateScalarInterfaceVarsForReplacement(
    Instruction* interface_var_type, spv::StorageClass storage_class,
    uint32_t extra_array_length) {
  // Handle array case.
  if (interface_var_type->opcode() == spv::Op::OpTypeArray) {
    return CreateScalarInterfaceVarsForArray(interface_var_type, storage_class,
                                             extra_array_length);
  }

  // Handle matrix case.
  if (interface_var_type->opcode() == spv::Op::OpTypeMatrix) {
    return CreateScalarInterfaceVarsForMatrix(interface_var_type, storage_class,
                                              extra_array_length);
  }

  // Handle scalar or vector case.
  NestedCompositeComponents scalar_var;
  uint32_t type_id = interface_var_type->result_id();
  if (extra_array_length != 0) {
    type_id = GetArrayType(type_id, extra_array_length);
  }
  uint32_t ptr_type_id =
      context()->get_type_mgr()->FindPointerToType(type_id, storage_class);
  uint32_t id = TakeNextId();
  std::unique_ptr<Instruction> variable(
      new Instruction(context(), spv::Op::OpVariable, ptr_type_id, id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_STORAGE_CLASS,
                           {static_cast<uint32_t>(storage_class)}}}));
  scalar_var.SetSingleComponentVariable(variable.get());
  context()->AddGlobalValue(std::move(variable));
  return scalar_var;
}

Instruction* InterfaceVariableScalarReplacement::GetTypeOfVariable(
    Instruction* var) {
  uint32_t pointee_type_id = GetPointeeTypeIdOfVar(var);
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  return def_use_mgr->GetDef(pointee_type_id);
}

Pass::Status InterfaceVariableScalarReplacement::Process() {
  Pass::Status status = Status::SuccessWithoutChange;
  for (Instruction& entry_point : get_module()->entry_points()) {
    status =
        CombineStatus(status, ReplaceInterfaceVarsWithScalars(entry_point));
  }
  return status;
}

bool InterfaceVariableScalarReplacement::
    ReportErrorIfHasExtraArraynessForOtherEntry(Instruction* var) {
  if (vars_with_extra_arrayness.find(var) == vars_with_extra_arrayness.end())
    return false;

  std::string message(
      "A variable is arrayed for an entry point but it is not "
      "arrayed for another entry point");
  message +=
      "\n  " + var->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
  return true;
}

bool InterfaceVariableScalarReplacement::
    ReportErrorIfHasNoExtraArraynessForOtherEntry(Instruction* var) {
  if (vars_without_extra_arrayness.find(var) ==
      vars_without_extra_arrayness.end())
    return false;

  std::string message(
      "A variable is not arrayed for an entry point but it is "
      "arrayed for another entry point");
  message +=
      "\n  " + var->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
  return true;
}

Pass::Status
InterfaceVariableScalarReplacement::ReplaceInterfaceVarsWithScalars(
    Instruction& entry_point) {
  std::vector<Instruction*> interface_vars =
      CollectInterfaceVariables(entry_point);

  Pass::Status status = Status::SuccessWithoutChange;
  for (Instruction* interface_var : interface_vars) {
    uint32_t location, component;
    if (!GetVariableLocation(interface_var, &location)) continue;
    if (!GetVariableComponent(interface_var, &component)) component = 0;

    Instruction* interface_var_type = GetTypeOfVariable(interface_var);
    uint32_t extra_array_length = 0;
    if (HasExtraArrayness(entry_point, interface_var)) {
      extra_array_length =
          GetArrayLength(context()->get_def_use_mgr(), interface_var_type);
      interface_var_type =
          GetArrayElementType(context()->get_def_use_mgr(), interface_var_type);
      vars_with_extra_arrayness.insert(interface_var);
    } else {
      vars_without_extra_arrayness.insert(interface_var);
    }

    if (!CheckExtraArraynessConflictBetweenEntries(interface_var,
                                                   extra_array_length != 0)) {
      return Pass::Status::Failure;
    }

    if (interface_var_type->opcode() != spv::Op::OpTypeArray &&
        interface_var_type->opcode() != spv::Op::OpTypeMatrix) {
      continue;
    }

    if (!ReplaceInterfaceVariableWithScalars(interface_var, interface_var_type,
                                             location, component,
                                             extra_array_length)) {
      return Pass::Status::Failure;
    }
    status = Pass::Status::SuccessWithChange;
  }

  return status;
}

}  // namespace opt
}  // namespace spvtools
