// Copyright (c) 2019 Google LLC
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

#include "source/opt/desc_sroa.h"

#include "source/opt/desc_sroa_util.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {
namespace {

bool IsDecorationBinding(Instruction* inst) {
  if (inst->opcode() != spv::Op::OpDecorate) return false;
  return spv::Decoration(inst->GetSingleWordInOperand(1u)) ==
         spv::Decoration::Binding;
}

}  // namespace

Pass::Status DescriptorScalarReplacement::Process() {
  bool modified = false;

  std::vector<Instruction*> vars_to_kill;

  for (Instruction& var : context()->types_values()) {
    if (descsroautil::IsDescriptorArray(context(), &var)) {
      modified = true;
      if (!ReplaceCandidate(&var)) {
        return Status::Failure;
      }
      vars_to_kill.push_back(&var);
    }
  }

  for (Instruction* var : vars_to_kill) {
    context()->KillInst(var);
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool DescriptorScalarReplacement::ReplaceCandidate(Instruction* var) {
  std::vector<Instruction*> access_chain_work_list;
  std::vector<Instruction*> load_work_list;
  bool failed = !get_def_use_mgr()->WhileEachUser(
      var->result_id(),
      [this, &access_chain_work_list, &load_work_list](Instruction* use) {
        if (use->opcode() == spv::Op::OpName) {
          return true;
        }

        if (use->IsDecoration()) {
          return true;
        }

        switch (use->opcode()) {
          case spv::Op::OpAccessChain:
          case spv::Op::OpInBoundsAccessChain:
            access_chain_work_list.push_back(use);
            return true;
          case spv::Op::OpLoad:
            load_work_list.push_back(use);
            return true;
          default:
            context()->EmitErrorMessage(
                "Variable cannot be replaced: invalid instruction", use);
            return false;
        }
        return true;
      });

  if (failed) {
    return false;
  }

  for (Instruction* use : access_chain_work_list) {
    if (!ReplaceAccessChain(var, use)) {
      return false;
    }
  }
  for (Instruction* use : load_work_list) {
    if (!ReplaceLoadedValue(var, use)) {
      return false;
    }
  }
  return true;
}

bool DescriptorScalarReplacement::ReplaceAccessChain(Instruction* var,
                                                     Instruction* use) {
  if (use->NumInOperands() <= 1) {
    context()->EmitErrorMessage(
        "Variable cannot be replaced: invalid instruction", use);
    return false;
  }

  const analysis::Constant* const_index =
      descsroautil::GetAccessChainIndexAsConst(context(), use);
  if (const_index == nullptr) {
    context()->EmitErrorMessage("Variable cannot be replaced: invalid index",
                                use);
    return false;
  }

  uint32_t idx = const_index->GetU32();
  uint32_t replacement_var = GetReplacementVariable(var, idx);

  if (use->NumInOperands() == 2) {
    // We are not indexing into the replacement variable.  We can replaces the
    // access chain with the replacement variable itself.
    context()->ReplaceAllUsesWith(use->result_id(), replacement_var);
    context()->KillInst(use);
    return true;
  }

  // We need to build a new access chain with the replacement variable as the
  // base address.
  Instruction::OperandList new_operands;

  // Same result id and result type.
  new_operands.emplace_back(use->GetOperand(0));
  new_operands.emplace_back(use->GetOperand(1));

  // Use the replacement variable as the base address.
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {replacement_var}});

  // Drop the first index because it is consumed by the replacement, and copy
  // the rest.
  for (uint32_t i = 4; i < use->NumOperands(); i++) {
    new_operands.emplace_back(use->GetOperand(i));
  }

  use->ReplaceOperands(new_operands);
  context()->UpdateDefUse(use);
  return true;
}

uint32_t DescriptorScalarReplacement::GetReplacementVariable(Instruction* var,
                                                             uint32_t idx) {
  auto replacement_vars = replacement_variables_.find(var);
  if (replacement_vars == replacement_variables_.end()) {
    uint32_t number_of_elements =
        descsroautil::GetNumberOfElementsForArrayOrStruct(context(), var);
    replacement_vars =
        replacement_variables_
            .insert({var, std::vector<uint32_t>(number_of_elements, 0)})
            .first;
  }

  if (replacement_vars->second[idx] == 0) {
    replacement_vars->second[idx] = CreateReplacementVariable(var, idx);
  }

  return replacement_vars->second[idx];
}

void DescriptorScalarReplacement::CopyDecorationsForNewVariable(
    Instruction* old_var, uint32_t index, uint32_t new_var_id,
    uint32_t new_var_ptr_type_id, const bool is_old_var_array,
    const bool is_old_var_struct, Instruction* old_var_type) {
  // Handle OpDecorate and OpDecorateString instructions.
  for (auto old_decoration :
       get_decoration_mgr()->GetDecorationsFor(old_var->result_id(), true)) {
    uint32_t new_binding = 0;
    if (IsDecorationBinding(old_decoration)) {
      new_binding = GetNewBindingForElement(
          old_decoration->GetSingleWordInOperand(2), index, new_var_ptr_type_id,
          is_old_var_array, is_old_var_struct, old_var_type);
    }
    CreateNewDecorationForNewVariable(old_decoration, new_var_id, new_binding);
  }

  // Handle OpMemberDecorate instructions.
  for (auto old_decoration : get_decoration_mgr()->GetDecorationsFor(
           old_var_type->result_id(), true)) {
    assert(old_decoration->opcode() == spv::Op::OpMemberDecorate);
    if (old_decoration->GetSingleWordInOperand(1u) != index) continue;
    CreateNewDecorationForMemberDecorate(old_decoration, new_var_id);
  }
}

uint32_t DescriptorScalarReplacement::GetNewBindingForElement(
    uint32_t old_binding, uint32_t index, uint32_t new_var_ptr_type_id,
    const bool is_old_var_array, const bool is_old_var_struct,
    Instruction* old_var_type) {
  if (is_old_var_array) {
    return old_binding + index * GetNumBindingsUsedByType(new_var_ptr_type_id);
  }
  if (is_old_var_struct) {
    // The binding offset that should be added is the sum of binding
    // numbers used by previous members of the current struct.
    uint32_t new_binding = old_binding;
    for (uint32_t i = 0; i < index; ++i) {
      new_binding +=
          GetNumBindingsUsedByType(old_var_type->GetSingleWordInOperand(i));
    }
    return new_binding;
  }
  return old_binding;
}

void DescriptorScalarReplacement::CreateNewDecorationForNewVariable(
    Instruction* old_decoration, uint32_t new_var_id, uint32_t new_binding) {
  assert(old_decoration->opcode() == spv::Op::OpDecorate ||
         old_decoration->opcode() == spv::Op::OpDecorateString);
  std::unique_ptr<Instruction> new_decoration(old_decoration->Clone(context()));
  new_decoration->SetInOperand(0, {new_var_id});

  if (IsDecorationBinding(new_decoration.get())) {
    new_decoration->SetInOperand(2, {new_binding});
  }
  context()->AddAnnotationInst(std::move(new_decoration));
}

void DescriptorScalarReplacement::CreateNewDecorationForMemberDecorate(
    Instruction* old_member_decoration, uint32_t new_var_id) {
  std::vector<Operand> operands(
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {new_var_id}}});
  auto new_decorate_operand_begin = old_member_decoration->begin() + 2u;
  auto new_decorate_operand_end = old_member_decoration->end();
  operands.insert(operands.end(), new_decorate_operand_begin,
                  new_decorate_operand_end);
  get_decoration_mgr()->AddDecoration(spv::Op::OpDecorate, std::move(operands));
}

uint32_t DescriptorScalarReplacement::CreateReplacementVariable(
    Instruction* var, uint32_t idx) {
  // The storage class for the new variable is the same as the original.
  spv::StorageClass storage_class =
      static_cast<spv::StorageClass>(var->GetSingleWordInOperand(0));

  // The type for the new variable will be a pointer to type of the elements of
  // the array.
  uint32_t ptr_type_id = var->type_id();
  Instruction* ptr_type_inst = get_def_use_mgr()->GetDef(ptr_type_id);
  assert(ptr_type_inst->opcode() == spv::Op::OpTypePointer &&
         "Variable should be a pointer to an array or structure.");
  uint32_t pointee_type_id = ptr_type_inst->GetSingleWordInOperand(1);
  Instruction* pointee_type_inst = get_def_use_mgr()->GetDef(pointee_type_id);
  const bool is_array = pointee_type_inst->opcode() == spv::Op::OpTypeArray;
  const bool is_struct = pointee_type_inst->opcode() == spv::Op::OpTypeStruct;
  assert((is_array || is_struct) &&
         "Variable should be a pointer to an array or structure.");

  uint32_t element_type_id =
      is_array ? pointee_type_inst->GetSingleWordInOperand(0)
               : pointee_type_inst->GetSingleWordInOperand(idx);

  uint32_t ptr_element_type_id = context()->get_type_mgr()->FindPointerToType(
      element_type_id, storage_class);

  // Create the variable.
  uint32_t id = TakeNextId();
  std::unique_ptr<Instruction> variable(
      new Instruction(context(), spv::Op::OpVariable, ptr_element_type_id, id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_STORAGE_CLASS,
                           {static_cast<uint32_t>(storage_class)}}}));
  context()->AddGlobalValue(std::move(variable));

  CopyDecorationsForNewVariable(var, idx, id, ptr_element_type_id, is_array,
                                is_struct, pointee_type_inst);

  // Create a new OpName for the replacement variable.
  std::vector<std::unique_ptr<Instruction>> names_to_add;
  for (auto p : context()->GetNames(var->result_id())) {
    Instruction* name_inst = p.second;
    std::string name_str = utils::MakeString(name_inst->GetOperand(1).words);
    if (is_array) {
      name_str += "[" + utils::ToString(idx) + "]";
    }
    if (is_struct) {
      Instruction* member_name_inst =
          context()->GetMemberName(pointee_type_inst->result_id(), idx);
      name_str += ".";
      if (member_name_inst)
        name_str += utils::MakeString(member_name_inst->GetOperand(2).words);
      else
        // In case the member does not have a name assigned to it, use the
        // member index.
        name_str += utils::ToString(idx);
    }

    std::unique_ptr<Instruction> new_name(new Instruction(
        context(), spv::Op::OpName, 0, 0,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_ID, {id}},
            {SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector(name_str)}}));
    Instruction* new_name_inst = new_name.get();
    get_def_use_mgr()->AnalyzeInstDefUse(new_name_inst);
    names_to_add.push_back(std::move(new_name));
  }

  // We shouldn't add the new names when we are iterating over name ranges
  // above. We can add all the new names now.
  for (auto& new_name : names_to_add)
    context()->AddDebug2Inst(std::move(new_name));

  return id;
}

uint32_t DescriptorScalarReplacement::GetNumBindingsUsedByType(
    uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);

  // If it's a pointer, look at the underlying type.
  if (type_inst->opcode() == spv::Op::OpTypePointer) {
    type_id = type_inst->GetSingleWordInOperand(1);
    type_inst = get_def_use_mgr()->GetDef(type_id);
  }

  // Arrays consume N*M binding numbers where N is the array length, and M is
  // the number of bindings used by each array element.
  if (type_inst->opcode() == spv::Op::OpTypeArray) {
    uint32_t element_type_id = type_inst->GetSingleWordInOperand(0);
    uint32_t length_id = type_inst->GetSingleWordInOperand(1);
    const analysis::Constant* length_const =
        context()->get_constant_mgr()->FindDeclaredConstant(length_id);
    // OpTypeArray's length must always be a constant
    assert(length_const != nullptr);
    uint32_t num_elems = length_const->GetU32();
    return num_elems * GetNumBindingsUsedByType(element_type_id);
  }

  // The number of bindings consumed by a structure is the sum of the bindings
  // used by its members.
  if (type_inst->opcode() == spv::Op::OpTypeStruct &&
      !descsroautil::IsTypeOfStructuredBuffer(context(), type_inst)) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < type_inst->NumInOperands(); i++)
      sum += GetNumBindingsUsedByType(type_inst->GetSingleWordInOperand(i));
    return sum;
  }

  // All other types are considered to take up 1 binding number.
  return 1;
}

bool DescriptorScalarReplacement::ReplaceLoadedValue(Instruction* var,
                                                     Instruction* value) {
  // |var| is the global variable that has to be eliminated (OpVariable).
  // |value| is the OpLoad instruction that has loaded |var|.
  // The function expects all users of |value| to be OpCompositeExtract
  // instructions. Otherwise the function returns false with an error message.
  assert(value->opcode() == spv::Op::OpLoad);
  assert(value->GetSingleWordInOperand(0) == var->result_id());
  std::vector<Instruction*> work_list;
  bool failed = !get_def_use_mgr()->WhileEachUser(
      value->result_id(), [this, &work_list](Instruction* use) {
        if (use->opcode() != spv::Op::OpCompositeExtract) {
          context()->EmitErrorMessage(
              "Variable cannot be replaced: invalid instruction", use);
          return false;
        }
        work_list.push_back(use);
        return true;
      });

  if (failed) {
    return false;
  }

  for (Instruction* use : work_list) {
    if (!ReplaceCompositeExtract(var, use)) {
      return false;
    }
  }

  // All usages of the loaded value have been killed. We can kill the OpLoad.
  context()->KillInst(value);
  return true;
}

bool DescriptorScalarReplacement::ReplaceCompositeExtract(
    Instruction* var, Instruction* extract) {
  assert(extract->opcode() == spv::Op::OpCompositeExtract);
  // We're currently only supporting extractions of one index at a time. If we
  // need to, we can handle cases with multiple indexes in the future.
  if (extract->NumInOperands() != 2) {
    context()->EmitErrorMessage(
        "Variable cannot be replaced: invalid instruction", extract);
    return false;
  }

  uint32_t replacement_var =
      GetReplacementVariable(var, extract->GetSingleWordInOperand(1));

  // The result type of the OpLoad is the same as the result type of the
  // OpCompositeExtract.
  uint32_t load_id = TakeNextId();
  std::unique_ptr<Instruction> load(
      new Instruction(context(), spv::Op::OpLoad, extract->type_id(), load_id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_ID, {replacement_var}}}));
  Instruction* load_instr = load.get();
  get_def_use_mgr()->AnalyzeInstDefUse(load_instr);
  context()->set_instr_block(load_instr, context()->get_instr_block(extract));
  extract->InsertBefore(std::move(load));
  context()->ReplaceAllUsesWith(extract->result_id(), load_id);
  context()->KillInst(extract);
  return true;
}

}  // namespace opt
}  // namespace spvtools
