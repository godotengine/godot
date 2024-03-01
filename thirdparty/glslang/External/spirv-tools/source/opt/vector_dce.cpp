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

#include "source/opt/vector_dce.h"

#include <utility>

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kExtractCompositeIdInIdx = 0;
constexpr uint32_t kInsertObjectIdInIdx = 0;
constexpr uint32_t kInsertCompositeIdInIdx = 1;
}  // namespace

Pass::Status VectorDCE::Process() {
  bool modified = false;
  for (Function& function : *get_module()) {
    modified |= VectorDCEFunction(&function);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool VectorDCE::VectorDCEFunction(Function* function) {
  LiveComponentMap live_components;
  FindLiveComponents(function, &live_components);
  return RewriteInstructions(function, live_components);
}

void VectorDCE::FindLiveComponents(Function* function,
                                   LiveComponentMap* live_components) {
  std::vector<WorkListItem> work_list;

  // Prime the work list.  We will assume that any instruction that does
  // not result in a vector is live.
  //
  // Extending to structures and matrices is not as straight forward because of
  // the nesting.  We cannot simply us a bit vector to keep track of which
  // components are live because of arbitrary nesting of structs.
  function->ForEachInst(
      [&work_list, this, live_components](Instruction* current_inst) {
        if (current_inst->IsCommonDebugInstr()) {
          return;
        }
        if (!HasVectorOrScalarResult(current_inst) ||
            !context()->IsCombinatorInstruction(current_inst)) {
          MarkUsesAsLive(current_inst, all_components_live_, live_components,
                         &work_list);
        }
      });

  // Process the work list propagating liveness.
  for (uint32_t i = 0; i < work_list.size(); i++) {
    WorkListItem current_item = work_list[i];
    Instruction* current_inst = current_item.instruction;

    switch (current_inst->opcode()) {
      case spv::Op::OpCompositeExtract:
        MarkExtractUseAsLive(current_inst, current_item.components,
                             live_components, &work_list);
        break;
      case spv::Op::OpCompositeInsert:
        MarkInsertUsesAsLive(current_item, live_components, &work_list);
        break;
      case spv::Op::OpVectorShuffle:
        MarkVectorShuffleUsesAsLive(current_item, live_components, &work_list);
        break;
      case spv::Op::OpCompositeConstruct:
        MarkCompositeContructUsesAsLive(current_item, live_components,
                                        &work_list);
        break;
      default:
        if (current_inst->IsScalarizable()) {
          MarkUsesAsLive(current_inst, current_item.components, live_components,
                         &work_list);
        } else {
          MarkUsesAsLive(current_inst, all_components_live_, live_components,
                         &work_list);
        }
        break;
    }
  }
}

void VectorDCE::MarkExtractUseAsLive(const Instruction* current_inst,
                                     const utils::BitVector& live_elements,
                                     LiveComponentMap* live_components,
                                     std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t operand_id =
      current_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
  Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

  if (HasVectorOrScalarResult(operand_inst)) {
    WorkListItem new_item;
    new_item.instruction = operand_inst;
    if (current_inst->NumInOperands() < 2) {
      new_item.components = live_elements;
    } else {
      uint32_t element_index = current_inst->GetSingleWordInOperand(1);
      uint32_t item_size = GetVectorComponentCount(operand_inst->type_id());
      if (element_index < item_size) {
        new_item.components.Set(element_index);
      }
    }
    AddItemToWorkListIfNeeded(new_item, live_components, work_list);
  }
}

void VectorDCE::MarkInsertUsesAsLive(
    const VectorDCE::WorkListItem& current_item,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  if (current_item.instruction->NumInOperands() > 2) {
    uint32_t insert_position =
        current_item.instruction->GetSingleWordInOperand(2);

    // Add the elements of the composite object that are used.
    uint32_t operand_id = current_item.instruction->GetSingleWordInOperand(
        kInsertCompositeIdInIdx);
    Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

    WorkListItem new_item;
    new_item.instruction = operand_inst;
    new_item.components = current_item.components;
    new_item.components.Clear(insert_position);

    AddItemToWorkListIfNeeded(new_item, live_components, work_list);

    // Add the element being inserted if it is used.
    if (current_item.components.Get(insert_position)) {
      uint32_t obj_operand_id =
          current_item.instruction->GetSingleWordInOperand(
              kInsertObjectIdInIdx);
      Instruction* obj_operand_inst = def_use_mgr->GetDef(obj_operand_id);
      WorkListItem new_item_for_obj;
      new_item_for_obj.instruction = obj_operand_inst;
      new_item_for_obj.components.Set(0);
      AddItemToWorkListIfNeeded(new_item_for_obj, live_components, work_list);
    }
  } else {
    // If there are no indices, then this is a copy of the object being
    // inserted.
    uint32_t object_id =
        current_item.instruction->GetSingleWordInOperand(kInsertObjectIdInIdx);
    Instruction* object_inst = def_use_mgr->GetDef(object_id);

    WorkListItem new_item;
    new_item.instruction = object_inst;
    new_item.components = current_item.components;
    AddItemToWorkListIfNeeded(new_item, live_components, work_list);
  }
}

void VectorDCE::MarkVectorShuffleUsesAsLive(
    const WorkListItem& current_item,
    VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  WorkListItem first_operand;
  first_operand.instruction =
      def_use_mgr->GetDef(current_item.instruction->GetSingleWordInOperand(0));
  WorkListItem second_operand;
  second_operand.instruction =
      def_use_mgr->GetDef(current_item.instruction->GetSingleWordInOperand(1));

  uint32_t size_of_first_operand =
      GetVectorComponentCount(first_operand.instruction->type_id());
  uint32_t size_of_second_operand =
      GetVectorComponentCount(second_operand.instruction->type_id());

  for (uint32_t in_op = 2; in_op < current_item.instruction->NumInOperands();
       ++in_op) {
    uint32_t index = current_item.instruction->GetSingleWordInOperand(in_op);
    if (current_item.components.Get(in_op - 2)) {
      if (index < size_of_first_operand) {
        first_operand.components.Set(index);
      } else if (index - size_of_first_operand < size_of_second_operand) {
        second_operand.components.Set(index - size_of_first_operand);
      }
    }
  }

  AddItemToWorkListIfNeeded(first_operand, live_components, work_list);
  AddItemToWorkListIfNeeded(second_operand, live_components, work_list);
}

void VectorDCE::MarkCompositeContructUsesAsLive(
    VectorDCE::WorkListItem work_item,
    VectorDCE::LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  uint32_t current_component = 0;
  Instruction* current_inst = work_item.instruction;
  uint32_t num_in_operands = current_inst->NumInOperands();
  for (uint32_t i = 0; i < num_in_operands; ++i) {
    uint32_t id = current_inst->GetSingleWordInOperand(i);
    Instruction* op_inst = def_use_mgr->GetDef(id);

    if (HasScalarResult(op_inst)) {
      WorkListItem new_work_item;
      new_work_item.instruction = op_inst;
      if (work_item.components.Get(current_component)) {
        new_work_item.components.Set(0);
      }
      AddItemToWorkListIfNeeded(new_work_item, live_components, work_list);
      current_component++;
    } else {
      assert(HasVectorResult(op_inst));
      WorkListItem new_work_item;
      new_work_item.instruction = op_inst;
      uint32_t op_vector_size = GetVectorComponentCount(op_inst->type_id());

      for (uint32_t op_vector_idx = 0; op_vector_idx < op_vector_size;
           op_vector_idx++, current_component++) {
        if (work_item.components.Get(current_component)) {
          new_work_item.components.Set(op_vector_idx);
        }
      }
      AddItemToWorkListIfNeeded(new_work_item, live_components, work_list);
    }
  }
}

void VectorDCE::MarkUsesAsLive(
    Instruction* current_inst, const utils::BitVector& live_elements,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  current_inst->ForEachInId([&work_list, &live_elements, this, live_components,
                             def_use_mgr](uint32_t* operand_id) {
    Instruction* operand_inst = def_use_mgr->GetDef(*operand_id);

    if (HasVectorResult(operand_inst)) {
      WorkListItem new_item;
      new_item.instruction = operand_inst;
      new_item.components = live_elements;
      AddItemToWorkListIfNeeded(new_item, live_components, work_list);
    } else if (HasScalarResult(operand_inst)) {
      WorkListItem new_item;
      new_item.instruction = operand_inst;
      new_item.components.Set(0);
      AddItemToWorkListIfNeeded(new_item, live_components, work_list);
    }
  });
}

bool VectorDCE::HasVectorOrScalarResult(const Instruction* inst) const {
  return HasScalarResult(inst) || HasVectorResult(inst);
}

bool VectorDCE::HasVectorResult(const Instruction* inst) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  if (inst->type_id() == 0) {
    return false;
  }

  const analysis::Type* current_type = type_mgr->GetType(inst->type_id());
  switch (current_type->kind()) {
    case analysis::Type::kVector:
      return true;
    default:
      return false;
  }
}

bool VectorDCE::HasScalarResult(const Instruction* inst) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  if (inst->type_id() == 0) {
    return false;
  }

  const analysis::Type* current_type = type_mgr->GetType(inst->type_id());
  switch (current_type->kind()) {
    case analysis::Type::kBool:
    case analysis::Type::kInteger:
    case analysis::Type::kFloat:
      return true;
    default:
      return false;
  }
}

uint32_t VectorDCE::GetVectorComponentCount(uint32_t type_id) {
  assert(type_id != 0 &&
         "Trying to get the vector element count, but the type id is 0");
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  const analysis::Type* type = type_mgr->GetType(type_id);
  const analysis::Vector* vector_type = type->AsVector();
  assert(
      vector_type &&
      "Trying to get the vector element count, but the type is not a vector");
  return vector_type->element_count();
}

bool VectorDCE::RewriteInstructions(
    Function* function, const VectorDCE::LiveComponentMap& live_components) {
  bool modified = false;

  // Kill DebugValue in the middle of the instruction iteration will result
  // in accessing a dangling pointer. We keep dead DebugValue instructions
  // in |dead_dbg_value| to kill them once after the iteration.
  std::vector<Instruction*> dead_dbg_value;

  function->ForEachInst([&modified, this, live_components,
                         &dead_dbg_value](Instruction* current_inst) {
    if (!context()->IsCombinatorInstruction(current_inst)) {
      return;
    }

    auto live_component = live_components.find(current_inst->result_id());
    if (live_component == live_components.end()) {
      // If this instruction is not in live_components then it does not
      // produce a vector, or it is never referenced and ADCE will remove
      // it.  No point in trying to differentiate.
      return;
    }

    // If no element in the current instruction is used replace it with an
    // OpUndef.
    if (live_component->second.Empty()) {
      modified = true;
      MarkDebugValueUsesAsDead(current_inst, &dead_dbg_value);
      uint32_t undef_id = this->Type2Undef(current_inst->type_id());
      context()->KillNamesAndDecorates(current_inst);
      context()->ReplaceAllUsesWith(current_inst->result_id(), undef_id);
      context()->KillInst(current_inst);
      return;
    }

    switch (current_inst->opcode()) {
      case spv::Op::OpCompositeInsert:
        modified |= RewriteInsertInstruction(
            current_inst, live_component->second, &dead_dbg_value);
        break;
      case spv::Op::OpCompositeConstruct:
        // TODO: The members that are not live can be replaced by an undef
        // or constant. This will remove uses of those values, and possibly
        // create opportunities for ADCE.
        break;
      default:
        // Do nothing.
        break;
    }
  });
  for (auto* i : dead_dbg_value) context()->KillInst(i);
  return modified;
}

bool VectorDCE::RewriteInsertInstruction(
    Instruction* current_inst, const utils::BitVector& live_components,
    std::vector<Instruction*>* dead_dbg_value) {
  // If the value being inserted is not live, then we can skip the insert.

  if (current_inst->NumInOperands() == 2) {
    // If there are no indices, then this is the same as a copy.
    context()->KillNamesAndDecorates(current_inst->result_id());
    uint32_t object_id =
        current_inst->GetSingleWordInOperand(kInsertObjectIdInIdx);
    context()->ReplaceAllUsesWith(current_inst->result_id(), object_id);
    return true;
  }

  uint32_t insert_index = current_inst->GetSingleWordInOperand(2);
  if (!live_components.Get(insert_index)) {
    MarkDebugValueUsesAsDead(current_inst, dead_dbg_value);
    context()->KillNamesAndDecorates(current_inst->result_id());
    uint32_t composite_id =
        current_inst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
    context()->ReplaceAllUsesWith(current_inst->result_id(), composite_id);
    return true;
  }

  // If the values already in the composite are not used, then replace it with
  // an undef.
  utils::BitVector temp = live_components;
  temp.Clear(insert_index);
  if (temp.Empty()) {
    context()->ForgetUses(current_inst);
    uint32_t undef_id = Type2Undef(current_inst->type_id());
    current_inst->SetInOperand(kInsertCompositeIdInIdx, {undef_id});
    context()->AnalyzeUses(current_inst);
    return true;
  }

  return false;
}

void VectorDCE::MarkDebugValueUsesAsDead(
    Instruction* composite, std::vector<Instruction*>* dead_dbg_value) {
  context()->get_def_use_mgr()->ForEachUser(
      composite, [&dead_dbg_value](Instruction* use) {
        if (use->GetCommonDebugOpcode() == CommonDebugInfoDebugValue)
          dead_dbg_value->push_back(use);
      });
}

void VectorDCE::AddItemToWorkListIfNeeded(
    WorkListItem work_item, VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  Instruction* current_inst = work_item.instruction;
  auto it = live_components->find(current_inst->result_id());
  if (it == live_components->end()) {
    live_components->emplace(
        std::make_pair(current_inst->result_id(), work_item.components));
    work_list->emplace_back(work_item);
  } else {
    if (it->second.Or(work_item.components)) {
      work_list->emplace_back(work_item);
    }
  }
}

}  // namespace opt
}  // namespace spvtools
