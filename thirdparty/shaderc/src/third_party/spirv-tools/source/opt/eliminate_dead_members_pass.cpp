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

#include "source/opt/eliminate_dead_members_pass.h"

#include "ir_builder.h"
#include "source/opt/ir_context.h"

namespace {
const uint32_t kRemovedMember = 0xFFFFFFFF;
}

namespace spvtools {
namespace opt {

Pass::Status EliminateDeadMembersPass::Process() {
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  FindLiveMembers();
  if (RemoveDeadMembers()) {
    return Status::SuccessWithChange;
  }
  return Status::SuccessWithoutChange;
}

void EliminateDeadMembersPass::FindLiveMembers() {
  // Until we have implemented the rewritting of OpSpecConsantOp instructions,
  // we have to mark them as fully used just to be safe.
  for (auto& inst : get_module()->types_values()) {
    if (inst.opcode() == SpvOpSpecConstantOp) {
      MarkTypeAsFullyUsed(inst.type_id());
    } else if (inst.opcode() == SpvOpVariable) {
      switch (inst.GetSingleWordInOperand(0)) {
        case SpvStorageClassInput:
        case SpvStorageClassOutput:
          MarkPointeeTypeAsFullUsed(inst.type_id());
          break;
        default:
          break;
      }
    }
  }

  for (const Function& func : *get_module()) {
    FindLiveMembers(func);
  }
}

void EliminateDeadMembersPass::FindLiveMembers(const Function& function) {
  function.ForEachInst(
      [this](const Instruction* inst) { FindLiveMembers(inst); });
}

void EliminateDeadMembersPass::FindLiveMembers(const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpStore:
      MarkMembersAsLiveForStore(inst);
      break;
    case SpvOpCopyMemory:
    case SpvOpCopyMemorySized:
      MarkMembersAsLiveForCopyMemory(inst);
      break;
    case SpvOpCompositeExtract:
      MarkMembersAsLiveForExtract(inst);
      break;
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpPtrAccessChain:
    case SpvOpInBoundsPtrAccessChain:
      MarkMembersAsLiveForAccessChain(inst);
      break;
    case SpvOpReturnValue:
      // This should be an issue only if we are returning from the entry point.
      // However, for now I will keep it more conservative because functions are
      // often inlined leaving only the entry points.
      MarkOperandTypeAsFullyUsed(inst, 0);
      break;
    case SpvOpArrayLength:
      MarkMembersAsLiveForArrayLength(inst);
      break;
    case SpvOpLoad:
    case SpvOpCompositeInsert:
    case SpvOpCompositeConstruct:
      break;
    default:
      // This path is here for safety.  All instructions that can reference
      // structs in a function body should be handled above.  However, this will
      // keep the pass valid, but not optimal, as new instructions get added
      // or if something was missed.
      MarkStructOperandsAsFullyUsed(inst);
      break;
  }
}

void EliminateDeadMembersPass::MarkMembersAsLiveForStore(
    const Instruction* inst) {
  // We should only have to mark the members as live if the store is to
  // memory that is read outside of the shader.  Other passes can remove all
  // store to memory that is not visible outside of the shader, so we do not
  // complicate the code for now.
  assert(inst->opcode() == SpvOpStore);
  uint32_t object_id = inst->GetSingleWordInOperand(1);
  Instruction* object_inst = context()->get_def_use_mgr()->GetDef(object_id);
  uint32_t object_type_id = object_inst->type_id();
  MarkTypeAsFullyUsed(object_type_id);
}

void EliminateDeadMembersPass::MarkTypeAsFullyUsed(uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  assert(type_inst != nullptr);
  if (type_inst->opcode() != SpvOpTypeStruct) {
    return;
  }

  // Mark every member of the current struct as used.
  for (uint32_t i = 0; i < type_inst->NumInOperands(); ++i) {
    used_members_[type_id].insert(i);
  }

  // Mark any sub struct as fully used.
  for (uint32_t i = 0; i < type_inst->NumInOperands(); ++i) {
    MarkTypeAsFullyUsed(type_inst->GetSingleWordInOperand(i));
  }
}

void EliminateDeadMembersPass::MarkPointeeTypeAsFullUsed(uint32_t ptr_type_id) {
  Instruction* ptr_type_inst = get_def_use_mgr()->GetDef(ptr_type_id);
  assert(ptr_type_inst->opcode() == SpvOpTypePointer);
  MarkTypeAsFullyUsed(ptr_type_inst->GetSingleWordInOperand(1));
}

void EliminateDeadMembersPass::MarkMembersAsLiveForCopyMemory(
    const Instruction* inst) {
  uint32_t target_id = inst->GetSingleWordInOperand(0);
  Instruction* target_inst = get_def_use_mgr()->GetDef(target_id);
  uint32_t pointer_type_id = target_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);
  MarkTypeAsFullyUsed(type_id);
}

void EliminateDeadMembersPass::MarkMembersAsLiveForExtract(
    const Instruction* inst) {
  assert(inst->opcode() == SpvOpCompositeExtract);

  uint32_t composite_id = inst->GetSingleWordInOperand(0);
  Instruction* composite_inst = get_def_use_mgr()->GetDef(composite_id);
  uint32_t type_id = composite_inst->type_id();

  for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    uint32_t member_idx = inst->GetSingleWordInOperand(i);
    switch (type_inst->opcode()) {
      case SpvOpTypeStruct:
        used_members_[type_id].insert(member_idx);
        type_id = type_inst->GetSingleWordInOperand(member_idx);
        break;
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
    }
  }
}

void EliminateDeadMembersPass::MarkMembersAsLiveForAccessChain(
    const Instruction* inst) {
  assert(inst->opcode() == SpvOpAccessChain ||
         inst->opcode() == SpvOpInBoundsAccessChain ||
         inst->opcode() == SpvOpPtrAccessChain ||
         inst->opcode() == SpvOpInBoundsPtrAccessChain);

  uint32_t pointer_id = inst->GetSingleWordInOperand(0);
  Instruction* pointer_inst = get_def_use_mgr()->GetDef(pointer_id);
  uint32_t pointer_type_id = pointer_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);

  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  // For a pointer access chain, we need to skip the |element| index.  It is not
  // a reference to the member of a struct, and it does not change the type.
  uint32_t i = (inst->opcode() == SpvOpAccessChain ||
                        inst->opcode() == SpvOpInBoundsAccessChain
                    ? 1
                    : 2);
  for (; i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case SpvOpTypeStruct: {
        const analysis::IntConstant* member_idx =
            const_mgr->FindDeclaredConstant(inst->GetSingleWordInOperand(i))
                ->AsIntConstant();
        assert(member_idx);
        if (member_idx->type()->AsInteger()->width() == 32) {
          used_members_[type_id].insert(member_idx->GetU32());
          type_id = type_inst->GetSingleWordInOperand(member_idx->GetU32());
        } else {
          used_members_[type_id].insert(
              static_cast<uint32_t>(member_idx->GetU64()));
          type_id = type_inst->GetSingleWordInOperand(
              static_cast<uint32_t>(member_idx->GetU64()));
        }
      } break;
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
    }
  }
}

void EliminateDeadMembersPass::MarkOperandTypeAsFullyUsed(
    const Instruction* inst, uint32_t in_idx) {
  uint32_t op_id = inst->GetSingleWordInOperand(in_idx);
  Instruction* op_inst = get_def_use_mgr()->GetDef(op_id);
  MarkTypeAsFullyUsed(op_inst->type_id());
}

void EliminateDeadMembersPass::MarkMembersAsLiveForArrayLength(
    const Instruction* inst) {
  assert(inst->opcode() == SpvOpArrayLength);
  uint32_t object_id = inst->GetSingleWordInOperand(0);
  Instruction* object_inst = get_def_use_mgr()->GetDef(object_id);
  uint32_t pointer_type_id = object_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);
  used_members_[type_id].insert(inst->GetSingleWordInOperand(1));
}

bool EliminateDeadMembersPass::RemoveDeadMembers() {
  bool modified = false;

  // First update all of the OpTypeStruct instructions.
  get_module()->ForEachInst([&modified, this](Instruction* inst) {
    switch (inst->opcode()) {
      case SpvOpTypeStruct:
        modified |= UpdateOpTypeStruct(inst);
        break;
      default:
        break;
    }
  });

  // Now update all of the instructions that reference the OpTypeStructs.
  get_module()->ForEachInst([&modified, this](Instruction* inst) {
    switch (inst->opcode()) {
      case SpvOpMemberName:
        modified |= UpdateOpMemberNameOrDecorate(inst);
        break;
      case SpvOpMemberDecorate:
        modified |= UpdateOpMemberNameOrDecorate(inst);
        break;
      case SpvOpGroupMemberDecorate:
        modified |= UpdateOpGroupMemberDecorate(inst);
        break;
      case SpvOpSpecConstantComposite:
      case SpvOpConstantComposite:
      case SpvOpCompositeConstruct:
        modified |= UpdateConstantComposite(inst);
        break;
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpInBoundsPtrAccessChain:
        modified |= UpdateAccessChain(inst);
        break;
      case SpvOpCompositeExtract:
        modified |= UpdateCompsiteExtract(inst);
        break;
      case SpvOpCompositeInsert:
        modified |= UpdateCompositeInsert(inst);
        break;
      case SpvOpArrayLength:
        modified |= UpdateOpArrayLength(inst);
        break;
      case SpvOpSpecConstantOp:
        assert(false && "Not yet implemented.");
        // with OpCompositeExtract, OpCompositeInsert
        // For kernels: OpAccessChain, OpInBoundsAccessChain, OpPtrAccessChain,
        // OpInBoundsPtrAccessChain
        break;
      default:
        break;
    }
  });
  return modified;
}

bool EliminateDeadMembersPass::UpdateOpTypeStruct(Instruction* inst) {
  assert(inst->opcode() == SpvOpTypeStruct);

  const auto& live_members = used_members_[inst->result_id()];
  if (live_members.size() == inst->NumInOperands()) {
    return false;
  }

  Instruction::OperandList new_operands;
  for (uint32_t idx : live_members) {
    new_operands.emplace_back(inst->GetInOperand(idx));
  }

  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool EliminateDeadMembersPass::UpdateOpMemberNameOrDecorate(Instruction* inst) {
  assert(inst->opcode() == SpvOpMemberName ||
         inst->opcode() == SpvOpMemberDecorate);

  uint32_t type_id = inst->GetSingleWordInOperand(0);
  auto live_members = used_members_.find(type_id);
  if (live_members == used_members_.end()) {
    return false;
  }

  uint32_t orig_member_idx = inst->GetSingleWordInOperand(1);
  uint32_t new_member_idx = GetNewMemberIndex(type_id, orig_member_idx);

  if (new_member_idx == kRemovedMember) {
    context()->KillInst(inst);
    return true;
  }

  if (new_member_idx == orig_member_idx) {
    return false;
  }

  inst->SetInOperand(1, {new_member_idx});
  return true;
}

bool EliminateDeadMembersPass::UpdateOpGroupMemberDecorate(Instruction* inst) {
  assert(inst->opcode() == SpvOpGroupMemberDecorate);

  bool modified = false;

  Instruction::OperandList new_operands;
  new_operands.emplace_back(inst->GetInOperand(0));
  for (uint32_t i = 1; i < inst->NumInOperands(); i += 2) {
    uint32_t type_id = inst->GetSingleWordInOperand(i);
    uint32_t member_idx = inst->GetSingleWordInOperand(i + 1);
    uint32_t new_member_idx = GetNewMemberIndex(type_id, member_idx);

    if (new_member_idx == kRemovedMember) {
      modified = true;
      continue;
    }

    new_operands.emplace_back(inst->GetOperand(i));
    if (new_member_idx != member_idx) {
      new_operands.emplace_back(
          Operand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {new_member_idx}}));
      modified = true;
    } else {
      new_operands.emplace_back(inst->GetOperand(i + 1));
    }
  }

  if (!modified) {
    return false;
  }

  if (new_operands.size() == 1) {
    context()->KillInst(inst);
    return true;
  }

  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool EliminateDeadMembersPass::UpdateConstantComposite(Instruction* inst) {
  assert(inst->opcode() == SpvOpConstantComposite ||
         inst->opcode() == SpvOpCompositeConstruct);
  uint32_t type_id = inst->type_id();

  bool modified = false;
  Instruction::OperandList new_operands;
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    uint32_t new_idx = GetNewMemberIndex(type_id, i);
    if (new_idx == kRemovedMember) {
      modified = true;
    } else {
      new_operands.emplace_back(inst->GetInOperand(i));
    }
  }
  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return modified;
}

bool EliminateDeadMembersPass::UpdateAccessChain(Instruction* inst) {
  assert(inst->opcode() == SpvOpAccessChain ||
         inst->opcode() == SpvOpInBoundsAccessChain ||
         inst->opcode() == SpvOpPtrAccessChain ||
         inst->opcode() == SpvOpInBoundsPtrAccessChain);

  uint32_t pointer_id = inst->GetSingleWordInOperand(0);
  Instruction* pointer_inst = get_def_use_mgr()->GetDef(pointer_id);
  uint32_t pointer_type_id = pointer_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);

  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  Instruction::OperandList new_operands;
  bool modified = false;
  new_operands.emplace_back(inst->GetInOperand(0));

  // For pointer access chains we want to copy the element operand.
  if (inst->opcode() == SpvOpPtrAccessChain ||
      inst->opcode() == SpvOpInBoundsPtrAccessChain) {
    new_operands.emplace_back(inst->GetInOperand(1));
  }

  for (uint32_t i = static_cast<uint32_t>(new_operands.size());
       i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case SpvOpTypeStruct: {
        const analysis::IntConstant* member_idx =
            const_mgr->FindDeclaredConstant(inst->GetSingleWordInOperand(i))
                ->AsIntConstant();
        assert(member_idx);
        uint32_t orig_member_idx;
        if (member_idx->type()->AsInteger()->width() == 32) {
          orig_member_idx = member_idx->GetU32();
        } else {
          orig_member_idx = static_cast<uint32_t>(member_idx->GetU64());
        }
        uint32_t new_member_idx = GetNewMemberIndex(type_id, orig_member_idx);
        assert(new_member_idx != kRemovedMember);
        if (orig_member_idx != new_member_idx) {
          InstructionBuilder ir_builder(
              context(), inst,
              IRContext::kAnalysisDefUse |
                  IRContext::kAnalysisInstrToBlockMapping);
          uint32_t const_id =
              ir_builder.GetUintConstant(new_member_idx)->result_id();
          new_operands.emplace_back(Operand({SPV_OPERAND_TYPE_ID, {const_id}}));
          modified = true;
        } else {
          new_operands.emplace_back(inst->GetInOperand(i));
        }
        // The type will have already been rewritten, so use the new member
        // index.
        type_id = type_inst->GetSingleWordInOperand(new_member_idx);
      } break;
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
        new_operands.emplace_back(inst->GetInOperand(i));
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
        break;
    }
  }

  if (!modified) {
    return false;
  }
  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return true;
}

uint32_t EliminateDeadMembersPass::GetNewMemberIndex(uint32_t type_id,
                                                     uint32_t member_idx) {
  auto live_members = used_members_.find(type_id);
  if (live_members == used_members_.end()) {
    return member_idx;
  }

  auto current_member = live_members->second.find(member_idx);
  if (current_member == live_members->second.end()) {
    return kRemovedMember;
  }

  return static_cast<uint32_t>(
      std::distance(live_members->second.begin(), current_member));
}

bool EliminateDeadMembersPass::UpdateCompsiteExtract(Instruction* inst) {
  uint32_t object_id = inst->GetSingleWordInOperand(0);
  Instruction* object_inst = get_def_use_mgr()->GetDef(object_id);
  uint32_t type_id = object_inst->type_id();

  Instruction::OperandList new_operands;
  bool modified = false;
  new_operands.emplace_back(inst->GetInOperand(0));
  for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
    uint32_t member_idx = inst->GetSingleWordInOperand(i);
    uint32_t new_member_idx = GetNewMemberIndex(type_id, member_idx);
    assert(new_member_idx != kRemovedMember);
    if (member_idx != new_member_idx) {
      modified = true;
    }
    new_operands.emplace_back(
        Operand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {new_member_idx}}));

    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case SpvOpTypeStruct:
        assert(i != 1 || (inst->opcode() != SpvOpPtrAccessChain &&
                          inst->opcode() != SpvOpInBoundsPtrAccessChain));
        // The type will have already been rewriten, so use the new member
        // index.
        type_id = type_inst->GetSingleWordInOperand(new_member_idx);
        break;
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
    }
  }

  if (!modified) {
    return false;
  }
  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool EliminateDeadMembersPass::UpdateCompositeInsert(Instruction* inst) {
  uint32_t composite_id = inst->GetSingleWordInOperand(1);
  Instruction* composite_inst = get_def_use_mgr()->GetDef(composite_id);
  uint32_t type_id = composite_inst->type_id();

  Instruction::OperandList new_operands;
  bool modified = false;
  new_operands.emplace_back(inst->GetInOperand(0));
  new_operands.emplace_back(inst->GetInOperand(1));
  for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
    uint32_t member_idx = inst->GetSingleWordInOperand(i);
    uint32_t new_member_idx = GetNewMemberIndex(type_id, member_idx);
    if (new_member_idx == kRemovedMember) {
      context()->KillInst(inst);
      return true;
    }

    if (member_idx != new_member_idx) {
      modified = true;
    }
    new_operands.emplace_back(
        Operand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {new_member_idx}}));

    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case SpvOpTypeStruct:
        // The type will have already been rewritten, so use the new member
        // index.
        type_id = type_inst->GetSingleWordInOperand(new_member_idx);
        break;
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
      case SpvOpTypeVector:
      case SpvOpTypeMatrix:
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
    }
  }

  if (!modified) {
    return false;
  }
  inst->SetInOperands(std::move(new_operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool EliminateDeadMembersPass::UpdateOpArrayLength(Instruction* inst) {
  uint32_t struct_id = inst->GetSingleWordInOperand(0);
  Instruction* struct_inst = get_def_use_mgr()->GetDef(struct_id);
  uint32_t pointer_type_id = struct_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);

  uint32_t member_idx = inst->GetSingleWordInOperand(1);
  uint32_t new_member_idx = GetNewMemberIndex(type_id, member_idx);
  assert(new_member_idx != kRemovedMember);

  if (member_idx == new_member_idx) {
    return false;
  }

  inst->SetInOperand(1, {new_member_idx});
  context()->UpdateDefUse(inst);
  return true;
}

void EliminateDeadMembersPass::MarkStructOperandsAsFullyUsed(
    const Instruction* inst) {
  if (inst->type_id() != 0) {
    MarkTypeAsFullyUsed(inst->type_id());
  }

  inst->ForEachInId([this](const uint32_t* id) {
    Instruction* instruction = get_def_use_mgr()->GetDef(*id);
    if (instruction->type_id() != 0) {
      MarkTypeAsFullyUsed(instruction->type_id());
    }
  });
}
}  // namespace opt
}  // namespace spvtools
