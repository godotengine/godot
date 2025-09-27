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

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kRemovedMember = 0xFFFFFFFF;
constexpr uint32_t kSpecConstOpOpcodeIdx = 0;
constexpr uint32_t kArrayElementTypeIdx = 0;
}  // namespace

Pass::Status EliminateDeadMembersPass::Process() {
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return Status::SuccessWithoutChange;

  FindLiveMembers();
  if (RemoveDeadMembers()) {
    return Status::SuccessWithChange;
  }
  return Status::SuccessWithoutChange;
}

void EliminateDeadMembersPass::FindLiveMembers() {
  // Until we have implemented the rewriting of OpSpecConsantOp instructions,
  // we have to mark them as fully used just to be safe.
  for (auto& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpSpecConstantOp) {
      switch (spv::Op(inst.GetSingleWordInOperand(kSpecConstOpOpcodeIdx))) {
        case spv::Op::OpCompositeExtract:
          MarkMembersAsLiveForExtract(&inst);
          break;
        case spv::Op::OpCompositeInsert:
          // Nothing specific to do.
          break;
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpPtrAccessChain:
        case spv::Op::OpInBoundsPtrAccessChain:
          assert(false && "Not implemented yet.");
          break;
        default:
          break;
      }
    } else if (inst.opcode() == spv::Op::OpVariable) {
      switch (spv::StorageClass(inst.GetSingleWordInOperand(0))) {
        case spv::StorageClass::Input:
        case spv::StorageClass::Output:
          MarkPointeeTypeAsFullUsed(inst.type_id());
          break;
        default:
          // Ignore structured buffers as layout(offset) qualifiers cannot be
          // applied to structure fields
          if (inst.IsVulkanStorageBufferVariable())
            MarkPointeeTypeAsFullUsed(inst.type_id());
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
    case spv::Op::OpStore:
      MarkMembersAsLiveForStore(inst);
      break;
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      MarkMembersAsLiveForCopyMemory(inst);
      break;
    case spv::Op::OpCompositeExtract:
      MarkMembersAsLiveForExtract(inst);
      break;
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
      MarkMembersAsLiveForAccessChain(inst);
      break;
    case spv::Op::OpReturnValue:
      // This should be an issue only if we are returning from the entry point.
      // However, for now I will keep it more conservative because functions are
      // often inlined leaving only the entry points.
      MarkOperandTypeAsFullyUsed(inst, 0);
      break;
    case spv::Op::OpArrayLength:
      MarkMembersAsLiveForArrayLength(inst);
      break;
    case spv::Op::OpLoad:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpCompositeConstruct:
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
  assert(inst->opcode() == spv::Op::OpStore);
  uint32_t object_id = inst->GetSingleWordInOperand(1);
  Instruction* object_inst = context()->get_def_use_mgr()->GetDef(object_id);
  uint32_t object_type_id = object_inst->type_id();
  MarkTypeAsFullyUsed(object_type_id);
}

void EliminateDeadMembersPass::MarkTypeAsFullyUsed(uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  assert(type_inst != nullptr);

  switch (type_inst->opcode()) {
    case spv::Op::OpTypeStruct:
      // Mark every member and its type as fully used.
      for (uint32_t i = 0; i < type_inst->NumInOperands(); ++i) {
        used_members_[type_id].insert(i);
        MarkTypeAsFullyUsed(type_inst->GetSingleWordInOperand(i));
      }
      break;
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      MarkTypeAsFullyUsed(
          type_inst->GetSingleWordInOperand(kArrayElementTypeIdx));
      break;
    default:
      break;
  }
}

void EliminateDeadMembersPass::MarkPointeeTypeAsFullUsed(uint32_t ptr_type_id) {
  Instruction* ptr_type_inst = get_def_use_mgr()->GetDef(ptr_type_id);
  assert(ptr_type_inst->opcode() == spv::Op::OpTypePointer);
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
  assert(inst->opcode() == spv::Op::OpCompositeExtract ||
         (inst->opcode() == spv::Op::OpSpecConstantOp &&
          spv::Op(inst->GetSingleWordInOperand(kSpecConstOpOpcodeIdx)) ==
              spv::Op::OpCompositeExtract));

  uint32_t first_operand =
      (inst->opcode() == spv::Op::OpSpecConstantOp ? 1 : 0);
  uint32_t composite_id = inst->GetSingleWordInOperand(first_operand);
  Instruction* composite_inst = get_def_use_mgr()->GetDef(composite_id);
  uint32_t type_id = composite_inst->type_id();

  for (uint32_t i = first_operand + 1; i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    uint32_t member_idx = inst->GetSingleWordInOperand(i);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeStruct:
        used_members_[type_id].insert(member_idx);
        type_id = type_inst->GetSingleWordInOperand(member_idx);
        break;
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
        type_id = type_inst->GetSingleWordInOperand(0);
        break;
      default:
        assert(false);
    }
  }
}

void EliminateDeadMembersPass::MarkMembersAsLiveForAccessChain(
    const Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpAccessChain ||
         inst->opcode() == spv::Op::OpInBoundsAccessChain ||
         inst->opcode() == spv::Op::OpPtrAccessChain ||
         inst->opcode() == spv::Op::OpInBoundsPtrAccessChain);

  uint32_t pointer_id = inst->GetSingleWordInOperand(0);
  Instruction* pointer_inst = get_def_use_mgr()->GetDef(pointer_id);
  uint32_t pointer_type_id = pointer_inst->type_id();
  Instruction* pointer_type_inst = get_def_use_mgr()->GetDef(pointer_type_id);
  uint32_t type_id = pointer_type_inst->GetSingleWordInOperand(1);

  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  // For a pointer access chain, we need to skip the |element| index.  It is not
  // a reference to the member of a struct, and it does not change the type.
  uint32_t i = (inst->opcode() == spv::Op::OpAccessChain ||
                        inst->opcode() == spv::Op::OpInBoundsAccessChain
                    ? 1
                    : 2);
  for (; i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeStruct: {
        const analysis::IntConstant* member_idx =
            const_mgr->FindDeclaredConstant(inst->GetSingleWordInOperand(i))
                ->AsIntConstant();
        assert(member_idx);
        uint32_t index =
            static_cast<uint32_t>(member_idx->GetZeroExtendedValue());
        used_members_[type_id].insert(index);
        type_id = type_inst->GetSingleWordInOperand(index);
      } break;
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
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
  assert(inst->opcode() == spv::Op::OpArrayLength);
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
      case spv::Op::OpTypeStruct:
        modified |= UpdateOpTypeStruct(inst);
        break;
      default:
        break;
    }
  });

  // Now update all of the instructions that reference the OpTypeStructs.
  get_module()->ForEachInst([&modified, this](Instruction* inst) {
    switch (inst->opcode()) {
      case spv::Op::OpMemberName:
        modified |= UpdateOpMemberNameOrDecorate(inst);
        break;
      case spv::Op::OpMemberDecorate:
        modified |= UpdateOpMemberNameOrDecorate(inst);
        break;
      case spv::Op::OpGroupMemberDecorate:
        modified |= UpdateOpGroupMemberDecorate(inst);
        break;
      case spv::Op::OpSpecConstantComposite:
      case spv::Op::OpConstantComposite:
      case spv::Op::OpCompositeConstruct:
        modified |= UpdateConstantComposite(inst);
        break;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpPtrAccessChain:
      case spv::Op::OpInBoundsPtrAccessChain:
        modified |= UpdateAccessChain(inst);
        break;
      case spv::Op::OpCompositeExtract:
        modified |= UpdateCompsiteExtract(inst);
        break;
      case spv::Op::OpCompositeInsert:
        modified |= UpdateCompositeInsert(inst);
        break;
      case spv::Op::OpArrayLength:
        modified |= UpdateOpArrayLength(inst);
        break;
      case spv::Op::OpSpecConstantOp:
        switch (spv::Op(inst->GetSingleWordInOperand(kSpecConstOpOpcodeIdx))) {
          case spv::Op::OpCompositeExtract:
            modified |= UpdateCompsiteExtract(inst);
            break;
          case spv::Op::OpCompositeInsert:
            modified |= UpdateCompositeInsert(inst);
            break;
          case spv::Op::OpAccessChain:
          case spv::Op::OpInBoundsAccessChain:
          case spv::Op::OpPtrAccessChain:
          case spv::Op::OpInBoundsPtrAccessChain:
            assert(false && "Not implemented yet.");
            break;
          default:
            break;
        }
        break;
      default:
        break;
    }
  });
  return modified;
}

bool EliminateDeadMembersPass::UpdateOpTypeStruct(Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpTypeStruct);

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
  assert(inst->opcode() == spv::Op::OpMemberName ||
         inst->opcode() == spv::Op::OpMemberDecorate);

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
  assert(inst->opcode() == spv::Op::OpGroupMemberDecorate);

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
  assert(inst->opcode() == spv::Op::OpSpecConstantComposite ||
         inst->opcode() == spv::Op::OpConstantComposite ||
         inst->opcode() == spv::Op::OpCompositeConstruct);
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
  assert(inst->opcode() == spv::Op::OpAccessChain ||
         inst->opcode() == spv::Op::OpInBoundsAccessChain ||
         inst->opcode() == spv::Op::OpPtrAccessChain ||
         inst->opcode() == spv::Op::OpInBoundsPtrAccessChain);

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
  if (inst->opcode() == spv::Op::OpPtrAccessChain ||
      inst->opcode() == spv::Op::OpInBoundsPtrAccessChain) {
    new_operands.emplace_back(inst->GetInOperand(1));
  }

  for (uint32_t i = static_cast<uint32_t>(new_operands.size());
       i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeStruct: {
        const analysis::IntConstant* member_idx =
            const_mgr->FindDeclaredConstant(inst->GetSingleWordInOperand(i))
                ->AsIntConstant();
        assert(member_idx);
        uint32_t orig_member_idx =
            static_cast<uint32_t>(member_idx->GetZeroExtendedValue());
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
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
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
  assert(inst->opcode() == spv::Op::OpCompositeExtract ||
         (inst->opcode() == spv::Op::OpSpecConstantOp &&
          spv::Op(inst->GetSingleWordInOperand(kSpecConstOpOpcodeIdx)) ==
              spv::Op::OpCompositeExtract));

  uint32_t first_operand = 0;
  if (inst->opcode() == spv::Op::OpSpecConstantOp) {
    first_operand = 1;
  }
  uint32_t object_id = inst->GetSingleWordInOperand(first_operand);
  Instruction* object_inst = get_def_use_mgr()->GetDef(object_id);
  uint32_t type_id = object_inst->type_id();

  Instruction::OperandList new_operands;
  bool modified = false;
  for (uint32_t i = 0; i < first_operand + 1; i++) {
    new_operands.emplace_back(inst->GetInOperand(i));
  }
  for (uint32_t i = first_operand + 1; i < inst->NumInOperands(); ++i) {
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
      case spv::Op::OpTypeStruct:
        // The type will have already been rewritten, so use the new member
        // index.
        type_id = type_inst->GetSingleWordInOperand(new_member_idx);
        break;
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
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
  assert(inst->opcode() == spv::Op::OpCompositeInsert ||
         (inst->opcode() == spv::Op::OpSpecConstantOp &&
          spv::Op(inst->GetSingleWordInOperand(kSpecConstOpOpcodeIdx)) ==
              spv::Op::OpCompositeInsert));

  uint32_t first_operand = 0;
  if (inst->opcode() == spv::Op::OpSpecConstantOp) {
    first_operand = 1;
  }

  uint32_t composite_id = inst->GetSingleWordInOperand(first_operand + 1);
  Instruction* composite_inst = get_def_use_mgr()->GetDef(composite_id);
  uint32_t type_id = composite_inst->type_id();

  Instruction::OperandList new_operands;
  bool modified = false;

  for (uint32_t i = 0; i < first_operand + 2; ++i) {
    new_operands.emplace_back(inst->GetInOperand(i));
  }
  for (uint32_t i = first_operand + 2; i < inst->NumInOperands(); ++i) {
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
      case spv::Op::OpTypeStruct:
        // The type will have already been rewritten, so use the new member
        // index.
        type_id = type_inst->GetSingleWordInOperand(new_member_idx);
        break;
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
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
