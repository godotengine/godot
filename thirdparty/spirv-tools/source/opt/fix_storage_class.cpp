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

#include "fix_storage_class.h"

#include <set>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status FixStorageClass::Process() {
  bool modified = false;

  get_module()->ForEachInst([this, &modified](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpVariable) {
      std::set<uint32_t> seen;
      std::vector<std::pair<Instruction*, uint32_t>> uses;
      get_def_use_mgr()->ForEachUse(inst,
                                    [&uses](Instruction* use, uint32_t op_idx) {
                                      uses.push_back({use, op_idx});
                                    });

      for (auto& use : uses) {
        modified |= PropagateStorageClass(
            use.first,
            static_cast<spv::StorageClass>(inst->GetSingleWordInOperand(0)),
            &seen);
        assert(seen.empty() && "Seen was not properly reset.");
        modified |=
            PropagateType(use.first, inst->type_id(), use.second, &seen);
        assert(seen.empty() && "Seen was not properly reset.");
      }
    }
  });
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FixStorageClass::PropagateStorageClass(Instruction* inst,
                                            spv::StorageClass storage_class,
                                            std::set<uint32_t>* seen) {
  if (!IsPointerResultType(inst)) {
    return false;
  }

  if (IsPointerToStorageClass(inst, storage_class)) {
    if (inst->opcode() == spv::Op::OpPhi) {
      if (!seen->insert(inst->result_id()).second) {
        return false;
      }
    }

    bool modified = false;
    std::vector<Instruction*> uses;
    get_def_use_mgr()->ForEachUser(
        inst, [&uses](Instruction* use) { uses.push_back(use); });
    for (Instruction* use : uses) {
      modified |= PropagateStorageClass(use, storage_class, seen);
    }

    if (inst->opcode() == spv::Op::OpPhi) {
      seen->erase(inst->result_id());
    }
    return modified;
  }

  switch (inst->opcode()) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpCopyObject:
    case spv::Op::OpPhi:
    case spv::Op::OpSelect:
      FixInstructionStorageClass(inst, storage_class, seen);
      return true;
    case spv::Op::OpFunctionCall:
      // We cannot be sure of the actual connection between the storage class
      // of the parameter and the storage class of the result, so we should not
      // do anything.  If the result type needs to be fixed, the function call
      // should be inlined.
      return false;
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
    case spv::Op::OpVariable:
    case spv::Op::OpBitcast:
      // Nothing to change for these opcode.  The result type is the same
      // regardless of the storage class of the operand.
      return false;
    default:
      assert(false &&
             "Not expecting instruction to have a pointer result type.");
      return false;
  }
}

void FixStorageClass::FixInstructionStorageClass(
    Instruction* inst, spv::StorageClass storage_class,
    std::set<uint32_t>* seen) {
  assert(IsPointerResultType(inst) &&
         "The result type of the instruction must be a pointer.");

  ChangeResultStorageClass(inst, storage_class);

  std::vector<Instruction*> uses;
  get_def_use_mgr()->ForEachUser(
      inst, [&uses](Instruction* use) { uses.push_back(use); });
  for (Instruction* use : uses) {
    PropagateStorageClass(use, storage_class, seen);
  }
}

void FixStorageClass::ChangeResultStorageClass(
    Instruction* inst, spv::StorageClass storage_class) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  Instruction* result_type_inst = get_def_use_mgr()->GetDef(inst->type_id());
  assert(result_type_inst->opcode() == spv::Op::OpTypePointer);
  uint32_t pointee_type_id = result_type_inst->GetSingleWordInOperand(1);
  uint32_t new_result_type_id =
      type_mgr->FindPointerToType(pointee_type_id, storage_class);
  inst->SetResultType(new_result_type_id);
  context()->UpdateDefUse(inst);
}

bool FixStorageClass::IsPointerResultType(Instruction* inst) {
  if (inst->type_id() == 0) {
    return false;
  }
  const analysis::Type* ret_type =
      context()->get_type_mgr()->GetType(inst->type_id());
  return ret_type->AsPointer() != nullptr;
}

bool FixStorageClass::IsPointerToStorageClass(Instruction* inst,
                                              spv::StorageClass storage_class) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Type* pType = type_mgr->GetType(inst->type_id());
  const analysis::Pointer* result_type = pType->AsPointer();

  if (result_type == nullptr) {
    return false;
  }

  return (result_type->storage_class() == storage_class);
}

bool FixStorageClass::ChangeResultType(Instruction* inst,
                                       uint32_t new_type_id) {
  if (inst->type_id() == new_type_id) {
    return false;
  }

  context()->ForgetUses(inst);
  inst->SetResultType(new_type_id);
  context()->AnalyzeUses(inst);
  return true;
}

bool FixStorageClass::PropagateType(Instruction* inst, uint32_t type_id,
                                    uint32_t op_idx, std::set<uint32_t>* seen) {
  assert(type_id != 0 && "Not given a valid type in PropagateType");
  bool modified = false;

  // If the type of operand |op_idx| forces the result type of |inst| to a
  // particular type, then we want find that type.
  uint32_t new_type_id = 0;
  switch (inst->opcode()) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
      if (op_idx == 2) {
        new_type_id = WalkAccessChainType(inst, type_id);
      }
      break;
    case spv::Op::OpCopyObject:
      new_type_id = type_id;
      break;
    case spv::Op::OpPhi:
      if (seen->insert(inst->result_id()).second) {
        new_type_id = type_id;
      }
      break;
    case spv::Op::OpSelect:
      if (op_idx > 2) {
        new_type_id = type_id;
      }
      break;
    case spv::Op::OpFunctionCall:
      // We cannot be sure of the actual connection between the type
      // of the parameter and the type of the result, so we should not
      // do anything.  If the result type needs to be fixed, the function call
      // should be inlined.
      return false;
    case spv::Op::OpLoad: {
      Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
      new_type_id = type_inst->GetSingleWordInOperand(1);
      break;
    }
    case spv::Op::OpStore: {
      uint32_t obj_id = inst->GetSingleWordInOperand(1);
      Instruction* obj_inst = get_def_use_mgr()->GetDef(obj_id);
      uint32_t obj_type_id = obj_inst->type_id();

      uint32_t ptr_id = inst->GetSingleWordInOperand(0);
      Instruction* ptr_inst = get_def_use_mgr()->GetDef(ptr_id);
      uint32_t pointee_type_id = GetPointeeTypeId(ptr_inst);

      if (obj_type_id != pointee_type_id) {
        if (context()->get_type_mgr()->GetType(obj_type_id)->AsImage() &&
            context()->get_type_mgr()->GetType(pointee_type_id)->AsImage()) {
          // When storing an image, allow the type mismatch
          // and let the later legalization passes eliminate the OpStore.
          // This is to support assigning an image to a variable,
          // where the assigned image does not have a pre-defined
          // image format.
          return false;
        }

        uint32_t copy_id = GenerateCopy(obj_inst, pointee_type_id, inst);
        inst->SetInOperand(1, {copy_id});
        context()->UpdateDefUse(inst);
      }
    } break;
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      // TODO: May need to expand the copy as we do with the stores.
      break;
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
      // TODO: DXC does not seem to generate code that will require changes to
      // these opcode.  The can be implemented when they come up.
      break;
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpBitcast:
      // Nothing to change for these opcode.  The result type is the same
      // regardless of the type of the operand.
      return false;
    default:
      // I expect the remaining instructions to act on types that are guaranteed
      // to be unique, so no change will be necessary.
      break;
  }

  // If the operand forces the result type, then make sure the result type
  // matches, and update the uses of |inst|.  We do not have to check the uses
  // of |inst| in the result type is not forced because we are only looking for
  // issue that come from mismatches between function formal and actual
  // parameters after the function has been inlined.  These parameters are
  // pointers. Once the type no longer depends on the type of the parameter,
  // then the types should have be correct.
  if (new_type_id != 0) {
    modified = ChangeResultType(inst, new_type_id);

    std::vector<std::pair<Instruction*, uint32_t>> uses;
    get_def_use_mgr()->ForEachUse(inst,
                                  [&uses](Instruction* use, uint32_t idx) {
                                    uses.push_back({use, idx});
                                  });

    for (auto& use : uses) {
      PropagateType(use.first, new_type_id, use.second, seen);
    }

    if (inst->opcode() == spv::Op::OpPhi) {
      seen->erase(inst->result_id());
    }
  }
  return modified;
}

uint32_t FixStorageClass::WalkAccessChainType(Instruction* inst, uint32_t id) {
  uint32_t start_idx = 0;
  switch (inst->opcode()) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
      start_idx = 1;
      break;
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
      start_idx = 2;
      break;
    default:
      assert(false);
      break;
  }

  Instruction* orig_type_inst = get_def_use_mgr()->GetDef(id);
  assert(orig_type_inst->opcode() == spv::Op::OpTypePointer);
  id = orig_type_inst->GetSingleWordInOperand(1);

  for (uint32_t i = start_idx; i < inst->NumInOperands(); ++i) {
    Instruction* type_inst = get_def_use_mgr()->GetDef(id);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeMatrix:
      case spv::Op::OpTypeVector:
        id = type_inst->GetSingleWordInOperand(0);
        break;
      case spv::Op::OpTypeStruct: {
        const analysis::Constant* index_const =
            context()->get_constant_mgr()->FindDeclaredConstant(
                inst->GetSingleWordInOperand(i));
        uint32_t index = index_const->GetU32();
        id = type_inst->GetSingleWordInOperand(index);
        break;
      }
      default:
        break;
    }
    assert(id != 0 &&
           "Tried to extract from an object where it cannot be done.");
  }

  return context()->get_type_mgr()->FindPointerToType(
      id, static_cast<spv::StorageClass>(
              orig_type_inst->GetSingleWordInOperand(0)));
}

// namespace opt

}  // namespace opt
}  // namespace spvtools
