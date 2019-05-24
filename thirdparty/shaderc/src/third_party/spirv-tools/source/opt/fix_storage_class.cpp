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

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status FixStorageClass::Process() {
  bool modified = false;

  get_module()->ForEachInst([this, &modified](Instruction* inst) {
    if (inst->opcode() == SpvOpVariable) {
      std::vector<Instruction*> uses;
      get_def_use_mgr()->ForEachUser(
          inst, [&uses](Instruction* use) { uses.push_back(use); });
      for (Instruction* use : uses) {
        modified |= PropagateStorageClass(
            use, static_cast<SpvStorageClass>(inst->GetSingleWordInOperand(0)));
      }
    }
  });
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FixStorageClass::PropagateStorageClass(Instruction* inst,
                                            SpvStorageClass storage_class) {
  if (!IsPointerResultType(inst)) {
    return false;
  }

  if (IsPointerToStorageClass(inst, storage_class)) {
    return false;
  }

  switch (inst->opcode()) {
    case SpvOpAccessChain:
    case SpvOpPtrAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpCopyObject:
    case SpvOpPhi:
    case SpvOpSelect:
      FixInstruction(inst, storage_class);
      return true;
    case SpvOpFunctionCall:
      // We cannot be sure of the actual connection between the storage class
      // of the parameter and the storage class of the result, so we should not
      // do anything.  If the result type needs to be fixed, the function call
      // should be inlined.
      return false;
    case SpvOpImageTexelPointer:
    case SpvOpLoad:
    case SpvOpStore:
    case SpvOpCopyMemory:
    case SpvOpCopyMemorySized:
    case SpvOpVariable:
      // Nothing to change for these opcode.  The result type is the same
      // regardless of the storage class of the operand.
      return false;
    default:
      assert(false &&
             "Not expecting instruction to have a pointer result type.");
      return false;
  }
}

void FixStorageClass::FixInstruction(Instruction* inst,
                                     SpvStorageClass storage_class) {
  assert(IsPointerResultType(inst) &&
         "The result type of the instruction must be a pointer.");

  ChangeResultStorageClass(inst, storage_class);

  std::vector<Instruction*> uses;
  get_def_use_mgr()->ForEachUser(
      inst, [&uses](Instruction* use) { uses.push_back(use); });
  for (Instruction* use : uses) {
    PropagateStorageClass(use, storage_class);
  }
}

void FixStorageClass::ChangeResultStorageClass(
    Instruction* inst, SpvStorageClass storage_class) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  Instruction* result_type_inst = get_def_use_mgr()->GetDef(inst->type_id());
  assert(result_type_inst->opcode() == SpvOpTypePointer);
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
                                              SpvStorageClass storage_class) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Type* pType = type_mgr->GetType(inst->type_id());
  const analysis::Pointer* result_type = pType->AsPointer();

  if (result_type == nullptr) {
    return false;
  }

  return (result_type->storage_class() == storage_class);
}

// namespace opt

}  // namespace opt
}  // namespace spvtools
