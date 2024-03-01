// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/pass.h"

#include "source/opt/ir_builder.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kTypePointerTypeIdInIdx = 1;
}  // namespace

Pass::Pass() : consumer_(nullptr), context_(nullptr), already_run_(false) {}

Pass::Status Pass::Run(IRContext* ctx) {
  if (already_run_) {
    return Status::Failure;
  }
  already_run_ = true;

  context_ = ctx;
  Pass::Status status = Process();
  context_ = nullptr;

  if (status == Status::SuccessWithChange) {
    ctx->InvalidateAnalysesExceptFor(GetPreservedAnalyses());
  }
  if (!(status == Status::Failure || ctx->IsConsistent()))
    assert(false && "An analysis in the context is out of date.");
  return status;
}

uint32_t Pass::GetPointeeTypeId(const Instruction* ptrInst) const {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
  return ptrTypeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx);
}

Instruction* Pass::GetBaseType(uint32_t ty_id) {
  Instruction* ty_inst = get_def_use_mgr()->GetDef(ty_id);
  if (ty_inst->opcode() == spv::Op::OpTypeMatrix) {
    uint32_t vty_id = ty_inst->GetSingleWordInOperand(0);
    ty_inst = get_def_use_mgr()->GetDef(vty_id);
  }
  if (ty_inst->opcode() == spv::Op::OpTypeVector) {
    uint32_t cty_id = ty_inst->GetSingleWordInOperand(0);
    ty_inst = get_def_use_mgr()->GetDef(cty_id);
  }
  return ty_inst;
}

bool Pass::IsFloat(uint32_t ty_id, uint32_t width) {
  Instruction* ty_inst = GetBaseType(ty_id);
  if (ty_inst->opcode() != spv::Op::OpTypeFloat) return false;
  return ty_inst->GetSingleWordInOperand(0) == width;
}

uint32_t Pass::GetNullId(uint32_t type_id) {
  if (IsFloat(type_id, 16)) context()->AddCapability(spv::Capability::Float16);
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  const analysis::Type* type = type_mgr->GetType(type_id);
  const analysis::Constant* null_const = const_mgr->GetConstant(type, {});
  Instruction* null_inst =
      const_mgr->GetDefiningInstruction(null_const, type_id);
  return null_inst->result_id();
}

uint32_t Pass::GenerateCopy(Instruction* object_to_copy, uint32_t new_type_id,
                            Instruction* insertion_position) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  uint32_t original_type_id = object_to_copy->type_id();
  if (original_type_id == new_type_id) {
    return object_to_copy->result_id();
  }

  InstructionBuilder ir_builder(
      context(), insertion_position,
      IRContext::kAnalysisInstrToBlockMapping | IRContext::kAnalysisDefUse);

  analysis::Type* original_type = type_mgr->GetType(original_type_id);
  analysis::Type* new_type = type_mgr->GetType(new_type_id);

  if (const analysis::Array* original_array_type = original_type->AsArray()) {
    uint32_t original_element_type_id =
        type_mgr->GetId(original_array_type->element_type());

    analysis::Array* new_array_type = new_type->AsArray();
    assert(new_array_type != nullptr && "Can't copy an array to a non-array.");
    uint32_t new_element_type_id =
        type_mgr->GetId(new_array_type->element_type());

    std::vector<uint32_t> element_ids;
    const analysis::Constant* length_const =
        const_mgr->FindDeclaredConstant(original_array_type->LengthId());
    assert(length_const->AsIntConstant());
    uint32_t array_length = length_const->AsIntConstant()->GetU32();
    for (uint32_t i = 0; i < array_length; i++) {
      Instruction* extract = ir_builder.AddCompositeExtract(
          original_element_type_id, object_to_copy->result_id(), {i});
      element_ids.push_back(
          GenerateCopy(extract, new_element_type_id, insertion_position));
    }

    return ir_builder.AddCompositeConstruct(new_type_id, element_ids)
        ->result_id();
  } else if (const analysis::Struct* original_struct_type =
                 original_type->AsStruct()) {
    analysis::Struct* new_struct_type = new_type->AsStruct();

    const std::vector<const analysis::Type*>& original_types =
        original_struct_type->element_types();
    const std::vector<const analysis::Type*>& new_types =
        new_struct_type->element_types();
    std::vector<uint32_t> element_ids;
    for (uint32_t i = 0; i < original_types.size(); i++) {
      Instruction* extract = ir_builder.AddCompositeExtract(
          type_mgr->GetId(original_types[i]), object_to_copy->result_id(), {i});
      element_ids.push_back(GenerateCopy(extract, type_mgr->GetId(new_types[i]),
                                         insertion_position));
    }
    return ir_builder.AddCompositeConstruct(new_type_id, element_ids)
        ->result_id();
  } else {
    // If we do not have an aggregate type, then we have a problem.  Either we
    // found multiple instances of the same type, or we are copying to an
    // incompatible type.  Either way the code is illegal.
    assert(false &&
           "Don't know how to copy this type.  Code is likely illegal.");
  }
  return 0;
}

}  // namespace opt
}  // namespace spvtools
