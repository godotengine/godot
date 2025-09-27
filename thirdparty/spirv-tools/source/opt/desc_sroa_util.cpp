// Copyright (c) 2021 Google LLC
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

#include "source/opt/desc_sroa_util.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kOpAccessChainInOperandIndexes = 1;

// Returns the length of array type |type|.
uint32_t GetLengthOfArrayType(IRContext* context, Instruction* type) {
  assert(type->opcode() == spv::Op::OpTypeArray && "type must be array");
  uint32_t length_id = type->GetSingleWordInOperand(1);
  const analysis::Constant* length_const =
      context->get_constant_mgr()->FindDeclaredConstant(length_id);
  assert(length_const != nullptr);
  return length_const->GetU32();
}

}  // namespace

namespace descsroautil {

bool IsDescriptorArray(IRContext* context, Instruction* var) {
  if (var->opcode() != spv::Op::OpVariable) {
    return false;
  }

  uint32_t ptr_type_id = var->type_id();
  Instruction* ptr_type_inst = context->get_def_use_mgr()->GetDef(ptr_type_id);
  if (ptr_type_inst->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  uint32_t var_type_id = ptr_type_inst->GetSingleWordInOperand(1);
  Instruction* var_type_inst = context->get_def_use_mgr()->GetDef(var_type_id);
  if (var_type_inst->opcode() != spv::Op::OpTypeArray &&
      var_type_inst->opcode() != spv::Op::OpTypeStruct) {
    return false;
  }

  // All structures with descriptor assignments must be replaced by variables,
  // one for each of their members - with the exceptions of buffers.
  if (IsTypeOfStructuredBuffer(context, var_type_inst)) {
    return false;
  }

  if (!context->get_decoration_mgr()->HasDecoration(
          var->result_id(), uint32_t(spv::Decoration::DescriptorSet))) {
    return false;
  }

  return context->get_decoration_mgr()->HasDecoration(
      var->result_id(), uint32_t(spv::Decoration::Binding));
}

bool IsTypeOfStructuredBuffer(IRContext* context, const Instruction* type) {
  if (type->opcode() != spv::Op::OpTypeStruct) {
    return false;
  }

  // All buffers have offset decorations for members of their structure types.
  // This is how we distinguish it from a structure of descriptors.
  return context->get_decoration_mgr()->HasDecoration(
      type->result_id(), uint32_t(spv::Decoration::Offset));
}

const analysis::Constant* GetAccessChainIndexAsConst(
    IRContext* context, Instruction* access_chain) {
  if (access_chain->NumInOperands() <= 1) {
    return nullptr;
  }
  uint32_t idx_id = GetFirstIndexOfAccessChain(access_chain);
  const analysis::Constant* idx_const =
      context->get_constant_mgr()->FindDeclaredConstant(idx_id);
  return idx_const;
}

uint32_t GetFirstIndexOfAccessChain(Instruction* access_chain) {
  assert(access_chain->NumInOperands() > 1 &&
         "OpAccessChain does not have Indexes operand");
  return access_chain->GetSingleWordInOperand(kOpAccessChainInOperandIndexes);
}

uint32_t GetNumberOfElementsForArrayOrStruct(IRContext* context,
                                             Instruction* var) {
  uint32_t ptr_type_id = var->type_id();
  Instruction* ptr_type_inst = context->get_def_use_mgr()->GetDef(ptr_type_id);
  assert(ptr_type_inst->opcode() == spv::Op::OpTypePointer &&
         "Variable should be a pointer to an array or structure.");
  uint32_t pointee_type_id = ptr_type_inst->GetSingleWordInOperand(1);
  Instruction* pointee_type_inst =
      context->get_def_use_mgr()->GetDef(pointee_type_id);
  if (pointee_type_inst->opcode() == spv::Op::OpTypeArray) {
    return GetLengthOfArrayType(context, pointee_type_inst);
  }
  assert(pointee_type_inst->opcode() == spv::Op::OpTypeStruct &&
         "Variable should be a pointer to an array or structure.");
  return pointee_type_inst->NumInOperands();
}

}  // namespace descsroautil
}  // namespace opt
}  // namespace spvtools
