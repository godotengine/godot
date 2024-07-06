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

#include "source/fuzz/transformation_add_type_array.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypeArray::TransformationAddTypeArray(
    protobufs::TransformationAddTypeArray message)
    : message_(std::move(message)) {}

TransformationAddTypeArray::TransformationAddTypeArray(uint32_t fresh_id,
                                                       uint32_t element_type_id,
                                                       uint32_t size_id) {
  message_.set_fresh_id(fresh_id);
  message_.set_element_type_id(element_type_id);
  message_.set_size_id(size_id);
}

bool TransformationAddTypeArray::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // A fresh id is required.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  auto element_type =
      ir_context->get_type_mgr()->GetType(message_.element_type_id());
  if (!element_type || element_type->AsFunction() ||
      fuzzerutil::HasBlockOrBufferBlockDecoration(ir_context,
                                                  message_.element_type_id())) {
    // The element type id either does not refer to a type, refers to a function
    // type, or refers to a block-decorated struct. These cases are all illegal.
    return false;
  }
  auto constant =
      ir_context->get_constant_mgr()->GetConstantsFromIds({message_.size_id()});
  if (constant.empty()) {
    // The size id does not refer to a constant.
    return false;
  }
  assert(constant.size() == 1 &&
         "Only one constant id was provided, so only one constant should have "
         "been returned");

  auto int_constant = constant[0]->AsIntConstant();
  if (!int_constant) {
    // The size constant is not an integer.
    return false;
  }
  // We require that the size constant be a 32-bit value that is positive when
  // interpreted as being signed.
  return int_constant->words().size() == 1 && int_constant->GetS32() >= 1;
}

void TransformationAddTypeArray::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.element_type_id()}});
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.size_id()}});
  auto type_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpTypeArray, 0, message_.fresh_id(), in_operands);
  auto type_instruction_ptr = type_instruction.get();
  ir_context->module()->AddType(std::move(type_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Inform the def use manager that there is a new definition. Invalidate the
  // type manager since we have added a new type.
  ir_context->get_def_use_mgr()->AnalyzeInstDef(type_instruction_ptr);
  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisTypes);
}

protobufs::Transformation TransformationAddTypeArray::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_array() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddTypeArray::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
