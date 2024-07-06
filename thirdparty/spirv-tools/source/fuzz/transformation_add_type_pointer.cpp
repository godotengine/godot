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

#include "source/fuzz/transformation_add_type_pointer.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypePointer::TransformationAddTypePointer(
    protobufs::TransformationAddTypePointer message)
    : message_(std::move(message)) {}

TransformationAddTypePointer::TransformationAddTypePointer(
    uint32_t fresh_id, spv::StorageClass storage_class, uint32_t base_type_id) {
  message_.set_fresh_id(fresh_id);
  message_.set_storage_class(uint32_t(storage_class));
  message_.set_base_type_id(base_type_id);
}

bool TransformationAddTypePointer::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The base type must be known.
  return ir_context->get_type_mgr()->GetType(message_.base_type_id()) !=
         nullptr;
}

void TransformationAddTypePointer::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Add the pointer type.
  opt::Instruction::OperandList in_operands = {
      {SPV_OPERAND_TYPE_STORAGE_CLASS, {message_.storage_class()}},
      {SPV_OPERAND_TYPE_ID, {message_.base_type_id()}}};
  auto type_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpTypePointer, 0, message_.fresh_id(), in_operands);
  auto type_instruction_ptr = type_instruction.get();
  ir_context->module()->AddType(std::move(type_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Inform the def use manager that there is a new definition. Invalidate the
  // type manager since we have added a new type.
  ir_context->get_def_use_mgr()->AnalyzeInstDef(type_instruction_ptr);
  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisTypes);
}

protobufs::Transformation TransformationAddTypePointer::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_pointer() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddTypePointer::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
