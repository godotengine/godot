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

#include "source/fuzz/transformation_add_type_matrix.h"

#include "fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypeMatrix::TransformationAddTypeMatrix(
    protobufs::TransformationAddTypeMatrix message)
    : message_(std::move(message)) {}

TransformationAddTypeMatrix::TransformationAddTypeMatrix(
    uint32_t fresh_id, uint32_t column_type_id, uint32_t column_count) {
  message_.set_fresh_id(fresh_id);
  message_.set_column_type_id(column_type_id);
  message_.set_column_count(column_count);
}

bool TransformationAddTypeMatrix::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The column type must be a floating-point vector.
  auto column_type =
      ir_context->get_type_mgr()->GetType(message_.column_type_id());
  if (!column_type) {
    return false;
  }
  return column_type->AsVector() &&
         column_type->AsVector()->element_type()->AsFloat();
}

void TransformationAddTypeMatrix::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.column_type_id()}});
  in_operands.push_back(
      {SPV_OPERAND_TYPE_LITERAL_INTEGER, {message_.column_count()}});
  auto type_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpTypeMatrix, 0, message_.fresh_id(), in_operands);
  auto type_instruction_ptr = type_instruction.get();
  ir_context->module()->AddType(std::move(type_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Inform the def use manager that there is a new definition. Invalidate the
  // type manager since we have added a new type.
  ir_context->get_def_use_mgr()->AnalyzeInstDef(type_instruction_ptr);
  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisTypes);
}

protobufs::Transformation TransformationAddTypeMatrix::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_matrix() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddTypeMatrix::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
