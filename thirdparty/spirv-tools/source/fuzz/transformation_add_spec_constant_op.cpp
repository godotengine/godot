// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_add_spec_constant_op.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddSpecConstantOp::TransformationAddSpecConstantOp(
    spvtools::fuzz::protobufs::TransformationAddSpecConstantOp message)
    : message_(std::move(message)) {}

TransformationAddSpecConstantOp::TransformationAddSpecConstantOp(
    uint32_t fresh_id, uint32_t type_id, spv::Op opcode,
    const opt::Instruction::OperandList& operands) {
  message_.set_fresh_id(fresh_id);
  message_.set_type_id(type_id);
  message_.set_opcode(uint32_t(opcode));
  for (const auto& operand : operands) {
    auto* op = message_.add_operand();
    op->set_operand_type(operand.type);
    for (auto word : operand.words) {
      op->add_operand_data(word);
    }
  }
}

bool TransformationAddSpecConstantOp::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto clone = fuzzerutil::CloneIRContext(ir_context);
  ApplyImpl(clone.get());
  return fuzzerutil::IsValid(clone.get(),
                             transformation_context.GetValidatorOptions(),
                             fuzzerutil::kSilentMessageConsumer);
}

void TransformationAddSpecConstantOp::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  ApplyImpl(ir_context);
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

void TransformationAddSpecConstantOp::ApplyImpl(
    opt::IRContext* ir_context) const {
  opt::Instruction::OperandList operands = {
      {SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER, {message_.opcode()}}};

  for (const auto& operand : message_.operand()) {
    std::vector<uint32_t> words(operand.operand_data().begin(),
                                operand.operand_data().end());
    operands.push_back({static_cast<spv_operand_type_t>(operand.operand_type()),
                        std::move(words)});
  }

  ir_context->AddGlobalValue(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpSpecConstantOp, message_.type_id(),
      message_.fresh_id(), std::move(operands)));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
}

protobufs::Transformation TransformationAddSpecConstantOp::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_spec_constant_op() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddSpecConstantOp::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
