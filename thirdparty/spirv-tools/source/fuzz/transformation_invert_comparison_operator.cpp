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

#include "source/fuzz/transformation_invert_comparison_operator.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationInvertComparisonOperator::TransformationInvertComparisonOperator(
    protobufs::TransformationInvertComparisonOperator message)
    : message_(std::move(message)) {}

TransformationInvertComparisonOperator::TransformationInvertComparisonOperator(
    uint32_t operator_id, uint32_t fresh_id) {
  message_.set_operator_id(operator_id);
  message_.set_fresh_id(fresh_id);
}

bool TransformationInvertComparisonOperator::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.operator_id| must be valid and inversion must be supported for
  // it.
  auto* inst = ir_context->get_def_use_mgr()->GetDef(message_.operator_id());
  if (!inst || !IsInversionSupported(inst->opcode())) {
    return false;
  }

  // Check that we can insert negation instruction.
  auto* block = ir_context->get_instr_block(inst);
  assert(block && "Instruction must have a basic block");

  auto iter = fuzzerutil::GetIteratorForInstruction(block, inst);
  ++iter;
  assert(iter != block->end() && "Instruction can't be the last in the block");
  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLogicalNot,
                                                      iter) &&
         "Can't insert negation after comparison operator");

  // |message_.fresh_id| must be fresh.
  return fuzzerutil::IsFreshId(ir_context, message_.fresh_id());
}

void TransformationInvertComparisonOperator::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = ir_context->get_def_use_mgr()->GetDef(message_.operator_id());
  assert(inst && "Result id of an operator is invalid");

  // Insert negation after |inst|.
  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);
  ++iter;

  iter.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLogicalNot, inst->type_id(), inst->result_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}}));

  // Change the result id of the original operator to |fresh_id|.
  inst->SetResultId(message_.fresh_id());

  // Invert the operator.
  inst->SetOpcode(InvertOpcode(inst->opcode()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

bool TransformationInvertComparisonOperator::IsInversionSupported(
    spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpSLessThan:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFUnordNotEqual:
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFUnordLessThan:
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFUnordLessThanEqual:
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFUnordGreaterThan:
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
      return true;
    default:
      return false;
  }
}

spv::Op TransformationInvertComparisonOperator::InvertOpcode(spv::Op opcode) {
  assert(IsInversionSupported(opcode) && "Inversion must be supported");

  switch (opcode) {
    case spv::Op::OpSGreaterThan:
      return spv::Op::OpSLessThanEqual;
    case spv::Op::OpSGreaterThanEqual:
      return spv::Op::OpSLessThan;
    case spv::Op::OpSLessThan:
      return spv::Op::OpSGreaterThanEqual;
    case spv::Op::OpSLessThanEqual:
      return spv::Op::OpSGreaterThan;
    case spv::Op::OpUGreaterThan:
      return spv::Op::OpULessThanEqual;
    case spv::Op::OpUGreaterThanEqual:
      return spv::Op::OpULessThan;
    case spv::Op::OpULessThan:
      return spv::Op::OpUGreaterThanEqual;
    case spv::Op::OpULessThanEqual:
      return spv::Op::OpUGreaterThan;
    case spv::Op::OpIEqual:
      return spv::Op::OpINotEqual;
    case spv::Op::OpINotEqual:
      return spv::Op::OpIEqual;
    case spv::Op::OpFOrdEqual:
      return spv::Op::OpFUnordNotEqual;
    case spv::Op::OpFUnordEqual:
      return spv::Op::OpFOrdNotEqual;
    case spv::Op::OpFOrdNotEqual:
      return spv::Op::OpFUnordEqual;
    case spv::Op::OpFUnordNotEqual:
      return spv::Op::OpFOrdEqual;
    case spv::Op::OpFOrdLessThan:
      return spv::Op::OpFUnordGreaterThanEqual;
    case spv::Op::OpFUnordLessThan:
      return spv::Op::OpFOrdGreaterThanEqual;
    case spv::Op::OpFOrdLessThanEqual:
      return spv::Op::OpFUnordGreaterThan;
    case spv::Op::OpFUnordLessThanEqual:
      return spv::Op::OpFOrdGreaterThan;
    case spv::Op::OpFOrdGreaterThan:
      return spv::Op::OpFUnordLessThanEqual;
    case spv::Op::OpFUnordGreaterThan:
      return spv::Op::OpFOrdLessThanEqual;
    case spv::Op::OpFOrdGreaterThanEqual:
      return spv::Op::OpFUnordLessThan;
    case spv::Op::OpFUnordGreaterThanEqual:
      return spv::Op::OpFOrdLessThan;
    default:
      // The program will fail in the debug mode because of the assertion
      // at the beginning of the function.
      return spv::Op::OpNop;
  }
}

protobufs::Transformation TransformationInvertComparisonOperator::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_invert_comparison_operator() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationInvertComparisonOperator::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
