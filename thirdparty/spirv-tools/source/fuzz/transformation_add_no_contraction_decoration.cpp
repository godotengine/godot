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

#include "source/fuzz/transformation_add_no_contraction_decoration.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddNoContractionDecoration::
    TransformationAddNoContractionDecoration(
        protobufs::TransformationAddNoContractionDecoration message)
    : message_(std::move(message)) {}

TransformationAddNoContractionDecoration::
    TransformationAddNoContractionDecoration(uint32_t result_id) {
  message_.set_result_id(result_id);
}

bool TransformationAddNoContractionDecoration::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.result_id| must be the id of an instruction.
  auto instr = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!instr) {
    return false;
  }
  // |instr| must not be decorated with NoContraction.
  if (ir_context->get_decoration_mgr()->HasDecoration(
          message_.result_id(), spv::Decoration::NoContraction)) {
    return false;
  }
  // The instruction must be arithmetic.
  return IsArithmetic(instr->opcode());
}

void TransformationAddNoContractionDecoration::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Add a NoContraction decoration targeting |message_.result_id|.
  ir_context->get_decoration_mgr()->AddDecoration(
      message_.result_id(), uint32_t(spv::Decoration::NoContraction));
}

protobufs::Transformation TransformationAddNoContractionDecoration::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_add_no_contraction_decoration() = message_;
  return result;
}

bool TransformationAddNoContractionDecoration::IsArithmetic(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpUMod:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpFRem:
    case spv::Op::OpFMod:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpMatrixTimesScalar:
    case spv::Op::OpVectorTimesMatrix:
    case spv::Op::OpMatrixTimesVector:
    case spv::Op::OpMatrixTimesMatrix:
    case spv::Op::OpOuterProduct:
    case spv::Op::OpDot:
    case spv::Op::OpIAddCarry:
    case spv::Op::OpISubBorrow:
    case spv::Op::OpUMulExtended:
    case spv::Op::OpSMulExtended:
    case spv::Op::OpAny:
    case spv::Op::OpAll:
    case spv::Op::OpIsNan:
    case spv::Op::OpIsInf:
    case spv::Op::OpIsFinite:
    case spv::Op::OpIsNormal:
    case spv::Op::OpSignBitSet:
    case spv::Op::OpLessOrGreater:
    case spv::Op::OpOrdered:
    case spv::Op::OpUnordered:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot:
      return true;
    default:
      return false;
  }
}

std::unordered_set<uint32_t>
TransformationAddNoContractionDecoration::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
