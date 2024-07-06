// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_add_relaxed_decoration.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddRelaxedDecoration::TransformationAddRelaxedDecoration(
    protobufs::TransformationAddRelaxedDecoration message)
    : message_(std::move(message)) {}

TransformationAddRelaxedDecoration::TransformationAddRelaxedDecoration(
    uint32_t result_id) {
  message_.set_result_id(result_id);
}

bool TransformationAddRelaxedDecoration::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // |message_.result_id| must be the id of an instruction.
  auto instr = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!instr) {
    return false;
  }
  // |instr| must not be decorated with RelaxedPrecision.
  if (ir_context->get_decoration_mgr()->HasDecoration(
          message_.result_id(), spv::Decoration::RelaxedPrecision)) {
    return false;
  }
  opt::BasicBlock* cur_block = ir_context->get_instr_block(instr);
  // The instruction must have a block.
  if (cur_block == nullptr) {
    return false;
  }
  // |cur_block| must be a dead block.
  if (!(transformation_context.GetFactManager()->BlockIsDead(
          cur_block->id()))) {
    return false;
  }

  // The instruction must be numeric.
  return IsNumeric(instr->opcode());
}

void TransformationAddRelaxedDecoration::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Add a RelaxedPrecision decoration targeting |message_.result_id|.
  ir_context->get_decoration_mgr()->AddDecoration(
      message_.result_id(), uint32_t(spv::Decoration::RelaxedPrecision));
}

protobufs::Transformation TransformationAddRelaxedDecoration::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_add_relaxed_decoration() = message_;
  return result;
}

bool TransformationAddRelaxedDecoration::IsNumeric(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpConvertPtrToU:
    case spv::Op::OpSatConvertSToU:
    case spv::Op::OpSatConvertUToS:
    case spv::Op::OpVectorExtractDynamic:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpVectorShuffle:
    case spv::Op::OpTranspose:
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
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
    case spv::Op::OpBitFieldInsert:
    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicStore:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
      return true;
    default:
      return false;
  }
}

std::unordered_set<uint32_t> TransformationAddRelaxedDecoration::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
