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

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

namespace {
const uint32_t kOpCompositeExtractIndexLowOrderBits = 0;
const uint32_t kArithmeticInstructionIndexLeftInOperand = 0;
const uint32_t kArithmeticInstructionIndexRightInOperand = 1;
}  // namespace

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(
        protobufs::TransformationReplaceAddSubMulWithCarryingExtended message)
    : message_(std::move(message)) {}

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(uint32_t struct_fresh_id,
                                                       uint32_t result_id) {
  message_.set_struct_fresh_id(struct_fresh_id);
  message_.set_result_id(result_id);
}

bool TransformationReplaceAddSubMulWithCarryingExtended::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.struct_fresh_id())) {
    return false;
  }

  // |message_.result_id| must refer to a suitable OpIAdd, OpISub or OpIMul
  // instruction. The instruction must be defined.
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (instruction == nullptr) {
    return false;
  }
  if (!TransformationReplaceAddSubMulWithCarryingExtended::
          IsInstructionSuitable(ir_context, *instruction)) {
    return false;
  }

  // The struct type for holding the intermediate result must exist in the
  // module. The struct type is based on the operand type.
  uint32_t operand_type_id = ir_context->get_def_use_mgr()
                                 ->GetDef(instruction->GetSingleWordInOperand(
                                     kArithmeticInstructionIndexLeftInOperand))
                                 ->type_id();

  uint32_t struct_type_id = fuzzerutil::MaybeGetStructType(
      ir_context, {operand_type_id, operand_type_id});
  if (struct_type_id == 0) {
    return false;
  }
  return true;
}

void TransformationReplaceAddSubMulWithCarryingExtended::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.struct_fresh_id()) &&
         "|message_.struct_fresh_id| must be fresh");

  // Get the signedness of an operand if it is an int or the signedness of a
  // component if it is a vector.
  auto type_id =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id())->type_id();
  auto type = ir_context->get_type_mgr()->GetType(type_id);
  bool operand_is_signed;
  if (type->kind() == opt::analysis::Type::kVector) {
    auto operand_type = type->AsVector()->element_type();
    operand_is_signed = operand_type->AsInteger()->IsSigned();
  } else {
    operand_is_signed = type->AsInteger()->IsSigned();
  }

  auto original_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.struct_fresh_id());

  // Determine the opcode of the new instruction that computes the result into a
  // struct.
  spv::Op new_instruction_opcode;

  switch (original_instruction->opcode()) {
    case spv::Op::OpIAdd:
      new_instruction_opcode = spv::Op::OpIAddCarry;
      break;
    case spv::Op::OpISub:
      new_instruction_opcode = spv::Op::OpISubBorrow;
      break;
    case spv::Op::OpIMul:
      if (!operand_is_signed) {
        new_instruction_opcode = spv::Op::OpUMulExtended;
      } else {
        new_instruction_opcode = spv::Op::OpSMulExtended;
      }
      break;
    default:
      assert(false && "The instruction has an unsupported opcode.");
      return;
  }
  // Get the type of struct type id holding the intermediate result based on the
  // operand type.
  uint32_t operand_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(original_instruction->GetSingleWordInOperand(
              kArithmeticInstructionIndexLeftInOperand))
          ->type_id();

  uint32_t struct_type_id = fuzzerutil::MaybeGetStructType(
      ir_context, {operand_type_id, operand_type_id});
  // Avoid unused variables in release mode.
  (void)struct_type_id;
  assert(struct_type_id && "The struct type must exist in the module.");

  // Insert the new instruction that computes the result into a struct before
  // the |original_instruction|.
  original_instruction->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, new_instruction_opcode, struct_type_id,
      message_.struct_fresh_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID,
            {original_instruction->GetSingleWordInOperand(
                kArithmeticInstructionIndexLeftInOperand)}},
           {SPV_OPERAND_TYPE_ID,
            {original_instruction->GetSingleWordInOperand(
                kArithmeticInstructionIndexRightInOperand)}}})));

  // Insert the OpCompositeExtract after the added instruction. This instruction
  // takes the first component of the struct which represents low-order bits of
  // the operation. This is the original result.
  original_instruction->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpCompositeExtract, original_instruction->type_id(),
      message_.result_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.struct_fresh_id()}},
           {SPV_OPERAND_TYPE_LITERAL_INTEGER,
            {kOpCompositeExtractIndexLowOrderBits}}})));

  // Remove the original instruction.
  ir_context->KillInst(original_instruction);

  // We have modified the module so most analyzes are now invalid.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

bool TransformationReplaceAddSubMulWithCarryingExtended::IsInstructionSuitable(
    opt::IRContext* ir_context, const opt::Instruction& instruction) {
  auto instruction_opcode = instruction.opcode();

  // Only instructions OpIAdd, OpISub, OpIMul are supported.
  switch (instruction_opcode) {
    case spv::Op::OpIAdd:
    case spv::Op::OpISub:
    case spv::Op::OpIMul:
      break;
    default:
      return false;
  }
  uint32_t operand_1_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction.GetSingleWordInOperand(
              kArithmeticInstructionIndexLeftInOperand))
          ->type_id();

  uint32_t operand_2_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction.GetSingleWordInOperand(
              kArithmeticInstructionIndexRightInOperand))
          ->type_id();

  uint32_t result_type_id = instruction.type_id();

  // Both type ids of the operands and the result type ids must be equal.
  if (operand_1_type_id != operand_2_type_id) {
    return false;
  }
  if (operand_2_type_id != result_type_id) {
    return false;
  }

  // In case of OpIAdd and OpISub, the type must be unsigned.
  auto type = ir_context->get_type_mgr()->GetType(instruction.type_id());

  switch (instruction_opcode) {
    case spv::Op::OpIAdd:
    case spv::Op::OpISub: {
      // In case of OpIAdd and OpISub if the operand is a vector, the component
      // type must be unsigned. Otherwise (if the operand is an int), the
      // operand must be unsigned.
      bool operand_is_signed =
          type->AsVector()
              ? type->AsVector()->element_type()->AsInteger()->IsSigned()
              : type->AsInteger()->IsSigned();
      if (operand_is_signed) {
        return false;
      }
    } break;
    default:
      break;
  }
  return true;
}

protobufs::Transformation
TransformationReplaceAddSubMulWithCarryingExtended::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_add_sub_mul_with_carrying_extended() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceAddSubMulWithCarryingExtended::GetFreshIds() const {
  return {message_.struct_fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
