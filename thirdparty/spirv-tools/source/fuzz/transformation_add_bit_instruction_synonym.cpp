// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_add_bit_instruction_synonym.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddBitInstructionSynonym::TransformationAddBitInstructionSynonym(
    protobufs::TransformationAddBitInstructionSynonym message)
    : message_(std::move(message)) {}

TransformationAddBitInstructionSynonym::TransformationAddBitInstructionSynonym(
    const uint32_t instruction_result_id,
    const std::vector<uint32_t>& fresh_ids) {
  message_.set_instruction_result_id(instruction_result_id);
  *message_.mutable_fresh_ids() =
      google::protobuf::RepeatedField<google::protobuf::uint32>(
          fresh_ids.begin(), fresh_ids.end());
}

bool TransformationAddBitInstructionSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());

  // Checks on: only integer operands are supported, instructions are bitwise
  // operations only. Signedness of the operands must be the same.
  if (!IsInstructionSupported(ir_context, instruction)) {
    return false;
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3791):
  //  This condition could be relaxed if the index exists as another integer
  //  type.
  // All bit indexes must be defined as 32-bit unsigned integers.
  uint32_t width = ir_context->get_type_mgr()
                       ->GetType(instruction->type_id())
                       ->AsInteger()
                       ->width();
  for (uint32_t i = 0; i < width; i++) {
    if (!fuzzerutil::MaybeGetIntegerConstant(ir_context, transformation_context,
                                             {i}, 32, false, false)) {
      return false;
    }
  }

  // |message_.fresh_ids.size| must have the exact number of fresh ids required
  // to apply the transformation.
  if (static_cast<uint32_t>(message_.fresh_ids().size()) !=
      GetRequiredFreshIdCount(ir_context, instruction)) {
    return false;
  }

  // All ids in |message_.fresh_ids| must be fresh.
  for (uint32_t fresh_id : message_.fresh_ids()) {
    if (!fuzzerutil::IsFreshId(ir_context, fresh_id)) {
      return false;
    }
  }

  return true;
}

void TransformationAddBitInstructionSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto bit_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());

  // Use an appropriate helper function to add the new instruction and new
  // synonym fact.  The helper function should take care of invalidating
  // analyses before adding facts.
  switch (bit_instruction->opcode()) {
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
      AddOpBitwiseOrOpNotSynonym(ir_context, transformation_context,
                                 bit_instruction);
      break;
    default:
      assert(false && "Should be unreachable.");
      break;
  }
}

bool TransformationAddBitInstructionSynonym::IsInstructionSupported(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3557):
  //  Right now we only support certain operations. When this issue is addressed
  //  the following conditional can use the function |spvOpcodeIsBit|.
  // |instruction| must be defined and must be a supported bit instruction.
  if (!instruction || (instruction->opcode() != spv::Op::OpBitwiseOr &&
                       instruction->opcode() != spv::Op::OpBitwiseXor &&
                       instruction->opcode() != spv::Op::OpBitwiseAnd &&
                       instruction->opcode() != spv::Op::OpNot)) {
    return false;
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3792):
  //  Right now, only integer operands are supported.
  if (ir_context->get_type_mgr()->GetType(instruction->type_id())->AsVector()) {
    return false;
  }

  if (instruction->opcode() == spv::Op::OpNot) {
    auto operand = instruction->GetInOperand(0).words[0];
    auto operand_inst = ir_context->get_def_use_mgr()->GetDef(operand);
    auto operand_type =
        ir_context->get_type_mgr()->GetType(operand_inst->type_id());
    auto operand_sign = operand_type->AsInteger()->IsSigned();

    auto type_id_sign = ir_context->get_type_mgr()
                            ->GetType(instruction->type_id())
                            ->AsInteger()
                            ->IsSigned();

    return operand_sign == type_id_sign;

  } else {
    // Other BitWise operations that takes two operands.
    auto first_operand = instruction->GetInOperand(0).words[0];
    auto first_operand_inst =
        ir_context->get_def_use_mgr()->GetDef(first_operand);
    auto first_operand_type =
        ir_context->get_type_mgr()->GetType(first_operand_inst->type_id());
    auto first_operand_sign = first_operand_type->AsInteger()->IsSigned();

    auto second_operand = instruction->GetInOperand(1).words[0];
    auto second_operand_inst =
        ir_context->get_def_use_mgr()->GetDef(second_operand);
    auto second_operand_type =
        ir_context->get_type_mgr()->GetType(second_operand_inst->type_id());
    auto second_operand_sign = second_operand_type->AsInteger()->IsSigned();

    auto type_id_sign = ir_context->get_type_mgr()
                            ->GetType(instruction->type_id())
                            ->AsInteger()
                            ->IsSigned();

    return first_operand_sign == second_operand_sign &&
           first_operand_sign == type_id_sign;
  }
}

protobufs::Transformation TransformationAddBitInstructionSynonym::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_add_bit_instruction_synonym() = message_;
  return result;
}

uint32_t TransformationAddBitInstructionSynonym::GetRequiredFreshIdCount(
    opt::IRContext* ir_context, opt::Instruction* bit_instruction) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3557):
  //  Right now, only certain operations are supported.
  switch (bit_instruction->opcode()) {
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
      return (2 + bit_instruction->NumInOperands()) *
                 ir_context->get_type_mgr()
                     ->GetType(bit_instruction->type_id())
                     ->AsInteger()
                     ->width() -
             1;
    default:
      assert(false && "Unsupported bit instruction.");
      return 0;
  }
}

void TransformationAddBitInstructionSynonym::AddOpBitwiseOrOpNotSynonym(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    opt::Instruction* bit_instruction) const {
  // Fresh id iterator.
  auto fresh_id = message_.fresh_ids().begin();

  // |width| is the bit width of operands (8, 16, 32 or 64).
  const uint32_t width = ir_context->get_type_mgr()
                             ->GetType(bit_instruction->type_id())
                             ->AsInteger()
                             ->width();

  // |count| is the number of bits to be extracted and inserted at a time.
  const uint32_t count = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {1}, 32, false, false);

  // |extracted_bit_instructions| is the collection of OpBiwise* or OpNot
  // instructions that evaluate the extracted bits. Those ids will be used to
  // insert the result bits.
  std::vector<uint32_t> extracted_bit_instructions(width);

  for (uint32_t i = 0; i < width; i++) {
    // |offset| is the current bit index.
    uint32_t offset = fuzzerutil::MaybeGetIntegerConstant(
        ir_context, *transformation_context, {i}, 32, false, false);

    // |bit_extract_ids| are the two extracted bits from the operands.
    opt::Instruction::OperandList bit_extract_ids;

    // Extracts the i-th bit from operands.
    for (auto operand = bit_instruction->begin() + 2;
         operand != bit_instruction->end(); operand++) {
      auto bit_extract =
          opt::Instruction(ir_context, spv::Op::OpBitFieldUExtract,
                           bit_instruction->type_id(), *fresh_id++,
                           {{SPV_OPERAND_TYPE_ID, operand->words},
                            {SPV_OPERAND_TYPE_ID, {offset}},
                            {SPV_OPERAND_TYPE_ID, {count}}});
      bit_instruction->InsertBefore(MakeUnique<opt::Instruction>(bit_extract));
      fuzzerutil::UpdateModuleIdBound(ir_context, bit_extract.result_id());
      bit_extract_ids.push_back(
          {SPV_OPERAND_TYPE_ID, {bit_extract.result_id()}});
    }

    // Applies |bit_instruction| to the extracted bits.
    auto extracted_bit_instruction = opt::Instruction(
        ir_context, bit_instruction->opcode(), bit_instruction->type_id(),
        *fresh_id++, bit_extract_ids);
    bit_instruction->InsertBefore(
        MakeUnique<opt::Instruction>(extracted_bit_instruction));
    fuzzerutil::UpdateModuleIdBound(ir_context,
                                    extracted_bit_instruction.result_id());
    extracted_bit_instructions[i] = extracted_bit_instruction.result_id();
  }

  // The first two ids in |extracted_bit_instructions| are used to insert the
  // first two bits of the result.
  uint32_t offset = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {1}, 32, false, false);
  auto bit_insert =
      opt::Instruction(ir_context, spv::Op::OpBitFieldInsert,
                       bit_instruction->type_id(), *fresh_id++,
                       {{SPV_OPERAND_TYPE_ID, {extracted_bit_instructions[0]}},
                        {SPV_OPERAND_TYPE_ID, {extracted_bit_instructions[1]}},
                        {SPV_OPERAND_TYPE_ID, {offset}},
                        {SPV_OPERAND_TYPE_ID, {count}}});
  bit_instruction->InsertBefore(MakeUnique<opt::Instruction>(bit_insert));
  fuzzerutil::UpdateModuleIdBound(ir_context, bit_insert.result_id());

  // Inserts the remaining bits.
  for (uint32_t i = 2; i < width; i++) {
    offset = fuzzerutil::MaybeGetIntegerConstant(
        ir_context, *transformation_context, {i}, 32, false, false);
    bit_insert = opt::Instruction(
        ir_context, spv::Op::OpBitFieldInsert, bit_instruction->type_id(),
        *fresh_id++,
        {{SPV_OPERAND_TYPE_ID, {bit_insert.result_id()}},
         {SPV_OPERAND_TYPE_ID, {extracted_bit_instructions[i]}},
         {SPV_OPERAND_TYPE_ID, {offset}},
         {SPV_OPERAND_TYPE_ID, {count}}});
    bit_instruction->InsertBefore(MakeUnique<opt::Instruction>(bit_insert));
    fuzzerutil::UpdateModuleIdBound(ir_context, bit_insert.result_id());
  }

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // We only add a synonym fact if the bit instruction is not irrelevant, and if
  // the new result id we would make it synonymous with is not irrelevant.  (It
  // could be irrelevant if we are in a dead block.)
  if (!transformation_context->GetFactManager()->IdIsIrrelevant(
          bit_instruction->result_id()) &&
      !transformation_context->GetFactManager()->IdIsIrrelevant(
          bit_insert.result_id())) {
    // Adds the fact that the last |bit_insert| instruction is synonymous of
    // |bit_instruction|.
    transformation_context->GetFactManager()->AddFactDataSynonym(
        MakeDataDescriptor(bit_insert.result_id(), {}),
        MakeDataDescriptor(bit_instruction->result_id(), {}));
  }
}

std::unordered_set<uint32_t>
TransformationAddBitInstructionSynonym::GetFreshIds() const {
  std::unordered_set<uint32_t> result;
  for (auto id : message_.fresh_ids()) {
    result.insert(id);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
