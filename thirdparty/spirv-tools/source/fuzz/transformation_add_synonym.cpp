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

#include "source/fuzz/transformation_add_synonym.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddSynonym::TransformationAddSynonym(
    protobufs::TransformationAddSynonym message)
    : message_(std::move(message)) {}

TransformationAddSynonym::TransformationAddSynonym(
    uint32_t result_id,
    protobufs::TransformationAddSynonym::SynonymType synonym_type,
    uint32_t synonym_fresh_id,
    const protobufs::InstructionDescriptor& insert_before) {
  message_.set_result_id(result_id);
  message_.set_synonym_type(synonym_type);
  message_.set_synonym_fresh_id(synonym_fresh_id);
  *message_.mutable_insert_before() = insert_before;
}

bool TransformationAddSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  assert(protobufs::TransformationAddSynonym::SynonymType_IsValid(
             message_.synonym_type()) &&
         "Synonym type is invalid");

  // |synonym_fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.synonym_fresh_id())) {
    return false;
  }

  // Check that |message_.result_id| is valid.
  auto* synonym = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!synonym) {
    return false;
  }

  // Check that we can apply |synonym_type| to |result_id|.
  if (!IsInstructionValid(ir_context, transformation_context, synonym,
                          message_.synonym_type())) {
    return false;
  }

  // Check that |insert_before| is valid.
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  if (!insert_before_inst) {
    return false;
  }

  const auto* insert_before_inst_block =
      ir_context->get_instr_block(insert_before_inst);
  assert(insert_before_inst_block &&
         "|insert_before_inst| must be in some block");

  if (transformation_context.GetFactManager()->BlockIsDead(
          insert_before_inst_block->id())) {
    // We don't create synonyms in dead blocks.
    return false;
  }

  // Check that we can insert |message._synonymous_instruction| before
  // |message_.insert_before| instruction. We use OpIAdd to represent some
  // instruction that can produce a synonym.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpIAdd,
                                                    insert_before_inst)) {
    return false;
  }

  // A constant instruction must be present in the module if required.
  if (IsAdditionalConstantRequired(message_.synonym_type()) &&
      MaybeGetConstantId(ir_context, transformation_context) == 0) {
    return false;
  }

  // Domination rules must be satisfied.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, insert_before_inst, message_.result_id());
}

void TransformationAddSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Add a synonymous instruction.
  auto new_instruction =
      MakeSynonymousInstruction(ir_context, *transformation_context);
  auto new_instruction_ptr = new_instruction.get();
  auto insert_before = FindInstruction(message_.insert_before(), ir_context);
  insert_before->InsertBefore(std::move(new_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.synonym_fresh_id());

  // Inform the def-use manager about the new instruction and record its basic
  // block.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr,
                              ir_context->get_instr_block(insert_before));

  // Propagate PointeeValueIsIrrelevant fact.
  const auto* new_synonym_type = ir_context->get_type_mgr()->GetType(
      fuzzerutil::GetTypeId(ir_context, message_.synonym_fresh_id()));
  assert(new_synonym_type && "New synonym should have a valid type");

  if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
          message_.result_id()) &&
      new_synonym_type->AsPointer()) {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.synonym_fresh_id());
  }

  // Mark two ids as synonymous.
  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.result_id(), {}),
      MakeDataDescriptor(message_.synonym_fresh_id(), {}));
}

protobufs::Transformation TransformationAddSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_synonym() = message_;
  return result;
}

bool TransformationAddSynonym::IsInstructionValid(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context, opt::Instruction* inst,
    protobufs::TransformationAddSynonym::SynonymType synonym_type) {
  // Instruction must have a result id, type id. We skip OpUndef and
  // OpConstantNull.
  if (!inst || !inst->result_id() || !inst->type_id() ||
      inst->opcode() == spv::Op::OpUndef ||
      inst->opcode() == spv::Op::OpConstantNull) {
    return false;
  }

  if (!fuzzerutil::CanMakeSynonymOf(ir_context, transformation_context,
                                    *inst)) {
    return false;
  }

  switch (synonym_type) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::MUL_ONE: {
      // The instruction must be either scalar or vector of integers or floats.
      const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
      assert(type && "Instruction's result id is invalid");

      if (const auto* vector = type->AsVector()) {
        return vector->element_type()->AsInteger() ||
               vector->element_type()->AsFloat();
      }

      return type->AsInteger() || type->AsFloat();
    }
    case protobufs::TransformationAddSynonym::BITWISE_OR:
    case protobufs::TransformationAddSynonym::BITWISE_XOR: {
      // The instruction must be either an integer or a vector of integers.
      const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
      assert(type && "Instruction's result id is invalid");

      if (const auto* vector = type->AsVector()) {
        return vector->element_type()->AsInteger();
      }

      return type->AsInteger();
    }
    case protobufs::TransformationAddSynonym::COPY_OBJECT:
      // All checks for OpCopyObject are handled by
      // fuzzerutil::CanMakeSynonymOf.
      return true;
    case protobufs::TransformationAddSynonym::LOGICAL_AND:
    case protobufs::TransformationAddSynonym::LOGICAL_OR: {
      // The instruction must be either a scalar or a vector of booleans.
      const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
      assert(type && "Instruction's result id is invalid");
      return (type->AsVector() && type->AsVector()->element_type()->AsBool()) ||
             type->AsBool();
    }
    default:
      assert(false && "Synonym type is not supported");
      return false;
  }
}

std::unique_ptr<opt::Instruction>
TransformationAddSynonym::MakeSynonymousInstruction(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto synonym_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.result_id());
  assert(synonym_type_id && "Synonym has invalid type id");
  auto opcode = spv::Op::OpNop;
  const auto* synonym_type =
      ir_context->get_type_mgr()->GetType(synonym_type_id);
  assert(synonym_type && "Synonym has invalid type");

  auto is_integral = (synonym_type->AsVector() &&
                      synonym_type->AsVector()->element_type()->AsInteger()) ||
                     synonym_type->AsInteger();

  switch (message_.synonym_type()) {
    case protobufs::TransformationAddSynonym::SUB_ZERO:
      opcode = is_integral ? spv::Op::OpISub : spv::Op::OpFSub;
      break;
    case protobufs::TransformationAddSynonym::MUL_ONE:
      opcode = is_integral ? spv::Op::OpIMul : spv::Op::OpFMul;
      break;
    case protobufs::TransformationAddSynonym::ADD_ZERO:
      opcode = is_integral ? spv::Op::OpIAdd : spv::Op::OpFAdd;
      break;
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
      opcode = spv::Op::OpLogicalOr;
      break;
    case protobufs::TransformationAddSynonym::LOGICAL_AND:
      opcode = spv::Op::OpLogicalAnd;
      break;
    case protobufs::TransformationAddSynonym::BITWISE_OR:
      opcode = spv::Op::OpBitwiseOr;
      break;
    case protobufs::TransformationAddSynonym::BITWISE_XOR:
      opcode = spv::Op::OpBitwiseXor;
      break;

    case protobufs::TransformationAddSynonym::COPY_OBJECT:
      return MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCopyObject, synonym_type_id,
          message_.synonym_fresh_id(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {message_.result_id()}}});

    default:
      assert(false && "Unhandled synonym type");
      return nullptr;
  }

  return MakeUnique<opt::Instruction>(
      ir_context, opcode, synonym_type_id, message_.synonym_fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.result_id()}},
          {SPV_OPERAND_TYPE_ID,
           {MaybeGetConstantId(ir_context, transformation_context)}}});
}

uint32_t TransformationAddSynonym::MaybeGetConstantId(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  assert(IsAdditionalConstantRequired(message_.synonym_type()) &&
         "Synonym type doesn't require an additional constant");

  auto synonym_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.result_id());
  assert(synonym_type_id && "Synonym has invalid type id");

  switch (message_.synonym_type()) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
    case protobufs::TransformationAddSynonym::BITWISE_OR:
    case protobufs::TransformationAddSynonym::BITWISE_XOR:
      return fuzzerutil::MaybeGetZeroConstant(
          ir_context, transformation_context, synonym_type_id, false);
    case protobufs::TransformationAddSynonym::MUL_ONE:
    case protobufs::TransformationAddSynonym::LOGICAL_AND: {
      auto synonym_type = ir_context->get_type_mgr()->GetType(synonym_type_id);
      assert(synonym_type && "Synonym has invalid type");

      if (const auto* vector = synonym_type->AsVector()) {
        auto element_type_id =
            ir_context->get_type_mgr()->GetId(vector->element_type());
        assert(element_type_id && "Vector's element type is invalid");

        auto one_word =
            vector->element_type()->AsFloat() ? fuzzerutil::FloatToWord(1) : 1u;
        if (auto scalar_one_id = fuzzerutil::MaybeGetScalarConstant(
                ir_context, transformation_context, {one_word}, element_type_id,
                false)) {
          return fuzzerutil::MaybeGetCompositeConstant(
              ir_context, transformation_context,
              std::vector<uint32_t>(vector->element_count(), scalar_one_id),
              synonym_type_id, false);
        }

        return 0;
      } else {
        return fuzzerutil::MaybeGetScalarConstant(
            ir_context, transformation_context,
            {synonym_type->AsFloat() ? fuzzerutil::FloatToWord(1) : 1u},
            synonym_type_id, false);
      }
    }
    default:
      // The assertion at the beginning of the function will fail in the debug
      // mode.
      return 0;
  }
}

bool TransformationAddSynonym::IsAdditionalConstantRequired(
    protobufs::TransformationAddSynonym::SynonymType synonym_type) {
  switch (synonym_type) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
    case protobufs::TransformationAddSynonym::MUL_ONE:
    case protobufs::TransformationAddSynonym::LOGICAL_AND:
    case protobufs::TransformationAddSynonym::BITWISE_OR:
    case protobufs::TransformationAddSynonym::BITWISE_XOR:
      return true;
    default:
      return false;
  }
}

std::unordered_set<uint32_t> TransformationAddSynonym::GetFreshIds() const {
  return {message_.synonym_fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
