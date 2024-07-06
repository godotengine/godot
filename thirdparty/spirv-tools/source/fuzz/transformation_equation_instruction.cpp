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

#include "source/fuzz/transformation_equation_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationEquationInstruction::TransformationEquationInstruction(
    protobufs::TransformationEquationInstruction message)
    : message_(std::move(message)) {}

TransformationEquationInstruction::TransformationEquationInstruction(
    uint32_t fresh_id, spv::Op opcode,
    const std::vector<uint32_t>& in_operand_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_opcode(uint32_t(opcode));
  for (auto id : in_operand_id) {
    message_.add_in_operand_id(id);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationEquationInstruction::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // The instruction to insert before must exist.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!insert_before) {
    return false;
  }
  // The input ids must all exist, not be OpUndef, not be irrelevant, and be
  // available before this instruction.
  for (auto id : message_.in_operand_id()) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(id);
    if (!inst) {
      return false;
    }
    if (inst->opcode() == spv::Op::OpUndef) {
      return false;
    }
    if (transformation_context.GetFactManager()->IdIsIrrelevant(id)) {
      return false;
    }
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    id)) {
      return false;
    }
  }

  return MaybeGetResultTypeId(ir_context) != 0;
}

void TransformationEquationInstruction::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  opt::Instruction::OperandList in_operands;
  std::vector<uint32_t> rhs_id;
  for (auto id : message_.in_operand_id()) {
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
    rhs_id.push_back(id);
  }

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  opt::Instruction* new_instruction =
      insert_before->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, static_cast<spv::Op>(message_.opcode()),
          MaybeGetResultTypeId(ir_context), message_.fresh_id(),
          std::move(in_operands)));

  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction);
  ir_context->set_instr_block(new_instruction,
                              ir_context->get_instr_block(insert_before));

  // Add an equation fact as long as the result id is not irrelevant (it could
  // be if we are inserting into a dead block).
  if (!transformation_context->GetFactManager()->IdIsIrrelevant(
          message_.fresh_id())) {
    transformation_context->GetFactManager()->AddFactIdEquation(
        message_.fresh_id(), static_cast<spv::Op>(message_.opcode()), rhs_id);
  }
}

protobufs::Transformation TransformationEquationInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_equation_instruction() = message_;
  return result;
}

uint32_t TransformationEquationInstruction::MaybeGetResultTypeId(
    opt::IRContext* ir_context) const {
  auto opcode = static_cast<spv::Op>(message_.opcode());
  switch (opcode) {
    case spv::Op::OpConvertUToF:
    case spv::Op::OpConvertSToF: {
      if (message_.in_operand_id_size() != 1) {
        return 0;
      }

      const auto* type = ir_context->get_type_mgr()->GetType(
          fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0)));
      if (!type) {
        return 0;
      }

      if (const auto* vector = type->AsVector()) {
        if (!vector->element_type()->AsInteger()) {
          return 0;
        }

        if (auto element_type_id = fuzzerutil::MaybeGetFloatType(
                ir_context, vector->element_type()->AsInteger()->width())) {
          return fuzzerutil::MaybeGetVectorType(ir_context, element_type_id,
                                                vector->element_count());
        }

        return 0;
      } else {
        if (!type->AsInteger()) {
          return 0;
        }

        return fuzzerutil::MaybeGetFloatType(ir_context,
                                             type->AsInteger()->width());
      }
    }
    case spv::Op::OpBitcast: {
      if (message_.in_operand_id_size() != 1) {
        return 0;
      }

      const auto* operand_inst =
          ir_context->get_def_use_mgr()->GetDef(message_.in_operand_id(0));
      if (!operand_inst) {
        return 0;
      }

      const auto* operand_type =
          ir_context->get_type_mgr()->GetType(operand_inst->type_id());
      if (!operand_type) {
        return 0;
      }

      // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3539):
      //  The only constraint on the types of OpBitcast's parameters is that
      //  they must have the same number of bits. Consider improving the code
      //  below to support this in full.
      if (const auto* vector = operand_type->AsVector()) {
        uint32_t component_type_id;
        if (const auto* int_type = vector->element_type()->AsInteger()) {
          component_type_id =
              fuzzerutil::MaybeGetFloatType(ir_context, int_type->width());
        } else if (const auto* float_type = vector->element_type()->AsFloat()) {
          component_type_id = fuzzerutil::MaybeGetIntegerType(
              ir_context, float_type->width(), true);
          if (component_type_id == 0 ||
              fuzzerutil::MaybeGetVectorType(ir_context, component_type_id,
                                             vector->element_count()) == 0) {
            component_type_id = fuzzerutil::MaybeGetIntegerType(
                ir_context, float_type->width(), false);
          }
        } else {
          assert(false && "Only vectors of numerical components are supported");
          return 0;
        }

        if (component_type_id == 0) {
          return 0;
        }

        return fuzzerutil::MaybeGetVectorType(ir_context, component_type_id,
                                              vector->element_count());
      } else if (const auto* int_type = operand_type->AsInteger()) {
        return fuzzerutil::MaybeGetFloatType(ir_context, int_type->width());
      } else if (const auto* float_type = operand_type->AsFloat()) {
        if (auto existing_id = fuzzerutil::MaybeGetIntegerType(
                ir_context, float_type->width(), true)) {
          return existing_id;
        }

        return fuzzerutil::MaybeGetIntegerType(ir_context, float_type->width(),
                                               false);
      } else {
        assert(false &&
               "Operand is not a scalar or a vector of numerical type");
        return 0;
      }
    }
    case spv::Op::OpIAdd:
    case spv::Op::OpISub: {
      if (message_.in_operand_id_size() != 2) {
        return 0;
      }
      uint32_t first_operand_width = 0;
      uint32_t first_operand_type_id = 0;
      for (uint32_t index = 0; index < 2; index++) {
        auto operand_inst = ir_context->get_def_use_mgr()->GetDef(
            message_.in_operand_id(index));
        if (!operand_inst || !operand_inst->type_id()) {
          return 0;
        }
        auto operand_type =
            ir_context->get_type_mgr()->GetType(operand_inst->type_id());
        if (!(operand_type->AsInteger() ||
              (operand_type->AsVector() &&
               operand_type->AsVector()->element_type()->AsInteger()))) {
          return 0;
        }
        uint32_t operand_width =
            operand_type->AsInteger()
                ? 1
                : operand_type->AsVector()->element_count();
        if (index == 0) {
          first_operand_width = operand_width;
          first_operand_type_id = operand_inst->type_id();
        } else {
          assert(first_operand_width != 0 &&
                 "The first operand should have been processed.");
          if (operand_width != first_operand_width) {
            return 0;
          }
        }
      }
      assert(first_operand_type_id != 0 &&
             "A type must have been found for the first operand.");
      return first_operand_type_id;
    }
    case spv::Op::OpLogicalNot: {
      if (message_.in_operand_id().size() != 1) {
        return 0;
      }
      auto operand_inst =
          ir_context->get_def_use_mgr()->GetDef(message_.in_operand_id(0));
      if (!operand_inst || !operand_inst->type_id()) {
        return 0;
      }
      auto operand_type =
          ir_context->get_type_mgr()->GetType(operand_inst->type_id());
      if (!(operand_type->AsBool() ||
            (operand_type->AsVector() &&
             operand_type->AsVector()->element_type()->AsBool()))) {
        return 0;
      }
      return operand_inst->type_id();
    }
    case spv::Op::OpSNegate: {
      if (message_.in_operand_id().size() != 1) {
        return 0;
      }
      auto operand_inst =
          ir_context->get_def_use_mgr()->GetDef(message_.in_operand_id(0));
      if (!operand_inst || !operand_inst->type_id()) {
        return 0;
      }
      auto operand_type =
          ir_context->get_type_mgr()->GetType(operand_inst->type_id());
      if (!(operand_type->AsInteger() ||
            (operand_type->AsVector() &&
             operand_type->AsVector()->element_type()->AsInteger()))) {
        return 0;
      }
      return operand_inst->type_id();
    }
    default:
      assert(false && "Inappropriate opcode for equation instruction.");
      return 0;
  }
}

std::unordered_set<uint32_t> TransformationEquationInstruction::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
