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

#include "source/fuzz/transformation_replace_boolean_constant_with_constant_binary.h"

#include <cmath>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"

namespace spvtools {
namespace fuzz {

namespace {

// Given floating-point values |lhs| and |rhs|, and a floating-point binary
// operator |binop|, returns true if it is certain that 'lhs binop rhs'
// evaluates to |required_value|.
template <typename T>
bool float_binop_evaluates_to(T lhs, T rhs, spv::Op binop,
                              bool required_value) {
  // Infinity and NaN values are conservatively treated as out of scope.
  if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
    return false;
  }
  bool binop_result;
  // The following captures the binary operators that spirv-fuzz can actually
  // generate when turning a boolean constant into a binary expression.
  switch (binop) {
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
      binop_result = (lhs >= rhs);
      break;
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFUnordGreaterThan:
      binop_result = (lhs > rhs);
      break;
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFUnordLessThanEqual:
      binop_result = (lhs <= rhs);
      break;
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFUnordLessThan:
      binop_result = (lhs < rhs);
      break;
    default:
      return false;
  }
  return binop_result == required_value;
}

// Analogous to 'float_binop_evaluates_to', but for signed int values.
template <typename T>
bool signed_int_binop_evaluates_to(T lhs, T rhs, spv::Op binop,
                                   bool required_value) {
  bool binop_result;
  switch (binop) {
    case spv::Op::OpSGreaterThanEqual:
      binop_result = (lhs >= rhs);
      break;
    case spv::Op::OpSGreaterThan:
      binop_result = (lhs > rhs);
      break;
    case spv::Op::OpSLessThanEqual:
      binop_result = (lhs <= rhs);
      break;
    case spv::Op::OpSLessThan:
      binop_result = (lhs < rhs);
      break;
    default:
      return false;
  }
  return binop_result == required_value;
}

// Analogous to 'float_binop_evaluates_to', but for unsigned int values.
template <typename T>
bool unsigned_int_binop_evaluates_to(T lhs, T rhs, spv::Op binop,
                                     bool required_value) {
  bool binop_result;
  switch (binop) {
    case spv::Op::OpUGreaterThanEqual:
      binop_result = (lhs >= rhs);
      break;
    case spv::Op::OpUGreaterThan:
      binop_result = (lhs > rhs);
      break;
    case spv::Op::OpULessThanEqual:
      binop_result = (lhs <= rhs);
      break;
    case spv::Op::OpULessThan:
      binop_result = (lhs < rhs);
      break;
    default:
      return false;
  }
  return binop_result == required_value;
}

}  // namespace

TransformationReplaceBooleanConstantWithConstantBinary::
    TransformationReplaceBooleanConstantWithConstantBinary(
        protobufs::TransformationReplaceBooleanConstantWithConstantBinary
            message)
    : message_(std::move(message)) {}

TransformationReplaceBooleanConstantWithConstantBinary::
    TransformationReplaceBooleanConstantWithConstantBinary(
        const protobufs::IdUseDescriptor& id_use_descriptor, uint32_t lhs_id,
        uint32_t rhs_id, spv::Op comparison_opcode,
        uint32_t fresh_id_for_binary_operation) {
  *message_.mutable_id_use_descriptor() = id_use_descriptor;
  message_.set_lhs_id(lhs_id);
  message_.set_rhs_id(rhs_id);
  message_.set_opcode(uint32_t(comparison_opcode));
  message_.set_fresh_id_for_binary_operation(fresh_id_for_binary_operation);
}

bool TransformationReplaceBooleanConstantWithConstantBinary::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The id for the binary result must be fresh
  if (!fuzzerutil::IsFreshId(ir_context,
                             message_.fresh_id_for_binary_operation())) {
    return false;
  }

  // The used id must be for a boolean constant
  auto boolean_constant = ir_context->get_def_use_mgr()->GetDef(
      message_.id_use_descriptor().id_of_interest());
  if (!boolean_constant) {
    return false;
  }
  if (!(boolean_constant->opcode() == spv::Op::OpConstantFalse ||
        boolean_constant->opcode() == spv::Op::OpConstantTrue)) {
    return false;
  }

  // The left-hand-side id must correspond to a constant instruction.
  auto lhs_constant_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.lhs_id());
  if (!lhs_constant_inst) {
    return false;
  }
  if (lhs_constant_inst->opcode() != spv::Op::OpConstant) {
    return false;
  }

  // The right-hand-side id must correspond to a constant instruction.
  auto rhs_constant_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.rhs_id());
  if (!rhs_constant_inst) {
    return false;
  }
  if (rhs_constant_inst->opcode() != spv::Op::OpConstant) {
    return false;
  }

  // The left- and right-hand side instructions must have the same type.
  if (lhs_constant_inst->type_id() != rhs_constant_inst->type_id()) {
    return false;
  }

  // The expression 'LHS opcode RHS' must evaluate to the boolean constant.
  auto lhs_constant =
      ir_context->get_constant_mgr()->FindDeclaredConstant(message_.lhs_id());
  auto rhs_constant =
      ir_context->get_constant_mgr()->FindDeclaredConstant(message_.rhs_id());
  bool expected_result =
      (boolean_constant->opcode() == spv::Op::OpConstantTrue);

  const auto binary_opcode = static_cast<spv::Op>(message_.opcode());

  // We consider the floating point, signed and unsigned integer cases
  // separately.  In each case the logic is very similar.
  if (lhs_constant->AsFloatConstant()) {
    assert(rhs_constant->AsFloatConstant() &&
           "Both constants should be of the same type.");
    if (lhs_constant->type()->AsFloat()->width() == 32) {
      if (!float_binop_evaluates_to(lhs_constant->GetFloat(),
                                    rhs_constant->GetFloat(), binary_opcode,
                                    expected_result)) {
        return false;
      }
    } else {
      assert(lhs_constant->type()->AsFloat()->width() == 64);
      if (!float_binop_evaluates_to(lhs_constant->GetDouble(),
                                    rhs_constant->GetDouble(), binary_opcode,
                                    expected_result)) {
        return false;
      }
    }
  } else {
    assert(lhs_constant->AsIntConstant() && "Constants should be in or float.");
    assert(rhs_constant->AsIntConstant() &&
           "Both constants should be of the same type.");
    if (lhs_constant->type()->AsInteger()->IsSigned()) {
      if (lhs_constant->type()->AsInteger()->width() == 32) {
        if (!signed_int_binop_evaluates_to(lhs_constant->GetS32(),
                                           rhs_constant->GetS32(),
                                           binary_opcode, expected_result)) {
          return false;
        }
      } else {
        assert(lhs_constant->type()->AsInteger()->width() == 64);
        if (!signed_int_binop_evaluates_to(lhs_constant->GetS64(),
                                           rhs_constant->GetS64(),
                                           binary_opcode, expected_result)) {
          return false;
        }
      }
    } else {
      if (lhs_constant->type()->AsInteger()->width() == 32) {
        if (!unsigned_int_binop_evaluates_to(lhs_constant->GetU32(),
                                             rhs_constant->GetU32(),
                                             binary_opcode, expected_result)) {
          return false;
        }
      } else {
        assert(lhs_constant->type()->AsInteger()->width() == 64);
        if (!unsigned_int_binop_evaluates_to(lhs_constant->GetU64(),
                                             rhs_constant->GetU64(),
                                             binary_opcode, expected_result)) {
          return false;
        }
      }
    }
  }

  // The id use descriptor must identify some instruction
  auto instruction =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  if (instruction == nullptr) {
    return false;
  }

  // The instruction must not be an OpVariable, because (a) we cannot insert
  // a binary operator before an OpVariable, but in any case (b) the
  // constant we would be replacing is the initializer constant of the
  // OpVariable, and this cannot be the result of a binary operation.
  if (instruction->opcode() == spv::Op::OpVariable) {
    return false;
  }

  return true;
}

void TransformationReplaceBooleanConstantWithConstantBinary::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  ApplyWithResult(ir_context, transformation_context);
}

opt::Instruction*
TransformationReplaceBooleanConstantWithConstantBinary::ApplyWithResult(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  opt::analysis::Bool bool_type;
  opt::Instruction::OperandList operands = {
      {SPV_OPERAND_TYPE_ID, {message_.lhs_id()}},
      {SPV_OPERAND_TYPE_ID, {message_.rhs_id()}}};
  auto binary_instruction = MakeUnique<opt::Instruction>(
      ir_context, static_cast<spv::Op>(message_.opcode()),
      ir_context->get_type_mgr()->GetId(&bool_type),
      message_.fresh_id_for_binary_operation(), operands);
  opt::Instruction* result = binary_instruction.get();
  auto instruction_containing_constant_use =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  auto instruction_before_which_to_insert = instruction_containing_constant_use;

  // If |instruction_before_which_to_insert| is an OpPhi instruction,
  // then |binary_instruction| will be inserted into the parent block associated
  // with the OpPhi variable operand.
  if (instruction_containing_constant_use->opcode() == spv::Op::OpPhi) {
    instruction_before_which_to_insert =
        ir_context->cfg()
            ->block(instruction_containing_constant_use->GetSingleWordInOperand(
                message_.id_use_descriptor().in_operand_index() + 1))
            ->terminator();
  }

  // We want to insert the new instruction before the instruction that contains
  // the use of the boolean, but we need to go backwards one more instruction if
  // the using instruction is preceded by a merge instruction.
  {
    opt::Instruction* previous_node =
        instruction_before_which_to_insert->PreviousNode();
    if (previous_node &&
        (previous_node->opcode() == spv::Op::OpLoopMerge ||
         previous_node->opcode() == spv::Op::OpSelectionMerge)) {
      instruction_before_which_to_insert = previous_node;
    }
  }

  instruction_before_which_to_insert->InsertBefore(
      std::move(binary_instruction));
  instruction_containing_constant_use->SetInOperand(
      message_.id_use_descriptor().in_operand_index(),
      {message_.fresh_id_for_binary_operation()});
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.fresh_id_for_binary_operation());
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
  return result;
}

protobufs::Transformation
TransformationReplaceBooleanConstantWithConstantBinary::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_boolean_constant_with_constant_binary() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceBooleanConstantWithConstantBinary::GetFreshIds() const {
  return {message_.fresh_id_for_binary_operation()};
}

}  // namespace fuzz
}  // namespace spvtools
