// Copyright (c) 2017 Google Inc.
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

#include "source/opt/fold.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "source/opt/const_folding_rules.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/folding_rules.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

#ifndef INT32_MIN
#define INT32_MIN (-2147483648)
#endif

#ifndef INT32_MAX
#define INT32_MAX 2147483647
#endif

#ifndef UINT32_MAX
#define UINT32_MAX 0xffffffff /* 4294967295U */
#endif

}  // namespace

uint32_t InstructionFolder::UnaryOperate(SpvOp opcode, uint32_t operand) const {
  switch (opcode) {
    // Arthimetics
    case SpvOp::SpvOpSNegate: {
      int32_t s_operand = static_cast<int32_t>(operand);
      if (s_operand == std::numeric_limits<int32_t>::min()) {
        return s_operand;
      }
      return -s_operand;
    }
    case SpvOp::SpvOpNot:
      return ~operand;
    case SpvOp::SpvOpLogicalNot:
      return !static_cast<bool>(operand);
    default:
      assert(false &&
             "Unsupported unary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

uint32_t InstructionFolder::BinaryOperate(SpvOp opcode, uint32_t a,
                                          uint32_t b) const {
  switch (opcode) {
    // Arthimetics
    case SpvOp::SpvOpIAdd:
      return a + b;
    case SpvOp::SpvOpISub:
      return a - b;
    case SpvOp::SpvOpIMul:
      return a * b;
    case SpvOp::SpvOpUDiv:
      if (b != 0) {
        return a / b;
      } else {
        // Dividing by 0 is undefined, so we will just pick 0.
        return 0;
      }
    case SpvOp::SpvOpSDiv:
      if (b != 0u) {
        return (static_cast<int32_t>(a)) / (static_cast<int32_t>(b));
      } else {
        // Dividing by 0 is undefined, so we will just pick 0.
        return 0;
      }
    case SpvOp::SpvOpSRem: {
      // The sign of non-zero result comes from the first operand: a. This is
      // guaranteed by C++11 rules for integer division operator. The division
      // result is rounded toward zero, so the result of '%' has the sign of
      // the first operand.
      if (b != 0u) {
        return static_cast<int32_t>(a) % static_cast<int32_t>(b);
      } else {
        // Remainder when dividing with 0 is undefined, so we will just pick 0.
        return 0;
      }
    }
    case SpvOp::SpvOpSMod: {
      // The sign of non-zero result comes from the second operand: b
      if (b != 0u) {
        int32_t rem = BinaryOperate(SpvOp::SpvOpSRem, a, b);
        int32_t b_prim = static_cast<int32_t>(b);
        return (rem + b_prim) % b_prim;
      } else {
        // Mod with 0 is undefined, so we will just pick 0.
        return 0;
      }
    }
    case SpvOp::SpvOpUMod:
      if (b != 0u) {
        return (a % b);
      } else {
        // Mod with 0 is undefined, so we will just pick 0.
        return 0;
      }

    // Shifting
    case SpvOp::SpvOpShiftRightLogical:
      if (b >= 32) {
        // This is undefined behaviour when |b| > 32.  Choose 0 for consistency.
        // When |b| == 32, doing the shift in C++ in undefined, but the result
        // will be 0, so just return that value.
        return 0;
      }
      return a >> b;
    case SpvOp::SpvOpShiftRightArithmetic:
      if (b > 32) {
        // This is undefined behaviour.  Choose 0 for consistency.
        return 0;
      }
      if (b == 32) {
        // Doing the shift in C++ is undefined, but the result is defined in the
        // spir-v spec.  Find that value another way.
        if (static_cast<int32_t>(a) >= 0) {
          return 0;
        } else {
          return static_cast<uint32_t>(-1);
        }
      }
      return (static_cast<int32_t>(a)) >> b;
    case SpvOp::SpvOpShiftLeftLogical:
      if (b >= 32) {
        // This is undefined behaviour when |b| > 32.  Choose 0 for consistency.
        // When |b| == 32, doing the shift in C++ in undefined, but the result
        // will be 0, so just return that value.
        return 0;
      }
      return a << b;

    // Bitwise operations
    case SpvOp::SpvOpBitwiseOr:
      return a | b;
    case SpvOp::SpvOpBitwiseAnd:
      return a & b;
    case SpvOp::SpvOpBitwiseXor:
      return a ^ b;

    // Logical
    case SpvOp::SpvOpLogicalEqual:
      return (static_cast<bool>(a)) == (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalNotEqual:
      return (static_cast<bool>(a)) != (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalOr:
      return (static_cast<bool>(a)) || (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalAnd:
      return (static_cast<bool>(a)) && (static_cast<bool>(b));

    // Comparison
    case SpvOp::SpvOpIEqual:
      return a == b;
    case SpvOp::SpvOpINotEqual:
      return a != b;
    case SpvOp::SpvOpULessThan:
      return a < b;
    case SpvOp::SpvOpSLessThan:
      return (static_cast<int32_t>(a)) < (static_cast<int32_t>(b));
    case SpvOp::SpvOpUGreaterThan:
      return a > b;
    case SpvOp::SpvOpSGreaterThan:
      return (static_cast<int32_t>(a)) > (static_cast<int32_t>(b));
    case SpvOp::SpvOpULessThanEqual:
      return a <= b;
    case SpvOp::SpvOpSLessThanEqual:
      return (static_cast<int32_t>(a)) <= (static_cast<int32_t>(b));
    case SpvOp::SpvOpUGreaterThanEqual:
      return a >= b;
    case SpvOp::SpvOpSGreaterThanEqual:
      return (static_cast<int32_t>(a)) >= (static_cast<int32_t>(b));
    default:
      assert(false &&
             "Unsupported binary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

uint32_t InstructionFolder::TernaryOperate(SpvOp opcode, uint32_t a, uint32_t b,
                                           uint32_t c) const {
  switch (opcode) {
    case SpvOp::SpvOpSelect:
      return (static_cast<bool>(a)) ? b : c;
    default:
      assert(false &&
             "Unsupported ternary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

uint32_t InstructionFolder::OperateWords(
    SpvOp opcode, const std::vector<uint32_t>& operand_words) const {
  switch (operand_words.size()) {
    case 1:
      return UnaryOperate(opcode, operand_words.front());
    case 2:
      return BinaryOperate(opcode, operand_words.front(), operand_words.back());
    case 3:
      return TernaryOperate(opcode, operand_words[0], operand_words[1],
                            operand_words[2]);
    default:
      assert(false && "Invalid number of operands");
      return 0;
  }
}

bool InstructionFolder::FoldInstructionInternal(Instruction* inst) const {
  auto identity_map = [](uint32_t id) { return id; };
  Instruction* folded_inst = FoldInstructionToConstant(inst, identity_map);
  if (folded_inst != nullptr) {
    inst->SetOpcode(SpvOpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {folded_inst->result_id()}}});
    return true;
  }

  SpvOp opcode = inst->opcode();
  analysis::ConstantManager* const_manager = context_->get_constant_mgr();

  std::vector<const analysis::Constant*> constants =
      const_manager->GetOperandConstants(inst);

  for (const FoldingRule& rule : GetFoldingRules().GetRulesForOpcode(opcode)) {
    if (rule(context_, inst, constants)) {
      return true;
    }
  }
  return false;
}

// Returns the result of performing an operation on scalar constant operands.
// This function extracts the operand values as 32 bit words and returns the
// result in 32 bit word. Scalar constants with longer than 32-bit width are
// not accepted in this function.
uint32_t InstructionFolder::FoldScalars(
    SpvOp opcode,
    const std::vector<const analysis::Constant*>& operands) const {
  assert(IsFoldableOpcode(opcode) &&
         "Unhandled instruction opcode in FoldScalars");
  std::vector<uint32_t> operand_values_in_raw_words;
  for (const auto& operand : operands) {
    if (const analysis::ScalarConstant* scalar = operand->AsScalarConstant()) {
      const auto& scalar_words = scalar->words();
      assert(scalar_words.size() == 1 &&
             "Scalar constants with longer than 32-bit width are not allowed "
             "in FoldScalars()");
      operand_values_in_raw_words.push_back(scalar_words.front());
    } else if (operand->AsNullConstant()) {
      operand_values_in_raw_words.push_back(0u);
    } else {
      assert(false &&
             "FoldScalars() only accepts ScalarConst or NullConst type of "
             "constant");
    }
  }
  return OperateWords(opcode, operand_values_in_raw_words);
}

bool InstructionFolder::FoldBinaryIntegerOpToConstant(
    Instruction* inst, const std::function<uint32_t(uint32_t)>& id_map,
    uint32_t* result) const {
  SpvOp opcode = inst->opcode();
  analysis::ConstantManager* const_manger = context_->get_constant_mgr();

  uint32_t ids[2];
  const analysis::IntConstant* constants[2];
  for (uint32_t i = 0; i < 2; i++) {
    const Operand* operand = &inst->GetInOperand(i);
    if (operand->type != SPV_OPERAND_TYPE_ID) {
      return false;
    }
    ids[i] = id_map(operand->words[0]);
    const analysis::Constant* constant =
        const_manger->FindDeclaredConstant(ids[i]);
    constants[i] = (constant != nullptr ? constant->AsIntConstant() : nullptr);
  }

  switch (opcode) {
    // Arthimetics
    case SpvOp::SpvOpIMul:
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr && constants[i]->IsZero()) {
          *result = 0;
          return true;
        }
      }
      break;
    case SpvOp::SpvOpUDiv:
    case SpvOp::SpvOpSDiv:
    case SpvOp::SpvOpSRem:
    case SpvOp::SpvOpSMod:
    case SpvOp::SpvOpUMod:
      // This changes undefined behaviour (ie divide by 0) into a 0.
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr && constants[i]->IsZero()) {
          *result = 0;
          return true;
        }
      }
      break;

    // Shifting
    case SpvOp::SpvOpShiftRightLogical:
    case SpvOp::SpvOpShiftLeftLogical:
      if (constants[1] != nullptr) {
        // When shifting by a value larger than the size of the result, the
        // result is undefined.  We are setting the undefined behaviour to a
        // result of 0.  If the shift amount is the same as the size of the
        // result, then the result is defined, and it 0.
        uint32_t shift_amount = constants[1]->GetU32BitValue();
        if (shift_amount >= 32) {
          *result = 0;
          return true;
        }
      }
      break;

    // Bitwise operations
    case SpvOp::SpvOpBitwiseOr:
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr) {
          // TODO: Change the mask against a value based on the bit width of the
          // instruction result type.  This way we can handle say 16-bit values
          // as well.
          uint32_t mask = constants[i]->GetU32BitValue();
          if (mask == 0xFFFFFFFF) {
            *result = 0xFFFFFFFF;
            return true;
          }
        }
      }
      break;
    case SpvOp::SpvOpBitwiseAnd:
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr) {
          if (constants[i]->IsZero()) {
            *result = 0;
            return true;
          }
        }
      }
      break;

    // Comparison
    case SpvOp::SpvOpULessThan:
      if (constants[0] != nullptr &&
          constants[0]->GetU32BitValue() == UINT32_MAX) {
        *result = false;
        return true;
      }
      if (constants[1] != nullptr && constants[1]->GetU32BitValue() == 0) {
        *result = false;
        return true;
      }
      break;
    case SpvOp::SpvOpSLessThan:
      if (constants[0] != nullptr &&
          constants[0]->GetS32BitValue() == INT32_MAX) {
        *result = false;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetS32BitValue() == INT32_MIN) {
        *result = false;
        return true;
      }
      break;
    case SpvOp::SpvOpUGreaterThan:
      if (constants[0] != nullptr && constants[0]->IsZero()) {
        *result = false;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetU32BitValue() == UINT32_MAX) {
        *result = false;
        return true;
      }
      break;
    case SpvOp::SpvOpSGreaterThan:
      if (constants[0] != nullptr &&
          constants[0]->GetS32BitValue() == INT32_MIN) {
        *result = false;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetS32BitValue() == INT32_MAX) {
        *result = false;
        return true;
      }
      break;
    case SpvOp::SpvOpULessThanEqual:
      if (constants[0] != nullptr && constants[0]->IsZero()) {
        *result = true;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetU32BitValue() == UINT32_MAX) {
        *result = true;
        return true;
      }
      break;
    case SpvOp::SpvOpSLessThanEqual:
      if (constants[0] != nullptr &&
          constants[0]->GetS32BitValue() == INT32_MIN) {
        *result = true;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetS32BitValue() == INT32_MAX) {
        *result = true;
        return true;
      }
      break;
    case SpvOp::SpvOpUGreaterThanEqual:
      if (constants[0] != nullptr &&
          constants[0]->GetU32BitValue() == UINT32_MAX) {
        *result = true;
        return true;
      }
      if (constants[1] != nullptr && constants[1]->GetU32BitValue() == 0) {
        *result = true;
        return true;
      }
      break;
    case SpvOp::SpvOpSGreaterThanEqual:
      if (constants[0] != nullptr &&
          constants[0]->GetS32BitValue() == INT32_MAX) {
        *result = true;
        return true;
      }
      if (constants[1] != nullptr &&
          constants[1]->GetS32BitValue() == INT32_MIN) {
        *result = true;
        return true;
      }
      break;
    default:
      break;
  }
  return false;
}

bool InstructionFolder::FoldBinaryBooleanOpToConstant(
    Instruction* inst, const std::function<uint32_t(uint32_t)>& id_map,
    uint32_t* result) const {
  SpvOp opcode = inst->opcode();
  analysis::ConstantManager* const_manger = context_->get_constant_mgr();

  uint32_t ids[2];
  const analysis::BoolConstant* constants[2];
  for (uint32_t i = 0; i < 2; i++) {
    const Operand* operand = &inst->GetInOperand(i);
    if (operand->type != SPV_OPERAND_TYPE_ID) {
      return false;
    }
    ids[i] = id_map(operand->words[0]);
    const analysis::Constant* constant =
        const_manger->FindDeclaredConstant(ids[i]);
    constants[i] = (constant != nullptr ? constant->AsBoolConstant() : nullptr);
  }

  switch (opcode) {
    // Logical
    case SpvOp::SpvOpLogicalOr:
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr) {
          if (constants[i]->value()) {
            *result = true;
            return true;
          }
        }
      }
      break;
    case SpvOp::SpvOpLogicalAnd:
      for (uint32_t i = 0; i < 2; i++) {
        if (constants[i] != nullptr) {
          if (!constants[i]->value()) {
            *result = false;
            return true;
          }
        }
      }
      break;

    default:
      break;
  }
  return false;
}

bool InstructionFolder::FoldIntegerOpToConstant(
    Instruction* inst, const std::function<uint32_t(uint32_t)>& id_map,
    uint32_t* result) const {
  assert(IsFoldableOpcode(inst->opcode()) &&
         "Unhandled instruction opcode in FoldScalars");
  switch (inst->NumInOperands()) {
    case 2:
      return FoldBinaryIntegerOpToConstant(inst, id_map, result) ||
             FoldBinaryBooleanOpToConstant(inst, id_map, result);
    default:
      return false;
  }
}

std::vector<uint32_t> InstructionFolder::FoldVectors(
    SpvOp opcode, uint32_t num_dims,
    const std::vector<const analysis::Constant*>& operands) const {
  assert(IsFoldableOpcode(opcode) &&
         "Unhandled instruction opcode in FoldVectors");
  std::vector<uint32_t> result;
  for (uint32_t d = 0; d < num_dims; d++) {
    std::vector<uint32_t> operand_values_for_one_dimension;
    for (const auto& operand : operands) {
      if (const analysis::VectorConstant* vector_operand =
              operand->AsVectorConstant()) {
        // Extract the raw value of the scalar component constants
        // in 32-bit words here. The reason of not using FoldScalars() here
        // is that we do not create temporary null constants as components
        // when the vector operand is a NullConstant because Constant creation
        // may need extra checks for the validity and that is not manageed in
        // here.
        if (const analysis::ScalarConstant* scalar_component =
                vector_operand->GetComponents().at(d)->AsScalarConstant()) {
          const auto& scalar_words = scalar_component->words();
          assert(
              scalar_words.size() == 1 &&
              "Vector components with longer than 32-bit width are not allowed "
              "in FoldVectors()");
          operand_values_for_one_dimension.push_back(scalar_words.front());
        } else if (operand->AsNullConstant()) {
          operand_values_for_one_dimension.push_back(0u);
        } else {
          assert(false &&
                 "VectorConst should only has ScalarConst or NullConst as "
                 "components");
        }
      } else if (operand->AsNullConstant()) {
        operand_values_for_one_dimension.push_back(0u);
      } else {
        assert(false &&
               "FoldVectors() only accepts VectorConst or NullConst type of "
               "constant");
      }
    }
    result.push_back(OperateWords(opcode, operand_values_for_one_dimension));
  }
  return result;
}

bool InstructionFolder::IsFoldableOpcode(SpvOp opcode) const {
  // NOTE: Extend to more opcodes as new cases are handled in the folder
  // functions.
  switch (opcode) {
    case SpvOp::SpvOpBitwiseAnd:
    case SpvOp::SpvOpBitwiseOr:
    case SpvOp::SpvOpBitwiseXor:
    case SpvOp::SpvOpIAdd:
    case SpvOp::SpvOpIEqual:
    case SpvOp::SpvOpIMul:
    case SpvOp::SpvOpINotEqual:
    case SpvOp::SpvOpISub:
    case SpvOp::SpvOpLogicalAnd:
    case SpvOp::SpvOpLogicalEqual:
    case SpvOp::SpvOpLogicalNot:
    case SpvOp::SpvOpLogicalNotEqual:
    case SpvOp::SpvOpLogicalOr:
    case SpvOp::SpvOpNot:
    case SpvOp::SpvOpSDiv:
    case SpvOp::SpvOpSelect:
    case SpvOp::SpvOpSGreaterThan:
    case SpvOp::SpvOpSGreaterThanEqual:
    case SpvOp::SpvOpShiftLeftLogical:
    case SpvOp::SpvOpShiftRightArithmetic:
    case SpvOp::SpvOpShiftRightLogical:
    case SpvOp::SpvOpSLessThan:
    case SpvOp::SpvOpSLessThanEqual:
    case SpvOp::SpvOpSMod:
    case SpvOp::SpvOpSNegate:
    case SpvOp::SpvOpSRem:
    case SpvOp::SpvOpUDiv:
    case SpvOp::SpvOpUGreaterThan:
    case SpvOp::SpvOpUGreaterThanEqual:
    case SpvOp::SpvOpULessThan:
    case SpvOp::SpvOpULessThanEqual:
    case SpvOp::SpvOpUMod:
      return true;
    default:
      return false;
  }
}

bool InstructionFolder::IsFoldableConstant(
    const analysis::Constant* cst) const {
  // Currently supported constants are 32-bit values or null constants.
  if (const analysis::ScalarConstant* scalar = cst->AsScalarConstant())
    return scalar->words().size() == 1;
  else
    return cst->AsNullConstant() != nullptr;
}

Instruction* InstructionFolder::FoldInstructionToConstant(
    Instruction* inst, std::function<uint32_t(uint32_t)> id_map) const {
  analysis::ConstantManager* const_mgr = context_->get_constant_mgr();

  if (!inst->IsFoldableByFoldScalar() &&
      !GetConstantFoldingRules().HasFoldingRule(inst->opcode())) {
    return nullptr;
  }
  // Collect the values of the constant parameters.
  std::vector<const analysis::Constant*> constants;
  bool missing_constants = false;
  inst->ForEachInId([&constants, &missing_constants, const_mgr,
                     &id_map](uint32_t* op_id) {
    uint32_t id = id_map(*op_id);
    const analysis::Constant* const_op = const_mgr->FindDeclaredConstant(id);
    if (!const_op) {
      constants.push_back(nullptr);
      missing_constants = true;
    } else {
      constants.push_back(const_op);
    }
  });

  if (GetConstantFoldingRules().HasFoldingRule(inst->opcode())) {
    const analysis::Constant* folded_const = nullptr;
    for (auto rule :
         GetConstantFoldingRules().GetRulesForOpcode(inst->opcode())) {
      folded_const = rule(context_, inst, constants);
      if (folded_const != nullptr) {
        Instruction* const_inst =
            const_mgr->GetDefiningInstruction(folded_const, inst->type_id());
        assert(const_inst->type_id() == inst->type_id());
        // May be a new instruction that needs to be analysed.
        context_->UpdateDefUse(const_inst);
        return const_inst;
      }
    }
  }

  uint32_t result_val = 0;
  bool successful = false;
  // If all parameters are constant, fold the instruction to a constant.
  if (!missing_constants && inst->IsFoldableByFoldScalar()) {
    result_val = FoldScalars(inst->opcode(), constants);
    successful = true;
  }

  if (!successful && inst->IsFoldableByFoldScalar()) {
    successful = FoldIntegerOpToConstant(inst, id_map, &result_val);
  }

  if (successful) {
    const analysis::Constant* result_const =
        const_mgr->GetConstant(const_mgr->GetType(inst), {result_val});
    Instruction* folded_inst =
        const_mgr->GetDefiningInstruction(result_const, inst->type_id());
    return folded_inst;
  }
  return nullptr;
}

bool InstructionFolder::IsFoldableType(Instruction* type_inst) const {
  // Support 32-bit integers.
  if (type_inst->opcode() == SpvOpTypeInt) {
    return type_inst->GetSingleWordInOperand(0) == 32;
  }
  // Support booleans.
  if (type_inst->opcode() == SpvOpTypeBool) {
    return true;
  }
  // Nothing else yet.
  return false;
}

bool InstructionFolder::FoldInstruction(Instruction* inst) const {
  bool modified = false;
  Instruction* folded_inst(inst);
  while (folded_inst->opcode() != SpvOpCopyObject &&
         FoldInstructionInternal(&*folded_inst)) {
    modified = true;
  }
  return modified;
}

}  // namespace opt
}  // namespace spvtools
