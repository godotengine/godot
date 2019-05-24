// Copyright (c) 2018 Google LLC
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

#include "source/opt/folding_rules.h"

#include <limits>
#include <memory>
#include <utility>

#include "source/latest_version_glsl_std_450_header.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;
const uint32_t kExtInstSetIdInIdx = 0;
const uint32_t kExtInstInstructionInIdx = 1;
const uint32_t kFMixXIdInIdx = 2;
const uint32_t kFMixYIdInIdx = 3;
const uint32_t kFMixAIdInIdx = 4;
const uint32_t kStoreObjectInIdx = 1;

// Returns the element width of |type|.
uint32_t ElementWidth(const analysis::Type* type) {
  if (const analysis::Vector* vec_type = type->AsVector()) {
    return ElementWidth(vec_type->element_type());
  } else if (const analysis::Float* float_type = type->AsFloat()) {
    return float_type->width();
  } else {
    assert(type->AsInteger());
    return type->AsInteger()->width();
  }
}

// Returns true if |type| is Float or a vector of Float.
bool HasFloatingPoint(const analysis::Type* type) {
  if (type->AsFloat()) {
    return true;
  } else if (const analysis::Vector* vec_type = type->AsVector()) {
    return vec_type->element_type()->AsFloat() != nullptr;
  }

  return false;
}

// Returns false if |val| is NaN, infinite or subnormal.
template <typename T>
bool IsValidResult(T val) {
  int classified = std::fpclassify(val);
  switch (classified) {
    case FP_NAN:
    case FP_INFINITE:
    case FP_SUBNORMAL:
      return false;
    default:
      return true;
  }
}

const analysis::Constant* ConstInput(
    const std::vector<const analysis::Constant*>& constants) {
  return constants[0] ? constants[0] : constants[1];
}

Instruction* NonConstInput(IRContext* context, const analysis::Constant* c,
                           Instruction* inst) {
  uint32_t in_op = c ? 1u : 0u;
  return context->get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(in_op));
}

// Returns the negation of |c|. |c| must be a 32 or 64 bit floating point
// constant.
uint32_t NegateFloatingPointConstant(analysis::ConstantManager* const_mgr,
                                     const analysis::Constant* c) {
  assert(c);
  assert(c->type()->AsFloat());
  uint32_t width = c->type()->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    utils::FloatProxy<double> result(c->GetDouble() * -1.0);
    words = result.GetWords();
  } else {
    utils::FloatProxy<float> result(c->GetFloat() * -1.0f);
    words = result.GetWords();
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

std::vector<uint32_t> ExtractInts(uint64_t val) {
  std::vector<uint32_t> words;
  words.push_back(static_cast<uint32_t>(val));
  words.push_back(static_cast<uint32_t>(val >> 32));
  return words;
}

// Negates the integer constant |c|. Returns the id of the defining instruction.
uint32_t NegateIntegerConstant(analysis::ConstantManager* const_mgr,
                               const analysis::Constant* c) {
  assert(c);
  assert(c->type()->AsInteger());
  uint32_t width = c->type()->AsInteger()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    uint64_t uval = static_cast<uint64_t>(0 - c->GetU64());
    words = ExtractInts(uval);
  } else {
    words.push_back(static_cast<uint32_t>(0 - c->GetU32()));
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

// Negates the vector constant |c|. Returns the id of the defining instruction.
uint32_t NegateVectorConstant(analysis::ConstantManager* const_mgr,
                              const analysis::Constant* c) {
  assert(const_mgr && c);
  assert(c->type()->AsVector());
  if (c->AsNullConstant()) {
    // 0.0 vs -0.0 shouldn't matter.
    return const_mgr->GetDefiningInstruction(c)->result_id();
  } else {
    const analysis::Type* component_type =
        c->AsVectorConstant()->component_type();
    std::vector<uint32_t> words;
    for (auto& comp : c->AsVectorConstant()->GetComponents()) {
      if (component_type->AsFloat()) {
        words.push_back(NegateFloatingPointConstant(const_mgr, comp));
      } else {
        assert(component_type->AsInteger());
        words.push_back(NegateIntegerConstant(const_mgr, comp));
      }
    }

    const analysis::Constant* negated_const =
        const_mgr->GetConstant(c->type(), std::move(words));
    return const_mgr->GetDefiningInstruction(negated_const)->result_id();
  }
}

// Negates |c|. Returns the id of the defining instruction.
uint32_t NegateConstant(analysis::ConstantManager* const_mgr,
                        const analysis::Constant* c) {
  if (c->type()->AsVector()) {
    return NegateVectorConstant(const_mgr, c);
  } else if (c->type()->AsFloat()) {
    return NegateFloatingPointConstant(const_mgr, c);
  } else {
    assert(c->type()->AsInteger());
    return NegateIntegerConstant(const_mgr, c);
  }
}

// Takes the reciprocal of |c|. |c|'s type must be Float or a vector of Float.
// Returns 0 if the reciprocal is NaN, infinite or subnormal.
uint32_t Reciprocal(analysis::ConstantManager* const_mgr,
                    const analysis::Constant* c) {
  assert(const_mgr && c);
  assert(c->type()->AsFloat());

  uint32_t width = c->type()->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    spvtools::utils::FloatProxy<double> result(1.0 / c->GetDouble());
    if (!IsValidResult(result.getAsFloat())) return 0;
    words = result.GetWords();
  } else {
    spvtools::utils::FloatProxy<float> result(1.0f / c->GetFloat());
    if (!IsValidResult(result.getAsFloat())) return 0;
    words = result.GetWords();
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

// Replaces fdiv where second operand is constant with fmul.
FoldingRule ReciprocalFDiv() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    if (constants[1] != nullptr) {
      uint32_t id = 0;
      if (const analysis::VectorConstant* vector_const =
              constants[1]->AsVectorConstant()) {
        std::vector<uint32_t> neg_ids;
        for (auto& comp : vector_const->GetComponents()) {
          id = Reciprocal(const_mgr, comp);
          if (id == 0) return false;
          neg_ids.push_back(id);
        }
        const analysis::Constant* negated_const =
            const_mgr->GetConstant(constants[1]->type(), std::move(neg_ids));
        id = const_mgr->GetDefiningInstruction(negated_const)->result_id();
      } else if (constants[1]->AsFloatConstant()) {
        id = Reciprocal(const_mgr, constants[1]);
        if (id == 0) return false;
      } else {
        // Don't fold a null constant.
        return false;
      }
      inst->SetOpcode(SpvOpFMul);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0u)}},
           {SPV_OPERAND_TYPE_ID, {id}}});
      return true;
    }

    return false;
  };
}

// Elides consecutive negate instructions.
FoldingRule MergeNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (op_inst->opcode() == inst->opcode()) {
      // Elide negates.
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op_inst->GetSingleWordInOperand(0u)}}});
      return true;
    }

    return false;
  };
}

// Merges negate into a mul or div operation if that operation contains a
// constant operand.
// Cases:
// -(x * 2) = x * -2
// -(2 * x) = x * -2
// -(x / 2) = x / -2
// -(2 / x) = -2 / x
FoldingRule MergeNegateMulDivArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    SpvOp opcode = op_inst->opcode();
    if (opcode == SpvOpFMul || opcode == SpvOpFDiv || opcode == SpvOpIMul ||
        opcode == SpvOpSDiv || opcode == SpvOpUDiv) {
      std::vector<const analysis::Constant*> op_constants =
          const_mgr->GetOperandConstants(op_inst);
      // Merge negate into mul or div if one operand is constant.
      if (op_constants[0] || op_constants[1]) {
        bool zero_is_variable = op_constants[0] == nullptr;
        const analysis::Constant* c = ConstInput(op_constants);
        uint32_t neg_id = NegateConstant(const_mgr, c);
        uint32_t non_const_id = zero_is_variable
                                    ? op_inst->GetSingleWordInOperand(0u)
                                    : op_inst->GetSingleWordInOperand(1u);
        // Change this instruction to a mul/div.
        inst->SetOpcode(op_inst->opcode());
        if (opcode == SpvOpFDiv || opcode == SpvOpUDiv || opcode == SpvOpSDiv) {
          uint32_t op0 = zero_is_variable ? non_const_id : neg_id;
          uint32_t op1 = zero_is_variable ? neg_id : non_const_id;
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {op0}}, {SPV_OPERAND_TYPE_ID, {op1}}});
        } else {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                               {SPV_OPERAND_TYPE_ID, {neg_id}}});
        }
        return true;
      }
    }

    return false;
  };
}

// Merges negate into a add or sub operation if that operation contains a
// constant operand.
// Cases:
// -(x + 2) = -2 - x
// -(2 + x) = -2 - x
// -(x - 2) = 2 - x
// -(2 - x) = x - 2
FoldingRule MergeNegateAddSubArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    if (op_inst->opcode() == SpvOpFAdd || op_inst->opcode() == SpvOpFSub ||
        op_inst->opcode() == SpvOpIAdd || op_inst->opcode() == SpvOpISub) {
      std::vector<const analysis::Constant*> op_constants =
          const_mgr->GetOperandConstants(op_inst);
      if (op_constants[0] || op_constants[1]) {
        bool zero_is_variable = op_constants[0] == nullptr;
        bool is_add = (op_inst->opcode() == SpvOpFAdd) ||
                      (op_inst->opcode() == SpvOpIAdd);
        bool swap_operands = !is_add || zero_is_variable;
        bool negate_const = is_add;
        const analysis::Constant* c = ConstInput(op_constants);
        uint32_t const_id = 0;
        if (negate_const) {
          const_id = NegateConstant(const_mgr, c);
        } else {
          const_id = zero_is_variable ? op_inst->GetSingleWordInOperand(1u)
                                      : op_inst->GetSingleWordInOperand(0u);
        }

        // Swap operands if necessary and make the instruction a subtraction.
        uint32_t op0 =
            zero_is_variable ? op_inst->GetSingleWordInOperand(0u) : const_id;
        uint32_t op1 =
            zero_is_variable ? const_id : op_inst->GetSingleWordInOperand(1u);
        if (swap_operands) std::swap(op0, op1);
        inst->SetOpcode(HasFloatingPoint(type) ? SpvOpFSub : SpvOpISub);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {op0}}, {SPV_OPERAND_TYPE_ID, {op1}}});
        return true;
      }
    }

    return false;
  };
}

// Returns true if |c| has a zero element.
bool HasZero(const analysis::Constant* c) {
  if (c->AsNullConstant()) {
    return true;
  }
  if (const analysis::VectorConstant* vec_const = c->AsVectorConstant()) {
    for (auto& comp : vec_const->GetComponents())
      if (HasZero(comp)) return true;
  } else {
    assert(c->AsScalarConstant());
    return c->AsScalarConstant()->IsZero();
  }

  return false;
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Float.
uint32_t PerformFloatingPointOperation(analysis::ConstantManager* const_mgr,
                                       SpvOp opcode,
                                       const analysis::Constant* input1,
                                       const analysis::Constant* input2) {
  const analysis::Type* type = input1->type();
  assert(type->AsFloat());
  uint32_t width = type->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
#define FOLD_OP(op)                                                          \
  if (width == 64) {                                                         \
    utils::FloatProxy<double> val =                                          \
        input1->GetDouble() op input2->GetDouble();                          \
    double dval = val.getAsFloat();                                          \
    if (!IsValidResult(dval)) return 0;                                      \
    words = val.GetWords();                                                  \
  } else {                                                                   \
    utils::FloatProxy<float> val = input1->GetFloat() op input2->GetFloat(); \
    float fval = val.getAsFloat();                                           \
    if (!IsValidResult(fval)) return 0;                                      \
    words = val.GetWords();                                                  \
  }
  switch (opcode) {
    case SpvOpFMul:
      FOLD_OP(*);
      break;
    case SpvOpFDiv:
      if (HasZero(input2)) return 0;
      FOLD_OP(/);
      break;
    case SpvOpFAdd:
      FOLD_OP(+);
      break;
    case SpvOpFSub:
      FOLD_OP(-);
      break;
    default:
      assert(false && "Unexpected operation");
      break;
  }
#undef FOLD_OP
  const analysis::Constant* merged_const = const_mgr->GetConstant(type, words);
  return const_mgr->GetDefiningInstruction(merged_const)->result_id();
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Integers.
uint32_t PerformIntegerOperation(analysis::ConstantManager* const_mgr,
                                 SpvOp opcode, const analysis::Constant* input1,
                                 const analysis::Constant* input2) {
  assert(input1->type()->AsInteger());
  const analysis::Integer* type = input1->type()->AsInteger();
  uint32_t width = type->AsInteger()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
#define FOLD_OP(op)                                        \
  if (width == 64) {                                       \
    if (type->IsSigned()) {                                \
      int64_t val = input1->GetS64() op input2->GetS64();  \
      words = ExtractInts(static_cast<uint64_t>(val));     \
    } else {                                               \
      uint64_t val = input1->GetU64() op input2->GetU64(); \
      words = ExtractInts(val);                            \
    }                                                      \
  } else {                                                 \
    if (type->IsSigned()) {                                \
      int32_t val = input1->GetS32() op input2->GetS32();  \
      words.push_back(static_cast<uint32_t>(val));         \
    } else {                                               \
      uint32_t val = input1->GetU32() op input2->GetU32(); \
      words.push_back(val);                                \
    }                                                      \
  }
  switch (opcode) {
    case SpvOpIMul:
      FOLD_OP(*);
      break;
    case SpvOpSDiv:
    case SpvOpUDiv:
      assert(false && "Should not merge integer division");
      break;
    case SpvOpIAdd:
      FOLD_OP(+);
      break;
    case SpvOpISub:
      FOLD_OP(-);
      break;
    default:
      assert(false && "Unexpected operation");
      break;
  }
#undef FOLD_OP
  const analysis::Constant* merged_const = const_mgr->GetConstant(type, words);
  return const_mgr->GetDefiningInstruction(merged_const)->result_id();
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Integers, Floats or Vectors of such.
uint32_t PerformOperation(analysis::ConstantManager* const_mgr, SpvOp opcode,
                          const analysis::Constant* input1,
                          const analysis::Constant* input2) {
  assert(input1 && input2);
  assert(input1->type() == input2->type());
  const analysis::Type* type = input1->type();
  std::vector<uint32_t> words;
  if (const analysis::Vector* vector_type = type->AsVector()) {
    const analysis::Type* ele_type = vector_type->element_type();
    for (uint32_t i = 0; i != vector_type->element_count(); ++i) {
      uint32_t id = 0;

      const analysis::Constant* input1_comp = nullptr;
      if (const analysis::VectorConstant* input1_vector =
              input1->AsVectorConstant()) {
        input1_comp = input1_vector->GetComponents()[i];
      } else {
        assert(input1->AsNullConstant());
        input1_comp = const_mgr->GetConstant(ele_type, {});
      }

      const analysis::Constant* input2_comp = nullptr;
      if (const analysis::VectorConstant* input2_vector =
              input2->AsVectorConstant()) {
        input2_comp = input2_vector->GetComponents()[i];
      } else {
        assert(input2->AsNullConstant());
        input2_comp = const_mgr->GetConstant(ele_type, {});
      }

      if (ele_type->AsFloat()) {
        id = PerformFloatingPointOperation(const_mgr, opcode, input1_comp,
                                           input2_comp);
      } else {
        assert(ele_type->AsInteger());
        id = PerformIntegerOperation(const_mgr, opcode, input1_comp,
                                     input2_comp);
      }
      if (id == 0) return 0;
      words.push_back(id);
    }
    const analysis::Constant* merged_const =
        const_mgr->GetConstant(type, words);
    return const_mgr->GetDefiningInstruction(merged_const)->result_id();
  } else if (type->AsFloat()) {
    return PerformFloatingPointOperation(const_mgr, opcode, input1, input2);
  } else {
    assert(type->AsInteger());
    return PerformIntegerOperation(const_mgr, opcode, input1, input2);
  }
}

// Merges consecutive multiplies where each contains one constant operand.
// Cases:
// 2 * (x * 2) = x * 4
// 2 * (2 * x) = x * 4
// (x * 2) * 2 = x * 4
// (2 * x) * 2 = x * 4
FoldingRule MergeMulMulArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul || inst->opcode() == SpvOpIMul);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    // Determine the constant input and the variable input in |inst|.
    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (HasFloatingPoint(type) && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == inst->opcode()) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      bool other_first_is_variable = other_constants[0] == nullptr;
      uint32_t merged_id = PerformOperation(const_mgr, inst->opcode(),
                                            const_input1, const_input2);
      if (merged_id == 0) return false;

      uint32_t non_const_id = other_first_is_variable
                                  ? other_inst->GetSingleWordInOperand(0u)
                                  : other_inst->GetSingleWordInOperand(1u);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                           {SPV_OPERAND_TYPE_ID, {merged_id}}});
      return true;
    }

    return false;
  };
}

// Merges divides into subsequent multiplies if each instruction contains one
// constant operand. Does not support integer operations.
// Cases:
// 2 * (x / 2) = x * 1
// 2 * (2 / x) = 4 / x
// (x / 2) * 2 = x * 1
// (2 / x) * 2 = 4 / x
// (y / x) * x = y
// x * (y / x) = y
FoldingRule MergeMulDivArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    for (uint32_t i = 0; i < 2; i++) {
      uint32_t op_id = inst->GetSingleWordInOperand(i);
      Instruction* op_inst = def_use_mgr->GetDef(op_id);
      if (op_inst->opcode() == SpvOpFDiv) {
        if (op_inst->GetSingleWordInOperand(1) ==
            inst->GetSingleWordInOperand(1 - i)) {
          inst->SetOpcode(SpvOpCopyObject);
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {op_inst->GetSingleWordInOperand(0)}}});
          return true;
        }
      }
    }

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (!other_inst->IsFloatingPointFoldingAllowed()) return false;

    if (other_inst->opcode() == SpvOpFDiv) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2 || HasZero(const_input2)) return false;

      bool other_first_is_variable = other_constants[0] == nullptr;
      // If the variable value is the second operand of the divide, multiply
      // the constants together. Otherwise divide the constants.
      uint32_t merged_id = PerformOperation(
          const_mgr,
          other_first_is_variable ? other_inst->opcode() : inst->opcode(),
          const_input1, const_input2);
      if (merged_id == 0) return false;

      uint32_t non_const_id = other_first_is_variable
                                  ? other_inst->GetSingleWordInOperand(0u)
                                  : other_inst->GetSingleWordInOperand(1u);

      // If the variable value is on the second operand of the div, then this
      // operation is a div. Otherwise it should be a multiply.
      inst->SetOpcode(other_first_is_variable ? inst->opcode()
                                              : other_inst->opcode());
      if (other_first_is_variable) {
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                             {SPV_OPERAND_TYPE_ID, {merged_id}}});
      } else {
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {merged_id}},
                             {SPV_OPERAND_TYPE_ID, {non_const_id}}});
      }
      return true;
    }

    return false;
  };
}

// Merges multiply of constant and negation.
// Cases:
// (-x) * 2 = x * -2
// 2 * (-x) = x * -2
FoldingRule MergeMulNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul || inst->opcode() == SpvOpIMul);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFNegate ||
        other_inst->opcode() == SpvOpSNegate) {
      uint32_t neg_id = NegateConstant(const_mgr, const_input1);

      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {other_inst->GetSingleWordInOperand(0u)}},
           {SPV_OPERAND_TYPE_ID, {neg_id}}});
      return true;
    }

    return false;
  };
}

// Merges consecutive divides if each instruction contains one constant operand.
// Does not support integer division.
// Cases:
// 2 / (x / 2) = 4 / x
// 4 / (2 / x) = 2 * x
// (4 / x) / 2 = 2 / x
// (x / 2) / 2 = x / 4
FoldingRule MergeDivDivArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1 || HasZero(const_input1)) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (!other_inst->IsFloatingPointFoldingAllowed()) return false;

    bool first_is_variable = constants[0] == nullptr;
    if (other_inst->opcode() == inst->opcode()) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2 || HasZero(const_input2)) return false;

      bool other_first_is_variable = other_constants[0] == nullptr;

      SpvOp merge_op = inst->opcode();
      if (other_first_is_variable) {
        // Constants magnify.
        merge_op = SpvOpFMul;
      }

      // This is an x / (*) case. Swap the inputs. Doesn't harm multiply
      // because it is commutative.
      if (first_is_variable) std::swap(const_input1, const_input2);
      uint32_t merged_id =
          PerformOperation(const_mgr, merge_op, const_input1, const_input2);
      if (merged_id == 0) return false;

      uint32_t non_const_id = other_first_is_variable
                                  ? other_inst->GetSingleWordInOperand(0u)
                                  : other_inst->GetSingleWordInOperand(1u);

      SpvOp op = inst->opcode();
      if (!first_is_variable && !other_first_is_variable) {
        // Effectively div of 1/x, so change to multiply.
        op = SpvOpFMul;
      }

      uint32_t op1 = merged_id;
      uint32_t op2 = non_const_id;
      if (first_is_variable && other_first_is_variable) std::swap(op1, op2);
      inst->SetOpcode(op);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }

    return false;
  };
}

// Fold multiplies succeeded by divides where each instruction contains a
// constant operand. Does not support integer divide.
// Cases:
// 4 / (x * 2) = 2 / x
// 4 / (2 * x) = 2 / x
// (x * 4) / 2 = x * 2
// (4 * x) / 2 = x * 2
// (x * y) / x = y
// (y * x) / x = y
FoldingRule MergeDivMulArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv);
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();

    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    uint32_t op_id = inst->GetSingleWordInOperand(0);
    Instruction* op_inst = def_use_mgr->GetDef(op_id);

    if (op_inst->opcode() == SpvOpFMul) {
      for (uint32_t i = 0; i < 2; i++) {
        if (op_inst->GetSingleWordInOperand(i) ==
            inst->GetSingleWordInOperand(1)) {
          inst->SetOpcode(SpvOpCopyObject);
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                                {op_inst->GetSingleWordInOperand(1 - i)}}});
          return true;
        }
      }
    }

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1 || HasZero(const_input1)) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (!other_inst->IsFloatingPointFoldingAllowed()) return false;

    bool first_is_variable = constants[0] == nullptr;
    if (other_inst->opcode() == SpvOpFMul) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      bool other_first_is_variable = other_constants[0] == nullptr;

      // This is an x / (*) case. Swap the inputs.
      if (first_is_variable) std::swap(const_input1, const_input2);
      uint32_t merged_id = PerformOperation(const_mgr, inst->opcode(),
                                            const_input1, const_input2);
      if (merged_id == 0) return false;

      uint32_t non_const_id = other_first_is_variable
                                  ? other_inst->GetSingleWordInOperand(0u)
                                  : other_inst->GetSingleWordInOperand(1u);

      uint32_t op1 = merged_id;
      uint32_t op2 = non_const_id;
      if (first_is_variable) std::swap(op1, op2);

      // Convert to multiply
      if (first_is_variable) inst->SetOpcode(other_inst->opcode());
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }

    return false;
  };
}

// Fold divides of a constant and a negation.
// Cases:
// (-x) / 2 = x / -2
// 2 / (-x) = 2 / -x
FoldingRule MergeDivNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv || inst->opcode() == SpvOpSDiv ||
           inst->opcode() == SpvOpUDiv);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    bool first_is_variable = constants[0] == nullptr;
    if (other_inst->opcode() == SpvOpFNegate ||
        other_inst->opcode() == SpvOpSNegate) {
      uint32_t neg_id = NegateConstant(const_mgr, const_input1);

      if (first_is_variable) {
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {other_inst->GetSingleWordInOperand(0u)}},
             {SPV_OPERAND_TYPE_ID, {neg_id}}});
      } else {
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {neg_id}},
             {SPV_OPERAND_TYPE_ID, {other_inst->GetSingleWordInOperand(0u)}}});
      }
      return true;
    }

    return false;
  };
}

// Folds addition of a constant and a negation.
// Cases:
// (-x) + 2 = 2 - x
// 2 + (-x) = 2 - x
FoldingRule MergeAddNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFAdd || inst->opcode() == SpvOpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpSNegate ||
        other_inst->opcode() == SpvOpFNegate) {
      inst->SetOpcode(HasFloatingPoint(type) ? SpvOpFSub : SpvOpISub);
      uint32_t const_id = constants[0] ? inst->GetSingleWordInOperand(0u)
                                       : inst->GetSingleWordInOperand(1u);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {const_id}},
           {SPV_OPERAND_TYPE_ID, {other_inst->GetSingleWordInOperand(0u)}}});
      return true;
    }
    return false;
  };
}

// Folds subtraction of a constant and a negation.
// Cases:
// (-x) - 2 = -2 - x
// 2 - (-x) = x + 2
FoldingRule MergeSubNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFSub || inst->opcode() == SpvOpISub);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpSNegate ||
        other_inst->opcode() == SpvOpFNegate) {
      uint32_t op1 = 0;
      uint32_t op2 = 0;
      SpvOp opcode = inst->opcode();
      if (constants[0] != nullptr) {
        op1 = other_inst->GetSingleWordInOperand(0u);
        op2 = inst->GetSingleWordInOperand(0u);
        opcode = HasFloatingPoint(type) ? SpvOpFAdd : SpvOpIAdd;
      } else {
        op1 = NegateConstant(const_mgr, const_input1);
        op2 = other_inst->GetSingleWordInOperand(0u);
      }

      inst->SetOpcode(opcode);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }
    return false;
  };
}

// Folds addition of an addition where each operation has a constant operand.
// Cases:
// (x + 2) + 2 = x + 4
// (2 + x) + 2 = x + 4
// 2 + (x + 2) = x + 4
// 2 + (2 + x) = x + 4
FoldingRule MergeAddAddArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFAdd || inst->opcode() == SpvOpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFAdd ||
        other_inst->opcode() == SpvOpIAdd) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      Instruction* non_const_input =
          NonConstInput(context, other_constants[0], other_inst);
      uint32_t merged_id = PerformOperation(const_mgr, inst->opcode(),
                                            const_input1, const_input2);
      if (merged_id == 0) return false;

      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {non_const_input->result_id()}},
           {SPV_OPERAND_TYPE_ID, {merged_id}}});
      return true;
    }
    return false;
  };
}

// Folds addition of a subtraction where each operation has a constant operand.
// Cases:
// (x - 2) + 2 = x + 0
// (2 - x) + 2 = 4 - x
// 2 + (x - 2) = x + 0
// 2 + (2 - x) = 4 - x
FoldingRule MergeAddSubArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFAdd || inst->opcode() == SpvOpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFSub ||
        other_inst->opcode() == SpvOpISub) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      bool first_is_variable = other_constants[0] == nullptr;
      SpvOp op = inst->opcode();
      uint32_t op1 = 0;
      uint32_t op2 = 0;
      if (first_is_variable) {
        // Subtract constants. Non-constant operand is first.
        op1 = other_inst->GetSingleWordInOperand(0u);
        op2 = PerformOperation(const_mgr, other_inst->opcode(), const_input1,
                               const_input2);
      } else {
        // Add constants. Constant operand is first. Change the opcode.
        op1 = PerformOperation(const_mgr, inst->opcode(), const_input1,
                               const_input2);
        op2 = other_inst->GetSingleWordInOperand(1u);
        op = other_inst->opcode();
      }
      if (op1 == 0 || op2 == 0) return false;

      inst->SetOpcode(op);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }
    return false;
  };
}

// Folds subtraction of an addition where each operand has a constant operand.
// Cases:
// (x + 2) - 2 = x + 0
// (2 + x) - 2 = x + 0
// 2 - (x + 2) = 0 - x
// 2 - (2 + x) = 0 - x
FoldingRule MergeSubAddArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFSub || inst->opcode() == SpvOpISub);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFAdd ||
        other_inst->opcode() == SpvOpIAdd) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      Instruction* non_const_input =
          NonConstInput(context, other_constants[0], other_inst);

      // If the first operand of the sub is not a constant, swap the constants
      // so the subtraction has the correct operands.
      if (constants[0] == nullptr) std::swap(const_input1, const_input2);
      // Subtract the constants.
      uint32_t merged_id = PerformOperation(const_mgr, inst->opcode(),
                                            const_input1, const_input2);
      SpvOp op = inst->opcode();
      uint32_t op1 = 0;
      uint32_t op2 = 0;
      if (constants[0] == nullptr) {
        // Non-constant operand is first. Change the opcode.
        op1 = non_const_input->result_id();
        op2 = merged_id;
        op = other_inst->opcode();
      } else {
        // Constant operand is first.
        op1 = merged_id;
        op2 = non_const_input->result_id();
      }
      if (op1 == 0 || op2 == 0) return false;

      inst->SetOpcode(op);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }
    return false;
  };
}

// Folds subtraction of a subtraction where each operand has a constant operand.
// Cases:
// (x - 2) - 2 = x - 4
// (2 - x) - 2 = 0 - x
// 2 - (x - 2) = 4 - x
// 2 - (2 - x) = x + 0
FoldingRule MergeSubSubArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFSub || inst->opcode() == SpvOpISub);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFSub ||
        other_inst->opcode() == SpvOpISub) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      Instruction* non_const_input =
          NonConstInput(context, other_constants[0], other_inst);

      // Merge the constants.
      uint32_t merged_id = 0;
      SpvOp merge_op = inst->opcode();
      if (other_constants[0] == nullptr) {
        merge_op = uses_float ? SpvOpFAdd : SpvOpIAdd;
      } else if (constants[0] == nullptr) {
        std::swap(const_input1, const_input2);
      }
      merged_id =
          PerformOperation(const_mgr, merge_op, const_input1, const_input2);
      if (merged_id == 0) return false;

      SpvOp op = inst->opcode();
      if (constants[0] != nullptr && other_constants[0] != nullptr) {
        // Change the operation.
        op = uses_float ? SpvOpFAdd : SpvOpIAdd;
      }

      uint32_t op1 = 0;
      uint32_t op2 = 0;
      if ((constants[0] == nullptr) ^ (other_constants[0] == nullptr)) {
        op1 = merged_id;
        op2 = non_const_input->result_id();
      } else {
        op1 = non_const_input->result_id();
        op2 = merged_id;
      }

      inst->SetOpcode(op);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op1}}, {SPV_OPERAND_TYPE_ID, {op2}}});
      return true;
    }
    return false;
  };
}

FoldingRule IntMultipleBy1() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpIMul && "Wrong opcode.  Should be OpIMul.");
    for (uint32_t i = 0; i < 2; i++) {
      if (constants[i] == nullptr) {
        continue;
      }
      const analysis::IntConstant* int_constant = constants[i]->AsIntConstant();
      if (int_constant) {
        uint32_t width = ElementWidth(int_constant->type());
        if (width != 32 && width != 64) return false;
        bool is_one = (width == 32) ? int_constant->GetU32BitValue() == 1u
                                    : int_constant->GetU64BitValue() == 1ull;
        if (is_one) {
          inst->SetOpcode(SpvOpCopyObject);
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1 - i)}}});
          return true;
        }
      }
    }
    return false;
  };
}

FoldingRule CompositeConstructFeedingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    // If the input to an OpCompositeExtract is an OpCompositeConstruct,
    // then we can simply use the appropriate element in the construction.
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeConstruct) {
      return false;
    }

    std::vector<Operand> operands;
    analysis::Type* composite_type = type_mgr->GetType(cinst->type_id());
    if (composite_type->AsVector() == nullptr) {
      // Get the element being extracted from the OpCompositeConstruct
      // Since it is not a vector, it is simple to extract the single element.
      uint32_t element_index = inst->GetSingleWordInOperand(1);
      uint32_t element_id = cinst->GetSingleWordInOperand(element_index);
      operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});

      // Add the remaining indices for extraction.
      for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {inst->GetSingleWordInOperand(i)}});
      }

    } else {
      // With vectors we have to handle the case where it is concatenating
      // vectors.
      assert(inst->NumInOperands() == 2 &&
             "Expecting a vector of scalar values.");

      uint32_t element_index = inst->GetSingleWordInOperand(1);
      for (uint32_t construct_index = 0;
           construct_index < cinst->NumInOperands(); ++construct_index) {
        uint32_t element_id = cinst->GetSingleWordInOperand(construct_index);
        Instruction* element_def = def_use_mgr->GetDef(element_id);
        analysis::Vector* element_type =
            type_mgr->GetType(element_def->type_id())->AsVector();
        if (element_type) {
          uint32_t vector_size = element_type->element_count();
          if (vector_size < element_index) {
            // The element we want comes after this vector.
            element_index -= vector_size;
          } else {
            // We want an element of this vector.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            operands.push_back(
                {SPV_OPERAND_TYPE_LITERAL_INTEGER, {element_index}});
            break;
          }
        } else {
          if (element_index == 0) {
            // This is a scalar, and we this is the element we are extracting.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            break;
          } else {
            // Skip over this scalar value.
            --element_index;
          }
        }
      }
    }

    // If there were no extra indices, then we have the final object.  No need
    // to extract even more.
    if (operands.size() == 1) {
      inst->SetOpcode(SpvOpCopyObject);
    }

    inst->SetInOperands(std::move(operands));
    return true;
  };
}

FoldingRule CompositeExtractFeedingConstruct() {
  // If the OpCompositeConstruct is simply putting back together elements that
  // where extracted from the same souce, we can simlpy reuse the source.
  //
  // This is a common code pattern because of the way that scalar replacement
  // works.
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeConstruct &&
           "Wrong opcode.  Should be OpCompositeConstruct.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    uint32_t original_id = 0;

    // Check each element to make sure they are:
    // - extractions
    // - extracting the same position they are inserting
    // - all extract from the same id.
    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      uint32_t element_id = inst->GetSingleWordInOperand(i);
      Instruction* element_inst = def_use_mgr->GetDef(element_id);

      if (element_inst->opcode() != SpvOpCompositeExtract) {
        return false;
      }

      if (element_inst->NumInOperands() != 2) {
        return false;
      }

      if (element_inst->GetSingleWordInOperand(1) != i) {
        return false;
      }

      if (i == 0) {
        original_id =
            element_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
      } else if (original_id != element_inst->GetSingleWordInOperand(
                                    kExtractCompositeIdInIdx)) {
        return false;
      }
    }

    // The last check it to see that the object being extracted from is the
    // correct type.
    Instruction* original_inst = def_use_mgr->GetDef(original_id);
    if (original_inst->type_id() != inst->type_id()) {
      return false;
    }

    // Simplify by using the original object.
    inst->SetOpcode(SpvOpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {original_id}}});
    return true;
  };
}

FoldingRule InsertFeedingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeInsert) {
      return false;
    }

    // Find the first position where the list of insert and extract indicies
    // differ, if at all.
    uint32_t i;
    for (i = 1; i < inst->NumInOperands(); ++i) {
      if (i + 1 >= cinst->NumInOperands()) {
        break;
      }

      if (inst->GetSingleWordInOperand(i) !=
          cinst->GetSingleWordInOperand(i + 1)) {
        break;
      }
    }

    // We are extracting the element that was inserted.
    if (i == inst->NumInOperands() && i + 1 == cinst->NumInOperands()) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID,
            {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}}});
      return true;
    }

    // Extracting the value that was inserted along with values for the base
    // composite.  Cannot do anything.
    if (i == inst->NumInOperands()) {
      return false;
    }

    // Extracting an element of the value that was inserted.  Extract from
    // that value directly.
    if (i + 1 == cinst->NumInOperands()) {
      std::vector<Operand> operands;
      operands.push_back(
          {SPV_OPERAND_TYPE_ID,
           {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}});
      for (; i < inst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {inst->GetSingleWordInOperand(i)}});
      }
      inst->SetInOperands(std::move(operands));
      return true;
    }

    // Extracting a value that is disjoint from the element being inserted.
    // Rewrite the extract to use the composite input to the insert.
    std::vector<Operand> operands;
    operands.push_back(
        {SPV_OPERAND_TYPE_ID,
         {cinst->GetSingleWordInOperand(kInsertCompositeIdInIdx)}});
    for (i = 1; i < inst->NumInOperands(); ++i) {
      operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {inst->GetSingleWordInOperand(i)}});
    }
    inst->SetInOperands(std::move(operands));
    return true;
  };
}

// When a VectorShuffle is feeding an Extract, we can extract from one of the
// operands of the VectorShuffle.  We just need to adjust the index in the
// extract instruction.
FoldingRule VectorShuffleFeedingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpVectorShuffle) {
      return false;
    }

    // Find the size of the first vector operand of the VectorShuffle
    Instruction* first_input =
        def_use_mgr->GetDef(cinst->GetSingleWordInOperand(0));
    analysis::Type* first_input_type =
        type_mgr->GetType(first_input->type_id());
    assert(first_input_type->AsVector() &&
           "Input to vector shuffle should be vectors.");
    uint32_t first_input_size = first_input_type->AsVector()->element_count();

    // Get index of the element the vector shuffle is placing in the position
    // being extracted.
    uint32_t new_index =
        cinst->GetSingleWordInOperand(2 + inst->GetSingleWordInOperand(1));

    // Extracting an undefined value so fold this extract into an undef.
    const uint32_t undef_literal_value = 0xffffffff;
    if (new_index == undef_literal_value) {
      inst->SetOpcode(SpvOpUndef);
      inst->SetInOperands({});
      return true;
    }

    // Get the id of the of the vector the elemtent comes from, and update the
    // index if needed.
    uint32_t new_vector = 0;
    if (new_index < first_input_size) {
      new_vector = cinst->GetSingleWordInOperand(0);
    } else {
      new_vector = cinst->GetSingleWordInOperand(1);
      new_index -= first_input_size;
    }

    // Update the extract instruction.
    inst->SetInOperand(kExtractCompositeIdInIdx, {new_vector});
    inst->SetInOperand(1, {new_index});
    return true;
  };
}

// When an FMix with is feeding an Extract that extracts an element whose
// corresponding |a| in the FMix is 0 or 1, we can extract from one of the
// operands of the FMix.
FoldingRule FMixFeedingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();

    uint32_t composite_id =
        inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* composite_inst = def_use_mgr->GetDef(composite_id);

    if (composite_inst->opcode() != SpvOpExtInst) {
      return false;
    }

    uint32_t inst_set_id =
        context->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

    if (composite_inst->GetSingleWordInOperand(kExtInstSetIdInIdx) !=
            inst_set_id ||
        composite_inst->GetSingleWordInOperand(kExtInstInstructionInIdx) !=
            GLSLstd450FMix) {
      return false;
    }

    // Get the |a| for the FMix instruction.
    uint32_t a_id = composite_inst->GetSingleWordInOperand(kFMixAIdInIdx);
    std::unique_ptr<Instruction> a(inst->Clone(context));
    a->SetInOperand(kExtractCompositeIdInIdx, {a_id});
    context->get_instruction_folder().FoldInstruction(a.get());

    if (a->opcode() != SpvOpCopyObject) {
      return false;
    }

    const analysis::Constant* a_const =
        const_mgr->FindDeclaredConstant(a->GetSingleWordInOperand(0));

    if (!a_const) {
      return false;
    }

    bool use_x = false;

    assert(a_const->type()->AsFloat());
    double element_value = a_const->GetValueAsDouble();
    if (element_value == 0.0) {
      use_x = true;
    } else if (element_value == 1.0) {
      use_x = false;
    } else {
      return false;
    }

    // Get the id of the of the vector the element comes from.
    uint32_t new_vector = 0;
    if (use_x) {
      new_vector = composite_inst->GetSingleWordInOperand(kFMixXIdInIdx);
    } else {
      new_vector = composite_inst->GetSingleWordInOperand(kFMixYIdInIdx);
    }

    // Update the extract instruction.
    inst->SetInOperand(kExtractCompositeIdInIdx, {new_vector});
    return true;
  };
}

FoldingRule RedundantPhi() {
  // An OpPhi instruction where all values are the same or the result of the phi
  // itself, can be replaced by the value itself.
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpPhi && "Wrong opcode.  Should be OpPhi.");

    uint32_t incoming_value = 0;

    for (uint32_t i = 0; i < inst->NumInOperands(); i += 2) {
      uint32_t op_id = inst->GetSingleWordInOperand(i);
      if (op_id == inst->result_id()) {
        continue;
      }

      if (incoming_value == 0) {
        incoming_value = op_id;
      } else if (op_id != incoming_value) {
        // Found two possible value.  Can't simplify.
        return false;
      }
    }

    if (incoming_value == 0) {
      // Code looks invalid.  Don't do anything.
      return false;
    }

    // We have a single incoming value.  Simplify using that value.
    inst->SetOpcode(SpvOpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {incoming_value}}});
    return true;
  };
}

FoldingRule RedundantSelect() {
  // An OpSelect instruction where both values are the same or the condition is
  // constant can be replaced by one of the values
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpSelect &&
           "Wrong opcode.  Should be OpSelect.");
    assert(inst->NumInOperands() == 3);
    assert(constants.size() == 3);

    uint32_t true_id = inst->GetSingleWordInOperand(1);
    uint32_t false_id = inst->GetSingleWordInOperand(2);

    if (true_id == false_id) {
      // Both results are the same, condition doesn't matter
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {true_id}}});
      return true;
    } else if (constants[0]) {
      const analysis::Type* type = constants[0]->type();
      if (type->AsBool()) {
        // Scalar constant value, select the corresponding value.
        inst->SetOpcode(SpvOpCopyObject);
        if (constants[0]->AsNullConstant() ||
            !constants[0]->AsBoolConstant()->value()) {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {false_id}}});
        } else {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {true_id}}});
        }
        return true;
      } else {
        assert(type->AsVector());
        if (constants[0]->AsNullConstant()) {
          // All values come from false id.
          inst->SetOpcode(SpvOpCopyObject);
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {false_id}}});
          return true;
        } else {
          // Convert to a vector shuffle.
          std::vector<Operand> ops;
          ops.push_back({SPV_OPERAND_TYPE_ID, {true_id}});
          ops.push_back({SPV_OPERAND_TYPE_ID, {false_id}});
          const analysis::VectorConstant* vector_const =
              constants[0]->AsVectorConstant();
          uint32_t size =
              static_cast<uint32_t>(vector_const->GetComponents().size());
          for (uint32_t i = 0; i != size; ++i) {
            const analysis::Constant* component =
                vector_const->GetComponents()[i];
            if (component->AsNullConstant() ||
                !component->AsBoolConstant()->value()) {
              // Selecting from the false vector which is the second input
              // vector to the shuffle. Offset the index by |size|.
              ops.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {i + size}});
            } else {
              // Selecting from true vector which is the first input vector to
              // the shuffle.
              ops.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}});
            }
          }

          inst->SetOpcode(SpvOpVectorShuffle);
          inst->SetInOperands(std::move(ops));
          return true;
        }
      }
    }

    return false;
  };
}

enum class FloatConstantKind { Unknown, Zero, One };

FloatConstantKind getFloatConstantKind(const analysis::Constant* constant) {
  if (constant == nullptr) {
    return FloatConstantKind::Unknown;
  }

  assert(HasFloatingPoint(constant->type()) && "Unexpected constant type");

  if (constant->AsNullConstant()) {
    return FloatConstantKind::Zero;
  } else if (const analysis::VectorConstant* vc =
                 constant->AsVectorConstant()) {
    const std::vector<const analysis::Constant*>& components =
        vc->GetComponents();
    assert(!components.empty());

    FloatConstantKind kind = getFloatConstantKind(components[0]);

    for (size_t i = 1; i < components.size(); ++i) {
      if (getFloatConstantKind(components[i]) != kind) {
        return FloatConstantKind::Unknown;
      }
    }

    return kind;
  } else if (const analysis::FloatConstant* fc = constant->AsFloatConstant()) {
    if (fc->IsZero()) return FloatConstantKind::Zero;

    uint32_t width = fc->type()->AsFloat()->width();
    if (width != 32 && width != 64) return FloatConstantKind::Unknown;

    double value = (width == 64) ? fc->GetDoubleValue() : fc->GetFloatValue();

    if (value == 0.0) {
      return FloatConstantKind::Zero;
    } else if (value == 1.0) {
      return FloatConstantKind::One;
    } else {
      return FloatConstantKind::Unknown;
    }
  } else {
    return FloatConstantKind::Unknown;
  }
}

FoldingRule RedundantFAdd() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFAdd && "Wrong opcode.  Should be OpFAdd.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::Zero ? 1 : 0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFSub() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFSub && "Wrong opcode.  Should be OpFSub.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpFNegate);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFMul() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul && "Wrong opcode.  Should be OpFMul.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::Zero ? 0 : 1)}}});
      return true;
    }

    if (kind0 == FloatConstantKind::One || kind1 == FloatConstantKind::One) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::One ? 1 : 0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFDiv() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv && "Wrong opcode.  Should be OpFDiv.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::One) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFMix() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpExtInst &&
           "Wrong opcode.  Should be OpExtInst.");

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    uint32_t instSetId =
        context->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

    if (inst->GetSingleWordInOperand(kExtInstSetIdInIdx) == instSetId &&
        inst->GetSingleWordInOperand(kExtInstInstructionInIdx) ==
            GLSLstd450FMix) {
      assert(constants.size() == 5);

      FloatConstantKind kind4 = getFloatConstantKind(constants[4]);

      if (kind4 == FloatConstantKind::Zero || kind4 == FloatConstantKind::One) {
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID,
              {inst->GetSingleWordInOperand(kind4 == FloatConstantKind::Zero
                                                ? kFMixXIdInIdx
                                                : kFMixYIdInIdx)}}});
        return true;
      }
    }

    return false;
  };
}

// This rule handles addition of zero for integers.
FoldingRule RedundantIAdd() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpIAdd && "Wrong opcode. Should be OpIAdd.");

    uint32_t operand = std::numeric_limits<uint32_t>::max();
    const analysis::Type* operand_type = nullptr;
    if (constants[0] && constants[0]->IsZero()) {
      operand = inst->GetSingleWordInOperand(1);
      operand_type = constants[0]->type();
    } else if (constants[1] && constants[1]->IsZero()) {
      operand = inst->GetSingleWordInOperand(0);
      operand_type = constants[1]->type();
    }

    if (operand != std::numeric_limits<uint32_t>::max()) {
      const analysis::Type* inst_type =
          context->get_type_mgr()->GetType(inst->type_id());
      if (inst_type->IsSame(operand_type)) {
        inst->SetOpcode(SpvOpCopyObject);
      } else {
        inst->SetOpcode(SpvOpBitcast);
      }
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {operand}}});
      return true;
    }
    return false;
  };
}

// This rule look for a dot with a constant vector containing a single 1 and
// the rest 0s.  This is the same as doing an extract.
FoldingRule DotProductDoingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpDot && "Wrong opcode.  Should be OpDot.");

    analysis::ConstantManager* const_mgr = context->get_constant_mgr();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    for (int i = 0; i < 2; ++i) {
      if (!constants[i]) {
        continue;
      }

      const analysis::Vector* vector_type = constants[i]->type()->AsVector();
      assert(vector_type && "Inputs to OpDot must be vectors.");
      const analysis::Float* element_type =
          vector_type->element_type()->AsFloat();
      assert(element_type && "Inputs to OpDot must be vectors of floats.");
      uint32_t element_width = element_type->width();
      if (element_width != 32 && element_width != 64) {
        return false;
      }

      std::vector<const analysis::Constant*> components;
      components = constants[i]->GetVectorComponents(const_mgr);

      const uint32_t kNotFound = std::numeric_limits<uint32_t>::max();

      uint32_t component_with_one = kNotFound;
      bool all_others_zero = true;
      for (uint32_t j = 0; j < components.size(); ++j) {
        const analysis::Constant* element = components[j];
        double value =
            (element_width == 32 ? element->GetFloat() : element->GetDouble());
        if (value == 0.0) {
          continue;
        } else if (value == 1.0) {
          if (component_with_one == kNotFound) {
            component_with_one = j;
          } else {
            component_with_one = kNotFound;
            break;
          }
        } else {
          all_others_zero = false;
          break;
        }
      }

      if (!all_others_zero || component_with_one == kNotFound) {
        continue;
      }

      std::vector<Operand> operands;
      operands.push_back(
          {SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1u - i)}});
      operands.push_back(
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {component_with_one}});

      inst->SetOpcode(SpvOpCompositeExtract);
      inst->SetInOperands(std::move(operands));
      return true;
    }
    return false;
  };
}

// If we are storing an undef, then we can remove the store.
//
// TODO: We can do something similar for OpImageWrite, but checking for volatile
// is complicated.  Waiting to see if it is needed.
FoldingRule StoringUndef() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpStore && "Wrong opcode.  Should be OpStore.");

    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

    // If this is a volatile store, the store cannot be removed.
    if (inst->NumInOperands() == 3) {
      if (inst->GetSingleWordInOperand(2) & SpvMemoryAccessVolatileMask) {
        return false;
      }
    }

    uint32_t object_id = inst->GetSingleWordInOperand(kStoreObjectInIdx);
    Instruction* object_inst = def_use_mgr->GetDef(object_id);
    if (object_inst->opcode() == SpvOpUndef) {
      inst->ToNop();
      return true;
    }
    return false;
  };
}

FoldingRule VectorShuffleFeedingShuffle() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpVectorShuffle &&
           "Wrong opcode.  Should be OpVectorShuffle.");

    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();

    Instruction* feeding_shuffle_inst =
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    analysis::Vector* op0_type =
        type_mgr->GetType(feeding_shuffle_inst->type_id())->AsVector();
    uint32_t op0_length = op0_type->element_count();

    bool feeder_is_op0 = true;
    if (feeding_shuffle_inst->opcode() != SpvOpVectorShuffle) {
      feeding_shuffle_inst =
          def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));
      feeder_is_op0 = false;
    }

    if (feeding_shuffle_inst->opcode() != SpvOpVectorShuffle) {
      return false;
    }

    Instruction* feeder2 =
        def_use_mgr->GetDef(feeding_shuffle_inst->GetSingleWordInOperand(0));
    analysis::Vector* feeder_op0_type =
        type_mgr->GetType(feeder2->type_id())->AsVector();
    uint32_t feeder_op0_length = feeder_op0_type->element_count();

    uint32_t new_feeder_id = 0;
    std::vector<Operand> new_operands;
    new_operands.resize(
        2, {SPV_OPERAND_TYPE_ID, {0}});  // Place holders for vector operands.
    const uint32_t undef_literal = 0xffffffff;
    for (uint32_t op = 2; op < inst->NumInOperands(); ++op) {
      uint32_t component_index = inst->GetSingleWordInOperand(op);

      // Do not interpret the undefined value literal as coming from operand 1.
      if (component_index != undef_literal &&
          feeder_is_op0 == (component_index < op0_length)) {
        // This component comes from the feeding_shuffle_inst.  Update
        // |component_index| to be the index into the operand of the feeder.

        // Adjust component_index to get the index into the operands of the
        // feeding_shuffle_inst.
        if (component_index >= op0_length) {
          component_index -= op0_length;
        }
        component_index =
            feeding_shuffle_inst->GetSingleWordInOperand(component_index + 2);

        // Check if we are using a component from the first or second operand of
        // the feeding instruction.
        if (component_index < feeder_op0_length) {
          if (new_feeder_id == 0) {
            // First time through, save the id of the operand the element comes
            // from.
            new_feeder_id = feeding_shuffle_inst->GetSingleWordInOperand(0);
          } else if (new_feeder_id !=
                     feeding_shuffle_inst->GetSingleWordInOperand(0)) {
            // We need both elements of the feeding_shuffle_inst, so we cannot
            // fold.
            return false;
          }
        } else {
          if (new_feeder_id == 0) {
            // First time through, save the id of the operand the element comes
            // from.
            new_feeder_id = feeding_shuffle_inst->GetSingleWordInOperand(1);
          } else if (new_feeder_id !=
                     feeding_shuffle_inst->GetSingleWordInOperand(1)) {
            // We need both elements of the feeding_shuffle_inst, so we cannot
            // fold.
            return false;
          }
          component_index -= feeder_op0_length;
        }

        if (!feeder_is_op0) {
          component_index += op0_length;
        }
      }
      new_operands.push_back(
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {component_index}});
    }

    if (new_feeder_id == 0) {
      analysis::ConstantManager* const_mgr = context->get_constant_mgr();
      const analysis::Type* type =
          type_mgr->GetType(feeding_shuffle_inst->type_id());
      const analysis::Constant* null_const = const_mgr->GetConstant(type, {});
      new_feeder_id =
          const_mgr->GetDefiningInstruction(null_const, 0)->result_id();
    }

    if (feeder_is_op0) {
      // If the size of the first vector operand changed then the indices
      // referring to the second operand need to be adjusted.
      Instruction* new_feeder_inst = def_use_mgr->GetDef(new_feeder_id);
      analysis::Type* new_feeder_type =
          type_mgr->GetType(new_feeder_inst->type_id());
      uint32_t new_op0_size = new_feeder_type->AsVector()->element_count();
      int32_t adjustment = op0_length - new_op0_size;

      if (adjustment != 0) {
        for (uint32_t i = 2; i < new_operands.size(); i++) {
          if (inst->GetSingleWordInOperand(i) >= op0_length) {
            new_operands[i].words[0] -= adjustment;
          }
        }
      }

      new_operands[0].words[0] = new_feeder_id;
      new_operands[1] = inst->GetInOperand(1);
    } else {
      new_operands[1].words[0] = new_feeder_id;
      new_operands[0] = inst->GetInOperand(0);
    }

    inst->SetInOperands(std::move(new_operands));
    return true;
  };
}

// Removes duplicate ids from the interface list of an OpEntryPoint
// instruction.
FoldingRule RemoveRedundantOperands() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpEntryPoint &&
           "Wrong opcode.  Should be OpEntryPoint.");
    bool has_redundant_operand = false;
    std::unordered_set<uint32_t> seen_operands;
    std::vector<Operand> new_operands;

    new_operands.emplace_back(inst->GetOperand(0));
    new_operands.emplace_back(inst->GetOperand(1));
    new_operands.emplace_back(inst->GetOperand(2));
    for (uint32_t i = 3; i < inst->NumOperands(); ++i) {
      if (seen_operands.insert(inst->GetSingleWordOperand(i)).second) {
        new_operands.emplace_back(inst->GetOperand(i));
      } else {
        has_redundant_operand = true;
      }
    }

    if (!has_redundant_operand) {
      return false;
    }

    inst->SetInOperands(std::move(new_operands));
    return true;
  };
}

}  // namespace

FoldingRules::FoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.
  rules_[SpvOpCompositeConstruct].push_back(CompositeExtractFeedingConstruct());

  rules_[SpvOpCompositeExtract].push_back(InsertFeedingExtract());
  rules_[SpvOpCompositeExtract].push_back(CompositeConstructFeedingExtract());
  rules_[SpvOpCompositeExtract].push_back(VectorShuffleFeedingExtract());
  rules_[SpvOpCompositeExtract].push_back(FMixFeedingExtract());

  rules_[SpvOpDot].push_back(DotProductDoingExtract());

  rules_[SpvOpEntryPoint].push_back(RemoveRedundantOperands());

  rules_[SpvOpExtInst].push_back(RedundantFMix());

  rules_[SpvOpFAdd].push_back(RedundantFAdd());
  rules_[SpvOpFAdd].push_back(MergeAddNegateArithmetic());
  rules_[SpvOpFAdd].push_back(MergeAddAddArithmetic());
  rules_[SpvOpFAdd].push_back(MergeAddSubArithmetic());

  rules_[SpvOpFDiv].push_back(RedundantFDiv());
  rules_[SpvOpFDiv].push_back(ReciprocalFDiv());
  rules_[SpvOpFDiv].push_back(MergeDivDivArithmetic());
  rules_[SpvOpFDiv].push_back(MergeDivMulArithmetic());
  rules_[SpvOpFDiv].push_back(MergeDivNegateArithmetic());

  rules_[SpvOpFMul].push_back(RedundantFMul());
  rules_[SpvOpFMul].push_back(MergeMulMulArithmetic());
  rules_[SpvOpFMul].push_back(MergeMulDivArithmetic());
  rules_[SpvOpFMul].push_back(MergeMulNegateArithmetic());

  rules_[SpvOpFNegate].push_back(MergeNegateArithmetic());
  rules_[SpvOpFNegate].push_back(MergeNegateAddSubArithmetic());
  rules_[SpvOpFNegate].push_back(MergeNegateMulDivArithmetic());

  rules_[SpvOpFSub].push_back(RedundantFSub());
  rules_[SpvOpFSub].push_back(MergeSubNegateArithmetic());
  rules_[SpvOpFSub].push_back(MergeSubAddArithmetic());
  rules_[SpvOpFSub].push_back(MergeSubSubArithmetic());

  rules_[SpvOpIAdd].push_back(RedundantIAdd());
  rules_[SpvOpIAdd].push_back(MergeAddNegateArithmetic());
  rules_[SpvOpIAdd].push_back(MergeAddAddArithmetic());
  rules_[SpvOpIAdd].push_back(MergeAddSubArithmetic());

  rules_[SpvOpIMul].push_back(IntMultipleBy1());
  rules_[SpvOpIMul].push_back(MergeMulMulArithmetic());
  rules_[SpvOpIMul].push_back(MergeMulNegateArithmetic());

  rules_[SpvOpISub].push_back(MergeSubNegateArithmetic());
  rules_[SpvOpISub].push_back(MergeSubAddArithmetic());
  rules_[SpvOpISub].push_back(MergeSubSubArithmetic());

  rules_[SpvOpPhi].push_back(RedundantPhi());

  rules_[SpvOpSDiv].push_back(MergeDivNegateArithmetic());

  rules_[SpvOpSNegate].push_back(MergeNegateArithmetic());
  rules_[SpvOpSNegate].push_back(MergeNegateMulDivArithmetic());
  rules_[SpvOpSNegate].push_back(MergeNegateAddSubArithmetic());

  rules_[SpvOpSelect].push_back(RedundantSelect());

  rules_[SpvOpStore].push_back(StoringUndef());

  rules_[SpvOpUDiv].push_back(MergeDivNegateArithmetic());

  rules_[SpvOpVectorShuffle].push_back(VectorShuffleFeedingShuffle());
}
}  // namespace opt
}  // namespace spvtools
