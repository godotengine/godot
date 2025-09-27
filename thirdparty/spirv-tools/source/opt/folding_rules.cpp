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

#include <climits>
#include <limits>
#include <memory>
#include <utility>

#include "ir_builder.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

constexpr uint32_t kExtractCompositeIdInIdx = 0;
constexpr uint32_t kInsertObjectIdInIdx = 0;
constexpr uint32_t kInsertCompositeIdInIdx = 1;
constexpr uint32_t kExtInstSetIdInIdx = 0;
constexpr uint32_t kExtInstInstructionInIdx = 1;
constexpr uint32_t kFMixXIdInIdx = 2;
constexpr uint32_t kFMixYIdInIdx = 3;
constexpr uint32_t kFMixAIdInIdx = 4;
constexpr uint32_t kStoreObjectInIdx = 1;

// Some image instructions may contain an "image operands" argument.
// Returns the operand index for the "image operands".
// Returns -1 if the instruction does not have image operands.
int32_t ImageOperandsMaskInOperandIndex(Instruction* inst) {
  const auto opcode = inst->opcode();
  switch (opcode) {
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageFetch:
    case spv::Op::OpImageRead:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseFetch:
    case spv::Op::OpImageSparseRead:
      return inst->NumOperands() > 4 ? 2 : -1;
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
      return inst->NumOperands() > 5 ? 3 : -1;
    case spv::Op::OpImageWrite:
      return inst->NumOperands() > 3 ? 3 : -1;
    default:
      return -1;
  }
}

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

std::vector<uint32_t> ExtractInts(uint64_t val) {
  std::vector<uint32_t> words;
  words.push_back(static_cast<uint32_t>(val));
  words.push_back(static_cast<uint32_t>(val >> 32));
  return words;
}

std::vector<uint32_t> GetWordsFromScalarIntConstant(
    const analysis::IntConstant* c) {
  assert(c != nullptr);
  uint32_t width = c->type()->AsInteger()->width();
  assert(width == 8 || width == 16 || width == 32 || width == 64);
  if (width == 64) {
    uint64_t uval = static_cast<uint64_t>(c->GetU64());
    return ExtractInts(uval);
  }
  // Section 2.2.1 of the SPIR-V spec guarantees that all integer types
  // smaller than 32-bits are automatically zero or sign extended to 32-bits.
  return {c->GetU32BitValue()};
}

std::vector<uint32_t> GetWordsFromScalarFloatConstant(
    const analysis::FloatConstant* c) {
  assert(c != nullptr);
  uint32_t width = c->type()->AsFloat()->width();
  assert(width == 16 || width == 32 || width == 64);
  if (width == 64) {
    utils::FloatProxy<double> result(c->GetDouble());
    return result.GetWords();
  }
  // Section 2.2.1 of the SPIR-V spec guarantees that all floating-point types
  // smaller than 32-bits are automatically zero extended to 32-bits.
  return {c->GetU32BitValue()};
}

std::vector<uint32_t> GetWordsFromNumericScalarOrVectorConstant(
    analysis::ConstantManager* const_mgr, const analysis::Constant* c) {
  if (const auto* float_constant = c->AsFloatConstant()) {
    return GetWordsFromScalarFloatConstant(float_constant);
  } else if (const auto* int_constant = c->AsIntConstant()) {
    return GetWordsFromScalarIntConstant(int_constant);
  } else if (const auto* vec_constant = c->AsVectorConstant()) {
    std::vector<uint32_t> words;
    for (const auto* comp : vec_constant->GetComponents()) {
      auto comp_in_words =
          GetWordsFromNumericScalarOrVectorConstant(const_mgr, comp);
      words.insert(words.end(), comp_in_words.begin(), comp_in_words.end());
    }
    return words;
  }
  return {};
}

const analysis::Constant* ConvertWordsToNumericScalarOrVectorConstant(
    analysis::ConstantManager* const_mgr, const std::vector<uint32_t>& words,
    const analysis::Type* type) {
  if (type->AsInteger() || type->AsFloat())
    return const_mgr->GetConstant(type, words);
  if (const auto* vec_type = type->AsVector())
    return const_mgr->GetNumericVectorConstantWithWords(vec_type, words);
  return nullptr;
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

  if (c->IsZero()) {
    return 0;
  }

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
    assert(inst->opcode() == spv::Op::OpFDiv);
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
      inst->SetOpcode(spv::Op::OpFMul);
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
    assert(inst->opcode() == spv::Op::OpFNegate ||
           inst->opcode() == spv::Op::OpSNegate);
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
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpFNegate ||
           inst->opcode() == spv::Op::OpSNegate);
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

    spv::Op opcode = op_inst->opcode();
    if (opcode == spv::Op::OpFMul || opcode == spv::Op::OpFDiv ||
        opcode == spv::Op::OpIMul || opcode == spv::Op::OpSDiv ||
        opcode == spv::Op::OpUDiv) {
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
        if (opcode == spv::Op::OpFDiv || opcode == spv::Op::OpUDiv ||
            opcode == spv::Op::OpSDiv) {
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
    assert(inst->opcode() == spv::Op::OpFNegate ||
           inst->opcode() == spv::Op::OpSNegate);
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

    if (op_inst->opcode() == spv::Op::OpFAdd ||
        op_inst->opcode() == spv::Op::OpFSub ||
        op_inst->opcode() == spv::Op::OpIAdd ||
        op_inst->opcode() == spv::Op::OpISub) {
      std::vector<const analysis::Constant*> op_constants =
          const_mgr->GetOperandConstants(op_inst);
      if (op_constants[0] || op_constants[1]) {
        bool zero_is_variable = op_constants[0] == nullptr;
        bool is_add = (op_inst->opcode() == spv::Op::OpFAdd) ||
                      (op_inst->opcode() == spv::Op::OpIAdd);
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
        inst->SetOpcode(HasFloatingPoint(type) ? spv::Op::OpFSub
                                               : spv::Op::OpISub);
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
                                       spv::Op opcode,
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
  }                                                                          \
  static_assert(true, "require extra semicolon")
  switch (opcode) {
    case spv::Op::OpFMul:
      FOLD_OP(*);
      break;
    case spv::Op::OpFDiv:
      if (HasZero(input2)) return 0;
      FOLD_OP(/);
      break;
    case spv::Op::OpFAdd:
      FOLD_OP(+);
      break;
    case spv::Op::OpFSub:
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
                                 spv::Op opcode,
                                 const analysis::Constant* input1,
                                 const analysis::Constant* input2) {
  assert(input1->type()->AsInteger());
  const analysis::Integer* type = input1->type()->AsInteger();
  uint32_t width = type->AsInteger()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  // Regardless of the sign of the constant, folding is performed on an unsigned
  // interpretation of the constant data. This avoids signed integer overflow
  // while folding, and works because sign is irrelevant for the IAdd, ISub and
  // IMul instructions.
#define FOLD_OP(op)                                      \
  if (width == 64) {                                     \
    uint64_t val = input1->GetU64() op input2->GetU64(); \
    words = ExtractInts(val);                            \
  } else {                                               \
    uint32_t val = input1->GetU32() op input2->GetU32(); \
    words.push_back(val);                                \
  }                                                      \
  static_assert(true, "require extra semicolon")
  switch (opcode) {
    case spv::Op::OpIMul:
      FOLD_OP(*);
      break;
    case spv::Op::OpSDiv:
    case spv::Op::OpUDiv:
      assert(false && "Should not merge integer division");
      break;
    case spv::Op::OpIAdd:
      FOLD_OP(+);
      break;
    case spv::Op::OpISub:
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
uint32_t PerformOperation(analysis::ConstantManager* const_mgr, spv::Op opcode,
                          const analysis::Constant* input1,
                          const analysis::Constant* input2) {
  assert(input1 && input2);
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
    assert(inst->opcode() == spv::Op::OpFMul ||
           inst->opcode() == spv::Op::OpIMul);
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
    assert(inst->opcode() == spv::Op::OpFMul);
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
      if (op_inst->opcode() == spv::Op::OpFDiv) {
        if (op_inst->GetSingleWordInOperand(1) ==
            inst->GetSingleWordInOperand(1 - i)) {
          inst->SetOpcode(spv::Op::OpCopyObject);
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

    if (other_inst->opcode() == spv::Op::OpFDiv) {
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
    assert(inst->opcode() == spv::Op::OpFMul ||
           inst->opcode() == spv::Op::OpIMul);
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

    if (other_inst->opcode() == spv::Op::OpFNegate ||
        other_inst->opcode() == spv::Op::OpSNegate) {
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
    assert(inst->opcode() == spv::Op::OpFDiv);
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

      spv::Op merge_op = inst->opcode();
      if (other_first_is_variable) {
        // Constants magnify.
        merge_op = spv::Op::OpFMul;
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

      spv::Op op = inst->opcode();
      if (!first_is_variable && !other_first_is_variable) {
        // Effectively div of 1/x, so change to multiply.
        op = spv::Op::OpFMul;
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
    assert(inst->opcode() == spv::Op::OpFDiv);
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();

    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    uint32_t op_id = inst->GetSingleWordInOperand(0);
    Instruction* op_inst = def_use_mgr->GetDef(op_id);

    if (op_inst->opcode() == spv::Op::OpFMul) {
      for (uint32_t i = 0; i < 2; i++) {
        if (op_inst->GetSingleWordInOperand(i) ==
            inst->GetSingleWordInOperand(1)) {
          inst->SetOpcode(spv::Op::OpCopyObject);
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
    if (other_inst->opcode() == spv::Op::OpFMul) {
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
// 2 / (-x) = -2 / x
FoldingRule MergeDivNegateArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == spv::Op::OpFDiv);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (!other_inst->IsFloatingPointFoldingAllowed()) return false;

    bool first_is_variable = constants[0] == nullptr;
    if (other_inst->opcode() == spv::Op::OpFNegate) {
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
    assert(inst->opcode() == spv::Op::OpFAdd ||
           inst->opcode() == spv::Op::OpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    const analysis::Constant* const_input1 = ConstInput(constants);
    if (!const_input1) return false;
    Instruction* other_inst = NonConstInput(context, constants[0], inst);
    if (uses_float && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == spv::Op::OpSNegate ||
        other_inst->opcode() == spv::Op::OpFNegate) {
      inst->SetOpcode(HasFloatingPoint(type) ? spv::Op::OpFSub
                                             : spv::Op::OpISub);
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
    assert(inst->opcode() == spv::Op::OpFSub ||
           inst->opcode() == spv::Op::OpISub);
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

    if (other_inst->opcode() == spv::Op::OpSNegate ||
        other_inst->opcode() == spv::Op::OpFNegate) {
      uint32_t op1 = 0;
      uint32_t op2 = 0;
      spv::Op opcode = inst->opcode();
      if (constants[0] != nullptr) {
        op1 = other_inst->GetSingleWordInOperand(0u);
        op2 = inst->GetSingleWordInOperand(0u);
        opcode = HasFloatingPoint(type) ? spv::Op::OpFAdd : spv::Op::OpIAdd;
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
    assert(inst->opcode() == spv::Op::OpFAdd ||
           inst->opcode() == spv::Op::OpIAdd);
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

    if (other_inst->opcode() == spv::Op::OpFAdd ||
        other_inst->opcode() == spv::Op::OpIAdd) {
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
    assert(inst->opcode() == spv::Op::OpFAdd ||
           inst->opcode() == spv::Op::OpIAdd);
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

    if (other_inst->opcode() == spv::Op::OpFSub ||
        other_inst->opcode() == spv::Op::OpISub) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      bool first_is_variable = other_constants[0] == nullptr;
      spv::Op op = inst->opcode();
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
    assert(inst->opcode() == spv::Op::OpFSub ||
           inst->opcode() == spv::Op::OpISub);
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

    if (other_inst->opcode() == spv::Op::OpFAdd ||
        other_inst->opcode() == spv::Op::OpIAdd) {
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
      spv::Op op = inst->opcode();
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
    assert(inst->opcode() == spv::Op::OpFSub ||
           inst->opcode() == spv::Op::OpISub);
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

    if (other_inst->opcode() == spv::Op::OpFSub ||
        other_inst->opcode() == spv::Op::OpISub) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      const analysis::Constant* const_input2 = ConstInput(other_constants);
      if (!const_input2) return false;

      Instruction* non_const_input =
          NonConstInput(context, other_constants[0], other_inst);

      // Merge the constants.
      uint32_t merged_id = 0;
      spv::Op merge_op = inst->opcode();
      if (other_constants[0] == nullptr) {
        merge_op = uses_float ? spv::Op::OpFAdd : spv::Op::OpIAdd;
      } else if (constants[0] == nullptr) {
        std::swap(const_input1, const_input2);
      }
      merged_id =
          PerformOperation(const_mgr, merge_op, const_input1, const_input2);
      if (merged_id == 0) return false;

      spv::Op op = inst->opcode();
      if (constants[0] != nullptr && other_constants[0] != nullptr) {
        // Change the operation.
        op = uses_float ? spv::Op::OpFAdd : spv::Op::OpIAdd;
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

// Helper function for MergeGenericAddSubArithmetic. If |addend| and
// subtrahend of |sub| is the same, merge to copy of minuend of |sub|.
bool MergeGenericAddendSub(uint32_t addend, uint32_t sub, Instruction* inst) {
  IRContext* context = inst->context();
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  Instruction* sub_inst = def_use_mgr->GetDef(sub);
  if (sub_inst->opcode() != spv::Op::OpFSub &&
      sub_inst->opcode() != spv::Op::OpISub)
    return false;
  if (sub_inst->opcode() == spv::Op::OpFSub &&
      !sub_inst->IsFloatingPointFoldingAllowed())
    return false;
  if (addend != sub_inst->GetSingleWordInOperand(1)) return false;
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetInOperands(
      {{SPV_OPERAND_TYPE_ID, {sub_inst->GetSingleWordInOperand(0)}}});
  context->UpdateDefUse(inst);
  return true;
}

// Folds addition of a subtraction where the subtrahend is equal to the
// other addend. Return a copy of the minuend. Accepts generic (const and
// non-const) operands.
// Cases:
// (a - b) + b = a
// b + (a - b) = a
FoldingRule MergeGenericAddSubArithmetic() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == spv::Op::OpFAdd ||
           inst->opcode() == spv::Op::OpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    uint32_t add_op0 = inst->GetSingleWordInOperand(0);
    uint32_t add_op1 = inst->GetSingleWordInOperand(1);
    if (MergeGenericAddendSub(add_op0, add_op1, inst)) return true;
    return MergeGenericAddendSub(add_op1, add_op0, inst);
  };
}

// Helper function for FactorAddMuls. If |factor0_0| is the same as |factor1_0|,
// generate |factor0_0| * (|factor0_1| + |factor1_1|).
bool FactorAddMulsOpnds(uint32_t factor0_0, uint32_t factor0_1,
                        uint32_t factor1_0, uint32_t factor1_1,
                        Instruction* inst) {
  IRContext* context = inst->context();
  if (factor0_0 != factor1_0) return false;
  InstructionBuilder ir_builder(
      context, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* new_add_inst = ir_builder.AddBinaryOp(
      inst->type_id(), inst->opcode(), factor0_1, factor1_1);
  inst->SetOpcode(inst->opcode() == spv::Op::OpFAdd ? spv::Op::OpFMul
                                                    : spv::Op::OpIMul);
  inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {factor0_0}},
                       {SPV_OPERAND_TYPE_ID, {new_add_inst->result_id()}}});
  context->UpdateDefUse(inst);
  return true;
}

// Perform the following factoring identity, handling all operand order
// combinations: (a * b) + (a * c) = a * (b + c)
FoldingRule FactorAddMuls() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == spv::Op::OpFAdd ||
           inst->opcode() == spv::Op::OpIAdd);
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    bool uses_float = HasFloatingPoint(type);
    if (uses_float && !inst->IsFloatingPointFoldingAllowed()) return false;

    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    uint32_t add_op0 = inst->GetSingleWordInOperand(0);
    Instruction* add_op0_inst = def_use_mgr->GetDef(add_op0);
    if (add_op0_inst->opcode() != spv::Op::OpFMul &&
        add_op0_inst->opcode() != spv::Op::OpIMul)
      return false;
    uint32_t add_op1 = inst->GetSingleWordInOperand(1);
    Instruction* add_op1_inst = def_use_mgr->GetDef(add_op1);
    if (add_op1_inst->opcode() != spv::Op::OpFMul &&
        add_op1_inst->opcode() != spv::Op::OpIMul)
      return false;

    // Only perform this optimization if both of the muls only have one use.
    // Otherwise this is a deoptimization in size and performance.
    if (def_use_mgr->NumUses(add_op0_inst) > 1) return false;
    if (def_use_mgr->NumUses(add_op1_inst) > 1) return false;

    if (add_op0_inst->opcode() == spv::Op::OpFMul &&
        (!add_op0_inst->IsFloatingPointFoldingAllowed() ||
         !add_op1_inst->IsFloatingPointFoldingAllowed()))
      return false;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        // Check if operand i in add_op0_inst matches operand j in add_op1_inst.
        if (FactorAddMulsOpnds(add_op0_inst->GetSingleWordInOperand(i),
                               add_op0_inst->GetSingleWordInOperand(1 - i),
                               add_op1_inst->GetSingleWordInOperand(j),
                               add_op1_inst->GetSingleWordInOperand(1 - j),
                               inst))
          return true;
      }
    }
    return false;
  };
}

// Replaces |inst| inplace with an FMA instruction |(x*y)+a|.
void ReplaceWithFma(Instruction* inst, uint32_t x, uint32_t y, uint32_t a) {
  uint32_t ext =
      inst->context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

  if (ext == 0) {
    inst->context()->AddExtInstImport("GLSL.std.450");
    ext = inst->context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
    assert(ext != 0 &&
           "Could not add the GLSL.std.450 extended instruction set");
  }

  std::vector<Operand> operands;
  operands.push_back({SPV_OPERAND_TYPE_ID, {ext}});
  operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {GLSLstd450Fma}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {x}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {y}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {a}});

  inst->SetOpcode(spv::Op::OpExtInst);
  inst->SetInOperands(std::move(operands));
}

// Folds a multiple and add into an Fma.
//
// Cases:
// (x * y) + a = Fma x y a
// a + (x * y) = Fma x y a
bool MergeMulAddArithmetic(IRContext* context, Instruction* inst,
                           const std::vector<const analysis::Constant*>&) {
  assert(inst->opcode() == spv::Op::OpFAdd);

  if (!inst->IsFloatingPointFoldingAllowed()) {
    return false;
  }

  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  for (int i = 0; i < 2; i++) {
    uint32_t op_id = inst->GetSingleWordInOperand(i);
    Instruction* op_inst = def_use_mgr->GetDef(op_id);

    if (op_inst->opcode() != spv::Op::OpFMul) {
      continue;
    }

    if (!op_inst->IsFloatingPointFoldingAllowed()) {
      continue;
    }

    uint32_t x = op_inst->GetSingleWordInOperand(0);
    uint32_t y = op_inst->GetSingleWordInOperand(1);
    uint32_t a = inst->GetSingleWordInOperand((i + 1) % 2);
    ReplaceWithFma(inst, x, y, a);
    return true;
  }
  return false;
}

// Replaces |sub| inplace with an FMA instruction |(x*y)+a| where |a| first gets
// negated if |negate_addition| is true, otherwise |x| gets negated.
void ReplaceWithFmaAndNegate(Instruction* sub, uint32_t x, uint32_t y,
                             uint32_t a, bool negate_addition) {
  uint32_t ext =
      sub->context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

  if (ext == 0) {
    sub->context()->AddExtInstImport("GLSL.std.450");
    ext = sub->context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
    assert(ext != 0 &&
           "Could not add the GLSL.std.450 extended instruction set");
  }

  InstructionBuilder ir_builder(
      sub->context(), sub,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  Instruction* neg = ir_builder.AddUnaryOp(sub->type_id(), spv::Op::OpFNegate,
                                           negate_addition ? a : x);
  uint32_t neg_op = neg->result_id();  // -a : -x

  std::vector<Operand> operands;
  operands.push_back({SPV_OPERAND_TYPE_ID, {ext}});
  operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {GLSLstd450Fma}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {negate_addition ? x : neg_op}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {y}});
  operands.push_back({SPV_OPERAND_TYPE_ID, {negate_addition ? neg_op : a}});

  sub->SetOpcode(spv::Op::OpExtInst);
  sub->SetInOperands(std::move(operands));
}

// Folds a multiply and subtract into an Fma and negation.
//
// Cases:
// (x * y) - a = Fma x y -a
// a - (x * y) = Fma -x y a
bool MergeMulSubArithmetic(IRContext* context, Instruction* sub,
                           const std::vector<const analysis::Constant*>&) {
  assert(sub->opcode() == spv::Op::OpFSub);

  if (!sub->IsFloatingPointFoldingAllowed()) {
    return false;
  }

  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  for (int i = 0; i < 2; i++) {
    uint32_t op_id = sub->GetSingleWordInOperand(i);
    Instruction* mul = def_use_mgr->GetDef(op_id);

    if (mul->opcode() != spv::Op::OpFMul) {
      continue;
    }

    if (!mul->IsFloatingPointFoldingAllowed()) {
      continue;
    }

    uint32_t x = mul->GetSingleWordInOperand(0);
    uint32_t y = mul->GetSingleWordInOperand(1);
    uint32_t a = sub->GetSingleWordInOperand((i + 1) % 2);
    ReplaceWithFmaAndNegate(sub, x, y, a, i == 0);
    return true;
  }
  return false;
}

FoldingRule IntMultipleBy1() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == spv::Op::OpIMul &&
           "Wrong opcode.  Should be OpIMul.");
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
          inst->SetOpcode(spv::Op::OpCopyObject);
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1 - i)}}});
          return true;
        }
      }
    }
    return false;
  };
}

// Returns the number of elements that the |index|th in operand in |inst|
// contributes to the result of |inst|.  |inst| must be an
// OpCompositeConstructInstruction.
uint32_t GetNumOfElementsContributedByOperand(IRContext* context,
                                              const Instruction* inst,
                                              uint32_t index) {
  assert(inst->opcode() == spv::Op::OpCompositeConstruct);
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context->get_type_mgr();

  analysis::Vector* result_type =
      type_mgr->GetType(inst->type_id())->AsVector();
  if (result_type == nullptr) {
    // If the result of the OpCompositeConstruct is not a vector then every
    // operands corresponds to a single element in the result.
    return 1;
  }

  // If the result type is a vector then the operands are either scalars or
  // vectors. If it is a scalar, then it corresponds to a single element.  If it
  // is a vector, then each element in the vector will be an element in the
  // result.
  uint32_t id = inst->GetSingleWordInOperand(index);
  Instruction* def = def_use_mgr->GetDef(id);
  analysis::Vector* type = type_mgr->GetType(def->type_id())->AsVector();
  if (type == nullptr) {
    return 1;
  }
  return type->element_count();
}

// Returns the in-operands for an OpCompositeExtract instruction that are needed
// to extract the |result_index|th element in the result of |inst| without using
// the result of |inst|. Returns the empty vector if |result_index| is
// out-of-bounds. |inst| must be an |OpCompositeConstruct| instruction.
std::vector<Operand> GetExtractOperandsForElementOfCompositeConstruct(
    IRContext* context, const Instruction* inst, uint32_t result_index) {
  assert(inst->opcode() == spv::Op::OpCompositeConstruct);
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context->get_type_mgr();

  analysis::Type* result_type = type_mgr->GetType(inst->type_id());
  if (result_type->AsVector() == nullptr) {
    if (result_index < inst->NumInOperands()) {
      uint32_t id = inst->GetSingleWordInOperand(result_index);
      return {Operand(SPV_OPERAND_TYPE_ID, {id})};
    }
    return {};
  }

  // If the result type is a vector, then vector operands are concatenated.
  uint32_t total_element_count = 0;
  for (uint32_t idx = 0; idx < inst->NumInOperands(); ++idx) {
    uint32_t element_count =
        GetNumOfElementsContributedByOperand(context, inst, idx);
    total_element_count += element_count;
    if (result_index < total_element_count) {
      std::vector<Operand> operands;
      uint32_t id = inst->GetSingleWordInOperand(idx);
      Instruction* operand_def = def_use_mgr->GetDef(id);
      analysis::Type* operand_type = type_mgr->GetType(operand_def->type_id());

      operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
      if (operand_type->AsVector()) {
        uint32_t start_index_of_id = total_element_count - element_count;
        uint32_t index_into_id = result_index - start_index_of_id;
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {index_into_id}});
      }
      return operands;
    }
  }
  return {};
}

bool CompositeConstructFeedingExtract(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>&) {
  // If the input to an OpCompositeExtract is an OpCompositeConstruct,
  // then we can simply use the appropriate element in the construction.
  assert(inst->opcode() == spv::Op::OpCompositeExtract &&
         "Wrong opcode.  Should be OpCompositeExtract.");
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

  // If there are no index operands, then this rule cannot do anything.
  if (inst->NumInOperands() <= 1) {
    return false;
  }

  uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
  Instruction* cinst = def_use_mgr->GetDef(cid);

  if (cinst->opcode() != spv::Op::OpCompositeConstruct) {
    return false;
  }

  uint32_t index_into_result = inst->GetSingleWordInOperand(1);
  std::vector<Operand> operands =
      GetExtractOperandsForElementOfCompositeConstruct(context, cinst,
                                                       index_into_result);

  if (operands.empty()) {
    return false;
  }

  // Add the remaining indices for extraction.
  for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
    operands.push_back(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {inst->GetSingleWordInOperand(i)}});
  }

  if (operands.size() == 1) {
    // If there were no extra indices, then we have the final object.  No need
    // to extract any more.
    inst->SetOpcode(spv::Op::OpCopyObject);
  }

  inst->SetInOperands(std::move(operands));
  return true;
}

// Walks the indexes chain from |start| to |end| of an OpCompositeInsert or
// OpCompositeExtract instruction, and returns the type of the final element
// being accessed.
const analysis::Type* GetElementType(uint32_t type_id,
                                     Instruction::iterator start,
                                     Instruction::iterator end,
                                     const analysis::TypeManager* type_mgr) {
  const analysis::Type* type = type_mgr->GetType(type_id);
  for (auto index : make_range(std::move(start), std::move(end))) {
    assert(index.type == SPV_OPERAND_TYPE_LITERAL_INTEGER &&
           index.words.size() == 1);
    if (auto* array_type = type->AsArray()) {
      type = array_type->element_type();
    } else if (auto* matrix_type = type->AsMatrix()) {
      type = matrix_type->element_type();
    } else if (auto* struct_type = type->AsStruct()) {
      type = struct_type->element_types()[index.words[0]];
    } else {
      type = nullptr;
    }
  }
  return type;
}

// Returns true of |inst_1| and |inst_2| have the same indexes that will be used
// to index into a composite object, excluding the last index.  The two
// instructions must have the same opcode, and be either OpCompositeExtract or
// OpCompositeInsert instructions.
bool HaveSameIndexesExceptForLast(Instruction* inst_1, Instruction* inst_2) {
  assert(inst_1->opcode() == inst_2->opcode() &&
         "Expecting the opcodes to be the same.");
  assert((inst_1->opcode() == spv::Op::OpCompositeInsert ||
          inst_1->opcode() == spv::Op::OpCompositeExtract) &&
         "Instructions must be OpCompositeInsert or OpCompositeExtract.");

  if (inst_1->NumInOperands() != inst_2->NumInOperands()) {
    return false;
  }

  uint32_t first_index_position =
      (inst_1->opcode() == spv::Op::OpCompositeInsert ? 2 : 1);
  for (uint32_t i = first_index_position; i < inst_1->NumInOperands() - 1;
       i++) {
    if (inst_1->GetSingleWordInOperand(i) !=
        inst_2->GetSingleWordInOperand(i)) {
      return false;
    }
  }
  return true;
}

// If the OpCompositeConstruct is simply putting back together elements that
// where extracted from the same source, we can simply reuse the source.
//
// This is a common code pattern because of the way that scalar replacement
// works.
bool CompositeExtractFeedingConstruct(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>&) {
  assert(inst->opcode() == spv::Op::OpCompositeConstruct &&
         "Wrong opcode.  Should be OpCompositeConstruct.");
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  uint32_t original_id = 0;

  if (inst->NumInOperands() == 0) {
    // The struct being constructed has no members.
    return false;
  }

  // Check each element to make sure they are:
  // - extractions
  // - extracting the same position they are inserting
  // - all extract from the same id.
  Instruction* first_element_inst = nullptr;
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    const uint32_t element_id = inst->GetSingleWordInOperand(i);
    Instruction* element_inst = def_use_mgr->GetDef(element_id);
    if (first_element_inst == nullptr) {
      first_element_inst = element_inst;
    }

    if (element_inst->opcode() != spv::Op::OpCompositeExtract) {
      return false;
    }

    if (!HaveSameIndexesExceptForLast(element_inst, first_element_inst)) {
      return false;
    }

    if (element_inst->GetSingleWordInOperand(element_inst->NumInOperands() -
                                             1) != i) {
      return false;
    }

    if (i == 0) {
      original_id =
          element_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    } else if (original_id !=
               element_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx)) {
      return false;
    }
  }

  // The last check it to see that the object being extracted from is the
  // correct type.
  Instruction* original_inst = def_use_mgr->GetDef(original_id);
  analysis::TypeManager* type_mgr = context->get_type_mgr();
  const analysis::Type* original_type =
      GetElementType(original_inst->type_id(), first_element_inst->begin() + 3,
                     first_element_inst->end() - 1, type_mgr);

  if (original_type == nullptr) {
    return false;
  }

  if (inst->type_id() != type_mgr->GetId(original_type)) {
    return false;
  }

  if (first_element_inst->NumInOperands() == 2) {
    // Simplify by using the original object.
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {original_id}}});
    return true;
  }

  // Copies the original id and all indexes except for the last to the new
  // extract instruction.
  inst->SetOpcode(spv::Op::OpCompositeExtract);
  inst->SetInOperands(std::vector<Operand>(first_element_inst->begin() + 2,
                                           first_element_inst->end() - 1));
  return true;
}

FoldingRule InsertFeedingExtract() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == spv::Op::OpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != spv::Op::OpCompositeInsert) {
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
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != spv::Op::OpVectorShuffle) {
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
      inst->SetOpcode(spv::Op::OpUndef);
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
    assert(inst->opcode() == spv::Op::OpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();

    uint32_t composite_id =
        inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    Instruction* composite_inst = def_use_mgr->GetDef(composite_id);

    if (composite_inst->opcode() != spv::Op::OpExtInst) {
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

    if (a->opcode() != spv::Op::OpCopyObject) {
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

// Returns the number of elements in the composite type |type|.  Returns 0 if
// |type| is a scalar value.
uint32_t GetNumberOfElements(const analysis::Type* type) {
  if (auto* vector_type = type->AsVector()) {
    return vector_type->element_count();
  }
  if (auto* matrix_type = type->AsMatrix()) {
    return matrix_type->element_count();
  }
  if (auto* struct_type = type->AsStruct()) {
    return static_cast<uint32_t>(struct_type->element_types().size());
  }
  if (auto* array_type = type->AsArray()) {
    return array_type->length_info().words[0];
  }
  return 0;
}

// Returns a map with the set of values that were inserted into an object by
// the chain of OpCompositeInsertInstruction starting with |inst|.
// The map will map the index to the value inserted at that index.
std::map<uint32_t, uint32_t> GetInsertedValues(Instruction* inst) {
  analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
  std::map<uint32_t, uint32_t> values_inserted;
  Instruction* current_inst = inst;
  while (current_inst->opcode() == spv::Op::OpCompositeInsert) {
    if (current_inst->NumInOperands() > inst->NumInOperands()) {
      // This is the catch the case
      //   %2 = OpCompositeInsert %m2x2int %v2int_1_0 %m2x2int_undef 0
      //   %3 = OpCompositeInsert %m2x2int %int_4 %2 0 0
      //   %4 = OpCompositeInsert %m2x2int %v2int_2_3 %3 1
      // In this case we cannot do a single construct to get the matrix.
      uint32_t partially_inserted_element_index =
          current_inst->GetSingleWordInOperand(inst->NumInOperands() - 1);
      if (values_inserted.count(partially_inserted_element_index) == 0)
        return {};
    }
    if (HaveSameIndexesExceptForLast(inst, current_inst)) {
      values_inserted.insert(
          {current_inst->GetSingleWordInOperand(current_inst->NumInOperands() -
                                                1),
           current_inst->GetSingleWordInOperand(kInsertObjectIdInIdx)});
    }
    current_inst = def_use_mgr->GetDef(
        current_inst->GetSingleWordInOperand(kInsertCompositeIdInIdx));
  }
  return values_inserted;
}

// Returns true of there is an entry in |values_inserted| for every element of
// |Type|.
bool DoInsertedValuesCoverEntireObject(
    const analysis::Type* type, std::map<uint32_t, uint32_t>& values_inserted) {
  uint32_t container_size = GetNumberOfElements(type);
  if (container_size != values_inserted.size()) {
    return false;
  }

  if (values_inserted.rbegin()->first >= container_size) {
    return false;
  }
  return true;
}

// Returns the type of the element that immediately contains the element being
// inserted by the OpCompositeInsert instruction |inst|.
const analysis::Type* GetContainerType(Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpCompositeInsert);
  analysis::TypeManager* type_mgr = inst->context()->get_type_mgr();
  return GetElementType(inst->type_id(), inst->begin() + 4, inst->end() - 1,
                        type_mgr);
}

// Returns an OpCompositeConstruct instruction that build an object with
// |type_id| out of the values in |values_inserted|.  Each value will be
// placed at the index corresponding to the value.  The new instruction will
// be placed before |insert_before|.
Instruction* BuildCompositeConstruct(
    uint32_t type_id, const std::map<uint32_t, uint32_t>& values_inserted,
    Instruction* insert_before) {
  InstructionBuilder ir_builder(
      insert_before->context(), insert_before,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  std::vector<uint32_t> ids_in_order;
  for (auto it : values_inserted) {
    ids_in_order.push_back(it.second);
  }
  Instruction* construct =
      ir_builder.AddCompositeConstruct(type_id, ids_in_order);
  return construct;
}

// Replaces the OpCompositeInsert |inst| that inserts |construct| into the same
// object as |inst| with final index removed.  If the resulting
// OpCompositeInsert instruction would have no remaining indexes, the
// instruction is replaced with an OpCopyObject instead.
void InsertConstructedObject(Instruction* inst, const Instruction* construct) {
  if (inst->NumInOperands() == 3) {
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {construct->result_id()}}});
  } else {
    inst->SetInOperand(kInsertObjectIdInIdx, {construct->result_id()});
    inst->RemoveOperand(inst->NumOperands() - 1);
  }
}

// Replaces a series of |OpCompositeInsert| instruction that cover the entire
// object with an |OpCompositeConstruct|.
bool CompositeInsertToCompositeConstruct(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>&) {
  assert(inst->opcode() == spv::Op::OpCompositeInsert &&
         "Wrong opcode.  Should be OpCompositeInsert.");
  if (inst->NumInOperands() < 3) return false;

  std::map<uint32_t, uint32_t> values_inserted = GetInsertedValues(inst);
  const analysis::Type* container_type = GetContainerType(inst);
  if (container_type == nullptr) {
    return false;
  }

  if (!DoInsertedValuesCoverEntireObject(container_type, values_inserted)) {
    return false;
  }

  analysis::TypeManager* type_mgr = context->get_type_mgr();
  Instruction* construct = BuildCompositeConstruct(
      type_mgr->GetId(container_type), values_inserted, inst);
  InsertConstructedObject(inst, construct);
  return true;
}

FoldingRule RedundantPhi() {
  // An OpPhi instruction where all values are the same or the result of the phi
  // itself, can be replaced by the value itself.
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == spv::Op::OpPhi &&
           "Wrong opcode.  Should be OpPhi.");

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
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {incoming_value}}});
    return true;
  };
}

FoldingRule BitCastScalarOrVector() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == spv::Op::OpBitcast && constants.size() == 1);
    if (constants[0] == nullptr) return false;

    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    std::vector<uint32_t> words =
        GetWordsFromNumericScalarOrVectorConstant(const_mgr, constants[0]);
    if (words.size() == 0) return false;

    const analysis::Constant* bitcasted_constant =
        ConvertWordsToNumericScalarOrVectorConstant(const_mgr, words, type);
    if (!bitcasted_constant) return false;

    auto new_feeder_id =
        const_mgr->GetDefiningInstruction(bitcasted_constant, inst->type_id())
            ->result_id();
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {new_feeder_id}}});
    return true;
  };
}

FoldingRule RedundantSelect() {
  // An OpSelect instruction where both values are the same or the condition is
  // constant can be replaced by one of the values
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == spv::Op::OpSelect &&
           "Wrong opcode.  Should be OpSelect.");
    assert(inst->NumInOperands() == 3);
    assert(constants.size() == 3);

    uint32_t true_id = inst->GetSingleWordInOperand(1);
    uint32_t false_id = inst->GetSingleWordInOperand(2);

    if (true_id == false_id) {
      // Both results are the same, condition doesn't matter
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {true_id}}});
      return true;
    } else if (constants[0]) {
      const analysis::Type* type = constants[0]->type();
      if (type->AsBool()) {
        // Scalar constant value, select the corresponding value.
        inst->SetOpcode(spv::Op::OpCopyObject);
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
          inst->SetOpcode(spv::Op::OpCopyObject);
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

          inst->SetOpcode(spv::Op::OpVectorShuffle);
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
    assert(inst->opcode() == spv::Op::OpFAdd &&
           "Wrong opcode.  Should be OpFAdd.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpFSub &&
           "Wrong opcode.  Should be OpFSub.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(spv::Op::OpFNegate);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpFMul &&
           "Wrong opcode.  Should be OpFMul.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::Zero ? 0 : 1)}}});
      return true;
    }

    if (kind0 == FloatConstantKind::One || kind1 == FloatConstantKind::One) {
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpFDiv &&
           "Wrong opcode.  Should be OpFDiv.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::One) {
      inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpExtInst &&
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
        inst->SetOpcode(spv::Op::OpCopyObject);
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
    assert(inst->opcode() == spv::Op::OpIAdd &&
           "Wrong opcode. Should be OpIAdd.");

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
        inst->SetOpcode(spv::Op::OpCopyObject);
      } else {
        inst->SetOpcode(spv::Op::OpBitcast);
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
    assert(inst->opcode() == spv::Op::OpDot &&
           "Wrong opcode.  Should be OpDot.");

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

      constexpr uint32_t kNotFound = std::numeric_limits<uint32_t>::max();

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

      inst->SetOpcode(spv::Op::OpCompositeExtract);
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
    assert(inst->opcode() == spv::Op::OpStore &&
           "Wrong opcode.  Should be OpStore.");

    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

    // If this is a volatile store, the store cannot be removed.
    if (inst->NumInOperands() == 3) {
      if (inst->GetSingleWordInOperand(2) &
          uint32_t(spv::MemoryAccessMask::Volatile)) {
        return false;
      }
    }

    uint32_t object_id = inst->GetSingleWordInOperand(kStoreObjectInIdx);
    Instruction* object_inst = def_use_mgr->GetDef(object_id);
    if (object_inst->opcode() == spv::Op::OpUndef) {
      inst->ToNop();
      return true;
    }
    return false;
  };
}

FoldingRule VectorShuffleFeedingShuffle() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == spv::Op::OpVectorShuffle &&
           "Wrong opcode.  Should be OpVectorShuffle.");

    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();

    Instruction* feeding_shuffle_inst =
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    analysis::Vector* op0_type =
        type_mgr->GetType(feeding_shuffle_inst->type_id())->AsVector();
    uint32_t op0_length = op0_type->element_count();

    bool feeder_is_op0 = true;
    if (feeding_shuffle_inst->opcode() != spv::Op::OpVectorShuffle) {
      feeding_shuffle_inst =
          def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));
      feeder_is_op0 = false;
    }

    if (feeding_shuffle_inst->opcode() != spv::Op::OpVectorShuffle) {
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
        } else if (component_index != undef_literal) {
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

        if (!feeder_is_op0 && component_index != undef_literal) {
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
          uint32_t operand = inst->GetSingleWordInOperand(i);
          if (operand >= op0_length && operand != undef_literal) {
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
    assert(inst->opcode() == spv::Op::OpEntryPoint &&
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

// If an image instruction's operand is a constant, updates the image operand
// flag from Offset to ConstOffset.
FoldingRule UpdateImageOperands() {
  return [](IRContext*, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    const auto opcode = inst->opcode();
    (void)opcode;
    assert((opcode == spv::Op::OpImageSampleImplicitLod ||
            opcode == spv::Op::OpImageSampleExplicitLod ||
            opcode == spv::Op::OpImageSampleDrefImplicitLod ||
            opcode == spv::Op::OpImageSampleDrefExplicitLod ||
            opcode == spv::Op::OpImageSampleProjImplicitLod ||
            opcode == spv::Op::OpImageSampleProjExplicitLod ||
            opcode == spv::Op::OpImageSampleProjDrefImplicitLod ||
            opcode == spv::Op::OpImageSampleProjDrefExplicitLod ||
            opcode == spv::Op::OpImageFetch ||
            opcode == spv::Op::OpImageGather ||
            opcode == spv::Op::OpImageDrefGather ||
            opcode == spv::Op::OpImageRead || opcode == spv::Op::OpImageWrite ||
            opcode == spv::Op::OpImageSparseSampleImplicitLod ||
            opcode == spv::Op::OpImageSparseSampleExplicitLod ||
            opcode == spv::Op::OpImageSparseSampleDrefImplicitLod ||
            opcode == spv::Op::OpImageSparseSampleDrefExplicitLod ||
            opcode == spv::Op::OpImageSparseSampleProjImplicitLod ||
            opcode == spv::Op::OpImageSparseSampleProjExplicitLod ||
            opcode == spv::Op::OpImageSparseSampleProjDrefImplicitLod ||
            opcode == spv::Op::OpImageSparseSampleProjDrefExplicitLod ||
            opcode == spv::Op::OpImageSparseFetch ||
            opcode == spv::Op::OpImageSparseGather ||
            opcode == spv::Op::OpImageSparseDrefGather ||
            opcode == spv::Op::OpImageSparseRead) &&
           "Wrong opcode.  Should be an image instruction.");

    int32_t operand_index = ImageOperandsMaskInOperandIndex(inst);
    if (operand_index >= 0) {
      auto image_operands = inst->GetSingleWordInOperand(operand_index);
      if (image_operands & uint32_t(spv::ImageOperandsMask::Offset)) {
        uint32_t offset_operand_index = operand_index + 1;
        if (image_operands & uint32_t(spv::ImageOperandsMask::Bias))
          offset_operand_index++;
        if (image_operands & uint32_t(spv::ImageOperandsMask::Lod))
          offset_operand_index++;
        if (image_operands & uint32_t(spv::ImageOperandsMask::Grad))
          offset_operand_index += 2;
        assert(((image_operands &
                 uint32_t(spv::ImageOperandsMask::ConstOffset)) == 0) &&
               "Offset and ConstOffset may not be used together");
        if (offset_operand_index < inst->NumOperands()) {
          if (constants[offset_operand_index]) {
            image_operands =
                image_operands | uint32_t(spv::ImageOperandsMask::ConstOffset);
            image_operands =
                image_operands & ~uint32_t(spv::ImageOperandsMask::Offset);
            inst->SetInOperand(operand_index, {image_operands});
            return true;
          }
        }
      }
    }

    return false;
  };
}

}  // namespace

void FoldingRules::AddFoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.
  rules_[spv::Op::OpBitcast].push_back(BitCastScalarOrVector());

  rules_[spv::Op::OpCompositeConstruct].push_back(
      CompositeExtractFeedingConstruct);

  rules_[spv::Op::OpCompositeExtract].push_back(InsertFeedingExtract());
  rules_[spv::Op::OpCompositeExtract].push_back(
      CompositeConstructFeedingExtract);
  rules_[spv::Op::OpCompositeExtract].push_back(VectorShuffleFeedingExtract());
  rules_[spv::Op::OpCompositeExtract].push_back(FMixFeedingExtract());

  rules_[spv::Op::OpCompositeInsert].push_back(
      CompositeInsertToCompositeConstruct);

  rules_[spv::Op::OpDot].push_back(DotProductDoingExtract());

  rules_[spv::Op::OpEntryPoint].push_back(RemoveRedundantOperands());

  rules_[spv::Op::OpFAdd].push_back(RedundantFAdd());
  rules_[spv::Op::OpFAdd].push_back(MergeAddNegateArithmetic());
  rules_[spv::Op::OpFAdd].push_back(MergeAddAddArithmetic());
  rules_[spv::Op::OpFAdd].push_back(MergeAddSubArithmetic());
  rules_[spv::Op::OpFAdd].push_back(MergeGenericAddSubArithmetic());
  rules_[spv::Op::OpFAdd].push_back(FactorAddMuls());
  rules_[spv::Op::OpFAdd].push_back(MergeMulAddArithmetic);

  rules_[spv::Op::OpFDiv].push_back(RedundantFDiv());
  rules_[spv::Op::OpFDiv].push_back(ReciprocalFDiv());
  rules_[spv::Op::OpFDiv].push_back(MergeDivDivArithmetic());
  rules_[spv::Op::OpFDiv].push_back(MergeDivMulArithmetic());
  rules_[spv::Op::OpFDiv].push_back(MergeDivNegateArithmetic());

  rules_[spv::Op::OpFMul].push_back(RedundantFMul());
  rules_[spv::Op::OpFMul].push_back(MergeMulMulArithmetic());
  rules_[spv::Op::OpFMul].push_back(MergeMulDivArithmetic());
  rules_[spv::Op::OpFMul].push_back(MergeMulNegateArithmetic());

  rules_[spv::Op::OpFNegate].push_back(MergeNegateArithmetic());
  rules_[spv::Op::OpFNegate].push_back(MergeNegateAddSubArithmetic());
  rules_[spv::Op::OpFNegate].push_back(MergeNegateMulDivArithmetic());

  rules_[spv::Op::OpFSub].push_back(RedundantFSub());
  rules_[spv::Op::OpFSub].push_back(MergeSubNegateArithmetic());
  rules_[spv::Op::OpFSub].push_back(MergeSubAddArithmetic());
  rules_[spv::Op::OpFSub].push_back(MergeSubSubArithmetic());
  rules_[spv::Op::OpFSub].push_back(MergeMulSubArithmetic);

  rules_[spv::Op::OpIAdd].push_back(RedundantIAdd());
  rules_[spv::Op::OpIAdd].push_back(MergeAddNegateArithmetic());
  rules_[spv::Op::OpIAdd].push_back(MergeAddAddArithmetic());
  rules_[spv::Op::OpIAdd].push_back(MergeAddSubArithmetic());
  rules_[spv::Op::OpIAdd].push_back(MergeGenericAddSubArithmetic());
  rules_[spv::Op::OpIAdd].push_back(FactorAddMuls());

  rules_[spv::Op::OpIMul].push_back(IntMultipleBy1());
  rules_[spv::Op::OpIMul].push_back(MergeMulMulArithmetic());
  rules_[spv::Op::OpIMul].push_back(MergeMulNegateArithmetic());

  rules_[spv::Op::OpISub].push_back(MergeSubNegateArithmetic());
  rules_[spv::Op::OpISub].push_back(MergeSubAddArithmetic());
  rules_[spv::Op::OpISub].push_back(MergeSubSubArithmetic());

  rules_[spv::Op::OpPhi].push_back(RedundantPhi());

  rules_[spv::Op::OpSNegate].push_back(MergeNegateArithmetic());
  rules_[spv::Op::OpSNegate].push_back(MergeNegateMulDivArithmetic());
  rules_[spv::Op::OpSNegate].push_back(MergeNegateAddSubArithmetic());

  rules_[spv::Op::OpSelect].push_back(RedundantSelect());

  rules_[spv::Op::OpStore].push_back(StoringUndef());

  rules_[spv::Op::OpVectorShuffle].push_back(VectorShuffleFeedingShuffle());

  rules_[spv::Op::OpImageSampleImplicitLod].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSampleExplicitLod].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSampleDrefImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSampleDrefExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSampleProjImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSampleProjExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSampleProjDrefImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSampleProjDrefExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageFetch].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageGather].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageDrefGather].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageRead].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageWrite].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleDrefImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleDrefExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleProjImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleProjExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleProjDrefImplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseSampleProjDrefExplicitLod].push_back(
      UpdateImageOperands());
  rules_[spv::Op::OpImageSparseFetch].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSparseGather].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSparseDrefGather].push_back(UpdateImageOperands());
  rules_[spv::Op::OpImageSparseRead].push_back(UpdateImageOperands());

  FeatureManager* feature_manager = context_->get_feature_mgr();
  // Add rules for GLSLstd450
  uint32_t ext_inst_glslstd450_id =
      feature_manager->GetExtInstImportId_GLSLstd450();
  if (ext_inst_glslstd450_id != 0) {
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FMix}].push_back(
        RedundantFMix());
  }
}
}  // namespace opt
}  // namespace spvtools
