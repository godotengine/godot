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

#include "source/opt/const_folding_rules.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

const uint32_t kExtractCompositeIdInIdx = 0;

// Returns true if |type| is Float or a vector of Float.
bool HasFloatingPoint(const analysis::Type* type) {
  if (type->AsFloat()) {
    return true;
  } else if (const analysis::Vector* vec_type = type->AsVector()) {
    return vec_type->element_type()->AsFloat() != nullptr;
  }

  return false;
}

// Folds an OpcompositeExtract where input is a composite constant.
ConstantFoldingRule FoldExtractWithConstants() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    const analysis::Constant* c = constants[kExtractCompositeIdInIdx];
    if (c == nullptr) {
      return nullptr;
    }

    for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
      uint32_t element_index = inst->GetSingleWordInOperand(i);
      if (c->AsNullConstant()) {
        // Return Null for the return type.
        analysis::ConstantManager* const_mgr = context->get_constant_mgr();
        analysis::TypeManager* type_mgr = context->get_type_mgr();
        return const_mgr->GetConstant(type_mgr->GetType(inst->type_id()), {});
      }

      auto cc = c->AsCompositeConstant();
      assert(cc != nullptr);
      auto components = cc->GetComponents();
      c = components[element_index];
    }
    return c;
  };
}

ConstantFoldingRule FoldVectorShuffleWithConstants() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(inst->opcode() == SpvOpVectorShuffle);
    const analysis::Constant* c1 = constants[0];
    const analysis::Constant* c2 = constants[1];
    if (c1 == nullptr || c2 == nullptr) {
      return nullptr;
    }

    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* element_type = c1->type()->AsVector()->element_type();

    std::vector<const analysis::Constant*> c1_components;
    if (const analysis::VectorConstant* vec_const = c1->AsVectorConstant()) {
      c1_components = vec_const->GetComponents();
    } else {
      assert(c1->AsNullConstant());
      const analysis::Constant* element =
          const_mgr->GetConstant(element_type, {});
      c1_components.resize(c1->type()->AsVector()->element_count(), element);
    }
    std::vector<const analysis::Constant*> c2_components;
    if (const analysis::VectorConstant* vec_const = c2->AsVectorConstant()) {
      c2_components = vec_const->GetComponents();
    } else {
      assert(c2->AsNullConstant());
      const analysis::Constant* element =
          const_mgr->GetConstant(element_type, {});
      c2_components.resize(c2->type()->AsVector()->element_count(), element);
    }

    std::vector<uint32_t> ids;
    const uint32_t undef_literal_value = 0xffffffff;
    for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
      uint32_t index = inst->GetSingleWordInOperand(i);
      if (index == undef_literal_value) {
        // Don't fold shuffle with undef literal value.
        return nullptr;
      } else if (index < c1_components.size()) {
        Instruction* member_inst =
            const_mgr->GetDefiningInstruction(c1_components[index]);
        ids.push_back(member_inst->result_id());
      } else {
        Instruction* member_inst = const_mgr->GetDefiningInstruction(
            c2_components[index - c1_components.size()]);
        ids.push_back(member_inst->result_id());
      }
    }

    analysis::TypeManager* type_mgr = context->get_type_mgr();
    return const_mgr->GetConstant(type_mgr->GetType(inst->type_id()), ids);
  };
}

ConstantFoldingRule FoldVectorTimesScalar() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(inst->opcode() == SpvOpVectorTimesScalar);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      if (HasFloatingPoint(type_mgr->GetType(inst->type_id()))) {
        return nullptr;
      }
    }

    const analysis::Constant* c1 = constants[0];
    const analysis::Constant* c2 = constants[1];

    if (c1 && c1->IsZero()) {
      return c1;
    }

    if (c2 && c2->IsZero()) {
      // Get or create the NullConstant for this type.
      std::vector<uint32_t> ids;
      return const_mgr->GetConstant(type_mgr->GetType(inst->type_id()), ids);
    }

    if (c1 == nullptr || c2 == nullptr) {
      return nullptr;
    }

    // Check result type.
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();
    assert(vector_type != nullptr);
    const analysis::Type* element_type = vector_type->element_type();
    assert(element_type != nullptr);
    const analysis::Float* float_type = element_type->AsFloat();
    assert(float_type != nullptr);

    // Check types of c1 and c2.
    assert(c1->type()->AsVector() == vector_type);
    assert(c1->type()->AsVector()->element_type() == element_type &&
           c2->type() == element_type);

    // Get a float vector that is the result of vector-times-scalar.
    std::vector<const analysis::Constant*> c1_components =
        c1->GetVectorComponents(const_mgr);
    std::vector<uint32_t> ids;
    if (float_type->width() == 32) {
      float scalar = c2->GetFloat();
      for (uint32_t i = 0; i < c1_components.size(); ++i) {
        utils::FloatProxy<float> result(c1_components[i]->GetFloat() * scalar);
        std::vector<uint32_t> words = result.GetWords();
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else if (float_type->width() == 64) {
      double scalar = c2->GetDouble();
      for (uint32_t i = 0; i < c1_components.size(); ++i) {
        utils::FloatProxy<double> result(c1_components[i]->GetDouble() *
                                         scalar);
        std::vector<uint32_t> words = result.GetWords();
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    }
    return nullptr;
  };
}

ConstantFoldingRule FoldCompositeWithConstants() {
  // Folds an OpCompositeConstruct where all of the inputs are constants to a
  // constant.  A new constant is created if necessary.
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* new_type = type_mgr->GetType(inst->type_id());
    Instruction* type_inst =
        context->get_def_use_mgr()->GetDef(inst->type_id());

    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < constants.size(); ++i) {
      const analysis::Constant* element_const = constants[i];
      if (element_const == nullptr) {
        return nullptr;
      }

      uint32_t component_type_id = 0;
      if (type_inst->opcode() == SpvOpTypeStruct) {
        component_type_id = type_inst->GetSingleWordInOperand(i);
      } else if (type_inst->opcode() == SpvOpTypeArray) {
        component_type_id = type_inst->GetSingleWordInOperand(0);
      }

      uint32_t element_id =
          const_mgr->FindDeclaredConstant(element_const, component_type_id);
      if (element_id == 0) {
        return nullptr;
      }
      ids.push_back(element_id);
    }
    return const_mgr->GetConstant(new_type, ids);
  };
}

// The interface for a function that returns the result of applying a scalar
// floating-point binary operation on |a| and |b|.  The type of the return value
// will be |type|.  The input constants must also be of type |type|.
using UnaryScalarFoldingRule = std::function<const analysis::Constant*(
    const analysis::Type* result_type, const analysis::Constant* a,
    analysis::ConstantManager*)>;

// The interface for a function that returns the result of applying a scalar
// floating-point binary operation on |a| and |b|.  The type of the return value
// will be |type|.  The input constants must also be of type |type|.
using BinaryScalarFoldingRule = std::function<const analysis::Constant*(
    const analysis::Type* result_type, const analysis::Constant* a,
    const analysis::Constant* b, analysis::ConstantManager*)>;

// Returns a |ConstantFoldingRule| that folds unary floating point scalar ops
// using |scalar_rule| and unary float point vectors ops by applying
// |scalar_rule| to the elements of the vector.  The |ConstantFoldingRule|
// that is returned assumes that |constants| contains 1 entry.  If they are
// not |nullptr|, then their type is either |Float| or |Integer| or a |Vector|
// whose element type is |Float| or |Integer|.
ConstantFoldingRule FoldFPUnaryOp(UnaryScalarFoldingRule scalar_rule) {
  return [scalar_rule](IRContext* context, Instruction* inst,
                       const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    if (constants[0] == nullptr) {
      return nullptr;
    }

    if (vector_type != nullptr) {
      std::vector<const analysis::Constant*> a_components;
      std::vector<const analysis::Constant*> results_components;

      a_components = constants[0]->GetVectorComponents(const_mgr);

      // Fold each component of the vector.
      for (uint32_t i = 0; i < a_components.size(); ++i) {
        results_components.push_back(scalar_rule(vector_type->element_type(),
                                                 a_components[i], const_mgr));
        if (results_components[i] == nullptr) {
          return nullptr;
        }
      }

      // Build the constant object and return it.
      std::vector<uint32_t> ids;
      for (const analysis::Constant* member : results_components) {
        ids.push_back(const_mgr->GetDefiningInstruction(member)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else {
      return scalar_rule(result_type, constants[0], const_mgr);
    }
  };
}

// Returns a |ConstantFoldingRule| that folds floating point scalars using
// |scalar_rule| and vectors of floating point by applying |scalar_rule| to the
// elements of the vector.  The |ConstantFoldingRule| that is returned assumes
// that |constants| contains 2 entries.  If they are not |nullptr|, then their
// type is either |Float| or a |Vector| whose element type is |Float|.
ConstantFoldingRule FoldFPBinaryOp(BinaryScalarFoldingRule scalar_rule) {
  return [scalar_rule](IRContext* context, Instruction* inst,
                       const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    if (constants[0] == nullptr || constants[1] == nullptr) {
      return nullptr;
    }

    if (vector_type != nullptr) {
      std::vector<const analysis::Constant*> a_components;
      std::vector<const analysis::Constant*> b_components;
      std::vector<const analysis::Constant*> results_components;

      a_components = constants[0]->GetVectorComponents(const_mgr);
      b_components = constants[1]->GetVectorComponents(const_mgr);

      // Fold each component of the vector.
      for (uint32_t i = 0; i < a_components.size(); ++i) {
        results_components.push_back(scalar_rule(vector_type->element_type(),
                                                 a_components[i],
                                                 b_components[i], const_mgr));
        if (results_components[i] == nullptr) {
          return nullptr;
        }
      }

      // Build the constant object and return it.
      std::vector<uint32_t> ids;
      for (const analysis::Constant* member : results_components) {
        ids.push_back(const_mgr->GetDefiningInstruction(member)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else {
      return scalar_rule(result_type, constants[0], constants[1], const_mgr);
    }
  };
}

// This macro defines a |UnaryScalarFoldingRule| that performs float to
// integer conversion.
// TODO(greg-lunarg): Support for 64-bit integer types.
UnaryScalarFoldingRule FoldFToIOp() {
  return [](const analysis::Type* result_type, const analysis::Constant* a,
            analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
    assert(result_type != nullptr && a != nullptr);
    const analysis::Integer* integer_type = result_type->AsInteger();
    const analysis::Float* float_type = a->type()->AsFloat();
    assert(float_type != nullptr);
    assert(integer_type != nullptr);
    if (integer_type->width() != 32) return nullptr;
    if (float_type->width() == 32) {
      float fa = a->GetFloat();
      uint32_t result = integer_type->IsSigned()
                            ? static_cast<uint32_t>(static_cast<int32_t>(fa))
                            : static_cast<uint32_t>(fa);
      std::vector<uint32_t> words = {result};
      return const_mgr->GetConstant(result_type, words);
    } else if (float_type->width() == 64) {
      double fa = a->GetDouble();
      uint32_t result = integer_type->IsSigned()
                            ? static_cast<uint32_t>(static_cast<int32_t>(fa))
                            : static_cast<uint32_t>(fa);
      std::vector<uint32_t> words = {result};
      return const_mgr->GetConstant(result_type, words);
    }
    return nullptr;
  };
}

// This function defines a |UnaryScalarFoldingRule| that performs integer to
// float conversion.
// TODO(greg-lunarg): Support for 64-bit integer types.
UnaryScalarFoldingRule FoldIToFOp() {
  return [](const analysis::Type* result_type, const analysis::Constant* a,
            analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
    assert(result_type != nullptr && a != nullptr);
    const analysis::Integer* integer_type = a->type()->AsInteger();
    const analysis::Float* float_type = result_type->AsFloat();
    assert(float_type != nullptr);
    assert(integer_type != nullptr);
    if (integer_type->width() != 32) return nullptr;
    uint32_t ua = a->GetU32();
    if (float_type->width() == 32) {
      float result_val = integer_type->IsSigned()
                             ? static_cast<float>(static_cast<int32_t>(ua))
                             : static_cast<float>(ua);
      utils::FloatProxy<float> result(result_val);
      std::vector<uint32_t> words = {result.data()};
      return const_mgr->GetConstant(result_type, words);
    } else if (float_type->width() == 64) {
      double result_val = integer_type->IsSigned()
                              ? static_cast<double>(static_cast<int32_t>(ua))
                              : static_cast<double>(ua);
      utils::FloatProxy<double> result(result_val);
      std::vector<uint32_t> words = result.GetWords();
      return const_mgr->GetConstant(result_type, words);
    }
    return nullptr;
  };
}

// This macro defines a |BinaryScalarFoldingRule| that applies |op|.  The
// operator |op| must work for both float and double, and use syntax "f1 op f2".
#define FOLD_FPARITH_OP(op)                                                \
  [](const analysis::Type* result_type, const analysis::Constant* a,       \
     const analysis::Constant* b,                                          \
     analysis::ConstantManager* const_mgr_in_macro)                        \
      -> const analysis::Constant* {                                       \
    assert(result_type != nullptr && a != nullptr && b != nullptr);        \
    assert(result_type == a->type() && result_type == b->type());          \
    const analysis::Float* float_type_in_macro = result_type->AsFloat();   \
    assert(float_type_in_macro != nullptr);                                \
    if (float_type_in_macro->width() == 32) {                              \
      float fa = a->GetFloat();                                            \
      float fb = b->GetFloat();                                            \
      utils::FloatProxy<float> result_in_macro(fa op fb);                  \
      std::vector<uint32_t> words_in_macro = result_in_macro.GetWords();   \
      return const_mgr_in_macro->GetConstant(result_type, words_in_macro); \
    } else if (float_type_in_macro->width() == 64) {                       \
      double fa = a->GetDouble();                                          \
      double fb = b->GetDouble();                                          \
      utils::FloatProxy<double> result_in_macro(fa op fb);                 \
      std::vector<uint32_t> words_in_macro = result_in_macro.GetWords();   \
      return const_mgr_in_macro->GetConstant(result_type, words_in_macro); \
    }                                                                      \
    return nullptr;                                                        \
  }

// Define the folding rule for conversion between floating point and integer
ConstantFoldingRule FoldFToI() { return FoldFPUnaryOp(FoldFToIOp()); }
ConstantFoldingRule FoldIToF() { return FoldFPUnaryOp(FoldIToFOp()); }

// Define the folding rules for subtraction, addition, multiplication, and
// division for floating point values.
ConstantFoldingRule FoldFSub() { return FoldFPBinaryOp(FOLD_FPARITH_OP(-)); }
ConstantFoldingRule FoldFAdd() { return FoldFPBinaryOp(FOLD_FPARITH_OP(+)); }
ConstantFoldingRule FoldFMul() { return FoldFPBinaryOp(FOLD_FPARITH_OP(*)); }
ConstantFoldingRule FoldFDiv() { return FoldFPBinaryOp(FOLD_FPARITH_OP(/)); }

bool CompareFloatingPoint(bool op_result, bool op_unordered,
                          bool need_ordered) {
  if (need_ordered) {
    // operands are ordered and Operand 1 is |op| Operand 2
    return !op_unordered && op_result;
  } else {
    // operands are unordered or Operand 1 is |op| Operand 2
    return op_unordered || op_result;
  }
}

// This macro defines a |BinaryScalarFoldingRule| that applies |op|.  The
// operator |op| must work for both float and double, and use syntax "f1 op f2".
#define FOLD_FPCMP_OP(op, ord)                                            \
  [](const analysis::Type* result_type, const analysis::Constant* a,      \
     const analysis::Constant* b,                                         \
     analysis::ConstantManager* const_mgr) -> const analysis::Constant* { \
    assert(result_type != nullptr && a != nullptr && b != nullptr);       \
    assert(result_type->AsBool());                                        \
    assert(a->type() == b->type());                                       \
    const analysis::Float* float_type = a->type()->AsFloat();             \
    assert(float_type != nullptr);                                        \
    if (float_type->width() == 32) {                                      \
      float fa = a->GetFloat();                                           \
      float fb = b->GetFloat();                                           \
      bool result = CompareFloatingPoint(                                 \
          fa op fb, std::isnan(fa) || std::isnan(fb), ord);               \
      std::vector<uint32_t> words = {uint32_t(result)};                   \
      return const_mgr->GetConstant(result_type, words);                  \
    } else if (float_type->width() == 64) {                               \
      double fa = a->GetDouble();                                         \
      double fb = b->GetDouble();                                         \
      bool result = CompareFloatingPoint(                                 \
          fa op fb, std::isnan(fa) || std::isnan(fb), ord);               \
      std::vector<uint32_t> words = {uint32_t(result)};                   \
      return const_mgr->GetConstant(result_type, words);                  \
    }                                                                     \
    return nullptr;                                                       \
  }

// Define the folding rules for ordered and unordered comparison for floating
// point values.
ConstantFoldingRule FoldFOrdEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(==, true));
}
ConstantFoldingRule FoldFUnordEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(==, false));
}
ConstantFoldingRule FoldFOrdNotEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(!=, true));
}
ConstantFoldingRule FoldFUnordNotEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(!=, false));
}
ConstantFoldingRule FoldFOrdLessThan() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(<, true));
}
ConstantFoldingRule FoldFUnordLessThan() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(<, false));
}
ConstantFoldingRule FoldFOrdGreaterThan() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(>, true));
}
ConstantFoldingRule FoldFUnordGreaterThan() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(>, false));
}
ConstantFoldingRule FoldFOrdLessThanEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(<=, true));
}
ConstantFoldingRule FoldFUnordLessThanEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(<=, false));
}
ConstantFoldingRule FoldFOrdGreaterThanEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(>=, true));
}
ConstantFoldingRule FoldFUnordGreaterThanEqual() {
  return FoldFPBinaryOp(FOLD_FPCMP_OP(>=, false));
}

// Folds an OpDot where all of the inputs are constants to a
// constant.  A new constant is created if necessary.
ConstantFoldingRule FoldOpDotWithConstants() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* new_type = type_mgr->GetType(inst->type_id());
    assert(new_type->AsFloat() && "OpDot should have a float return type.");
    const analysis::Float* float_type = new_type->AsFloat();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    // If one of the operands is 0, then the result is 0.
    bool has_zero_operand = false;

    for (int i = 0; i < 2; ++i) {
      if (constants[i]) {
        if (constants[i]->AsNullConstant() ||
            constants[i]->AsVectorConstant()->IsZero()) {
          has_zero_operand = true;
          break;
        }
      }
    }

    if (has_zero_operand) {
      if (float_type->width() == 32) {
        utils::FloatProxy<float> result(0.0f);
        std::vector<uint32_t> words = result.GetWords();
        return const_mgr->GetConstant(float_type, words);
      }
      if (float_type->width() == 64) {
        utils::FloatProxy<double> result(0.0);
        std::vector<uint32_t> words = result.GetWords();
        return const_mgr->GetConstant(float_type, words);
      }
      return nullptr;
    }

    if (constants[0] == nullptr || constants[1] == nullptr) {
      return nullptr;
    }

    std::vector<const analysis::Constant*> a_components;
    std::vector<const analysis::Constant*> b_components;

    a_components = constants[0]->GetVectorComponents(const_mgr);
    b_components = constants[1]->GetVectorComponents(const_mgr);

    utils::FloatProxy<double> result(0.0);
    std::vector<uint32_t> words = result.GetWords();
    const analysis::Constant* result_const =
        const_mgr->GetConstant(float_type, words);
    for (uint32_t i = 0; i < a_components.size() && result_const != nullptr;
         ++i) {
      if (a_components[i] == nullptr || b_components[i] == nullptr) {
        return nullptr;
      }

      const analysis::Constant* component = FOLD_FPARITH_OP(*)(
          new_type, a_components[i], b_components[i], const_mgr);
      if (component == nullptr) {
        return nullptr;
      }
      result_const =
          FOLD_FPARITH_OP(+)(new_type, result_const, component, const_mgr);
    }
    return result_const;
  };
}

// This function defines a |UnaryScalarFoldingRule| that subtracts the constant
// from zero.
UnaryScalarFoldingRule FoldFNegateOp() {
  return [](const analysis::Type* result_type, const analysis::Constant* a,
            analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
    assert(result_type != nullptr && a != nullptr);
    assert(result_type == a->type());
    const analysis::Float* float_type = result_type->AsFloat();
    assert(float_type != nullptr);
    if (float_type->width() == 32) {
      float fa = a->GetFloat();
      utils::FloatProxy<float> result(-fa);
      std::vector<uint32_t> words = result.GetWords();
      return const_mgr->GetConstant(result_type, words);
    } else if (float_type->width() == 64) {
      double da = a->GetDouble();
      utils::FloatProxy<double> result(-da);
      std::vector<uint32_t> words = result.GetWords();
      return const_mgr->GetConstant(result_type, words);
    }
    return nullptr;
  };
}

ConstantFoldingRule FoldFNegate() { return FoldFPUnaryOp(FoldFNegateOp()); }

ConstantFoldingRule FoldFClampFeedingCompare(uint32_t cmp_opcode) {
  return [cmp_opcode](IRContext* context, Instruction* inst,
                      const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    uint32_t non_const_idx = (constants[0] ? 1 : 0);
    uint32_t operand_id = inst->GetSingleWordInOperand(non_const_idx);
    Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* operand_type =
        type_mgr->GetType(operand_inst->type_id());

    if (!operand_type->AsFloat()) {
      return nullptr;
    }

    if (operand_type->AsFloat()->width() != 32 &&
        operand_type->AsFloat()->width() != 64) {
      return nullptr;
    }

    if (operand_inst->opcode() != SpvOpExtInst) {
      return nullptr;
    }

    if (operand_inst->GetSingleWordInOperand(1) != GLSLstd450FClamp) {
      return nullptr;
    }

    if (constants[1] == nullptr && constants[0] == nullptr) {
      return nullptr;
    }

    uint32_t max_id = operand_inst->GetSingleWordInOperand(4);
    const analysis::Constant* max_const =
        const_mgr->FindDeclaredConstant(max_id);

    uint32_t min_id = operand_inst->GetSingleWordInOperand(3);
    const analysis::Constant* min_const =
        const_mgr->FindDeclaredConstant(min_id);

    bool found_result = false;
    bool result = false;

    switch (cmp_opcode) {
      case SpvOpFOrdLessThan:
      case SpvOpFUnordLessThan:
      case SpvOpFOrdGreaterThanEqual:
      case SpvOpFUnordGreaterThanEqual:
        if (constants[0]) {
          if (min_const) {
            if (constants[0]->GetValueAsDouble() <
                min_const->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == SpvOpFOrdLessThan ||
                        cmp_opcode == SpvOpFUnordLessThan);
            }
          }
          if (max_const) {
            if (constants[0]->GetValueAsDouble() >=
                max_const->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == SpvOpFOrdLessThan ||
                         cmp_opcode == SpvOpFUnordLessThan);
            }
          }
        }

        if (constants[1]) {
          if (max_const) {
            if (max_const->GetValueAsDouble() <
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == SpvOpFOrdLessThan ||
                        cmp_opcode == SpvOpFUnordLessThan);
            }
          }

          if (min_const) {
            if (min_const->GetValueAsDouble() >=
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == SpvOpFOrdLessThan ||
                         cmp_opcode == SpvOpFUnordLessThan);
            }
          }
        }
        break;
      case SpvOpFOrdGreaterThan:
      case SpvOpFUnordGreaterThan:
      case SpvOpFOrdLessThanEqual:
      case SpvOpFUnordLessThanEqual:
        if (constants[0]) {
          if (min_const) {
            if (constants[0]->GetValueAsDouble() <=
                min_const->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == SpvOpFOrdLessThanEqual ||
                        cmp_opcode == SpvOpFUnordLessThanEqual);
            }
          }
          if (max_const) {
            if (constants[0]->GetValueAsDouble() >
                max_const->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == SpvOpFOrdLessThanEqual ||
                         cmp_opcode == SpvOpFUnordLessThanEqual);
            }
          }
        }

        if (constants[1]) {
          if (max_const) {
            if (max_const->GetValueAsDouble() <=
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == SpvOpFOrdLessThanEqual ||
                        cmp_opcode == SpvOpFUnordLessThanEqual);
            }
          }

          if (min_const) {
            if (min_const->GetValueAsDouble() >
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == SpvOpFOrdLessThanEqual ||
                         cmp_opcode == SpvOpFUnordLessThanEqual);
            }
          }
        }
        break;
      default:
        return nullptr;
    }

    if (!found_result) {
      return nullptr;
    }

    const analysis::Type* bool_type =
        context->get_type_mgr()->GetType(inst->type_id());
    const analysis::Constant* result_const =
        const_mgr->GetConstant(bool_type, {static_cast<uint32_t>(result)});
    assert(result_const);
    return result_const;
  };
}

}  // namespace

ConstantFoldingRules::ConstantFoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[SpvOpCompositeConstruct].push_back(FoldCompositeWithConstants());

  rules_[SpvOpCompositeExtract].push_back(FoldExtractWithConstants());

  rules_[SpvOpConvertFToS].push_back(FoldFToI());
  rules_[SpvOpConvertFToU].push_back(FoldFToI());
  rules_[SpvOpConvertSToF].push_back(FoldIToF());
  rules_[SpvOpConvertUToF].push_back(FoldIToF());

  rules_[SpvOpDot].push_back(FoldOpDotWithConstants());
  rules_[SpvOpFAdd].push_back(FoldFAdd());
  rules_[SpvOpFDiv].push_back(FoldFDiv());
  rules_[SpvOpFMul].push_back(FoldFMul());
  rules_[SpvOpFSub].push_back(FoldFSub());

  rules_[SpvOpFOrdEqual].push_back(FoldFOrdEqual());

  rules_[SpvOpFUnordEqual].push_back(FoldFUnordEqual());

  rules_[SpvOpFOrdNotEqual].push_back(FoldFOrdNotEqual());

  rules_[SpvOpFUnordNotEqual].push_back(FoldFUnordNotEqual());

  rules_[SpvOpFOrdLessThan].push_back(FoldFOrdLessThan());
  rules_[SpvOpFOrdLessThan].push_back(
      FoldFClampFeedingCompare(SpvOpFOrdLessThan));

  rules_[SpvOpFUnordLessThan].push_back(FoldFUnordLessThan());
  rules_[SpvOpFUnordLessThan].push_back(
      FoldFClampFeedingCompare(SpvOpFUnordLessThan));

  rules_[SpvOpFOrdGreaterThan].push_back(FoldFOrdGreaterThan());
  rules_[SpvOpFOrdGreaterThan].push_back(
      FoldFClampFeedingCompare(SpvOpFOrdGreaterThan));

  rules_[SpvOpFUnordGreaterThan].push_back(FoldFUnordGreaterThan());
  rules_[SpvOpFUnordGreaterThan].push_back(
      FoldFClampFeedingCompare(SpvOpFUnordGreaterThan));

  rules_[SpvOpFOrdLessThanEqual].push_back(FoldFOrdLessThanEqual());
  rules_[SpvOpFOrdLessThanEqual].push_back(
      FoldFClampFeedingCompare(SpvOpFOrdLessThanEqual));

  rules_[SpvOpFUnordLessThanEqual].push_back(FoldFUnordLessThanEqual());
  rules_[SpvOpFUnordLessThanEqual].push_back(
      FoldFClampFeedingCompare(SpvOpFUnordLessThanEqual));

  rules_[SpvOpFOrdGreaterThanEqual].push_back(FoldFOrdGreaterThanEqual());
  rules_[SpvOpFOrdGreaterThanEqual].push_back(
      FoldFClampFeedingCompare(SpvOpFOrdGreaterThanEqual));

  rules_[SpvOpFUnordGreaterThanEqual].push_back(FoldFUnordGreaterThanEqual());
  rules_[SpvOpFUnordGreaterThanEqual].push_back(
      FoldFClampFeedingCompare(SpvOpFUnordGreaterThanEqual));

  rules_[SpvOpVectorShuffle].push_back(FoldVectorShuffleWithConstants());
  rules_[SpvOpVectorTimesScalar].push_back(FoldVectorTimesScalar());

  rules_[SpvOpFNegate].push_back(FoldFNegate());
}
}  // namespace opt
}  // namespace spvtools
