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
constexpr uint32_t kExtractCompositeIdInIdx = 0;

// Returns a constants with the value NaN of the given type.  Only works for
// 32-bit and 64-bit float point types.  Returns |nullptr| if an error occurs.
const analysis::Constant* GetNan(const analysis::Type* type,
                                 analysis::ConstantManager* const_mgr) {
  const analysis::Float* float_type = type->AsFloat();
  if (float_type == nullptr) {
    return nullptr;
  }

  switch (float_type->width()) {
    case 32:
      return const_mgr->GetFloatConst(std::numeric_limits<float>::quiet_NaN());
    case 64:
      return const_mgr->GetDoubleConst(
          std::numeric_limits<double>::quiet_NaN());
    default:
      return nullptr;
  }
}

// Returns a constants with the value INF of the given type.  Only works for
// 32-bit and 64-bit float point types.  Returns |nullptr| if an error occurs.
const analysis::Constant* GetInf(const analysis::Type* type,
                                 analysis::ConstantManager* const_mgr) {
  const analysis::Float* float_type = type->AsFloat();
  if (float_type == nullptr) {
    return nullptr;
  }

  switch (float_type->width()) {
    case 32:
      return const_mgr->GetFloatConst(std::numeric_limits<float>::infinity());
    case 64:
      return const_mgr->GetDoubleConst(std::numeric_limits<double>::infinity());
    default:
      return nullptr;
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

// Returns a constants with the value |-val| of the given type.  Only works for
// 32-bit and 64-bit float point types.  Returns |nullptr| if an error occurs.
const analysis::Constant* NegateFPConst(const analysis::Type* result_type,
                                        const analysis::Constant* val,
                                        analysis::ConstantManager* const_mgr) {
  const analysis::Float* float_type = result_type->AsFloat();
  assert(float_type != nullptr);
  if (float_type->width() == 32) {
    float fa = val->GetFloat();
    return const_mgr->GetFloatConst(-fa);
  } else if (float_type->width() == 64) {
    double da = val->GetDouble();
    return const_mgr->GetDoubleConst(-da);
  }
  return nullptr;
}

// Returns a constants with the value |-val| of the given type.
const analysis::Constant* NegateIntConst(const analysis::Type* result_type,
                                         const analysis::Constant* val,
                                         analysis::ConstantManager* const_mgr) {
  const analysis::Integer* int_type = result_type->AsInteger();
  assert(int_type != nullptr);

  if (val->AsNullConstant()) {
    return val;
  }

  uint64_t new_value = static_cast<uint64_t>(-val->GetSignExtendedValue());
  return const_mgr->GetIntConst(new_value, int_type->width(),
                                int_type->IsSigned());
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
      // Protect against invalid IR.  Refuse to fold if the index is out
      // of bounds.
      if (element_index >= components.size()) return nullptr;
      c = components[element_index];
    }
    return c;
  };
}

// Folds an OpcompositeInsert where input is a composite constant.
ConstantFoldingRule FoldInsertWithConstants() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Constant* object = constants[0];
    const analysis::Constant* composite = constants[1];
    if (object == nullptr || composite == nullptr) {
      return nullptr;
    }

    // If there is more than 1 index, then each additional constant used by the
    // index will need to be recreated to use the inserted object.
    std::vector<const analysis::Constant*> chain;
    std::vector<const analysis::Constant*> components;
    const analysis::Type* type = nullptr;
    const uint32_t final_index = (inst->NumInOperands() - 1);

    // Work down hierarchy of all indexes
    for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
      type = composite->type();

      if (composite->AsNullConstant()) {
        // Make new composite so it can be inserted in the index with the
        // non-null value
        if (const auto new_composite =
                const_mgr->GetNullCompositeConstant(type)) {
          // Keep track of any indexes along the way to last index
          if (i != final_index) {
            chain.push_back(new_composite);
          }
          components = new_composite->AsCompositeConstant()->GetComponents();
        } else {
          // Unsupported input type (such as structs)
          return nullptr;
        }
      } else {
        // Keep track of any indexes along the way to last index
        if (i != final_index) {
          chain.push_back(composite);
        }
        components = composite->AsCompositeConstant()->GetComponents();
      }
      const uint32_t index = inst->GetSingleWordInOperand(i);
      composite = components[index];
    }

    // Final index in hierarchy is inserted with new object.
    const uint32_t final_operand = inst->GetSingleWordInOperand(final_index);
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < components.size(); i++) {
      const analysis::Constant* constant =
          (i == final_operand) ? object : components[i];
      Instruction* member_inst = const_mgr->GetDefiningInstruction(constant);
      ids.push_back(member_inst->result_id());
    }
    const analysis::Constant* new_constant = const_mgr->GetConstant(type, ids);

    // Work backwards up the chain and replace each index with new constant.
    for (size_t i = chain.size(); i > 0; i--) {
      // Need to insert any previous instruction into the module first.
      // Can't just insert in types_values_begin() because it will move above
      // where the types are declared.
      // Can't compare with location of inst because not all new added
      // instructions are added to types_values_
      auto iter = context->types_values_end();
      Module::inst_iterator* pos = &iter;
      const_mgr->BuildInstructionAndAddToModule(new_constant, pos);

      composite = chain[i - 1];
      components = composite->AsCompositeConstant()->GetComponents();
      type = composite->type();
      ids.clear();
      for (size_t k = 0; k < components.size(); k++) {
        const uint32_t index =
            inst->GetSingleWordInOperand(1 + static_cast<uint32_t>(i));
        const analysis::Constant* constant =
            (k == index) ? new_constant : components[k];
        const uint32_t constant_id =
            const_mgr->FindDeclaredConstant(constant, 0);
        ids.push_back(constant_id);
      }
      new_constant = const_mgr->GetConstant(type, ids);
    }

    // If multiple constants were created, only need to return the top index.
    return new_constant;
  };
}

ConstantFoldingRule FoldVectorShuffleWithConstants() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(inst->opcode() == spv::Op::OpVectorShuffle);
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
    assert(inst->opcode() == spv::Op::OpVectorTimesScalar);
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

// Returns to the constant that results from tranposing |matrix|. The result
// will have type |result_type|, and |matrix| must exist in |context|. The
// result constant will also exist in |context|.
const analysis::Constant* TransposeMatrix(const analysis::Constant* matrix,
                                          analysis::Matrix* result_type,
                                          IRContext* context) {
  analysis::ConstantManager* const_mgr = context->get_constant_mgr();
  if (matrix->AsNullConstant() != nullptr) {
    return const_mgr->GetNullCompositeConstant(result_type);
  }

  const auto& columns = matrix->AsMatrixConstant()->GetComponents();
  uint32_t number_of_rows = columns[0]->type()->AsVector()->element_count();

  // Collect the ids of the elements in their new positions.
  std::vector<std::vector<uint32_t>> result_elements(number_of_rows);
  for (const analysis::Constant* column : columns) {
    if (column->AsNullConstant()) {
      column = const_mgr->GetNullCompositeConstant(column->type());
    }
    const auto& column_components = column->AsVectorConstant()->GetComponents();

    for (uint32_t row = 0; row < number_of_rows; ++row) {
      result_elements[row].push_back(
          const_mgr->GetDefiningInstruction(column_components[row])
              ->result_id());
    }
  }

  // Create the constant for each row in the result, and collect the ids.
  std::vector<uint32_t> result_columns(number_of_rows);
  for (uint32_t col = 0; col < number_of_rows; ++col) {
    auto* element = const_mgr->GetConstant(result_type->element_type(),
                                           result_elements[col]);
    result_columns[col] =
        const_mgr->GetDefiningInstruction(element)->result_id();
  }

  // Create the matrix constant from the row ids, and return it.
  return const_mgr->GetConstant(result_type, result_columns);
}

const analysis::Constant* FoldTranspose(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>& constants) {
  assert(inst->opcode() == spv::Op::OpTranspose);

  analysis::TypeManager* type_mgr = context->get_type_mgr();
  if (!inst->IsFloatingPointFoldingAllowed()) {
    if (HasFloatingPoint(type_mgr->GetType(inst->type_id()))) {
      return nullptr;
    }
  }

  const analysis::Constant* matrix = constants[0];
  if (matrix == nullptr) {
    return nullptr;
  }

  auto* result_type = type_mgr->GetType(inst->type_id());
  return TransposeMatrix(matrix, result_type->AsMatrix(), context);
}

ConstantFoldingRule FoldVectorTimesMatrix() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(inst->opcode() == spv::Op::OpVectorTimesMatrix);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      if (HasFloatingPoint(type_mgr->GetType(inst->type_id()))) {
        return nullptr;
      }
    }

    const analysis::Constant* c1 = constants[0];
    const analysis::Constant* c2 = constants[1];

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
           c2->type()->AsMatrix()->element_type() == vector_type);

    uint32_t resultVectorSize = result_type->AsVector()->element_count();
    std::vector<uint32_t> ids;

    if ((c1 && c1->IsZero()) || (c2 && c2->IsZero())) {
      std::vector<uint32_t> words(float_type->width() / 32, 0);
      for (uint32_t i = 0; i < resultVectorSize; ++i) {
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    }

    // Get a float vector that is the result of vector-times-matrix.
    std::vector<const analysis::Constant*> c1_components =
        c1->GetVectorComponents(const_mgr);
    std::vector<const analysis::Constant*> c2_components =
        c2->AsMatrixConstant()->GetComponents();

    if (float_type->width() == 32) {
      for (uint32_t i = 0; i < resultVectorSize; ++i) {
        float result_scalar = 0.0f;
        if (!c2_components[i]->AsNullConstant()) {
          const analysis::VectorConstant* c2_vec =
              c2_components[i]->AsVectorConstant();
          for (uint32_t j = 0; j < c2_vec->GetComponents().size(); ++j) {
            float c1_scalar = c1_components[j]->GetFloat();
            float c2_scalar = c2_vec->GetComponents()[j]->GetFloat();
            result_scalar += c1_scalar * c2_scalar;
          }
        }
        utils::FloatProxy<float> result(result_scalar);
        std::vector<uint32_t> words = result.GetWords();
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else if (float_type->width() == 64) {
      for (uint32_t i = 0; i < c2_components.size(); ++i) {
        double result_scalar = 0.0;
        if (!c2_components[i]->AsNullConstant()) {
          const analysis::VectorConstant* c2_vec =
              c2_components[i]->AsVectorConstant();
          for (uint32_t j = 0; j < c2_vec->GetComponents().size(); ++j) {
            double c1_scalar = c1_components[j]->GetDouble();
            double c2_scalar = c2_vec->GetComponents()[j]->GetDouble();
            result_scalar += c1_scalar * c2_scalar;
          }
        }
        utils::FloatProxy<double> result(result_scalar);
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

ConstantFoldingRule FoldMatrixTimesVector() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(inst->opcode() == spv::Op::OpMatrixTimesVector);
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      if (HasFloatingPoint(type_mgr->GetType(inst->type_id()))) {
        return nullptr;
      }
    }

    const analysis::Constant* c1 = constants[0];
    const analysis::Constant* c2 = constants[1];

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
    assert(c1->type()->AsMatrix()->element_type() == vector_type);
    assert(c2->type()->AsVector()->element_type() == element_type);

    uint32_t resultVectorSize = result_type->AsVector()->element_count();
    std::vector<uint32_t> ids;

    if ((c1 && c1->IsZero()) || (c2 && c2->IsZero())) {
      std::vector<uint32_t> words(float_type->width() / 32, 0);
      for (uint32_t i = 0; i < resultVectorSize; ++i) {
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    }

    // Get a float vector that is the result of matrix-times-vector.
    std::vector<const analysis::Constant*> c1_components =
        c1->AsMatrixConstant()->GetComponents();
    std::vector<const analysis::Constant*> c2_components =
        c2->GetVectorComponents(const_mgr);

    if (float_type->width() == 32) {
      for (uint32_t i = 0; i < resultVectorSize; ++i) {
        float result_scalar = 0.0f;
        for (uint32_t j = 0; j < c1_components.size(); ++j) {
          if (!c1_components[j]->AsNullConstant()) {
            float c1_scalar = c1_components[j]
                                  ->AsVectorConstant()
                                  ->GetComponents()[i]
                                  ->GetFloat();
            float c2_scalar = c2_components[j]->GetFloat();
            result_scalar += c1_scalar * c2_scalar;
          }
        }
        utils::FloatProxy<float> result(result_scalar);
        std::vector<uint32_t> words = result.GetWords();
        const analysis::Constant* new_elem =
            const_mgr->GetConstant(float_type, words);
        ids.push_back(const_mgr->GetDefiningInstruction(new_elem)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else if (float_type->width() == 64) {
      for (uint32_t i = 0; i < resultVectorSize; ++i) {
        double result_scalar = 0.0;
        for (uint32_t j = 0; j < c1_components.size(); ++j) {
          if (!c1_components[j]->AsNullConstant()) {
            double c1_scalar = c1_components[j]
                                   ->AsVectorConstant()
                                   ->GetComponents()[i]
                                   ->GetDouble();
            double c2_scalar = c2_components[j]->GetDouble();
            result_scalar += c1_scalar * c2_scalar;
          }
        }
        utils::FloatProxy<double> result(result_scalar);
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
      if (type_inst->opcode() == spv::Op::OpTypeStruct) {
        component_type_id = type_inst->GetSingleWordInOperand(i);
      } else if (type_inst->opcode() == spv::Op::OpTypeArray) {
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

// Returns a |ConstantFoldingRule| that folds unary scalar ops
// using |scalar_rule| and unary vectors ops by applying
// |scalar_rule| to the elements of the vector.  The |ConstantFoldingRule|
// that is returned assumes that |constants| contains 1 entry.  If they are
// not |nullptr|, then their type is either |Float| or |Integer| or a |Vector|
// whose element type is |Float| or |Integer|.
ConstantFoldingRule FoldUnaryOp(UnaryScalarFoldingRule scalar_rule) {
  return [scalar_rule](IRContext* context, Instruction* inst,
                       const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();

    const analysis::Constant* arg =
        (inst->opcode() == spv::Op::OpExtInst) ? constants[1] : constants[0];

    if (arg == nullptr) {
      return nullptr;
    }

    if (vector_type != nullptr) {
      std::vector<const analysis::Constant*> a_components;
      std::vector<const analysis::Constant*> results_components;

      a_components = arg->GetVectorComponents(const_mgr);

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
      return scalar_rule(result_type, arg, const_mgr);
    }
  };
}

// Returns a |ConstantFoldingRule| that folds binary scalar ops
// using |scalar_rule| and binary vectors ops by applying
// |scalar_rule| to the elements of the vector. The folding rule assumes that op
// has two inputs. For regular instruction, those are in operands 0 and 1. For
// extended instruction, they are in operands 1 and 2. If an element in
// |constants| is not nullprt, then the constant's type is |Float|, |Integer|,
// or |Vector| whose element type is |Float| or |Integer|.
ConstantFoldingRule FoldBinaryOp(BinaryScalarFoldingRule scalar_rule) {
  return [scalar_rule](IRContext* context, Instruction* inst,
                       const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    assert(constants.size() == inst->NumInOperands());
    assert(constants.size() == (inst->opcode() == spv::Op::OpExtInst ? 3 : 2));
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();

    const analysis::Constant* arg1 =
        (inst->opcode() == spv::Op::OpExtInst) ? constants[1] : constants[0];
    const analysis::Constant* arg2 =
        (inst->opcode() == spv::Op::OpExtInst) ? constants[2] : constants[1];

    if (arg1 == nullptr || arg2 == nullptr) {
      return nullptr;
    }

    if (vector_type == nullptr) {
      return scalar_rule(result_type, arg1, arg2, const_mgr);
    }

    std::vector<const analysis::Constant*> a_components;
    std::vector<const analysis::Constant*> b_components;
    std::vector<const analysis::Constant*> results_components;

    a_components = arg1->GetVectorComponents(const_mgr);
    b_components = arg2->GetVectorComponents(const_mgr);
    assert(a_components.size() == b_components.size());

    // Fold each component of the vector.
    for (uint32_t i = 0; i < a_components.size(); ++i) {
      results_components.push_back(scalar_rule(vector_type->element_type(),
                                               a_components[i], b_components[i],
                                               const_mgr));
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
  };
}

// Returns a |ConstantFoldingRule| that folds unary floating point scalar ops
// using |scalar_rule| and unary float point vectors ops by applying
// |scalar_rule| to the elements of the vector.  The |ConstantFoldingRule|
// that is returned assumes that |constants| contains 1 entry.  If they are
// not |nullptr|, then their type is either |Float| or |Integer| or a |Vector|
// whose element type is |Float| or |Integer|.
ConstantFoldingRule FoldFPUnaryOp(UnaryScalarFoldingRule scalar_rule) {
  auto folding_rule = FoldUnaryOp(scalar_rule);
  return [folding_rule](IRContext* context, Instruction* inst,
                        const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    return folding_rule(context, inst, constants);
  };
}

// Returns the result of folding the constants in |constants| according the
// |scalar_rule|.  If |result_type| is a vector, then |scalar_rule| is applied
// per component.
const analysis::Constant* FoldFPBinaryOp(
    BinaryScalarFoldingRule scalar_rule, uint32_t result_type_id,
    const std::vector<const analysis::Constant*>& constants,
    IRContext* context) {
  analysis::ConstantManager* const_mgr = context->get_constant_mgr();
  analysis::TypeManager* type_mgr = context->get_type_mgr();
  const analysis::Type* result_type = type_mgr->GetType(result_type_id);
  const analysis::Vector* vector_type = result_type->AsVector();

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
                                               a_components[i], b_components[i],
                                               const_mgr));
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
    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }
    if (inst->opcode() == spv::Op::OpExtInst) {
      return FoldFPBinaryOp(scalar_rule, inst->type_id(),
                            {constants[1], constants[2]}, context);
    }
    return FoldFPBinaryOp(scalar_rule, inst->type_id(), constants, context);
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

// This defines a |UnaryScalarFoldingRule| that performs |OpQuantizeToF16|.
UnaryScalarFoldingRule FoldQuantizeToF16Scalar() {
  return [](const analysis::Type* result_type, const analysis::Constant* a,
            analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
    assert(result_type != nullptr && a != nullptr);
    const analysis::Float* float_type = a->type()->AsFloat();
    assert(float_type != nullptr);
    if (float_type->width() != 32) {
      return nullptr;
    }

    float fa = a->GetFloat();
    utils::HexFloat<utils::FloatProxy<float>> orignal(fa);
    utils::HexFloat<utils::FloatProxy<utils::Float16>> quantized(0);
    utils::HexFloat<utils::FloatProxy<float>> result(0.0f);
    orignal.castTo(quantized, utils::round_direction::kToZero);
    quantized.castTo(result, utils::round_direction::kToZero);
    std::vector<uint32_t> words = {result.getBits()};
    return const_mgr->GetConstant(result_type, words);
  };
}

// This macro defines a |BinaryScalarFoldingRule| that applies |op|.  The
// operator |op| must work for both float and double, and use syntax "f1 op f2".
#define FOLD_FPARITH_OP(op)                                                   \
  [](const analysis::Type* result_type_in_macro, const analysis::Constant* a, \
     const analysis::Constant* b,                                             \
     analysis::ConstantManager* const_mgr_in_macro)                           \
      -> const analysis::Constant* {                                          \
    assert(result_type_in_macro != nullptr && a != nullptr && b != nullptr);  \
    assert(result_type_in_macro == a->type() &&                               \
           result_type_in_macro == b->type());                                \
    const analysis::Float* float_type_in_macro =                              \
        result_type_in_macro->AsFloat();                                      \
    assert(float_type_in_macro != nullptr);                                   \
    if (float_type_in_macro->width() == 32) {                                 \
      float fa = a->GetFloat();                                               \
      float fb = b->GetFloat();                                               \
      utils::FloatProxy<float> result_in_macro(fa op fb);                     \
      std::vector<uint32_t> words_in_macro = result_in_macro.GetWords();      \
      return const_mgr_in_macro->GetConstant(result_type_in_macro,            \
                                             words_in_macro);                 \
    } else if (float_type_in_macro->width() == 64) {                          \
      double fa = a->GetDouble();                                             \
      double fb = b->GetDouble();                                             \
      utils::FloatProxy<double> result_in_macro(fa op fb);                    \
      std::vector<uint32_t> words_in_macro = result_in_macro.GetWords();      \
      return const_mgr_in_macro->GetConstant(result_type_in_macro,            \
                                             words_in_macro);                 \
    }                                                                         \
    return nullptr;                                                           \
  }

// Define the folding rule for conversion between floating point and integer
ConstantFoldingRule FoldFToI() { return FoldFPUnaryOp(FoldFToIOp()); }
ConstantFoldingRule FoldIToF() { return FoldFPUnaryOp(FoldIToFOp()); }
ConstantFoldingRule FoldQuantizeToF16() {
  return FoldFPUnaryOp(FoldQuantizeToF16Scalar());
}

// Define the folding rules for subtraction, addition, multiplication, and
// division for floating point values.
ConstantFoldingRule FoldFSub() { return FoldFPBinaryOp(FOLD_FPARITH_OP(-)); }
ConstantFoldingRule FoldFAdd() { return FoldFPBinaryOp(FOLD_FPARITH_OP(+)); }
ConstantFoldingRule FoldFMul() { return FoldFPBinaryOp(FOLD_FPARITH_OP(*)); }

// Returns the constant that results from evaluating |numerator| / 0.0.  Returns
// |nullptr| if the result could not be evaluated.
const analysis::Constant* FoldFPScalarDivideByZero(
    const analysis::Type* result_type, const analysis::Constant* numerator,
    analysis::ConstantManager* const_mgr) {
  if (numerator == nullptr) {
    return nullptr;
  }

  if (numerator->IsZero()) {
    return GetNan(result_type, const_mgr);
  }

  const analysis::Constant* result = GetInf(result_type, const_mgr);
  if (result == nullptr) {
    return nullptr;
  }

  if (numerator->AsFloatConstant()->GetValueAsDouble() < 0.0) {
    result = NegateFPConst(result_type, result, const_mgr);
  }
  return result;
}

// Returns the result of folding |numerator| / |denominator|.  Returns |nullptr|
// if it cannot be folded.
const analysis::Constant* FoldScalarFPDivide(
    const analysis::Type* result_type, const analysis::Constant* numerator,
    const analysis::Constant* denominator,
    analysis::ConstantManager* const_mgr) {
  if (denominator == nullptr) {
    return nullptr;
  }

  if (denominator->IsZero()) {
    return FoldFPScalarDivideByZero(result_type, numerator, const_mgr);
  }

  uint32_t width = denominator->type()->AsFloat()->width();
  if (width != 32 && width != 64) {
    return nullptr;
  }

  const analysis::FloatConstant* denominator_float =
      denominator->AsFloatConstant();
  if (denominator_float && denominator->GetValueAsDouble() == -0.0) {
    const analysis::Constant* result =
        FoldFPScalarDivideByZero(result_type, numerator, const_mgr);
    if (result != nullptr)
      result = NegateFPConst(result_type, result, const_mgr);
    return result;
  } else {
    return FOLD_FPARITH_OP(/)(result_type, numerator, denominator, const_mgr);
  }
}

// Returns the constant folding rule to fold |OpFDiv| with two constants.
ConstantFoldingRule FoldFDiv() { return FoldFPBinaryOp(FoldScalarFPDivide); }

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

ConstantFoldingRule FoldFNegate() { return FoldFPUnaryOp(NegateFPConst); }
ConstantFoldingRule FoldSNegate() { return FoldUnaryOp(NegateIntConst); }

ConstantFoldingRule FoldFClampFeedingCompare(spv::Op cmp_opcode) {
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

    if (operand_inst->opcode() != spv::Op::OpExtInst) {
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
      case spv::Op::OpFOrdLessThan:
      case spv::Op::OpFUnordLessThan:
      case spv::Op::OpFOrdGreaterThanEqual:
      case spv::Op::OpFUnordGreaterThanEqual:
        if (constants[0]) {
          if (min_const) {
            if (constants[0]->GetValueAsDouble() <
                min_const->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == spv::Op::OpFOrdLessThan ||
                        cmp_opcode == spv::Op::OpFUnordLessThan);
            }
          }
          if (max_const) {
            if (constants[0]->GetValueAsDouble() >=
                max_const->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == spv::Op::OpFOrdLessThan ||
                         cmp_opcode == spv::Op::OpFUnordLessThan);
            }
          }
        }

        if (constants[1]) {
          if (max_const) {
            if (max_const->GetValueAsDouble() <
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == spv::Op::OpFOrdLessThan ||
                        cmp_opcode == spv::Op::OpFUnordLessThan);
            }
          }

          if (min_const) {
            if (min_const->GetValueAsDouble() >=
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == spv::Op::OpFOrdLessThan ||
                         cmp_opcode == spv::Op::OpFUnordLessThan);
            }
          }
        }
        break;
      case spv::Op::OpFOrdGreaterThan:
      case spv::Op::OpFUnordGreaterThan:
      case spv::Op::OpFOrdLessThanEqual:
      case spv::Op::OpFUnordLessThanEqual:
        if (constants[0]) {
          if (min_const) {
            if (constants[0]->GetValueAsDouble() <=
                min_const->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == spv::Op::OpFOrdLessThanEqual ||
                        cmp_opcode == spv::Op::OpFUnordLessThanEqual);
            }
          }
          if (max_const) {
            if (constants[0]->GetValueAsDouble() >
                max_const->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == spv::Op::OpFOrdLessThanEqual ||
                         cmp_opcode == spv::Op::OpFUnordLessThanEqual);
            }
          }
        }

        if (constants[1]) {
          if (max_const) {
            if (max_const->GetValueAsDouble() <=
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = (cmp_opcode == spv::Op::OpFOrdLessThanEqual ||
                        cmp_opcode == spv::Op::OpFUnordLessThanEqual);
            }
          }

          if (min_const) {
            if (min_const->GetValueAsDouble() >
                constants[1]->GetValueAsDouble()) {
              found_result = true;
              result = !(cmp_opcode == spv::Op::OpFOrdLessThanEqual ||
                         cmp_opcode == spv::Op::OpFUnordLessThanEqual);
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

ConstantFoldingRule FoldFMix() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    assert(inst->opcode() == spv::Op::OpExtInst &&
           "Expecting an extended instruction.");
    assert(inst->GetSingleWordInOperand(0) ==
               context->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
           "Expecting a GLSLstd450 extended instruction.");
    assert(inst->GetSingleWordInOperand(1) == GLSLstd450FMix &&
           "Expecting and FMix instruction.");

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    // Make sure all FMix operands are constants.
    for (uint32_t i = 1; i < 4; i++) {
      if (constants[i] == nullptr) {
        return nullptr;
      }
    }

    const analysis::Constant* one;
    bool is_vector = false;
    const analysis::Type* result_type = constants[1]->type();
    const analysis::Type* base_type = result_type;
    if (base_type->AsVector()) {
      is_vector = true;
      base_type = base_type->AsVector()->element_type();
    }
    assert(base_type->AsFloat() != nullptr &&
           "FMix is suppose to act on floats or vectors of floats.");

    if (base_type->AsFloat()->width() == 32) {
      one = const_mgr->GetConstant(base_type,
                                   utils::FloatProxy<float>(1.0f).GetWords());
    } else {
      one = const_mgr->GetConstant(base_type,
                                   utils::FloatProxy<double>(1.0).GetWords());
    }

    if (is_vector) {
      uint32_t one_id = const_mgr->GetDefiningInstruction(one)->result_id();
      one =
          const_mgr->GetConstant(result_type, std::vector<uint32_t>(4, one_id));
    }

    const analysis::Constant* temp1 = FoldFPBinaryOp(
        FOLD_FPARITH_OP(-), inst->type_id(), {one, constants[3]}, context);
    if (temp1 == nullptr) {
      return nullptr;
    }

    const analysis::Constant* temp2 = FoldFPBinaryOp(
        FOLD_FPARITH_OP(*), inst->type_id(), {constants[1], temp1}, context);
    if (temp2 == nullptr) {
      return nullptr;
    }
    const analysis::Constant* temp3 =
        FoldFPBinaryOp(FOLD_FPARITH_OP(*), inst->type_id(),
                       {constants[2], constants[3]}, context);
    if (temp3 == nullptr) {
      return nullptr;
    }
    return FoldFPBinaryOp(FOLD_FPARITH_OP(+), inst->type_id(), {temp2, temp3},
                          context);
  };
}

const analysis::Constant* FoldMin(const analysis::Type* result_type,
                                  const analysis::Constant* a,
                                  const analysis::Constant* b,
                                  analysis::ConstantManager*) {
  if (const analysis::Integer* int_type = result_type->AsInteger()) {
    if (int_type->width() == 32) {
      if (int_type->IsSigned()) {
        int32_t va = a->GetS32();
        int32_t vb = b->GetS32();
        return (va < vb ? a : b);
      } else {
        uint32_t va = a->GetU32();
        uint32_t vb = b->GetU32();
        return (va < vb ? a : b);
      }
    } else if (int_type->width() == 64) {
      if (int_type->IsSigned()) {
        int64_t va = a->GetS64();
        int64_t vb = b->GetS64();
        return (va < vb ? a : b);
      } else {
        uint64_t va = a->GetU64();
        uint64_t vb = b->GetU64();
        return (va < vb ? a : b);
      }
    }
  } else if (const analysis::Float* float_type = result_type->AsFloat()) {
    if (float_type->width() == 32) {
      float va = a->GetFloat();
      float vb = b->GetFloat();
      return (va < vb ? a : b);
    } else if (float_type->width() == 64) {
      double va = a->GetDouble();
      double vb = b->GetDouble();
      return (va < vb ? a : b);
    }
  }
  return nullptr;
}

const analysis::Constant* FoldMax(const analysis::Type* result_type,
                                  const analysis::Constant* a,
                                  const analysis::Constant* b,
                                  analysis::ConstantManager*) {
  if (const analysis::Integer* int_type = result_type->AsInteger()) {
    if (int_type->width() == 32) {
      if (int_type->IsSigned()) {
        int32_t va = a->GetS32();
        int32_t vb = b->GetS32();
        return (va > vb ? a : b);
      } else {
        uint32_t va = a->GetU32();
        uint32_t vb = b->GetU32();
        return (va > vb ? a : b);
      }
    } else if (int_type->width() == 64) {
      if (int_type->IsSigned()) {
        int64_t va = a->GetS64();
        int64_t vb = b->GetS64();
        return (va > vb ? a : b);
      } else {
        uint64_t va = a->GetU64();
        uint64_t vb = b->GetU64();
        return (va > vb ? a : b);
      }
    }
  } else if (const analysis::Float* float_type = result_type->AsFloat()) {
    if (float_type->width() == 32) {
      float va = a->GetFloat();
      float vb = b->GetFloat();
      return (va > vb ? a : b);
    } else if (float_type->width() == 64) {
      double va = a->GetDouble();
      double vb = b->GetDouble();
      return (va > vb ? a : b);
    }
  }
  return nullptr;
}

// Fold an clamp instruction when all three operands are constant.
const analysis::Constant* FoldClamp1(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>& constants) {
  assert(inst->opcode() == spv::Op::OpExtInst &&
         "Expecting an extended instruction.");
  assert(inst->GetSingleWordInOperand(0) ==
             context->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
         "Expecting a GLSLstd450 extended instruction.");

  // Make sure all Clamp operands are constants.
  for (uint32_t i = 1; i < 4; i++) {
    if (constants[i] == nullptr) {
      return nullptr;
    }
  }

  const analysis::Constant* temp = FoldFPBinaryOp(
      FoldMax, inst->type_id(), {constants[1], constants[2]}, context);
  if (temp == nullptr) {
    return nullptr;
  }
  return FoldFPBinaryOp(FoldMin, inst->type_id(), {temp, constants[3]},
                        context);
}

// Fold a clamp instruction when |x <= min_val|.
const analysis::Constant* FoldClamp2(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>& constants) {
  assert(inst->opcode() == spv::Op::OpExtInst &&
         "Expecting an extended instruction.");
  assert(inst->GetSingleWordInOperand(0) ==
             context->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
         "Expecting a GLSLstd450 extended instruction.");

  const analysis::Constant* x = constants[1];
  const analysis::Constant* min_val = constants[2];

  if (x == nullptr || min_val == nullptr) {
    return nullptr;
  }

  const analysis::Constant* temp =
      FoldFPBinaryOp(FoldMax, inst->type_id(), {x, min_val}, context);
  if (temp == min_val) {
    // We can assume that |min_val| is less than |max_val|.  Therefore, if the
    // result of the max operation is |min_val|, we know the result of the min
    // operation, even if |max_val| is not a constant.
    return min_val;
  }
  return nullptr;
}

// Fold a clamp instruction when |x >= max_val|.
const analysis::Constant* FoldClamp3(
    IRContext* context, Instruction* inst,
    const std::vector<const analysis::Constant*>& constants) {
  assert(inst->opcode() == spv::Op::OpExtInst &&
         "Expecting an extended instruction.");
  assert(inst->GetSingleWordInOperand(0) ==
             context->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
         "Expecting a GLSLstd450 extended instruction.");

  const analysis::Constant* x = constants[1];
  const analysis::Constant* max_val = constants[3];

  if (x == nullptr || max_val == nullptr) {
    return nullptr;
  }

  const analysis::Constant* temp =
      FoldFPBinaryOp(FoldMin, inst->type_id(), {x, max_val}, context);
  if (temp == max_val) {
    // We can assume that |min_val| is less than |max_val|.  Therefore, if the
    // result of the max operation is |min_val|, we know the result of the min
    // operation, even if |max_val| is not a constant.
    return max_val;
  }
  return nullptr;
}

UnaryScalarFoldingRule FoldFTranscendentalUnary(double (*fp)(double)) {
  return
      [fp](const analysis::Type* result_type, const analysis::Constant* a,
           analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
        assert(result_type != nullptr && a != nullptr);
        const analysis::Float* float_type = a->type()->AsFloat();
        assert(float_type != nullptr);
        assert(float_type == result_type->AsFloat());
        if (float_type->width() == 32) {
          float fa = a->GetFloat();
          float res = static_cast<float>(fp(fa));
          utils::FloatProxy<float> result(res);
          std::vector<uint32_t> words = result.GetWords();
          return const_mgr->GetConstant(result_type, words);
        } else if (float_type->width() == 64) {
          double fa = a->GetDouble();
          double res = fp(fa);
          utils::FloatProxy<double> result(res);
          std::vector<uint32_t> words = result.GetWords();
          return const_mgr->GetConstant(result_type, words);
        }
        return nullptr;
      };
}

BinaryScalarFoldingRule FoldFTranscendentalBinary(double (*fp)(double,
                                                               double)) {
  return
      [fp](const analysis::Type* result_type, const analysis::Constant* a,
           const analysis::Constant* b,
           analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
        assert(result_type != nullptr && a != nullptr);
        const analysis::Float* float_type = a->type()->AsFloat();
        assert(float_type != nullptr);
        assert(float_type == result_type->AsFloat());
        assert(float_type == b->type()->AsFloat());
        if (float_type->width() == 32) {
          float fa = a->GetFloat();
          float fb = b->GetFloat();
          float res = static_cast<float>(fp(fa, fb));
          utils::FloatProxy<float> result(res);
          std::vector<uint32_t> words = result.GetWords();
          return const_mgr->GetConstant(result_type, words);
        } else if (float_type->width() == 64) {
          double fa = a->GetDouble();
          double fb = b->GetDouble();
          double res = fp(fa, fb);
          utils::FloatProxy<double> result(res);
          std::vector<uint32_t> words = result.GetWords();
          return const_mgr->GetConstant(result_type, words);
        }
        return nullptr;
      };
}

enum Sign { Signed, Unsigned };

// Returns a BinaryScalarFoldingRule that applies `op` to the scalars.
// The `signedness` is used to determine if the operands should be interpreted
// as signed or unsigned. If the operands are signed, the value will be sign
// extended before the value is passed to `op`. Otherwise the values will be
// zero extended.
template <Sign signedness>
BinaryScalarFoldingRule FoldBinaryIntegerOperation(uint64_t (*op)(uint64_t,
                                                                  uint64_t)) {
  return
      [op](const analysis::Type* result_type, const analysis::Constant* a,
           const analysis::Constant* b,
           analysis::ConstantManager* const_mgr) -> const analysis::Constant* {
        assert(result_type != nullptr && a != nullptr && b != nullptr);
        const analysis::Integer* integer_type = result_type->AsInteger();
        assert(integer_type != nullptr);
        assert(a->type()->kind() == analysis::Type::kInteger);
        assert(b->type()->kind() == analysis::Type::kInteger);
        assert(integer_type->width() == a->type()->AsInteger()->width());
        assert(integer_type->width() == b->type()->AsInteger()->width());

        // In SPIR-V, all operations support unsigned types, but the way they
        // are interpreted depends on the opcode. This is why we use the
        // template argument to determine how to interpret the operands.
        uint64_t ia = (signedness == Signed ? a->GetSignExtendedValue()
                                            : a->GetZeroExtendedValue());
        uint64_t ib = (signedness == Signed ? b->GetSignExtendedValue()
                                            : b->GetZeroExtendedValue());
        uint64_t result = op(ia, ib);

        const analysis::Constant* result_constant =
            const_mgr->GenerateIntegerConstant(integer_type, result);
        return result_constant;
      };
}

// A scalar folding rule that folds OpSConvert.
const analysis::Constant* FoldScalarSConvert(
    const analysis::Type* result_type, const analysis::Constant* a,
    analysis::ConstantManager* const_mgr) {
  assert(result_type != nullptr);
  assert(a != nullptr);
  assert(const_mgr != nullptr);
  const analysis::Integer* integer_type = result_type->AsInteger();
  assert(integer_type && "The result type of an SConvert");
  int64_t value = a->GetSignExtendedValue();
  return const_mgr->GenerateIntegerConstant(integer_type, value);
}

// A scalar folding rule that folds OpUConvert.
const analysis::Constant* FoldScalarUConvert(
    const analysis::Type* result_type, const analysis::Constant* a,
    analysis::ConstantManager* const_mgr) {
  assert(result_type != nullptr);
  assert(a != nullptr);
  assert(const_mgr != nullptr);
  const analysis::Integer* integer_type = result_type->AsInteger();
  assert(integer_type && "The result type of an UConvert");
  uint64_t value = a->GetZeroExtendedValue();

  // If the operand was an unsigned value with less than 32-bit, it would have
  // been sign extended earlier, and we need to clear those bits.
  auto* operand_type = a->type()->AsInteger();
  value = utils::ClearHighBits(value, 64 - operand_type->width());
  return const_mgr->GenerateIntegerConstant(integer_type, value);
}
}  // namespace

void ConstantFoldingRules::AddFoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[spv::Op::OpCompositeConstruct].push_back(FoldCompositeWithConstants());

  rules_[spv::Op::OpCompositeExtract].push_back(FoldExtractWithConstants());
  rules_[spv::Op::OpCompositeInsert].push_back(FoldInsertWithConstants());

  rules_[spv::Op::OpConvertFToS].push_back(FoldFToI());
  rules_[spv::Op::OpConvertFToU].push_back(FoldFToI());
  rules_[spv::Op::OpConvertSToF].push_back(FoldIToF());
  rules_[spv::Op::OpConvertUToF].push_back(FoldIToF());
  rules_[spv::Op::OpSConvert].push_back(FoldUnaryOp(FoldScalarSConvert));
  rules_[spv::Op::OpUConvert].push_back(FoldUnaryOp(FoldScalarUConvert));

  rules_[spv::Op::OpDot].push_back(FoldOpDotWithConstants());
  rules_[spv::Op::OpFAdd].push_back(FoldFAdd());
  rules_[spv::Op::OpFDiv].push_back(FoldFDiv());
  rules_[spv::Op::OpFMul].push_back(FoldFMul());
  rules_[spv::Op::OpFSub].push_back(FoldFSub());

  rules_[spv::Op::OpFOrdEqual].push_back(FoldFOrdEqual());

  rules_[spv::Op::OpFUnordEqual].push_back(FoldFUnordEqual());

  rules_[spv::Op::OpFOrdNotEqual].push_back(FoldFOrdNotEqual());

  rules_[spv::Op::OpFUnordNotEqual].push_back(FoldFUnordNotEqual());

  rules_[spv::Op::OpFOrdLessThan].push_back(FoldFOrdLessThan());
  rules_[spv::Op::OpFOrdLessThan].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFOrdLessThan));

  rules_[spv::Op::OpFUnordLessThan].push_back(FoldFUnordLessThan());
  rules_[spv::Op::OpFUnordLessThan].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFUnordLessThan));

  rules_[spv::Op::OpFOrdGreaterThan].push_back(FoldFOrdGreaterThan());
  rules_[spv::Op::OpFOrdGreaterThan].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFOrdGreaterThan));

  rules_[spv::Op::OpFUnordGreaterThan].push_back(FoldFUnordGreaterThan());
  rules_[spv::Op::OpFUnordGreaterThan].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFUnordGreaterThan));

  rules_[spv::Op::OpFOrdLessThanEqual].push_back(FoldFOrdLessThanEqual());
  rules_[spv::Op::OpFOrdLessThanEqual].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFOrdLessThanEqual));

  rules_[spv::Op::OpFUnordLessThanEqual].push_back(FoldFUnordLessThanEqual());
  rules_[spv::Op::OpFUnordLessThanEqual].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFUnordLessThanEqual));

  rules_[spv::Op::OpFOrdGreaterThanEqual].push_back(FoldFOrdGreaterThanEqual());
  rules_[spv::Op::OpFOrdGreaterThanEqual].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFOrdGreaterThanEqual));

  rules_[spv::Op::OpFUnordGreaterThanEqual].push_back(
      FoldFUnordGreaterThanEqual());
  rules_[spv::Op::OpFUnordGreaterThanEqual].push_back(
      FoldFClampFeedingCompare(spv::Op::OpFUnordGreaterThanEqual));

  rules_[spv::Op::OpVectorShuffle].push_back(FoldVectorShuffleWithConstants());
  rules_[spv::Op::OpVectorTimesScalar].push_back(FoldVectorTimesScalar());
  rules_[spv::Op::OpVectorTimesMatrix].push_back(FoldVectorTimesMatrix());
  rules_[spv::Op::OpMatrixTimesVector].push_back(FoldMatrixTimesVector());
  rules_[spv::Op::OpTranspose].push_back(FoldTranspose);

  rules_[spv::Op::OpFNegate].push_back(FoldFNegate());
  rules_[spv::Op::OpSNegate].push_back(FoldSNegate());
  rules_[spv::Op::OpQuantizeToF16].push_back(FoldQuantizeToF16());

  rules_[spv::Op::OpIAdd].push_back(
      FoldBinaryOp(FoldBinaryIntegerOperation<Unsigned>(
          [](uint64_t a, uint64_t b) { return a + b; })));
  rules_[spv::Op::OpISub].push_back(
      FoldBinaryOp(FoldBinaryIntegerOperation<Unsigned>(
          [](uint64_t a, uint64_t b) { return a - b; })));
  rules_[spv::Op::OpIMul].push_back(
      FoldBinaryOp(FoldBinaryIntegerOperation<Unsigned>(
          [](uint64_t a, uint64_t b) { return a * b; })));
  rules_[spv::Op::OpUDiv].push_back(
      FoldBinaryOp(FoldBinaryIntegerOperation<Unsigned>(
          [](uint64_t a, uint64_t b) { return (b != 0 ? a / b : 0); })));
  rules_[spv::Op::OpSDiv].push_back(FoldBinaryOp(
      FoldBinaryIntegerOperation<Signed>([](uint64_t a, uint64_t b) {
        return (b != 0 ? static_cast<uint64_t>(static_cast<int64_t>(a) /
                                               static_cast<int64_t>(b))
                       : 0);
      })));
  rules_[spv::Op::OpUMod].push_back(
      FoldBinaryOp(FoldBinaryIntegerOperation<Unsigned>(
          [](uint64_t a, uint64_t b) { return (b != 0 ? a % b : 0); })));

  rules_[spv::Op::OpSRem].push_back(FoldBinaryOp(
      FoldBinaryIntegerOperation<Signed>([](uint64_t a, uint64_t b) {
        return (b != 0 ? static_cast<uint64_t>(static_cast<int64_t>(a) %
                                               static_cast<int64_t>(b))
                       : 0);
      })));

  rules_[spv::Op::OpSMod].push_back(FoldBinaryOp(
      FoldBinaryIntegerOperation<Signed>([](uint64_t a, uint64_t b) {
        if (b == 0) return static_cast<uint64_t>(0ull);

        int64_t signed_a = static_cast<int64_t>(a);
        int64_t signed_b = static_cast<int64_t>(b);
        int64_t result = signed_a % signed_b;
        if ((signed_b < 0) != (result < 0)) result += signed_b;
        return static_cast<uint64_t>(result);
      })));

  // Add rules for GLSLstd450
  FeatureManager* feature_manager = context_->get_feature_mgr();
  uint32_t ext_inst_glslstd450_id =
      feature_manager->GetExtInstImportId_GLSLstd450();
  if (ext_inst_glslstd450_id != 0) {
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FMix}].push_back(FoldFMix());
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450SMin}].push_back(
        FoldFPBinaryOp(FoldMin));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450UMin}].push_back(
        FoldFPBinaryOp(FoldMin));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FMin}].push_back(
        FoldFPBinaryOp(FoldMin));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450SMax}].push_back(
        FoldFPBinaryOp(FoldMax));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450UMax}].push_back(
        FoldFPBinaryOp(FoldMax));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FMax}].push_back(
        FoldFPBinaryOp(FoldMax));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450UClamp}].push_back(
        FoldClamp1);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450UClamp}].push_back(
        FoldClamp2);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450UClamp}].push_back(
        FoldClamp3);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450SClamp}].push_back(
        FoldClamp1);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450SClamp}].push_back(
        FoldClamp2);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450SClamp}].push_back(
        FoldClamp3);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FClamp}].push_back(
        FoldClamp1);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FClamp}].push_back(
        FoldClamp2);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450FClamp}].push_back(
        FoldClamp3);
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Sin}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::sin)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Cos}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::cos)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Tan}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::tan)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Asin}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::asin)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Acos}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::acos)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Atan}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::atan)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Exp}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::exp)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Log}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::log)));

#ifdef __ANDROID__
    // Android NDK r15c targeting ABI 15 doesn't have full support for C++11
    // (no std::exp2/log2). ::exp2 is available from C99 but ::log2 isn't
    // available up until ABI 18 so we use a shim
    auto log2_shim = [](double v) -> double { return log(v) / log(2.0); };
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Exp2}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(::exp2)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Log2}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(log2_shim)));
#else
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Exp2}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::exp2)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Log2}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::log2)));
#endif

    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Sqrt}].push_back(
        FoldFPUnaryOp(FoldFTranscendentalUnary(std::sqrt)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Atan2}].push_back(
        FoldFPBinaryOp(FoldFTranscendentalBinary(std::atan2)));
    ext_rules_[{ext_inst_glslstd450_id, GLSLstd450Pow}].push_back(
        FoldFPBinaryOp(FoldFTranscendentalBinary(std::pow)));
  }
}
}  // namespace opt
}  // namespace spvtools
