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

// Validates correctness of composite SPIR-V instructions.

#include "source/val/validate.h"

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns the type of the value accessed by OpCompositeExtract or
// OpCompositeInsert instruction. The function traverses the hierarchy of
// nested data structures (structs, arrays, vectors, matrices) as directed by
// the sequence of indices in the instruction. May return error if traversal
// fails (encountered non-composite, out of bounds, no indices, nesting too
// deep).
spv_result_t GetExtractInsertValueType(ValidationState_t& _,
                                       const Instruction* inst,
                                       uint32_t* member_type) {
  const spv::Op opcode = inst->opcode();
  assert(opcode == spv::Op::OpCompositeExtract ||
         opcode == spv::Op::OpCompositeInsert);
  uint32_t word_index = opcode == spv::Op::OpCompositeExtract ? 4 : 5;
  const uint32_t num_words = static_cast<uint32_t>(inst->words().size());
  const uint32_t composite_id_index = word_index - 1;
  const uint32_t num_indices = num_words - word_index;
  const uint32_t kCompositeExtractInsertMaxNumIndices = 255;

  if (num_indices == 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected at least one index to Op"
           << spvOpcodeString(inst->opcode()) << ", zero found";

  } else if (num_indices > kCompositeExtractInsertMaxNumIndices) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The number of indexes in Op" << spvOpcodeString(opcode)
           << " may not exceed " << kCompositeExtractInsertMaxNumIndices
           << ". Found " << num_indices << " indexes.";
  }

  *member_type = _.GetTypeId(inst->word(composite_id_index));
  if (*member_type == 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Composite to be an object of composite type";
  }

  for (; word_index < num_words; ++word_index) {
    const uint32_t component_index = inst->word(word_index);
    const Instruction* const type_inst = _.FindDef(*member_type);
    assert(type_inst);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeVector: {
        *member_type = type_inst->word(2);
        const uint32_t vector_size = type_inst->word(3);
        if (component_index >= vector_size) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Vector access is out of bounds, vector size is "
                 << vector_size << ", but access index is " << component_index;
        }
        break;
      }
      case spv::Op::OpTypeMatrix: {
        *member_type = type_inst->word(2);
        const uint32_t num_cols = type_inst->word(3);
        if (component_index >= num_cols) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Matrix access is out of bounds, matrix has " << num_cols
                 << " columns, but access index is " << component_index;
        }
        break;
      }
      case spv::Op::OpTypeArray: {
        uint64_t array_size = 0;
        auto size = _.FindDef(type_inst->word(3));
        *member_type = type_inst->word(2);
        if (spvOpcodeIsSpecConstant(size->opcode())) {
          // Cannot verify against the size of this array.
          break;
        }

        if (!_.GetConstantValUint64(type_inst->word(3), &array_size)) {
          assert(0 && "Array type definition is corrupt");
        }
        if (component_index >= array_size) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Array access is out of bounds, array size is "
                 << array_size << ", but access index is " << component_index;
        }
        break;
      }
      case spv::Op::OpTypeRuntimeArray: {
        *member_type = type_inst->word(2);
        // Array size is unknown.
        break;
      }
      case spv::Op::OpTypeStruct: {
        const size_t num_struct_members = type_inst->words().size() - 2;
        if (component_index >= num_struct_members) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Index is out of bounds, can not find index "
                 << component_index << " in the structure <id> '"
                 << type_inst->id() << "'. This structure has "
                 << num_struct_members << " members. Largest valid index is "
                 << num_struct_members - 1 << ".";
        }
        *member_type = type_inst->word(component_index + 2);
        break;
      }
      case spv::Op::OpTypeCooperativeMatrixNV: {
        *member_type = type_inst->word(2);
        break;
      }
      default:
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Reached non-composite type while indexes still remain to "
                  "be traversed.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateVectorExtractDynamic(ValidationState_t& _,
                                          const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  const spv::Op result_opcode = _.GetIdOpcode(result_type);
  if (!spvOpcodeIsScalarType(result_opcode)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be a scalar type";
  }

  const uint32_t vector_type = _.GetOperandTypeId(inst, 2);
  const spv::Op vector_opcode = _.GetIdOpcode(vector_type);
  if (vector_opcode != spv::Op::OpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Vector type to be OpTypeVector";
  }

  if (_.GetComponentType(vector_type) != result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Vector component type to be equal to Result Type";
  }

  const auto index = _.FindDef(inst->GetOperandAs<uint32_t>(3));
  if (!index || index->type_id() == 0 || !_.IsIntScalarType(index->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Index to be int scalar";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot extract from a vector of 8- or 16-bit types";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateVectorInsertDyanmic(ValidationState_t& _,
                                         const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  const spv::Op result_opcode = _.GetIdOpcode(result_type);
  if (result_opcode != spv::Op::OpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypeVector";
  }

  const uint32_t vector_type = _.GetOperandTypeId(inst, 2);
  if (vector_type != result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Vector type to be equal to Result Type";
  }

  const uint32_t component_type = _.GetOperandTypeId(inst, 3);
  if (_.GetComponentType(result_type) != component_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Component type to be equal to Result Type "
           << "component type";
  }

  const uint32_t index_type = _.GetOperandTypeId(inst, 4);
  if (!_.IsIntScalarType(index_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Index to be int scalar";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot insert into a vector of 8- or 16-bit types";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateCompositeConstruct(ValidationState_t& _,
                                        const Instruction* inst) {
  const uint32_t num_operands = static_cast<uint32_t>(inst->operands().size());
  const uint32_t result_type = inst->type_id();
  const spv::Op result_opcode = _.GetIdOpcode(result_type);
  switch (result_opcode) {
    case spv::Op::OpTypeVector: {
      const uint32_t num_result_components = _.GetDimension(result_type);
      const uint32_t result_component_type = _.GetComponentType(result_type);
      uint32_t given_component_count = 0;

      if (num_operands <= 3) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected number of constituents to be at least 2";
      }

      for (uint32_t operand_index = 2; operand_index < num_operands;
           ++operand_index) {
        const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
        if (operand_type == result_component_type) {
          ++given_component_count;
        } else {
          if (_.GetIdOpcode(operand_type) != spv::Op::OpTypeVector ||
              _.GetComponentType(operand_type) != result_component_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << "Expected Constituents to be scalars or vectors of"
                   << " the same type as Result Type components";
          }

          given_component_count += _.GetDimension(operand_type);
        }
      }

      if (num_result_components != given_component_count) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected total number of given components to be equal "
               << "to the size of Result Type vector";
      }

      break;
    }
    case spv::Op::OpTypeMatrix: {
      uint32_t result_num_rows = 0;
      uint32_t result_num_cols = 0;
      uint32_t result_col_type = 0;
      uint32_t result_component_type = 0;
      if (!_.GetMatrixTypeInfo(result_type, &result_num_rows, &result_num_cols,
                               &result_col_type, &result_component_type)) {
        assert(0);
      }

      if (result_num_cols + 2 != num_operands) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected total number of Constituents to be equal "
               << "to the number of columns of Result Type matrix";
      }

      for (uint32_t operand_index = 2; operand_index < num_operands;
           ++operand_index) {
        const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
        if (operand_type != result_col_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected Constituent type to be equal to the column "
                 << "type Result Type matrix";
        }
      }

      break;
    }
    case spv::Op::OpTypeArray: {
      const Instruction* const array_inst = _.FindDef(result_type);
      assert(array_inst);
      assert(array_inst->opcode() == spv::Op::OpTypeArray);

      auto size = _.FindDef(array_inst->word(3));
      if (spvOpcodeIsSpecConstant(size->opcode())) {
        // Cannot verify against the size of this array.
        break;
      }

      uint64_t array_size = 0;
      if (!_.GetConstantValUint64(array_inst->word(3), &array_size)) {
        assert(0 && "Array type definition is corrupt");
      }

      if (array_size + 2 != num_operands) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected total number of Constituents to be equal "
               << "to the number of elements of Result Type array";
      }

      const uint32_t result_component_type = array_inst->word(2);
      for (uint32_t operand_index = 2; operand_index < num_operands;
           ++operand_index) {
        const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
        if (operand_type != result_component_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected Constituent type to be equal to the column "
                 << "type Result Type array";
        }
      }

      break;
    }
    case spv::Op::OpTypeStruct: {
      const Instruction* const struct_inst = _.FindDef(result_type);
      assert(struct_inst);
      assert(struct_inst->opcode() == spv::Op::OpTypeStruct);

      if (struct_inst->operands().size() + 1 != num_operands) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected total number of Constituents to be equal "
               << "to the number of members of Result Type struct";
      }

      for (uint32_t operand_index = 2; operand_index < num_operands;
           ++operand_index) {
        const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
        const uint32_t member_type = struct_inst->word(operand_index);
        if (operand_type != member_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected Constituent type to be equal to the "
                 << "corresponding member type of Result Type struct";
        }
      }

      break;
    }
    case spv::Op::OpTypeCooperativeMatrixNV: {
      const auto result_type_inst = _.FindDef(result_type);
      assert(result_type_inst);
      const auto component_type_id =
          result_type_inst->GetOperandAs<uint32_t>(1);

      if (3 != num_operands) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected single constituent";
      }

      const uint32_t operand_type_id = _.GetOperandTypeId(inst, 2);

      if (operand_type_id != component_type_id) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Constituent type to be equal to the component type";
      }

      break;
    }
    default: {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Result Type to be a composite type";
    }
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot create a composite containing 8- or 16-bit types";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateCompositeExtract(ValidationState_t& _,
                                      const Instruction* inst) {
  uint32_t member_type = 0;
  if (spv_result_t error = GetExtractInsertValueType(_, inst, &member_type)) {
    return error;
  }

  const uint32_t result_type = inst->type_id();
  if (result_type != member_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result type (Op" << spvOpcodeString(_.GetIdOpcode(result_type))
           << ") does not match the type that results from indexing into "
              "the composite (Op"
           << spvOpcodeString(_.GetIdOpcode(member_type)) << ").";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot extract from a composite of 8- or 16-bit types";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateCompositeInsert(ValidationState_t& _,
                                     const Instruction* inst) {
  const uint32_t object_type = _.GetOperandTypeId(inst, 2);
  const uint32_t composite_type = _.GetOperandTypeId(inst, 3);
  const uint32_t result_type = inst->type_id();
  if (result_type != composite_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The Result Type must be the same as Composite type in Op"
           << spvOpcodeString(inst->opcode()) << " yielding Result Id "
           << result_type << ".";
  }

  uint32_t member_type = 0;
  if (spv_result_t error = GetExtractInsertValueType(_, inst, &member_type)) {
    return error;
  }

  if (object_type != member_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The Object type (Op"
           << spvOpcodeString(_.GetIdOpcode(object_type))
           << ") does not match the type that results from indexing into the "
              "Composite (Op"
           << spvOpcodeString(_.GetIdOpcode(member_type)) << ").";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot insert into a composite of 8- or 16-bit types";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateCopyObject(ValidationState_t& _, const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  const uint32_t operand_type = _.GetOperandTypeId(inst, 2);
  if (operand_type != result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type and Operand type to be the same";
  }
  if (_.IsVoidType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "OpCopyObject cannot have void result type";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTranspose(ValidationState_t& _, const Instruction* inst) {
  uint32_t result_num_rows = 0;
  uint32_t result_num_cols = 0;
  uint32_t result_col_type = 0;
  uint32_t result_component_type = 0;
  const uint32_t result_type = inst->type_id();
  if (!_.GetMatrixTypeInfo(result_type, &result_num_rows, &result_num_cols,
                           &result_col_type, &result_component_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be a matrix type";
  }

  const uint32_t matrix_type = _.GetOperandTypeId(inst, 2);
  uint32_t matrix_num_rows = 0;
  uint32_t matrix_num_cols = 0;
  uint32_t matrix_col_type = 0;
  uint32_t matrix_component_type = 0;
  if (!_.GetMatrixTypeInfo(matrix_type, &matrix_num_rows, &matrix_num_cols,
                           &matrix_col_type, &matrix_component_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Matrix to be of type OpTypeMatrix";
  }

  if (result_component_type != matrix_component_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected component types of Matrix and Result Type to be "
           << "identical";
  }

  if (result_num_rows != matrix_num_cols ||
      result_num_cols != matrix_num_rows) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected number of columns and the column size of Matrix "
           << "to be the reverse of those of Result Type";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot transpose matrices of 16-bit floats";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateVectorShuffle(ValidationState_t& _,
                                   const Instruction* inst) {
  auto resultType = _.FindDef(inst->type_id());
  if (!resultType || resultType->opcode() != spv::Op::OpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Result Type of OpVectorShuffle must be"
           << " OpTypeVector. Found Op"
           << spvOpcodeString(static_cast<spv::Op>(resultType->opcode()))
           << ".";
  }

  // The number of components in Result Type must be the same as the number of
  // Component operands.
  auto componentCount = inst->operands().size() - 4;
  auto resultVectorDimension = resultType->GetOperandAs<uint32_t>(2);
  if (componentCount != resultVectorDimension) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpVectorShuffle component literals count does not match "
              "Result Type <id> "
           << _.getIdName(resultType->id()) << "s vector component count.";
  }

  // Vector 1 and Vector 2 must both have vector types, with the same Component
  // Type as Result Type.
  auto vector1Object = _.FindDef(inst->GetOperandAs<uint32_t>(2));
  auto vector1Type = _.FindDef(vector1Object->type_id());
  auto vector2Object = _.FindDef(inst->GetOperandAs<uint32_t>(3));
  auto vector2Type = _.FindDef(vector2Object->type_id());
  if (!vector1Type || vector1Type->opcode() != spv::Op::OpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The type of Vector 1 must be OpTypeVector.";
  }
  if (!vector2Type || vector2Type->opcode() != spv::Op::OpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The type of Vector 2 must be OpTypeVector.";
  }

  auto resultComponentType = resultType->GetOperandAs<uint32_t>(1);
  if (vector1Type->GetOperandAs<uint32_t>(1) != resultComponentType) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Component Type of Vector 1 must be the same as ResultType.";
  }
  if (vector2Type->GetOperandAs<uint32_t>(1) != resultComponentType) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Component Type of Vector 2 must be the same as ResultType.";
  }

  // All Component literals must either be FFFFFFFF or in [0, N - 1].
  auto vector1ComponentCount = vector1Type->GetOperandAs<uint32_t>(2);
  auto vector2ComponentCount = vector2Type->GetOperandAs<uint32_t>(2);
  auto N = vector1ComponentCount + vector2ComponentCount;
  auto firstLiteralIndex = 4;
  for (size_t i = firstLiteralIndex; i < inst->operands().size(); ++i) {
    auto literal = inst->GetOperandAs<uint32_t>(i);
    if (literal != 0xFFFFFFFF && literal >= N) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Component index " << literal << " is out of bounds for "
             << "combined (Vector1 + Vector2) size of " << N << ".";
    }
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot shuffle a vector of 8- or 16-bit types";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateCopyLogical(ValidationState_t& _,
                                 const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  const auto source = _.FindDef(inst->GetOperandAs<uint32_t>(2u));
  const auto source_type = _.FindDef(source->type_id());
  if (!source_type || !result_type || source_type == result_type) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Result Type must not equal the Operand type";
  }

  if (!_.LogicallyMatch(source_type, result_type, false)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Result Type does not logically match the Operand type";
  }

  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Cannot copy composites of 8- or 16-bit types";
  }

  return SPV_SUCCESS;
}

}  // anonymous namespace

// Validates correctness of composite instructions.
spv_result_t CompositesPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpVectorExtractDynamic:
      return ValidateVectorExtractDynamic(_, inst);
    case spv::Op::OpVectorInsertDynamic:
      return ValidateVectorInsertDyanmic(_, inst);
    case spv::Op::OpVectorShuffle:
      return ValidateVectorShuffle(_, inst);
    case spv::Op::OpCompositeConstruct:
      return ValidateCompositeConstruct(_, inst);
    case spv::Op::OpCompositeExtract:
      return ValidateCompositeExtract(_, inst);
    case spv::Op::OpCompositeInsert:
      return ValidateCompositeInsert(_, inst);
    case spv::Op::OpCopyObject:
      return ValidateCopyObject(_, inst);
    case spv::Op::OpTranspose:
      return ValidateTranspose(_, inst);
    case spv::Op::OpCopyLogical:
      return ValidateCopyLogical(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
