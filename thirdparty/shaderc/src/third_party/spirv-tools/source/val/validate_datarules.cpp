// Copyright (c) 2016 Google Inc.
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

// Ensures Data Rules are followed according to the specifications.

#include "source/val/validate.h"

#include <cassert>
#include <sstream>
#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Validates that the number of components in the vector is valid.
// Vector types can only be parameterized as having 2, 3, or 4 components.
// If the Vector16 capability is added, 8 and 16 components are also allowed.
spv_result_t ValidateVecNumComponents(ValidationState_t& _,
                                      const Instruction* inst) {
  // Operand 2 specifies the number of components in the vector.
  auto num_components = inst->GetOperandAs<const uint32_t>(2);
  if (num_components == 2 || num_components == 3 || num_components == 4) {
    return SPV_SUCCESS;
  }
  if (num_components == 8 || num_components == 16) {
    if (_.HasCapability(SpvCapabilityVector16)) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Having " << num_components << " components for "
           << spvOpcodeString(inst->opcode())
           << " requires the Vector16 capability";
  }
  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << "Illegal number of components (" << num_components << ") for "
         << spvOpcodeString(inst->opcode());
}

// Validates that the number of bits specifed for a float type is valid.
// Scalar floating-point types can be parameterized only with 32-bits.
// Float16 capability allows using a 16-bit OpTypeFloat.
// Float16Buffer capability allows creation of a 16-bit OpTypeFloat.
// Float64 capability allows using a 64-bit OpTypeFloat.
spv_result_t ValidateFloatSize(ValidationState_t& _, const Instruction* inst) {
  // Operand 1 is the number of bits for this float
  auto num_bits = inst->GetOperandAs<const uint32_t>(1);
  if (num_bits == 32) {
    return SPV_SUCCESS;
  }
  if (num_bits == 16) {
    if (_.features().declare_float16_type) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Using a 16-bit floating point "
           << "type requires the Float16 or Float16Buffer capability,"
              " or an extension that explicitly enables 16-bit floating point.";
  }
  if (num_bits == 64) {
    if (_.HasCapability(SpvCapabilityFloat64)) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Using a 64-bit floating point "
           << "type requires the Float64 capability.";
  }
  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << "Invalid number of bits (" << num_bits << ") used for OpTypeFloat.";
}

// Validates that the number of bits specified for an Int type is valid.
// Scalar integer types can be parameterized only with 32-bits.
// Int8, Int16, and Int64 capabilities allow using 8-bit, 16-bit, and 64-bit
// integers, respectively.
spv_result_t ValidateIntSize(ValidationState_t& _, const Instruction* inst) {
  // Operand 1 is the number of bits for this integer.
  auto num_bits = inst->GetOperandAs<const uint32_t>(1);
  if (num_bits == 32) {
    return SPV_SUCCESS;
  }
  if (num_bits == 8) {
    if (_.features().declare_int8_type) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Using an 8-bit integer type requires the Int8 capability,"
              " or an extension that explicitly enables 8-bit integers.";
  }
  if (num_bits == 16) {
    if (_.features().declare_int16_type) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Using a 16-bit integer type requires the Int16 capability,"
              " or an extension that explicitly enables 16-bit integers.";
  }
  if (num_bits == 64) {
    if (_.HasCapability(SpvCapabilityInt64)) {
      return SPV_SUCCESS;
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Using a 64-bit integer type requires the Int64 capability.";
  }
  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << "Invalid number of bits (" << num_bits << ") used for OpTypeInt.";
}

// Validates that the matrix is parameterized with floating-point types.
spv_result_t ValidateMatrixColumnType(ValidationState_t& _,
                                      const Instruction* inst) {
  // Find the component type of matrix columns (must be vector).
  // Operand 1 is the <id> of the type specified for matrix columns.
  auto type_id = inst->GetOperandAs<const uint32_t>(1);
  auto col_type_instr = _.FindDef(type_id);
  if (col_type_instr->opcode() != SpvOpTypeVector) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Columns in a matrix must be of type vector.";
  }

  // Trace back once more to find out the type of components in the vector.
  // Operand 1 is the <id> of the type of data in the vector.
  auto comp_type_id =
      col_type_instr->words()[col_type_instr->operands()[1].offset];
  auto comp_type_instruction = _.FindDef(comp_type_id);
  if (comp_type_instruction->opcode() != SpvOpTypeFloat) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Matrix types can only be "
                                                   "parameterized with "
                                                   "floating-point types.";
  }
  return SPV_SUCCESS;
}

// Validates that the matrix has 2,3, or 4 columns.
spv_result_t ValidateMatrixNumCols(ValidationState_t& _,
                                   const Instruction* inst) {
  // Operand 2 is the number of columns in the matrix.
  auto num_cols = inst->GetOperandAs<const uint32_t>(2);
  if (num_cols != 2 && num_cols != 3 && num_cols != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Matrix types can only be "
                                                   "parameterized as having "
                                                   "only 2, 3, or 4 columns.";
  }
  return SPV_SUCCESS;
}

// Validates that OpSpecConstant specializes to either int or float type.
spv_result_t ValidateSpecConstNumerical(ValidationState_t& _,
                                        const Instruction* inst) {
  // Operand 0 is the <id> of the type that we're specializing to.
  auto type_id = inst->GetOperandAs<const uint32_t>(0);
  auto type_instruction = _.FindDef(type_id);
  auto type_opcode = type_instruction->opcode();
  if (type_opcode != SpvOpTypeInt && type_opcode != SpvOpTypeFloat) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Specialization constant "
                                                   "must be an integer or "
                                                   "floating-point number.";
  }
  return SPV_SUCCESS;
}

// Validates that OpSpecConstantTrue and OpSpecConstantFalse specialize to bool.
spv_result_t ValidateSpecConstBoolean(ValidationState_t& _,
                                      const Instruction* inst) {
  // Find out the type that we're specializing to.
  auto type_instruction = _.FindDef(inst->type_id());
  if (type_instruction->opcode() != SpvOpTypeBool) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Specialization constant must be a boolean type.";
  }
  return SPV_SUCCESS;
}

// Records the <id> of the forward pointer to be used for validation.
spv_result_t ValidateForwardPointer(ValidationState_t& _,
                                    const Instruction* inst) {
  // Record the <id> (which is operand 0) to ensure it's used properly.
  // OpTypeStruct can only include undefined pointers that are
  // previously declared as a ForwardPointer
  return (_.RegisterForwardPointer(inst->GetOperandAs<uint32_t>(0)));
}

// Validates that any undefined component of the struct is a forward pointer.
// It is valid to declare a forward pointer, and use its <id> as one of the
// components of a struct.
spv_result_t ValidateStruct(ValidationState_t& _, const Instruction* inst) {
  // Struct components are operands 1, 2, etc.
  for (unsigned i = 1; i < inst->operands().size(); i++) {
    auto type_id = inst->GetOperandAs<const uint32_t>(i);
    auto type_instruction = _.FindDef(type_id);
    if (type_instruction == nullptr && !_.IsForwardPointer(type_id)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Forward reference operands in an OpTypeStruct must first be "
                "declared using OpTypeForwardPointer.";
    }
  }
  return SPV_SUCCESS;
}

// Validates that any undefined type of the array is a forward pointer.
// It is valid to declare a forward pointer, and use its <id> as the element
// type of the array.
spv_result_t ValidateArray(ValidationState_t& _, const Instruction* inst) {
  auto element_type_id = inst->GetOperandAs<const uint32_t>(1);
  auto element_type_instruction = _.FindDef(element_type_id);
  if (element_type_instruction == nullptr &&
      !_.IsForwardPointer(element_type_id)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Forward reference operands in an OpTypeArray must first be "
              "declared using OpTypeForwardPointer.";
  }
  return SPV_SUCCESS;
}

}  // namespace

// Validates that Data Rules are followed according to the specifications.
// (Data Rules subsection of 2.16.1 Universal Validation Rules)
spv_result_t DataRulesPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpTypeVector: {
      if (auto error = ValidateVecNumComponents(_, inst)) return error;
      break;
    }
    case SpvOpTypeFloat: {
      if (auto error = ValidateFloatSize(_, inst)) return error;
      break;
    }
    case SpvOpTypeInt: {
      if (auto error = ValidateIntSize(_, inst)) return error;
      break;
    }
    case SpvOpTypeMatrix: {
      if (auto error = ValidateMatrixColumnType(_, inst)) return error;
      if (auto error = ValidateMatrixNumCols(_, inst)) return error;
      break;
    }
    // TODO(ehsan): Add OpSpecConstantComposite validation code.
    // TODO(ehsan): Add OpSpecConstantOp validation code (if any).
    case SpvOpSpecConstant: {
      if (auto error = ValidateSpecConstNumerical(_, inst)) return error;
      break;
    }
    case SpvOpSpecConstantFalse:
    case SpvOpSpecConstantTrue: {
      if (auto error = ValidateSpecConstBoolean(_, inst)) return error;
      break;
    }
    case SpvOpTypeForwardPointer: {
      if (auto error = ValidateForwardPointer(_, inst)) return error;
      break;
    }
    case SpvOpTypeStruct: {
      if (auto error = ValidateStruct(_, inst)) return error;
      break;
    }
    case SpvOpTypeArray: {
      if (auto error = ValidateArray(_, inst)) return error;
      break;
    }
    // TODO(ehsan): add more data rules validation here.
    default: { break; }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
