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

// Validates correctness of conversion instructions.

#include "source/val/validate.h"

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

// Validates correctness of conversion instructions.
spv_result_t ConversionPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case SpvOpConvertFToU: {
      if (!_.IsUnsignedIntScalarType(result_type) &&
          !_.IsUnsignedIntVectorType(result_type) &&
          !_.IsUnsignedIntCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected unsigned int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type || (!_.IsFloatScalarType(input_type) &&
                          !_.IsFloatVectorType(input_type) &&
                          !_.IsFloatCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be float scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (!_.features().use_int8_type && (8 == _.GetBitWidth(result_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Invalid cast to 8-bit integer from a floating-point: "
               << spvOpcodeString(opcode);

      break;
    }

    case SpvOpConvertFToS: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type) &&
          !_.IsIntCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type || (!_.IsFloatScalarType(input_type) &&
                          !_.IsFloatVectorType(input_type) &&
                          !_.IsFloatCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be float scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (!_.features().use_int8_type && (8 == _.GetBitWidth(result_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Invalid cast to 8-bit integer from a floating-point: "
               << spvOpcodeString(opcode);

      break;
    }

    case SpvOpConvertSToF:
    case SpvOpConvertUToF: {
      if (!_.IsFloatScalarType(result_type) &&
          !_.IsFloatVectorType(result_type) &&
          !_.IsFloatCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected float scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type ||
          (!_.IsIntScalarType(input_type) && !_.IsIntVectorType(input_type) &&
           !_.IsIntCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be int scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (!_.features().use_int8_type && (8 == _.GetBitWidth(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Invalid cast to floating-point from an 8-bit integer: "
               << spvOpcodeString(opcode);

      break;
    }

    case SpvOpUConvert: {
      if (!_.IsUnsignedIntScalarType(result_type) &&
          !_.IsUnsignedIntVectorType(result_type) &&
          !_.IsUnsignedIntCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected unsigned int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type ||
          (!_.IsIntScalarType(input_type) && !_.IsIntVectorType(input_type) &&
           !_.IsIntCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be int scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (_.GetBitWidth(result_type) == _.GetBitWidth(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have different bit width from Result "
                  "Type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpSConvert: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type) &&
          !_.IsIntCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type ||
          (!_.IsIntScalarType(input_type) && !_.IsIntVectorType(input_type) &&
           !_.IsIntCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be int scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (_.GetBitWidth(result_type) == _.GetBitWidth(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have different bit width from Result "
                  "Type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpFConvert: {
      if (!_.IsFloatScalarType(result_type) &&
          !_.IsFloatVectorType(result_type) &&
          !_.IsFloatCooperativeMatrixType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected float scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type || (!_.IsFloatScalarType(input_type) &&
                          !_.IsFloatVectorType(input_type) &&
                          !_.IsFloatCooperativeMatrixType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be float scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.IsCooperativeMatrixType(result_type) ||
          _.IsCooperativeMatrixType(input_type)) {
        spv_result_t ret =
            _.CooperativeMatrixShapesMatch(inst, result_type, input_type);
        if (ret != SPV_SUCCESS) return ret;
      } else {
        if (_.GetDimension(result_type) != _.GetDimension(input_type))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same dimension as Result Type: "
                 << spvOpcodeString(opcode);
      }

      if (_.GetBitWidth(result_type) == _.GetBitWidth(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have different bit width from Result "
                  "Type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpQuantizeToF16: {
      if ((!_.IsFloatScalarType(result_type) &&
           !_.IsFloatVectorType(result_type)) ||
          _.GetBitWidth(result_type) != 32)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected 32-bit float scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (input_type != result_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input type to be equal to Result Type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpConvertPtrToU: {
      if (!_.IsUnsignedIntScalarType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected unsigned int scalar type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!_.IsPointerType(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer: " << spvOpcodeString(opcode);

      if (_.addressing_model() == SpvAddressingModelLogical)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Logical addressing not supported: "
               << spvOpcodeString(opcode);

      if (_.addressing_model() ==
          SpvAddressingModelPhysicalStorageBuffer64EXT) {
        uint32_t input_storage_class = 0;
        uint32_t input_data_type = 0;
        _.GetPointerTypeInfo(input_type, &input_data_type,
                             &input_storage_class);
        if (input_storage_class != SpvStorageClassPhysicalStorageBufferEXT)
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Pointer storage class must be PhysicalStorageBufferEXT: "
                 << spvOpcodeString(opcode);
      }
      break;
    }

    case SpvOpSatConvertSToU:
    case SpvOpSatConvertUToS: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type ||
          (!_.IsIntScalarType(input_type) && !_.IsIntVectorType(input_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector as input: "
               << spvOpcodeString(opcode);

      if (_.GetDimension(result_type) != _.GetDimension(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have the same dimension as Result Type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpConvertUToPtr: {
      if (!_.IsPointerType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be a pointer: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type || !_.IsIntScalarType(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar as input: " << spvOpcodeString(opcode);

      if (_.addressing_model() == SpvAddressingModelLogical)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Logical addressing not supported: "
               << spvOpcodeString(opcode);

      if (_.addressing_model() ==
          SpvAddressingModelPhysicalStorageBuffer64EXT) {
        uint32_t result_storage_class = 0;
        uint32_t result_data_type = 0;
        _.GetPointerTypeInfo(result_type, &result_data_type,
                             &result_storage_class);
        if (result_storage_class != SpvStorageClassPhysicalStorageBufferEXT)
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Pointer storage class must be PhysicalStorageBufferEXT: "
                 << spvOpcodeString(opcode);
      }
      break;
    }

    case SpvOpPtrCastToGeneric: {
      uint32_t result_storage_class = 0;
      uint32_t result_data_type = 0;
      if (!_.GetPointerTypeInfo(result_type, &result_data_type,
                                &result_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be a pointer: "
               << spvOpcodeString(opcode);

      if (result_storage_class != SpvStorageClassGeneric)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to have storage class Generic: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      uint32_t input_storage_class = 0;
      uint32_t input_data_type = 0;
      if (!_.GetPointerTypeInfo(input_type, &input_data_type,
                                &input_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer: " << spvOpcodeString(opcode);

      if (input_storage_class != SpvStorageClassWorkgroup &&
          input_storage_class != SpvStorageClassCrossWorkgroup &&
          input_storage_class != SpvStorageClassFunction)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have storage class Workgroup, "
               << "CrossWorkgroup or Function: " << spvOpcodeString(opcode);

      if (result_data_type != input_data_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input and Result Type to point to the same type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpGenericCastToPtr: {
      uint32_t result_storage_class = 0;
      uint32_t result_data_type = 0;
      if (!_.GetPointerTypeInfo(result_type, &result_data_type,
                                &result_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be a pointer: "
               << spvOpcodeString(opcode);

      if (result_storage_class != SpvStorageClassWorkgroup &&
          result_storage_class != SpvStorageClassCrossWorkgroup &&
          result_storage_class != SpvStorageClassFunction)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to have storage class Workgroup, "
               << "CrossWorkgroup or Function: " << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      uint32_t input_storage_class = 0;
      uint32_t input_data_type = 0;
      if (!_.GetPointerTypeInfo(input_type, &input_data_type,
                                &input_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer: " << spvOpcodeString(opcode);

      if (input_storage_class != SpvStorageClassGeneric)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have storage class Generic: "
               << spvOpcodeString(opcode);

      if (result_data_type != input_data_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input and Result Type to point to the same type: "
               << spvOpcodeString(opcode);
      break;
    }

    case SpvOpGenericCastToPtrExplicit: {
      uint32_t result_storage_class = 0;
      uint32_t result_data_type = 0;
      if (!_.GetPointerTypeInfo(result_type, &result_data_type,
                                &result_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be a pointer: "
               << spvOpcodeString(opcode);

      const uint32_t target_storage_class = inst->word(4);
      if (result_storage_class != target_storage_class)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be of target storage class: "
               << spvOpcodeString(opcode);

      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      uint32_t input_storage_class = 0;
      uint32_t input_data_type = 0;
      if (!_.GetPointerTypeInfo(input_type, &input_data_type,
                                &input_storage_class))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer: " << spvOpcodeString(opcode);

      if (input_storage_class != SpvStorageClassGeneric)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have storage class Generic: "
               << spvOpcodeString(opcode);

      if (result_data_type != input_data_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input and Result Type to point to the same type: "
               << spvOpcodeString(opcode);

      if (target_storage_class != SpvStorageClassWorkgroup &&
          target_storage_class != SpvStorageClassCrossWorkgroup &&
          target_storage_class != SpvStorageClassFunction)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected target storage class to be Workgroup, "
               << "CrossWorkgroup or Function: " << spvOpcodeString(opcode);
      break;
    }

    case SpvOpBitcast: {
      const uint32_t input_type = _.GetOperandTypeId(inst, 2);
      if (!input_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to have a type: " << spvOpcodeString(opcode);

      const bool result_is_pointer = _.IsPointerType(result_type);
      const bool result_is_int_scalar = _.IsIntScalarType(result_type);
      const bool input_is_pointer = _.IsPointerType(input_type);
      const bool input_is_int_scalar = _.IsIntScalarType(input_type);

      if (!result_is_pointer && !result_is_int_scalar &&
          !_.IsIntVectorType(result_type) &&
          !_.IsFloatScalarType(result_type) &&
          !_.IsFloatVectorType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Result Type to be a pointer or int or float vector "
               << "or scalar type: " << spvOpcodeString(opcode);

      if (!input_is_pointer && !input_is_int_scalar &&
          !_.IsIntVectorType(input_type) && !_.IsFloatScalarType(input_type) &&
          !_.IsFloatVectorType(input_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer or int or float vector "
               << "or scalar: " << spvOpcodeString(opcode);

      if (result_is_pointer && !input_is_pointer && !input_is_int_scalar)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected input to be a pointer or int scalar if Result Type "
               << "is pointer: " << spvOpcodeString(opcode);

      if (input_is_pointer && !result_is_pointer && !result_is_int_scalar)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Pointer can only be converted to another pointer or int "
               << "scalar: " << spvOpcodeString(opcode);

      if (!result_is_pointer && !input_is_pointer) {
        const uint32_t result_size =
            _.GetBitWidth(result_type) * _.GetDimension(result_type);
        const uint32_t input_size =
            _.GetBitWidth(input_type) * _.GetDimension(input_type);
        if (result_size != input_size)
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected input to have the same total bit width as "
                 << "Result Type: " << spvOpcodeString(opcode);
      }
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
