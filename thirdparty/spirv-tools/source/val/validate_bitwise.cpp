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

// Validates correctness of bitwise instructions.

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

// Validates when base and result need to be the same type
spv_result_t ValidateBaseType(ValidationState_t& _, const Instruction* inst,
                              const uint32_t base_type) {
  const spv::Op opcode = inst->opcode();

  if (!_.IsIntScalarType(base_type) && !_.IsIntVectorType(base_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(4781)
           << "Expected int scalar or vector type for Base operand: "
           << spvOpcodeString(opcode);
  }

  // Vulkan has a restriction to 32 bit for base
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (_.GetBitWidth(base_type) != 32) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4781)
             << "Expected 32-bit int type for Base operand: "
             << spvOpcodeString(opcode);
    }
  }

  // OpBitCount just needs same number of components
  if (base_type != inst->type_id() && opcode != spv::Op::OpBitCount) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Base Type to be equal to Result Type: "
           << spvOpcodeString(opcode);
  }

  return SPV_SUCCESS;
}

// Validates correctness of bitwise instructions.
spv_result_t BitwisePass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t result_dimension = _.GetDimension(result_type);
      const uint32_t base_type = _.GetOperandTypeId(inst, 2);
      const uint32_t shift_type = _.GetOperandTypeId(inst, 3);

      if (!base_type ||
          (!_.IsIntScalarType(base_type) && !_.IsIntVectorType(base_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Base to be int scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.GetDimension(base_type) != result_dimension)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Base to have the same dimension "
               << "as Result Type: " << spvOpcodeString(opcode);

      if (_.GetBitWidth(base_type) != _.GetBitWidth(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Base to have the same bit width "
               << "as Result Type: " << spvOpcodeString(opcode);

      if (!shift_type ||
          (!_.IsIntScalarType(shift_type) && !_.IsIntVectorType(shift_type)))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Shift to be int scalar or vector: "
               << spvOpcodeString(opcode);

      if (_.GetDimension(shift_type) != result_dimension)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Shift to have the same dimension "
               << "as Result Type: " << spvOpcodeString(opcode);
      break;
    }

    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t result_dimension = _.GetDimension(result_type);
      const uint32_t result_bit_width = _.GetBitWidth(result_type);

      for (size_t operand_index = 2; operand_index < inst->operands().size();
           ++operand_index) {
        const uint32_t type_id = _.GetOperandTypeId(inst, operand_index);
        if (!type_id ||
            (!_.IsIntScalarType(type_id) && !_.IsIntVectorType(type_id)))
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected int scalar or vector as operand: "
                 << spvOpcodeString(opcode) << " operand index "
                 << operand_index;

        if (_.GetDimension(type_id) != result_dimension)
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected operands to have the same dimension "
                 << "as Result Type: " << spvOpcodeString(opcode)
                 << " operand index " << operand_index;

        if (_.GetBitWidth(type_id) != result_bit_width)
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Expected operands to have the same bit width "
                 << "as Result Type: " << spvOpcodeString(opcode)
                 << " operand index " << operand_index;
      }
      break;
    }

    case spv::Op::OpBitFieldInsert: {
      const uint32_t base_type = _.GetOperandTypeId(inst, 2);
      const uint32_t insert_type = _.GetOperandTypeId(inst, 3);
      const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
      const uint32_t count_type = _.GetOperandTypeId(inst, 5);

      if (spv_result_t error = ValidateBaseType(_, inst, base_type)) {
        return error;
      }

      if (insert_type != result_type)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Insert Type to be equal to Result Type: "
               << spvOpcodeString(opcode);

      if (!offset_type || !_.IsIntScalarType(offset_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Offset Type to be int scalar: "
               << spvOpcodeString(opcode);

      if (!count_type || !_.IsIntScalarType(count_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Count Type to be int scalar: "
               << spvOpcodeString(opcode);
      break;
    }

    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract: {
      const uint32_t base_type = _.GetOperandTypeId(inst, 2);
      const uint32_t offset_type = _.GetOperandTypeId(inst, 3);
      const uint32_t count_type = _.GetOperandTypeId(inst, 4);

      if (spv_result_t error = ValidateBaseType(_, inst, base_type)) {
        return error;
      }

      if (!offset_type || !_.IsIntScalarType(offset_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Offset Type to be int scalar: "
               << spvOpcodeString(opcode);

      if (!count_type || !_.IsIntScalarType(count_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Count Type to be int scalar: "
               << spvOpcodeString(opcode);
      break;
    }

    case spv::Op::OpBitReverse: {
      const uint32_t base_type = _.GetOperandTypeId(inst, 2);

      if (spv_result_t error = ValidateBaseType(_, inst, base_type)) {
        return error;
      }

      break;
    }

    case spv::Op::OpBitCount: {
      if (!_.IsIntScalarType(result_type) && !_.IsIntVectorType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected int scalar or vector type as Result Type: "
               << spvOpcodeString(opcode);

      const uint32_t base_type = _.GetOperandTypeId(inst, 2);

      if (spv_result_t error = ValidateBaseType(_, inst, base_type)) {
        return error;
      }

      const uint32_t base_dimension = _.GetDimension(base_type);
      const uint32_t result_dimension = _.GetDimension(result_type);

      if (base_dimension != result_dimension)
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Base dimension to be equal to Result Type "
                  "dimension: "
               << spvOpcodeString(opcode);
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
