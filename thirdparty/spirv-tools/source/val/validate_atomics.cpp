// Copyright (c) 2017 Google Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights
// reserved.
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

// Validates correctness of atomic SPIR-V instructions.

#include "source/val/validate.h"

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validate_memory_semantics.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace {

bool IsStorageClassAllowedByUniversalRules(spv::StorageClass storage_class) {
  switch (storage_class) {
    case spv::StorageClass::Uniform:
    case spv::StorageClass::StorageBuffer:
    case spv::StorageClass::Workgroup:
    case spv::StorageClass::CrossWorkgroup:
    case spv::StorageClass::Generic:
    case spv::StorageClass::AtomicCounter:
    case spv::StorageClass::Image:
    case spv::StorageClass::Function:
    case spv::StorageClass::PhysicalStorageBuffer:
    case spv::StorageClass::TaskPayloadWorkgroupEXT:
      return true;
      break;
    default:
      return false;
  }
}

bool HasReturnType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicStore:
    case spv::Op::OpAtomicFlagClear:
      return false;
      break;
    default:
      return true;
  }
}

bool HasOnlyFloatReturnType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicFAddEXT:
    case spv::Op::OpAtomicFMinEXT:
    case spv::Op::OpAtomicFMaxEXT:
      return true;
      break;
    default:
      return false;
  }
}

bool HasOnlyIntReturnType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
      return true;
      break;
    default:
      return false;
  }
}

bool HasIntOrFloatReturnType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicExchange:
      return true;
      break;
    default:
      return false;
  }
}

bool HasOnlyBoolReturnType(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpAtomicFlagTestAndSet:
      return true;
      break;
    default:
      return false;
  }
}

}  // namespace

namespace spvtools {
namespace val {

// Validates correctness of atomic instructions.
spv_result_t AtomicsPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  switch (opcode) {
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicStore:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicFAddEXT:
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicFMinEXT:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicFMaxEXT:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
    case spv::Op::OpAtomicFlagTestAndSet:
    case spv::Op::OpAtomicFlagClear: {
      const uint32_t result_type = inst->type_id();

      // All current atomics only are scalar result
      // Validate return type first so can just check if pointer type is same
      // (if applicable)
      if (HasReturnType(opcode)) {
        if (HasOnlyFloatReturnType(opcode) &&
            !_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be float scalar type";
        } else if (HasOnlyIntReturnType(opcode) &&
                   !_.IsIntScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be integer scalar type";
        } else if (HasIntOrFloatReturnType(opcode) &&
                   !_.IsFloatScalarType(result_type) &&
                   !_.IsIntScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be integer or float scalar type";
        } else if (HasOnlyBoolReturnType(opcode) &&
                   !_.IsBoolScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be bool scalar type";
        }
      }

      uint32_t operand_index = HasReturnType(opcode) ? 2 : 0;
      const uint32_t pointer_type = _.GetOperandTypeId(inst, operand_index++);
      uint32_t data_type = 0;
      spv::StorageClass storage_class;
      if (!_.GetPointerTypeInfo(pointer_type, &data_type, &storage_class)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Pointer to be of type OpTypePointer";
      }

      // Can't use result_type because OpAtomicStore doesn't have a result
      if (_.IsIntScalarType(data_type) && _.GetBitWidth(data_type) == 64 &&
          !_.HasCapability(spv::Capability::Int64Atomics)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": 64-bit atomics require the Int64Atomics capability";
      }

      // Validate storage class against universal rules
      if (!IsStorageClassAllowedByUniversalRules(storage_class)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": storage class forbidden by universal validation rules.";
      }

      // Then Shader rules
      if (_.HasCapability(spv::Capability::Shader)) {
        // Vulkan environment rule
        if (spvIsVulkanEnv(_.context()->target_env)) {
          if ((storage_class != spv::StorageClass::Uniform) &&
              (storage_class != spv::StorageClass::StorageBuffer) &&
              (storage_class != spv::StorageClass::Workgroup) &&
              (storage_class != spv::StorageClass::Image) &&
              (storage_class != spv::StorageClass::PhysicalStorageBuffer) &&
              (storage_class != spv::StorageClass::TaskPayloadWorkgroupEXT)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << _.VkErrorID(4686) << spvOpcodeString(opcode)
                   << ": Vulkan spec only allows storage classes for atomic to "
                      "be: Uniform, Workgroup, Image, StorageBuffer, "
                      "PhysicalStorageBuffer or TaskPayloadWorkgroupEXT.";
          }
        } else if (storage_class == spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": Function storage class forbidden when the Shader "
                    "capability is declared.";
        }

        if (opcode == spv::Op::OpAtomicFAddEXT) {
          // result type being float checked already
          if ((_.GetBitWidth(result_type) == 16) &&
              (!_.HasCapability(spv::Capability::AtomicFloat16AddEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float add atomics require the AtomicFloat32AddEXT "
                      "capability";
          }
          if ((_.GetBitWidth(result_type) == 32) &&
              (!_.HasCapability(spv::Capability::AtomicFloat32AddEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float add atomics require the AtomicFloat32AddEXT "
                      "capability";
          }
          if ((_.GetBitWidth(result_type) == 64) &&
              (!_.HasCapability(spv::Capability::AtomicFloat64AddEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float add atomics require the AtomicFloat64AddEXT "
                      "capability";
          }
        } else if (opcode == spv::Op::OpAtomicFMinEXT ||
                   opcode == spv::Op::OpAtomicFMaxEXT) {
          if ((_.GetBitWidth(result_type) == 16) &&
              (!_.HasCapability(spv::Capability::AtomicFloat16MinMaxEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float min/max atomics require the "
                      "AtomicFloat16MinMaxEXT capability";
          }
          if ((_.GetBitWidth(result_type) == 32) &&
              (!_.HasCapability(spv::Capability::AtomicFloat32MinMaxEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float min/max atomics require the "
                      "AtomicFloat32MinMaxEXT capability";
          }
          if ((_.GetBitWidth(result_type) == 64) &&
              (!_.HasCapability(spv::Capability::AtomicFloat64MinMaxEXT))) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": float min/max atomics require the "
                      "AtomicFloat64MinMaxEXT capability";
          }
        }
      }

      // And finally OpenCL environment rules
      if (spvIsOpenCLEnv(_.context()->target_env)) {
        if ((storage_class != spv::StorageClass::Function) &&
            (storage_class != spv::StorageClass::Workgroup) &&
            (storage_class != spv::StorageClass::CrossWorkgroup) &&
            (storage_class != spv::StorageClass::Generic)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": storage class must be Function, Workgroup, "
                    "CrossWorkGroup or Generic in the OpenCL environment.";
        }

        if (_.context()->target_env == SPV_ENV_OPENCL_1_2) {
          if (storage_class == spv::StorageClass::Generic) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << "Storage class cannot be Generic in OpenCL 1.2 "
                      "environment";
          }
        }
      }

      // If result and pointer type are different, need to do special check here
      if (opcode == spv::Op::OpAtomicFlagTestAndSet ||
          opcode == spv::Op::OpAtomicFlagClear) {
        if (!_.IsIntScalarType(data_type) || _.GetBitWidth(data_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Pointer to point to a value of 32-bit integer "
                    "type";
        }
      } else if (opcode == spv::Op::OpAtomicStore) {
        if (!_.IsFloatScalarType(data_type) && !_.IsIntScalarType(data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Pointer to be a pointer to integer or float "
                 << "scalar type";
        }
      } else if (data_type != result_type) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Pointer to point to a value of type Result "
                  "Type";
      }

      auto memory_scope = inst->GetOperandAs<const uint32_t>(operand_index++);
      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      const auto equal_semantics_index = operand_index++;
      if (auto error = ValidateMemorySemantics(_, inst, equal_semantics_index,
                                               memory_scope))
        return error;

      if (opcode == spv::Op::OpAtomicCompareExchange ||
          opcode == spv::Op::OpAtomicCompareExchangeWeak) {
        const auto unequal_semantics_index = operand_index++;
        if (auto error = ValidateMemorySemantics(
                _, inst, unequal_semantics_index, memory_scope))
          return error;

        // Volatile bits must match for equal and unequal semantics. Previous
        // checks guarantee they are 32-bit constants, but we need to recheck
        // whether they are evaluatable constants.
        bool is_int32 = false;
        bool is_equal_const = false;
        bool is_unequal_const = false;
        uint32_t equal_value = 0;
        uint32_t unequal_value = 0;
        std::tie(is_int32, is_equal_const, equal_value) = _.EvalInt32IfConst(
            inst->GetOperandAs<uint32_t>(equal_semantics_index));
        std::tie(is_int32, is_unequal_const, unequal_value) =
            _.EvalInt32IfConst(
                inst->GetOperandAs<uint32_t>(unequal_semantics_index));
        if (is_equal_const && is_unequal_const &&
            ((equal_value & uint32_t(spv::MemorySemanticsMask::Volatile)) ^
             (unequal_value & uint32_t(spv::MemorySemanticsMask::Volatile)))) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "Volatile mask setting must match for Equal and Unequal "
                    "memory semantics";
        }
      }

      if (opcode == spv::Op::OpAtomicStore) {
        const uint32_t value_type = _.GetOperandTypeId(inst, 3);
        if (value_type != data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Value type and the type pointed to by "
                    "Pointer to be the same";
        }
      } else if (opcode != spv::Op::OpAtomicLoad &&
                 opcode != spv::Op::OpAtomicIIncrement &&
                 opcode != spv::Op::OpAtomicIDecrement &&
                 opcode != spv::Op::OpAtomicFlagTestAndSet &&
                 opcode != spv::Op::OpAtomicFlagClear) {
        const uint32_t value_type = _.GetOperandTypeId(inst, operand_index++);
        if (value_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Value to be of type Result Type";
        }
      }

      if (opcode == spv::Op::OpAtomicCompareExchange ||
          opcode == spv::Op::OpAtomicCompareExchangeWeak) {
        const uint32_t comparator_type =
            _.GetOperandTypeId(inst, operand_index++);
        if (comparator_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": expected Comparator to be of type Result Type";
        }
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
