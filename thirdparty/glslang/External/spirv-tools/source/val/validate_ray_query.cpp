// Copyright (c) 2022 The Khronos Group Inc.
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

// Validates ray query instructions from SPV_KHR_ray_query

#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateRayQueryPointer(ValidationState_t& _,
                                     const Instruction* inst,
                                     uint32_t ray_query_index) {
  const uint32_t ray_query_id = inst->GetOperandAs<uint32_t>(ray_query_index);
  auto variable = _.FindDef(ray_query_id);
  const auto var_opcode = variable->opcode();
  if (!variable || (var_opcode != spv::Op::OpVariable &&
                    var_opcode != spv::Op::OpFunctionParameter &&
                    var_opcode != spv::Op::OpAccessChain)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Ray Query must be a memory object declaration";
  }
  auto pointer = _.FindDef(variable->GetOperandAs<uint32_t>(0));
  if (!pointer || pointer->opcode() != spv::Op::OpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Ray Query must be a pointer";
  }
  auto type = _.FindDef(pointer->GetOperandAs<uint32_t>(2));
  if (!type || type->opcode() != spv::Op::OpTypeRayQueryKHR) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Ray Query must be a pointer to OpTypeRayQueryKHR";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateIntersectionId(ValidationState_t& _,
                                    const Instruction* inst,
                                    uint32_t intersection_index) {
  const uint32_t intersection_id =
      inst->GetOperandAs<uint32_t>(intersection_index);
  const uint32_t intersection_type = _.GetTypeId(intersection_id);
  const spv::Op intersection_opcode = _.GetIdOpcode(intersection_id);
  if (!_.IsIntScalarType(intersection_type) ||
      _.GetBitWidth(intersection_type) != 32 ||
      !spvOpcodeIsConstant(intersection_opcode)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "expected Intersection ID to be a constant 32-bit int scalar";
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t RayQueryPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case spv::Op::OpRayQueryInitializeKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 0)) return error;

      if (_.GetIdOpcode(_.GetOperandTypeId(inst, 1)) !=
          spv::Op::OpTypeAccelerationStructureKHR) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Acceleration Structure to be of type "
                  "OpTypeAccelerationStructureKHR";
      }

      const uint32_t ray_flags = _.GetOperandTypeId(inst, 2);
      if (!_.IsIntScalarType(ray_flags) || _.GetBitWidth(ray_flags) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Flags must be a 32-bit int scalar";
      }

      const uint32_t cull_mask = _.GetOperandTypeId(inst, 3);
      if (!_.IsIntScalarType(cull_mask) || _.GetBitWidth(cull_mask) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Cull Mask must be a 32-bit int scalar";
      }

      const uint32_t ray_origin = _.GetOperandTypeId(inst, 4);
      if (!_.IsFloatVectorType(ray_origin) || _.GetDimension(ray_origin) != 3 ||
          _.GetBitWidth(ray_origin) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Origin must be a 32-bit float 3-component vector";
      }

      const uint32_t ray_tmin = _.GetOperandTypeId(inst, 5);
      if (!_.IsFloatScalarType(ray_tmin) || _.GetBitWidth(ray_tmin) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray TMin must be a 32-bit float scalar";
      }

      const uint32_t ray_direction = _.GetOperandTypeId(inst, 6);
      if (!_.IsFloatVectorType(ray_direction) ||
          _.GetDimension(ray_direction) != 3 ||
          _.GetBitWidth(ray_direction) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Direction must be a 32-bit float 3-component vector";
      }

      const uint32_t ray_tmax = _.GetOperandTypeId(inst, 7);
      if (!_.IsFloatScalarType(ray_tmax) || _.GetBitWidth(ray_tmax) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray TMax must be a 32-bit float scalar";
      }
      break;
    }

    case spv::Op::OpRayQueryTerminateKHR:
    case spv::Op::OpRayQueryConfirmIntersectionKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 0)) return error;
      break;
    }

    case spv::Op::OpRayQueryGenerateIntersectionKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 0)) return error;

      const uint32_t hit_t_id = _.GetOperandTypeId(inst, 1);
      if (!_.IsFloatScalarType(hit_t_id) || _.GetBitWidth(hit_t_id) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Hit T must be a 32-bit float scalar";
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionFrontFaceKHR:
    case spv::Op::OpRayQueryProceedKHR:
    case spv::Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;

      if (!_.IsBoolScalarType(result_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be bool scalar type";
      }

      if (opcode == spv::Op::OpRayQueryGetIntersectionFrontFaceKHR) {
        if (auto error = ValidateIntersectionId(_, inst, 3)) return error;
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionTKHR:
    case spv::Op::OpRayQueryGetRayTMinKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;

      if (!_.IsFloatScalarType(result_type) ||
          _.GetBitWidth(result_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be 32-bit float scalar type";
      }

      if (opcode == spv::Op::OpRayQueryGetIntersectionTKHR) {
        if (auto error = ValidateIntersectionId(_, inst, 3)) return error;
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionTypeKHR:
    case spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR:
    case spv::Op::OpRayQueryGetIntersectionInstanceIdKHR:
    case spv::Op::
        OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
    case spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR:
    case spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR:
    case spv::Op::OpRayQueryGetRayFlagsKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;

      if (!_.IsIntScalarType(result_type) || _.GetBitWidth(result_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be 32-bit int scalar type";
      }

      if (opcode != spv::Op::OpRayQueryGetRayFlagsKHR) {
        if (auto error = ValidateIntersectionId(_, inst, 3)) return error;
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR:
    case spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR:
    case spv::Op::OpRayQueryGetWorldRayDirectionKHR:
    case spv::Op::OpRayQueryGetWorldRayOriginKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;

      if (!_.IsFloatVectorType(result_type) ||
          _.GetDimension(result_type) != 3 ||
          _.GetBitWidth(result_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be 32-bit float 3-component "
                  "vector type";
      }

      if (opcode == spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR ||
          opcode == spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR) {
        if (auto error = ValidateIntersectionId(_, inst, 3)) return error;
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionBarycentricsKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;
      if (auto error = ValidateIntersectionId(_, inst, 3)) return error;

      if (!_.IsFloatVectorType(result_type) ||
          _.GetDimension(result_type) != 2 ||
          _.GetBitWidth(result_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be 32-bit float 2-component "
                  "vector type";
      }

      break;
    }

    case spv::Op::OpRayQueryGetIntersectionObjectToWorldKHR:
    case spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR: {
      if (auto error = ValidateRayQueryPointer(_, inst, 2)) return error;
      if (auto error = ValidateIntersectionId(_, inst, 3)) return error;

      uint32_t num_rows = 0;
      uint32_t num_cols = 0;
      uint32_t col_type = 0;
      uint32_t component_type = 0;
      if (!_.GetMatrixTypeInfo(result_type, &num_rows, &num_cols, &col_type,
                               &component_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected matrix type as Result Type";
      }

      if (num_cols != 4) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type matrix to have a Column Count of 4";
      }

      if (!_.IsFloatScalarType(component_type) ||
          _.GetBitWidth(result_type) != 32 || num_rows != 3) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type matrix to have a Column Type of "
                  "3-component 32-bit float vectors";
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
