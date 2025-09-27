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

// Validates ray tracing instructions from SPV_KHR_ray_tracing

#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t RayTracingPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case spv::Op::OpTraceRayKHR: {
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              [](spv::ExecutionModel model, std::string* message) {
                if (model != spv::ExecutionModel::RayGenerationKHR &&
                    model != spv::ExecutionModel::ClosestHitKHR &&
                    model != spv::ExecutionModel::MissKHR) {
                  if (message) {
                    *message =
                        "OpTraceRayKHR requires RayGenerationKHR, "
                        "ClosestHitKHR and MissKHR execution models";
                  }
                  return false;
                }
                return true;
              });

      if (_.GetIdOpcode(_.GetOperandTypeId(inst, 0)) !=
          spv::Op::OpTypeAccelerationStructureKHR) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Acceleration Structure to be of type "
                  "OpTypeAccelerationStructureKHR";
      }

      const uint32_t ray_flags = _.GetOperandTypeId(inst, 1);
      if (!_.IsIntScalarType(ray_flags) || _.GetBitWidth(ray_flags) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Flags must be a 32-bit int scalar";
      }

      const uint32_t cull_mask = _.GetOperandTypeId(inst, 2);
      if (!_.IsIntScalarType(cull_mask) || _.GetBitWidth(cull_mask) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Cull Mask must be a 32-bit int scalar";
      }

      const uint32_t sbt_offset = _.GetOperandTypeId(inst, 3);
      if (!_.IsIntScalarType(sbt_offset) || _.GetBitWidth(sbt_offset) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "SBT Offset must be a 32-bit int scalar";
      }

      const uint32_t sbt_stride = _.GetOperandTypeId(inst, 4);
      if (!_.IsIntScalarType(sbt_stride) || _.GetBitWidth(sbt_stride) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "SBT Stride must be a 32-bit int scalar";
      }

      const uint32_t miss_index = _.GetOperandTypeId(inst, 5);
      if (!_.IsIntScalarType(miss_index) || _.GetBitWidth(miss_index) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Miss Index must be a 32-bit int scalar";
      }

      const uint32_t ray_origin = _.GetOperandTypeId(inst, 6);
      if (!_.IsFloatVectorType(ray_origin) || _.GetDimension(ray_origin) != 3 ||
          _.GetBitWidth(ray_origin) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Origin must be a 32-bit float 3-component vector";
      }

      const uint32_t ray_tmin = _.GetOperandTypeId(inst, 7);
      if (!_.IsFloatScalarType(ray_tmin) || _.GetBitWidth(ray_tmin) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray TMin must be a 32-bit float scalar";
      }

      const uint32_t ray_direction = _.GetOperandTypeId(inst, 8);
      if (!_.IsFloatVectorType(ray_direction) ||
          _.GetDimension(ray_direction) != 3 ||
          _.GetBitWidth(ray_direction) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray Direction must be a 32-bit float 3-component vector";
      }

      const uint32_t ray_tmax = _.GetOperandTypeId(inst, 9);
      if (!_.IsFloatScalarType(ray_tmax) || _.GetBitWidth(ray_tmax) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ray TMax must be a 32-bit float scalar";
      }

      const Instruction* payload = _.FindDef(inst->GetOperandAs<uint32_t>(10));
      if (payload->opcode() != spv::Op::OpVariable) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Payload must be the result of a OpVariable";
      } else if (payload->GetOperandAs<spv::StorageClass>(2) !=
                     spv::StorageClass::RayPayloadKHR &&
                 payload->GetOperandAs<spv::StorageClass>(2) !=
                     spv::StorageClass::IncomingRayPayloadKHR) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Payload must have storage class RayPayloadKHR or "
                  "IncomingRayPayloadKHR";
      }
      break;
    }

    case spv::Op::OpReportIntersectionKHR: {
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              [](spv::ExecutionModel model, std::string* message) {
                if (model != spv::ExecutionModel::IntersectionKHR) {
                  if (message) {
                    *message =
                        "OpReportIntersectionKHR requires IntersectionKHR "
                        "execution model";
                  }
                  return false;
                }
                return true;
              });

      if (!_.IsBoolScalarType(result_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "expected Result Type to be bool scalar type";
      }

      const uint32_t hit = _.GetOperandTypeId(inst, 2);
      if (!_.IsFloatScalarType(hit) || _.GetBitWidth(hit) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Hit must be a 32-bit int scalar";
      }

      const uint32_t hit_kind = _.GetOperandTypeId(inst, 3);
      if (!_.IsUnsignedIntScalarType(hit_kind) ||
          _.GetBitWidth(hit_kind) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Hit Kind must be a 32-bit unsigned int scalar";
      }
      break;
    }

    case spv::Op::OpExecuteCallableKHR: {
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation([](spv::ExecutionModel model,
                                                std::string* message) {
            if (model != spv::ExecutionModel::RayGenerationKHR &&
                model != spv::ExecutionModel::ClosestHitKHR &&
                model != spv::ExecutionModel::MissKHR &&
                model != spv::ExecutionModel::CallableKHR) {
              if (message) {
                *message =
                    "OpExecuteCallableKHR requires RayGenerationKHR, "
                    "ClosestHitKHR, MissKHR and CallableKHR execution models";
              }
              return false;
            }
            return true;
          });

      const uint32_t sbt_index = _.GetOperandTypeId(inst, 0);
      if (!_.IsUnsignedIntScalarType(sbt_index) ||
          _.GetBitWidth(sbt_index) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "SBT Index must be a 32-bit unsigned int scalar";
      }

      const auto callable_data = _.FindDef(inst->GetOperandAs<uint32_t>(1));
      if (callable_data->opcode() != spv::Op::OpVariable) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Callable Data must be the result of a OpVariable";
      } else if (callable_data->GetOperandAs<spv::StorageClass>(2) !=
                     spv::StorageClass::CallableDataKHR &&
                 callable_data->GetOperandAs<spv::StorageClass>(2) !=
                     spv::StorageClass::IncomingCallableDataKHR) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Callable Data must have storage class CallableDataKHR or "
                  "IncomingCallableDataKHR";
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
