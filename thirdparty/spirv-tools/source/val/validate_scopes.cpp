// Copyright (c) 2018 Google LLC.
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

#include "source/val/validate_scopes.h"

#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

bool IsValidScope(uint32_t scope) {
  // Deliberately avoid a default case so we have to update the list when the
  // scopes list changes.
  switch (static_cast<spv::Scope>(scope)) {
    case spv::Scope::CrossDevice:
    case spv::Scope::Device:
    case spv::Scope::Workgroup:
    case spv::Scope::Subgroup:
    case spv::Scope::Invocation:
    case spv::Scope::QueueFamilyKHR:
    case spv::Scope::ShaderCallKHR:
      return true;
    case spv::Scope::Max:
      break;
  }
  return false;
}

spv_result_t ValidateScope(ValidationState_t& _, const Instruction* inst,
                           uint32_t scope) {
  spv::Op opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode) << ": expected scope to be a 32-bit int";
  }

  if (!is_const_int32) {
    if (_.HasCapability(spv::Capability::Shader) &&
        !_.HasCapability(spv::Capability::CooperativeMatrixNV)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Scope ids must be OpConstant when Shader capability is "
             << "present";
    }
    if (_.HasCapability(spv::Capability::Shader) &&
        _.HasCapability(spv::Capability::CooperativeMatrixNV) &&
        !spvOpcodeIsConstant(_.GetIdOpcode(scope))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Scope ids must be constant or specialization constant when "
             << "CooperativeMatrixNV capability is present";
    }
  }

  if (is_const_int32 && !IsValidScope(value)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Invalid scope value:\n " << _.Disassemble(*_.FindDef(scope));
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExecutionScope(ValidationState_t& _,
                                    const Instruction* inst, uint32_t scope) {
  spv::Op opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t tmp_value = 0;
  std::tie(is_int32, is_const_int32, tmp_value) = _.EvalInt32IfConst(scope);

  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  spv::Scope value = spv::Scope(tmp_value);

  // Vulkan specific rules
  if (spvIsVulkanEnv(_.context()->target_env)) {
    // Vulkan 1.1 specific rules
    if (_.context()->target_env != SPV_ENV_VULKAN_1_0) {
      // Scope for Non Uniform Group Operations must be limited to Subgroup
      if ((spvOpcodeIsNonUniformGroupOperation(opcode) &&
           (opcode != spv::Op::OpGroupNonUniformQuadAllKHR) &&
           (opcode != spv::Op::OpGroupNonUniformQuadAnyKHR)) &&
          (value != spv::Scope::Subgroup)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(4642) << spvOpcodeString(opcode)
               << ": in Vulkan environment Execution scope is limited to "
               << "Subgroup";
      }
    }

    // OpControlBarrier must only use Subgroup execution scope for a subset of
    // execution models.
    if (opcode == spv::Op::OpControlBarrier && value != spv::Scope::Subgroup) {
      std::string errorVUID = _.VkErrorID(4682);
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation([errorVUID](
                                                 spv::ExecutionModel model,
                                                 std::string* message) {
            if (model == spv::ExecutionModel::Fragment ||
                model == spv::ExecutionModel::Vertex ||
                model == spv::ExecutionModel::Geometry ||
                model == spv::ExecutionModel::TessellationEvaluation ||
                model == spv::ExecutionModel::RayGenerationKHR ||
                model == spv::ExecutionModel::IntersectionKHR ||
                model == spv::ExecutionModel::AnyHitKHR ||
                model == spv::ExecutionModel::ClosestHitKHR ||
                model == spv::ExecutionModel::MissKHR) {
              if (message) {
                *message =
                    errorVUID +
                    "in Vulkan environment, OpControlBarrier execution scope "
                    "must be Subgroup for Fragment, Vertex, Geometry, "
                    "TessellationEvaluation, RayGeneration, Intersection, "
                    "AnyHit, ClosestHit, and Miss execution models";
              }
              return false;
            }
            return true;
          });
    }

    // Only subset of execution models support Workgroup.
    if (value == spv::Scope::Workgroup) {
      std::string errorVUID = _.VkErrorID(4637);
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              [errorVUID](spv::ExecutionModel model, std::string* message) {
                if (model != spv::ExecutionModel::TaskNV &&
                    model != spv::ExecutionModel::MeshNV &&
                    model != spv::ExecutionModel::TaskEXT &&
                    model != spv::ExecutionModel::MeshEXT &&
                    model != spv::ExecutionModel::TessellationControl &&
                    model != spv::ExecutionModel::GLCompute) {
                  if (message) {
                    *message =
                        errorVUID +
                        "in Vulkan environment, Workgroup execution scope is "
                        "only for TaskNV, MeshNV, TaskEXT, MeshEXT, "
                        "TessellationControl, and GLCompute execution models";
                  }
                  return false;
                }
                return true;
              });
    }

    // Vulkan generic rules
    // Scope for execution must be limited to Workgroup or Subgroup
    if (value != spv::Scope::Workgroup && value != spv::Scope::Subgroup) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4636) << spvOpcodeString(opcode)
             << ": in Vulkan environment Execution Scope is limited to "
             << "Workgroup and Subgroup";
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL environments.

  // General SPIRV rules
  // Scope for execution must be limited to Workgroup or Subgroup for
  // non-uniform operations
  if (spvOpcodeIsNonUniformGroupOperation(opcode) &&
      opcode != spv::Op::OpGroupNonUniformQuadAllKHR &&
      opcode != spv::Op::OpGroupNonUniformQuadAnyKHR &&
      value != spv::Scope::Subgroup && value != spv::Scope::Workgroup) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Execution scope is limited to Subgroup or Workgroup";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateMemoryScope(ValidationState_t& _, const Instruction* inst,
                                 uint32_t scope) {
  const spv::Op opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t tmp_value = 0;
  std::tie(is_int32, is_const_int32, tmp_value) = _.EvalInt32IfConst(scope);

  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  spv::Scope value = spv::Scope(tmp_value);

  if (value == spv::Scope::QueueFamilyKHR) {
    if (_.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
      return SPV_SUCCESS;
    } else {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Scope QueueFamilyKHR requires capability "
             << "VulkanMemoryModelKHR";
    }
  }

  if (value == spv::Scope::Device &&
      _.HasCapability(spv::Capability::VulkanMemoryModelKHR) &&
      !_.HasCapability(spv::Capability::VulkanMemoryModelDeviceScopeKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Use of device scope with VulkanKHR memory model requires the "
           << "VulkanMemoryModelDeviceScopeKHR capability";
  }

  // Vulkan Specific rules
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (value != spv::Scope::Device && value != spv::Scope::Workgroup &&
        value != spv::Scope::Subgroup && value != spv::Scope::Invocation &&
        value != spv::Scope::ShaderCallKHR &&
        value != spv::Scope::QueueFamily) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4638) << spvOpcodeString(opcode)
             << ": in Vulkan environment Memory Scope is limited to Device, "
                "QueueFamily, Workgroup, ShaderCallKHR, Subgroup, or "
                "Invocation";
    } else if (_.context()->target_env == SPV_ENV_VULKAN_1_0 &&
               value == spv::Scope::Subgroup &&
               !_.HasCapability(spv::Capability::SubgroupBallotKHR) &&
               !_.HasCapability(spv::Capability::SubgroupVoteKHR)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(7951) << spvOpcodeString(opcode)
             << ": in Vulkan 1.0 environment Memory Scope is can not be "
                "Subgroup without SubgroupBallotKHR or SubgroupVoteKHR "
                "declared";
    }

    if (value == spv::Scope::ShaderCallKHR) {
      std::string errorVUID = _.VkErrorID(4640);
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              [errorVUID](spv::ExecutionModel model, std::string* message) {
                if (model != spv::ExecutionModel::RayGenerationKHR &&
                    model != spv::ExecutionModel::IntersectionKHR &&
                    model != spv::ExecutionModel::AnyHitKHR &&
                    model != spv::ExecutionModel::ClosestHitKHR &&
                    model != spv::ExecutionModel::MissKHR &&
                    model != spv::ExecutionModel::CallableKHR) {
                  if (message) {
                    *message =
                        errorVUID +
                        "ShaderCallKHR Memory Scope requires a ray tracing "
                        "execution model";
                  }
                  return false;
                }
                return true;
              });
    }

    if (value == spv::Scope::Workgroup) {
      std::string errorVUID = _.VkErrorID(7321);
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              [errorVUID](spv::ExecutionModel model, std::string* message) {
                if (model != spv::ExecutionModel::GLCompute &&
                    model != spv::ExecutionModel::TessellationControl &&
                    model != spv::ExecutionModel::TaskNV &&
                    model != spv::ExecutionModel::MeshNV &&
                    model != spv::ExecutionModel::TaskEXT &&
                    model != spv::ExecutionModel::MeshEXT) {
                  if (message) {
                    *message = errorVUID +
                               "Workgroup Memory Scope is limited to MeshNV, "
                               "TaskNV, MeshEXT, TaskEXT, TessellationControl, "
                               "and GLCompute execution model";
                  }
                  return false;
                }
                return true;
              });

      if (_.memory_model() == spv::MemoryModel::GLSL450) {
        errorVUID = _.VkErrorID(7320);
        _.function(inst->function()->id())
            ->RegisterExecutionModelLimitation(
                [errorVUID](spv::ExecutionModel model, std::string* message) {
                  if (model == spv::ExecutionModel::TessellationControl) {
                    if (message) {
                      *message =
                          errorVUID +
                          "Workgroup Memory Scope can't be used with "
                          "TessellationControl using GLSL450 Memory Model";
                    }
                    return false;
                  }
                  return true;
                });
      }
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL environments.

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
