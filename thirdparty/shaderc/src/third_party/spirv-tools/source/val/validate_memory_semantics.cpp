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

#include "source/val/validate_memory_semantics.h"

#include "source/diagnostic.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ValidateMemorySemantics(ValidationState_t& _,
                                     const Instruction* inst,
                                     uint32_t operand_index) {
  const SpvOp opcode = inst->opcode();
  const auto id = inst->GetOperandAs<const uint32_t>(operand_index);
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(id);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": expected Memory Semantics to be a 32-bit int";
  }

  if (!is_const_int32) {
    if (_.HasCapability(SpvCapabilityShader)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics ids must be OpConstant when Shader "
                "capability is present";
    }
    return SPV_SUCCESS;
  }

  if (spvIsWebGPUEnv(_.context()->target_env)) {
    uint32_t valid_bits = SpvMemorySemanticsAcquireMask |
                          SpvMemorySemanticsReleaseMask |
                          SpvMemorySemanticsAcquireReleaseMask |
                          SpvMemorySemanticsUniformMemoryMask |
                          SpvMemorySemanticsWorkgroupMemoryMask |
                          SpvMemorySemanticsImageMemoryMask |
                          SpvMemorySemanticsOutputMemoryKHRMask |
                          SpvMemorySemanticsMakeAvailableKHRMask |
                          SpvMemorySemanticsMakeVisibleKHRMask;
    if (value & ~valid_bits) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "WebGPU spec disallows any bit masks in Memory Semantics that "
                "are not Acquire, Release, AcquireRelease, UniformMemory, "
                "WorkgroupMemory, ImageMemory, OutputMemoryKHR, "
                "MakeAvailableKHR, or MakeVisibleKHR";
    }
  }

  const size_t num_memory_order_set_bits = spvtools::utils::CountSetBits(
      value & (SpvMemorySemanticsAcquireMask | SpvMemorySemanticsReleaseMask |
               SpvMemorySemanticsAcquireReleaseMask |
               SpvMemorySemanticsSequentiallyConsistentMask));

  if (num_memory_order_set_bits > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics can have at most one of the following "
              "bits "
              "set: Acquire, Release, AcquireRelease or "
              "SequentiallyConsistent";
  }

  if (_.memory_model() == SpvMemoryModelVulkanKHR &&
      value & SpvMemorySemanticsSequentiallyConsistentMask) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "SequentiallyConsistent memory "
              "semantics cannot be used with "
              "the VulkanKHR memory model.";
  }

  if (value & SpvMemorySemanticsMakeAvailableKHRMask &&
      !_.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics MakeAvailableKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & SpvMemorySemanticsMakeVisibleKHRMask &&
      !_.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics MakeVisibleKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & SpvMemorySemanticsOutputMemoryKHRMask &&
      !_.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics OutputMemoryKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & SpvMemorySemanticsUniformMemoryMask &&
      !_.HasCapability(SpvCapabilityShader)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics UniformMemory requires capability Shader";
  }

  // Checking for SpvCapabilityAtomicStorage is intentionally not done here. See
  // https://github.com/KhronosGroup/glslang/issues/1618 for the reasoning why.

  if (value & (SpvMemorySemanticsMakeAvailableKHRMask |
               SpvMemorySemanticsMakeVisibleKHRMask)) {
    const bool includes_storage_class =
        value & (SpvMemorySemanticsUniformMemoryMask |
                 SpvMemorySemanticsSubgroupMemoryMask |
                 SpvMemorySemanticsWorkgroupMemoryMask |
                 SpvMemorySemanticsCrossWorkgroupMemoryMask |
                 SpvMemorySemanticsAtomicCounterMemoryMask |
                 SpvMemorySemanticsImageMemoryMask |
                 SpvMemorySemanticsOutputMemoryKHRMask);

    if (!includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a storage class";
    }
  }

  if (value & SpvMemorySemanticsMakeVisibleKHRMask &&
      !(value & (SpvMemorySemanticsAcquireMask |
                 SpvMemorySemanticsAcquireReleaseMask))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": MakeVisibleKHR Memory Semantics also requires either Acquire "
              "or AcquireRelease Memory Semantics";
  }

  if (value & SpvMemorySemanticsMakeAvailableKHRMask &&
      !(value & (SpvMemorySemanticsReleaseMask |
                 SpvMemorySemanticsAcquireReleaseMask))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": MakeAvailableKHR Memory Semantics also requires either "
              "Release or AcquireRelease Memory Semantics";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const bool includes_storage_class =
        value & (SpvMemorySemanticsUniformMemoryMask |
                 SpvMemorySemanticsWorkgroupMemoryMask |
                 SpvMemorySemanticsImageMemoryMask |
                 SpvMemorySemanticsOutputMemoryKHRMask);

    if (opcode == SpvOpMemoryBarrier && !num_memory_order_set_bits) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Vulkan specification requires Memory Semantics to have "
                "one "
                "of the following bits set: Acquire, Release, "
                "AcquireRelease "
                "or SequentiallyConsistent";
    }

    if (opcode == SpvOpMemoryBarrier && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class";
    }

#if 0
    // TODO(atgoo@github.com): this check fails Vulkan CTS, reenable once fixed.
    if (opcode == SpvOpControlBarrier && value && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class if Memory Semantics is not None";
    }
#endif
  }

  if (opcode == SpvOpAtomicFlagClear &&
      (value & SpvMemorySemanticsAcquireMask ||
       value & SpvMemorySemanticsAcquireReleaseMask)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Memory Semantics Acquire and AcquireRelease cannot be used "
              "with "
           << spvOpcodeString(opcode);
  }

  if (opcode == SpvOpAtomicCompareExchange && operand_index == 5 &&
      (value & SpvMemorySemanticsReleaseMask ||
       value & SpvMemorySemanticsAcquireReleaseMask)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics Release and AcquireRelease cannot be "
              "used "
              "for operand Unequal";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (opcode == SpvOpAtomicLoad &&
        (value & SpvMemorySemanticsReleaseMask ||
         value & SpvMemorySemanticsAcquireReleaseMask ||
         value & SpvMemorySemanticsSequentiallyConsistentMask)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Vulkan spec disallows OpAtomicLoad with Memory Semantics "
                "Release, AcquireRelease and SequentiallyConsistent";
    }

    if (opcode == SpvOpAtomicStore &&
        (value & SpvMemorySemanticsAcquireMask ||
         value & SpvMemorySemanticsAcquireReleaseMask ||
         value & SpvMemorySemanticsSequentiallyConsistentMask)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Vulkan spec disallows OpAtomicStore with Memory Semantics "
                "Acquire, AcquireRelease and SequentiallyConsistent";
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL environments.

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
