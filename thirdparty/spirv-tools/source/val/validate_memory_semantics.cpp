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
                                     uint32_t operand_index,
                                     uint32_t memory_scope) {
  const spv::Op opcode = inst->opcode();
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
    if (_.HasCapability(spv::Capability::Shader) &&
        !_.HasCapability(spv::Capability::CooperativeMatrixNV)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics ids must be OpConstant when Shader "
                "capability is present";
    }

    if (_.HasCapability(spv::Capability::Shader) &&
        _.HasCapability(spv::Capability::CooperativeMatrixNV) &&
        !spvOpcodeIsConstant(_.GetIdOpcode(id))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics must be a constant instruction when "
                "CooperativeMatrixNV capability is present";
    }
    return SPV_SUCCESS;
  }

  const size_t num_memory_order_set_bits = spvtools::utils::CountSetBits(
      value & uint32_t(spv::MemorySemanticsMask::Acquire |
                       spv::MemorySemanticsMask::Release |
                       spv::MemorySemanticsMask::AcquireRelease |
                       spv::MemorySemanticsMask::SequentiallyConsistent));

  if (num_memory_order_set_bits > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics can have at most one of the following "
              "bits "
              "set: Acquire, Release, AcquireRelease or "
              "SequentiallyConsistent";
  }

  if (_.memory_model() == spv::MemoryModel::VulkanKHR &&
      value & uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "SequentiallyConsistent memory "
              "semantics cannot be used with "
              "the VulkanKHR memory model.";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeAvailableKHR) &&
      !_.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics MakeAvailableKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeVisibleKHR) &&
      !_.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics MakeVisibleKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::OutputMemoryKHR) &&
      !_.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics OutputMemoryKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::Volatile)) {
    if (!_.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Semantics Volatile requires capability "
                "VulkanMemoryModelKHR";
    }

    if (!spvOpcodeIsAtomicOp(inst->opcode())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics Volatile can only be used with atomic "
                "instructions";
    }
  }

  if (value & uint32_t(spv::MemorySemanticsMask::UniformMemory) &&
      !_.HasCapability(spv::Capability::Shader)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics UniformMemory requires capability Shader";
  }

  // Checking for spv::Capability::AtomicStorage is intentionally not done here.
  // See https://github.com/KhronosGroup/glslang/issues/1618 for the reasoning
  // why.

  if (value & uint32_t(spv::MemorySemanticsMask::MakeAvailableKHR |
                       spv::MemorySemanticsMask::MakeVisibleKHR)) {
    const bool includes_storage_class =
        value & uint32_t(spv::MemorySemanticsMask::UniformMemory |
                         spv::MemorySemanticsMask::SubgroupMemory |
                         spv::MemorySemanticsMask::WorkgroupMemory |
                         spv::MemorySemanticsMask::CrossWorkgroupMemory |
                         spv::MemorySemanticsMask::AtomicCounterMemory |
                         spv::MemorySemanticsMask::ImageMemory |
                         spv::MemorySemanticsMask::OutputMemoryKHR);

    if (!includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a storage class";
    }
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeVisibleKHR) &&
      !(value & uint32_t(spv::MemorySemanticsMask::Acquire |
                         spv::MemorySemanticsMask::AcquireRelease))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": MakeVisibleKHR Memory Semantics also requires either Acquire "
              "or AcquireRelease Memory Semantics";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeAvailableKHR) &&
      !(value & uint32_t(spv::MemorySemanticsMask::Release |
                         spv::MemorySemanticsMask::AcquireRelease))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": MakeAvailableKHR Memory Semantics also requires either "
              "Release or AcquireRelease Memory Semantics";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const bool includes_storage_class =
        value & uint32_t(spv::MemorySemanticsMask::UniformMemory |
                         spv::MemorySemanticsMask::WorkgroupMemory |
                         spv::MemorySemanticsMask::ImageMemory |
                         spv::MemorySemanticsMask::OutputMemoryKHR);

    if (opcode == spv::Op::OpMemoryBarrier && !num_memory_order_set_bits) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4732) << spvOpcodeString(opcode)
             << ": Vulkan specification requires Memory Semantics to have "
                "one "
                "of the following bits set: Acquire, Release, "
                "AcquireRelease "
                "or SequentiallyConsistent";
    } else if (opcode != spv::Op::OpMemoryBarrier &&
               num_memory_order_set_bits) {
      // should leave only atomics and control barriers for Vulkan env
      bool memory_is_int32 = false, memory_is_const_int32 = false;
      uint32_t memory_value = 0;
      std::tie(memory_is_int32, memory_is_const_int32, memory_value) =
          _.EvalInt32IfConst(memory_scope);
      if (memory_is_int32 &&
          spv::Scope(memory_value) == spv::Scope::Invocation) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(4641) << spvOpcodeString(opcode)
               << ": Vulkan specification requires Memory Semantics to be None "
                  "if used with Invocation Memory Scope";
      }
    }

    if (opcode == spv::Op::OpMemoryBarrier && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4733) << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class";
    }

#if 0
    // TODO(atgoo@github.com): this check fails Vulkan CTS, reenable once fixed.
    if (opcode == spv::Op::OpControlBarrier && value && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class if Memory Semantics is not None";
    }
#endif
  }

  if (opcode == spv::Op::OpAtomicFlagClear &&
      (value & uint32_t(spv::MemorySemanticsMask::Acquire) ||
       value & uint32_t(spv::MemorySemanticsMask::AcquireRelease))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Memory Semantics Acquire and AcquireRelease cannot be used "
              "with "
           << spvOpcodeString(opcode);
  }

  if (opcode == spv::Op::OpAtomicCompareExchange && operand_index == 5 &&
      (value & uint32_t(spv::MemorySemanticsMask::Release) ||
       value & uint32_t(spv::MemorySemanticsMask::AcquireRelease))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics Release and AcquireRelease cannot be "
              "used "
              "for operand Unequal";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (opcode == spv::Op::OpAtomicLoad &&
        (value & uint32_t(spv::MemorySemanticsMask::Release) ||
         value & uint32_t(spv::MemorySemanticsMask::AcquireRelease) ||
         value & uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4731)
             << "Vulkan spec disallows OpAtomicLoad with Memory Semantics "
                "Release, AcquireRelease and SequentiallyConsistent";
    }

    if (opcode == spv::Op::OpAtomicStore &&
        (value & uint32_t(spv::MemorySemanticsMask::Acquire) ||
         value & uint32_t(spv::MemorySemanticsMask::AcquireRelease) ||
         value & uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4730)
             << "Vulkan spec disallows OpAtomicStore with Memory Semantics "
                "Acquire, AcquireRelease and SequentiallyConsistent";
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL environments.

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
