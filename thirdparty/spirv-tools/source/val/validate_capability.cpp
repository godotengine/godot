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

// Validates OpCapability instruction.

#include <cassert>
#include <string>
#include <unordered_set>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

bool IsSupportGuaranteedVulkan_1_0(uint32_t capability) {
  switch (spv::Capability(capability)) {
    case spv::Capability::Matrix:
    case spv::Capability::Shader:
    case spv::Capability::InputAttachment:
    case spv::Capability::Sampled1D:
    case spv::Capability::Image1D:
    case spv::Capability::SampledBuffer:
    case spv::Capability::ImageBuffer:
    case spv::Capability::ImageQuery:
    case spv::Capability::DerivativeControl:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportGuaranteedVulkan_1_1(uint32_t capability) {
  if (IsSupportGuaranteedVulkan_1_0(capability)) return true;
  switch (spv::Capability(capability)) {
    case spv::Capability::DeviceGroup:
    case spv::Capability::MultiView:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportGuaranteedVulkan_1_2(uint32_t capability) {
  if (IsSupportGuaranteedVulkan_1_1(capability)) return true;
  switch (spv::Capability(capability)) {
    case spv::Capability::ShaderNonUniform:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportOptionalVulkan_1_0(uint32_t capability) {
  switch (spv::Capability(capability)) {
    case spv::Capability::Geometry:
    case spv::Capability::Tessellation:
    case spv::Capability::Float64:
    case spv::Capability::Int64:
    case spv::Capability::Int16:
    case spv::Capability::TessellationPointSize:
    case spv::Capability::GeometryPointSize:
    case spv::Capability::ImageGatherExtended:
    case spv::Capability::StorageImageMultisample:
    case spv::Capability::UniformBufferArrayDynamicIndexing:
    case spv::Capability::SampledImageArrayDynamicIndexing:
    case spv::Capability::StorageBufferArrayDynamicIndexing:
    case spv::Capability::StorageImageArrayDynamicIndexing:
    case spv::Capability::ClipDistance:
    case spv::Capability::CullDistance:
    case spv::Capability::ImageCubeArray:
    case spv::Capability::SampleRateShading:
    case spv::Capability::SparseResidency:
    case spv::Capability::MinLod:
    case spv::Capability::SampledCubeArray:
    case spv::Capability::ImageMSArray:
    case spv::Capability::StorageImageExtendedFormats:
    case spv::Capability::InterpolationFunction:
    case spv::Capability::StorageImageReadWithoutFormat:
    case spv::Capability::StorageImageWriteWithoutFormat:
    case spv::Capability::MultiViewport:
    case spv::Capability::Int64Atomics:
    case spv::Capability::TransformFeedback:
    case spv::Capability::GeometryStreams:
    case spv::Capability::Float16:
    case spv::Capability::Int8:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportOptionalVulkan_1_1(uint32_t capability) {
  if (IsSupportOptionalVulkan_1_0(capability)) return true;

  switch (spv::Capability(capability)) {
    case spv::Capability::GroupNonUniform:
    case spv::Capability::GroupNonUniformVote:
    case spv::Capability::GroupNonUniformArithmetic:
    case spv::Capability::GroupNonUniformBallot:
    case spv::Capability::GroupNonUniformShuffle:
    case spv::Capability::GroupNonUniformShuffleRelative:
    case spv::Capability::GroupNonUniformClustered:
    case spv::Capability::GroupNonUniformQuad:
    case spv::Capability::DrawParameters:
    // Alias spv::Capability::StorageBuffer16BitAccess.
    case spv::Capability::StorageUniformBufferBlock16:
    // Alias spv::Capability::UniformAndStorageBuffer16BitAccess.
    case spv::Capability::StorageUniform16:
    case spv::Capability::StoragePushConstant16:
    case spv::Capability::StorageInputOutput16:
    case spv::Capability::DeviceGroup:
    case spv::Capability::MultiView:
    case spv::Capability::VariablePointersStorageBuffer:
    case spv::Capability::VariablePointers:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportOptionalVulkan_1_2(uint32_t capability) {
  if (IsSupportOptionalVulkan_1_1(capability)) return true;

  switch (spv::Capability(capability)) {
    case spv::Capability::DenormPreserve:
    case spv::Capability::DenormFlushToZero:
    case spv::Capability::SignedZeroInfNanPreserve:
    case spv::Capability::RoundingModeRTE:
    case spv::Capability::RoundingModeRTZ:
    case spv::Capability::VulkanMemoryModel:
    case spv::Capability::VulkanMemoryModelDeviceScope:
    case spv::Capability::StorageBuffer8BitAccess:
    case spv::Capability::UniformAndStorageBuffer8BitAccess:
    case spv::Capability::StoragePushConstant8:
    case spv::Capability::ShaderViewportIndex:
    case spv::Capability::ShaderLayer:
    case spv::Capability::PhysicalStorageBufferAddresses:
    case spv::Capability::RuntimeDescriptorArray:
    case spv::Capability::UniformTexelBufferArrayDynamicIndexing:
    case spv::Capability::StorageTexelBufferArrayDynamicIndexing:
    case spv::Capability::UniformBufferArrayNonUniformIndexing:
    case spv::Capability::SampledImageArrayNonUniformIndexing:
    case spv::Capability::StorageBufferArrayNonUniformIndexing:
    case spv::Capability::StorageImageArrayNonUniformIndexing:
    case spv::Capability::InputAttachmentArrayNonUniformIndexing:
    case spv::Capability::UniformTexelBufferArrayNonUniformIndexing:
    case spv::Capability::StorageTexelBufferArrayNonUniformIndexing:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_1_2(uint32_t capability, bool embedded_profile) {
  switch (spv::Capability(capability)) {
    case spv::Capability::Addresses:
    case spv::Capability::Float16Buffer:
    case spv::Capability::Int16:
    case spv::Capability::Int8:
    case spv::Capability::Kernel:
    case spv::Capability::Linkage:
    case spv::Capability::Vector16:
      return true;
    case spv::Capability::Int64:
      return !embedded_profile;
    default:
      break;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_2_0(uint32_t capability, bool embedded_profile) {
  if (IsSupportGuaranteedOpenCL_1_2(capability, embedded_profile)) return true;

  switch (spv::Capability(capability)) {
    case spv::Capability::DeviceEnqueue:
    case spv::Capability::GenericPointer:
    case spv::Capability::Groups:
    case spv::Capability::Pipes:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_2_2(uint32_t capability, bool embedded_profile) {
  if (IsSupportGuaranteedOpenCL_2_0(capability, embedded_profile)) return true;

  switch (spv::Capability(capability)) {
    case spv::Capability::SubgroupDispatch:
    case spv::Capability::PipeStorage:
      return true;
    default:
      break;
  }
  return false;
}

bool IsSupportOptionalOpenCL_1_2(uint32_t capability) {
  switch (spv::Capability(capability)) {
    case spv::Capability::ImageBasic:
    case spv::Capability::Float64:
      return true;
    default:
      break;
  }
  return false;
}

// Checks if |capability| was enabled by extension.
bool IsEnabledByExtension(ValidationState_t& _, uint32_t capability) {
  spv_operand_desc operand_desc = nullptr;
  _.grammar().lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, capability,
                            &operand_desc);

  // operand_desc is expected to be not null, otherwise validator would have
  // failed at an earlier stage. This 'assert' is 'just in case'.
  assert(operand_desc);

  ExtensionSet operand_exts(operand_desc->numExtensions,
                            operand_desc->extensions);
  if (operand_exts.IsEmpty()) return false;

  return _.HasAnyOfExtensions(operand_exts);
}

bool IsEnabledByCapabilityOpenCL_1_2(ValidationState_t& _,
                                     uint32_t capability) {
  if (_.HasCapability(spv::Capability::ImageBasic)) {
    switch (spv::Capability(capability)) {
      case spv::Capability::LiteralSampler:
      case spv::Capability::Sampled1D:
      case spv::Capability::Image1D:
      case spv::Capability::SampledBuffer:
      case spv::Capability::ImageBuffer:
        return true;
      default:
        break;
    }
    return false;
  }
  return false;
}

bool IsEnabledByCapabilityOpenCL_2_0(ValidationState_t& _,
                                     uint32_t capability) {
  if (_.HasCapability(spv::Capability::ImageBasic)) {
    switch (spv::Capability(capability)) {
      case spv::Capability::ImageReadWrite:
      case spv::Capability::LiteralSampler:
      case spv::Capability::Sampled1D:
      case spv::Capability::Image1D:
      case spv::Capability::SampledBuffer:
      case spv::Capability::ImageBuffer:
        return true;
      default:
        break;
    }
    return false;
  }
  return false;
}

}  // namespace

// Validates that capability declarations use operands allowed in the current
// context.
spv_result_t CapabilityPass(ValidationState_t& _, const Instruction* inst) {
  if (inst->opcode() != spv::Op::OpCapability) return SPV_SUCCESS;

  assert(inst->operands().size() == 1);

  const spv_parsed_operand_t& operand = inst->operand(0);

  assert(operand.num_words == 1);
  assert(operand.offset < inst->words().size());

  const uint32_t capability = inst->word(operand.offset);
  const auto capability_str = [&_, capability]() {
    spv_operand_desc desc = nullptr;
    if (_.grammar().lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, capability,
                                  &desc) != SPV_SUCCESS ||
        !desc) {
      return std::string("Unknown");
    }
    return std::string(desc->name);
  };

  const auto env = _.context()->target_env;
  const bool opencl_embedded = env == SPV_ENV_OPENCL_EMBEDDED_1_2 ||
                               env == SPV_ENV_OPENCL_EMBEDDED_2_0 ||
                               env == SPV_ENV_OPENCL_EMBEDDED_2_1 ||
                               env == SPV_ENV_OPENCL_EMBEDDED_2_2;
  const std::string opencl_profile = opencl_embedded ? "Embedded" : "Full";
  if (env == SPV_ENV_VULKAN_1_0) {
    if (!IsSupportGuaranteedVulkan_1_0(capability) &&
        !IsSupportOptionalVulkan_1_0(capability) &&
        !IsEnabledByExtension(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by Vulkan 1.0 specification"
             << " (or requires extension)";
    }
  } else if (env == SPV_ENV_VULKAN_1_1) {
    if (!IsSupportGuaranteedVulkan_1_1(capability) &&
        !IsSupportOptionalVulkan_1_1(capability) &&
        !IsEnabledByExtension(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by Vulkan 1.1 specification"
             << " (or requires extension)";
    }
  } else if (env == SPV_ENV_VULKAN_1_2) {
    if (!IsSupportGuaranteedVulkan_1_2(capability) &&
        !IsSupportOptionalVulkan_1_2(capability) &&
        !IsEnabledByExtension(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by Vulkan 1.2 specification"
             << " (or requires extension)";
    }
  } else if (env == SPV_ENV_OPENCL_1_2 || env == SPV_ENV_OPENCL_EMBEDDED_1_2) {
    if (!IsSupportGuaranteedOpenCL_1_2(capability, opencl_embedded) &&
        !IsSupportOptionalOpenCL_1_2(capability) &&
        !IsEnabledByExtension(_, capability) &&
        !IsEnabledByCapabilityOpenCL_1_2(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by OpenCL 1.2 " << opencl_profile
             << " Profile specification"
             << " (or requires extension or capability)";
    }
  } else if (env == SPV_ENV_OPENCL_2_0 || env == SPV_ENV_OPENCL_EMBEDDED_2_0 ||
             env == SPV_ENV_OPENCL_2_1 || env == SPV_ENV_OPENCL_EMBEDDED_2_1) {
    if (!IsSupportGuaranteedOpenCL_2_0(capability, opencl_embedded) &&
        !IsSupportOptionalOpenCL_1_2(capability) &&
        !IsEnabledByExtension(_, capability) &&
        !IsEnabledByCapabilityOpenCL_2_0(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by OpenCL 2.0/2.1 " << opencl_profile
             << " Profile specification"
             << " (or requires extension or capability)";
    }
  } else if (env == SPV_ENV_OPENCL_2_2 || env == SPV_ENV_OPENCL_EMBEDDED_2_2) {
    if (!IsSupportGuaranteedOpenCL_2_2(capability, opencl_embedded) &&
        !IsSupportOptionalOpenCL_1_2(capability) &&
        !IsEnabledByExtension(_, capability) &&
        !IsEnabledByCapabilityOpenCL_2_0(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by OpenCL 2.2 " << opencl_profile
             << " Profile specification"
             << " (or requires extension or capability)";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
