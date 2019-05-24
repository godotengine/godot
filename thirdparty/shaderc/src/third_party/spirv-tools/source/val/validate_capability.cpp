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

#include "source/val/validate.h"

#include <cassert>
#include <string>
#include <unordered_set>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

bool IsSupportGuaranteedVulkan_1_0(uint32_t capability) {
  switch (capability) {
    case SpvCapabilityMatrix:
    case SpvCapabilityShader:
    case SpvCapabilityInputAttachment:
    case SpvCapabilitySampled1D:
    case SpvCapabilityImage1D:
    case SpvCapabilitySampledBuffer:
    case SpvCapabilityImageBuffer:
    case SpvCapabilityImageQuery:
    case SpvCapabilityDerivativeControl:
      return true;
  }
  return false;
}

bool IsSupportGuaranteedVulkan_1_1(uint32_t capability) {
  if (IsSupportGuaranteedVulkan_1_0(capability)) return true;
  switch (capability) {
    case SpvCapabilityDeviceGroup:
    case SpvCapabilityMultiView:
      return true;
  }
  return false;
}

bool IsSupportOptionalVulkan_1_0(uint32_t capability) {
  switch (capability) {
    case SpvCapabilityGeometry:
    case SpvCapabilityTessellation:
    case SpvCapabilityFloat64:
    case SpvCapabilityInt64:
    case SpvCapabilityInt16:
    case SpvCapabilityTessellationPointSize:
    case SpvCapabilityGeometryPointSize:
    case SpvCapabilityImageGatherExtended:
    case SpvCapabilityStorageImageMultisample:
    case SpvCapabilityUniformBufferArrayDynamicIndexing:
    case SpvCapabilitySampledImageArrayDynamicIndexing:
    case SpvCapabilityStorageBufferArrayDynamicIndexing:
    case SpvCapabilityStorageImageArrayDynamicIndexing:
    case SpvCapabilityClipDistance:
    case SpvCapabilityCullDistance:
    case SpvCapabilityImageCubeArray:
    case SpvCapabilitySampleRateShading:
    case SpvCapabilitySparseResidency:
    case SpvCapabilityMinLod:
    case SpvCapabilitySampledCubeArray:
    case SpvCapabilityImageMSArray:
    case SpvCapabilityStorageImageExtendedFormats:
    case SpvCapabilityInterpolationFunction:
    case SpvCapabilityStorageImageReadWithoutFormat:
    case SpvCapabilityStorageImageWriteWithoutFormat:
    case SpvCapabilityMultiViewport:
    case SpvCapabilityInt64Atomics:
    case SpvCapabilityTransformFeedback:
    case SpvCapabilityGeometryStreams:
    case SpvCapabilityFloat16:
    case SpvCapabilityInt8:
      return true;
  }
  return false;
}

bool IsSupportOptionalVulkan_1_1(uint32_t capability) {
  if (IsSupportOptionalVulkan_1_0(capability)) return true;

  switch (capability) {
    case SpvCapabilityGroupNonUniform:
    case SpvCapabilityGroupNonUniformVote:
    case SpvCapabilityGroupNonUniformArithmetic:
    case SpvCapabilityGroupNonUniformBallot:
    case SpvCapabilityGroupNonUniformShuffle:
    case SpvCapabilityGroupNonUniformShuffleRelative:
    case SpvCapabilityGroupNonUniformClustered:
    case SpvCapabilityGroupNonUniformQuad:
    case SpvCapabilityDrawParameters:
    // Alias SpvCapabilityStorageBuffer16BitAccess.
    case SpvCapabilityStorageUniformBufferBlock16:
    // Alias SpvCapabilityUniformAndStorageBuffer16BitAccess.
    case SpvCapabilityStorageUniform16:
    case SpvCapabilityStoragePushConstant16:
    case SpvCapabilityStorageInputOutput16:
    case SpvCapabilityDeviceGroup:
    case SpvCapabilityMultiView:
    case SpvCapabilityVariablePointersStorageBuffer:
    case SpvCapabilityVariablePointers:
      return true;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_1_2(uint32_t capability, bool embedded_profile) {
  switch (capability) {
    case SpvCapabilityAddresses:
    case SpvCapabilityFloat16Buffer:
    case SpvCapabilityGroups:
    case SpvCapabilityInt16:
    case SpvCapabilityInt8:
    case SpvCapabilityKernel:
    case SpvCapabilityLinkage:
    case SpvCapabilityVector16:
      return true;
    case SpvCapabilityInt64:
      return !embedded_profile;
    case SpvCapabilityPipes:
      return embedded_profile;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_2_0(uint32_t capability, bool embedded_profile) {
  if (IsSupportGuaranteedOpenCL_1_2(capability, embedded_profile)) return true;

  switch (capability) {
    case SpvCapabilityDeviceEnqueue:
    case SpvCapabilityGenericPointer:
    case SpvCapabilityPipes:
      return true;
  }
  return false;
}

bool IsSupportGuaranteedOpenCL_2_2(uint32_t capability, bool embedded_profile) {
  if (IsSupportGuaranteedOpenCL_2_0(capability, embedded_profile)) return true;

  switch (capability) {
    case SpvCapabilitySubgroupDispatch:
    case SpvCapabilityPipeStorage:
      return true;
  }
  return false;
}

bool IsSupportOptionalOpenCL_1_2(uint32_t capability) {
  switch (capability) {
    case SpvCapabilityImageBasic:
    case SpvCapabilityFloat64:
      return true;
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
  if (_.HasCapability(SpvCapabilityImageBasic)) {
    switch (capability) {
      case SpvCapabilityLiteralSampler:
      case SpvCapabilitySampled1D:
      case SpvCapabilityImage1D:
      case SpvCapabilitySampledBuffer:
      case SpvCapabilityImageBuffer:
        return true;
    }
    return false;
  }
  return false;
}

bool IsEnabledByCapabilityOpenCL_2_0(ValidationState_t& _,
                                     uint32_t capability) {
  if (_.HasCapability(SpvCapabilityImageBasic)) {
    switch (capability) {
      case SpvCapabilityImageReadWrite:
      case SpvCapabilityLiteralSampler:
      case SpvCapabilitySampled1D:
      case SpvCapabilityImage1D:
      case SpvCapabilitySampledBuffer:
      case SpvCapabilityImageBuffer:
        return true;
    }
    return false;
  }
  return false;
}

bool IsSupportGuaranteedWebGPU(uint32_t capability) {
  switch (capability) {
    case SpvCapabilityMatrix:
    case SpvCapabilityShader:
    case SpvCapabilitySampled1D:
    case SpvCapabilityImage1D:
    case SpvCapabilityDerivativeControl:
    case SpvCapabilityImageQuery:
      return true;
  }
  return false;
}

}  // namespace

// Validates that capability declarations use operands allowed in the current
// context.
spv_result_t CapabilityPass(ValidationState_t& _, const Instruction* inst) {
  if (inst->opcode() != SpvOpCapability) return SPV_SUCCESS;

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
  } else if (env == SPV_ENV_WEBGPU_0) {
    if (!IsSupportGuaranteedWebGPU(capability) &&
        !IsEnabledByExtension(_, capability)) {
      return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Capability " << capability_str()
             << " is not allowed by WebGPU specification"
             << " (or requires extension)";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
