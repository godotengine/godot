// Copyright (c) 2023 Google Inc.
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

#ifndef SOURCE_OPT_TRIM_CAPABILITIES_PASS_H_
#define SOURCE_OPT_TRIM_CAPABILITIES_PASS_H_

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "source/enum_set.h"
#include "source/extensions.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/spirv_target_env.h"

namespace spvtools {
namespace opt {

// This is required for NDK build. The unordered_set/unordered_map
// implementation don't work with class enums.
struct ClassEnumHash {
  std::size_t operator()(spv::Capability value) const {
    using StoringType = typename std::underlying_type_t<spv::Capability>;
    return std::hash<StoringType>{}(static_cast<StoringType>(value));
  }

  std::size_t operator()(spv::Op value) const {
    using StoringType = typename std::underlying_type_t<spv::Op>;
    return std::hash<StoringType>{}(static_cast<StoringType>(value));
  }
};

// An opcode handler is a function which, given an instruction, returns either
// the required capability, or nothing.
// Each handler checks one case for a capability requirement.
//
// Example:
//  - `OpTypeImage` can have operand `A` operand which requires capability 1
//  - `OpTypeImage` can also have operand `B` which requires capability 2.
//    -> We have 2 handlers: `Handler_OpTypeImage_1` and
//    `Handler_OpTypeImage_2`.
using OpcodeHandler =
    std::optional<spv::Capability> (*)(const Instruction* instruction);

// This pass tried to remove superfluous capabilities declared in the module.
// - If all the capabilities listed by an extension are removed, the extension
//   is also trimmed.
// - If the module countains any capability listed in `kForbiddenCapabilities`,
//   the module is left untouched.
// - No capabilities listed in `kUntouchableCapabilities` are trimmed, even when
//   not used.
// - Only capabilitied listed in `kSupportedCapabilities` are supported.
// - If the module contains unsupported capabilities, results might be
//   incorrect.
class TrimCapabilitiesPass : public Pass {
 private:
  // All the capabilities supported by this optimization pass. If your module
  // contains unsupported instruction, the pass could yield bad results.
  static constexpr std::array kSupportedCapabilities{
      // clang-format off
      spv::Capability::ComputeDerivativeGroupLinearNV,
      spv::Capability::ComputeDerivativeGroupQuadsNV,
      spv::Capability::Float16,
      spv::Capability::Float64,
      spv::Capability::FragmentShaderPixelInterlockEXT,
      spv::Capability::FragmentShaderSampleInterlockEXT,
      spv::Capability::FragmentShaderShadingRateInterlockEXT,
      spv::Capability::GroupNonUniform,
      spv::Capability::GroupNonUniformArithmetic,
      spv::Capability::GroupNonUniformClustered,
      spv::Capability::GroupNonUniformPartitionedNV,
      spv::Capability::GroupNonUniformVote,
      spv::Capability::Groups,
      spv::Capability::ImageMSArray,
      spv::Capability::Int16,
      spv::Capability::Int64,
      spv::Capability::Linkage,
      spv::Capability::MinLod,
      spv::Capability::PhysicalStorageBufferAddresses,
      spv::Capability::RayQueryKHR,
      spv::Capability::RayTracingKHR,
      spv::Capability::RayTraversalPrimitiveCullingKHR,
      spv::Capability::Shader,
      spv::Capability::ShaderClockKHR,
      spv::Capability::StorageImageReadWithoutFormat,
      spv::Capability::StorageInputOutput16,
      spv::Capability::StoragePushConstant16,
      spv::Capability::StorageUniform16,
      spv::Capability::StorageUniformBufferBlock16,
      spv::Capability::VulkanMemoryModelDeviceScope,
      // clang-format on
  };

  // Those capabilities disable all transformation of the module.
  static constexpr std::array kForbiddenCapabilities{
      spv::Capability::Linkage,
  };

  // Those capabilities are never removed from a module because we cannot
  // guess from the SPIR-V only if they are required or not.
  static constexpr std::array kUntouchableCapabilities{
      spv::Capability::Shader,
  };

 public:
  TrimCapabilitiesPass();
  TrimCapabilitiesPass(const TrimCapabilitiesPass&) = delete;
  TrimCapabilitiesPass(TrimCapabilitiesPass&&) = delete;

 private:
  // Inserts every capability listed by `descriptor` this pass supports into
  // `output`. Expects a Descriptor like `spv_opcode_desc_t` or
  // `spv_operand_desc_t`.
  template <class Descriptor>
  inline void addSupportedCapabilitiesToSet(const Descriptor* const descriptor,
                                            CapabilitySet* output) const {
    const uint32_t capabilityCount = descriptor->numCapabilities;
    for (uint32_t i = 0; i < capabilityCount; ++i) {
      const auto capability = descriptor->capabilities[i];
      if (supportedCapabilities_.contains(capability)) {
        output->insert(capability);
      }
    }
  }

  // Inserts every extension listed by `descriptor` required by the module into
  // `output`. Expects a Descriptor like `spv_opcode_desc_t` or
  // `spv_operand_desc_t`.
  template <class Descriptor>
  inline void addSupportedExtensionsToSet(const Descriptor* const descriptor,
                                          ExtensionSet* output) const {
    if (descriptor->minVersion <=
        spvVersionForTargetEnv(context()->GetTargetEnv())) {
      return;
    }
    output->insert(descriptor->extensions,
                   descriptor->extensions + descriptor->numExtensions);
  }

  void addInstructionRequirementsForOpcode(spv::Op opcode,
                                           CapabilitySet* capabilities,
                                           ExtensionSet* extensions) const;
  void addInstructionRequirementsForOperand(const Operand& operand,
                                            CapabilitySet* capabilities,
                                            ExtensionSet* extensions) const;

  // Given an `instruction`, determines the capabilities it requires, and output
  // them in `capabilities`. The returned capabilities form a subset of
  // kSupportedCapabilities.
  void addInstructionRequirements(Instruction* instruction,
                                  CapabilitySet* capabilities,
                                  ExtensionSet* extensions) const;

  // Given an operand `type` and `value`, adds the extensions it would require
  // to `extensions`.
  void AddExtensionsForOperand(const spv_operand_type_t type,
                               const uint32_t value,
                               ExtensionSet* extensions) const;

  // Returns the list of required capabilities and extensions for the module.
  // The returned capabilities form a subset of kSupportedCapabilities.
  std::pair<CapabilitySet, ExtensionSet>
  DetermineRequiredCapabilitiesAndExtensions() const;

  // Trims capabilities not listed in `required_capabilities` if possible.
  // Returns whether or not the module was modified.
  Pass::Status TrimUnrequiredCapabilities(
      const CapabilitySet& required_capabilities) const;

  // Trims extensions not listed in `required_extensions` if supported by this
  // pass. An extensions is considered supported as soon as one capability this
  // pass support requires it.
  Pass::Status TrimUnrequiredExtensions(
      const ExtensionSet& required_extensions) const;

  // Returns if the analyzed module contains any forbidden capability.
  bool HasForbiddenCapabilities() const;

 public:
  const char* name() const override { return "trim-capabilities"; }
  Status Process() override;

 private:
  const CapabilitySet supportedCapabilities_;
  const CapabilitySet forbiddenCapabilities_;
  const CapabilitySet untouchableCapabilities_;
  const std::unordered_multimap<spv::Op, OpcodeHandler, ClassEnumHash>
      opcodeHandlers_;
};

}  // namespace opt
}  // namespace spvtools
#endif  // SOURCE_OPT_TRIM_CAPABILITIES_H_
