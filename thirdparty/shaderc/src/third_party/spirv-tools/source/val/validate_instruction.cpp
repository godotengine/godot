// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Performs validation on instructions that appear inside of a SPIR-V block.

#include "source/val/validate.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include "source/binary.h"
#include "source/diagnostic.h"
#include "source/enum_set.h"
#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/spirv_constant.h"
#include "source/spirv_definition.h"
#include "source/spirv_target_env.h"
#include "source/spirv_validator_options.h"
#include "source/util/string_utils.h"
#include "source/val/function.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

std::string ToString(const CapabilitySet& capabilities,
                     const AssemblyGrammar& grammar) {
  std::stringstream ss;
  capabilities.ForEach([&grammar, &ss](SpvCapability cap) {
    spv_operand_desc desc;
    if (SPV_SUCCESS ==
        grammar.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
      ss << desc->name << " ";
    else
      ss << cap << " ";
  });
  return ss.str();
}

bool IsValidWebGPUStorageClass(SpvStorageClass storage_class) {
  return storage_class == SpvStorageClassUniformConstant ||
         storage_class == SpvStorageClassUniform ||
         storage_class == SpvStorageClassStorageBuffer ||
         storage_class == SpvStorageClassInput ||
         storage_class == SpvStorageClassOutput ||
         storage_class == SpvStorageClassImage ||
         storage_class == SpvStorageClassWorkgroup ||
         storage_class == SpvStorageClassPrivate ||
         storage_class == SpvStorageClassFunction;
}

// Returns capabilities that enable an opcode.  An empty result is interpreted
// as no prohibition of use of the opcode.  If the result is non-empty, then
// the opcode may only be used if at least one of the capabilities is specified
// by the module.
CapabilitySet EnablingCapabilitiesForOp(const ValidationState_t& state,
                                        SpvOp opcode) {
  // Exceptions for SPV_AMD_shader_ballot
  switch (opcode) {
    // Normally these would require Group capability
    case SpvOpGroupIAddNonUniformAMD:
    case SpvOpGroupFAddNonUniformAMD:
    case SpvOpGroupFMinNonUniformAMD:
    case SpvOpGroupUMinNonUniformAMD:
    case SpvOpGroupSMinNonUniformAMD:
    case SpvOpGroupFMaxNonUniformAMD:
    case SpvOpGroupUMaxNonUniformAMD:
    case SpvOpGroupSMaxNonUniformAMD:
      if (state.HasExtension(kSPV_AMD_shader_ballot)) return CapabilitySet();
      break;
    default:
      break;
  }
  // Look it up in the grammar
  spv_opcode_desc opcode_desc = {};
  if (SPV_SUCCESS == state.grammar().lookupOpcode(opcode, &opcode_desc)) {
    return state.grammar().filterCapsAgainstTargetEnv(
        opcode_desc->capabilities, opcode_desc->numCapabilities);
  }
  return CapabilitySet();
}

// Returns SPV_SUCCESS if the given operand is enabled by capabilities declared
// in the module.  Otherwise issues an error message and returns
// SPV_ERROR_INVALID_CAPABILITY.
spv_result_t CheckRequiredCapabilities(ValidationState_t& state,
                                       const Instruction* inst,
                                       size_t which_operand,
                                       spv_operand_type_t type,
                                       uint32_t operand) {
  // Mere mention of PointSize, ClipDistance, or CullDistance in a Builtin
  // decoration does not require the associated capability.  The use of such
  // a variable value should trigger the capability requirement, but that's
  // not implemented yet.  This rule is independent of target environment.
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/365
  if (type == SPV_OPERAND_TYPE_BUILT_IN) {
    switch (operand) {
      case SpvBuiltInPointSize:
      case SpvBuiltInClipDistance:
      case SpvBuiltInCullDistance:
        return SPV_SUCCESS;
      default:
        break;
    }
  } else if (type == SPV_OPERAND_TYPE_FP_ROUNDING_MODE) {
    // Allow all FP rounding modes if requested
    if (state.features().free_fp_rounding_mode) {
      return SPV_SUCCESS;
    }
  } else if (type == SPV_OPERAND_TYPE_GROUP_OPERATION &&
             state.features().group_ops_reduce_and_scans &&
             (operand <= uint32_t(SpvGroupOperationExclusiveScan))) {
    // Allow certain group operations if requested.
    return SPV_SUCCESS;
  }

  CapabilitySet enabling_capabilities;
  spv_operand_desc operand_desc = nullptr;
  const auto lookup_result =
      state.grammar().lookupOperand(type, operand, &operand_desc);
  if (lookup_result == SPV_SUCCESS) {
    // Allow FPRoundingMode decoration if requested.
    if (type == SPV_OPERAND_TYPE_DECORATION &&
        operand_desc->value == SpvDecorationFPRoundingMode) {
      if (state.features().free_fp_rounding_mode) return SPV_SUCCESS;

      // Vulkan API requires more capabilities on rounding mode.
      if (spvIsVulkanEnv(state.context()->target_env)) {
        enabling_capabilities.Add(SpvCapabilityStorageUniformBufferBlock16);
        enabling_capabilities.Add(SpvCapabilityStorageUniform16);
        enabling_capabilities.Add(SpvCapabilityStoragePushConstant16);
        enabling_capabilities.Add(SpvCapabilityStorageInputOutput16);
      }
    } else {
      enabling_capabilities = state.grammar().filterCapsAgainstTargetEnv(
          operand_desc->capabilities, operand_desc->numCapabilities);
    }

    if (!state.HasAnyOfCapabilities(enabling_capabilities)) {
      return state.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
             << "Operand " << which_operand << " of "
             << spvOpcodeString(inst->opcode())
             << " requires one of these capabilities: "
             << ToString(enabling_capabilities, state.grammar());
    }
  }

  return SPV_SUCCESS;
}

// Returns operand's required extensions.
ExtensionSet RequiredExtensions(const ValidationState_t& state,
                                spv_operand_type_t type, uint32_t operand) {
  spv_operand_desc operand_desc;
  if (state.grammar().lookupOperand(type, operand, &operand_desc) ==
      SPV_SUCCESS) {
    assert(operand_desc);
    // If this operand is incorporated into core SPIR-V before or in the current
    // target environment, we don't require extensions anymore.
    if (spvVersionForTargetEnv(state.grammar().target_env()) >=
        operand_desc->minVersion)
      return {};
    return {operand_desc->numExtensions, operand_desc->extensions};
  }

  return {};
}

// Returns SPV_ERROR_INVALID_BINARY and emits a diagnostic if the instruction
// is explicitly reserved in the SPIR-V core spec.  Otherwise return
// SPV_SUCCESS.
spv_result_t ReservedCheck(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  switch (opcode) {
    // These instructions are enabled by a capability, but should never
    // be used anyway.
    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod: {
      spv_opcode_desc inst_desc;
      _.grammar().lookupOpcode(opcode, &inst_desc);
      return _.diag(SPV_ERROR_INVALID_BINARY, inst)
             << "Invalid Opcode name 'Op" << inst_desc->name << "'";
    }
    default:
      break;
  }
  return SPV_SUCCESS;
}

// Returns SPV_ERROR_INVALID_BINARY and emits a diagnostic if the instruction
// is invalid because of an execution environment constraint.
spv_result_t EnvironmentCheck(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  switch (opcode) {
    case SpvOpUndef:
      if (_.features().bans_op_undef) {
        return _.diag(SPV_ERROR_INVALID_BINARY, inst)
               << "OpUndef is disallowed";
      }
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

// Returns SPV_ERROR_INVALID_CAPABILITY and emits a diagnostic if the
// instruction is invalid because the required capability isn't declared
// in the module.
spv_result_t CapabilityCheck(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  CapabilitySet opcode_caps = EnablingCapabilitiesForOp(_, opcode);
  if (!_.HasAnyOfCapabilities(opcode_caps)) {
    return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
           << "Opcode " << spvOpcodeString(opcode)
           << " requires one of these capabilities: "
           << ToString(opcode_caps, _.grammar());
  }
  for (size_t i = 0; i < inst->operands().size(); ++i) {
    const auto& operand = inst->operand(i);
    const auto word = inst->word(operand.offset);
    if (spvOperandIsConcreteMask(operand.type)) {
      // Check for required capabilities for each bit position of the mask.
      for (uint32_t mask_bit = 0x80000000; mask_bit; mask_bit >>= 1) {
        if (word & mask_bit) {
          spv_result_t status =
              CheckRequiredCapabilities(_, inst, i + 1, operand.type, mask_bit);
          if (status != SPV_SUCCESS) return status;
        }
      }
    } else if (spvIsIdType(operand.type)) {
      // TODO(dneto): Check the value referenced by this Id, if we can compute
      // it.  For now, just punt, to fix issue 248:
      // https://github.com/KhronosGroup/SPIRV-Tools/issues/248
    } else {
      // Check the operand word as a whole.
      spv_result_t status =
          CheckRequiredCapabilities(_, inst, i + 1, operand.type, word);
      if (status != SPV_SUCCESS) return status;
    }
  }
  return SPV_SUCCESS;
}

// Checks that all extensions required by the given instruction's operands were
// declared in the module.
spv_result_t ExtensionCheck(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  for (size_t operand_index = 0; operand_index < inst->operands().size();
       ++operand_index) {
    const auto& operand = inst->operand(operand_index);
    const uint32_t word = inst->word(operand.offset);
    const ExtensionSet required_extensions =
        RequiredExtensions(_, operand.type, word);
    if (!_.HasAnyOfExtensions(required_extensions)) {
      return _.diag(SPV_ERROR_MISSING_EXTENSION, inst)
             << spvtools::utils::CardinalToOrdinal(operand_index + 1)
             << " operand of " << spvOpcodeString(opcode) << ": operand "
             << word << " requires one of these extensions: "
             << ExtensionSetToString(required_extensions);
    }
  }
  return SPV_SUCCESS;
}

// Checks that the instruction can be used in this target environment's base
// version. Assumes that CapabilityCheck has checked direct capability
// dependencies for the opcode.
spv_result_t VersionCheck(ValidationState_t& _, const Instruction* inst) {
  const auto opcode = inst->opcode();
  spv_opcode_desc inst_desc;
  const spv_result_t r = _.grammar().lookupOpcode(opcode, &inst_desc);
  assert(r == SPV_SUCCESS);
  (void)r;

  const auto min_version = inst_desc->minVersion;

  if (inst_desc->numCapabilities > 0u) {
    // We already checked that the direct capability dependency has been
    // satisfied. We don't need to check any further.
    return SPV_SUCCESS;
  }

  ExtensionSet exts(inst_desc->numExtensions, inst_desc->extensions);
  if (exts.IsEmpty()) {
    // If no extensions can enable this instruction, then emit error messages
    // only concerning core SPIR-V versions if errors happen.
    if (min_version == ~0u) {
      return _.diag(SPV_ERROR_WRONG_VERSION, inst)
             << spvOpcodeString(opcode) << " is reserved for future use.";
    }

    if (spvVersionForTargetEnv(_.grammar().target_env()) < min_version) {
      return _.diag(SPV_ERROR_WRONG_VERSION, inst)
             << spvOpcodeString(opcode) << " requires "
             << spvTargetEnvDescription(
                    static_cast<spv_target_env>(min_version))
             << " at minimum.";
    }
  // Otherwise, we only error out when no enabling extensions are registered.
  } else if (!_.HasAnyOfExtensions(exts)) {
    if (min_version == ~0u) {
      return _.diag(SPV_ERROR_MISSING_EXTENSION, inst)
             << spvOpcodeString(opcode)
             << " requires one of the following extensions: "
             << ExtensionSetToString(exts);
    }

    if (static_cast<uint32_t>(_.grammar().target_env()) < min_version) {
      return _.diag(SPV_ERROR_WRONG_VERSION, inst)
             << spvOpcodeString(opcode) << " requires "
             << spvTargetEnvDescription(
                    static_cast<spv_target_env>(min_version))
             << " at minimum or one of the following extensions: "
             << ExtensionSetToString(exts);
    }
  }

  return SPV_SUCCESS;
}

// Checks that the Resuld <id> is within the valid bound.
spv_result_t LimitCheckIdBound(ValidationState_t& _, const Instruction* inst) {
  if (inst->id() >= _.getIdBound()) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "Result <id> '" << inst->id()
           << "' must be less than the ID bound '" << _.getIdBound() << "'.";
  }
  return SPV_SUCCESS;
}

// Checks that the number of OpTypeStruct members is within the limit.
spv_result_t LimitCheckStruct(ValidationState_t& _, const Instruction* inst) {
  if (SpvOpTypeStruct != inst->opcode()) {
    return SPV_SUCCESS;
  }

  // Number of members is the number of operands of the instruction minus 1.
  // One operand is the result ID.
  const uint16_t limit =
      static_cast<uint16_t>(_.options()->universal_limits_.max_struct_members);
  if (inst->operands().size() - 1 > limit) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "Number of OpTypeStruct members (" << inst->operands().size() - 1
           << ") has exceeded the limit (" << limit << ").";
  }

  // Section 2.17 of SPIRV Spec specifies that the "Structure Nesting Depth"
  // must be less than or equal to 255.
  // This is interpreted as structures including other structures as members.
  // The code does not follow pointers or look into arrays to see if we reach a
  // structure downstream.
  // The nesting depth of a struct is 1+(largest depth of any member).
  // Scalars are at depth 0.
  uint32_t max_member_depth = 0;
  // Struct members start at word 2 of OpTypeStruct instruction.
  for (size_t word_i = 2; word_i < inst->words().size(); ++word_i) {
    auto member = inst->word(word_i);
    auto memberTypeInstr = _.FindDef(member);
    if (memberTypeInstr && SpvOpTypeStruct == memberTypeInstr->opcode()) {
      max_member_depth = std::max(
          max_member_depth, _.struct_nesting_depth(memberTypeInstr->id()));
    }
  }

  const uint32_t depth_limit = _.options()->universal_limits_.max_struct_depth;
  const uint32_t cur_depth = 1 + max_member_depth;
  _.set_struct_nesting_depth(inst->id(), cur_depth);
  if (cur_depth > depth_limit) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "Structure Nesting Depth may not be larger than " << depth_limit
           << ". Found " << cur_depth << ".";
  }
  return SPV_SUCCESS;
}

// Checks that the number of (literal, label) pairs in OpSwitch is within the
// limit.
spv_result_t LimitCheckSwitch(ValidationState_t& _, const Instruction* inst) {
  if (SpvOpSwitch == inst->opcode()) {
    // The instruction syntax is as follows:
    // OpSwitch <selector ID> <Default ID> literal label literal label ...
    // literal,label pairs come after the first 2 operands.
    // It is guaranteed at this point that num_operands is an even numner.
    size_t num_pairs = (inst->operands().size() - 2) / 2;
    const unsigned int num_pairs_limit =
        _.options()->universal_limits_.max_switch_branches;
    if (num_pairs > num_pairs_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY, inst)
             << "Number of (literal, label) pairs in OpSwitch (" << num_pairs
             << ") exceeds the limit (" << num_pairs_limit << ").";
    }
  }
  return SPV_SUCCESS;
}

// Ensure the number of variables of the given class does not exceed the limit.
spv_result_t LimitCheckNumVars(ValidationState_t& _, const uint32_t var_id,
                               const SpvStorageClass storage_class) {
  if (SpvStorageClassFunction == storage_class) {
    _.registerLocalVariable(var_id);
    const uint32_t num_local_vars_limit =
        _.options()->universal_limits_.max_local_variables;
    if (_.num_local_vars() > num_local_vars_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY, nullptr)
             << "Number of local variables ('Function' Storage Class) "
                "exceeded the valid limit ("
             << num_local_vars_limit << ").";
    }
  } else {
    _.registerGlobalVariable(var_id);
    const uint32_t num_global_vars_limit =
        _.options()->universal_limits_.max_global_variables;
    if (_.num_global_vars() > num_global_vars_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY, nullptr)
             << "Number of Global Variables (Storage Class other than "
                "'Function') exceeded the valid limit ("
             << num_global_vars_limit << ").";
    }
  }
  return SPV_SUCCESS;
}

// Parses OpExtension instruction and logs warnings if unsuccessful.
spv_result_t CheckIfKnownExtension(ValidationState_t& _,
                                   const Instruction* inst) {
  const std::string extension_str = GetExtensionString(&(inst->c_inst()));
  Extension extension;
  if (!GetExtensionFromString(extension_str.c_str(), &extension)) {
    return _.diag(SPV_WARNING, inst)
           << "Found unrecognized extension " << extension_str;
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t InstructionPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  if (opcode == SpvOpExtension) {
    CheckIfKnownExtension(_, inst);
  } else if (opcode == SpvOpCapability) {
    _.RegisterCapability(inst->GetOperandAs<SpvCapability>(0));
  } else if (opcode == SpvOpMemoryModel) {
    if (_.has_memory_model_specified()) {
      return _.diag(SPV_ERROR_INVALID_LAYOUT, inst)
             << "OpMemoryModel should only be provided once.";
    }
    _.set_addressing_model(inst->GetOperandAs<SpvAddressingModel>(0));
    _.set_memory_model(inst->GetOperandAs<SpvMemoryModel>(1));

    if (_.memory_model() != SpvMemoryModelVulkanKHR &&
        _.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "VulkanMemoryModelKHR capability must only be specified if the "
                "VulkanKHR memory model is used.";
    }

    if (spvIsWebGPUEnv(_.context()->target_env)) {
      if (_.addressing_model() != SpvAddressingModelLogical) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Addressing model must be Logical for WebGPU environment.";
      }
      if (_.memory_model() != SpvMemoryModelVulkanKHR) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Memory model must be VulkanKHR for WebGPU environment.";
      }
    }
  } else if (opcode == SpvOpExecutionMode) {
    const uint32_t entry_point = inst->word(1);
    _.RegisterExecutionModeForEntryPoint(entry_point,
                                         SpvExecutionMode(inst->word(2)));
  } else if (opcode == SpvOpVariable) {
    const auto storage_class = inst->GetOperandAs<SpvStorageClass>(2);
    if (auto error = LimitCheckNumVars(_, inst->id(), storage_class)) {
      return error;
    }

    if (spvIsWebGPUEnv(_.context()->target_env) &&
        !IsValidWebGPUStorageClass(storage_class)) {
      return _.diag(SPV_ERROR_INVALID_BINARY, inst)
             << "For WebGPU, OpVariable storage class must be one of "
                "UniformConstant, Uniform, StorageBuffer, Input, Output, "
                "Image, Workgroup, Private, Function for WebGPU";
    }

    if (storage_class == SpvStorageClassGeneric)
      return _.diag(SPV_ERROR_INVALID_BINARY, inst)
             << "OpVariable storage class cannot be Generic";
    if (_.current_layout_section() == kLayoutFunctionDefinitions) {
      if (storage_class != SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT, inst)
               << "Variables must have a function[7] storage class inside"
                  " of a function";
      }
      if (_.current_function().IsFirstBlock(
              _.current_function().current_block()->id()) == false) {
        return _.diag(SPV_ERROR_INVALID_CFG, inst)
               << "Variables can only be defined "
                  "in the first block of a "
                  "function";
      }
    } else {
      if (storage_class == SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT, inst)
               << "Variables can not have a function[7] storage class "
                  "outside of a function";
      }
    }
  } else if (opcode == SpvOpTypePointer) {
    const auto storage_class = inst->GetOperandAs<SpvStorageClass>(1);
    if (spvIsWebGPUEnv(_.context()->target_env) &&
        !IsValidWebGPUStorageClass(storage_class)) {
      return _.diag(SPV_ERROR_INVALID_BINARY, inst)
             << "For WebGPU, OpTypePointer storage class must be one of "
                "UniformConstant, Uniform, StorageBuffer, Input, Output, "
                "Image, Workgroup, Private, Function";
    }
  }

  // SPIR-V Spec 2.16.3: Validation Rules for Kernel Capabilities: The
  // Signedness in OpTypeInt must always be 0.
  if (SpvOpTypeInt == inst->opcode() && _.HasCapability(SpvCapabilityKernel) &&
      inst->GetOperandAs<uint32_t>(2) != 0u) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "The Signedness in OpTypeInt "
              "must always be 0 when Kernel "
              "capability is used.";
  }

  if (auto error = ExtensionCheck(_, inst)) return error;
  if (auto error = ReservedCheck(_, inst)) return error;
  if (auto error = EnvironmentCheck(_, inst)) return error;
  if (auto error = CapabilityCheck(_, inst)) return error;
  if (auto error = LimitCheckIdBound(_, inst)) return error;
  if (auto error = LimitCheckStruct(_, inst)) return error;
  if (auto error = LimitCheckSwitch(_, inst)) return error;
  if (auto error = VersionCheck(_, inst)) return error;

  // All instruction checks have passed.
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
