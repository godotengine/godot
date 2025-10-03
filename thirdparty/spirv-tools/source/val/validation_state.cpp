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

#include "source/val/validation_state.h"

#include <cassert>
#include <stack>
#include <utility>

#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/val/basic_block.h"
#include "source/val/construct.h"
#include "source/val/function.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace val {
namespace {

ModuleLayoutSection InstructionLayoutSection(
    ModuleLayoutSection current_section, spv::Op op) {
  // See Section 2.4
  if (spvOpcodeGeneratesType(op) || spvOpcodeIsConstant(op))
    return kLayoutTypes;

  switch (op) {
    case spv::Op::OpCapability:
      return kLayoutCapabilities;
    case spv::Op::OpExtension:
      return kLayoutExtensions;
    case spv::Op::OpExtInstImport:
      return kLayoutExtInstImport;
    case spv::Op::OpMemoryModel:
      return kLayoutMemoryModel;
    case spv::Op::OpEntryPoint:
      return kLayoutEntryPoint;
    case spv::Op::OpExecutionMode:
    case spv::Op::OpExecutionModeId:
      return kLayoutExecutionMode;
    case spv::Op::OpSourceContinued:
    case spv::Op::OpSource:
    case spv::Op::OpSourceExtension:
    case spv::Op::OpString:
      return kLayoutDebug1;
    case spv::Op::OpName:
    case spv::Op::OpMemberName:
      return kLayoutDebug2;
    case spv::Op::OpModuleProcessed:
      return kLayoutDebug3;
    case spv::Op::OpDecorate:
    case spv::Op::OpMemberDecorate:
    case spv::Op::OpGroupDecorate:
    case spv::Op::OpGroupMemberDecorate:
    case spv::Op::OpDecorationGroup:
    case spv::Op::OpDecorateId:
    case spv::Op::OpDecorateStringGOOGLE:
    case spv::Op::OpMemberDecorateStringGOOGLE:
      return kLayoutAnnotations;
    case spv::Op::OpTypeForwardPointer:
      return kLayoutTypes;
    case spv::Op::OpVariable:
      if (current_section == kLayoutTypes) return kLayoutTypes;
      return kLayoutFunctionDefinitions;
    case spv::Op::OpExtInst:
      // spv::Op::OpExtInst is only allowed in types section for certain
      // extended instruction sets. This will be checked separately.
      if (current_section == kLayoutTypes) return kLayoutTypes;
      return kLayoutFunctionDefinitions;
    case spv::Op::OpLine:
    case spv::Op::OpNoLine:
    case spv::Op::OpUndef:
      if (current_section == kLayoutTypes) return kLayoutTypes;
      return kLayoutFunctionDefinitions;
    case spv::Op::OpFunction:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpFunctionEnd:
      if (current_section == kLayoutFunctionDeclarations)
        return kLayoutFunctionDeclarations;
      return kLayoutFunctionDefinitions;
    case spv::Op::OpSamplerImageAddressingModeNV:
      return kLayoutSamplerImageAddressMode;
    default:
      break;
  }
  return kLayoutFunctionDefinitions;
}

bool IsInstructionInLayoutSection(ModuleLayoutSection layout, spv::Op op) {
  return layout == InstructionLayoutSection(layout, op);
}

// Counts the number of instructions and functions in the file.
spv_result_t CountInstructions(void* user_data,
                               const spv_parsed_instruction_t* inst) {
  ValidationState_t& _ = *(reinterpret_cast<ValidationState_t*>(user_data));
  if (spv::Op(inst->opcode) == spv::Op::OpFunction) {
    _.increment_total_functions();
  }
  _.increment_total_instructions();

  return SPV_SUCCESS;
}

spv_result_t setHeader(void* user_data, spv_endianness_t, uint32_t,
                       uint32_t version, uint32_t generator, uint32_t id_bound,
                       uint32_t) {
  ValidationState_t& vstate =
      *(reinterpret_cast<ValidationState_t*>(user_data));
  vstate.setIdBound(id_bound);
  vstate.setGenerator(generator);
  vstate.setVersion(version);

  return SPV_SUCCESS;
}

// Add features based on SPIR-V core version number.
void UpdateFeaturesBasedOnSpirvVersion(ValidationState_t::Feature* features,
                                       uint32_t version) {
  assert(features);
  if (version >= SPV_SPIRV_VERSION_WORD(1, 4)) {
    features->select_between_composites = true;
    features->copy_memory_permits_two_memory_accesses = true;
    features->uconvert_spec_constant_op = true;
    features->nonwritable_var_in_function_or_private = true;
  }
}

}  // namespace

ValidationState_t::ValidationState_t(const spv_const_context ctx,
                                     const spv_const_validator_options opt,
                                     const uint32_t* words,
                                     const size_t num_words,
                                     const uint32_t max_warnings)
    : context_(ctx),
      options_(opt),
      words_(words),
      num_words_(num_words),
      unresolved_forward_ids_{},
      operand_names_{},
      current_layout_section_(kLayoutCapabilities),
      module_functions_(),
      module_capabilities_(),
      module_extensions_(),
      ordered_instructions_(),
      all_definitions_(),
      global_vars_(),
      local_vars_(),
      struct_nesting_depth_(),
      struct_has_nested_blockorbufferblock_struct_(),
      grammar_(ctx),
      addressing_model_(spv::AddressingModel::Max),
      memory_model_(spv::MemoryModel::Max),
      pointer_size_and_alignment_(0),
      sampler_image_addressing_mode_(0),
      in_function_(false),
      num_of_warnings_(0),
      max_num_of_warnings_(max_warnings) {
  assert(opt && "Validator options may not be Null.");

  const auto env = context_->target_env;

  if (spvIsVulkanEnv(env)) {
    // Vulkan 1.1 includes VK_KHR_relaxed_block_layout in core.
    if (env != SPV_ENV_VULKAN_1_0) {
      features_.env_relaxed_block_layout = true;
    }
  }

  // LocalSizeId is only disallowed prior to Vulkan 1.3 without maintenance4.
  switch (env) {
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_VULKAN_1_2:
      features_.env_allow_localsizeid = false;
      break;
    default:
      features_.env_allow_localsizeid = true;
      break;
  }

  // Only attempt to count if we have words, otherwise let the other validation
  // fail and generate an error.
  if (num_words > 0) {
    // Count the number of instructions in the binary.
    // This parse should not produce any error messages. Hijack the context and
    // replace the message consumer so that we do not pollute any state in input
    // consumer.
    spv_context_t hijacked_context = *ctx;
    hijacked_context.consumer = [](spv_message_level_t, const char*,
                                   const spv_position_t&, const char*) {};
    spvBinaryParse(&hijacked_context, this, words, num_words, setHeader,
                   CountInstructions,
                   /* diagnostic = */ nullptr);
    preallocateStorage();
  }
  UpdateFeaturesBasedOnSpirvVersion(&features_, version_);

  name_mapper_ = spvtools::GetTrivialNameMapper();
  if (options_->use_friendly_names) {
    friendly_mapper_ = spvtools::MakeUnique<spvtools::FriendlyNameMapper>(
        context_, words_, num_words_);
    name_mapper_ = friendly_mapper_->GetNameMapper();
  }
}

void ValidationState_t::preallocateStorage() {
  ordered_instructions_.reserve(total_instructions_);
  module_functions_.reserve(total_functions_);
}

spv_result_t ValidationState_t::ForwardDeclareId(uint32_t id) {
  unresolved_forward_ids_.insert(id);
  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::RemoveIfForwardDeclared(uint32_t id) {
  unresolved_forward_ids_.erase(id);
  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::RegisterForwardPointer(uint32_t id) {
  forward_pointer_ids_.insert(id);
  return SPV_SUCCESS;
}

bool ValidationState_t::IsForwardPointer(uint32_t id) const {
  return (forward_pointer_ids_.find(id) != forward_pointer_ids_.end());
}

void ValidationState_t::AssignNameToId(uint32_t id, std::string name) {
  operand_names_[id] = name;
}

std::string ValidationState_t::getIdName(uint32_t id) const {
  const std::string id_name = name_mapper_(id);

  std::stringstream out;
  out << "'" << id << "[%" << id_name << "]'";
  return out.str();
}

size_t ValidationState_t::unresolved_forward_id_count() const {
  return unresolved_forward_ids_.size();
}

std::vector<uint32_t> ValidationState_t::UnresolvedForwardIds() const {
  std::vector<uint32_t> out(std::begin(unresolved_forward_ids_),
                            std::end(unresolved_forward_ids_));
  return out;
}

bool ValidationState_t::IsDefinedId(uint32_t id) const {
  return all_definitions_.find(id) != std::end(all_definitions_);
}

const Instruction* ValidationState_t::FindDef(uint32_t id) const {
  auto it = all_definitions_.find(id);
  if (it == all_definitions_.end()) return nullptr;
  return it->second;
}

Instruction* ValidationState_t::FindDef(uint32_t id) {
  auto it = all_definitions_.find(id);
  if (it == all_definitions_.end()) return nullptr;
  return it->second;
}

ModuleLayoutSection ValidationState_t::current_layout_section() const {
  return current_layout_section_;
}

void ValidationState_t::ProgressToNextLayoutSectionOrder() {
  // Guard against going past the last element(kLayoutFunctionDefinitions)
  if (current_layout_section_ <= kLayoutFunctionDefinitions) {
    current_layout_section_ =
        static_cast<ModuleLayoutSection>(current_layout_section_ + 1);
  }
}

bool ValidationState_t::IsOpcodeInPreviousLayoutSection(spv::Op op) {
  ModuleLayoutSection section =
      InstructionLayoutSection(current_layout_section_, op);
  return section < current_layout_section_;
}

bool ValidationState_t::IsOpcodeInCurrentLayoutSection(spv::Op op) {
  return IsInstructionInLayoutSection(current_layout_section_, op);
}

DiagnosticStream ValidationState_t::diag(spv_result_t error_code,
                                         const Instruction* inst) {
  if (error_code == SPV_WARNING) {
    if (num_of_warnings_ == max_num_of_warnings_) {
      DiagnosticStream({0, 0, 0}, context_->consumer, "", error_code)
          << "Other warnings have been suppressed.\n";
    }
    if (num_of_warnings_ >= max_num_of_warnings_) {
      return DiagnosticStream({0, 0, 0}, nullptr, "", error_code);
    }
    ++num_of_warnings_;
  }

  std::string disassembly;
  if (inst) disassembly = Disassemble(*inst);

  return DiagnosticStream({0, 0, inst ? inst->LineNum() : 0},
                          context_->consumer, disassembly, error_code);
}

std::vector<Function>& ValidationState_t::functions() {
  return module_functions_;
}

Function& ValidationState_t::current_function() {
  assert(in_function_body());
  return module_functions_.back();
}

const Function& ValidationState_t::current_function() const {
  assert(in_function_body());
  return module_functions_.back();
}

const Function* ValidationState_t::function(uint32_t id) const {
  const auto it = id_to_function_.find(id);
  if (it == id_to_function_.end()) return nullptr;
  return it->second;
}

Function* ValidationState_t::function(uint32_t id) {
  auto it = id_to_function_.find(id);
  if (it == id_to_function_.end()) return nullptr;
  return it->second;
}

bool ValidationState_t::in_function_body() const { return in_function_; }

bool ValidationState_t::in_block() const {
  return module_functions_.empty() == false &&
         module_functions_.back().current_block() != nullptr;
}

void ValidationState_t::RegisterCapability(spv::Capability cap) {
  // Avoid redundant work.  Otherwise the recursion could induce work
  // quadrdatic in the capability dependency depth. (Ok, not much, but
  // it's something.)
  if (module_capabilities_.Contains(cap)) return;

  module_capabilities_.Add(cap);
  spv_operand_desc desc;
  if (SPV_SUCCESS == grammar_.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY,
                                            uint32_t(cap), &desc)) {
    CapabilitySet(desc->numCapabilities, desc->capabilities)
        .ForEach([this](spv::Capability c) { RegisterCapability(c); });
  }

  switch (cap) {
    case spv::Capability::Kernel:
      features_.group_ops_reduce_and_scans = true;
      break;
    case spv::Capability::Int8:
      features_.use_int8_type = true;
      features_.declare_int8_type = true;
      break;
    case spv::Capability::StorageBuffer8BitAccess:
    case spv::Capability::UniformAndStorageBuffer8BitAccess:
    case spv::Capability::StoragePushConstant8:
    case spv::Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR:
      features_.declare_int8_type = true;
      break;
    case spv::Capability::Int16:
      features_.declare_int16_type = true;
      break;
    case spv::Capability::Float16:
    case spv::Capability::Float16Buffer:
      features_.declare_float16_type = true;
      break;
    case spv::Capability::StorageUniformBufferBlock16:
    case spv::Capability::StorageUniform16:
    case spv::Capability::StoragePushConstant16:
    case spv::Capability::StorageInputOutput16:
    case spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR:
      features_.declare_int16_type = true;
      features_.declare_float16_type = true;
      features_.free_fp_rounding_mode = true;
      break;
    case spv::Capability::VariablePointers:
    case spv::Capability::VariablePointersStorageBuffer:
      features_.variable_pointers = true;
      break;
    default:
      // TODO(dneto): For now don't validate SPV_NV_ray_tracing, which uses
      // capability spv::Capability::RayTracingNV.
      // spv::Capability::RayTracingProvisionalKHR would need the same
      // treatment. One of the differences going from SPV_KHR_ray_tracing from
      // provisional to final spec was the provisional spec uses Locations
      // for variables in certain storage classes, just like the
      // SPV_NV_ray_tracing extension.  So it mimics the NVIDIA extension.
      // The final SPV_KHR_ray_tracing uses a different capability token
      // number, so it doesn't fall into this case.
      break;
  }
}

void ValidationState_t::RegisterExtension(Extension ext) {
  if (module_extensions_.Contains(ext)) return;

  module_extensions_.Add(ext);

  switch (ext) {
    case kSPV_AMD_gpu_shader_half_float:
    case kSPV_AMD_gpu_shader_half_float_fetch:
      // SPV_AMD_gpu_shader_half_float enables float16 type.
      // https://github.com/KhronosGroup/SPIRV-Tools/issues/1375
      features_.declare_float16_type = true;
      break;
    case kSPV_AMD_gpu_shader_int16:
      // This is not yet in the extension, but it's recommended for it.
      // See https://github.com/KhronosGroup/glslang/issues/848
      features_.uconvert_spec_constant_op = true;
      break;
    case kSPV_AMD_shader_ballot:
      // The grammar doesn't encode the fact that SPV_AMD_shader_ballot
      // enables the use of group operations Reduce, InclusiveScan,
      // and ExclusiveScan.  Enable it manually.
      // https://github.com/KhronosGroup/SPIRV-Tools/issues/991
      features_.group_ops_reduce_and_scans = true;
      break;
    default:
      break;
  }
}

bool ValidationState_t::HasAnyOfCapabilities(
    const CapabilitySet& capabilities) const {
  return module_capabilities_.HasAnyOf(capabilities);
}

bool ValidationState_t::HasAnyOfExtensions(
    const ExtensionSet& extensions) const {
  return module_extensions_.HasAnyOf(extensions);
}

void ValidationState_t::set_addressing_model(spv::AddressingModel am) {
  addressing_model_ = am;
  switch (am) {
    case spv::AddressingModel::Physical32:
      pointer_size_and_alignment_ = 4;
      break;
    default:
    // fall through
    case spv::AddressingModel::Physical64:
    case spv::AddressingModel::PhysicalStorageBuffer64:
      pointer_size_and_alignment_ = 8;
      break;
  }
}

spv::AddressingModel ValidationState_t::addressing_model() const {
  return addressing_model_;
}

void ValidationState_t::set_memory_model(spv::MemoryModel mm) {
  memory_model_ = mm;
}

spv::MemoryModel ValidationState_t::memory_model() const {
  return memory_model_;
}

void ValidationState_t::set_samplerimage_variable_address_mode(
    uint32_t bit_width) {
  sampler_image_addressing_mode_ = bit_width;
}

uint32_t ValidationState_t::samplerimage_variable_address_mode() const {
  return sampler_image_addressing_mode_;
}

spv_result_t ValidationState_t::RegisterFunction(
    uint32_t id, uint32_t ret_type_id,
    spv::FunctionControlMask function_control, uint32_t function_type_id) {
  assert(in_function_body() == false &&
         "RegisterFunction can only be called when parsing the binary outside "
         "of another function");
  in_function_ = true;
  module_functions_.emplace_back(id, ret_type_id, function_control,
                                 function_type_id);
  id_to_function_.emplace(id, &current_function());

  // TODO(umar): validate function type and type_id

  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::RegisterFunctionEnd() {
  assert(in_function_body() == true &&
         "RegisterFunctionEnd can only be called when parsing the binary "
         "inside of another function");
  assert(in_block() == false &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "outside of a block");
  current_function().RegisterFunctionEnd();
  in_function_ = false;
  return SPV_SUCCESS;
}

Instruction* ValidationState_t::AddOrderedInstruction(
    const spv_parsed_instruction_t* inst) {
  ordered_instructions_.emplace_back(inst);
  ordered_instructions_.back().SetLineNum(ordered_instructions_.size());
  return &ordered_instructions_.back();
}

// Improves diagnostic messages by collecting names of IDs
void ValidationState_t::RegisterDebugInstruction(const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpName: {
      const auto target = inst->GetOperandAs<uint32_t>(0);
      const std::string str = inst->GetOperandAs<std::string>(1);
      AssignNameToId(target, str);
      break;
    }
    case spv::Op::OpMemberName: {
      const auto target = inst->GetOperandAs<uint32_t>(0);
      const std::string str = inst->GetOperandAs<std::string>(2);
      AssignNameToId(target, str);
      break;
    }
    case spv::Op::OpSourceContinued:
    case spv::Op::OpSource:
    case spv::Op::OpSourceExtension:
    case spv::Op::OpString:
    case spv::Op::OpLine:
    case spv::Op::OpNoLine:
    default:
      break;
  }
}

void ValidationState_t::RegisterInstruction(Instruction* inst) {
  if (inst->id()) all_definitions_.insert(std::make_pair(inst->id(), inst));

  // Some validation checks are easier by getting all the consumers
  for (size_t i = 0; i < inst->operands().size(); ++i) {
    const spv_parsed_operand_t& operand = inst->operand(i);
    if ((SPV_OPERAND_TYPE_ID == operand.type) ||
        (SPV_OPERAND_TYPE_TYPE_ID == operand.type)) {
      const uint32_t operand_word = inst->word(operand.offset);
      Instruction* operand_inst = FindDef(operand_word);
      if (!operand_inst) {
        continue;
      }

      // If the instruction is using an OpTypeSampledImage as an operand, it
      // should be recorded. The validator will ensure that all usages of an
      // OpTypeSampledImage and its definition are in the same basic block.
      if ((SPV_OPERAND_TYPE_ID == operand.type) &&
          (spv::Op::OpSampledImage == operand_inst->opcode())) {
        RegisterSampledImageConsumer(operand_word, inst);
      }

      // In order to track storage classes (not Function) used per execution
      // model we can't use RegisterExecutionModelLimitation on instructions
      // like OpTypePointer which are going to be in the pre-function section.
      // Instead just need to register storage class usage for consumers in a
      // function block.
      if (inst->function()) {
        if (operand_inst->opcode() == spv::Op::OpTypePointer) {
          RegisterStorageClassConsumer(
              operand_inst->GetOperandAs<spv::StorageClass>(1), inst);
        } else if (operand_inst->opcode() == spv::Op::OpVariable) {
          RegisterStorageClassConsumer(
              operand_inst->GetOperandAs<spv::StorageClass>(2), inst);
        }
      }
    }
  }
}

std::vector<Instruction*> ValidationState_t::getSampledImageConsumers(
    uint32_t sampled_image_id) const {
  std::vector<Instruction*> result;
  auto iter = sampled_image_consumers_.find(sampled_image_id);
  if (iter != sampled_image_consumers_.end()) {
    result = iter->second;
  }
  return result;
}

void ValidationState_t::RegisterSampledImageConsumer(uint32_t sampled_image_id,
                                                     Instruction* consumer) {
  sampled_image_consumers_[sampled_image_id].push_back(consumer);
}

void ValidationState_t::RegisterStorageClassConsumer(
    spv::StorageClass storage_class, Instruction* consumer) {
  if (spvIsVulkanEnv(context()->target_env)) {
    if (storage_class == spv::StorageClass::Output) {
      std::string errorVUID = VkErrorID(4644);
      function(consumer->function()->id())
          ->RegisterExecutionModelLimitation([errorVUID](
                                                 spv::ExecutionModel model,
                                                 std::string* message) {
            if (model == spv::ExecutionModel::GLCompute ||
                model == spv::ExecutionModel::RayGenerationKHR ||
                model == spv::ExecutionModel::IntersectionKHR ||
                model == spv::ExecutionModel::AnyHitKHR ||
                model == spv::ExecutionModel::ClosestHitKHR ||
                model == spv::ExecutionModel::MissKHR ||
                model == spv::ExecutionModel::CallableKHR) {
              if (message) {
                *message =
                    errorVUID +
                    "in Vulkan environment, Output Storage Class must not be "
                    "used in GLCompute, RayGenerationKHR, IntersectionKHR, "
                    "AnyHitKHR, ClosestHitKHR, MissKHR, or CallableKHR "
                    "execution models";
              }
              return false;
            }
            return true;
          });
    }

    if (storage_class == spv::StorageClass::Workgroup) {
      std::string errorVUID = VkErrorID(4645);
      function(consumer->function()->id())
          ->RegisterExecutionModelLimitation([errorVUID](
                                                 spv::ExecutionModel model,
                                                 std::string* message) {
            if (model != spv::ExecutionModel::GLCompute &&
                model != spv::ExecutionModel::TaskNV &&
                model != spv::ExecutionModel::MeshNV &&
                model != spv::ExecutionModel::TaskEXT &&
                model != spv::ExecutionModel::MeshEXT) {
              if (message) {
                *message =
                    errorVUID +
                    "in Vulkan environment, Workgroup Storage Class is limited "
                    "to MeshNV, TaskNV, and GLCompute execution model";
              }
              return false;
            }
            return true;
          });
    }
  }

  if (storage_class == spv::StorageClass::CallableDataKHR) {
    std::string errorVUID = VkErrorID(4704);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation([errorVUID](
                                               spv::ExecutionModel model,
                                               std::string* message) {
          if (model != spv::ExecutionModel::RayGenerationKHR &&
              model != spv::ExecutionModel::ClosestHitKHR &&
              model != spv::ExecutionModel::CallableKHR &&
              model != spv::ExecutionModel::MissKHR) {
            if (message) {
              *message = errorVUID +
                         "CallableDataKHR Storage Class is limited to "
                         "RayGenerationKHR, ClosestHitKHR, CallableKHR, and "
                         "MissKHR execution model";
            }
            return false;
          }
          return true;
        });
  } else if (storage_class == spv::StorageClass::IncomingCallableDataKHR) {
    std::string errorVUID = VkErrorID(4705);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation([errorVUID](
                                               spv::ExecutionModel model,
                                               std::string* message) {
          if (model != spv::ExecutionModel::CallableKHR) {
            if (message) {
              *message = errorVUID +
                         "IncomingCallableDataKHR Storage Class is limited to "
                         "CallableKHR execution model";
            }
            return false;
          }
          return true;
        });
  } else if (storage_class == spv::StorageClass::RayPayloadKHR) {
    std::string errorVUID = VkErrorID(4698);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation([errorVUID](
                                               spv::ExecutionModel model,
                                               std::string* message) {
          if (model != spv::ExecutionModel::RayGenerationKHR &&
              model != spv::ExecutionModel::ClosestHitKHR &&
              model != spv::ExecutionModel::MissKHR) {
            if (message) {
              *message =
                  errorVUID +
                  "RayPayloadKHR Storage Class is limited to RayGenerationKHR, "
                  "ClosestHitKHR, and MissKHR execution model";
            }
            return false;
          }
          return true;
        });
  } else if (storage_class == spv::StorageClass::HitAttributeKHR) {
    std::string errorVUID = VkErrorID(4701);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation(
            [errorVUID](spv::ExecutionModel model, std::string* message) {
              if (model != spv::ExecutionModel::IntersectionKHR &&
                  model != spv::ExecutionModel::AnyHitKHR &&
                  model != spv::ExecutionModel::ClosestHitKHR) {
                if (message) {
                  *message = errorVUID +
                             "HitAttributeKHR Storage Class is limited to "
                             "IntersectionKHR, AnyHitKHR, sand ClosestHitKHR "
                             "execution model";
                }
                return false;
              }
              return true;
            });
  } else if (storage_class == spv::StorageClass::IncomingRayPayloadKHR) {
    std::string errorVUID = VkErrorID(4699);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation(
            [errorVUID](spv::ExecutionModel model, std::string* message) {
              if (model != spv::ExecutionModel::AnyHitKHR &&
                  model != spv::ExecutionModel::ClosestHitKHR &&
                  model != spv::ExecutionModel::MissKHR) {
                if (message) {
                  *message =
                      errorVUID +
                      "IncomingRayPayloadKHR Storage Class is limited to "
                      "AnyHitKHR, ClosestHitKHR, and MissKHR execution model";
                }
                return false;
              }
              return true;
            });
  } else if (storage_class == spv::StorageClass::ShaderRecordBufferKHR) {
    std::string errorVUID = VkErrorID(7119);
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation(
            [errorVUID](spv::ExecutionModel model, std::string* message) {
              if (model != spv::ExecutionModel::RayGenerationKHR &&
                  model != spv::ExecutionModel::IntersectionKHR &&
                  model != spv::ExecutionModel::AnyHitKHR &&
                  model != spv::ExecutionModel::ClosestHitKHR &&
                  model != spv::ExecutionModel::CallableKHR &&
                  model != spv::ExecutionModel::MissKHR) {
                if (message) {
                  *message =
                      errorVUID +
                      "ShaderRecordBufferKHR Storage Class is limited to "
                      "RayGenerationKHR, IntersectionKHR, AnyHitKHR, "
                      "ClosestHitKHR, CallableKHR, and MissKHR execution model";
                }
                return false;
              }
              return true;
            });
  } else if (storage_class == spv::StorageClass::TaskPayloadWorkgroupEXT) {
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation(
            [](spv::ExecutionModel model, std::string* message) {
              if (model != spv::ExecutionModel::TaskEXT &&
                  model != spv::ExecutionModel::MeshEXT) {
                if (message) {
                  *message =
                      "TaskPayloadWorkgroupEXT Storage Class is limited to "
                      "TaskEXT and MeshKHR execution model";
                }
                return false;
              }
              return true;
            });
  } else if (storage_class == spv::StorageClass::HitObjectAttributeNV) {
    function(consumer->function()->id())
        ->RegisterExecutionModelLimitation([](spv::ExecutionModel model,
                                              std::string* message) {
          if (model != spv::ExecutionModel::RayGenerationKHR &&
              model != spv::ExecutionModel::ClosestHitKHR &&
              model != spv::ExecutionModel::MissKHR) {
            if (message) {
              *message =
                  "HitObjectAttributeNV Storage Class is limited to "
                  "RayGenerationKHR, ClosestHitKHR or MissKHR execution model";
            }
            return false;
          }
          return true;
        });
  }
}

uint32_t ValidationState_t::getIdBound() const { return id_bound_; }

void ValidationState_t::setIdBound(const uint32_t bound) { id_bound_ = bound; }

bool ValidationState_t::RegisterUniqueTypeDeclaration(const Instruction* inst) {
  std::vector<uint32_t> key;
  key.push_back(static_cast<uint32_t>(inst->opcode()));
  for (size_t index = 0; index < inst->operands().size(); ++index) {
    const spv_parsed_operand_t& operand = inst->operand(index);

    if (operand.type == SPV_OPERAND_TYPE_RESULT_ID) continue;

    const int words_begin = operand.offset;
    const int words_end = words_begin + operand.num_words;
    assert(words_end <= static_cast<int>(inst->words().size()));

    key.insert(key.end(), inst->words().begin() + words_begin,
               inst->words().begin() + words_end);
  }

  return unique_type_declarations_.insert(std::move(key)).second;
}

uint32_t ValidationState_t::GetTypeId(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst ? inst->type_id() : 0;
}

spv::Op ValidationState_t::GetIdOpcode(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst ? inst->opcode() : spv::Op::OpNop;
}

uint32_t ValidationState_t::GetComponentType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  switch (inst->opcode()) {
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeBool:
      return id;

    case spv::Op::OpTypeVector:
      return inst->word(2);

    case spv::Op::OpTypeMatrix:
      return GetComponentType(inst->word(2));

    case spv::Op::OpTypeCooperativeMatrixNV:
      return inst->word(2);

    default:
      break;
  }

  if (inst->type_id()) return GetComponentType(inst->type_id());

  assert(0);
  return 0;
}

uint32_t ValidationState_t::GetDimension(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  switch (inst->opcode()) {
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeBool:
      return 1;

    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
      return inst->word(3);

    case spv::Op::OpTypeCooperativeMatrixNV:
      // Actual dimension isn't known, return 0
      return 0;

    default:
      break;
  }

  if (inst->type_id()) return GetDimension(inst->type_id());

  assert(0);
  return 0;
}

uint32_t ValidationState_t::GetBitWidth(uint32_t id) const {
  const uint32_t component_type_id = GetComponentType(id);
  const Instruction* inst = FindDef(component_type_id);
  assert(inst);

  if (inst->opcode() == spv::Op::OpTypeFloat ||
      inst->opcode() == spv::Op::OpTypeInt)
    return inst->word(2);

  if (inst->opcode() == spv::Op::OpTypeBool) return 1;

  assert(0);
  return 0;
}

bool ValidationState_t::IsVoidType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeVoid;
}

bool ValidationState_t::IsFloatScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeFloat;
}

bool ValidationState_t::IsFloatVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsFloatScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsFloatScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeFloat) {
    return true;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsFloatScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeInt;
}

bool ValidationState_t::IsIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsIntScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeInt) {
    return true;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsUnsignedIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeInt && inst->word(3) == 0;
}

bool ValidationState_t::IsUnsignedIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsUnsignedIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsSignedIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeInt && inst->word(3) == 1;
}

bool ValidationState_t::IsSignedIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsSignedIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsBoolScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeBool;
}

bool ValidationState_t::IsBoolVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsBoolScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsBoolScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeBool) {
    return true;
  }

  if (inst->opcode() == spv::Op::OpTypeVector) {
    return IsBoolScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsFloatMatrixType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    return false;
  }

  if (inst->opcode() == spv::Op::OpTypeMatrix) {
    return IsFloatScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::GetMatrixTypeInfo(uint32_t id, uint32_t* num_rows,
                                          uint32_t* num_cols,
                                          uint32_t* column_type,
                                          uint32_t* component_type) const {
  if (!id) return false;

  const Instruction* mat_inst = FindDef(id);
  assert(mat_inst);
  if (mat_inst->opcode() != spv::Op::OpTypeMatrix) return false;

  const uint32_t vec_type = mat_inst->word(2);
  const Instruction* vec_inst = FindDef(vec_type);
  assert(vec_inst);

  if (vec_inst->opcode() != spv::Op::OpTypeVector) {
    assert(0);
    return false;
  }

  *num_cols = mat_inst->word(3);
  *num_rows = vec_inst->word(3);
  *column_type = mat_inst->word(2);
  *component_type = vec_inst->word(2);

  return true;
}

bool ValidationState_t::GetStructMemberTypes(
    uint32_t struct_type_id, std::vector<uint32_t>* member_types) const {
  member_types->clear();
  if (!struct_type_id) return false;

  const Instruction* inst = FindDef(struct_type_id);
  assert(inst);
  if (inst->opcode() != spv::Op::OpTypeStruct) return false;

  *member_types =
      std::vector<uint32_t>(inst->words().cbegin() + 2, inst->words().cend());

  if (member_types->empty()) return false;

  return true;
}

bool ValidationState_t::IsPointerType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypePointer;
}

bool ValidationState_t::GetPointerTypeInfo(
    uint32_t id, uint32_t* data_type, spv::StorageClass* storage_class) const {
  *storage_class = spv::StorageClass::Max;
  if (!id) return false;

  const Instruction* inst = FindDef(id);
  assert(inst);
  if (inst->opcode() != spv::Op::OpTypePointer) return false;

  *storage_class = spv::StorageClass(inst->word(2));
  *data_type = inst->word(3);
  return true;
}

bool ValidationState_t::IsAccelerationStructureType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeAccelerationStructureKHR;
}

bool ValidationState_t::IsCooperativeMatrixType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst && inst->opcode() == spv::Op::OpTypeCooperativeMatrixNV;
}

bool ValidationState_t::IsFloatCooperativeMatrixType(uint32_t id) const {
  if (!IsCooperativeMatrixType(id)) return false;
  return IsFloatScalarType(FindDef(id)->word(2));
}

bool ValidationState_t::IsIntCooperativeMatrixType(uint32_t id) const {
  if (!IsCooperativeMatrixType(id)) return false;
  return IsIntScalarType(FindDef(id)->word(2));
}

bool ValidationState_t::IsUnsignedIntCooperativeMatrixType(uint32_t id) const {
  if (!IsCooperativeMatrixType(id)) return false;
  return IsUnsignedIntScalarType(FindDef(id)->word(2));
}

// Either a 32 bit 2-component uint vector or a 64 bit uint scalar
bool ValidationState_t::IsUnsigned64BitHandle(uint32_t id) const {
  return ((IsUnsignedIntScalarType(id) && GetBitWidth(id) == 64) ||
          (IsUnsignedIntVectorType(id) && GetDimension(id) == 2 &&
           GetBitWidth(id) == 32));
}

spv_result_t ValidationState_t::CooperativeMatrixShapesMatch(
    const Instruction* inst, uint32_t m1, uint32_t m2) {
  const auto m1_type = FindDef(m1);
  const auto m2_type = FindDef(m2);

  if (m1_type->opcode() != spv::Op::OpTypeCooperativeMatrixNV ||
      m2_type->opcode() != spv::Op::OpTypeCooperativeMatrixNV) {
    return diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected cooperative matrix types";
  }

  uint32_t m1_scope_id = m1_type->GetOperandAs<uint32_t>(2);
  uint32_t m1_rows_id = m1_type->GetOperandAs<uint32_t>(3);
  uint32_t m1_cols_id = m1_type->GetOperandAs<uint32_t>(4);

  uint32_t m2_scope_id = m2_type->GetOperandAs<uint32_t>(2);
  uint32_t m2_rows_id = m2_type->GetOperandAs<uint32_t>(3);
  uint32_t m2_cols_id = m2_type->GetOperandAs<uint32_t>(4);

  bool m1_is_int32 = false, m1_is_const_int32 = false, m2_is_int32 = false,
       m2_is_const_int32 = false;
  uint32_t m1_value = 0, m2_value = 0;

  std::tie(m1_is_int32, m1_is_const_int32, m1_value) =
      EvalInt32IfConst(m1_scope_id);
  std::tie(m2_is_int32, m2_is_const_int32, m2_value) =
      EvalInt32IfConst(m2_scope_id);

  if (m1_is_const_int32 && m2_is_const_int32 && m1_value != m2_value) {
    return diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected scopes of Matrix and Result Type to be "
           << "identical";
  }

  std::tie(m1_is_int32, m1_is_const_int32, m1_value) =
      EvalInt32IfConst(m1_rows_id);
  std::tie(m2_is_int32, m2_is_const_int32, m2_value) =
      EvalInt32IfConst(m2_rows_id);

  if (m1_is_const_int32 && m2_is_const_int32 && m1_value != m2_value) {
    return diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected rows of Matrix type and Result Type to be "
           << "identical";
  }

  std::tie(m1_is_int32, m1_is_const_int32, m1_value) =
      EvalInt32IfConst(m1_cols_id);
  std::tie(m2_is_int32, m2_is_const_int32, m2_value) =
      EvalInt32IfConst(m2_cols_id);

  if (m1_is_const_int32 && m2_is_const_int32 && m1_value != m2_value) {
    return diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected columns of Matrix type and Result Type to be "
           << "identical";
  }

  return SPV_SUCCESS;
}

uint32_t ValidationState_t::GetOperandTypeId(const Instruction* inst,
                                             size_t operand_index) const {
  return GetTypeId(inst->GetOperandAs<uint32_t>(operand_index));
}

bool ValidationState_t::GetConstantValUint64(uint32_t id, uint64_t* val) const {
  const Instruction* inst = FindDef(id);
  if (!inst) {
    assert(0 && "Instruction not found");
    return false;
  }

  if (inst->opcode() != spv::Op::OpConstant &&
      inst->opcode() != spv::Op::OpSpecConstant)
    return false;

  if (!IsIntScalarType(inst->type_id())) return false;

  if (inst->words().size() == 4) {
    *val = inst->word(3);
  } else {
    assert(inst->words().size() == 5);
    *val = inst->word(3);
    *val |= uint64_t(inst->word(4)) << 32;
  }
  return true;
}

std::tuple<bool, bool, uint32_t> ValidationState_t::EvalInt32IfConst(
    uint32_t id) const {
  const Instruction* const inst = FindDef(id);
  assert(inst);
  const uint32_t type = inst->type_id();

  if (type == 0 || !IsIntScalarType(type) || GetBitWidth(type) != 32) {
    return std::make_tuple(false, false, 0);
  }

  // Spec constant values cannot be evaluated so don't consider constant for
  // the purpose of this method.
  if (!spvOpcodeIsConstant(inst->opcode()) ||
      spvOpcodeIsSpecConstant(inst->opcode())) {
    return std::make_tuple(true, false, 0);
  }

  if (inst->opcode() == spv::Op::OpConstantNull) {
    return std::make_tuple(true, true, 0);
  }

  assert(inst->words().size() == 4);
  return std::make_tuple(true, true, inst->word(3));
}

void ValidationState_t::ComputeFunctionToEntryPointMapping() {
  for (const uint32_t entry_point : entry_points()) {
    std::stack<uint32_t> call_stack;
    std::set<uint32_t> visited;
    call_stack.push(entry_point);
    while (!call_stack.empty()) {
      const uint32_t called_func_id = call_stack.top();
      call_stack.pop();
      if (!visited.insert(called_func_id).second) continue;

      function_to_entry_points_[called_func_id].push_back(entry_point);

      const Function* called_func = function(called_func_id);
      if (called_func) {
        // Other checks should error out on this invalid SPIR-V.
        for (const uint32_t new_call : called_func->function_call_targets()) {
          call_stack.push(new_call);
        }
      }
    }
  }
}

void ValidationState_t::ComputeRecursiveEntryPoints() {
  for (const Function& func : functions()) {
    std::stack<uint32_t> call_stack;
    std::set<uint32_t> visited;

    for (const uint32_t new_call : func.function_call_targets()) {
      call_stack.push(new_call);
    }

    while (!call_stack.empty()) {
      const uint32_t called_func_id = call_stack.top();
      call_stack.pop();

      if (!visited.insert(called_func_id).second) continue;

      if (called_func_id == func.id()) {
        for (const uint32_t entry_point :
             function_to_entry_points_[called_func_id])
          recursive_entry_points_.insert(entry_point);
        break;
      }

      const Function* called_func = function(called_func_id);
      if (called_func) {
        // Other checks should error out on this invalid SPIR-V.
        for (const uint32_t new_call : called_func->function_call_targets()) {
          call_stack.push(new_call);
        }
      }
    }
  }
}

const std::vector<uint32_t>& ValidationState_t::FunctionEntryPoints(
    uint32_t func) const {
  auto iter = function_to_entry_points_.find(func);
  if (iter == function_to_entry_points_.end()) {
    return empty_ids_;
  } else {
    return iter->second;
  }
}

std::set<uint32_t> ValidationState_t::EntryPointReferences(uint32_t id) const {
  std::set<uint32_t> referenced_entry_points;
  const auto inst = FindDef(id);
  if (!inst) return referenced_entry_points;

  std::vector<const Instruction*> stack;
  stack.push_back(inst);
  while (!stack.empty()) {
    const auto current_inst = stack.back();
    stack.pop_back();

    if (const auto func = current_inst->function()) {
      // Instruction lives in a function, we can stop searching.
      const auto function_entry_points = FunctionEntryPoints(func->id());
      referenced_entry_points.insert(function_entry_points.begin(),
                                     function_entry_points.end());
    } else {
      // Instruction is in the global scope, keep searching its uses.
      for (auto pair : current_inst->uses()) {
        const auto next_inst = pair.first;
        stack.push_back(next_inst);
      }
    }
  }

  return referenced_entry_points;
}

std::string ValidationState_t::Disassemble(const Instruction& inst) const {
  const spv_parsed_instruction_t& c_inst(inst.c_inst());
  return Disassemble(c_inst.words, c_inst.num_words);
}

std::string ValidationState_t::Disassemble(const uint32_t* words,
                                           uint16_t num_words) const {
  uint32_t disassembly_options = SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                                 SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES;

  return spvInstructionBinaryToText(context()->target_env, words, num_words,
                                    words_, num_words_, disassembly_options);
}

bool ValidationState_t::LogicallyMatch(const Instruction* lhs,
                                       const Instruction* rhs,
                                       bool check_decorations) {
  if (lhs->opcode() != rhs->opcode()) {
    return false;
  }

  if (check_decorations) {
    const auto& dec_a = id_decorations(lhs->id());
    const auto& dec_b = id_decorations(rhs->id());

    for (const auto& dec : dec_b) {
      if (std::find(dec_a.begin(), dec_a.end(), dec) == dec_a.end()) {
        return false;
      }
    }
  }

  if (lhs->opcode() == spv::Op::OpTypeArray) {
    // Size operands must match.
    if (lhs->GetOperandAs<uint32_t>(2u) != rhs->GetOperandAs<uint32_t>(2u)) {
      return false;
    }

    // Elements must match or logically match.
    const auto lhs_ele_id = lhs->GetOperandAs<uint32_t>(1u);
    const auto rhs_ele_id = rhs->GetOperandAs<uint32_t>(1u);
    if (lhs_ele_id == rhs_ele_id) {
      return true;
    }

    const auto lhs_ele = FindDef(lhs_ele_id);
    const auto rhs_ele = FindDef(rhs_ele_id);
    if (!lhs_ele || !rhs_ele) {
      return false;
    }
    return LogicallyMatch(lhs_ele, rhs_ele, check_decorations);
  } else if (lhs->opcode() == spv::Op::OpTypeStruct) {
    // Number of elements must match.
    if (lhs->operands().size() != rhs->operands().size()) {
      return false;
    }

    for (size_t i = 1u; i < lhs->operands().size(); ++i) {
      const auto lhs_ele_id = lhs->GetOperandAs<uint32_t>(i);
      const auto rhs_ele_id = rhs->GetOperandAs<uint32_t>(i);
      // Elements must match or logically match.
      if (lhs_ele_id == rhs_ele_id) {
        continue;
      }

      const auto lhs_ele = FindDef(lhs_ele_id);
      const auto rhs_ele = FindDef(rhs_ele_id);
      if (!lhs_ele || !rhs_ele) {
        return false;
      }

      if (!LogicallyMatch(lhs_ele, rhs_ele, check_decorations)) {
        return false;
      }
    }

    // All checks passed.
    return true;
  }

  // No other opcodes are acceptable at this point. Arrays and structs are
  // caught above and if they're elements are not arrays or structs they are
  // required to match exactly.
  return false;
}

const Instruction* ValidationState_t::TracePointer(
    const Instruction* inst) const {
  auto base_ptr = inst;
  while (base_ptr->opcode() == spv::Op::OpAccessChain ||
         base_ptr->opcode() == spv::Op::OpInBoundsAccessChain ||
         base_ptr->opcode() == spv::Op::OpPtrAccessChain ||
         base_ptr->opcode() == spv::Op::OpInBoundsPtrAccessChain ||
         base_ptr->opcode() == spv::Op::OpCopyObject) {
    base_ptr = FindDef(base_ptr->GetOperandAs<uint32_t>(2u));
  }
  return base_ptr;
}

bool ValidationState_t::ContainsType(
    uint32_t id, const std::function<bool(const Instruction*)>& f,
    bool traverse_all_types) const {
  const auto inst = FindDef(id);
  if (!inst) return false;

  if (f(inst)) return true;

  switch (inst->opcode()) {
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypeCooperativeMatrixNV:
      return ContainsType(inst->GetOperandAs<uint32_t>(1u), f,
                          traverse_all_types);
    case spv::Op::OpTypePointer:
      if (IsForwardPointer(id)) return false;
      if (traverse_all_types) {
        return ContainsType(inst->GetOperandAs<uint32_t>(2u), f,
                            traverse_all_types);
      }
      break;
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeStruct:
      if (inst->opcode() == spv::Op::OpTypeFunction && !traverse_all_types) {
        return false;
      }
      for (uint32_t i = 1; i < inst->operands().size(); ++i) {
        if (ContainsType(inst->GetOperandAs<uint32_t>(i), f,
                         traverse_all_types)) {
          return true;
        }
      }
      break;
    default:
      break;
  }

  return false;
}

bool ValidationState_t::ContainsSizedIntOrFloatType(uint32_t id, spv::Op type,
                                                    uint32_t width) const {
  if (type != spv::Op::OpTypeInt && type != spv::Op::OpTypeFloat) return false;

  const auto f = [type, width](const Instruction* inst) {
    if (inst->opcode() == type) {
      return inst->GetOperandAs<uint32_t>(1u) == width;
    }
    return false;
  };
  return ContainsType(id, f);
}

bool ValidationState_t::ContainsLimitedUseIntOrFloatType(uint32_t id) const {
  if ((!HasCapability(spv::Capability::Int16) &&
       ContainsSizedIntOrFloatType(id, spv::Op::OpTypeInt, 16)) ||
      (!HasCapability(spv::Capability::Int8) &&
       ContainsSizedIntOrFloatType(id, spv::Op::OpTypeInt, 8)) ||
      (!HasCapability(spv::Capability::Float16) &&
       ContainsSizedIntOrFloatType(id, spv::Op::OpTypeFloat, 16))) {
    return true;
  }
  return false;
}

bool ValidationState_t::ContainsRuntimeArray(uint32_t id) const {
  const auto f = [](const Instruction* inst) {
    return inst->opcode() == spv::Op::OpTypeRuntimeArray;
  };
  return ContainsType(id, f, /* traverse_all_types = */ false);
}

bool ValidationState_t::IsValidStorageClass(
    spv::StorageClass storage_class) const {
  if (spvIsVulkanEnv(context()->target_env)) {
    switch (storage_class) {
      case spv::StorageClass::UniformConstant:
      case spv::StorageClass::Uniform:
      case spv::StorageClass::StorageBuffer:
      case spv::StorageClass::Input:
      case spv::StorageClass::Output:
      case spv::StorageClass::Image:
      case spv::StorageClass::Workgroup:
      case spv::StorageClass::Private:
      case spv::StorageClass::Function:
      case spv::StorageClass::PushConstant:
      case spv::StorageClass::PhysicalStorageBuffer:
      case spv::StorageClass::RayPayloadKHR:
      case spv::StorageClass::IncomingRayPayloadKHR:
      case spv::StorageClass::HitAttributeKHR:
      case spv::StorageClass::CallableDataKHR:
      case spv::StorageClass::IncomingCallableDataKHR:
      case spv::StorageClass::ShaderRecordBufferKHR:
      case spv::StorageClass::TaskPayloadWorkgroupEXT:
      case spv::StorageClass::HitObjectAttributeNV:
        return true;
      default:
        return false;
    }
  }

  return true;
}

#define VUID_WRAP(vuid) "[" #vuid "] "

// Currently no 2 VUID share the same id, so no need for |reference|
std::string ValidationState_t::VkErrorID(uint32_t id,
                                         const char* /*reference*/) const {
  if (!spvIsVulkanEnv(context_->target_env)) {
    return "";
  }

  // This large switch case is only searched when an error has occurred.
  // If an id is changed, the old case must be modified or removed. Each string
  // here is interpreted as being "implemented"

  // Clang format adds spaces between hyphens
  // clang-format off
  switch (id) {
    case 4154:
      return VUID_WRAP(VUID-BaryCoordKHR-BaryCoordKHR-04154);
    case 4155:
      return VUID_WRAP(VUID-BaryCoordKHR-BaryCoordKHR-04155);
    case 4156:
      return VUID_WRAP(VUID-BaryCoordKHR-BaryCoordKHR-04156);
    case 4160:
      return VUID_WRAP(VUID-BaryCoordNoPerspKHR-BaryCoordNoPerspKHR-04160);
    case 4161:
      return VUID_WRAP(VUID-BaryCoordNoPerspKHR-BaryCoordNoPerspKHR-04161);
    case 4162:
      return VUID_WRAP(VUID-BaryCoordNoPerspKHR-BaryCoordNoPerspKHR-04162);
    case 4181:
      return VUID_WRAP(VUID-BaseInstance-BaseInstance-04181);
    case 4182:
      return VUID_WRAP(VUID-BaseInstance-BaseInstance-04182);
    case 4183:
      return VUID_WRAP(VUID-BaseInstance-BaseInstance-04183);
    case 4184:
      return VUID_WRAP(VUID-BaseVertex-BaseVertex-04184);
    case 4185:
      return VUID_WRAP(VUID-BaseVertex-BaseVertex-04185);
    case 4186:
      return VUID_WRAP(VUID-BaseVertex-BaseVertex-04186);
    case 4187:
      return VUID_WRAP(VUID-ClipDistance-ClipDistance-04187);
    case 4188:
      return VUID_WRAP(VUID-ClipDistance-ClipDistance-04188);
    case 4189:
      return VUID_WRAP(VUID-ClipDistance-ClipDistance-04189);
    case 4190:
      return VUID_WRAP(VUID-ClipDistance-ClipDistance-04190);
    case 4191:
      return VUID_WRAP(VUID-ClipDistance-ClipDistance-04191);
    case 4196:
      return VUID_WRAP(VUID-CullDistance-CullDistance-04196);
    case 4197:
      return VUID_WRAP(VUID-CullDistance-CullDistance-04197);
    case 4198:
      return VUID_WRAP(VUID-CullDistance-CullDistance-04198);
    case 4199:
      return VUID_WRAP(VUID-CullDistance-CullDistance-04199);
    case 4200:
      return VUID_WRAP(VUID-CullDistance-CullDistance-04200);
    case 6735:
      return VUID_WRAP(VUID-CullMaskKHR-CullMaskKHR-06735); // Execution Model
    case 6736:
      return VUID_WRAP(VUID-CullMaskKHR-CullMaskKHR-06736); // input storage
    case 6737:
      return VUID_WRAP(VUID-CullMaskKHR-CullMaskKHR-06737); // 32 int scalar
    case 4205:
      return VUID_WRAP(VUID-DeviceIndex-DeviceIndex-04205);
    case 4206:
      return VUID_WRAP(VUID-DeviceIndex-DeviceIndex-04206);
    case 4207:
      return VUID_WRAP(VUID-DrawIndex-DrawIndex-04207);
    case 4208:
      return VUID_WRAP(VUID-DrawIndex-DrawIndex-04208);
    case 4209:
      return VUID_WRAP(VUID-DrawIndex-DrawIndex-04209);
    case 4210:
      return VUID_WRAP(VUID-FragCoord-FragCoord-04210);
    case 4211:
      return VUID_WRAP(VUID-FragCoord-FragCoord-04211);
    case 4212:
      return VUID_WRAP(VUID-FragCoord-FragCoord-04212);
    case 4213:
      return VUID_WRAP(VUID-FragDepth-FragDepth-04213);
    case 4214:
      return VUID_WRAP(VUID-FragDepth-FragDepth-04214);
    case 4215:
      return VUID_WRAP(VUID-FragDepth-FragDepth-04215);
    case 4216:
      return VUID_WRAP(VUID-FragDepth-FragDepth-04216);
    case 4217:
      return VUID_WRAP(VUID-FragInvocationCountEXT-FragInvocationCountEXT-04217);
    case 4218:
      return VUID_WRAP(VUID-FragInvocationCountEXT-FragInvocationCountEXT-04218);
    case 4219:
      return VUID_WRAP(VUID-FragInvocationCountEXT-FragInvocationCountEXT-04219);
    case 4220:
      return VUID_WRAP(VUID-FragSizeEXT-FragSizeEXT-04220);
    case 4221:
      return VUID_WRAP(VUID-FragSizeEXT-FragSizeEXT-04221);
    case 4222:
      return VUID_WRAP(VUID-FragSizeEXT-FragSizeEXT-04222);
    case 4223:
      return VUID_WRAP(VUID-FragStencilRefEXT-FragStencilRefEXT-04223);
    case 4224:
      return VUID_WRAP(VUID-FragStencilRefEXT-FragStencilRefEXT-04224);
    case 4225:
      return VUID_WRAP(VUID-FragStencilRefEXT-FragStencilRefEXT-04225);
    case 4229:
      return VUID_WRAP(VUID-FrontFacing-FrontFacing-04229);
    case 4230:
      return VUID_WRAP(VUID-FrontFacing-FrontFacing-04230);
    case 4231:
      return VUID_WRAP(VUID-FrontFacing-FrontFacing-04231);
    case 4232:
      return VUID_WRAP(VUID-FullyCoveredEXT-FullyCoveredEXT-04232);
    case 4233:
      return VUID_WRAP(VUID-FullyCoveredEXT-FullyCoveredEXT-04233);
    case 4234:
      return VUID_WRAP(VUID-FullyCoveredEXT-FullyCoveredEXT-04234);
    case 4236:
      return VUID_WRAP(VUID-GlobalInvocationId-GlobalInvocationId-04236);
    case 4237:
      return VUID_WRAP(VUID-GlobalInvocationId-GlobalInvocationId-04237);
    case 4238:
      return VUID_WRAP(VUID-GlobalInvocationId-GlobalInvocationId-04238);
    case 4239:
      return VUID_WRAP(VUID-HelperInvocation-HelperInvocation-04239);
    case 4240:
      return VUID_WRAP(VUID-HelperInvocation-HelperInvocation-04240);
    case 4241:
      return VUID_WRAP(VUID-HelperInvocation-HelperInvocation-04241);
    case 4242:
      return VUID_WRAP(VUID-HitKindKHR-HitKindKHR-04242);
    case 4243:
      return VUID_WRAP(VUID-HitKindKHR-HitKindKHR-04243);
    case 4244:
      return VUID_WRAP(VUID-HitKindKHR-HitKindKHR-04244);
    case 4245:
      return VUID_WRAP(VUID-HitTNV-HitTNV-04245);
    case 4246:
      return VUID_WRAP(VUID-HitTNV-HitTNV-04246);
    case 4247:
      return VUID_WRAP(VUID-HitTNV-HitTNV-04247);
    case 4248:
      return VUID_WRAP(VUID-IncomingRayFlagsKHR-IncomingRayFlagsKHR-04248);
    case 4249:
      return VUID_WRAP(VUID-IncomingRayFlagsKHR-IncomingRayFlagsKHR-04249);
    case 4250:
      return VUID_WRAP(VUID-IncomingRayFlagsKHR-IncomingRayFlagsKHR-04250);
    case 4251:
      return VUID_WRAP(VUID-InstanceCustomIndexKHR-InstanceCustomIndexKHR-04251);
    case 4252:
      return VUID_WRAP(VUID-InstanceCustomIndexKHR-InstanceCustomIndexKHR-04252);
    case 4253:
      return VUID_WRAP(VUID-InstanceCustomIndexKHR-InstanceCustomIndexKHR-04253);
    case 4254:
      return VUID_WRAP(VUID-InstanceId-InstanceId-04254);
    case 4255:
      return VUID_WRAP(VUID-InstanceId-InstanceId-04255);
    case 4256:
      return VUID_WRAP(VUID-InstanceId-InstanceId-04256);
    case 4257:
      return VUID_WRAP(VUID-InvocationId-InvocationId-04257);
    case 4258:
      return VUID_WRAP(VUID-InvocationId-InvocationId-04258);
    case 4259:
      return VUID_WRAP(VUID-InvocationId-InvocationId-04259);
    case 4263:
      return VUID_WRAP(VUID-InstanceIndex-InstanceIndex-04263);
    case 4264:
      return VUID_WRAP(VUID-InstanceIndex-InstanceIndex-04264);
    case 4265:
      return VUID_WRAP(VUID-InstanceIndex-InstanceIndex-04265);
    case 4266:
      return VUID_WRAP(VUID-LaunchIdKHR-LaunchIdKHR-04266);
    case 4267:
      return VUID_WRAP(VUID-LaunchIdKHR-LaunchIdKHR-04267);
    case 4268:
      return VUID_WRAP(VUID-LaunchIdKHR-LaunchIdKHR-04268);
    case 4269:
      return VUID_WRAP(VUID-LaunchSizeKHR-LaunchSizeKHR-04269);
    case 4270:
      return VUID_WRAP(VUID-LaunchSizeKHR-LaunchSizeKHR-04270);
    case 4271:
      return VUID_WRAP(VUID-LaunchSizeKHR-LaunchSizeKHR-04271);
    case 4272:
      return VUID_WRAP(VUID-Layer-Layer-04272);
    case 4273:
      return VUID_WRAP(VUID-Layer-Layer-04273);
    case 4274:
      return VUID_WRAP(VUID-Layer-Layer-04274);
    case 4275:
      return VUID_WRAP(VUID-Layer-Layer-04275);
    case 4276:
      return VUID_WRAP(VUID-Layer-Layer-04276);
    case 4281:
      return VUID_WRAP(VUID-LocalInvocationId-LocalInvocationId-04281);
    case 4282:
      return VUID_WRAP(VUID-LocalInvocationId-LocalInvocationId-04282);
    case 4283:
      return VUID_WRAP(VUID-LocalInvocationId-LocalInvocationId-04283);
    case 4293:
      return VUID_WRAP(VUID-NumSubgroups-NumSubgroups-04293);
    case 4294:
      return VUID_WRAP(VUID-NumSubgroups-NumSubgroups-04294);
    case 4295:
      return VUID_WRAP(VUID-NumSubgroups-NumSubgroups-04295);
    case 4296:
      return VUID_WRAP(VUID-NumWorkgroups-NumWorkgroups-04296);
    case 4297:
      return VUID_WRAP(VUID-NumWorkgroups-NumWorkgroups-04297);
    case 4298:
      return VUID_WRAP(VUID-NumWorkgroups-NumWorkgroups-04298);
    case 4299:
      return VUID_WRAP(VUID-ObjectRayDirectionKHR-ObjectRayDirectionKHR-04299);
    case 4300:
      return VUID_WRAP(VUID-ObjectRayDirectionKHR-ObjectRayDirectionKHR-04300);
    case 4301:
      return VUID_WRAP(VUID-ObjectRayDirectionKHR-ObjectRayDirectionKHR-04301);
    case 4302:
      return VUID_WRAP(VUID-ObjectRayOriginKHR-ObjectRayOriginKHR-04302);
    case 4303:
      return VUID_WRAP(VUID-ObjectRayOriginKHR-ObjectRayOriginKHR-04303);
    case 4304:
      return VUID_WRAP(VUID-ObjectRayOriginKHR-ObjectRayOriginKHR-04304);
    case 4305:
      return VUID_WRAP(VUID-ObjectToWorldKHR-ObjectToWorldKHR-04305);
    case 4306:
      return VUID_WRAP(VUID-ObjectToWorldKHR-ObjectToWorldKHR-04306);
    case 4307:
      return VUID_WRAP(VUID-ObjectToWorldKHR-ObjectToWorldKHR-04307);
    case 4308:
      return VUID_WRAP(VUID-PatchVertices-PatchVertices-04308);
    case 4309:
      return VUID_WRAP(VUID-PatchVertices-PatchVertices-04309);
    case 4310:
      return VUID_WRAP(VUID-PatchVertices-PatchVertices-04310);
    case 4311:
      return VUID_WRAP(VUID-PointCoord-PointCoord-04311);
    case 4312:
      return VUID_WRAP(VUID-PointCoord-PointCoord-04312);
    case 4313:
      return VUID_WRAP(VUID-PointCoord-PointCoord-04313);
    case 4314:
      return VUID_WRAP(VUID-PointSize-PointSize-04314);
    case 4315:
      return VUID_WRAP(VUID-PointSize-PointSize-04315);
    case 4316:
      return VUID_WRAP(VUID-PointSize-PointSize-04316);
    case 4317:
      return VUID_WRAP(VUID-PointSize-PointSize-04317);
    case 4318:
      return VUID_WRAP(VUID-Position-Position-04318);
    case 4319:
      return VUID_WRAP(VUID-Position-Position-04319);
    case 4320:
      return VUID_WRAP(VUID-Position-Position-04320);
    case 4321:
      return VUID_WRAP(VUID-Position-Position-04321);
    case 4330:
      return VUID_WRAP(VUID-PrimitiveId-PrimitiveId-04330);
    case 4334:
      return VUID_WRAP(VUID-PrimitiveId-PrimitiveId-04334);
    case 4337:
      return VUID_WRAP(VUID-PrimitiveId-PrimitiveId-04337);
    case 4345:
      return VUID_WRAP(VUID-RayGeometryIndexKHR-RayGeometryIndexKHR-04345);
    case 4346:
      return VUID_WRAP(VUID-RayGeometryIndexKHR-RayGeometryIndexKHR-04346);
    case 4347:
      return VUID_WRAP(VUID-RayGeometryIndexKHR-RayGeometryIndexKHR-04347);
    case 4348:
      return VUID_WRAP(VUID-RayTmaxKHR-RayTmaxKHR-04348);
    case 4349:
      return VUID_WRAP(VUID-RayTmaxKHR-RayTmaxKHR-04349);
    case 4350:
      return VUID_WRAP(VUID-RayTmaxKHR-RayTmaxKHR-04350);
    case 4351:
      return VUID_WRAP(VUID-RayTminKHR-RayTminKHR-04351);
    case 4352:
      return VUID_WRAP(VUID-RayTminKHR-RayTminKHR-04352);
    case 4353:
      return VUID_WRAP(VUID-RayTminKHR-RayTminKHR-04353);
    case 4354:
      return VUID_WRAP(VUID-SampleId-SampleId-04354);
    case 4355:
      return VUID_WRAP(VUID-SampleId-SampleId-04355);
    case 4356:
      return VUID_WRAP(VUID-SampleId-SampleId-04356);
    case 4357:
      return VUID_WRAP(VUID-SampleMask-SampleMask-04357);
    case 4358:
      return VUID_WRAP(VUID-SampleMask-SampleMask-04358);
    case 4359:
      return VUID_WRAP(VUID-SampleMask-SampleMask-04359);
    case 4360:
      return VUID_WRAP(VUID-SamplePosition-SamplePosition-04360);
    case 4361:
      return VUID_WRAP(VUID-SamplePosition-SamplePosition-04361);
    case 4362:
      return VUID_WRAP(VUID-SamplePosition-SamplePosition-04362);
    case 4367:
      return VUID_WRAP(VUID-SubgroupId-SubgroupId-04367);
    case 4368:
      return VUID_WRAP(VUID-SubgroupId-SubgroupId-04368);
    case 4369:
      return VUID_WRAP(VUID-SubgroupId-SubgroupId-04369);
    case 4370:
      return VUID_WRAP(VUID-SubgroupEqMask-SubgroupEqMask-04370);
    case 4371:
      return VUID_WRAP(VUID-SubgroupEqMask-SubgroupEqMask-04371);
    case 4372:
      return VUID_WRAP(VUID-SubgroupGeMask-SubgroupGeMask-04372);
    case 4373:
      return VUID_WRAP(VUID-SubgroupGeMask-SubgroupGeMask-04373);
    case 4374:
      return VUID_WRAP(VUID-SubgroupGtMask-SubgroupGtMask-04374);
    case 4375:
      return VUID_WRAP(VUID-SubgroupGtMask-SubgroupGtMask-04375);
    case 4376:
      return VUID_WRAP(VUID-SubgroupLeMask-SubgroupLeMask-04376);
    case 4377:
      return VUID_WRAP(VUID-SubgroupLeMask-SubgroupLeMask-04377);
    case 4378:
      return VUID_WRAP(VUID-SubgroupLtMask-SubgroupLtMask-04378);
    case 4379:
      return VUID_WRAP(VUID-SubgroupLtMask-SubgroupLtMask-04379);
    case 4380:
      return VUID_WRAP(VUID-SubgroupLocalInvocationId-SubgroupLocalInvocationId-04380);
    case 4381:
      return VUID_WRAP(VUID-SubgroupLocalInvocationId-SubgroupLocalInvocationId-04381);
    case 4382:
      return VUID_WRAP(VUID-SubgroupSize-SubgroupSize-04382);
    case 4383:
      return VUID_WRAP(VUID-SubgroupSize-SubgroupSize-04383);
    case 4387:
      return VUID_WRAP(VUID-TessCoord-TessCoord-04387);
    case 4388:
      return VUID_WRAP(VUID-TessCoord-TessCoord-04388);
    case 4389:
      return VUID_WRAP(VUID-TessCoord-TessCoord-04389);
    case 4390:
      return VUID_WRAP(VUID-TessLevelOuter-TessLevelOuter-04390);
    case 4391:
      return VUID_WRAP(VUID-TessLevelOuter-TessLevelOuter-04391);
    case 4392:
      return VUID_WRAP(VUID-TessLevelOuter-TessLevelOuter-04392);
    case 4393:
      return VUID_WRAP(VUID-TessLevelOuter-TessLevelOuter-04393);
    case 4394:
      return VUID_WRAP(VUID-TessLevelInner-TessLevelInner-04394);
    case 4395:
      return VUID_WRAP(VUID-TessLevelInner-TessLevelInner-04395);
    case 4396:
      return VUID_WRAP(VUID-TessLevelInner-TessLevelInner-04396);
    case 4397:
      return VUID_WRAP(VUID-TessLevelInner-TessLevelInner-04397);
    case 4398:
      return VUID_WRAP(VUID-VertexIndex-VertexIndex-04398);
    case 4399:
      return VUID_WRAP(VUID-VertexIndex-VertexIndex-04399);
    case 4400:
      return VUID_WRAP(VUID-VertexIndex-VertexIndex-04400);
    case 4401:
      return VUID_WRAP(VUID-ViewIndex-ViewIndex-04401);
    case 4402:
      return VUID_WRAP(VUID-ViewIndex-ViewIndex-04402);
    case 4403:
      return VUID_WRAP(VUID-ViewIndex-ViewIndex-04403);
    case 4404:
      return VUID_WRAP(VUID-ViewportIndex-ViewportIndex-04404);
    case 4405:
      return VUID_WRAP(VUID-ViewportIndex-ViewportIndex-04405);
    case 4406:
      return VUID_WRAP(VUID-ViewportIndex-ViewportIndex-04406);
    case 4407:
      return VUID_WRAP(VUID-ViewportIndex-ViewportIndex-04407);
    case 4408:
      return VUID_WRAP(VUID-ViewportIndex-ViewportIndex-04408);
    case 4422:
      return VUID_WRAP(VUID-WorkgroupId-WorkgroupId-04422);
    case 4423:
      return VUID_WRAP(VUID-WorkgroupId-WorkgroupId-04423);
    case 4424:
      return VUID_WRAP(VUID-WorkgroupId-WorkgroupId-04424);
    case 4425:
      return VUID_WRAP(VUID-WorkgroupSize-WorkgroupSize-04425);
    case 4426:
      return VUID_WRAP(VUID-WorkgroupSize-WorkgroupSize-04426);
    case 4427:
      return VUID_WRAP(VUID-WorkgroupSize-WorkgroupSize-04427);
    case 4428:
      return VUID_WRAP(VUID-WorldRayDirectionKHR-WorldRayDirectionKHR-04428);
    case 4429:
      return VUID_WRAP(VUID-WorldRayDirectionKHR-WorldRayDirectionKHR-04429);
    case 4430:
      return VUID_WRAP(VUID-WorldRayDirectionKHR-WorldRayDirectionKHR-04430);
    case 4431:
      return VUID_WRAP(VUID-WorldRayOriginKHR-WorldRayOriginKHR-04431);
    case 4432:
      return VUID_WRAP(VUID-WorldRayOriginKHR-WorldRayOriginKHR-04432);
    case 4433:
      return VUID_WRAP(VUID-WorldRayOriginKHR-WorldRayOriginKHR-04433);
    case 4434:
      return VUID_WRAP(VUID-WorldToObjectKHR-WorldToObjectKHR-04434);
    case 4435:
      return VUID_WRAP(VUID-WorldToObjectKHR-WorldToObjectKHR-04435);
    case 4436:
      return VUID_WRAP(VUID-WorldToObjectKHR-WorldToObjectKHR-04436);
    case 4484:
      return VUID_WRAP(VUID-PrimitiveShadingRateKHR-PrimitiveShadingRateKHR-04484);
    case 4485:
      return VUID_WRAP(VUID-PrimitiveShadingRateKHR-PrimitiveShadingRateKHR-04485);
    case 4486:
      return VUID_WRAP(VUID-PrimitiveShadingRateKHR-PrimitiveShadingRateKHR-04486);
    case 4490:
      return VUID_WRAP(VUID-ShadingRateKHR-ShadingRateKHR-04490);
    case 4491:
      return VUID_WRAP(VUID-ShadingRateKHR-ShadingRateKHR-04491);
    case 4492:
      return VUID_WRAP(VUID-ShadingRateKHR-ShadingRateKHR-04492);
    case 4633:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04633);
    case 4634:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04634);
    case 4635:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04635);
    case 4636:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04636);
    case 4637:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04637);
    case 4638:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04638);
    case 7321:
      return VUID_WRAP(VUID-StandaloneSpirv-None-07321);
    case 4640:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04640);
    case 4641:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04641);
    case 4642:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04642);
    case 4643:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04643);
    case 4644:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04644);
    case 4645:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04645);
    case 4651:
      return VUID_WRAP(VUID-StandaloneSpirv-OpVariable-04651);
    case 4652:
      return VUID_WRAP(VUID-StandaloneSpirv-OpReadClockKHR-04652);
    case 4653:
      return VUID_WRAP(VUID-StandaloneSpirv-OriginLowerLeft-04653);
    case 4654:
      return VUID_WRAP(VUID-StandaloneSpirv-PixelCenterInteger-04654);
    case 4655:
      return VUID_WRAP(VUID-StandaloneSpirv-UniformConstant-04655);
    case 4656:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeImage-04656);
    case 4657:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeImage-04657);
    case 4658:
      return VUID_WRAP(VUID-StandaloneSpirv-OpImageTexelPointer-04658);
    case 4659:
      return VUID_WRAP(VUID-StandaloneSpirv-OpImageQuerySizeLod-04659);
    case 4662:
      return VUID_WRAP(VUID-StandaloneSpirv-Offset-04662);
    case 4663:
      return VUID_WRAP(VUID-StandaloneSpirv-Offset-04663);
    case 4664:
      return VUID_WRAP(VUID-StandaloneSpirv-OpImageGather-04664);
    case 4667:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04667);
    case 4669:
      return VUID_WRAP(VUID-StandaloneSpirv-GLSLShared-04669);
    case 4670:
      return VUID_WRAP(VUID-StandaloneSpirv-Flat-04670);
    case 4675:
      return VUID_WRAP(VUID-StandaloneSpirv-FPRoundingMode-04675);
    case 4677:
      return VUID_WRAP(VUID-StandaloneSpirv-Invariant-04677);
    case 4680:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeRuntimeArray-04680);
    case 4682:
      return VUID_WRAP(VUID-StandaloneSpirv-OpControlBarrier-04682);
    case 6426:
      return VUID_WRAP(VUID-StandaloneSpirv-LocalSize-06426); // formally 04683
    case 4685:
      return VUID_WRAP(VUID-StandaloneSpirv-OpGroupNonUniformBallotBitCount-04685);
    case 4686:
      return VUID_WRAP(VUID-StandaloneSpirv-None-04686);
    case 4698:
      return VUID_WRAP(VUID-StandaloneSpirv-RayPayloadKHR-04698);
    case 4699:
      return VUID_WRAP(VUID-StandaloneSpirv-IncomingRayPayloadKHR-04699);
    case 4701:
      return VUID_WRAP(VUID-StandaloneSpirv-HitAttributeKHR-04701);
    case 4703:
      return VUID_WRAP(VUID-StandaloneSpirv-HitAttributeKHR-04703);
    case 4704:
      return VUID_WRAP(VUID-StandaloneSpirv-CallableDataKHR-04704);
    case 4705:
      return VUID_WRAP(VUID-StandaloneSpirv-IncomingCallableDataKHR-04705);
    case 7119:
      return VUID_WRAP(VUID-StandaloneSpirv-ShaderRecordBufferKHR-07119);
    case 4708:
      return VUID_WRAP(VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708);
    case 4710:
      return VUID_WRAP(VUID-StandaloneSpirv-PhysicalStorageBuffer64-04710);
    case 4711:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeForwardPointer-04711);
    case 4730:
      return VUID_WRAP(VUID-StandaloneSpirv-OpAtomicStore-04730);
    case 4731:
      return VUID_WRAP(VUID-StandaloneSpirv-OpAtomicLoad-04731);
    case 4732:
      return VUID_WRAP(VUID-StandaloneSpirv-OpMemoryBarrier-04732);
    case 4733:
      return VUID_WRAP(VUID-StandaloneSpirv-OpMemoryBarrier-04733);
    case 4734:
      return VUID_WRAP(VUID-StandaloneSpirv-OpVariable-04734);
    case 4744:
      return VUID_WRAP(VUID-StandaloneSpirv-Flat-04744);
    case 4777:
      return VUID_WRAP(VUID-StandaloneSpirv-OpImage-04777);
    case 4780:
      return VUID_WRAP(VUID-StandaloneSpirv-Result-04780);
    case 4781:
      return VUID_WRAP(VUID-StandaloneSpirv-Base-04781);
    case 4915:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-04915);
    case 4916:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-04916);
    case 4917:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-04917);
    case 4918:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-04918);
    case 4919:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-04919);
    case 4920:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-04920);
    case 4921:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-04921);
    case 4922:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-04922);
    case 4923:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-04923);
    case 4924:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-04924);
    case 6201:
      return VUID_WRAP(VUID-StandaloneSpirv-Flat-06201);
    case 6202:
      return VUID_WRAP(VUID-StandaloneSpirv-Flat-06202);
    case 6214:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeImage-06214);
    case 6491:
      return VUID_WRAP(VUID-StandaloneSpirv-DescriptorSet-06491);
    case 6671:
      return VUID_WRAP(VUID-StandaloneSpirv-OpTypeSampledImage-06671);
    case 6672:
      return VUID_WRAP(VUID-StandaloneSpirv-Location-06672);
    case 6674:
      return VUID_WRAP(VUID-StandaloneSpirv-OpEntryPoint-06674);
    case 6675:
      return VUID_WRAP(VUID-StandaloneSpirv-PushConstant-06675);
    case 6676:
      return VUID_WRAP(VUID-StandaloneSpirv-Uniform-06676);
    case 6677:
      return VUID_WRAP(VUID-StandaloneSpirv-UniformConstant-06677);
    case 6678:
      return VUID_WRAP(VUID-StandaloneSpirv-InputAttachmentIndex-06678);
    case 6777:
      return VUID_WRAP(VUID-StandaloneSpirv-PerVertexKHR-06777);
    case 6778:
      return VUID_WRAP(VUID-StandaloneSpirv-Input-06778);
    case 6807:
      return VUID_WRAP(VUID-StandaloneSpirv-Uniform-06807);
    case 6808:
      return VUID_WRAP(VUID-StandaloneSpirv-PushConstant-06808);
    case 6925:
      return VUID_WRAP(VUID-StandaloneSpirv-Uniform-06925);
    case 6997:
      return VUID_WRAP(VUID-StandaloneSpirv-SubgroupVoteKHR-06997);
    case 7102:
      return VUID_WRAP(VUID-StandaloneSpirv-MeshEXT-07102);
    case 7320:
      return VUID_WRAP(VUID-StandaloneSpirv-ExecutionModel-07320);
    case 7290:
      return VUID_WRAP(VUID-StandaloneSpirv-Input-07290);
    case 7650:
      return VUID_WRAP(VUID-StandaloneSpirv-Base-07650);
    case 7651:
      return VUID_WRAP(VUID-StandaloneSpirv-Base-07651);
    case 7652:
      return VUID_WRAP(VUID-StandaloneSpirv-Base-07652);
    case 7703:
      return VUID_WRAP(VUID-StandaloneSpirv-Component-07703);
    default:
      return "";  // unknown id
  }
  // clang-format on
}

}  // namespace val
}  // namespace spvtools
