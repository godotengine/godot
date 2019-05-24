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
#include "source/spirv_target_env.h"
#include "source/val/basic_block.h"
#include "source/val/construct.h"
#include "source/val/function.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace val {
namespace {

bool IsInstructionInLayoutSection(ModuleLayoutSection layout, SpvOp op) {
  // See Section 2.4
  bool out = false;
  // clang-format off
  switch (layout) {
    case kLayoutCapabilities:  out = op == SpvOpCapability;    break;
    case kLayoutExtensions:    out = op == SpvOpExtension;     break;
    case kLayoutExtInstImport: out = op == SpvOpExtInstImport; break;
    case kLayoutMemoryModel:   out = op == SpvOpMemoryModel;   break;
    case kLayoutEntryPoint:    out = op == SpvOpEntryPoint;    break;
    case kLayoutExecutionMode:
      out = op == SpvOpExecutionMode || op == SpvOpExecutionModeId;
      break;
    case kLayoutDebug1:
      switch (op) {
        case SpvOpSourceContinued:
        case SpvOpSource:
        case SpvOpSourceExtension:
        case SpvOpString:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutDebug2:
      switch (op) {
        case SpvOpName:
        case SpvOpMemberName:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutDebug3:
      // Only OpModuleProcessed is allowed here.
      out = (op == SpvOpModuleProcessed);
      break;
    case kLayoutAnnotations:
      switch (op) {
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpGroupDecorate:
        case SpvOpGroupMemberDecorate:
        case SpvOpDecorationGroup:
        case SpvOpDecorateId:
        case SpvOpDecorateStringGOOGLE:
        case SpvOpMemberDecorateStringGOOGLE:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutTypes:
      if (spvOpcodeGeneratesType(op) || spvOpcodeIsConstant(op)) {
        out = true;
        break;
      }
      switch (op) {
        case SpvOpTypeForwardPointer:
        case SpvOpVariable:
        case SpvOpLine:
        case SpvOpNoLine:
        case SpvOpUndef:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutFunctionDeclarations:
    case kLayoutFunctionDefinitions:
      // NOTE: These instructions should NOT be in these layout sections
      if (spvOpcodeGeneratesType(op) || spvOpcodeIsConstant(op)) {
        out = false;
        break;
      }
      switch (op) {
        case SpvOpCapability:
        case SpvOpExtension:
        case SpvOpExtInstImport:
        case SpvOpMemoryModel:
        case SpvOpEntryPoint:
        case SpvOpExecutionMode:
        case SpvOpExecutionModeId:
        case SpvOpSourceContinued:
        case SpvOpSource:
        case SpvOpSourceExtension:
        case SpvOpString:
        case SpvOpName:
        case SpvOpMemberName:
        case SpvOpModuleProcessed:
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpGroupDecorate:
        case SpvOpGroupMemberDecorate:
        case SpvOpDecorationGroup:
        case SpvOpTypeForwardPointer:
          out = false;
          break;
      default:
        out = true;
        break;
      }
  }
  // clang-format on
  return out;
}

// Counts the number of instructions and functions in the file.
spv_result_t CountInstructions(void* user_data,
                               const spv_parsed_instruction_t* inst) {
  ValidationState_t& _ = *(reinterpret_cast<ValidationState_t*>(user_data));
  if (inst->opcode == SpvOpFunction) _.increment_total_functions();
  _.increment_total_instructions();

  return SPV_SUCCESS;
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
      addressing_model_(SpvAddressingModelMax),
      memory_model_(SpvMemoryModelMax),
      pointer_size_and_alignment_(0),
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

  switch (env) {
    case SPV_ENV_WEBGPU_0:
      features_.bans_op_undef = true;
      break;
    default:
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
    spvBinaryParse(&hijacked_context, this, words, num_words,
                   /* parsed_header = */ nullptr, CountInstructions,
                   /* diagnostic = */ nullptr);
    preallocateStorage();
  }

  friendly_mapper_ = spvtools::MakeUnique<spvtools::FriendlyNameMapper>(
      context_, words_, num_words_);
  name_mapper_ = friendly_mapper_->GetNameMapper();
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
  out << id << "[%" << id_name << "]";
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

bool ValidationState_t::IsOpcodeInCurrentLayoutSection(SpvOp op) {
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

void ValidationState_t::RegisterCapability(SpvCapability cap) {
  // Avoid redundant work.  Otherwise the recursion could induce work
  // quadrdatic in the capability dependency depth. (Ok, not much, but
  // it's something.)
  if (module_capabilities_.Contains(cap)) return;

  module_capabilities_.Add(cap);
  spv_operand_desc desc;
  if (SPV_SUCCESS ==
      grammar_.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc)) {
    CapabilitySet(desc->numCapabilities, desc->capabilities)
        .ForEach([this](SpvCapability c) { RegisterCapability(c); });
  }

  switch (cap) {
    case SpvCapabilityKernel:
      features_.group_ops_reduce_and_scans = true;
      break;
    case SpvCapabilityInt8:
      features_.use_int8_type = true;
      features_.declare_int8_type = true;
      break;
    case SpvCapabilityStorageBuffer8BitAccess:
    case SpvCapabilityUniformAndStorageBuffer8BitAccess:
    case SpvCapabilityStoragePushConstant8:
      features_.declare_int8_type = true;
      break;
    case SpvCapabilityInt16:
      features_.declare_int16_type = true;
      break;
    case SpvCapabilityFloat16:
    case SpvCapabilityFloat16Buffer:
      features_.declare_float16_type = true;
      break;
    case SpvCapabilityStorageUniformBufferBlock16:
    case SpvCapabilityStorageUniform16:
    case SpvCapabilityStoragePushConstant16:
    case SpvCapabilityStorageInputOutput16:
      features_.declare_int16_type = true;
      features_.declare_float16_type = true;
      features_.free_fp_rounding_mode = true;
      break;
    case SpvCapabilityVariablePointers:
      features_.variable_pointers = true;
      features_.variable_pointers_storage_buffer = true;
      break;
    case SpvCapabilityVariablePointersStorageBuffer:
      features_.variable_pointers_storage_buffer = true;
      break;
    default:
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

void ValidationState_t::set_addressing_model(SpvAddressingModel am) {
  addressing_model_ = am;
  switch (am) {
    case SpvAddressingModelPhysical32:
      pointer_size_and_alignment_ = 4;
      break;
    default:
      // fall through
    case SpvAddressingModelPhysical64:
    case SpvAddressingModelPhysicalStorageBuffer64EXT:
      pointer_size_and_alignment_ = 8;
      break;
  }
}

SpvAddressingModel ValidationState_t::addressing_model() const {
  return addressing_model_;
}

void ValidationState_t::set_memory_model(SpvMemoryModel mm) {
  memory_model_ = mm;
}

SpvMemoryModel ValidationState_t::memory_model() const { return memory_model_; }

spv_result_t ValidationState_t::RegisterFunction(
    uint32_t id, uint32_t ret_type_id, SpvFunctionControlMask function_control,
    uint32_t function_type_id) {
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
         "ouside of a block");
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
    case SpvOpName: {
      const auto target = inst->GetOperandAs<uint32_t>(0);
      const auto* str = reinterpret_cast<const char*>(inst->words().data() +
                                                      inst->operand(1).offset);
      AssignNameToId(target, str);
      break;
    }
    case SpvOpMemberName: {
      const auto target = inst->GetOperandAs<uint32_t>(0);
      const auto* str = reinterpret_cast<const char*>(inst->words().data() +
                                                      inst->operand(2).offset);
      AssignNameToId(target, str);
      break;
    }
    case SpvOpSourceContinued:
    case SpvOpSource:
    case SpvOpSourceExtension:
    case SpvOpString:
    case SpvOpLine:
    case SpvOpNoLine:
    default:
      break;
  }
}

void ValidationState_t::RegisterInstruction(Instruction* inst) {
  if (inst->id()) all_definitions_.insert(std::make_pair(inst->id(), inst));

  // If the instruction is using an OpTypeSampledImage as an operand, it should
  // be recorded. The validator will ensure that all usages of an
  // OpTypeSampledImage and its definition are in the same basic block.
  for (uint16_t i = 0; i < inst->operands().size(); ++i) {
    const spv_parsed_operand_t& operand = inst->operand(i);
    if (SPV_OPERAND_TYPE_ID == operand.type) {
      const uint32_t operand_word = inst->word(operand.offset);
      Instruction* operand_inst = FindDef(operand_word);
      if (operand_inst && SpvOpSampledImage == operand_inst->opcode()) {
        RegisterSampledImageConsumer(operand_word, inst);
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

SpvOp ValidationState_t::GetIdOpcode(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  return inst ? inst->opcode() : SpvOpNop;
}

uint32_t ValidationState_t::GetComponentType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  switch (inst->opcode()) {
    case SpvOpTypeFloat:
    case SpvOpTypeInt:
    case SpvOpTypeBool:
      return id;

    case SpvOpTypeVector:
      return inst->word(2);

    case SpvOpTypeMatrix:
      return GetComponentType(inst->word(2));

    case SpvOpTypeCooperativeMatrixNV:
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
    case SpvOpTypeFloat:
    case SpvOpTypeInt:
    case SpvOpTypeBool:
      return 1;

    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
      return inst->word(3);

    case SpvOpTypeCooperativeMatrixNV:
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

  if (inst->opcode() == SpvOpTypeFloat || inst->opcode() == SpvOpTypeInt)
    return inst->word(2);

  if (inst->opcode() == SpvOpTypeBool) return 1;

  assert(0);
  return 0;
}

bool ValidationState_t::IsFloatScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeFloat;
}

bool ValidationState_t::IsFloatVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeVector) {
    return IsFloatScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsFloatScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeFloat) {
    return true;
  }

  if (inst->opcode() == SpvOpTypeVector) {
    return IsFloatScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeInt;
}

bool ValidationState_t::IsIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeVector) {
    return IsIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsIntScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeInt) {
    return true;
  }

  if (inst->opcode() == SpvOpTypeVector) {
    return IsIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsUnsignedIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeInt && inst->word(3) == 0;
}

bool ValidationState_t::IsUnsignedIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeVector) {
    return IsUnsignedIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsSignedIntScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeInt && inst->word(3) == 1;
}

bool ValidationState_t::IsSignedIntVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeVector) {
    return IsSignedIntScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsBoolScalarType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeBool;
}

bool ValidationState_t::IsBoolVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeVector) {
    return IsBoolScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsBoolScalarOrVectorType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeBool) {
    return true;
  }

  if (inst->opcode() == SpvOpTypeVector) {
    return IsBoolScalarType(GetComponentType(id));
  }

  return false;
}

bool ValidationState_t::IsFloatMatrixType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeMatrix) {
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
  if (mat_inst->opcode() != SpvOpTypeMatrix) return false;

  const uint32_t vec_type = mat_inst->word(2);
  const Instruction* vec_inst = FindDef(vec_type);
  assert(vec_inst);

  if (vec_inst->opcode() != SpvOpTypeVector) {
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
  if (inst->opcode() != SpvOpTypeStruct) return false;

  *member_types =
      std::vector<uint32_t>(inst->words().cbegin() + 2, inst->words().cend());

  if (member_types->empty()) return false;

  return true;
}

bool ValidationState_t::IsPointerType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypePointer;
}

bool ValidationState_t::GetPointerTypeInfo(uint32_t id, uint32_t* data_type,
                                           uint32_t* storage_class) const {
  if (!id) return false;

  const Instruction* inst = FindDef(id);
  assert(inst);
  if (inst->opcode() != SpvOpTypePointer) return false;

  *storage_class = inst->word(2);
  *data_type = inst->word(3);
  return true;
}

bool ValidationState_t::IsCooperativeMatrixType(uint32_t id) const {
  const Instruction* inst = FindDef(id);
  assert(inst);
  return inst->opcode() == SpvOpTypeCooperativeMatrixNV;
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

spv_result_t ValidationState_t::CooperativeMatrixShapesMatch(
    const Instruction* inst, uint32_t m1, uint32_t m2) {
  const auto m1_type = FindDef(m1);
  const auto m2_type = FindDef(m2);

  if (m1_type->opcode() != SpvOpTypeCooperativeMatrixNV ||
      m2_type->opcode() != SpvOpTypeCooperativeMatrixNV) {
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

  if (inst->opcode() != SpvOpConstant && inst->opcode() != SpvOpSpecConstant)
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

  if (inst->opcode() == SpvOpConstantNull) {
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
  for (const Function func : functions()) {
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

}  // namespace val
}  // namespace spvtools
