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

// Ensures type declarations are unique unless allowed by the specification.

#include "source/val/validate.h"

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// True if the integer constant is > 0. |const_words| are words of the
// constant-defining instruction (either OpConstant or
// OpSpecConstant). typeWords are the words of the constant's-type-defining
// OpTypeInt.
bool AboveZero(const std::vector<uint32_t>& const_words,
               const std::vector<uint32_t>& type_words) {
  const uint32_t width = type_words[2];
  const bool is_signed = type_words[3] > 0;
  const uint32_t lo_word = const_words[3];
  if (width > 32) {
    // The spec currently doesn't allow integers wider than 64 bits.
    const uint32_t hi_word = const_words[4];  // Must exist, per spec.
    if (is_signed && (hi_word >> 31)) return false;
    return (lo_word | hi_word) > 0;
  } else {
    if (is_signed && (lo_word >> 31)) return false;
    return lo_word > 0;
  }
}

// Validates that type declarations are unique, unless multiple declarations
// of the same data type are allowed by the specification.
// (see section 2.8 Types and Variables)
// Doesn't do anything if SPV_VAL_ignore_type_decl_unique was declared in the
// module.
spv_result_t ValidateUniqueness(ValidationState_t& _, const Instruction* inst) {
  if (_.HasExtension(Extension::kSPV_VALIDATOR_ignore_type_decl_unique))
    return SPV_SUCCESS;

  const auto opcode = inst->opcode();
  if (opcode != SpvOpTypeArray && opcode != SpvOpTypeRuntimeArray &&
      opcode != SpvOpTypeStruct && opcode != SpvOpTypePointer &&
      !_.RegisterUniqueTypeDeclaration(inst)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Duplicate non-aggregate type declarations are not allowed. "
              "Opcode: "
           << spvOpcodeString(opcode) << " id: " << inst->id();
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeVector(ValidationState_t& _, const Instruction* inst) {
  const auto component_index = 1;
  const auto component_id = inst->GetOperandAs<uint32_t>(component_index);
  const auto component_type = _.FindDef(component_id);
  if (!component_type || !spvOpcodeIsScalarType(component_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeVector Component Type <id> '" << _.getIdName(component_id)
           << "' is not a scalar type.";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTypeMatrix(ValidationState_t& _, const Instruction* inst) {
  const auto column_type_index = 1;
  const auto column_type_id = inst->GetOperandAs<uint32_t>(column_type_index);
  const auto column_type = _.FindDef(column_type_id);
  if (!column_type || SpvOpTypeVector != column_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeMatrix Column Type <id> '" << _.getIdName(column_type_id)
           << "' is not a vector.";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTypeArray(ValidationState_t& _, const Instruction* inst) {
  const auto element_type_index = 1;
  const auto element_type_id = inst->GetOperandAs<uint32_t>(element_type_index);
  const auto element_type = _.FindDef(element_type_id);
  if (!element_type || !spvOpcodeGeneratesType(element_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeArray Element Type <id> '" << _.getIdName(element_type_id)
           << "' is not a type.";
  }

  if (element_type->opcode() == SpvOpTypeVoid) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeArray Element Type <id> '" << _.getIdName(element_type_id)
           << "' is a void type.";
  }

  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env) &&
      element_type->opcode() == SpvOpTypeRuntimeArray) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeArray Element Type <id> '" << _.getIdName(element_type_id)
           << "' is not valid in "
           << spvLogStringForEnv(_.context()->target_env) << " environments.";
  }

  const auto length_index = 2;
  const auto length_id = inst->GetOperandAs<uint32_t>(length_index);
  const auto length = _.FindDef(length_id);
  if (!length || !spvOpcodeIsConstant(length->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeArray Length <id> '" << _.getIdName(length_id)
           << "' is not a scalar constant type.";
  }

  // NOTE: Check the initialiser value of the constant
  const auto const_inst = length->words();
  const auto const_result_type_index = 1;
  const auto const_result_type = _.FindDef(const_inst[const_result_type_index]);
  if (!const_result_type || SpvOpTypeInt != const_result_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeArray Length <id> '" << _.getIdName(length_id)
           << "' is not a constant integer type.";
  }

  switch (length->opcode()) {
    case SpvOpSpecConstant:
    case SpvOpConstant:
      if (AboveZero(length->words(), const_result_type->words())) break;
    // Else fall through!
    case SpvOpConstantNull: {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpTypeArray Length <id> '" << _.getIdName(length_id)
             << "' default value must be at least 1.";
    }
    case SpvOpSpecConstantOp:
      // Assume it's OK, rather than try to evaluate the operation.
      break;
    default:
      assert(0 && "bug in spvOpcodeIsConstant() or result type isn't int");
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTypeRuntimeArray(ValidationState_t& _,
                                      const Instruction* inst) {
  const auto element_type_index = 1;
  const auto element_id = inst->GetOperandAs<uint32_t>(element_type_index);
  const auto element_type = _.FindDef(element_id);
  if (!element_type || !spvOpcodeGeneratesType(element_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeRuntimeArray Element Type <id> '"
           << _.getIdName(element_id) << "' is not a type.";
  }

  if (element_type->opcode() == SpvOpTypeVoid) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeRuntimeArray Element Type <id> '"
           << _.getIdName(element_id) << "' is a void type.";
  }

  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env) &&
      element_type->opcode() == SpvOpTypeRuntimeArray) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeRuntimeArray Element Type <id> '"
           << _.getIdName(element_id) << "' is not valid in "
           << spvLogStringForEnv(_.context()->target_env) << " environments.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeStruct(ValidationState_t& _, const Instruction* inst) {
  const uint32_t struct_id = inst->GetOperandAs<uint32_t>(0);
  for (size_t member_type_index = 1;
       member_type_index < inst->operands().size(); ++member_type_index) {
    auto member_type_id = inst->GetOperandAs<uint32_t>(member_type_index);
    auto member_type = _.FindDef(member_type_id);
    if (!member_type || !spvOpcodeGeneratesType(member_type->opcode())) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpTypeStruct Member Type <id> '" << _.getIdName(member_type_id)
             << "' is not a type.";
    }
    if (member_type->opcode() == SpvOpTypeVoid) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Structures cannot contain a void type.";
    }
    if (SpvOpTypeStruct == member_type->opcode() &&
        _.IsStructTypeWithBuiltInMember(member_type_id)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Structure <id> " << _.getIdName(member_type_id)
             << " contains members with BuiltIn decoration. Therefore this "
             << "structure may not be contained as a member of another "
             << "structure "
             << "type. Structure <id> " << _.getIdName(struct_id)
             << " contains structure <id> " << _.getIdName(member_type_id)
             << ".";
    }
    if (_.IsForwardPointer(member_type_id)) {
      // If we're dealing with a forward pointer:
      // Find out the type that the pointer is pointing to (must be struct)
      // word 3 is the <id> of the type being pointed to.
      auto type_pointing_to = _.FindDef(member_type->words()[3]);
      if (type_pointing_to && type_pointing_to->opcode() != SpvOpTypeStruct) {
        // Forward declared operands of a struct may only point to a struct.
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "A forward reference operand in an OpTypeStruct must be an "
                  "OpTypePointer that points to an OpTypeStruct. "
                  "Found OpTypePointer that points to Op"
               << spvOpcodeString(
                      static_cast<SpvOp>(type_pointing_to->opcode()))
               << ".";
      }
    }

    if (spvIsVulkanOrWebGPUEnv(_.context()->target_env) &&
        member_type->opcode() == SpvOpTypeRuntimeArray) {
      const bool is_last_member =
          member_type_index == inst->operands().size() - 1;
      if (!is_last_member) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "In " << spvLogStringForEnv(_.context()->target_env)
               << ", OpTypeRuntimeArray must only be used for the last member "
                  "of an OpTypeStruct";
      }
    }
  }

  bool has_nested_blockOrBufferBlock_struct = false;
  // Struct members start at word 2 of OpTypeStruct instruction.
  for (size_t word_i = 2; word_i < inst->words().size(); ++word_i) {
    auto member = inst->word(word_i);
    auto memberTypeInstr = _.FindDef(member);
    if (memberTypeInstr && SpvOpTypeStruct == memberTypeInstr->opcode()) {
      if (_.HasDecoration(memberTypeInstr->id(), SpvDecorationBlock) ||
          _.HasDecoration(memberTypeInstr->id(), SpvDecorationBufferBlock) ||
          _.GetHasNestedBlockOrBufferBlockStruct(memberTypeInstr->id()))
        has_nested_blockOrBufferBlock_struct = true;
    }
  }

  _.SetHasNestedBlockOrBufferBlockStruct(inst->id(),
                                         has_nested_blockOrBufferBlock_struct);
  if (_.GetHasNestedBlockOrBufferBlockStruct(inst->id()) &&
      (_.HasDecoration(inst->id(), SpvDecorationBufferBlock) ||
       _.HasDecoration(inst->id(), SpvDecorationBlock))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "rules: A Block or BufferBlock cannot be nested within another "
              "Block or BufferBlock. ";
  }

  std::unordered_set<uint32_t> built_in_members;
  for (auto decoration : _.id_decorations(struct_id)) {
    if (decoration.dec_type() == SpvDecorationBuiltIn &&
        decoration.struct_member_index() != Decoration::kInvalidMember) {
      built_in_members.insert(decoration.struct_member_index());
    }
  }
  int num_struct_members = static_cast<int>(inst->operands().size() - 1);
  int num_builtin_members = static_cast<int>(built_in_members.size());
  if (num_builtin_members > 0 && num_builtin_members != num_struct_members) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "When BuiltIn decoration is applied to a structure-type member, "
           << "all members of that structure type must also be decorated with "
           << "BuiltIn (No allowed mixing of built-in variables and "
           << "non-built-in variables within a single structure). Structure id "
           << struct_id << " does not meet this requirement.";
  }
  if (num_builtin_members > 0) {
    _.RegisterStructTypeWithBuiltInMember(struct_id);
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTypePointer(ValidationState_t& _,
                                 const Instruction* inst) {
  auto type_id = inst->GetOperandAs<uint32_t>(2);
  auto type = _.FindDef(type_id);
  if (!type || !spvOpcodeGeneratesType(type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypePointer Type <id> '" << _.getIdName(type_id)
           << "' is not a type.";
  }
  // See if this points to a storage image.
  const auto storage_class = inst->GetOperandAs<SpvStorageClass>(1);
  if (storage_class == SpvStorageClassUniformConstant) {
    // Unpack an optional level of arraying.
    if (type->opcode() == SpvOpTypeArray ||
        type->opcode() == SpvOpTypeRuntimeArray) {
      type_id = type->GetOperandAs<uint32_t>(1);
      type = _.FindDef(type_id);
    }
    if (type->opcode() == SpvOpTypeImage) {
      const auto sampled = type->GetOperandAs<uint32_t>(6);
      // 2 indicates this image is known to be be used without a sampler, i.e.
      // a storage image.
      if (sampled == 2) _.RegisterPointerToStorageImage(inst->id());
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateTypeFunction(ValidationState_t& _,
                                  const Instruction* inst) {
  const auto return_type_id = inst->GetOperandAs<uint32_t>(1);
  const auto return_type = _.FindDef(return_type_id);
  if (!return_type || !spvOpcodeGeneratesType(return_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeFunction Return Type <id> '" << _.getIdName(return_type_id)
           << "' is not a type.";
  }
  size_t num_args = 0;
  for (size_t param_type_index = 2; param_type_index < inst->operands().size();
       ++param_type_index, ++num_args) {
    const auto param_id = inst->GetOperandAs<uint32_t>(param_type_index);
    const auto param_type = _.FindDef(param_id);
    if (!param_type || !spvOpcodeGeneratesType(param_type->opcode())) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpTypeFunction Parameter Type <id> '" << _.getIdName(param_id)
             << "' is not a type.";
    }

    if (param_type->opcode() == SpvOpTypeVoid) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpTypeFunction Parameter Type <id> '" << _.getIdName(param_id)
             << "' cannot be OpTypeVoid.";
    }
  }
  const uint32_t num_function_args_limit =
      _.options()->universal_limits_.max_function_args;
  if (num_args > num_function_args_limit) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeFunction may not take more than "
           << num_function_args_limit << " arguments. OpTypeFunction <id> '"
           << _.getIdName(inst->GetOperandAs<uint32_t>(0)) << "' has "
           << num_args << " arguments.";
  }

  // The only valid uses of OpTypeFunction are in an OpFunction, debugging, or
  // decoration instruction.
  for (auto& pair : inst->uses()) {
    const auto* use = pair.first;
    if (use->opcode() != SpvOpFunction && !spvOpcodeIsDebug(use->opcode()) &&
        !spvOpcodeIsDecoration(use->opcode())) {
      return _.diag(SPV_ERROR_INVALID_ID, use)
             << "Invalid use of function type result id "
             << _.getIdName(inst->id()) << ".";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeForwardPointer(ValidationState_t& _,
                                        const Instruction* inst) {
  const auto pointer_type_id = inst->GetOperandAs<uint32_t>(0);
  const auto pointer_type_inst = _.FindDef(pointer_type_id);
  if (pointer_type_inst->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Pointer type in OpTypeForwardPointer is not a pointer type.";
  }

  if (inst->GetOperandAs<uint32_t>(1) !=
      pointer_type_inst->GetOperandAs<uint32_t>(1)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Storage class in OpTypeForwardPointer does not match the "
           << "pointer definition.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeCooperativeMatrixNV(ValidationState_t& _,
                                             const Instruction* inst) {
  const auto component_type_index = 1;
  const auto component_type_id =
      inst->GetOperandAs<uint32_t>(component_type_index);
  const auto component_type = _.FindDef(component_type_id);
  if (!component_type || (SpvOpTypeFloat != component_type->opcode() &&
                          SpvOpTypeInt != component_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeCooperativeMatrixNV Component Type <id> '"
           << _.getIdName(component_type_id)
           << "' is not a scalar numerical type.";
  }

  const auto scope_index = 2;
  const auto scope_id = inst->GetOperandAs<uint32_t>(scope_index);
  const auto scope = _.FindDef(scope_id);
  if (!scope || !_.IsIntScalarType(scope->type_id()) ||
      !spvOpcodeIsConstant(scope->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeCooperativeMatrixNV Scope <id> '" << _.getIdName(scope_id)
           << "' is not a constant instruction with scalar integer type.";
  }

  const auto rows_index = 3;
  const auto rows_id = inst->GetOperandAs<uint32_t>(rows_index);
  const auto rows = _.FindDef(rows_id);
  if (!rows || !_.IsIntScalarType(rows->type_id()) ||
      !spvOpcodeIsConstant(rows->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeCooperativeMatrixNV Rows <id> '" << _.getIdName(rows_id)
           << "' is not a constant instruction with scalar integer type.";
  }

  const auto cols_index = 4;
  const auto cols_id = inst->GetOperandAs<uint32_t>(cols_index);
  const auto cols = _.FindDef(cols_id);
  if (!cols || !_.IsIntScalarType(cols->type_id()) ||
      !spvOpcodeIsConstant(cols->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpTypeCooperativeMatrixNV Cols <id> '" << _.getIdName(rows_id)
           << "' is not a constant instruction with scalar integer type.";
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t TypePass(ValidationState_t& _, const Instruction* inst) {
  if (!spvOpcodeGeneratesType(inst->opcode()) &&
      inst->opcode() != SpvOpTypeForwardPointer) {
    return SPV_SUCCESS;
  }

  if (auto error = ValidateUniqueness(_, inst)) return error;

  switch (inst->opcode()) {
    case SpvOpTypeVector:
      if (auto error = ValidateTypeVector(_, inst)) return error;
      break;
    case SpvOpTypeMatrix:
      if (auto error = ValidateTypeMatrix(_, inst)) return error;
      break;
    case SpvOpTypeArray:
      if (auto error = ValidateTypeArray(_, inst)) return error;
      break;
    case SpvOpTypeRuntimeArray:
      if (auto error = ValidateTypeRuntimeArray(_, inst)) return error;
      break;
    case SpvOpTypeStruct:
      if (auto error = ValidateTypeStruct(_, inst)) return error;
      break;
    case SpvOpTypePointer:
      if (auto error = ValidateTypePointer(_, inst)) return error;
      break;
    case SpvOpTypeFunction:
      if (auto error = ValidateTypeFunction(_, inst)) return error;
      break;
    case SpvOpTypeForwardPointer:
      if (auto error = ValidateTypeForwardPointer(_, inst)) return error;
      break;
    case SpvOpTypeCooperativeMatrixNV:
      if (auto error = ValidateTypeCooperativeMatrixNV(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
