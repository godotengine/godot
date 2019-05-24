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

#include "source/val/validate.h"

#include <algorithm>
#include <string>
#include <vector>

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

bool AreLayoutCompatibleStructs(ValidationState_t&, const Instruction*,
                                const Instruction*);
bool HaveLayoutCompatibleMembers(ValidationState_t&, const Instruction*,
                                 const Instruction*);
bool HaveSameLayoutDecorations(ValidationState_t&, const Instruction*,
                               const Instruction*);
bool HasConflictingMemberOffsets(const std::vector<Decoration>&,
                                 const std::vector<Decoration>&);

bool IsAllowedTypeOrArrayOfSame(ValidationState_t& _, const Instruction* type,
                                std::initializer_list<uint32_t> allowed) {
  if (std::find(allowed.begin(), allowed.end(), type->opcode()) !=
      allowed.end()) {
    return true;
  }
  if (type->opcode() == SpvOpTypeArray ||
      type->opcode() == SpvOpTypeRuntimeArray) {
    auto elem_type = _.FindDef(type->word(2));
    return std::find(allowed.begin(), allowed.end(), elem_type->opcode()) !=
           allowed.end();
  }
  return false;
}

// Returns true if the two instructions represent structs that, as far as the
// validator can tell, have the exact same data layout.
bool AreLayoutCompatibleStructs(ValidationState_t& _, const Instruction* type1,
                                const Instruction* type2) {
  if (type1->opcode() != SpvOpTypeStruct) {
    return false;
  }
  if (type2->opcode() != SpvOpTypeStruct) {
    return false;
  }

  if (!HaveLayoutCompatibleMembers(_, type1, type2)) return false;

  return HaveSameLayoutDecorations(_, type1, type2);
}

// Returns true if the operands to the OpTypeStruct instruction defining the
// types are the same or are layout compatible types. |type1| and |type2| must
// be OpTypeStruct instructions.
bool HaveLayoutCompatibleMembers(ValidationState_t& _, const Instruction* type1,
                                 const Instruction* type2) {
  assert(type1->opcode() == SpvOpTypeStruct &&
         "type1 must be an OpTypeStruct instruction.");
  assert(type2->opcode() == SpvOpTypeStruct &&
         "type2 must be an OpTypeStruct instruction.");
  const auto& type1_operands = type1->operands();
  const auto& type2_operands = type2->operands();
  if (type1_operands.size() != type2_operands.size()) {
    return false;
  }

  for (size_t operand = 2; operand < type1_operands.size(); ++operand) {
    if (type1->word(operand) != type2->word(operand)) {
      auto def1 = _.FindDef(type1->word(operand));
      auto def2 = _.FindDef(type2->word(operand));
      if (!AreLayoutCompatibleStructs(_, def1, def2)) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if all decorations that affect the data layout of the struct
// (like Offset), are the same for the two types. |type1| and |type2| must be
// OpTypeStruct instructions.
bool HaveSameLayoutDecorations(ValidationState_t& _, const Instruction* type1,
                               const Instruction* type2) {
  assert(type1->opcode() == SpvOpTypeStruct &&
         "type1 must be an OpTypeStruct instruction.");
  assert(type2->opcode() == SpvOpTypeStruct &&
         "type2 must be an OpTypeStruct instruction.");
  const std::vector<Decoration>& type1_decorations =
      _.id_decorations(type1->id());
  const std::vector<Decoration>& type2_decorations =
      _.id_decorations(type2->id());

  // TODO: Will have to add other check for arrays an matricies if we want to
  // handle them.
  if (HasConflictingMemberOffsets(type1_decorations, type2_decorations)) {
    return false;
  }

  return true;
}

bool HasConflictingMemberOffsets(
    const std::vector<Decoration>& type1_decorations,
    const std::vector<Decoration>& type2_decorations) {
  {
    // We are interested in conflicting decoration.  If a decoration is in one
    // list but not the other, then we will assume the code is correct.  We are
    // looking for things we know to be wrong.
    //
    // We do not have to traverse type2_decoration because, after traversing
    // type1_decorations, anything new will not be found in
    // type1_decoration.  Therefore, it cannot lead to a conflict.
    for (const Decoration& decoration : type1_decorations) {
      switch (decoration.dec_type()) {
        case SpvDecorationOffset: {
          // Since these affect the layout of the struct, they must be present
          // in both structs.
          auto compare = [&decoration](const Decoration& rhs) {
            if (rhs.dec_type() != SpvDecorationOffset) return false;
            return decoration.struct_member_index() ==
                   rhs.struct_member_index();
          };
          auto i = std::find_if(type2_decorations.begin(),
                                type2_decorations.end(), compare);
          if (i != type2_decorations.end() &&
              decoration.params().front() != i->params().front()) {
            return true;
          }
        } break;
        default:
          // This decoration does not affect the layout of the structure, so
          // just moving on.
          break;
      }
    }
  }
  return false;
}

// If |skip_builtin| is true, returns true if |storage| contains bool within
// it and no storage that contains the bool is builtin.
// If |skip_builtin| is false, returns true if |storage| contains bool within
// it.
bool ContainsInvalidBool(ValidationState_t& _, const Instruction* storage,
                         bool skip_builtin) {
  if (skip_builtin) {
    for (const Decoration& decoration : _.id_decorations(storage->id())) {
      if (decoration.dec_type() == SpvDecorationBuiltIn) return false;
    }
  }

  const size_t elem_type_index = 1;
  uint32_t elem_type_id;
  Instruction* elem_type;

  switch (storage->opcode()) {
    case SpvOpTypeBool:
      return true;
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
      elem_type_id = storage->GetOperandAs<uint32_t>(elem_type_index);
      elem_type = _.FindDef(elem_type_id);
      return ContainsInvalidBool(_, elem_type, skip_builtin);
    case SpvOpTypeStruct:
      for (size_t member_type_index = 1;
           member_type_index < storage->operands().size();
           ++member_type_index) {
        auto member_type_id =
            storage->GetOperandAs<uint32_t>(member_type_index);
        auto member_type = _.FindDef(member_type_id);
        if (ContainsInvalidBool(_, member_type, skip_builtin)) return true;
      }
    default:
      break;
  }
  return false;
}

bool ContainsCooperativeMatrix(ValidationState_t& _,
                               const Instruction* storage) {
  const size_t elem_type_index = 1;
  uint32_t elem_type_id;
  Instruction* elem_type;

  switch (storage->opcode()) {
    case SpvOpTypeCooperativeMatrixNV:
      return true;
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
      elem_type_id = storage->GetOperandAs<uint32_t>(elem_type_index);
      elem_type = _.FindDef(elem_type_id);
      return ContainsCooperativeMatrix(_, elem_type);
    case SpvOpTypeStruct:
      for (size_t member_type_index = 1;
           member_type_index < storage->operands().size();
           ++member_type_index) {
        auto member_type_id =
            storage->GetOperandAs<uint32_t>(member_type_index);
        auto member_type = _.FindDef(member_type_id);
        if (ContainsCooperativeMatrix(_, member_type)) return true;
      }
      break;
    default:
      break;
  }
  return false;
}

std::pair<SpvStorageClass, SpvStorageClass> GetStorageClass(
    ValidationState_t& _, const Instruction* inst) {
  SpvStorageClass dst_sc = SpvStorageClassMax;
  SpvStorageClass src_sc = SpvStorageClassMax;
  switch (inst->opcode()) {
    case SpvOpCooperativeMatrixLoadNV:
    case SpvOpLoad: {
      auto load_pointer = _.FindDef(inst->GetOperandAs<uint32_t>(2));
      auto load_pointer_type = _.FindDef(load_pointer->type_id());
      dst_sc = load_pointer_type->GetOperandAs<SpvStorageClass>(1);
      break;
    }
    case SpvOpCooperativeMatrixStoreNV:
    case SpvOpStore: {
      auto store_pointer = _.FindDef(inst->GetOperandAs<uint32_t>(0));
      auto store_pointer_type = _.FindDef(store_pointer->type_id());
      dst_sc = store_pointer_type->GetOperandAs<SpvStorageClass>(1);
      break;
    }
    case SpvOpCopyMemory:
    case SpvOpCopyMemorySized: {
      auto dst = _.FindDef(inst->GetOperandAs<uint32_t>(0));
      auto dst_type = _.FindDef(dst->type_id());
      dst_sc = dst_type->GetOperandAs<SpvStorageClass>(1);
      auto src = _.FindDef(inst->GetOperandAs<uint32_t>(1));
      auto src_type = _.FindDef(src->type_id());
      src_sc = src_type->GetOperandAs<SpvStorageClass>(1);
      break;
    }
    default:
      break;
  }

  return std::make_pair(dst_sc, src_sc);
}

// This function is only called for OpLoad, OpStore, OpCopyMemory and
// OpCopyMemorySized, OpCooperativeMatrixLoadNV, and
// OpCooperativeMatrixStoreNV.
uint32_t GetMakeAvailableScope(const Instruction* inst, uint32_t mask) {
  uint32_t offset = 1;
  if (mask & SpvMemoryAccessAlignedMask) ++offset;

  uint32_t scope_id = 0;
  switch (inst->opcode()) {
    case SpvOpLoad:
    case SpvOpCopyMemorySized:
      return inst->GetOperandAs<uint32_t>(3 + offset);
    case SpvOpStore:
    case SpvOpCopyMemory:
      return inst->GetOperandAs<uint32_t>(2 + offset);
    case SpvOpCooperativeMatrixLoadNV:
      return inst->GetOperandAs<uint32_t>(5 + offset);
    case SpvOpCooperativeMatrixStoreNV:
      return inst->GetOperandAs<uint32_t>(4 + offset);
    default:
      assert(false && "unexpected opcode");
      break;
  }

  return scope_id;
}

// This function is only called for OpLoad, OpStore, OpCopyMemory,
// OpCopyMemorySized, OpCooperativeMatrixLoadNV, and
// OpCooperativeMatrixStoreNV.
uint32_t GetMakeVisibleScope(const Instruction* inst, uint32_t mask) {
  uint32_t offset = 1;
  if (mask & SpvMemoryAccessAlignedMask) ++offset;
  if (mask & SpvMemoryAccessMakePointerAvailableKHRMask) ++offset;

  uint32_t scope_id = 0;
  switch (inst->opcode()) {
    case SpvOpLoad:
    case SpvOpCopyMemorySized:
      return inst->GetOperandAs<uint32_t>(3 + offset);
    case SpvOpStore:
    case SpvOpCopyMemory:
      return inst->GetOperandAs<uint32_t>(2 + offset);
    case SpvOpCooperativeMatrixLoadNV:
      return inst->GetOperandAs<uint32_t>(5 + offset);
    case SpvOpCooperativeMatrixStoreNV:
      return inst->GetOperandAs<uint32_t>(4 + offset);
    default:
      assert(false && "unexpected opcode");
      break;
  }

  return scope_id;
}

bool DoesStructContainRTA(const ValidationState_t& _, const Instruction* inst) {
  for (size_t member_index = 1; member_index < inst->operands().size();
       ++member_index) {
    const auto member_id = inst->GetOperandAs<uint32_t>(member_index);
    const auto member_type = _.FindDef(member_id);
    if (member_type->opcode() == SpvOpTypeRuntimeArray) return true;
  }
  return false;
}

spv_result_t CheckMemoryAccess(ValidationState_t& _, const Instruction* inst,
                               uint32_t index) {
  SpvStorageClass dst_sc, src_sc;
  std::tie(dst_sc, src_sc) = GetStorageClass(_, inst);
  if (inst->operands().size() <= index) {
    if (src_sc == SpvStorageClassPhysicalStorageBufferEXT ||
        dst_sc == SpvStorageClassPhysicalStorageBufferEXT) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Memory accesses with PhysicalStorageBufferEXT must use "
                "Aligned.";
    }
    return SPV_SUCCESS;
  }

  uint32_t mask = inst->GetOperandAs<uint32_t>(index);
  if (mask & SpvMemoryAccessMakePointerAvailableKHRMask) {
    if (inst->opcode() == SpvOpLoad ||
        inst->opcode() == SpvOpCooperativeMatrixLoadNV) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "MakePointerAvailableKHR cannot be used with OpLoad.";
    }

    if (!(mask & SpvMemoryAccessNonPrivatePointerKHRMask)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "NonPrivatePointerKHR must be specified if "
                "MakePointerAvailableKHR is specified.";
    }

    // Check the associated scope for MakeAvailableKHR.
    const auto available_scope = GetMakeAvailableScope(inst, mask);
    if (auto error = ValidateMemoryScope(_, inst, available_scope))
      return error;
  }

  if (mask & SpvMemoryAccessMakePointerVisibleKHRMask) {
    if (inst->opcode() == SpvOpStore ||
        inst->opcode() == SpvOpCooperativeMatrixStoreNV) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "MakePointerVisibleKHR cannot be used with OpStore.";
    }

    if (!(mask & SpvMemoryAccessNonPrivatePointerKHRMask)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "NonPrivatePointerKHR must be specified if "
             << "MakePointerVisibleKHR is specified.";
    }

    // Check the associated scope for MakeVisibleKHR.
    const auto visible_scope = GetMakeVisibleScope(inst, mask);
    if (auto error = ValidateMemoryScope(_, inst, visible_scope)) return error;
  }

  if (mask & SpvMemoryAccessNonPrivatePointerKHRMask) {
    if (dst_sc != SpvStorageClassUniform &&
        dst_sc != SpvStorageClassWorkgroup &&
        dst_sc != SpvStorageClassCrossWorkgroup &&
        dst_sc != SpvStorageClassGeneric && dst_sc != SpvStorageClassImage &&
        dst_sc != SpvStorageClassStorageBuffer &&
        dst_sc != SpvStorageClassPhysicalStorageBufferEXT) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "NonPrivatePointerKHR requires a pointer in Uniform, "
             << "Workgroup, CrossWorkgroup, Generic, Image or StorageBuffer "
             << "storage classes.";
    }
    if (src_sc != SpvStorageClassMax && src_sc != SpvStorageClassUniform &&
        src_sc != SpvStorageClassWorkgroup &&
        src_sc != SpvStorageClassCrossWorkgroup &&
        src_sc != SpvStorageClassGeneric && src_sc != SpvStorageClassImage &&
        src_sc != SpvStorageClassStorageBuffer &&
        src_sc != SpvStorageClassPhysicalStorageBufferEXT) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "NonPrivatePointerKHR requires a pointer in Uniform, "
             << "Workgroup, CrossWorkgroup, Generic, Image or StorageBuffer "
             << "storage classes.";
    }
  }

  if (!(mask & SpvMemoryAccessAlignedMask)) {
    if (src_sc == SpvStorageClassPhysicalStorageBufferEXT ||
        dst_sc == SpvStorageClassPhysicalStorageBufferEXT) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Memory accesses with PhysicalStorageBufferEXT must use "
                "Aligned.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateVariable(ValidationState_t& _, const Instruction* inst) {
  auto result_type = _.FindDef(inst->type_id());
  if (!result_type || result_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpVariable Result Type <id> '" << _.getIdName(inst->type_id())
           << "' is not a pointer type.";
  }

  const auto initializer_index = 3;
  const auto storage_class_index = 2;
  if (initializer_index < inst->operands().size()) {
    const auto initializer_id = inst->GetOperandAs<uint32_t>(initializer_index);
    const auto initializer = _.FindDef(initializer_id);
    const auto is_module_scope_var =
        initializer && (initializer->opcode() == SpvOpVariable) &&
        (initializer->GetOperandAs<SpvStorageClass>(storage_class_index) !=
         SpvStorageClassFunction);
    const auto is_constant =
        initializer && spvOpcodeIsConstant(initializer->opcode());
    if (!initializer || !(is_constant || is_module_scope_var)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpVariable Initializer <id> '" << _.getIdName(initializer_id)
             << "' is not a constant or module-scope variable.";
    }
  }

  const auto storage_class =
      inst->GetOperandAs<SpvStorageClass>(storage_class_index);
  if (storage_class != SpvStorageClassWorkgroup &&
      storage_class != SpvStorageClassCrossWorkgroup &&
      storage_class != SpvStorageClassPrivate &&
      storage_class != SpvStorageClassFunction &&
      storage_class != SpvStorageClassRayPayloadNV &&
      storage_class != SpvStorageClassIncomingRayPayloadNV &&
      storage_class != SpvStorageClassHitAttributeNV &&
      storage_class != SpvStorageClassCallableDataNV &&
      storage_class != SpvStorageClassIncomingCallableDataNV) {
    const auto storage_index = 2;
    const auto storage_id = result_type->GetOperandAs<uint32_t>(storage_index);
    const auto storage = _.FindDef(storage_id);
    bool storage_input_or_output = storage_class == SpvStorageClassInput ||
                                   storage_class == SpvStorageClassOutput;
    bool builtin = false;
    if (storage_input_or_output) {
      for (const Decoration& decoration : _.id_decorations(inst->id())) {
        if (decoration.dec_type() == SpvDecorationBuiltIn) {
          builtin = true;
          break;
        }
      }
    }
    if (!(storage_input_or_output && builtin) &&
        ContainsInvalidBool(_, storage, storage_input_or_output)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "If OpTypeBool is stored in conjunction with OpVariable, it "
             << "can only be used with non-externally visible shader Storage "
             << "Classes: Workgroup, CrossWorkgroup, Private, and Function";
    }
  }

  // SPIR-V 3.32.8: Check that pointer type and variable type have the same
  // storage class.
  const auto result_storage_class_index = 1;
  const auto result_storage_class =
      result_type->GetOperandAs<uint32_t>(result_storage_class_index);
  if (storage_class != result_storage_class) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "From SPIR-V spec, section 3.32.8 on OpVariable:\n"
           << "Its Storage Class operand must be the same as the Storage Class "
           << "operand of the result type.";
  }

  // Variable pointer related restrictions.
  const auto pointee = _.FindDef(result_type->word(3));
  if (_.addressing_model() == SpvAddressingModelLogical &&
      !_.options()->relax_logical_pointer) {
    // VariablePointersStorageBuffer is implied by VariablePointers.
    if (pointee->opcode() == SpvOpTypePointer) {
      if (!_.HasCapability(SpvCapabilityVariablePointersStorageBuffer)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "In Logical addressing, variables may not allocate a pointer "
               << "type";
      } else if (storage_class != SpvStorageClassFunction &&
                 storage_class != SpvStorageClassPrivate) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "In Logical addressing with variable pointers, variables "
               << "that allocate pointers must be in Function or Private "
               << "storage classes";
      }
    }
  }

  // Vulkan 14.5.1: Check type of PushConstant variables.
  // Vulkan 14.5.2: Check type of UniformConstant and Uniform variables.
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (storage_class == SpvStorageClassPushConstant) {
      if (!IsAllowedTypeOrArrayOfSame(_, pointee, {SpvOpTypeStruct})) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "PushConstant OpVariable <id> '" << _.getIdName(inst->id())
               << "' has illegal type.\n"
               << "From Vulkan spec, section 14.5.1:\n"
               << "Such variables must be typed as OpTypeStruct, "
               << "or an array of this type";
      }
    }

    if (storage_class == SpvStorageClassUniformConstant) {
      if (!IsAllowedTypeOrArrayOfSame(
              _, pointee,
              {SpvOpTypeImage, SpvOpTypeSampler, SpvOpTypeSampledImage,
               SpvOpTypeAccelerationStructureNV})) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "UniformConstant OpVariable <id> '" << _.getIdName(inst->id())
               << "' has illegal type.\n"
               << "From Vulkan spec, section 14.5.2:\n"
               << "Variables identified with the UniformConstant storage class "
               << "are used only as handles to refer to opaque resources. Such "
               << "variables must be typed as OpTypeImage, OpTypeSampler, "
               << "OpTypeSampledImage, OpTypeAccelerationStructureNV, "
               << "or an array of one of these types.";
      }
    }

    if (storage_class == SpvStorageClassUniform) {
      if (!IsAllowedTypeOrArrayOfSame(_, pointee, {SpvOpTypeStruct})) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Uniform OpVariable <id> '" << _.getIdName(inst->id())
               << "' has illegal type.\n"
               << "From Vulkan spec, section 14.5.2:\n"
               << "Variables identified with the Uniform storage class are "
               << "used to access transparent buffer backed resources. Such "
               << "variables must be typed as OpTypeStruct, or an array of "
               << "this type";
      }
    }
  }

  // WebGPU & Vulkan Appendix A: Check that if contains initializer, then
  // storage class is Output, Private, or Function.
  if (inst->operands().size() > 3 && storage_class != SpvStorageClassOutput &&
      storage_class != SpvStorageClassPrivate &&
      storage_class != SpvStorageClassFunction) {
    if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpVariable, <id> '" << _.getIdName(inst->id())
             << "', has a disallowed initializer & storage class "
             << "combination.\n"
             << "From " << spvLogStringForEnv(_.context()->target_env)
             << " spec:\n"
             << "Variable declarations that include initializers must have "
             << "one of the following storage classes: Output, Private, or "
             << "Function";
    }
  }

  // WebGPU: All variables with storage class Output, Private, or Function MUST
  // have an initializer.
  if (spvIsWebGPUEnv(_.context()->target_env) && inst->operands().size() <= 3 &&
      (storage_class == SpvStorageClassOutput ||
       storage_class == SpvStorageClassPrivate ||
       storage_class == SpvStorageClassFunction)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpVariable, <id> '" << _.getIdName(inst->id())
           << "', must have an initializer.\n"
           << "From WebGPU execution environment spec:\n"
           << "All variables in the following storage classes must have an "
           << "initializer: Output, Private, or Function";
  }

  if (storage_class == SpvStorageClassPhysicalStorageBufferEXT) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "PhysicalStorageBufferEXT must not be used with OpVariable.";
  }

  auto pointee_base = pointee;
  while (pointee_base->opcode() == SpvOpTypeArray) {
    pointee_base = _.FindDef(pointee_base->GetOperandAs<uint32_t>(1u));
  }
  if (pointee_base->opcode() == SpvOpTypePointer) {
    if (pointee_base->GetOperandAs<uint32_t>(1u) ==
        SpvStorageClassPhysicalStorageBufferEXT) {
      // check for AliasedPointerEXT/RestrictPointerEXT
      bool foundAliased =
          _.HasDecoration(inst->id(), SpvDecorationAliasedPointerEXT);
      bool foundRestrict =
          _.HasDecoration(inst->id(), SpvDecorationRestrictPointerEXT);
      if (!foundAliased && !foundRestrict) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpVariable " << inst->id()
               << ": expected AliasedPointerEXT or RestrictPointerEXT for "
               << "PhysicalStorageBufferEXT pointer.";
      }
      if (foundAliased && foundRestrict) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpVariable " << inst->id()
               << ": can't specify both AliasedPointerEXT and "
               << "RestrictPointerEXT for PhysicalStorageBufferEXT pointer.";
      }
    }
  }

  // Vulkan specific validation rules for OpTypeRuntimeArray
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const auto type_index = 2;
    const auto value_id = result_type->GetOperandAs<uint32_t>(type_index);
    auto value_type = _.FindDef(value_id);
    // OpTypeRuntimeArray should only ever be in a container like OpTypeStruct,
    // so should never appear as a bare variable.
    // Unless the module has the RuntimeDescriptorArrayEXT capability.
    if (value_type && value_type->opcode() == SpvOpTypeRuntimeArray) {
      if (!_.HasCapability(SpvCapabilityRuntimeDescriptorArrayEXT)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpVariable, <id> '" << _.getIdName(inst->id())
               << "', is attempting to create memory for an illegal type, "
               << "OpTypeRuntimeArray.\nFor Vulkan OpTypeRuntimeArray can only "
               << "appear as the final member of an OpTypeStruct, thus cannot "
               << "be instantiated via OpVariable";
      } else {
        // A bare variable OpTypeRuntimeArray is allowed in this context, but
        // still need to check the storage class.
        if (storage_class != SpvStorageClassStorageBuffer &&
            storage_class != SpvStorageClassUniform &&
            storage_class != SpvStorageClassUniformConstant) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "For Vulkan with RuntimeDescriptorArrayEXT, a variable "
                 << "containing OpTypeRuntimeArray must have storage class of "
                 << "StorageBuffer, Uniform, or UniformConstant.";
        }
      }
    }

    // If an OpStruct has an OpTypeRuntimeArray somewhere within it, then it
    // must either have the storage class StorageBuffer and be decorated
    // with Block, or it must be in the Uniform storage class and be decorated
    // as BufferBlock.
    if (value_type && value_type->opcode() == SpvOpTypeStruct) {
      if (DoesStructContainRTA(_, value_type)) {
        if (storage_class == SpvStorageClassStorageBuffer) {
          if (!_.HasDecoration(value_id, SpvDecorationBlock)) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "For Vulkan, an OpTypeStruct variable containing an "
                   << "OpTypeRuntimeArray must be decorated with Block if it "
                   << "has storage class StorageBuffer.";
          }
        } else if (storage_class == SpvStorageClassUniform) {
          if (!_.HasDecoration(value_id, SpvDecorationBufferBlock)) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "For Vulkan, an OpTypeStruct variable containing an "
                   << "OpTypeRuntimeArray must be decorated with BufferBlock "
                   << "if it has storage class Uniform.";
          }
        } else {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "For Vulkan, OpTypeStruct variables containing "
                 << "OpTypeRuntimeArray must have storage class of "
                 << "StorageBuffer or Uniform.";
        }
      }
    }
  }

  // WebGPU specific validation rules for OpTypeRuntimeArray
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    const auto type_index = 2;
    const auto value_id = result_type->GetOperandAs<uint32_t>(type_index);
    auto value_type = _.FindDef(value_id);
    // OpTypeRuntimeArray should only ever be in an OpTypeStruct,
    // so should never appear as a bare variable.
    if (value_type && value_type->opcode() == SpvOpTypeRuntimeArray) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpVariable, <id> '" << _.getIdName(inst->id())
             << "', is attempting to create memory for an illegal type, "
             << "OpTypeRuntimeArray.\nFor WebGPU OpTypeRuntimeArray can only "
             << "appear as the final member of an OpTypeStruct, thus cannot "
             << "be instantiated via OpVariable";
    }

    // If an OpStruct has an OpTypeRuntimeArray somewhere within it, then it
    // must have the storage class StorageBuffer and be decorated
    // with Block.
    if (value_type && value_type->opcode() == SpvOpTypeStruct) {
      if (DoesStructContainRTA(_, value_type)) {
        if (storage_class == SpvStorageClassStorageBuffer) {
          if (!_.HasDecoration(value_id, SpvDecorationBlock)) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "For WebGPU, an OpTypeStruct variable containing an "
                   << "OpTypeRuntimeArray must be decorated with Block if it "
                   << "has storage class StorageBuffer.";
          }
        } else {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "For WebGPU, OpTypeStruct variables containing "
                 << "OpTypeRuntimeArray must have storage class of "
                 << "StorageBuffer";
        }
      }
    }
  }

  // Cooperative matrix types can only be allocated in Function or Private
  if ((storage_class != SpvStorageClassFunction &&
       storage_class != SpvStorageClassPrivate) &&
      ContainsCooperativeMatrix(_, pointee)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Cooperative matrix types (or types containing them) can only be "
              "allocated "
           << "in Function or Private storage classes or as function "
              "parameters";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateLoad(ValidationState_t& _, const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  if (!result_type) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpLoad Result Type <id> '" << _.getIdName(inst->type_id())
           << "' is not defined.";
  }

  const bool uses_variable_pointers =
      _.features().variable_pointers ||
      _.features().variable_pointers_storage_buffer;
  const auto pointer_index = 2;
  const auto pointer_id = inst->GetOperandAs<uint32_t>(pointer_index);
  const auto pointer = _.FindDef(pointer_id);
  if (!pointer ||
      ((_.addressing_model() == SpvAddressingModelLogical) &&
       ((!uses_variable_pointers &&
         !spvOpcodeReturnsLogicalPointer(pointer->opcode())) ||
        (uses_variable_pointers &&
         !spvOpcodeReturnsLogicalVariablePointer(pointer->opcode()))))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpLoad Pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a logical pointer.";
  }

  const auto pointer_type = _.FindDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpLoad type for pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a pointer type.";
  }

  const auto pointee_type = _.FindDef(pointer_type->GetOperandAs<uint32_t>(2));
  if (!pointee_type || result_type->id() != pointee_type->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpLoad Result Type <id> '" << _.getIdName(inst->type_id())
           << "' does not match Pointer <id> '" << _.getIdName(pointer->id())
           << "'s type.";
  }

  if (auto error = CheckMemoryAccess(_, inst, 3)) return error;

  return SPV_SUCCESS;
}

spv_result_t ValidateStore(ValidationState_t& _, const Instruction* inst) {
  const bool uses_variable_pointer =
      _.features().variable_pointers ||
      _.features().variable_pointers_storage_buffer;
  const auto pointer_index = 0;
  const auto pointer_id = inst->GetOperandAs<uint32_t>(pointer_index);
  const auto pointer = _.FindDef(pointer_id);
  if (!pointer ||
      (_.addressing_model() == SpvAddressingModelLogical &&
       ((!uses_variable_pointer &&
         !spvOpcodeReturnsLogicalPointer(pointer->opcode())) ||
        (uses_variable_pointer &&
         !spvOpcodeReturnsLogicalVariablePointer(pointer->opcode()))))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a logical pointer.";
  }
  const auto pointer_type = _.FindDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpStore type for pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a pointer type.";
  }
  const auto type_id = pointer_type->GetOperandAs<uint32_t>(2);
  const auto type = _.FindDef(type_id);
  if (!type || SpvOpTypeVoid == type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
           << "'s type is void.";
  }

  // validate storage class
  {
    uint32_t data_type;
    uint32_t storage_class;
    if (!_.GetPointerTypeInfo(pointer_type->id(), &data_type, &storage_class)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
             << "' is not pointer type";
    }

    if (storage_class == SpvStorageClassUniformConstant ||
        storage_class == SpvStorageClassInput ||
        storage_class == SpvStorageClassPushConstant) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
             << "' storage class is read-only";
    }
  }

  const auto object_index = 1;
  const auto object_id = inst->GetOperandAs<uint32_t>(object_index);
  const auto object = _.FindDef(object_id);
  if (!object || !object->type_id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpStore Object <id> '" << _.getIdName(object_id)
           << "' is not an object.";
  }
  const auto object_type = _.FindDef(object->type_id());
  if (!object_type || SpvOpTypeVoid == object_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpStore Object <id> '" << _.getIdName(object_id)
           << "'s type is void.";
  }

  if (type->id() != object_type->id()) {
    if (!_.options()->relax_struct_store || type->opcode() != SpvOpTypeStruct ||
        object_type->opcode() != SpvOpTypeStruct) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
             << "'s type does not match Object <id> '"
             << _.getIdName(object->id()) << "'s type.";
    }

    // TODO: Check for layout compatible matricies and arrays as well.
    if (!AreLayoutCompatibleStructs(_, type, object_type)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpStore Pointer <id> '" << _.getIdName(pointer_id)
             << "'s layout does not match Object <id> '"
             << _.getIdName(object->id()) << "'s layout.";
    }
  }

  if (auto error = CheckMemoryAccess(_, inst, 2)) return error;

  return SPV_SUCCESS;
}

spv_result_t ValidateCopyMemory(ValidationState_t& _, const Instruction* inst) {
  const auto target_index = 0;
  const auto target_id = inst->GetOperandAs<uint32_t>(target_index);
  const auto target = _.FindDef(target_id);
  if (!target) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Target operand <id> '" << _.getIdName(target_id)
           << "' is not defined.";
  }

  const auto source_index = 1;
  const auto source_id = inst->GetOperandAs<uint32_t>(source_index);
  const auto source = _.FindDef(source_id);
  if (!source) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Source operand <id> '" << _.getIdName(source_id)
           << "' is not defined.";
  }

  const auto target_pointer_type = _.FindDef(target->type_id());
  if (!target_pointer_type ||
      target_pointer_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Target operand <id> '" << _.getIdName(target_id)
           << "' is not a pointer.";
  }

  const auto source_pointer_type = _.FindDef(source->type_id());
  if (!source_pointer_type ||
      source_pointer_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Source operand <id> '" << _.getIdName(source_id)
           << "' is not a pointer.";
  }

  if (inst->opcode() == SpvOpCopyMemory) {
    const auto target_type =
        _.FindDef(target_pointer_type->GetOperandAs<uint32_t>(2));
    if (!target_type || target_type->opcode() == SpvOpTypeVoid) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Target operand <id> '" << _.getIdName(target_id)
             << "' cannot be a void pointer.";
    }

    const auto source_type =
        _.FindDef(source_pointer_type->GetOperandAs<uint32_t>(2));
    if (!source_type || source_type->opcode() == SpvOpTypeVoid) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Source operand <id> '" << _.getIdName(source_id)
             << "' cannot be a void pointer.";
    }

    if (target_type->id() != source_type->id()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Target <id> '" << _.getIdName(source_id)
             << "'s type does not match Source <id> '"
             << _.getIdName(source_type->id()) << "'s type.";
    }

    if (auto error = CheckMemoryAccess(_, inst, 2)) return error;
  } else {
    const auto size_id = inst->GetOperandAs<uint32_t>(2);
    const auto size = _.FindDef(size_id);
    if (!size) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Size operand <id> '" << _.getIdName(size_id)
             << "' is not defined.";
    }

    const auto size_type = _.FindDef(size->type_id());
    if (!_.IsIntScalarType(size_type->id())) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Size operand <id> '" << _.getIdName(size_id)
             << "' must be a scalar integer type.";
    }

    bool is_zero = true;
    switch (size->opcode()) {
      case SpvOpConstantNull:
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Size operand <id> '" << _.getIdName(size_id)
               << "' cannot be a constant zero.";
      case SpvOpConstant:
        if (size_type->word(3) == 1 &&
            size->word(size->words().size() - 1) & 0x80000000) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "Size operand <id> '" << _.getIdName(size_id)
                 << "' cannot have the sign bit set to 1.";
        }
        for (size_t i = 3; is_zero && i < size->words().size(); ++i) {
          is_zero &= (size->word(i) == 0);
        }
        if (is_zero) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "Size operand <id> '" << _.getIdName(size_id)
                 << "' cannot be a constant zero.";
        }
        break;
      default:
        // Cannot infer any other opcodes.
        break;
    }

    if (auto error = CheckMemoryAccess(_, inst, 3)) return error;
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateAccessChain(ValidationState_t& _,
                                 const Instruction* inst) {
  std::string instr_name =
      "Op" + std::string(spvOpcodeString(static_cast<SpvOp>(inst->opcode())));

  // The result type must be OpTypePointer.
  auto result_type = _.FindDef(inst->type_id());
  if (SpvOpTypePointer != result_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Result Type of " << instr_name << " <id> '"
           << _.getIdName(inst->id()) << "' must be OpTypePointer. Found Op"
           << spvOpcodeString(static_cast<SpvOp>(result_type->opcode())) << ".";
  }

  // Result type is a pointer. Find out what it's pointing to.
  // This will be used to make sure the indexing results in the same type.
  // OpTypePointer word 3 is the type being pointed to.
  const auto result_type_pointee = _.FindDef(result_type->word(3));

  // Base must be a pointer, pointing to the base of a composite object.
  const auto base_index = 2;
  const auto base_id = inst->GetOperandAs<uint32_t>(base_index);
  const auto base = _.FindDef(base_id);
  const auto base_type = _.FindDef(base->type_id());
  if (!base_type || SpvOpTypePointer != base_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Base <id> '" << _.getIdName(base_id) << "' in " << instr_name
           << " instruction must be a pointer.";
  }

  // The result pointer storage class and base pointer storage class must match.
  // Word 2 of OpTypePointer is the Storage Class.
  auto result_type_storage_class = result_type->word(2);
  auto base_type_storage_class = base_type->word(2);
  if (result_type_storage_class != base_type_storage_class) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The result pointer storage class and base "
              "pointer storage class in "
           << instr_name << " do not match.";
  }

  // The type pointed to by OpTypePointer (word 3) must be a composite type.
  auto type_pointee = _.FindDef(base_type->word(3));

  // Check Universal Limit (SPIR-V Spec. Section 2.17).
  // The number of indexes passed to OpAccessChain may not exceed 255
  // The instruction includes 4 words + N words (for N indexes)
  size_t num_indexes = inst->words().size() - 4;
  if (inst->opcode() == SpvOpPtrAccessChain ||
      inst->opcode() == SpvOpInBoundsPtrAccessChain) {
    // In pointer access chains, the element operand is required, but not
    // counted as an index.
    --num_indexes;
  }
  const size_t num_indexes_limit =
      _.options()->universal_limits_.max_access_chain_indexes;
  if (num_indexes > num_indexes_limit) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The number of indexes in " << instr_name << " may not exceed "
           << num_indexes_limit << ". Found " << num_indexes << " indexes.";
  }
  // Indexes walk the type hierarchy to the desired depth, potentially down to
  // scalar granularity. The first index in Indexes will select the top-level
  // member/element/component/element of the base composite. All composite
  // constituents use zero-based numbering, as described by their OpType...
  // instruction. The second index will apply similarly to that result, and so
  // on. Once any non-composite type is reached, there must be no remaining
  // (unused) indexes.
  auto starting_index = 4;
  if (inst->opcode() == SpvOpPtrAccessChain ||
      inst->opcode() == SpvOpInBoundsPtrAccessChain) {
    ++starting_index;
  }
  for (size_t i = starting_index; i < inst->words().size(); ++i) {
    const uint32_t cur_word = inst->words()[i];
    // Earlier ID checks ensure that cur_word definition exists.
    auto cur_word_instr = _.FindDef(cur_word);
    // The index must be a scalar integer type (See OpAccessChain in the Spec.)
    auto index_type = _.FindDef(cur_word_instr->type_id());
    if (!index_type || SpvOpTypeInt != index_type->opcode()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Indexes passed to " << instr_name
             << " must be of type integer.";
    }
    switch (type_pointee->opcode()) {
      case SpvOpTypeMatrix:
      case SpvOpTypeVector:
      case SpvOpTypeCooperativeMatrixNV:
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray: {
        // In OpTypeMatrix, OpTypeVector, SpvOpTypeCooperativeMatrixNV,
        // OpTypeArray, and OpTypeRuntimeArray, word 2 is the Element Type.
        type_pointee = _.FindDef(type_pointee->word(2));
        break;
      }
      case SpvOpTypeStruct: {
        // In case of structures, there is an additional constraint on the
        // index: the index must be an OpConstant.
        if (SpvOpConstant != cur_word_instr->opcode()) {
          return _.diag(SPV_ERROR_INVALID_ID, cur_word_instr)
                 << "The <id> passed to " << instr_name
                 << " to index into a "
                    "structure must be an OpConstant.";
        }
        // Get the index value from the OpConstant (word 3 of OpConstant).
        // OpConstant could be a signed integer. But it's okay to treat it as
        // unsigned because a negative constant int would never be seen as
        // correct as a struct offset, since structs can't have more than 2
        // billion members.
        const uint32_t cur_index = cur_word_instr->word(3);
        // The index points to the struct member we want, therefore, the index
        // should be less than the number of struct members.
        const uint32_t num_struct_members =
            static_cast<uint32_t>(type_pointee->words().size() - 2);
        if (cur_index >= num_struct_members) {
          return _.diag(SPV_ERROR_INVALID_ID, cur_word_instr)
                 << "Index is out of bounds: " << instr_name
                 << " can not find index " << cur_index
                 << " into the structure <id> '"
                 << _.getIdName(type_pointee->id()) << "'. This structure has "
                 << num_struct_members << " members. Largest valid index is "
                 << num_struct_members - 1 << ".";
        }
        // Struct members IDs start at word 2 of OpTypeStruct.
        auto structMemberId = type_pointee->word(cur_index + 2);
        type_pointee = _.FindDef(structMemberId);
        break;
      }
      default: {
        // Give an error. reached non-composite type while indexes still remain.
        return _.diag(SPV_ERROR_INVALID_ID, cur_word_instr)
               << instr_name
               << " reached non-composite type while indexes "
                  "still remain to be traversed.";
      }
    }
  }
  // At this point, we have fully walked down from the base using the indeces.
  // The type being pointed to should be the same as the result type.
  if (type_pointee->id() != result_type_pointee->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << instr_name << " result type (Op"
           << spvOpcodeString(static_cast<SpvOp>(result_type_pointee->opcode()))
           << ") does not match the type that results from indexing into the "
              "base "
              "<id> (Op"
           << spvOpcodeString(static_cast<SpvOp>(type_pointee->opcode()))
           << ").";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidatePtrAccessChain(ValidationState_t& _,
                                    const Instruction* inst) {
  if (_.addressing_model() == SpvAddressingModelLogical) {
    if (!_.features().variable_pointers &&
        !_.features().variable_pointers_storage_buffer) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Generating variable pointers requires capability "
             << "VariablePointers or VariablePointersStorageBuffer";
    }
  }
  return ValidateAccessChain(_, inst);
}

spv_result_t ValidateArrayLength(ValidationState_t& state,
                                 const Instruction* inst) {
  std::string instr_name =
      "Op" + std::string(spvOpcodeString(static_cast<SpvOp>(inst->opcode())));

  // Result type must be a 32-bit unsigned int.
  auto result_type = state.FindDef(inst->type_id());
  if (result_type->opcode() != SpvOpTypeInt ||
      result_type->GetOperandAs<uint32_t>(1) != 32 ||
      result_type->GetOperandAs<uint32_t>(2) != 0) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Result Type of " << instr_name << " <id> '"
           << state.getIdName(inst->id())
           << "' must be OpTypeInt with width 32 and signedness 0.";
  }

  // The structure that is passed in must be an pointer to a structure, whose
  // last element is a runtime array.
  auto pointer = state.FindDef(inst->GetOperandAs<uint32_t>(2));
  auto pointer_type = state.FindDef(pointer->type_id());
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Struture's type in " << instr_name << " <id> '"
           << state.getIdName(inst->id())
           << "' must be a pointer to an OpTypeStruct.";
  }

  auto structure_type = state.FindDef(pointer_type->GetOperandAs<uint32_t>(2));
  if (structure_type->opcode() != SpvOpTypeStruct) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Struture's type in " << instr_name << " <id> '"
           << state.getIdName(inst->id())
           << "' must be a pointer to an OpTypeStruct.";
  }

  auto num_of_members = structure_type->operands().size() - 1;
  auto last_member =
      state.FindDef(structure_type->GetOperandAs<uint32_t>(num_of_members));
  if (last_member->opcode() != SpvOpTypeRuntimeArray) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Struture's last member in " << instr_name << " <id> '"
           << state.getIdName(inst->id()) << "' must be an OpTypeRuntimeArray.";
  }

  // The array member must the the index of the last element (the run time
  // array).
  if (inst->GetOperandAs<uint32_t>(3) != num_of_members - 1) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The array member in " << instr_name << " <id> '"
           << state.getIdName(inst->id())
           << "' must be an the last member of the struct.";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateCooperativeMatrixLengthNV(ValidationState_t& state,
                                               const Instruction* inst) {
  std::string instr_name =
      "Op" + std::string(spvOpcodeString(static_cast<SpvOp>(inst->opcode())));

  // Result type must be a 32-bit unsigned int.
  auto result_type = state.FindDef(inst->type_id());
  if (result_type->opcode() != SpvOpTypeInt ||
      result_type->GetOperandAs<uint32_t>(1) != 32 ||
      result_type->GetOperandAs<uint32_t>(2) != 0) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The Result Type of " << instr_name << " <id> '"
           << state.getIdName(inst->id())
           << "' must be OpTypeInt with width 32 and signedness 0.";
  }

  auto type_id = inst->GetOperandAs<uint32_t>(2);
  auto type = state.FindDef(type_id);
  if (type->opcode() != SpvOpTypeCooperativeMatrixNV) {
    return state.diag(SPV_ERROR_INVALID_ID, inst)
           << "The type in " << instr_name << " <id> '"
           << state.getIdName(type_id)
           << "' must be OpTypeCooperativeMatrixNV.";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateCooperativeMatrixLoadStoreNV(ValidationState_t& _,
                                                  const Instruction* inst) {
  uint32_t type_id;
  const char* opname;
  if (inst->opcode() == SpvOpCooperativeMatrixLoadNV) {
    type_id = inst->type_id();
    opname = "SpvOpCooperativeMatrixLoadNV";
  } else {
    // get Object operand's type
    type_id = _.FindDef(inst->GetOperandAs<uint32_t>(1))->type_id();
    opname = "SpvOpCooperativeMatrixStoreNV";
  }

  auto matrix_type = _.FindDef(type_id);

  if (matrix_type->opcode() != SpvOpTypeCooperativeMatrixNV) {
    if (inst->opcode() == SpvOpCooperativeMatrixLoadNV) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "SpvOpCooperativeMatrixLoadNV Result Type <id> '"
             << _.getIdName(type_id) << "' is not a cooperative matrix type.";
    } else {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "SpvOpCooperativeMatrixStoreNV Object type <id> '"
             << _.getIdName(type_id) << "' is not a cooperative matrix type.";
    }
  }

  const bool uses_variable_pointers =
      _.features().variable_pointers ||
      _.features().variable_pointers_storage_buffer;
  const auto pointer_index =
      (inst->opcode() == SpvOpCooperativeMatrixLoadNV) ? 2u : 0u;
  const auto pointer_id = inst->GetOperandAs<uint32_t>(pointer_index);
  const auto pointer = _.FindDef(pointer_id);
  if (!pointer ||
      ((_.addressing_model() == SpvAddressingModelLogical) &&
       ((!uses_variable_pointers &&
         !spvOpcodeReturnsLogicalPointer(pointer->opcode())) ||
        (uses_variable_pointers &&
         !spvOpcodeReturnsLogicalVariablePointer(pointer->opcode()))))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opname << " Pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a logical pointer.";
  }

  const auto pointer_type_id = pointer->type_id();
  const auto pointer_type = _.FindDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opname << " type for pointer <id> '" << _.getIdName(pointer_id)
           << "' is not a pointer type.";
  }

  const auto storage_class_index = 1u;
  const auto storage_class =
      pointer_type->GetOperandAs<uint32_t>(storage_class_index);

  if (storage_class != SpvStorageClassWorkgroup &&
      storage_class != SpvStorageClassStorageBuffer &&
      storage_class != SpvStorageClassPhysicalStorageBufferEXT) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opname << " storage class for pointer type <id> '"
           << _.getIdName(pointer_type_id)
           << "' is not Workgroup or StorageBuffer.";
  }

  const auto pointee_id = pointer_type->GetOperandAs<uint32_t>(2);
  const auto pointee_type = _.FindDef(pointee_id);
  if (!pointee_type || !(_.IsIntScalarOrVectorType(pointee_id) ||
                         _.IsFloatScalarOrVectorType(pointee_id))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opname << " Pointer <id> '" << _.getIdName(pointer->id())
           << "'s Type must be a scalar or vector type.";
  }

  const auto stride_index =
      (inst->opcode() == SpvOpCooperativeMatrixLoadNV) ? 3u : 2u;
  const auto stride_id = inst->GetOperandAs<uint32_t>(stride_index);
  const auto stride = _.FindDef(stride_id);
  if (!stride || !_.IsIntScalarType(stride->type_id())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Stride operand <id> '" << _.getIdName(stride_id)
           << "' must be a scalar integer type.";
  }

  const auto colmajor_index =
      (inst->opcode() == SpvOpCooperativeMatrixLoadNV) ? 4u : 3u;
  const auto colmajor_id = inst->GetOperandAs<uint32_t>(colmajor_index);
  const auto colmajor = _.FindDef(colmajor_id);
  if (!colmajor || !_.IsBoolScalarType(colmajor->type_id()) ||
      !(spvOpcodeIsConstant(colmajor->opcode()) ||
        spvOpcodeIsSpecConstant(colmajor->opcode()))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Column Major operand <id> '" << _.getIdName(colmajor_id)
           << "' must be a boolean constant instruction.";
  }

  const auto memory_access_index =
      (inst->opcode() == SpvOpCooperativeMatrixLoadNV) ? 5u : 4u;
  if (inst->operands().size() > memory_access_index) {
    if (auto error = CheckMemoryAccess(_, inst, memory_access_index))
      return error;
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t MemoryPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpVariable:
      if (auto error = ValidateVariable(_, inst)) return error;
      break;
    case SpvOpLoad:
      if (auto error = ValidateLoad(_, inst)) return error;
      break;
    case SpvOpStore:
      if (auto error = ValidateStore(_, inst)) return error;
      break;
    case SpvOpCopyMemory:
    case SpvOpCopyMemorySized:
      if (auto error = ValidateCopyMemory(_, inst)) return error;
      break;
    case SpvOpPtrAccessChain:
      if (auto error = ValidatePtrAccessChain(_, inst)) return error;
      break;
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpInBoundsPtrAccessChain:
      if (auto error = ValidateAccessChain(_, inst)) return error;
      break;
    case SpvOpArrayLength:
      if (auto error = ValidateArrayLength(_, inst)) return error;
      break;
    case SpvOpCooperativeMatrixLoadNV:
    case SpvOpCooperativeMatrixStoreNV:
      if (auto error = ValidateCooperativeMatrixLoadStoreNV(_, inst))
        return error;
      break;
    case SpvOpCooperativeMatrixLengthNV:
      if (auto error = ValidateCooperativeMatrixLengthNV(_, inst)) return error;
      break;
    case SpvOpImageTexelPointer:
    case SpvOpGenericPtrMemSemantics:
    default:
      break;
  }

  return SPV_SUCCESS;
}
}  // namespace val
}  // namespace spvtools
