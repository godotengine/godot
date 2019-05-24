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

// Validates correctness of built-in variables.

#include "source/val/validate.h"

#include <functional>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns a short textual description of the id defined by the given
// instruction.
std::string GetIdDesc(const Instruction& inst) {
  std::ostringstream ss;
  ss << "ID <" << inst.id() << "> (Op" << spvOpcodeString(inst.opcode()) << ")";
  return ss.str();
}

// Gets underlying data type which is
// - member type if instruction is OpTypeStruct
//   (member index is taken from decoration).
// - data type if id creates a pointer.
// - type of the constant if instruction is OpConst or OpSpecConst.
//
// Fails in any other case. The function is based on built-ins allowed by
// the Vulkan spec.
// TODO: If non-Vulkan validation rules are added then it might need
// to be refactored.
spv_result_t GetUnderlyingType(ValidationState_t& _,
                               const Decoration& decoration,
                               const Instruction& inst,
                               uint32_t* underlying_type) {
  if (decoration.struct_member_index() != Decoration::kInvalidMember) {
    assert(inst.opcode() == SpvOpTypeStruct);
    *underlying_type = inst.word(decoration.struct_member_index() + 2);
    return SPV_SUCCESS;
  }

  assert(inst.opcode() != SpvOpTypeStruct);

  if (spvOpcodeIsConstant(inst.opcode())) {
    *underlying_type = inst.type_id();
    return SPV_SUCCESS;
  }

  uint32_t storage_class = 0;
  if (!_.GetPointerTypeInfo(inst.type_id(), underlying_type, &storage_class)) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << GetIdDesc(inst)
           << " is decorated with BuiltIn. BuiltIn decoration should only be "
              "applied to struct types, variables and constants.";
  }
  return SPV_SUCCESS;
}

// Returns Storage Class used by the instruction if applicable.
// Returns SpvStorageClassMax if not.
SpvStorageClass GetStorageClass(const Instruction& inst) {
  switch (inst.opcode()) {
    case SpvOpTypePointer:
    case SpvOpTypeForwardPointer: {
      return SpvStorageClass(inst.word(2));
    }
    case SpvOpVariable: {
      return SpvStorageClass(inst.word(3));
    }
    case SpvOpGenericCastToPtrExplicit: {
      return SpvStorageClass(inst.word(4));
    }
    default: { break; }
  }
  return SpvStorageClassMax;
}

bool IsBuiltInValidForWebGPU(SpvBuiltIn label) {
  switch (label) {
    case SpvBuiltInPosition:
    case SpvBuiltInVertexIndex:
    case SpvBuiltInInstanceIndex:
    case SpvBuiltInFrontFacing:
    case SpvBuiltInFragCoord:
    case SpvBuiltInFragDepth:
    case SpvBuiltInNumWorkgroups:
    case SpvBuiltInWorkgroupSize:
    case SpvBuiltInLocalInvocationId:
    case SpvBuiltInGlobalInvocationId:
    case SpvBuiltInLocalInvocationIndex: {
      return true;
    }
    default:
      break;
  }

  return false;
}

// Helper class managing validation of built-ins.
// TODO: Generic functionality of this class can be moved into
// ValidationState_t to be made available to other users.
class BuiltInsValidator {
 public:
  BuiltInsValidator(ValidationState_t& vstate) : _(vstate) {}

  // Run validation.
  spv_result_t Run();

 private:
  // Goes through all decorations in the module, if decoration is BuiltIn
  // calls ValidateSingleBuiltInAtDefinition().
  spv_result_t ValidateBuiltInsAtDefinition();

  // Validates the instruction defining an id with built-in decoration.
  // Can be called multiple times for the same id, if multiple built-ins are
  // specified. Seeds id_to_at_reference_checks_ with decorated ids if needed.
  spv_result_t ValidateSingleBuiltInAtDefinition(const Decoration& decoration,
                                                 const Instruction& inst);

  // The following section contains functions which are called when id defined
  // by |inst| is decorated with BuiltIn |decoration|.
  // Most functions are specific to a single built-in and have naming scheme:
  // ValidateXYZAtDefinition. Some functions are common to multiple kinds of
  // BuiltIn.
  spv_result_t ValidateClipOrCullDistanceAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateFragCoordAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidateFragDepthAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidateFrontFacingAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateHelperInvocationAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateInvocationIdAtDefinition(const Decoration& decoration,
                                                const Instruction& inst);
  spv_result_t ValidateInstanceIndexAtDefinition(const Decoration& decoration,
                                                 const Instruction& inst);
  spv_result_t ValidateLayerOrViewportIndexAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidatePatchVerticesAtDefinition(const Decoration& decoration,
                                                 const Instruction& inst);
  spv_result_t ValidatePointCoordAtDefinition(const Decoration& decoration,
                                              const Instruction& inst);
  spv_result_t ValidatePointSizeAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidatePositionAtDefinition(const Decoration& decoration,
                                            const Instruction& inst);
  spv_result_t ValidatePrimitiveIdAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateSampleIdAtDefinition(const Decoration& decoration,
                                            const Instruction& inst);
  spv_result_t ValidateSampleMaskAtDefinition(const Decoration& decoration,
                                              const Instruction& inst);
  spv_result_t ValidateSamplePositionAtDefinition(const Decoration& decoration,
                                                  const Instruction& inst);
  spv_result_t ValidateTessCoordAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidateTessLevelOuterAtDefinition(const Decoration& decoration,
                                                  const Instruction& inst);
  spv_result_t ValidateTessLevelInnerAtDefinition(const Decoration& decoration,
                                                  const Instruction& inst);
  spv_result_t ValidateVertexIndexAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateVertexIdOrInstanceIdAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateLocalInvocationIndexAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateWorkgroupSizeAtDefinition(const Decoration& decoration,
                                                 const Instruction& inst);
  // Used for GlobalInvocationId, LocalInvocationId, NumWorkgroups, WorkgroupId.
  spv_result_t ValidateComputeShaderI32Vec3InputAtDefinition(
      const Decoration& decoration, const Instruction& inst);

  // The following section contains functions which are called when id defined
  // by |referenced_inst| is
  // 1. referenced by |referenced_from_inst|
  // 2. dependent on |built_in_inst| which is decorated with BuiltIn
  // |decoration|. Most functions are specific to a single built-in and have
  // naming scheme: ValidateXYZAtReference. Some functions are common to
  // multiple kinds of BuiltIn.
  spv_result_t ValidateFragCoordAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFragDepthAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFrontFacingAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateHelperInvocationAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateInvocationIdAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateInstanceIdAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateInstanceIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePatchVerticesAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePointCoordAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePointSizeAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePositionAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePrimitiveIdAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateSampleIdAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateSampleMaskAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateSamplePositionAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateTessCoordAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateTessLevelAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateLocalInvocationIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateVertexIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateLayerOrViewportIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateWorkgroupSizeAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateClipOrCullDistanceAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Used for GlobalInvocationId, LocalInvocationId, NumWorkgroups, WorkgroupId.
  spv_result_t ValidateComputeShaderI32Vec3InputAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Validates that |built_in_inst| is not (even indirectly) referenced from
  // within a function which can be called with |execution_model|.
  //
  // |comment| - text explaining why the restriction was imposed.
  // |decoration| - BuiltIn decoration which causes the restriction.
  // |referenced_inst| - instruction which is dependent on |built_in_inst| and
  //                     defines the id which was referenced.
  // |referenced_from_inst| - instruction which references id defined by
  //                          |referenced_inst| from within a function.
  spv_result_t ValidateNotCalledWithExecutionModel(
      const char* comment, SpvExecutionModel execution_model,
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // The following section contains functions which check that the decorated
  // variable has the type specified in the function name. |diag| would be
  // called with a corresponding error message, if validation is not successful.
  spv_result_t ValidateBool(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateI32(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateI32Vec(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateI32Arr(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateF32(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateOptionalArrayedF32(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateF32Helper(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag,
      uint32_t underlying_type);
  spv_result_t ValidateF32Vec(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateOptionalArrayedF32Vec(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateF32VecHelper(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag,
      uint32_t underlying_type);
  // If |num_components| is zero, the number of components is not checked.
  spv_result_t ValidateF32Arr(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateOptionalArrayedF32Arr(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateF32ArrHelper(
      const Decoration& decoration, const Instruction& inst,
      uint32_t num_components,
      const std::function<spv_result_t(const std::string& message)>& diag,
      uint32_t underlying_type);

  // Generates strings like "Member #0 of struct ID <2>".
  std::string GetDefinitionDesc(const Decoration& decoration,
                                const Instruction& inst) const;

  // Generates strings like "ID <51> (OpTypePointer) is referencing ID <2>
  // (OpTypeStruct) which is decorated with BuiltIn Position".
  std::string GetReferenceDesc(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst,
      SpvExecutionModel execution_model = SpvExecutionModelMax) const;

  // Generates strings like "ID <51> (OpTypePointer) uses storage class
  // UniformConstant".
  std::string GetStorageClassDesc(const Instruction& inst) const;

  // Updates inner working of the class. Is called sequentially for every
  // instruction.
  void Update(const Instruction& inst);

  ValidationState_t& _;

  // Mapping id -> list of rules which validate instruction referencing the
  // id. Rules can create new rules and add them to this container.
  // Using std::map, and not std::unordered_map to avoid iterator invalidation
  // during rehashing.
  std::map<uint32_t, std::list<std::function<spv_result_t(const Instruction&)>>>
      id_to_at_reference_checks_;

  // Id of the function we are currently inside. 0 if not inside a function.
  uint32_t function_id_ = 0;

  // Entry points which can (indirectly) call the current function.
  // The pointer either points to a vector inside to function_to_entry_points_
  // or to no_entry_points_. The pointer is guaranteed to never be null.
  const std::vector<uint32_t> no_entry_points;
  const std::vector<uint32_t>* entry_points_ = &no_entry_points;

  // Execution models with which the current function can be called.
  std::set<SpvExecutionModel> execution_models_;
};

void BuiltInsValidator::Update(const Instruction& inst) {
  const SpvOp opcode = inst.opcode();
  if (opcode == SpvOpFunction) {
    // Entering a function.
    assert(function_id_ == 0);
    function_id_ = inst.id();
    execution_models_.clear();
    entry_points_ = &_.FunctionEntryPoints(function_id_);
    // Collect execution models from all entry points from which the current
    // function can be called.
    for (const uint32_t entry_point : *entry_points_) {
      if (const auto* models = _.GetExecutionModels(entry_point)) {
        execution_models_.insert(models->begin(), models->end());
      }
    }
  }

  if (opcode == SpvOpFunctionEnd) {
    // Exiting a function.
    assert(function_id_ != 0);
    function_id_ = 0;
    entry_points_ = &no_entry_points;
    execution_models_.clear();
  }
}

std::string BuiltInsValidator::GetDefinitionDesc(
    const Decoration& decoration, const Instruction& inst) const {
  std::ostringstream ss;
  if (decoration.struct_member_index() != Decoration::kInvalidMember) {
    assert(inst.opcode() == SpvOpTypeStruct);
    ss << "Member #" << decoration.struct_member_index();
    ss << " of struct ID <" << inst.id() << ">";
  } else {
    ss << GetIdDesc(inst);
  }
  return ss.str();
}

std::string BuiltInsValidator::GetReferenceDesc(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst, const Instruction& referenced_from_inst,
    SpvExecutionModel execution_model) const {
  std::ostringstream ss;
  ss << GetIdDesc(referenced_from_inst) << " is referencing "
     << GetIdDesc(referenced_inst);
  if (built_in_inst.id() != referenced_inst.id()) {
    ss << " which is dependent on " << GetIdDesc(built_in_inst);
  }

  ss << " which is decorated with BuiltIn ";
  ss << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                      decoration.params()[0]);
  if (function_id_) {
    ss << " in function <" << function_id_ << ">";
    if (execution_model != SpvExecutionModelMax) {
      ss << " called with execution model ";
      ss << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_EXECUTION_MODEL,
                                          execution_model);
    }
  }
  ss << ".";
  return ss.str();
}

std::string BuiltInsValidator::GetStorageClassDesc(
    const Instruction& inst) const {
  std::ostringstream ss;
  ss << GetIdDesc(inst) << " uses storage class ";
  ss << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_STORAGE_CLASS,
                                      GetStorageClass(inst));
  ss << ".";
  return ss.str();
}

spv_result_t BuiltInsValidator::ValidateBool(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  if (!_.IsBoolScalarType(underlying_type)) {
    return diag(GetDefinitionDesc(decoration, inst) + " is not a bool scalar.");
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateI32(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  if (!_.IsIntScalarType(underlying_type)) {
    return diag(GetDefinitionDesc(decoration, inst) + " is not an int scalar.");
  }

  const uint32_t bit_width = _.GetBitWidth(underlying_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst) << " has bit width " << bit_width
       << ".";
    return diag(ss.str());
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateOptionalArrayedF32(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  // Strip the array, if present.
  if (_.GetIdOpcode(underlying_type) == SpvOpTypeArray) {
    underlying_type = _.FindDef(underlying_type)->word(2u);
  }

  return ValidateF32Helper(decoration, inst, diag, underlying_type);
}

spv_result_t BuiltInsValidator::ValidateF32(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  return ValidateF32Helper(decoration, inst, diag, underlying_type);
}

spv_result_t BuiltInsValidator::ValidateF32Helper(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag,
    uint32_t underlying_type) {
  if (!_.IsFloatScalarType(underlying_type)) {
    return diag(GetDefinitionDesc(decoration, inst) +
                " is not a float scalar.");
  }

  const uint32_t bit_width = _.GetBitWidth(underlying_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst) << " has bit width " << bit_width
       << ".";
    return diag(ss.str());
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateI32Vec(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  if (!_.IsIntVectorType(underlying_type)) {
    return diag(GetDefinitionDesc(decoration, inst) + " is not an int vector.");
  }

  const uint32_t actual_num_components = _.GetDimension(underlying_type);
  if (_.GetDimension(underlying_type) != num_components) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst) << " has "
       << actual_num_components << " components.";
    return diag(ss.str());
  }

  const uint32_t bit_width = _.GetBitWidth(underlying_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst)
       << " has components with bit width " << bit_width << ".";
    return diag(ss.str());
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateOptionalArrayedF32Vec(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  // Strip the array, if present.
  if (_.GetIdOpcode(underlying_type) == SpvOpTypeArray) {
    underlying_type = _.FindDef(underlying_type)->word(2u);
  }

  return ValidateF32VecHelper(decoration, inst, num_components, diag,
                              underlying_type);
}

spv_result_t BuiltInsValidator::ValidateF32Vec(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  return ValidateF32VecHelper(decoration, inst, num_components, diag,
                              underlying_type);
}

spv_result_t BuiltInsValidator::ValidateF32VecHelper(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag,
    uint32_t underlying_type) {
  if (!_.IsFloatVectorType(underlying_type)) {
    return diag(GetDefinitionDesc(decoration, inst) +
                " is not a float vector.");
  }

  const uint32_t actual_num_components = _.GetDimension(underlying_type);
  if (_.GetDimension(underlying_type) != num_components) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst) << " has "
       << actual_num_components << " components.";
    return diag(ss.str());
  }

  const uint32_t bit_width = _.GetBitWidth(underlying_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst)
       << " has components with bit width " << bit_width << ".";
    return diag(ss.str());
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateI32Arr(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  const Instruction* const type_inst = _.FindDef(underlying_type);
  if (type_inst->opcode() != SpvOpTypeArray) {
    return diag(GetDefinitionDesc(decoration, inst) + " is not an array.");
  }

  const uint32_t component_type = type_inst->word(2);
  if (!_.IsIntScalarType(component_type)) {
    return diag(GetDefinitionDesc(decoration, inst) +
                " components are not int scalar.");
  }

  const uint32_t bit_width = _.GetBitWidth(component_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst)
       << " has components with bit width " << bit_width << ".";
    return diag(ss.str());
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateF32Arr(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  return ValidateF32ArrHelper(decoration, inst, num_components, diag,
                              underlying_type);
}

spv_result_t BuiltInsValidator::ValidateOptionalArrayedF32Arr(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  // Strip an extra layer of arraying if present.
  if (_.GetIdOpcode(underlying_type) == SpvOpTypeArray) {
    uint32_t subtype = _.FindDef(underlying_type)->word(2u);
    if (_.GetIdOpcode(subtype) == SpvOpTypeArray) {
      underlying_type = subtype;
    }
  }

  return ValidateF32ArrHelper(decoration, inst, num_components, diag,
                              underlying_type);
}

spv_result_t BuiltInsValidator::ValidateF32ArrHelper(
    const Decoration& decoration, const Instruction& inst,
    uint32_t num_components,
    const std::function<spv_result_t(const std::string& message)>& diag,
    uint32_t underlying_type) {
  const Instruction* const type_inst = _.FindDef(underlying_type);
  if (type_inst->opcode() != SpvOpTypeArray) {
    return diag(GetDefinitionDesc(decoration, inst) + " is not an array.");
  }

  const uint32_t component_type = type_inst->word(2);
  if (!_.IsFloatScalarType(component_type)) {
    return diag(GetDefinitionDesc(decoration, inst) +
                " components are not float scalar.");
  }

  const uint32_t bit_width = _.GetBitWidth(component_type);
  if (bit_width != 32) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst)
       << " has components with bit width " << bit_width << ".";
    return diag(ss.str());
  }

  if (num_components != 0) {
    uint64_t actual_num_components = 0;
    if (!_.GetConstantValUint64(type_inst->word(3), &actual_num_components)) {
      assert(0 && "Array type definition is corrupt");
    }
    if (actual_num_components != num_components) {
      std::ostringstream ss;
      ss << GetDefinitionDesc(decoration, inst) << " has "
         << actual_num_components << " components.";
      return diag(ss.str());
    }
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateNotCalledWithExecutionModel(
    const char* comment, SpvExecutionModel execution_model,
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (function_id_) {
    if (execution_models_.count(execution_model)) {
      const char* execution_model_str = _.grammar().lookupOperandName(
          SPV_OPERAND_TYPE_EXECUTION_MODEL, execution_model);
      const char* built_in_str = _.grammar().lookupOperandName(
          SPV_OPERAND_TYPE_BUILT_IN, decoration.params()[0]);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << comment << " " << GetIdDesc(referenced_inst) << " depends on "
             << GetIdDesc(built_in_inst) << " which is decorated with BuiltIn "
             << built_in_str << "."
             << " Id <" << referenced_inst.id() << "> is later referenced by "
             << GetIdDesc(referenced_from_inst) << " in function <"
             << function_id_ << "> which is called with execution model "
             << execution_model_str << ".";
    }
  } else {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
                  comment, execution_model, decoration, built_in_inst,
                  referenced_from_inst, std::placeholders::_1));
  }
  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateClipOrCullDistanceAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  // Seed at reference checks with this built-in.
  return ValidateClipOrCullDistanceAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateClipOrCullDistanceAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassInput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Input storage class if execution model is "
          "Vertex.",
          SpvExecutionModelVertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    if (storage_class == SpvStorageClassOutput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Output storage class if execution model is "
          "Fragment.",
          SpvExecutionModelFragment, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelFragment:
        case SpvExecutionModelVertex: {
          if (spv_result_t error = ValidateF32Arr(
                  decoration, built_in_inst, /* Any number of components */ 0,
                  [this, &decoration, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << "According to the Vulkan spec BuiltIn "
                           << _.grammar().lookupOperandName(
                                  SPV_OPERAND_TYPE_BUILT_IN,
                                  decoration.params()[0])
                           << " variable needs to be a 32-bit float array. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        case SpvExecutionModelTessellationControl:
        case SpvExecutionModelTessellationEvaluation:
        case SpvExecutionModelGeometry:
        case SpvExecutionModelMeshNV: {
          if (decoration.struct_member_index() != Decoration::kInvalidMember) {
            // The outer level of array is applied on the variable.
            if (spv_result_t error = ValidateF32Arr(
                    decoration, built_in_inst, /* Any number of components */ 0,
                    [this, &decoration, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn "
                             << _.grammar().lookupOperandName(
                                    SPV_OPERAND_TYPE_BUILT_IN,
                                    decoration.params()[0])
                             << " variable needs to be a 32-bit float array. "
                             << message;
                    })) {
              return error;
            }
          } else {
            if (spv_result_t error = ValidateOptionalArrayedF32Arr(
                    decoration, built_in_inst, /* Any number of components */ 0,
                    [this, &decoration, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn "
                             << _.grammar().lookupOperandName(
                                    SPV_OPERAND_TYPE_BUILT_IN,
                                    decoration.params()[0])
                             << " variable needs to be a 32-bit float array. "
                             << message;
                    })) {
              return error;
            }
          }
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  decoration.params()[0])
                 << " to be used only with Fragment, Vertex, "
                    "TessellationControl, TessellationEvaluation or Geometry "
                    "execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateClipOrCullDistanceAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFragCoordAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 4,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn FragCoord "
                        "variable needs to be a 4-component 32-bit float "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateFragCoordAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFragCoordAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FragCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn FragCoord to be used only with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragCoordAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFragDepthAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn FragDepth "
                        "variable needs to be a 32-bit float scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateFragDepthAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFragDepthAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FragDepth to be only used for "
                "variables with Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn FragDepth to be used only with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }

    for (const uint32_t entry_point : *entry_points_) {
      // Every entry point from which this function is called needs to have
      // Execution Mode DepthReplacing.
      const auto* modes = _.GetExecutionModes(entry_point);
      if (!modes || !modes->count(SpvExecutionModeDepthReplacing)) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec requires DepthReplacing execution mode to be "
                  "declared when using BuiltIn FragDepth. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragDepthAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFrontFacingAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateBool(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn FrontFacing "
                        "variable needs to be a bool scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateFrontFacingAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFrontFacingAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FrontFacing to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn FrontFacing to be used only with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFrontFacingAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateHelperInvocationAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateBool(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn HelperInvocation "
                        "variable needs to be a bool scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateHelperInvocationAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateHelperInvocationAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn HelperInvocation to be only used "
                "for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn HelperInvocation to be used only "
                  "with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateHelperInvocationAtReference, this,
                  decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateInvocationIdAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn InvocationId "
                        "variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateInvocationIdAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateInvocationIdAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn InvocationId to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelTessellationControl &&
          execution_model != SpvExecutionModelGeometry) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn InvocationId to be used only "
                  "with TessellationControl or Geometry execution models. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateInvocationIdAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateInstanceIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn InstanceIndex "
                        "variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateInstanceIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateInstanceIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn InstanceIndex to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelVertex) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn InstanceIndex to be used only "
                  "with Vertex execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateInstanceIndexAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePatchVerticesAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn PatchVertices "
                        "variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidatePatchVerticesAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePatchVerticesAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn PatchVertices to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelTessellationControl &&
          execution_model != SpvExecutionModelTessellationEvaluation) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn PatchVertices to be used only "
                  "with TessellationControl or TessellationEvaluation "
                  "execution models. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidatePatchVerticesAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePointCoordAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 2,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn PointCoord "
                        "variable needs to be a 2-component 32-bit float "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidatePointCoordAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePointCoordAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn PointCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn PointCoord to be used only with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidatePointCoordAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePointSizeAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  // Seed at reference checks with this built-in.
  return ValidatePointSizeAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePointSizeAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn PointSize to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassInput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn PointSize to be used for "
          "variables with Input storage class if execution model is Vertex.",
          SpvExecutionModelVertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelVertex: {
          if (spv_result_t error = ValidateF32(
                  decoration, built_in_inst,
                  [this, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << "According to the Vulkan spec BuiltIn PointSize "
                              "variable needs to be a 32-bit float scalar. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        case SpvExecutionModelTessellationControl:
        case SpvExecutionModelTessellationEvaluation:
        case SpvExecutionModelGeometry:
        case SpvExecutionModelMeshNV: {
          // PointSize can be a per-vertex variable for tessellation control,
          // tessellation evaluation and geometry shader stages. In such cases
          // variables will have an array of 32-bit floats.
          if (decoration.struct_member_index() != Decoration::kInvalidMember) {
            // The array is on the variable, so this must be a 32-bit float.
            if (spv_result_t error = ValidateF32(
                    decoration, built_in_inst,
                    [this, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn "
                                "PointSize variable needs to be a 32-bit "
                                "float scalar. "
                             << message;
                    })) {
              return error;
            }
          } else {
            if (spv_result_t error = ValidateOptionalArrayedF32(
                    decoration, built_in_inst,
                    [this, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn "
                                "PointSize variable needs to be a 32-bit "
                                "float scalar. "
                             << message;
                    })) {
              return error;
            }
          }
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn PointSize to be used only with "
                    "Vertex, TessellationControl, TessellationEvaluation or "
                    "Geometry execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidatePointSizeAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePositionAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  // Seed at reference checks with this built-in.
  return ValidatePositionAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePositionAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn Position to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassInput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn Position to be used for variables "
          "with Input storage class if execution model is Vertex.",
          SpvExecutionModelVertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelVertex: {
          if (spv_result_t error = ValidateF32Vec(
                  decoration, built_in_inst, 4,
                  [this, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << "According to the Vulkan spec BuiltIn Position "
                              "variable needs to be a 4-component 32-bit float "
                              "vector. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        case SpvExecutionModelGeometry:
        case SpvExecutionModelTessellationControl:
        case SpvExecutionModelTessellationEvaluation:
        case SpvExecutionModelMeshNV: {
          // Position can be a per-vertex variable for tessellation control,
          // tessellation evaluation, geometry and mesh shader stages. In such
          // cases variables will have an array of 4-component 32-bit float
          // vectors.
          if (decoration.struct_member_index() != Decoration::kInvalidMember) {
            // The array is on the variable, so this must be a 4-component
            // 32-bit float vector.
            if (spv_result_t error = ValidateF32Vec(
                    decoration, built_in_inst, 4,
                    [this, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn Position "
                                "variable needs to be a 4-component 32-bit "
                                "float vector. "
                             << message;
                    })) {
              return error;
            }
          } else {
            if (spv_result_t error = ValidateOptionalArrayedF32Vec(
                    decoration, built_in_inst, 4,
                    [this, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << "According to the Vulkan spec BuiltIn Position "
                                "variable needs to be a 4-component 32-bit "
                                "float vector. "
                             << message;
                    })) {
              return error;
            }
          }
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn Position to be used only "
                    "with Vertex, TessellationControl, TessellationEvaluation"
                    " or Geometry execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (spvIsWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "WebGPU spec allows BuiltIn Position to be only used for "
                "variables with Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelVertex: {
          if (spv_result_t error = ValidateF32Vec(
                  decoration, built_in_inst, 4,
                  [this, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << "According to the WebGPU spec BuiltIn Position "
                              "variable needs to be a 4-component 32-bit float "
                              "vector. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "WebGPU spec allows BuiltIn Position to be used only "
                    "with the Vertex execution model. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidatePositionAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePrimitiveIdAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn PrimitiveId "
                        "variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidatePrimitiveIdAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePrimitiveIdAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn PrimitiveId to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassOutput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "TessellationControl.",
          SpvExecutionModelTessellationControl, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "TessellationEvaluation.",
          SpvExecutionModelTessellationEvaluation, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "Fragment.",
          SpvExecutionModelFragment, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelFragment:
        case SpvExecutionModelTessellationControl:
        case SpvExecutionModelTessellationEvaluation:
        case SpvExecutionModelGeometry:
        case SpvExecutionModelMeshNV:
        case SpvExecutionModelRayGenerationNV:
        case SpvExecutionModelIntersectionNV:
        case SpvExecutionModelAnyHitNV:
        case SpvExecutionModelClosestHitNV:
        case SpvExecutionModelMissNV:
        case SpvExecutionModelCallableNV: {
          // Ok.
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn PrimitiveId to be used only "
                    "with Fragment, TessellationControl, "
                    "TessellationEvaluation or Geometry execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidatePrimitiveIdAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateSampleIdAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn SampleId "
                        "variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateSampleIdAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateSampleIdAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn SampleId to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn SampleId to be used only with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateSampleIdAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateSampleMaskAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32Arr(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn SampleMask "
                        "variable needs to be a 32-bit int array. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateSampleMaskAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateSampleMaskAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn SampleMask to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn SampleMask to be used only "
                  "with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateSampleMaskAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateSamplePositionAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 2,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn SamplePosition "
                        "variable needs to be a 2-component 32-bit float "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateSamplePositionAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateSamplePositionAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn SamplePosition to be only used "
                "for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelFragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn SamplePosition to be used only "
                  "with "
                  "Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateSamplePositionAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateTessCoordAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 3,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn TessCoord "
                        "variable needs to be a 3-component 32-bit float "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateTessCoordAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateTessCoordAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn TessCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelTessellationEvaluation) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "Vulkan spec allows BuiltIn TessCoord to be used only with "
                  "TessellationEvaluation execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateTessCoordAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateTessLevelOuterAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Arr(
            decoration, inst, 4,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn TessLevelOuter "
                        "variable needs to be a 4-component 32-bit float "
                        "array. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateTessLevelAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateTessLevelInnerAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Arr(
            decoration, inst, 2,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn TessLevelOuter "
                        "variable needs to be a 2-component 32-bit float "
                        "array. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateTessLevelAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateTessLevelAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassInput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
          "used "
          "for variables with Input storage class if execution model is "
          "TessellationControl.",
          SpvExecutionModelTessellationControl, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    if (storage_class == SpvStorageClassOutput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
          "used "
          "for variables with Output storage class if execution model is "
          "TessellationEvaluation.",
          SpvExecutionModelTessellationEvaluation, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelTessellationControl:
        case SpvExecutionModelTessellationEvaluation: {
          // Ok.
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  decoration.params()[0])
                 << " to be used only with TessellationControl or "
                    "TessellationEvaluation execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateTessLevelAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateVertexIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn VertexIndex variable needs to be a "
                        "32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateVertexIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateVertexIdOrInstanceIdAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  const SpvBuiltIn label = SpvBuiltIn(decoration.params()[0]);
  bool allow_instance_id = _.HasCapability(SpvCapabilityRayTracingNV) &&
                           label == SpvBuiltInInstanceId;
  if (spvIsVulkanEnv(_.context()->target_env) && !allow_instance_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << "Vulkan spec doesn't allow BuiltIn VertexId/InstanceId "
              "to be used.";
  }

  if (label == SpvBuiltInInstanceId) {
    return ValidateInstanceIdAtReference(decoration, inst, inst, inst);
  }
  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateInstanceIdAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelIntersectionNV:
        case SpvExecutionModelClosestHitNV:
        case SpvExecutionModelAnyHitNV:
          // Do nothing, valid stages
          break;
        default:
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn InstanceId to be used "
                    "only with IntersectionNV, ClosestHitNV and AnyHitNV "
                    "execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst);
          break;
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateInstanceIdAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateLocalInvocationIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the WebGPU spec BuiltIn "
                        "LocalInvocationIndex variable needs to be a 32-bit "
                        "int."
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateLocalInvocationIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateLocalInvocationIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "WebGPU spec allows BuiltIn LocalInvocationIndex to be only "
                "used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelGLCompute) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << "WebGPU spec allows BuiltIn VertexIndex to be used only "
                  "with GLCompute execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateLocalInvocationIndexAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateVertexIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn VertexIndex to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelVertex) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn VertexIndex to be used only with "
                  "Vertex execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateVertexIndexAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateLayerOrViewportIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &decoration,
             &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << "variable needs to be a 32-bit int scalar. " << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateLayerOrViewportIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateLayerOrViewportIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == SpvStorageClassInput) {
      assert(function_id_ == 0);
      for (const auto em :
           {SpvExecutionModelVertex, SpvExecutionModelTessellationEvaluation,
            SpvExecutionModelGeometry}) {
        id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
            std::bind(&BuiltInsValidator::ValidateNotCalledWithExecutionModel,
                      this,
                      "Vulkan spec doesn't allow BuiltIn Layer and "
                      "ViewportIndex to be "
                      "used for variables with Input storage class if "
                      "execution model is Vertex, TessellationEvaluation, or "
                      "Geometry.",
                      em, decoration, built_in_inst, referenced_from_inst,
                      std::placeholders::_1));
      }
    }

    if (storage_class == SpvStorageClassOutput) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this,
          "Vulkan spec doesn't allow BuiltIn Layer and "
          "ViewportIndex to be "
          "used for variables with Output storage class if "
          "execution model is "
          "Fragment.",
          SpvExecutionModelFragment, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case SpvExecutionModelGeometry:
        case SpvExecutionModelFragment:
        case SpvExecutionModelMeshNV: {
          // Ok.
          break;
          case SpvExecutionModelVertex:
          case SpvExecutionModelTessellationEvaluation:
            if (!_.HasCapability(SpvCapabilityShaderViewportIndexLayerEXT)) {
              return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                     << "Using BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " in Vertex or Tessellation execution model requires "
                        "the ShaderViewportIndexLayerEXT capability.";
            }
            break;
        }
        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  decoration.params()[0])
                 << " to be used only with Vertex, TessellationEvaluation, "
                    "Geometry, or Fragment execution models. "
                 << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                     referenced_from_inst, execution_model);
        }
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateLayerOrViewportIndexAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateComputeShaderI32Vec3InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 3,
            [this, &decoration,
             &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 3-component 32-bit int "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateComputeShaderI32Vec3InputAtReference(decoration, inst, inst,
                                                      inst);
}

spv_result_t BuiltInsValidator::ValidateComputeShaderI32Vec3InputAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    const SpvStorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != SpvStorageClassMax &&
        storage_class != SpvStorageClassInput) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const SpvExecutionModel execution_model : execution_models_) {
      bool has_vulkan_model = execution_model == SpvExecutionModelGLCompute ||
                              execution_model == SpvExecutionModelTaskNV ||
                              execution_model == SpvExecutionModelMeshNV;
      bool has_webgpu_model = execution_model == SpvExecutionModelGLCompute;
      if ((spvIsVulkanEnv(_.context()->target_env) && !has_vulkan_model) ||
          (spvIsWebGPUEnv(_.context()->target_env) && !has_webgpu_model)) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                decoration.params()[0])
               << " to be used only with GLCompute execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateComputeShaderI32Vec3InputAtReference, this,
        decoration, built_in_inst, referenced_from_inst,
        std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateWorkgroupSizeAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    if (spvIsVulkanEnv(_.context()->target_env) &&
        !spvOpcodeIsConstant(inst.opcode())) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << "Vulkan spec requires BuiltIn WorkgroupSize to be a "
                "constant. "
             << GetIdDesc(inst) << " is not a constant.";
    }

    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 3,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn WorkgroupSize variable needs to be a "
                        "3-component 32-bit int vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateWorkgroupSizeAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateWorkgroupSizeAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    for (const SpvExecutionModel execution_model : execution_models_) {
      if (execution_model != SpvExecutionModelGLCompute) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                decoration.params()[0])
               << " to be used only with GLCompute execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateWorkgroupSizeAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateSingleBuiltInAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  const SpvBuiltIn label = SpvBuiltIn(decoration.params()[0]);

  if (spvIsWebGPUEnv(_.context()->target_env) &&
      !IsBuiltInValidForWebGPU(label)) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << "WebGPU does not allow BuiltIn "
           << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                            decoration.params()[0]);
  }

  // If you are adding a new BuiltIn enum, please register it here.
  // If the newly added enum has validation rules associated with it
  // consider leaving a TODO and/or creating an issue.
  switch (label) {
    case SpvBuiltInClipDistance:
    case SpvBuiltInCullDistance: {
      return ValidateClipOrCullDistanceAtDefinition(decoration, inst);
    }
    case SpvBuiltInFragCoord: {
      return ValidateFragCoordAtDefinition(decoration, inst);
    }
    case SpvBuiltInFragDepth: {
      return ValidateFragDepthAtDefinition(decoration, inst);
    }
    case SpvBuiltInFrontFacing: {
      return ValidateFrontFacingAtDefinition(decoration, inst);
    }
    case SpvBuiltInGlobalInvocationId:
    case SpvBuiltInLocalInvocationId:
    case SpvBuiltInNumWorkgroups:
    case SpvBuiltInWorkgroupId: {
      return ValidateComputeShaderI32Vec3InputAtDefinition(decoration, inst);
    }
    case SpvBuiltInHelperInvocation: {
      return ValidateHelperInvocationAtDefinition(decoration, inst);
    }
    case SpvBuiltInInvocationId: {
      return ValidateInvocationIdAtDefinition(decoration, inst);
    }
    case SpvBuiltInInstanceIndex: {
      return ValidateInstanceIndexAtDefinition(decoration, inst);
    }
    case SpvBuiltInLayer:
    case SpvBuiltInViewportIndex: {
      return ValidateLayerOrViewportIndexAtDefinition(decoration, inst);
    }
    case SpvBuiltInPatchVertices: {
      return ValidatePatchVerticesAtDefinition(decoration, inst);
    }
    case SpvBuiltInPointCoord: {
      return ValidatePointCoordAtDefinition(decoration, inst);
    }
    case SpvBuiltInPointSize: {
      return ValidatePointSizeAtDefinition(decoration, inst);
    }
    case SpvBuiltInPosition: {
      return ValidatePositionAtDefinition(decoration, inst);
    }
    case SpvBuiltInPrimitiveId: {
      return ValidatePrimitiveIdAtDefinition(decoration, inst);
    }
    case SpvBuiltInSampleId: {
      return ValidateSampleIdAtDefinition(decoration, inst);
    }
    case SpvBuiltInSampleMask: {
      return ValidateSampleMaskAtDefinition(decoration, inst);
    }
    case SpvBuiltInSamplePosition: {
      return ValidateSamplePositionAtDefinition(decoration, inst);
    }
    case SpvBuiltInTessCoord: {
      return ValidateTessCoordAtDefinition(decoration, inst);
    }
    case SpvBuiltInTessLevelOuter: {
      return ValidateTessLevelOuterAtDefinition(decoration, inst);
    }
    case SpvBuiltInTessLevelInner: {
      return ValidateTessLevelInnerAtDefinition(decoration, inst);
    }
    case SpvBuiltInVertexIndex: {
      return ValidateVertexIndexAtDefinition(decoration, inst);
    }
    case SpvBuiltInWorkgroupSize: {
      return ValidateWorkgroupSizeAtDefinition(decoration, inst);
    }
    case SpvBuiltInVertexId:
    case SpvBuiltInInstanceId: {
      return ValidateVertexIdOrInstanceIdAtDefinition(decoration, inst);
    }
    case SpvBuiltInLocalInvocationIndex: {
      return ValidateLocalInvocationIndexAtDefinition(decoration, inst);
    }
    case SpvBuiltInWorkDim:
    case SpvBuiltInGlobalSize:
    case SpvBuiltInEnqueuedWorkgroupSize:
    case SpvBuiltInGlobalOffset:
    case SpvBuiltInGlobalLinearId:
    case SpvBuiltInSubgroupSize:
    case SpvBuiltInSubgroupMaxSize:
    case SpvBuiltInNumSubgroups:
    case SpvBuiltInNumEnqueuedSubgroups:
    case SpvBuiltInSubgroupId:
    case SpvBuiltInSubgroupLocalInvocationId:
    case SpvBuiltInSubgroupEqMaskKHR:
    case SpvBuiltInSubgroupGeMaskKHR:
    case SpvBuiltInSubgroupGtMaskKHR:
    case SpvBuiltInSubgroupLeMaskKHR:
    case SpvBuiltInSubgroupLtMaskKHR:
    case SpvBuiltInBaseVertex:
    case SpvBuiltInBaseInstance:
    case SpvBuiltInDrawIndex:
    case SpvBuiltInDeviceIndex:
    case SpvBuiltInViewIndex:
    case SpvBuiltInBaryCoordNoPerspAMD:
    case SpvBuiltInBaryCoordNoPerspCentroidAMD:
    case SpvBuiltInBaryCoordNoPerspSampleAMD:
    case SpvBuiltInBaryCoordSmoothAMD:
    case SpvBuiltInBaryCoordSmoothCentroidAMD:
    case SpvBuiltInBaryCoordSmoothSampleAMD:
    case SpvBuiltInBaryCoordPullModelAMD:
    case SpvBuiltInFragStencilRefEXT:
    case SpvBuiltInViewportMaskNV:
    case SpvBuiltInSecondaryPositionNV:
    case SpvBuiltInSecondaryViewportMaskNV:
    case SpvBuiltInPositionPerViewNV:
    case SpvBuiltInViewportMaskPerViewNV:
    case SpvBuiltInFullyCoveredEXT:
    case SpvBuiltInMax:
    case SpvBuiltInTaskCountNV:
    case SpvBuiltInPrimitiveCountNV:
    case SpvBuiltInPrimitiveIndicesNV:
    case SpvBuiltInClipDistancePerViewNV:
    case SpvBuiltInCullDistancePerViewNV:
    case SpvBuiltInLayerPerViewNV:
    case SpvBuiltInMeshViewCountNV:
    case SpvBuiltInMeshViewIndicesNV:
    case SpvBuiltInBaryCoordNV:
    case SpvBuiltInBaryCoordNoPerspNV:
    case SpvBuiltInFragmentSizeNV:         // alias SpvBuiltInFragSizeEXT
    case SpvBuiltInInvocationsPerPixelNV:  // alias
                                           // SpvBuiltInFragInvocationCountEXT
    case SpvBuiltInLaunchIdNV:
    case SpvBuiltInLaunchSizeNV:
    case SpvBuiltInWorldRayOriginNV:
    case SpvBuiltInWorldRayDirectionNV:
    case SpvBuiltInObjectRayOriginNV:
    case SpvBuiltInObjectRayDirectionNV:
    case SpvBuiltInRayTminNV:
    case SpvBuiltInRayTmaxNV:
    case SpvBuiltInInstanceCustomIndexNV:
    case SpvBuiltInObjectToWorldNV:
    case SpvBuiltInWorldToObjectNV:
    case SpvBuiltInHitTNV:
    case SpvBuiltInHitKindNV:
    case SpvBuiltInIncomingRayFlagsNV: {
      // No validation rules (for the moment).
      break;
    }
  }
  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateBuiltInsAtDefinition() {
  for (const auto& kv : _.id_decorations()) {
    const uint32_t id = kv.first;
    const auto& decorations = kv.second;
    if (decorations.empty()) {
      continue;
    }

    const Instruction* inst = _.FindDef(id);
    assert(inst);

    for (const auto& decoration : kv.second) {
      if (decoration.dec_type() != SpvDecorationBuiltIn) {
        continue;
      }

      if (spv_result_t error =
              ValidateSingleBuiltInAtDefinition(decoration, *inst)) {
        return error;
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::Run() {
  // First pass: validate all built-ins at definition and seed
  // id_to_at_reference_checks_ with built-ins.
  if (auto error = ValidateBuiltInsAtDefinition()) {
    return error;
  }

  if (id_to_at_reference_checks_.empty()) {
    // No validation tasks were seeded. Nothing else to do.
    return SPV_SUCCESS;
  }

  // Second pass: validate every id reference in the module using
  // rules in id_to_at_reference_checks_.
  for (const Instruction& inst : _.ordered_instructions()) {
    Update(inst);

    std::set<uint32_t> already_checked;

    for (const auto& operand : inst.operands()) {
      if (!spvIsIdType(operand.type)) {
        // Not id.
        continue;
      }

      const uint32_t id = inst.word(operand.offset);
      if (id == inst.id()) {
        // No need to check result id.
        continue;
      }

      if (!already_checked.insert(id).second) {
        // The instruction has already referenced this id.
        continue;
      }

      // Instruction references the id. Run all checks associated with the id
      // on the instruction. id_to_at_reference_checks_ can be modified in the
      // process, iterators are safe because it's a tree-based map.
      const auto it = id_to_at_reference_checks_.find(id);
      if (it != id_to_at_reference_checks_.end()) {
        for (const auto& check : it->second) {
          if (spv_result_t error = check(inst)) {
            return error;
          }
        }
      }
    }
  }

  return SPV_SUCCESS;
}

}  // namespace

// Validates correctness of built-in variables.
spv_result_t ValidateBuiltIns(ValidationState_t& _) {
  if (!spvIsVulkanOrWebGPUEnv(_.context()->target_env)) {
    // Early return. All currently implemented rules are based on Vulkan or
    // WebGPU spec.
    //
    // TODO: If you are adding validation rules for environments other than
    // Vulkan or WebGPU (or general rules which are not environment
    // independent), then you need to modify or remove this condition. Consider
    // also adding early returns into BuiltIn-specific rules, so that the system
    // doesn't spawn new rules which don't do anything.
    return SPV_SUCCESS;
  }

  BuiltInsValidator validator(_);
  return validator.Run();
}

}  // namespace val
}  // namespace spvtools
