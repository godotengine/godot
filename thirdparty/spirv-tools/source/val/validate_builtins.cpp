// Copyright (c) 2018 Google LLC.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights
// reserved.
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

#include <array>
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
#include "source/val/validate.h"
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
    if (inst.opcode() != spv::Op::OpTypeStruct) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << GetIdDesc(inst)
             << "Attempted to get underlying data type via member index for "
                "non-struct type.";
    }
    *underlying_type = inst.word(decoration.struct_member_index() + 2);
    return SPV_SUCCESS;
  }

  if (inst.opcode() == spv::Op::OpTypeStruct) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << GetIdDesc(inst)
           << " did not find an member index to get underlying data type for "
              "struct type.";
  }

  if (spvOpcodeIsConstant(inst.opcode())) {
    *underlying_type = inst.type_id();
    return SPV_SUCCESS;
  }

  spv::StorageClass storage_class;
  if (!_.GetPointerTypeInfo(inst.type_id(), underlying_type, &storage_class)) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << GetIdDesc(inst)
           << " is decorated with BuiltIn. BuiltIn decoration should only be "
              "applied to struct types, variables and constants.";
  }
  return SPV_SUCCESS;
}

// Returns Storage Class used by the instruction if applicable.
// Returns spv::StorageClass::Max if not.
spv::StorageClass GetStorageClass(const Instruction& inst) {
  switch (inst.opcode()) {
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeForwardPointer: {
      return spv::StorageClass(inst.word(2));
    }
    case spv::Op::OpVariable: {
      return spv::StorageClass(inst.word(3));
    }
    case spv::Op::OpGenericCastToPtrExplicit: {
      return spv::StorageClass(inst.word(4));
    }
    default: { break; }
  }
  return spv::StorageClass::Max;
}

typedef enum VUIDError_ {
  VUIDErrorExecutionModel = 0,
  VUIDErrorStorageClass = 1,
  VUIDErrorType = 2,
  VUIDErrorMax,
} VUIDError;

const static uint32_t NumVUIDBuiltins = 36;

typedef struct {
  spv::BuiltIn builtIn;
  uint32_t vuid[VUIDErrorMax];  // execution mode, storage class, type VUIDs
} BuiltinVUIDMapping;

std::array<BuiltinVUIDMapping, NumVUIDBuiltins> builtinVUIDInfo = {{
    // clang-format off
    {spv::BuiltIn::SubgroupEqMask,            {0,    4370, 4371}},
    {spv::BuiltIn::SubgroupGeMask,            {0,    4372, 4373}},
    {spv::BuiltIn::SubgroupGtMask,            {0,    4374, 4375}},
    {spv::BuiltIn::SubgroupLeMask,            {0,    4376, 4377}},
    {spv::BuiltIn::SubgroupLtMask,            {0,    4378, 4379}},
    {spv::BuiltIn::SubgroupLocalInvocationId, {0,    4380, 4381}},
    {spv::BuiltIn::SubgroupSize,              {0,    4382, 4383}},
    {spv::BuiltIn::GlobalInvocationId,        {4236, 4237, 4238}},
    {spv::BuiltIn::LocalInvocationId,         {4281, 4282, 4283}},
    {spv::BuiltIn::NumWorkgroups,             {4296, 4297, 4298}},
    {spv::BuiltIn::NumSubgroups,              {4293, 4294, 4295}},
    {spv::BuiltIn::SubgroupId,                {4367, 4368, 4369}},
    {spv::BuiltIn::WorkgroupId,               {4422, 4423, 4424}},
    {spv::BuiltIn::HitKindKHR,                {4242, 4243, 4244}},
    {spv::BuiltIn::HitTNV,                    {4245, 4246, 4247}},
    {spv::BuiltIn::InstanceCustomIndexKHR,    {4251, 4252, 4253}},
    {spv::BuiltIn::InstanceId,                {4254, 4255, 4256}},
    {spv::BuiltIn::RayGeometryIndexKHR,       {4345, 4346, 4347}},
    {spv::BuiltIn::ObjectRayDirectionKHR,     {4299, 4300, 4301}},
    {spv::BuiltIn::ObjectRayOriginKHR,        {4302, 4303, 4304}},
    {spv::BuiltIn::ObjectToWorldKHR,          {4305, 4306, 4307}},
    {spv::BuiltIn::WorldToObjectKHR,          {4434, 4435, 4436}},
    {spv::BuiltIn::IncomingRayFlagsKHR,       {4248, 4249, 4250}},
    {spv::BuiltIn::RayTminKHR,                {4351, 4352, 4353}},
    {spv::BuiltIn::RayTmaxKHR,                {4348, 4349, 4350}},
    {spv::BuiltIn::WorldRayDirectionKHR,      {4428, 4429, 4430}},
    {spv::BuiltIn::WorldRayOriginKHR,         {4431, 4432, 4433}},
    {spv::BuiltIn::LaunchIdKHR,               {4266, 4267, 4268}},
    {spv::BuiltIn::LaunchSizeKHR,             {4269, 4270, 4271}},
    {spv::BuiltIn::FragInvocationCountEXT,    {4217, 4218, 4219}},
    {spv::BuiltIn::FragSizeEXT,               {4220, 4221, 4222}},
    {spv::BuiltIn::FragStencilRefEXT,         {4223, 4224, 4225}},
    {spv::BuiltIn::FullyCoveredEXT,           {4232, 4233, 4234}},
    {spv::BuiltIn::CullMaskKHR,               {6735, 6736, 6737}},
    {spv::BuiltIn::BaryCoordKHR,              {4154, 4155, 4156}},
    {spv::BuiltIn::BaryCoordNoPerspKHR,       {4160, 4161, 4162}},
    // clang-format off
} };

uint32_t GetVUIDForBuiltin(spv::BuiltIn builtIn, VUIDError type) {
  uint32_t vuid = 0;
  for (const auto& iter: builtinVUIDInfo) {
    if (iter.builtIn == builtIn) {
      assert(type < VUIDErrorMax);
      vuid = iter.vuid[type];
      break;
    }
  }
  return vuid;
}

bool IsExecutionModelValidForRtBuiltIn(spv::BuiltIn builtin,
                                       spv::ExecutionModel stage) {
  switch (builtin) {
    case spv::BuiltIn::HitKindKHR:
    case spv::BuiltIn::HitTNV:
      if (stage == spv::ExecutionModel::AnyHitKHR ||
          stage == spv::ExecutionModel::ClosestHitKHR) {
        return true;
      }
      break;
    case spv::BuiltIn::InstanceCustomIndexKHR:
    case spv::BuiltIn::InstanceId:
    case spv::BuiltIn::RayGeometryIndexKHR:
    case spv::BuiltIn::ObjectRayDirectionKHR:
    case spv::BuiltIn::ObjectRayOriginKHR:
    case spv::BuiltIn::ObjectToWorldKHR:
    case spv::BuiltIn::WorldToObjectKHR:
      switch (stage) {
        case spv::ExecutionModel::IntersectionKHR:
        case spv::ExecutionModel::AnyHitKHR:
        case spv::ExecutionModel::ClosestHitKHR:
          return true;
        default:
          return false;
      }
      break;
    case spv::BuiltIn::IncomingRayFlagsKHR:
    case spv::BuiltIn::RayTminKHR:
    case spv::BuiltIn::RayTmaxKHR:
    case spv::BuiltIn::WorldRayDirectionKHR:
    case spv::BuiltIn::WorldRayOriginKHR:
    case spv::BuiltIn::CullMaskKHR:
      switch (stage) {
        case spv::ExecutionModel::IntersectionKHR:
        case spv::ExecutionModel::AnyHitKHR:
        case spv::ExecutionModel::ClosestHitKHR:
        case spv::ExecutionModel::MissKHR:
          return true;
        default:
          return false;
      }
      break;
    case spv::BuiltIn::LaunchIdKHR:
    case spv::BuiltIn::LaunchSizeKHR:
      switch (stage) {
        case spv::ExecutionModel::RayGenerationKHR:
        case spv::ExecutionModel::IntersectionKHR:
        case spv::ExecutionModel::AnyHitKHR:
        case spv::ExecutionModel::ClosestHitKHR:
        case spv::ExecutionModel::MissKHR:
        case spv::ExecutionModel::CallableKHR:
          return true;
        default:
          return false;
      }
      break;
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
  spv_result_t ValidateVertexIdAtDefinition(const Decoration& decoration,
                                            const Instruction& inst);
  spv_result_t ValidateLocalInvocationIndexAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateWorkgroupSizeAtDefinition(const Decoration& decoration,
                                                 const Instruction& inst);
  spv_result_t ValidateBaseInstanceOrVertexAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateDrawIndexAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidateViewIndexAtDefinition(const Decoration& decoration,
                                             const Instruction& inst);
  spv_result_t ValidateDeviceIndexAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateFragInvocationCountAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateFragSizeAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateFragStencilRefAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  spv_result_t ValidateFullyCoveredAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);
  // Used for GlobalInvocationId, LocalInvocationId, NumWorkgroups, WorkgroupId.
  spv_result_t ValidateComputeShaderI32Vec3InputAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  spv_result_t ValidateNVSMOrARMCoreBuiltinsAtDefinition(const Decoration& decoration,
                                              const Instruction& inst);
  // Used for BaryCoord, BaryCoordNoPersp.
  spv_result_t ValidateFragmentShaderF32Vec3InputAtDefinition(
      const Decoration& decoration, const Instruction& inst);
  // Used for SubgroupEqMask, SubgroupGeMask, SubgroupGtMask, SubgroupLtMask,
  // SubgroupLeMask.
  spv_result_t ValidateI32Vec4InputAtDefinition(const Decoration& decoration,
                                                const Instruction& inst);
  // Used for SubgroupLocalInvocationId, SubgroupSize.
  spv_result_t ValidateI32InputAtDefinition(const Decoration& decoration,
                                            const Instruction& inst);
  // Used for SubgroupId, NumSubgroups.
  spv_result_t ValidateComputeI32InputAtDefinition(const Decoration& decoration,
                                                   const Instruction& inst);

  spv_result_t ValidatePrimitiveShadingRateAtDefinition(
      const Decoration& decoration, const Instruction& inst);

  spv_result_t ValidateShadingRateAtDefinition(const Decoration& decoration,
                                               const Instruction& inst);

  spv_result_t ValidateRayTracingBuiltinsAtDefinition(
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

  spv_result_t ValidateBaseInstanceOrVertexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateDrawIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateViewIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateDeviceIndexAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFragInvocationCountAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFragSizeAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFragStencilRefAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateFullyCoveredAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Used for GlobalInvocationId, LocalInvocationId, NumWorkgroups, WorkgroupId.
  spv_result_t ValidateComputeShaderI32Vec3InputAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Used for BaryCoord, BaryCoordNoPersp.
  spv_result_t ValidateFragmentShaderF32Vec3InputAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Used for SubgroupId and NumSubgroups.
  spv_result_t ValidateComputeI32InputAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateNVSMOrARMCoreBuiltinsAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidatePrimitiveShadingRateAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateShadingRateAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  spv_result_t ValidateRayTracingBuiltinsAtReference(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // Validates that |built_in_inst| is not (even indirectly) referenced from
  // within a function which can be called with |execution_model|.
  //
  // |vuid| - Vulkan ID for the error, or a negative value if none.
  // |comment| - text explaining why the restriction was imposed.
  // |decoration| - BuiltIn decoration which causes the restriction.
  // |referenced_inst| - instruction which is dependent on |built_in_inst| and
  //                     defines the id which was referenced.
  // |referenced_from_inst| - instruction which references id defined by
  //                          |referenced_inst| from within a function.
  spv_result_t ValidateNotCalledWithExecutionModel(
      int vuid, const char* comment, spv::ExecutionModel execution_model,
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst);

  // The following section contains functions which check that the decorated
  // variable has the type specified in the function name. |diag| would be
  // called with a corresponding error message, if validation is not successful.
  spv_result_t ValidateBool(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateI(
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
  spv_result_t ValidateOptionalArrayedI32(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag);
  spv_result_t ValidateI32Helper(
      const Decoration& decoration, const Instruction& inst,
      const std::function<spv_result_t(const std::string& message)>& diag,
      uint32_t underlying_type);
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
  spv_result_t ValidateF32Mat(
      const Decoration& decoration, const Instruction& inst,
      uint32_t req_num_rows, uint32_t req_num_columns,
      const std::function<spv_result_t(const std::string& message)>& diag);

  // Generates strings like "Member #0 of struct ID <2>".
  std::string GetDefinitionDesc(const Decoration& decoration,
                                const Instruction& inst) const;

  // Generates strings like "ID <51> (OpTypePointer) is referencing ID <2>
  // (OpTypeStruct) which is decorated with BuiltIn Position".
  std::string GetReferenceDesc(
      const Decoration& decoration, const Instruction& built_in_inst,
      const Instruction& referenced_inst,
      const Instruction& referenced_from_inst,
      spv::ExecutionModel execution_model = spv::ExecutionModel::Max) const;

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
  std::set<spv::ExecutionModel> execution_models_;
};

void BuiltInsValidator::Update(const Instruction& inst) {
  const spv::Op opcode = inst.opcode();
  if (opcode == spv::Op::OpFunction) {
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

  if (opcode == spv::Op::OpFunctionEnd) {
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
    assert(inst.opcode() == spv::Op::OpTypeStruct);
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
    spv::ExecutionModel execution_model) const {
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
    if (execution_model != spv::ExecutionModel::Max) {
      ss << " called with execution model ";
      ss << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_EXECUTION_MODEL,
                                          uint32_t(execution_model));
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
                                      uint32_t(GetStorageClass(inst)));
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

spv_result_t BuiltInsValidator::ValidateI(
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

  return ValidateI32Helper(decoration, inst, diag, underlying_type);
}

spv_result_t BuiltInsValidator::ValidateOptionalArrayedI32(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }

  // Strip the array, if present.
  if (_.GetIdOpcode(underlying_type) == spv::Op::OpTypeArray) {
    underlying_type = _.FindDef(underlying_type)->word(2u);
  }

  return ValidateI32Helper(decoration, inst, diag, underlying_type);
}

spv_result_t BuiltInsValidator::ValidateI32Helper(
    const Decoration& decoration, const Instruction& inst,
    const std::function<spv_result_t(const std::string& message)>& diag,
    uint32_t underlying_type) {
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
  if (_.GetIdOpcode(underlying_type) == spv::Op::OpTypeArray) {
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
  if (_.GetIdOpcode(underlying_type) == spv::Op::OpTypeArray) {
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
  if (type_inst->opcode() != spv::Op::OpTypeArray) {
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
  if (_.GetIdOpcode(underlying_type) == spv::Op::OpTypeArray) {
    uint32_t subtype = _.FindDef(underlying_type)->word(2u);
    if (_.GetIdOpcode(subtype) == spv::Op::OpTypeArray) {
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
  if (type_inst->opcode() != spv::Op::OpTypeArray) {
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

spv_result_t BuiltInsValidator::ValidateF32Mat(
    const Decoration& decoration, const Instruction& inst,
    uint32_t req_num_rows, uint32_t req_num_columns,
    const std::function<spv_result_t(const std::string& message)>& diag) {
  uint32_t underlying_type = 0;
  uint32_t num_rows = 0;
  uint32_t num_cols = 0;
  uint32_t col_type = 0;
  uint32_t component_type = 0;
  if (spv_result_t error =
          GetUnderlyingType(_, decoration, inst, &underlying_type)) {
    return error;
  }
  if (!_.GetMatrixTypeInfo(underlying_type, &num_rows, &num_cols, &col_type,
                           &component_type) ||
      num_rows != req_num_rows || num_cols != req_num_columns) {
    std::ostringstream ss;
    ss << GetDefinitionDesc(decoration, inst) << " has columns " << num_cols
       << " and rows " << num_rows << " not equal to expected "
       << req_num_columns << "x" << req_num_rows << ".";
    return diag(ss.str());
  }

  return ValidateF32VecHelper(decoration, inst, req_num_rows, diag, col_type);
}

spv_result_t BuiltInsValidator::ValidateNotCalledWithExecutionModel(
    int vuid, const char* comment, spv::ExecutionModel execution_model,
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (function_id_) {
    if (execution_models_.count(execution_model)) {
      const char* execution_model_str = _.grammar().lookupOperandName(
          SPV_OPERAND_TYPE_EXECUTION_MODEL, uint32_t(execution_model));
      const char* built_in_str = _.grammar().lookupOperandName(
          SPV_OPERAND_TYPE_BUILT_IN, decoration.params()[0]);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << (vuid < 0 ? std::string("") : _.VkErrorID(vuid)) << comment
             << " " << GetIdDesc(referenced_inst) << " depends on "
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
                  vuid, comment, execution_model, decoration, built_in_inst,
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
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance) ? 4190 : 4199;
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Input) {
      assert(function_id_ == 0);
      uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance) ? 4188 : 4197;
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Input storage class if execution model is "
          "Vertex.",
          spv::ExecutionModel::Vertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Input storage class if execution model is "
          "MeshNV.",
          spv::ExecutionModel::MeshNV, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Input storage class if execution model is "
          "MeshEXT.",
          spv::ExecutionModel::MeshEXT, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    if (storage_class == spv::StorageClass::Output) {
      assert(function_id_ == 0);
      uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance) ? 4189 : 4198;
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow BuiltIn ClipDistance/CullDistance to be "
          "used for variables with Output storage class if execution model is "
          "Fragment.",
          spv::ExecutionModel::Fragment, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Fragment:
        case spv::ExecutionModel::Vertex: {
          if (spv_result_t error = ValidateF32Arr(
                  decoration, built_in_inst, /* Any number of components */ 0,
                  [this, &decoration, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    uint32_t vuid =
                        (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance)
                            ? 4191
                            : 4200;
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << _.VkErrorID(vuid)
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
        case spv::ExecutionModel::TessellationControl:
        case spv::ExecutionModel::TessellationEvaluation:
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT: {
          if (decoration.struct_member_index() != Decoration::kInvalidMember) {
            // The outer level of array is applied on the variable.
            if (spv_result_t error = ValidateF32Arr(
                    decoration, built_in_inst, /* Any number of components */ 0,
                    [this, &decoration, &referenced_from_inst](
                        const std::string& message) -> spv_result_t {
                      uint32_t vuid =
                          (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance)
                              ? 4191
                              : 4200;
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << _.VkErrorID(vuid)
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
                      uint32_t vuid =
                          (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance)
                              ? 4191
                              : 4200;
                      return _.diag(SPV_ERROR_INVALID_DATA,
                                    &referenced_from_inst)
                             << _.VkErrorID(vuid)
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
          uint32_t vuid =
              (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::ClipDistance) ? 4187 : 4196;
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  operand)
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 4,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4212) << "According to the "
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4211) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FragCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4210)
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateF32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4215) << "According to the "
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4214) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FragDepth to be only used for "
                "variables with Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4213)
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
      if (!modes || !modes->count(spv::ExecutionMode::DepthReplacing)) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4216)
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateBool(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4231) << "According to the "
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4230) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn FrontFacing to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4229)
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
                     << _.VkErrorID(4241)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4240)
             << "Vulkan spec allows BuiltIn HelperInvocation to be only used "
                "for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4239)
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
                     << _.VkErrorID(4259)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4258)
             << "Vulkan spec allows BuiltIn InvocationId to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::TessellationControl &&
          execution_model != spv::ExecutionModel::Geometry) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4257)
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4265) << "According to the "
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4264) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn InstanceIndex to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Vertex) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4263)
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
                     << _.VkErrorID(4310)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4309)
             << "Vulkan spec allows BuiltIn PatchVertices to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::TessellationControl &&
          execution_model != spv::ExecutionModel::TessellationEvaluation) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4308)
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
                     << _.VkErrorID(4313)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4312)
             << "Vulkan spec allows BuiltIn PointCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4311)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4316)
             << "Vulkan spec allows BuiltIn PointSize to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Input) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4315,
          "Vulkan spec doesn't allow BuiltIn PointSize to be used for "
          "variables with Input storage class if execution model is "
          "Vertex.",
          spv::ExecutionModel::Vertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Vertex: {
          if (spv_result_t error = ValidateF32(
                  decoration, built_in_inst,
                  [this, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << _.VkErrorID(4317)
                           << "According to the Vulkan spec BuiltIn PointSize "
                              "variable needs to be a 32-bit float scalar. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        case spv::ExecutionModel::TessellationControl:
        case spv::ExecutionModel::TessellationEvaluation:
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT: {
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
                             << _.VkErrorID(4317)
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
                             << _.VkErrorID(4317)
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
                 << _.VkErrorID(4314)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4320) << "Vulkan spec allows BuiltIn Position to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Input) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4319,
          "Vulkan spec doesn't allow BuiltIn Position to be used "
          "for variables "
          "with Input storage class if execution model is Vertex.",
          spv::ExecutionModel::Vertex, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4319,
          "Vulkan spec doesn't allow BuiltIn Position to be used "
          "for variables "
          "with Input storage class if execution model is MeshNV.",
          spv::ExecutionModel::MeshNV, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4319,
          "Vulkan spec doesn't allow BuiltIn Position to be used "
          "for variables "
          "with Input storage class if execution model is MeshEXT.",
          spv::ExecutionModel::MeshEXT, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Vertex: {
          if (spv_result_t error = ValidateF32Vec(
                  decoration, built_in_inst, 4,
                  [this, &referenced_from_inst](
                      const std::string& message) -> spv_result_t {
                    return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                           << _.VkErrorID(4321)
                           << "According to the Vulkan spec BuiltIn Position "
                              "variable needs to be a 4-component 32-bit float "
                              "vector. "
                           << message;
                  })) {
            return error;
          }
          break;
        }
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::TessellationControl:
        case spv::ExecutionModel::TessellationEvaluation:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT: {
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
                             << _.VkErrorID(4321)
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
                             << _.VkErrorID(4321)
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
                 << _.VkErrorID(4318)
                 << "Vulkan spec allows BuiltIn Position to be used only "
                    "with Vertex, TessellationControl, TessellationEvaluation"
                    " or Geometry execution models. "
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
    // PrimitiveId can be a per-primitive variable for mesh shader stage.
    // In such cases variable will have an array of 32-bit integers.
    if (decoration.struct_member_index() != Decoration::kInvalidMember) {
      // This must be a 32-bit int scalar.
      if (spv_result_t error = ValidateI32(
              decoration, inst,
              [this, &inst](const std::string& message) -> spv_result_t {
                return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                       << _.VkErrorID(4337)
                       << "According to the Vulkan spec BuiltIn PrimitiveId "
                          "variable needs to be a 32-bit int scalar. "
                       << message;
              })) {
        return error;
      }
    } else {
      if (spv_result_t error = ValidateOptionalArrayedI32(
              decoration, inst,
              [this, &inst](const std::string& message) -> spv_result_t {
                return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                       << _.VkErrorID(4337)
                       << "According to the Vulkan spec BuiltIn PrimitiveId "
                          "variable needs to be a 32-bit int scalar. "
                       << message;
              })) {
        return error;
      }
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn PrimitiveId to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Output) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "TessellationControl.",
          spv::ExecutionModel::TessellationControl, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "TessellationEvaluation.",
          spv::ExecutionModel::TessellationEvaluation, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "Fragment.",
          spv::ExecutionModel::Fragment, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "IntersectionKHR.",
          spv::ExecutionModel::IntersectionKHR, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "AnyHitKHR.",
          spv::ExecutionModel::AnyHitKHR, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, 4334,
          "Vulkan spec doesn't allow BuiltIn PrimitiveId to be used for "
          "variables with Output storage class if execution model is "
          "ClosestHitKHR.",
          spv::ExecutionModel::ClosestHitKHR, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Fragment:
        case spv::ExecutionModel::TessellationControl:
        case spv::ExecutionModel::TessellationEvaluation:
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT:
        case spv::ExecutionModel::IntersectionKHR:
        case spv::ExecutionModel::AnyHitKHR:
        case spv::ExecutionModel::ClosestHitKHR: {
          // Ok.
          break;
        }

        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << _.VkErrorID(4330)
                 << "Vulkan spec allows BuiltIn PrimitiveId to be used only "
                    "with Fragment, TessellationControl, "
                    "TessellationEvaluation, Geometry, MeshNV, MeshEXT, "
                    "IntersectionKHR, AnyHitKHR, and ClosestHitKHR execution models. "
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
                     << _.VkErrorID(4356)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4355)
             << "Vulkan spec allows BuiltIn SampleId to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4354)
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
                     << _.VkErrorID(4359)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4358)
             << "Vulkan spec allows BuiltIn SampleMask to be only used for "
                "variables with Input or Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4357)
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
                     << _.VkErrorID(4362)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4361)
             << "Vulkan spec allows BuiltIn SamplePosition to be only used "
                "for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4360)
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
                     << _.VkErrorID(4389)
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
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4388)
             << "Vulkan spec allows BuiltIn TessCoord to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::TessellationEvaluation) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4387)
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
                     << _.VkErrorID(4393)
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
                     << _.VkErrorID(4397)
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
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Input) {
      assert(function_id_ == 0);
      uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::TessLevelOuter) ? 4391 : 4395;
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
          "used "
          "for variables with Input storage class if execution model is "
          "TessellationControl.",
          spv::ExecutionModel::TessellationControl, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    if (storage_class == spv::StorageClass::Output) {
      assert(function_id_ == 0);
      uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::TessLevelOuter) ? 4392 : 4396;
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
          &BuiltInsValidator::ValidateNotCalledWithExecutionModel, this, vuid,
          "Vulkan spec doesn't allow TessLevelOuter/TessLevelInner to be "
          "used "
          "for variables with Output storage class if execution model is "
          "TessellationEvaluation.",
          spv::ExecutionModel::TessellationEvaluation, decoration, built_in_inst,
          referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::TessellationControl:
        case spv::ExecutionModel::TessellationEvaluation: {
          // Ok.
          break;
        }

        default: {
          uint32_t vuid = (spv::BuiltIn(operand) == spv::BuiltIn::TessLevelOuter) ? 4390 : 4394;
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  operand)
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4400) << "According to the "
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

spv_result_t BuiltInsValidator::ValidateVertexIdAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  (void)decoration;
  if (spvIsVulkanEnv(_.context()->target_env)) {
    return _.diag(SPV_ERROR_INVALID_DATA, &inst)
           << "Vulkan spec doesn't allow BuiltIn VertexId "
              "to be used.";
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateLocalInvocationIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  // Seed at reference checks with this built-in.
  return ValidateLocalInvocationIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateLocalInvocationIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction&,
    const Instruction& referenced_from_inst) {
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4399) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn VertexIndex to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Vertex) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4398)
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
    // This can be a per-primitive variable for mesh shader stage.
    // In such cases variable will have an array of 32-bit integers.
    if (decoration.struct_member_index() != Decoration::kInvalidMember) {
      // This must be a 32-bit int scalar.
      if (spv_result_t error = ValidateI32(
              decoration, inst,
              [this, &decoration,
               &inst](const std::string& message) -> spv_result_t {
                uint32_t vuid =
                    (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::Layer) ? 4276 : 4408;
                return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                       << _.VkErrorID(vuid)
                       << "According to the Vulkan spec BuiltIn "
                       << _.grammar().lookupOperandName(
                              SPV_OPERAND_TYPE_BUILT_IN, decoration.params()[0])
                       << "variable needs to be a 32-bit int scalar. "
                       << message;
              })) {
        return error;
      }
    } else {
      if (spv_result_t error = ValidateOptionalArrayedI32(
              decoration, inst,
              [this, &decoration,
               &inst](const std::string& message) -> spv_result_t {
                uint32_t vuid =
                    (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::Layer) ? 4276 : 4408;
                return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                       << _.VkErrorID(vuid)
                       << "According to the Vulkan spec BuiltIn "
                       << _.grammar().lookupOperandName(
                              SPV_OPERAND_TYPE_BUILT_IN, decoration.params()[0])
                       << "variable needs to be a 32-bit int scalar. "
                       << message;
              })) {
        return error;
      }
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateLayerOrViewportIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateLayerOrViewportIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input or Output storage "
                "class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    if (storage_class == spv::StorageClass::Input) {
      assert(function_id_ == 0);
      for (const auto em :
           {spv::ExecutionModel::Vertex, spv::ExecutionModel::TessellationEvaluation,
            spv::ExecutionModel::Geometry, spv::ExecutionModel::MeshNV,
            spv::ExecutionModel::MeshEXT}) {
        id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
            std::bind(&BuiltInsValidator::ValidateNotCalledWithExecutionModel,
                      this, ((spv::BuiltIn(operand) == spv::BuiltIn::Layer) ? 4274 : 4406),
                      "Vulkan spec doesn't allow BuiltIn Layer and "
                      "ViewportIndex to be "
                      "used for variables with Input storage class if "
                      "execution model is Vertex, TessellationEvaluation, "
                      "Geometry, MeshNV or MeshEXT.",
                      em, decoration, built_in_inst, referenced_from_inst,
                      std::placeholders::_1));
      }
    }

    if (storage_class == spv::StorageClass::Output) {
      assert(function_id_ == 0);
      id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
          std::bind(&BuiltInsValidator::ValidateNotCalledWithExecutionModel,
                    this, ((spv::BuiltIn(operand) == spv::BuiltIn::Layer) ? 4275 : 4407),
                    "Vulkan spec doesn't allow BuiltIn Layer and "
                    "ViewportIndex to be "
                    "used for variables with Output storage class if "
                    "execution model is "
                    "Fragment.",
                    spv::ExecutionModel::Fragment, decoration, built_in_inst,
                    referenced_from_inst, std::placeholders::_1));
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::Fragment:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT:
          // Ok.
          break;
        case spv::ExecutionModel::Vertex:
        case spv::ExecutionModel::TessellationEvaluation: {
          if (!_.HasCapability(spv::Capability::ShaderViewportIndexLayerEXT)) {
            if (spv::BuiltIn(operand) == spv::BuiltIn::ViewportIndex &&
                _.HasCapability(spv::Capability::ShaderViewportIndex))
              break;  // Ok
            if (spv::BuiltIn(operand) == spv::BuiltIn::Layer &&
                _.HasCapability(spv::Capability::ShaderLayer))
              break;  // Ok

            const char* capability = "ShaderViewportIndexLayerEXT";

            if (spv::BuiltIn(operand) == spv::BuiltIn::ViewportIndex)
              capability = "ShaderViewportIndexLayerEXT or ShaderViewportIndex";
            if (spv::BuiltIn(operand) == spv::BuiltIn::Layer)
              capability = "ShaderViewportIndexLayerEXT or ShaderLayer";

            uint32_t vuid = (spv::BuiltIn(operand) == spv::BuiltIn::Layer) ? 4273 : 4405;
            return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                   << _.VkErrorID(vuid) << "Using BuiltIn "
                   << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                    operand)
                   << " in Vertex or Tessellation execution model requires the "
                   << capability << " capability.";
          }
          break;
        }
        default: {
          uint32_t vuid = (spv::BuiltIn(operand) == spv::BuiltIn::Layer) ? 4272 : 4404;
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  operand)
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

spv_result_t BuiltInsValidator::ValidateFragmentShaderF32Vec3InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateF32Vec(
            decoration, inst, 3,
            [this, &inst, builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
                     << " variable needs to be a 3-component 32-bit float "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateFragmentShaderF32Vec3InputAtReference(decoration, inst, inst,
                                                      inst);
}

spv_result_t BuiltInsValidator::ValidateFragmentShaderF32Vec3InputAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragmentShaderF32Vec3InputAtReference, this,
        decoration, built_in_inst, referenced_from_inst,
        std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateComputeShaderI32Vec3InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 3,
            [this, &inst, builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      bool has_vulkan_model = execution_model == spv::ExecutionModel::GLCompute ||
                              execution_model == spv::ExecutionModel::TaskNV ||
                              execution_model == spv::ExecutionModel::MeshNV ||
                              execution_model == spv::ExecutionModel::TaskEXT ||
                              execution_model == spv::ExecutionModel::MeshEXT;

      if (spvIsVulkanEnv(_.context()->target_env) && !has_vulkan_model) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with GLCompute, MeshNV, TaskNV, MeshEXT or"
               << " TaskEXT execution model. "
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

spv_result_t BuiltInsValidator::ValidateComputeI32InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (decoration.struct_member_index() != Decoration::kInvalidMember) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << "BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " cannot be used as a member decoration ";
    }
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst, builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                     << " variable needs to be a 32-bit int "
                        "vector. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateComputeI32InputAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateComputeI32InputAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      bool has_vulkan_model = execution_model == spv::ExecutionModel::GLCompute ||
                              execution_model == spv::ExecutionModel::TaskNV ||
                              execution_model == spv::ExecutionModel::MeshNV ||
                              execution_model == spv::ExecutionModel::TaskEXT ||
                              execution_model == spv::ExecutionModel::MeshEXT;
      if (spvIsVulkanEnv(_.context()->target_env) && !has_vulkan_model) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with GLCompute, MeshNV, TaskNV, MeshEXT or "
               << "TaskEXT execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateComputeI32InputAtReference, this,
                  decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateI32InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (decoration.struct_member_index() != Decoration::kInvalidMember) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << "BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " cannot be used as a member decoration ";
    }
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst, builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                     << " variable needs to be a 32-bit int. " << message;
            })) {
      return error;
    }

    const spv::StorageClass storage_class = GetStorageClass(inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << _.VkErrorID(vuid)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, inst, inst, inst) << " "
             << GetStorageClassDesc(inst);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateI32Vec4InputAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (decoration.struct_member_index() != Decoration::kInvalidMember) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << "BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " cannot be used as a member decoration ";
    }
    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 4,
            [this, &inst, builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                     << " variable needs to be a 4-component 32-bit int "
                        "vector. "
                     << message;
            })) {
      return error;
    }

    const spv::StorageClass storage_class = GetStorageClass(inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << _.VkErrorID(vuid)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, inst, inst, inst) << " "
             << GetStorageClassDesc(inst);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateWorkgroupSizeAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spvIsVulkanEnv(_.context()->target_env) &&
        !spvOpcodeIsConstant(inst.opcode())) {
      return _.diag(SPV_ERROR_INVALID_DATA, &inst)
             << _.VkErrorID(4426)
             << "Vulkan spec requires BuiltIn WorkgroupSize to be a "
                "constant. "
             << GetIdDesc(inst) << " is not a constant.";
    }

    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 3,
            [this, &inst](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4427) << "According to the "
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
  if (spvIsVulkanEnv(_.context()->target_env)) {
    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::GLCompute &&
          execution_model != spv::ExecutionModel::TaskNV &&
          execution_model != spv::ExecutionModel::MeshNV &&
          execution_model != spv::ExecutionModel::TaskEXT &&
          execution_model != spv::ExecutionModel::MeshEXT) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4425)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                decoration.params()[0])
               << " to be used only with GLCompute, MeshNV, TaskNV, MeshEXT or "
               << "TaskEXT execution model. "
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

spv_result_t BuiltInsValidator::ValidateBaseInstanceOrVertexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              uint32_t vuid = (spv::BuiltIn(decoration.params()[0]) == spv::BuiltIn::BaseInstance)
                                  ? 4183
                                  : 4186;
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateBaseInstanceOrVertexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateBaseInstanceOrVertexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = (spv::BuiltIn(operand) == spv::BuiltIn::BaseInstance) ? 4182 : 4185;
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Vertex) {
        uint32_t vuid = (spv::BuiltIn(operand) == spv::BuiltIn::BaseInstance) ? 4181 : 4184;
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                operand)
               << " to be used only with Vertex execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateBaseInstanceOrVertexAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateDrawIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4209)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateDrawIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateDrawIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4208) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Vertex &&
          execution_model != spv::ExecutionModel::MeshNV &&
          execution_model != spv::ExecutionModel::TaskNV &&
          execution_model != spv::ExecutionModel::MeshEXT &&
          execution_model != spv::ExecutionModel::TaskEXT) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4207) << "Vulkan spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                operand)
               << " to be used only with Vertex, MeshNV, TaskNV , MeshEXT or"
               << " TaskEXT execution "
                  "model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateDrawIndexAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateViewIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4403)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateViewIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateViewIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4402) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model == spv::ExecutionModel::GLCompute) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4401) << "Vulkan spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                operand)
               << " to be not be used with GLCompute execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateViewIndexAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateDeviceIndexAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4206)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateDeviceIndexAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateDeviceIndexAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  uint32_t operand = decoration.params()[0];
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4205) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              operand)
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateDeviceIndexAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFragInvocationCountAtDefinition(const Decoration& decoration,
                                            const Instruction& inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst, &builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateFragInvocationCountAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFragInvocationCountAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragInvocationCountAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFragSizeAtDefinition(const Decoration& decoration,
                                            const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateI32Vec(
            decoration, inst, 2,
            [this, &inst, &builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
                     << " variable needs to be a 2-component 32-bit int vector. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateFragSizeAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFragSizeAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragSizeAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFragStencilRefAtDefinition(const Decoration& decoration,
                                            const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateI(
            decoration, inst,
            [this, &inst, &builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
                     << " variable needs to be a int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateFragStencilRefAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFragStencilRefAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Output) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFragStencilRefAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateFullyCoveredAtDefinition(const Decoration& decoration,
                                               const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    if (spv_result_t error = ValidateBool(
            decoration, inst,
            [this, &inst, &builtin](const std::string& message) -> spv_result_t {
              uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(vuid) << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      uint32_t(builtin))
                     << " variable needs to be a bool scalar. "
                     << message;
            })) {
      return error;
    }
  }

  return ValidateFullyCoveredAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateFullyCoveredAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {

  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid)
               << spvLogStringForEnv(_.context()->target_env)
               << " spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
               << " to be used only with Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateFullyCoveredAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateNVSMOrARMCoreBuiltinsAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "According to the "
                     << spvLogStringForEnv(_.context()->target_env)
                     << " spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateNVSMOrARMCoreBuiltinsAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateNVSMOrARMCoreBuiltinsAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << spvLogStringForEnv(_.context()->target_env)
             << " spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for "
                "variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateNVSMOrARMCoreBuiltinsAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidatePrimitiveShadingRateAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4486)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidatePrimitiveShadingRateAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidatePrimitiveShadingRateAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Output) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4485) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Output storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      switch (execution_model) {
        case spv::ExecutionModel::Vertex:
        case spv::ExecutionModel::Geometry:
        case spv::ExecutionModel::MeshNV:
        case spv::ExecutionModel::MeshEXT:
          break;
        default: {
          return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
                 << _.VkErrorID(4484) << "Vulkan spec allows BuiltIn "
                 << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                  decoration.params()[0])
                 << " to be used only with Vertex, Geometry, or MeshNV "
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
        std::bind(&BuiltInsValidator::ValidatePrimitiveShadingRateAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateShadingRateAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (spv_result_t error = ValidateI32(
            decoration, inst,
            [this, &inst,
             &decoration](const std::string& message) -> spv_result_t {
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << _.VkErrorID(4492)
                     << "According to the Vulkan spec BuiltIn "
                     << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                      decoration.params()[0])
                     << " variable needs to be a 32-bit int scalar. "
                     << message;
            })) {
      return error;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateShadingRateAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateShadingRateAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(4491) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (execution_model != spv::ExecutionModel::Fragment) {
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(4490) << "Vulkan spec allows BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                decoration.params()[0])
               << " to be used only with the Fragment execution model. "
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(std::bind(
        &BuiltInsValidator::ValidateShadingRateAtReference, this, decoration,
        built_in_inst, referenced_from_inst, std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateRayTracingBuiltinsAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    switch (builtin) {
      case spv::BuiltIn::HitTNV:
      case spv::BuiltIn::RayTminKHR:
      case spv::BuiltIn::RayTmaxKHR:
        // f32 scalar
        if (spv_result_t error = ValidateF32(
                decoration, inst,
                [this, &inst,
                 builtin](const std::string& message) -> spv_result_t {
                  uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
                  return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                         << _.VkErrorID(vuid)
                         << "According to the Vulkan spec BuiltIn "
                         << _.grammar().lookupOperandName(
                                SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                         << " variable needs to be a 32-bit float scalar. "
                         << message;
                })) {
          return error;
        }
        break;
      case spv::BuiltIn::HitKindKHR:
      case spv::BuiltIn::InstanceCustomIndexKHR:
      case spv::BuiltIn::InstanceId:
      case spv::BuiltIn::RayGeometryIndexKHR:
      case spv::BuiltIn::IncomingRayFlagsKHR:
      case spv::BuiltIn::CullMaskKHR:
        // i32 scalar
        if (spv_result_t error = ValidateI32(
                decoration, inst,
                [this, &inst,
                 builtin](const std::string& message) -> spv_result_t {
                  uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
                  return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                         << _.VkErrorID(vuid)
                         << "According to the Vulkan spec BuiltIn "
                         << _.grammar().lookupOperandName(
                                SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                         << " variable needs to be a 32-bit int scalar. "
                         << message;
                })) {
          return error;
        }
        break;
      case spv::BuiltIn::ObjectRayDirectionKHR:
      case spv::BuiltIn::ObjectRayOriginKHR:
      case spv::BuiltIn::WorldRayDirectionKHR:
      case spv::BuiltIn::WorldRayOriginKHR:
        // f32 vec3
        if (spv_result_t error = ValidateF32Vec(
                decoration, inst, 3,
                [this, &inst,
                 builtin](const std::string& message) -> spv_result_t {
                  uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
                  return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                         << _.VkErrorID(vuid)
                         << "According to the Vulkan spec BuiltIn "
                         << _.grammar().lookupOperandName(
                                SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                         << " variable needs to be a 3-component 32-bit float "
                            "vector. "
                         << message;
                })) {
          return error;
        }
        break;
      case spv::BuiltIn::LaunchIdKHR:
      case spv::BuiltIn::LaunchSizeKHR:
        // i32 vec3
        if (spv_result_t error = ValidateI32Vec(
                decoration, inst, 3,
                [this, &inst,
                 builtin](const std::string& message) -> spv_result_t {
                  uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
                  return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                         << _.VkErrorID(vuid)
                         << "According to the Vulkan spec BuiltIn "
                         << _.grammar().lookupOperandName(
                                SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                         << " variable needs to be a 3-component 32-bit int "
                            "vector. "
                         << message;
                })) {
          return error;
        }
        break;
      case spv::BuiltIn::ObjectToWorldKHR:
      case spv::BuiltIn::WorldToObjectKHR:
        // f32 mat4x3
        if (spv_result_t error = ValidateF32Mat(
                decoration, inst, 3, 4,
                [this, &inst,
                 builtin](const std::string& message) -> spv_result_t {
                  uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorType);
                  return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                         << _.VkErrorID(vuid)
                         << "According to the Vulkan spec BuiltIn "
                         << _.grammar().lookupOperandName(
                                SPV_OPERAND_TYPE_BUILT_IN, uint32_t(builtin))
                         << " variable needs to be a matrix with"
                         << " 4 columns of 3-component vectors of 32-bit "
                            "floats. "
                         << message;
                })) {
          return error;
        }
        break;
      default:
        assert(0 && "Unexpected ray tracing builtin");
        break;
    }
  }

  // Seed at reference checks with this built-in.
  return ValidateRayTracingBuiltinsAtReference(decoration, inst, inst, inst);
}

spv_result_t BuiltInsValidator::ValidateRayTracingBuiltinsAtReference(
    const Decoration& decoration, const Instruction& built_in_inst,
    const Instruction& referenced_inst,
    const Instruction& referenced_from_inst) {
  if (spvIsVulkanEnv(_.context()->target_env)) {
    const spv::BuiltIn builtin = spv::BuiltIn(decoration.params()[0]);
    const spv::StorageClass storage_class = GetStorageClass(referenced_from_inst);
    if (storage_class != spv::StorageClass::Max &&
        storage_class != spv::StorageClass::Input) {
      uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorStorageClass);
      return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
             << _.VkErrorID(vuid) << "Vulkan spec allows BuiltIn "
             << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                              decoration.params()[0])
             << " to be only used for variables with Input storage class. "
             << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                 referenced_from_inst)
             << " " << GetStorageClassDesc(referenced_from_inst);
    }

    for (const spv::ExecutionModel execution_model : execution_models_) {
      if (!IsExecutionModelValidForRtBuiltIn(builtin, execution_model)) {
        uint32_t vuid = GetVUIDForBuiltin(builtin, VUIDErrorExecutionModel);
        return _.diag(SPV_ERROR_INVALID_DATA, &referenced_from_inst)
               << _.VkErrorID(vuid) << "Vulkan spec does not allow BuiltIn "
               << _.grammar().lookupOperandName(SPV_OPERAND_TYPE_BUILT_IN,
                                                decoration.params()[0])
               << " to be used with the execution model "
               << _.grammar().lookupOperandName(
                      SPV_OPERAND_TYPE_EXECUTION_MODEL, uint32_t(execution_model))
               << ".\n"
               << GetReferenceDesc(decoration, built_in_inst, referenced_inst,
                                   referenced_from_inst, execution_model);
      }
    }
  }

  if (function_id_ == 0) {
    // Propagate this rule to all dependant ids in the global scope.
    id_to_at_reference_checks_[referenced_from_inst.id()].push_back(
        std::bind(&BuiltInsValidator::ValidateRayTracingBuiltinsAtReference,
                  this, decoration, built_in_inst, referenced_from_inst,
                  std::placeholders::_1));
  }

  return SPV_SUCCESS;
}

spv_result_t BuiltInsValidator::ValidateSingleBuiltInAtDefinition(
    const Decoration& decoration, const Instruction& inst) {
  const spv::BuiltIn label = spv::BuiltIn(decoration.params()[0]);

  if (!spvIsVulkanEnv(_.context()->target_env)) {
    // Early return. All currently implemented rules are based on Vulkan spec.
    //
    // TODO: If you are adding validation rules for environments other than
    // Vulkan (or general rules which are not environment independent), then
    // you need to modify or remove this condition. Consider also adding early
    // returns into BuiltIn-specific rules, so that the system doesn't spawn new
    // rules which don't do anything.
    return SPV_SUCCESS;
  }

  // If you are adding a new BuiltIn enum, please register it here.
  // If the newly added enum has validation rules associated with it
  // consider leaving a TODO and/or creating an issue.
  switch (label) {
    case spv::BuiltIn::ClipDistance:
    case spv::BuiltIn::CullDistance: {
      return ValidateClipOrCullDistanceAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FragCoord: {
      return ValidateFragCoordAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FragDepth: {
      return ValidateFragDepthAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FrontFacing: {
      return ValidateFrontFacingAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::GlobalInvocationId:
    case spv::BuiltIn::LocalInvocationId:
    case spv::BuiltIn::NumWorkgroups:
    case spv::BuiltIn::WorkgroupId: {
      return ValidateComputeShaderI32Vec3InputAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::BaryCoordKHR:
    case spv::BuiltIn::BaryCoordNoPerspKHR: {
      return ValidateFragmentShaderF32Vec3InputAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::HelperInvocation: {
      return ValidateHelperInvocationAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::InvocationId: {
      return ValidateInvocationIdAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::InstanceIndex: {
      return ValidateInstanceIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::Layer:
    case spv::BuiltIn::ViewportIndex: {
      return ValidateLayerOrViewportIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::PatchVertices: {
      return ValidatePatchVerticesAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::PointCoord: {
      return ValidatePointCoordAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::PointSize: {
      return ValidatePointSizeAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::Position: {
      return ValidatePositionAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::PrimitiveId: {
      return ValidatePrimitiveIdAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SampleId: {
      return ValidateSampleIdAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SampleMask: {
      return ValidateSampleMaskAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SamplePosition: {
      return ValidateSamplePositionAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SubgroupId:
    case spv::BuiltIn::NumSubgroups: {
      return ValidateComputeI32InputAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SubgroupLocalInvocationId:
    case spv::BuiltIn::SubgroupSize: {
      return ValidateI32InputAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::SubgroupEqMask:
    case spv::BuiltIn::SubgroupGeMask:
    case spv::BuiltIn::SubgroupGtMask:
    case spv::BuiltIn::SubgroupLeMask:
    case spv::BuiltIn::SubgroupLtMask: {
      return ValidateI32Vec4InputAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::TessCoord: {
      return ValidateTessCoordAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::TessLevelOuter: {
      return ValidateTessLevelOuterAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::TessLevelInner: {
      return ValidateTessLevelInnerAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::VertexIndex: {
      return ValidateVertexIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::WorkgroupSize: {
      return ValidateWorkgroupSizeAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::VertexId: {
      return ValidateVertexIdAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::LocalInvocationIndex: {
      return ValidateLocalInvocationIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::CoreIDARM:
    case spv::BuiltIn::CoreCountARM:
    case spv::BuiltIn::CoreMaxIDARM:
    case spv::BuiltIn::WarpIDARM:
    case spv::BuiltIn::WarpMaxIDARM:
    case spv::BuiltIn::WarpsPerSMNV:
    case spv::BuiltIn::SMCountNV:
    case spv::BuiltIn::WarpIDNV:
    case spv::BuiltIn::SMIDNV: {
      return ValidateNVSMOrARMCoreBuiltinsAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::BaseInstance:
    case spv::BuiltIn::BaseVertex: {
      return ValidateBaseInstanceOrVertexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::DrawIndex: {
      return ValidateDrawIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::ViewIndex: {
      return ValidateViewIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::DeviceIndex: {
      return ValidateDeviceIndexAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FragInvocationCountEXT: {
      // alias spv::BuiltIn::InvocationsPerPixelNV
      return ValidateFragInvocationCountAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FragSizeEXT: {
      // alias spv::BuiltIn::FragmentSizeNV
      return ValidateFragSizeAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FragStencilRefEXT: {
      return ValidateFragStencilRefAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::FullyCoveredEXT:{
      return ValidateFullyCoveredAtDefinition(decoration, inst);
    }
    // Ray tracing builtins
    case spv::BuiltIn::HitKindKHR:  // alias spv::BuiltIn::HitKindNV
    case spv::BuiltIn::HitTNV:      // NOT present in KHR
    case spv::BuiltIn::InstanceId:
    case spv::BuiltIn::LaunchIdKHR:           // alias spv::BuiltIn::LaunchIdNV
    case spv::BuiltIn::LaunchSizeKHR:         // alias spv::BuiltIn::LaunchSizeNV
    case spv::BuiltIn::WorldRayOriginKHR:     // alias spv::BuiltIn::WorldRayOriginNV
    case spv::BuiltIn::WorldRayDirectionKHR:  // alias spv::BuiltIn::WorldRayDirectionNV
    case spv::BuiltIn::ObjectRayOriginKHR:    // alias spv::BuiltIn::ObjectRayOriginNV
    case spv::BuiltIn::ObjectRayDirectionKHR:   // alias
                                            // spv::BuiltIn::ObjectRayDirectionNV
    case spv::BuiltIn::RayTminKHR:              // alias spv::BuiltIn::RayTminNV
    case spv::BuiltIn::RayTmaxKHR:              // alias spv::BuiltIn::RayTmaxNV
    case spv::BuiltIn::InstanceCustomIndexKHR:  // alias
                                            // spv::BuiltIn::InstanceCustomIndexNV
    case spv::BuiltIn::ObjectToWorldKHR:        // alias spv::BuiltIn::ObjectToWorldNV
    case spv::BuiltIn::WorldToObjectKHR:        // alias spv::BuiltIn::WorldToObjectNV
    case spv::BuiltIn::IncomingRayFlagsKHR:    // alias spv::BuiltIn::IncomingRayFlagsNV
    case spv::BuiltIn::RayGeometryIndexKHR:    // NOT present in NV
    case spv::BuiltIn::CullMaskKHR: {
      return ValidateRayTracingBuiltinsAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::PrimitiveShadingRateKHR: {
      return ValidatePrimitiveShadingRateAtDefinition(decoration, inst);
    }
    case spv::BuiltIn::ShadingRateKHR: {
      return ValidateShadingRateAtDefinition(decoration, inst);
    }
    default:
      // No validation rules (for the moment).
      break;
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
      if (decoration.dec_type() != spv::Decoration::BuiltIn) {
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
  BuiltInsValidator validator(_);
  return validator.Run();
}

}  // namespace val
}  // namespace spvtools
