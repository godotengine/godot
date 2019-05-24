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

// Validates correctness of image instructions.

#include "source/val/validate.h"

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Performs compile time check that all SpvImageOperandsXXX cases are handled in
// this module. If SpvImageOperandsXXX list changes, this function will fail the
// build.
// For all other purposes this is a dummy function.
bool CheckAllImageOperandsHandled() {
  SpvImageOperandsMask enum_val = SpvImageOperandsBiasMask;

  // Some improvised code to prevent the compiler from considering enum_val
  // constant and optimizing the switch away.
  uint32_t stack_var = 0;
  if (reinterpret_cast<uintptr_t>(&stack_var) % 256)
    enum_val = SpvImageOperandsLodMask;

  switch (enum_val) {
    // Please update the validation rules in this module if you are changing
    // the list of image operands, and add new enum values to this switch.
    case SpvImageOperandsMaskNone:
      return false;
    case SpvImageOperandsBiasMask:
    case SpvImageOperandsLodMask:
    case SpvImageOperandsGradMask:
    case SpvImageOperandsConstOffsetMask:
    case SpvImageOperandsOffsetMask:
    case SpvImageOperandsConstOffsetsMask:
    case SpvImageOperandsSampleMask:
    case SpvImageOperandsMinLodMask:

    // TODO(dneto): Support image operands related to the Vulkan memory model.
    // https://gitlab.khronos.org/spirv/spirv-tools/issues/32
    case SpvImageOperandsMakeTexelAvailableKHRMask:
    case SpvImageOperandsMakeTexelVisibleKHRMask:
    case SpvImageOperandsNonPrivateTexelKHRMask:
    case SpvImageOperandsVolatileTexelKHRMask:
      return true;
  }
  return false;
}

// Used by GetImageTypeInfo. See OpTypeImage spec for more information.
struct ImageTypeInfo {
  uint32_t sampled_type = 0;
  SpvDim dim = SpvDimMax;
  uint32_t depth = 0;
  uint32_t arrayed = 0;
  uint32_t multisampled = 0;
  uint32_t sampled = 0;
  SpvImageFormat format = SpvImageFormatMax;
  SpvAccessQualifier access_qualifier = SpvAccessQualifierMax;
};

// Provides information on image type. |id| should be object of either
// OpTypeImage or OpTypeSampledImage type. Returns false in case of failure
// (not a valid id, failed to parse the instruction, etc).
bool GetImageTypeInfo(const ValidationState_t& _, uint32_t id,
                      ImageTypeInfo* info) {
  if (!id || !info) return false;

  const Instruction* inst = _.FindDef(id);
  assert(inst);

  if (inst->opcode() == SpvOpTypeSampledImage) {
    inst = _.FindDef(inst->word(2));
    assert(inst);
  }

  if (inst->opcode() != SpvOpTypeImage) return false;

  const size_t num_words = inst->words().size();
  if (num_words != 9 && num_words != 10) return false;

  info->sampled_type = inst->word(2);
  info->dim = static_cast<SpvDim>(inst->word(3));
  info->depth = inst->word(4);
  info->arrayed = inst->word(5);
  info->multisampled = inst->word(6);
  info->sampled = inst->word(7);
  info->format = static_cast<SpvImageFormat>(inst->word(8));
  info->access_qualifier = num_words < 10
                               ? SpvAccessQualifierMax
                               : static_cast<SpvAccessQualifier>(inst->word(9));
  return true;
}

bool IsImplicitLod(SpvOp opcode) {
  switch (opcode) {
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
      return true;
    default:
      break;
  }
  return false;
}

bool IsExplicitLod(SpvOp opcode) {
  switch (opcode) {
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageSampleProjExplicitLod:
    case SpvOpImageSampleProjDrefExplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod:
      return true;
    default:
      break;
  }
  return false;
}

// Returns true if the opcode is a Image instruction which applies
// homogenous projection to the coordinates.
bool IsProj(SpvOp opcode) {
  switch (opcode) {
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOpImageSampleProjExplicitLod:
    case SpvOpImageSampleProjDrefExplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod:
      return true;
    default:
      break;
  }
  return false;
}

// Returns the number of components in a coordinate used to access a texel in
// a single plane of an image with the given parameters.
uint32_t GetPlaneCoordSize(const ImageTypeInfo& info) {
  uint32_t plane_size = 0;
  // If this switch breaks your build, please add new values below.
  switch (info.dim) {
    case SpvDim1D:
    case SpvDimBuffer:
      plane_size = 1;
      break;
    case SpvDim2D:
    case SpvDimRect:
    case SpvDimSubpassData:
      plane_size = 2;
      break;
    case SpvDim3D:
    case SpvDimCube:
      // For Cube direction vector is used instead of UV.
      plane_size = 3;
      break;
    case SpvDimMax:
      assert(0);
      break;
  }

  return plane_size;
}

// Returns minimal number of coordinates based on image dim, arrayed and whether
// the instruction uses projection coordinates.
uint32_t GetMinCoordSize(SpvOp opcode, const ImageTypeInfo& info) {
  if (info.dim == SpvDimCube &&
      (opcode == SpvOpImageRead || opcode == SpvOpImageWrite ||
       opcode == SpvOpImageSparseRead)) {
    // These opcodes use UV for Cube, not direction vector.
    return 3;
  }

  return GetPlaneCoordSize(info) + info.arrayed + (IsProj(opcode) ? 1 : 0);
}

// Checks ImageOperand bitfield and respective operands.
spv_result_t ValidateImageOperands(ValidationState_t& _,
                                   const Instruction* inst,
                                   const ImageTypeInfo& info, uint32_t mask,
                                   uint32_t word_index) {
  static const bool kAllImageOperandsHandled = CheckAllImageOperandsHandled();
  (void)kAllImageOperandsHandled;

  const SpvOp opcode = inst->opcode();
  const size_t num_words = inst->words().size();

  // NonPrivate and Volatile take no operand words.
  const uint32_t mask_bits_having_operands =
      mask & ~uint32_t(SpvImageOperandsNonPrivateTexelKHRMask |
                       SpvImageOperandsVolatileTexelKHRMask);
  size_t expected_num_image_operand_words =
      spvtools::utils::CountSetBits(mask_bits_having_operands);
  if (mask & SpvImageOperandsGradMask) {
    // Grad uses two words.
    ++expected_num_image_operand_words;
  }

  if (expected_num_image_operand_words != num_words - word_index) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Number of image operand ids doesn't correspond to the bit mask";
  }

  if (spvtools::utils::CountSetBits(
          mask & (SpvImageOperandsOffsetMask | SpvImageOperandsConstOffsetMask |
                  SpvImageOperandsConstOffsetsMask)) > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image Operands Offset, ConstOffset, ConstOffsets cannot be used "
           << "together";
  }

  const bool is_implicit_lod = IsImplicitLod(opcode);
  const bool is_explicit_lod = IsExplicitLod(opcode);

  // The checks should be done in the order of definition of OperandImage.

  if (mask & SpvImageOperandsBiasMask) {
    if (!is_implicit_lod) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Bias can only be used with ImplicitLod opcodes";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsFloatScalarType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Bias to be float scalar";
    }

    if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
        info.dim != SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Bias requires 'Dim' parameter to be 1D, 2D, 3D "
                "or Cube";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Bias requires 'MS' parameter to be 0";
    }
  }

  if (mask & SpvImageOperandsLodMask) {
    if (!is_explicit_lod && opcode != SpvOpImageFetch &&
        opcode != SpvOpImageSparseFetch) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Lod can only be used with ExplicitLod opcodes "
             << "and OpImageFetch";
    }

    if (mask & SpvImageOperandsGradMask) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand bits Lod and Grad cannot be set at the same "
                "time";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (is_explicit_lod) {
      if (!_.IsFloatScalarType(type_id)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Image Operand Lod to be float scalar when used "
               << "with ExplicitLod";
      }
    } else {
      if (!_.IsIntScalarType(type_id)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Image Operand Lod to be int scalar when used with "
               << "OpImageFetch";
      }
    }

    if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
        info.dim != SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Lod requires 'Dim' parameter to be 1D, 2D, 3D "
                "or Cube";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Lod requires 'MS' parameter to be 0";
    }
  }

  if (mask & SpvImageOperandsGradMask) {
    if (!is_explicit_lod) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Grad can only be used with ExplicitLod opcodes";
    }

    const uint32_t dx_type_id = _.GetTypeId(inst->word(word_index++));
    const uint32_t dy_type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsFloatScalarOrVectorType(dx_type_id) ||
        !_.IsFloatScalarOrVectorType(dy_type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected both Image Operand Grad ids to be float scalars or "
             << "vectors";
    }

    const uint32_t plane_size = GetPlaneCoordSize(info);
    const uint32_t dx_size = _.GetDimension(dx_type_id);
    const uint32_t dy_size = _.GetDimension(dy_type_id);
    if (plane_size != dx_size) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Grad dx to have " << plane_size
             << " components, but given " << dx_size;
    }

    if (plane_size != dy_size) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Grad dy to have " << plane_size
             << " components, but given " << dy_size;
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Grad requires 'MS' parameter to be 0";
    }
  }

  if (mask & SpvImageOperandsConstOffsetMask) {
    if (info.dim == SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand ConstOffset cannot be used with Cube Image "
                "'Dim'";
    }

    const uint32_t id = inst->word(word_index++);
    const uint32_t type_id = _.GetTypeId(id);
    if (!_.IsIntScalarOrVectorType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffset to be int scalar or "
             << "vector";
    }

    if (!spvOpcodeIsConstant(_.GetIdOpcode(id))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffset to be a const object";
    }

    const uint32_t plane_size = GetPlaneCoordSize(info);
    const uint32_t offset_size = _.GetDimension(type_id);
    if (plane_size != offset_size) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffset to have " << plane_size
             << " components, but given " << offset_size;
    }
  }

  if (mask & SpvImageOperandsOffsetMask) {
    if (info.dim == SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Offset cannot be used with Cube Image 'Dim'";
    }

    const uint32_t id = inst->word(word_index++);
    const uint32_t type_id = _.GetTypeId(id);
    if (!_.IsIntScalarOrVectorType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Offset to be int scalar or "
             << "vector";
    }

    const uint32_t plane_size = GetPlaneCoordSize(info);
    const uint32_t offset_size = _.GetDimension(type_id);
    if (plane_size != offset_size) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Offset to have " << plane_size
             << " components, but given " << offset_size;
    }
  }

  if (mask & SpvImageOperandsConstOffsetsMask) {
    if (opcode != SpvOpImageGather && opcode != SpvOpImageDrefGather &&
        opcode != SpvOpImageSparseGather &&
        opcode != SpvOpImageSparseDrefGather) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand ConstOffsets can only be used with "
                "OpImageGather and OpImageDrefGather";
    }

    if (info.dim == SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand ConstOffsets cannot be used with Cube Image "
                "'Dim'";
    }

    const uint32_t id = inst->word(word_index++);
    const uint32_t type_id = _.GetTypeId(id);
    const Instruction* type_inst = _.FindDef(type_id);
    assert(type_inst);

    if (type_inst->opcode() != SpvOpTypeArray) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffsets to be an array of size 4";
    }

    uint64_t array_size = 0;
    if (!_.GetConstantValUint64(type_inst->word(3), &array_size)) {
      assert(0 && "Array type definition is corrupt");
    }

    if (array_size != 4) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffsets to be an array of size 4";
    }

    const uint32_t component_type = type_inst->word(2);
    if (!_.IsIntVectorType(component_type) ||
        _.GetDimension(component_type) != 2) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffsets array componenets to be "
                "int vectors of size 2";
    }

    if (!spvOpcodeIsConstant(_.GetIdOpcode(id))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffsets to be a const object";
    }
  }

  if (mask & SpvImageOperandsSampleMask) {
    if (opcode != SpvOpImageFetch && opcode != SpvOpImageRead &&
        opcode != SpvOpImageWrite && opcode != SpvOpImageSparseFetch &&
        opcode != SpvOpImageSparseRead) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Sample can only be used with OpImageFetch, "
             << "OpImageRead, OpImageWrite, OpImageSparseFetch and "
             << "OpImageSparseRead";
    }

    if (info.multisampled == 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Sample requires non-zero 'MS' parameter";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsIntScalarType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Sample to be int scalar";
    }
  }

  if (mask & SpvImageOperandsMinLodMask) {
    if (!is_implicit_lod && !(mask & SpvImageOperandsGradMask)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod can only be used with ImplicitLod "
             << "opcodes or together with Image Operand Grad";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsFloatScalarType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand MinLod to be float scalar";
    }

    if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
        info.dim != SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod requires 'Dim' parameter to be 1D, 2D, "
                "3D or Cube";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod requires 'MS' parameter to be 0";
    }
  }

  if (mask & SpvImageOperandsMakeTexelAvailableKHRMask) {
    // Checked elsewhere: capability and memory model are correct.
    if (opcode != SpvOpImageWrite) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelAvailableKHR can only be used with Op"
             << spvOpcodeString(SpvOpImageWrite) << ": Op"
             << spvOpcodeString(opcode);
    }

    if (!(mask & SpvImageOperandsNonPrivateTexelKHRMask)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelAvailableKHR requires "
                "NonPrivateTexelKHR is also specified: Op"
             << spvOpcodeString(opcode);
    }

    const auto available_scope = inst->word(word_index++);
    if (auto error = ValidateMemoryScope(_, inst, available_scope))
      return error;
  }

  if (mask & SpvImageOperandsMakeTexelVisibleKHRMask) {
    // Checked elsewhere: capability and memory model are correct.
    if (opcode != SpvOpImageRead && opcode != SpvOpImageSparseRead) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelVisibleKHR can only be used with Op"
             << spvOpcodeString(SpvOpImageRead) << " or Op"
             << spvOpcodeString(SpvOpImageSparseRead) << ": Op"
             << spvOpcodeString(opcode);
    }

    if (!(mask & SpvImageOperandsNonPrivateTexelKHRMask)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelVisibleKHR requires NonPrivateTexelKHR "
                "is also specified: Op"
             << spvOpcodeString(opcode);
    }

    const auto visible_scope = inst->word(word_index++);
    if (auto error = ValidateMemoryScope(_, inst, visible_scope)) return error;
  }

  return SPV_SUCCESS;
}

// Checks some of the validation rules which are common to multiple opcodes.
spv_result_t ValidateImageCommon(ValidationState_t& _, const Instruction* inst,
                                 const ImageTypeInfo& info) {
  const SpvOp opcode = inst->opcode();
  if (IsProj(opcode)) {
    if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
        info.dim != SpvDimRect) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Dim' parameter to be 1D, 2D, 3D or Rect";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Image 'MS' parameter to be 0";
    }

    if (info.arrayed != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Image 'arrayed' parameter to be 0";
    }
  }

  if (opcode == SpvOpImageRead || opcode == SpvOpImageSparseRead ||
      opcode == SpvOpImageWrite) {
    if (info.sampled == 0) {
    } else if (info.sampled == 2) {
      if (info.dim == SpvDim1D && !_.HasCapability(SpvCapabilityImage1D)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Capability Image1D is required to access storage image";
      } else if (info.dim == SpvDimRect &&
                 !_.HasCapability(SpvCapabilityImageRect)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Capability ImageRect is required to access storage image";
      } else if (info.dim == SpvDimBuffer &&
                 !_.HasCapability(SpvCapabilityImageBuffer)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Capability ImageBuffer is required to access storage image";
      } else if (info.dim == SpvDimCube && info.arrayed == 1 &&
                 !_.HasCapability(SpvCapabilityImageCubeArray)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Capability ImageCubeArray is required to access "
               << "storage image";
      }

      if (info.multisampled == 1 &&
          !_.HasCapability(SpvCapabilityImageMSArray)) {
#if 0
        // TODO(atgoo@github.com) The description of this rule in the spec
        // is unclear and Glslang doesn't declare ImageMSArray. Need to clarify
        // and reenable.
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
            << "Capability ImageMSArray is required to access storage "
            << "image";
#endif
      }
    } else {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled' parameter to be 0 or 2";
    }
  }

  return SPV_SUCCESS;
}

// Returns true if opcode is *ImageSparse*, false otherwise.
bool IsSparse(SpvOp opcode) {
  switch (opcode) {
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod:
    case SpvOpImageSparseFetch:
    case SpvOpImageSparseGather:
    case SpvOpImageSparseDrefGather:
    case SpvOpImageSparseTexelsResident:
    case SpvOpImageSparseRead: {
      return true;
    }

    default: { return false; }
  }

  return false;
}

// Checks sparse image opcode result type and returns the second struct member.
// Returns inst.type_id for non-sparse image opcodes.
// Not valid for sparse image opcodes which do not return a struct.
spv_result_t GetActualResultType(ValidationState_t& _, const Instruction* inst,
                                 uint32_t* actual_result_type) {
  const SpvOp opcode = inst->opcode();

  if (IsSparse(opcode)) {
    const Instruction* const type_inst = _.FindDef(inst->type_id());
    assert(type_inst);

    if (!type_inst || type_inst->opcode() != SpvOpTypeStruct) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Result Type to be OpTypeStruct";
    }

    if (type_inst->words().size() != 4 ||
        !_.IsIntScalarType(type_inst->word(2))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Result Type to be a struct containing an int "
                "scalar and a texel";
    }

    *actual_result_type = type_inst->word(3);
  } else {
    *actual_result_type = inst->type_id();
  }

  return SPV_SUCCESS;
}

// Returns a string describing actual result type of an opcode.
// Not valid for sparse image opcodes which do not return a struct.
const char* GetActualResultTypeStr(SpvOp opcode) {
  if (IsSparse(opcode)) return "Result Type's second member";
  return "Result Type";
}

spv_result_t ValidateTypeImage(ValidationState_t& _, const Instruction* inst) {
  assert(inst->type_id() == 0);

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, inst->word(1), &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if ((!_.IsFloatScalarType(info.sampled_type) &&
         !_.IsIntScalarType(info.sampled_type)) ||
        32 != _.GetBitWidth(info.sampled_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Sampled Type to be a 32-bit int or float "
                "scalar type for Vulkan environment";
    }
  } else {
    const SpvOp sampled_type_opcode = _.GetIdOpcode(info.sampled_type);
    if (sampled_type_opcode != SpvOpTypeVoid &&
        sampled_type_opcode != SpvOpTypeInt &&
        sampled_type_opcode != SpvOpTypeFloat) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Sampled Type to be either void or"
             << " numerical scalar type";
    }
  }

  // Dim is checked elsewhere.

  if (info.depth > 2) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Invalid Depth " << info.depth << " (must be 0, 1 or 2)";
  }

  if (info.arrayed > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Invalid Arrayed " << info.arrayed << " (must be 0 or 1)";
  }

  if (info.multisampled > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Invalid MS " << info.multisampled << " (must be 0 or 1)";
  }

  if (info.sampled > 2) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Invalid Sampled " << info.sampled << " (must be 0, 1 or 2)";
  }

  if (info.dim == SpvDimSubpassData) {
    if (info.sampled != 2) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Dim SubpassData requires Sampled to be 2";
    }

    if (info.format != SpvImageFormatUnknown) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Dim SubpassData requires format Unknown";
    }
  }

  // Format and Access Qualifier are checked elsewhere.

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeSampledImage(ValidationState_t& _,
                                      const Instruction* inst) {
  const uint32_t image_type = inst->word(2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }
  return SPV_SUCCESS;
}

bool IsAllowedSampledImageOperand(SpvOp opcode) {
  switch (opcode) {
    case SpvOpSampledImage:
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjExplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSampleProjDrefExplicitLod:
    case SpvOpImageGather:
    case SpvOpImageDrefGather:
    case SpvOpImage:
    case SpvOpImageQueryLod:
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOpImageSparseGather:
    case SpvOpImageSparseDrefGather:
      return true;
    default:
      return false;
  }
}

spv_result_t ValidateSampledImage(ValidationState_t& _,
                                  const Instruction* inst) {
  if (_.GetIdOpcode(inst->type_id()) != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypeSampledImage.";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage.";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  // TODO(atgoo@github.com) Check compatibility of result type and received
  // image.

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (info.sampled != 1) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled' parameter to be 1 "
             << "for Vulkan environment.";
    }
  } else {
    if (info.sampled != 0 && info.sampled != 1) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled' parameter to be 0 or 1";
    }
  }

  if (info.dim == SpvDimSubpassData) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Dim' parameter to be not SubpassData.";
  }

  if (_.GetIdOpcode(_.GetOperandTypeId(inst, 3)) != SpvOpTypeSampler) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampler to be of type OpTypeSampler";
  }

  // We need to validate 2 things:
  // * All OpSampledImage instructions must be in the same block in which their
  // Result <id> are consumed.
  // * Result <id> from OpSampledImage instructions must not appear as operands
  // to OpPhi instructions or OpSelect instructions, or any instructions other
  // than the image lookup and query instructions specified to take an operand
  // whose type is OpTypeSampledImage.
  std::vector<Instruction*> consumers = _.getSampledImageConsumers(inst->id());
  if (!consumers.empty()) {
    for (auto consumer_instr : consumers) {
      const auto consumer_opcode = consumer_instr->opcode();
      if (consumer_instr->block() != inst->block()) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "All OpSampledImage instructions must be in the same block "
                  "in "
                  "which their Result <id> are consumed. OpSampledImage Result "
                  "Type <id> '"
               << _.getIdName(inst->id())
               << "' has a consumer in a different basic "
                  "block. The consumer instruction <id> is '"
               << _.getIdName(consumer_instr->id()) << "'.";
      }

      if (consumer_opcode == SpvOpPhi || consumer_opcode == SpvOpSelect) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Result <id> from OpSampledImage instruction must not appear "
                  "as "
                  "operands of Op"
               << spvOpcodeString(static_cast<SpvOp>(consumer_opcode)) << "."
               << " Found result <id> '" << _.getIdName(inst->id())
               << "' as an operand of <id> '"
               << _.getIdName(consumer_instr->id()) << "'.";
      }

      if (!IsAllowedSampledImageOperand(consumer_opcode)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Result <id> from OpSampledImage instruction must not appear "
                  "as operand for Op"
               << spvOpcodeString(static_cast<SpvOp>(consumer_opcode))
               << ", since it is not specificed as taking an "
               << "OpTypeSampledImage."
               << " Found result <id> '" << _.getIdName(inst->id())
               << "' as an operand of <id> '"
               << _.getIdName(consumer_instr->id()) << "'.";
      }
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageTexelPointer(ValidationState_t& _,
                                       const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  if (result_type->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer";
  }

  const auto storage_class = result_type->GetOperandAs<uint32_t>(1);
  if (storage_class != SpvStorageClassImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer whose Storage Class "
              "operand is Image";
  }

  const auto ptr_type = result_type->GetOperandAs<uint32_t>(2);
  const auto ptr_opcode = _.GetIdOpcode(ptr_type);
  if (ptr_opcode != SpvOpTypeInt && ptr_opcode != SpvOpTypeFloat &&
      ptr_opcode != SpvOpTypeVoid) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer whose Type operand "
              "must be a scalar numerical type or OpTypeVoid";
  }

  const auto image_ptr = _.FindDef(_.GetOperandTypeId(inst, 2));
  if (!image_ptr || image_ptr->opcode() != SpvOpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be OpTypePointer";
  }

  const auto image_type = image_ptr->GetOperandAs<uint32_t>(2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be OpTypePointer with Type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.sampled_type != ptr_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Sampled Type' to be the same as the Type "
              "pointed to by Result Type";
  }

  if (info.dim == SpvDimSubpassData) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image Dim SubpassData cannot be used with OpImageTexelPointer";
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (!coord_type || !_.IsIntScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be integer scalar or vector";
  }

  uint32_t expected_coord_size = 0;
  if (info.arrayed == 0) {
    expected_coord_size = GetPlaneCoordSize(info);
  } else if (info.arrayed == 1) {
    switch (info.dim) {
      case SpvDim1D:
        expected_coord_size = 2;
        break;
      case SpvDimCube:
      case SpvDim2D:
        expected_coord_size = 3;
        break;
      default:
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected Image 'Dim' must be one of 1D, 2D, or Cube when "
                  "Arrayed is 1";
        break;
    }
  }

  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (expected_coord_size != actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have " << expected_coord_size
           << " components, but given " << actual_coord_size;
  }

  const uint32_t sample_type = _.GetOperandTypeId(inst, 4);
  if (!sample_type || !_.IsIntScalarType(sample_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sample to be integer scalar";
  }

  if (info.multisampled == 0) {
    uint64_t ms = 0;
    if (!_.GetConstantValUint64(inst->GetOperandAs<uint32_t>(4), &ms) ||
        ms != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Sample for Image with MS 0 to be a valid <id> for "
                "the value 0";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageLod(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type)) {
    return error;
  }

  if (!_.IsIntVectorType(actual_result_type) &&
      !_.IsFloatVectorType(actual_result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to be int or float vector type";
  }

  if (_.GetDimension(actual_result_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to have 4 components";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (spv_result_t result = ValidateImageCommon(_, inst, info)) return result;

  if (_.GetIdOpcode(info.sampled_type) != SpvOpTypeVoid) {
    const uint32_t texel_component_type =
        _.GetComponentType(actual_result_type);
    if (texel_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if ((opcode == SpvOpImageSampleExplicitLod ||
       opcode == SpvOpImageSparseSampleExplicitLod) &&
      _.HasCapability(SpvCapabilityKernel)) {
    if (!_.IsFloatScalarOrVectorType(coord_type) &&
        !_.IsIntScalarOrVectorType(coord_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Coordinate to be int or float scalar or vector";
    }
  } else {
    if (!_.IsFloatScalarOrVectorType(coord_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Coordinate to be float scalar or vector";
    }
  }

  const uint32_t min_coord_size = GetMinCoordSize(opcode, info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  if (inst->words().size() <= 5) {
    assert(IsImplicitLod(opcode));
    return SPV_SUCCESS;
  }

  const uint32_t mask = inst->word(5);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageDrefLod(ValidationState_t& _,
                                  const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type)) {
    return error;
  }

  if (!_.IsIntScalarType(actual_result_type) &&
      !_.IsFloatScalarType(actual_result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to be int or float scalar type";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (spv_result_t result = ValidateImageCommon(_, inst, info)) return result;

  if (actual_result_type != info.sampled_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Sampled Type' to be the same as "
           << GetActualResultTypeStr(opcode);
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsFloatScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be float scalar or vector";
  }

  const uint32_t min_coord_size = GetMinCoordSize(opcode, info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  const uint32_t dref_type = _.GetOperandTypeId(inst, 4);
  if (!_.IsFloatScalarType(dref_type) || _.GetBitWidth(dref_type) != 32) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Dref to be of 32-bit float type";
  }

  if (inst->words().size() <= 6) {
    assert(IsImplicitLod(opcode));
    return SPV_SUCCESS;
  }

  const uint32_t mask = inst->word(6);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 7))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageFetch(ValidationState_t& _, const Instruction* inst) {
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type)) {
    return error;
  }

  const SpvOp opcode = inst->opcode();
  if (!_.IsIntVectorType(actual_result_type) &&
      !_.IsFloatVectorType(actual_result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to be int or float vector type";
  }

  if (_.GetDimension(actual_result_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to have 4 components";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (_.GetIdOpcode(info.sampled_type) != SpvOpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (info.dim == SpvDimCube) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Image 'Dim' cannot be Cube";
  }

  if (info.sampled != 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Sampled' parameter to be 1";
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsIntScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be int scalar or vector";
  }

  const uint32_t min_coord_size = GetMinCoordSize(opcode, info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  if (inst->words().size() <= 5) return SPV_SUCCESS;

  const uint32_t mask = inst->word(5);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageGather(ValidationState_t& _,
                                 const Instruction* inst) {
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type))
    return error;

  const SpvOp opcode = inst->opcode();
  if (!_.IsIntVectorType(actual_result_type) &&
      !_.IsFloatVectorType(actual_result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to be int or float vector type";
  }

  if (_.GetDimension(actual_result_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to have 4 components";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (opcode == SpvOpImageDrefGather || opcode == SpvOpImageSparseDrefGather ||
      _.GetIdOpcode(info.sampled_type) != SpvOpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (info.dim != SpvDim2D && info.dim != SpvDimCube &&
      info.dim != SpvDimRect) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Dim' cannot be Cube";
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsFloatScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be float scalar or vector";
  }

  const uint32_t min_coord_size = GetMinCoordSize(opcode, info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  if (opcode == SpvOpImageGather || opcode == SpvOpImageSparseGather) {
    const uint32_t component_index_type = _.GetOperandTypeId(inst, 4);
    if (!_.IsIntScalarType(component_index_type) ||
        _.GetBitWidth(component_index_type) != 32) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Component to be 32-bit int scalar";
    }
  } else {
    assert(opcode == SpvOpImageDrefGather ||
           opcode == SpvOpImageSparseDrefGather);
    const uint32_t dref_type = _.GetOperandTypeId(inst, 4);
    if (!_.IsFloatScalarType(dref_type) || _.GetBitWidth(dref_type) != 32) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Dref to be of 32-bit float type";
    }
  }

  if (inst->words().size() <= 6) return SPV_SUCCESS;

  const uint32_t mask = inst->word(6);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 7))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageRead(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type)) {
    return error;
  }

  if (!_.IsIntScalarOrVectorType(actual_result_type) &&
      !_.IsFloatScalarOrVectorType(actual_result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to be int or float scalar or vector type";
  }

#if 0
  // TODO(atgoo@github.com) Disabled until the spec is clarified.
  if (_.GetDimension(actual_result_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected " << GetActualResultTypeStr(opcode)
           << " to have 4 components";
  }
#endif

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.dim == SpvDimSubpassData) {
    if (opcode == SpvOpImageSparseRead) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Dim SubpassData cannot be used with ImageSparseRead";
    }

    _.function(inst->function()->id())
        ->RegisterExecutionModelLimitation(
            SpvExecutionModelFragment,
            std::string("Dim SubpassData requires Fragment execution model: ") +
                spvOpcodeString(opcode));
  }

  if (_.GetIdOpcode(info.sampled_type) != SpvOpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (spv_result_t result = ValidateImageCommon(_, inst, info)) return result;

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsIntScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be int scalar or vector";
  }

  const uint32_t min_coord_size = GetMinCoordSize(opcode, info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  if (info.format == SpvImageFormatUnknown && info.dim != SpvDimSubpassData &&
      !_.HasCapability(SpvCapabilityStorageImageReadWithoutFormat)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Capability StorageImageReadWithoutFormat is required to "
           << "read storage image";
  }

  if (inst->words().size() <= 5) return SPV_SUCCESS;

  const uint32_t mask = inst->word(5);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageWrite(ValidationState_t& _, const Instruction* inst) {
  const uint32_t image_type = _.GetOperandTypeId(inst, 0);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.dim == SpvDimSubpassData) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image 'Dim' cannot be SubpassData";
  }

  if (spv_result_t result = ValidateImageCommon(_, inst, info)) return result;

  const uint32_t coord_type = _.GetOperandTypeId(inst, 1);
  if (!_.IsIntScalarOrVectorType(coord_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to be int scalar or vector";
  }

  const uint32_t min_coord_size = GetMinCoordSize(inst->opcode(), info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }

  // TODO(atgoo@github.com) The spec doesn't explicitely say what the type
  // of texel should be.
  const uint32_t texel_type = _.GetOperandTypeId(inst, 2);
  if (!_.IsIntScalarOrVectorType(texel_type) &&
      !_.IsFloatScalarOrVectorType(texel_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Texel to be int or float vector or scalar";
  }

#if 0
  // TODO: See above.
  if (_.GetDimension(texel_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
        << "Expected Texel to have 4 components";
  }
#endif

  if (_.GetIdOpcode(info.sampled_type) != SpvOpTypeVoid) {
    const uint32_t texel_component_type = _.GetComponentType(texel_type);
    if (texel_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as Texel "
             << "components";
    }
  }

  if (info.format == SpvImageFormatUnknown && info.dim != SpvDimSubpassData &&
      !_.HasCapability(SpvCapabilityStorageImageWriteWithoutFormat)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Capability StorageImageWriteWithoutFormat is required to "
              "write "
           << "to storage image";
  }

  if (inst->words().size() <= 4) return SPV_SUCCESS;

  const uint32_t mask = inst->word(4);
  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, mask, /* word_index = */ 5))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImage(ValidationState_t& _, const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  if (_.GetIdOpcode(result_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypeImage";
  }

  const uint32_t sampled_image_type = _.GetOperandTypeId(inst, 2);
  const Instruction* sampled_image_type_inst = _.FindDef(sampled_image_type);
  assert(sampled_image_type_inst);

  if (sampled_image_type_inst->opcode() != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sample Image to be of type OpTypeSampleImage";
  }

  if (sampled_image_type_inst->word(2) != result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sample Image image type to be equal to Result Type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateImageQuerySizeLod(ValidationState_t& _,
                                       const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  if (!_.IsIntScalarOrVectorType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be int scalar or vector type";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  uint32_t expected_num_components = info.arrayed;
  switch (info.dim) {
    case SpvDim1D:
      expected_num_components += 1;
      break;
    case SpvDim2D:
    case SpvDimCube:
      expected_num_components += 2;
      break;
    case SpvDim3D:
      expected_num_components += 3;
      break;
    default:
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, 2D, 3D or Cube";
  }

  if (info.multisampled != 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Image 'MS' must be 0";
  }

  uint32_t result_num_components = _.GetDimension(result_type);
  if (result_num_components != expected_num_components) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result Type has " << result_num_components << " components, "
           << "but " << expected_num_components << " expected";
  }

  const uint32_t lod_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsIntScalarType(lod_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Level of Detail to be int scalar";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageQuerySize(ValidationState_t& _,
                                    const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  if (!_.IsIntScalarOrVectorType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be int scalar or vector type";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  uint32_t expected_num_components = info.arrayed;
  switch (info.dim) {
    case SpvDim1D:
    case SpvDimBuffer:
      expected_num_components += 1;
      break;
    case SpvDim2D:
    case SpvDimCube:
    case SpvDimRect:
      expected_num_components += 2;
      break;
    case SpvDim3D:
      expected_num_components += 3;
      break;
    default:
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, Buffer, 2D, Cube, 3D or Rect";
  }

  if (info.dim == SpvDim1D || info.dim == SpvDim2D || info.dim == SpvDim3D ||
      info.dim == SpvDimCube) {
    if (info.multisampled != 1 && info.sampled != 0 && info.sampled != 2) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image must have either 'MS'=1 or 'Sampled'=0 or 'Sampled'=2";
    }
  }

  uint32_t result_num_components = _.GetDimension(result_type);
  if (result_num_components != expected_num_components) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result Type has " << result_num_components << " components, "
           << "but " << expected_num_components << " expected";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateImageQueryFormatOrOrder(ValidationState_t& _,
                                             const Instruction* inst) {
  if (!_.IsIntScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be int scalar type";
  }

  if (_.GetIdOpcode(_.GetOperandTypeId(inst, 2)) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected operand to be of type OpTypeImage";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageQueryLod(ValidationState_t& _,
                                   const Instruction* inst) {
  _.function(inst->function()->id())
      ->RegisterExecutionModelLimitation(
          SpvExecutionModelFragment,
          "OpImageQueryLod requires Fragment execution model");

  const uint32_t result_type = inst->type_id();
  if (!_.IsFloatVectorType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be float vector type";
  }

  if (_.GetDimension(result_type) != 2) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to have 2 components";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image operand to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
      info.dim != SpvDimCube) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image 'Dim' must be 1D, 2D, 3D or Cube";
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (_.HasCapability(SpvCapabilityKernel)) {
    if (!_.IsFloatScalarOrVectorType(coord_type) &&
        !_.IsIntScalarOrVectorType(coord_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Coordinate to be int or float scalar or vector";
    }
  } else {
    if (!_.IsFloatScalarOrVectorType(coord_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Coordinate to be float scalar or vector";
    }
  }

  const uint32_t min_coord_size = GetPlaneCoordSize(info);
  const uint32_t actual_coord_size = _.GetDimension(coord_type);
  if (min_coord_size > actual_coord_size) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Coordinate to have at least " << min_coord_size
           << " components, but given only " << actual_coord_size;
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageSparseLod(ValidationState_t& _,
                                    const Instruction* inst) {
  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << "Instruction reserved for future use, use of this instruction "
         << "is invalid";
}

spv_result_t ValidateImageQueryLevelsOrSamples(ValidationState_t& _,
                                               const Instruction* inst) {
  if (!_.IsIntScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be int scalar type";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != SpvOpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  const SpvOp opcode = inst->opcode();
  if (opcode == SpvOpImageQueryLevels) {
    if (info.dim != SpvDim1D && info.dim != SpvDim2D && info.dim != SpvDim3D &&
        info.dim != SpvDimCube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, 2D, 3D or Cube";
    }
  } else {
    assert(opcode == SpvOpImageQuerySamples);
    if (info.dim != SpvDim2D) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Image 'Dim' must be 2D";
    }

    if (info.multisampled != 1) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Image 'MS' must be 1";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageSparseTexelsResident(ValidationState_t& _,
                                               const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be bool scalar type";
  }

  const uint32_t resident_code_type = _.GetOperandTypeId(inst, 2);
  if (!_.IsIntScalarType(resident_code_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Resident Code to be int scalar";
  }

  return SPV_SUCCESS;
}

}  // namespace

// Validates correctness of image instructions.
spv_result_t ImagePass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  if (IsImplicitLod(opcode)) {
    _.function(inst->function()->id())
        ->RegisterExecutionModelLimitation(
            SpvExecutionModelFragment,
            "ImplicitLod instructions require Fragment execution model");
  }

  switch (opcode) {
    case SpvOpTypeImage:
      return ValidateTypeImage(_, inst);
    case SpvOpTypeSampledImage:
      return ValidateTypeSampledImage(_, inst);
    case SpvOpSampledImage:
      return ValidateSampledImage(_, inst);
    case SpvOpImageTexelPointer:
      return ValidateImageTexelPointer(_, inst);

    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjExplicitLod:
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
      return ValidateImageLod(_, inst);

    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSampleProjDrefExplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
      return ValidateImageDrefLod(_, inst);

    case SpvOpImageFetch:
    case SpvOpImageSparseFetch:
      return ValidateImageFetch(_, inst);

    case SpvOpImageGather:
    case SpvOpImageDrefGather:
    case SpvOpImageSparseGather:
    case SpvOpImageSparseDrefGather:
      return ValidateImageGather(_, inst);

    case SpvOpImageRead:
    case SpvOpImageSparseRead:
      return ValidateImageRead(_, inst);

    case SpvOpImageWrite:
      return ValidateImageWrite(_, inst);

    case SpvOpImage:
      return ValidateImage(_, inst);

    case SpvOpImageQueryFormat:
    case SpvOpImageQueryOrder:
      return ValidateImageQueryFormatOrOrder(_, inst);

    case SpvOpImageQuerySizeLod:
      return ValidateImageQuerySizeLod(_, inst);
    case SpvOpImageQuerySize:
      return ValidateImageQuerySize(_, inst);
    case SpvOpImageQueryLod:
      return ValidateImageQueryLod(_, inst);

    case SpvOpImageQueryLevels:
    case SpvOpImageQuerySamples:
      return ValidateImageQueryLevelsOrSamples(_, inst);

    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod:
      return ValidateImageSparseLod(_, inst);

    case SpvOpImageSparseTexelsResident:
      return ValidateImageSparseTexelsResident(_, inst);

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
