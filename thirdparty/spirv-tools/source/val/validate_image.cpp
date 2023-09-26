// Copyright (c) 2017 Google Inc.
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

// Validates correctness of image instructions.

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Performs compile time check that all spv::ImageOperandsMask::XXX cases are
// handled in this module. If spv::ImageOperandsMask::XXX list changes, this
// function will fail the build. For all other purposes this is a placeholder
// function.
bool CheckAllImageOperandsHandled() {
  spv::ImageOperandsMask enum_val = spv::ImageOperandsMask::Bias;

  // Some improvised code to prevent the compiler from considering enum_val
  // constant and optimizing the switch away.
  uint32_t stack_var = 0;
  if (reinterpret_cast<uintptr_t>(&stack_var) % 256)
    enum_val = spv::ImageOperandsMask::Lod;

  switch (enum_val) {
    // Please update the validation rules in this module if you are changing
    // the list of image operands, and add new enum values to this switch.
    case spv::ImageOperandsMask::MaskNone:
      return false;
    case spv::ImageOperandsMask::Bias:
    case spv::ImageOperandsMask::Lod:
    case spv::ImageOperandsMask::Grad:
    case spv::ImageOperandsMask::ConstOffset:
    case spv::ImageOperandsMask::Offset:
    case spv::ImageOperandsMask::ConstOffsets:
    case spv::ImageOperandsMask::Sample:
    case spv::ImageOperandsMask::MinLod:

    // TODO(dneto): Support image operands related to the Vulkan memory model.
    // https://gitlab.khronos.org/spirv/spirv-tools/issues/32
    case spv::ImageOperandsMask::MakeTexelAvailableKHR:
    case spv::ImageOperandsMask::MakeTexelVisibleKHR:
    case spv::ImageOperandsMask::NonPrivateTexelKHR:
    case spv::ImageOperandsMask::VolatileTexelKHR:
    case spv::ImageOperandsMask::SignExtend:
    case spv::ImageOperandsMask::ZeroExtend:
    // TODO(jaebaek): Move this line properly after handling image offsets
    //                operand. This line temporarily fixes CI failure that
    //                blocks other PRs.
    // https://github.com/KhronosGroup/SPIRV-Tools/issues/4565
    case spv::ImageOperandsMask::Offsets:
    case spv::ImageOperandsMask::Nontemporal:
      return true;
  }
  return false;
}

// Used by GetImageTypeInfo. See OpTypeImage spec for more information.
struct ImageTypeInfo {
  uint32_t sampled_type = 0;
  spv::Dim dim = spv::Dim::Max;
  uint32_t depth = 0;
  uint32_t arrayed = 0;
  uint32_t multisampled = 0;
  uint32_t sampled = 0;
  spv::ImageFormat format = spv::ImageFormat::Max;
  spv::AccessQualifier access_qualifier = spv::AccessQualifier::Max;
};

// Provides information on image type. |id| should be object of either
// OpTypeImage or OpTypeSampledImage type. Returns false in case of failure
// (not a valid id, failed to parse the instruction, etc).
bool GetImageTypeInfo(const ValidationState_t& _, uint32_t id,
                      ImageTypeInfo* info) {
  if (!id || !info) return false;

  const Instruction* inst = _.FindDef(id);
  assert(inst);

  if (inst->opcode() == spv::Op::OpTypeSampledImage) {
    inst = _.FindDef(inst->word(2));
    assert(inst);
  }

  if (inst->opcode() != spv::Op::OpTypeImage) return false;

  const size_t num_words = inst->words().size();
  if (num_words != 9 && num_words != 10) return false;

  info->sampled_type = inst->word(2);
  info->dim = static_cast<spv::Dim>(inst->word(3));
  info->depth = inst->word(4);
  info->arrayed = inst->word(5);
  info->multisampled = inst->word(6);
  info->sampled = inst->word(7);
  info->format = static_cast<spv::ImageFormat>(inst->word(8));
  info->access_qualifier =
      num_words < 10 ? spv::AccessQualifier::Max
                     : static_cast<spv::AccessQualifier>(inst->word(9));
  return true;
}

bool IsImplicitLod(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
      return true;
    default:
      break;
  }
  return false;
}

bool IsExplicitLod(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
      return true;
    default:
      break;
  }
  return false;
}

bool IsValidLodOperand(const ValidationState_t& _, spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageRead:
    case spv::Op::OpImageWrite:
    case spv::Op::OpImageSparseRead:
      return _.HasCapability(spv::Capability::ImageReadWriteLodAMD);
    default:
      return IsExplicitLod(opcode);
  }
}

bool IsValidGatherLodBiasAMD(const ValidationState_t& _, spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageGather:
    case spv::Op::OpImageSparseGather:
      return _.HasCapability(spv::Capability::ImageGatherBiasLodAMD);
    default:
      break;
  }
  return false;
}

// Returns true if the opcode is a Image instruction which applies
// homogenous projection to the coordinates.
bool IsProj(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
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
    case spv::Dim::Dim1D:
    case spv::Dim::Buffer:
      plane_size = 1;
      break;
    case spv::Dim::Dim2D:
    case spv::Dim::Rect:
    case spv::Dim::SubpassData:
      plane_size = 2;
      break;
    case spv::Dim::Dim3D:
    case spv::Dim::Cube:
      // For Cube direction vector is used instead of UV.
      plane_size = 3;
      break;
    case spv::Dim::Max:
      assert(0);
      break;
  }

  return plane_size;
}

// Returns minimal number of coordinates based on image dim, arrayed and whether
// the instruction uses projection coordinates.
uint32_t GetMinCoordSize(spv::Op opcode, const ImageTypeInfo& info) {
  if (info.dim == spv::Dim::Cube &&
      (opcode == spv::Op::OpImageRead || opcode == spv::Op::OpImageWrite ||
       opcode == spv::Op::OpImageSparseRead)) {
    // These opcodes use UV for Cube, not direction vector.
    return 3;
  }

  return GetPlaneCoordSize(info) + info.arrayed + (IsProj(opcode) ? 1 : 0);
}

// Checks ImageOperand bitfield and respective operands.
// word_index is the index of the first word after the image-operand mask word.
spv_result_t ValidateImageOperands(ValidationState_t& _,
                                   const Instruction* inst,
                                   const ImageTypeInfo& info,
                                   uint32_t word_index) {
  static const bool kAllImageOperandsHandled = CheckAllImageOperandsHandled();
  (void)kAllImageOperandsHandled;

  const spv::Op opcode = inst->opcode();
  const size_t num_words = inst->words().size();

  const bool have_explicit_mask = (word_index - 1 < num_words);
  const uint32_t mask = have_explicit_mask ? inst->word(word_index - 1) : 0u;

  if (have_explicit_mask) {
    // NonPrivate, Volatile, SignExtend, ZeroExtend take no operand words.
    const uint32_t mask_bits_having_operands =
        mask & ~uint32_t(spv::ImageOperandsMask::NonPrivateTexelKHR |
                         spv::ImageOperandsMask::VolatileTexelKHR |
                         spv::ImageOperandsMask::SignExtend |
                         spv::ImageOperandsMask::ZeroExtend |
                         spv::ImageOperandsMask::Nontemporal);
    size_t expected_num_image_operand_words =
        spvtools::utils::CountSetBits(mask_bits_having_operands);
    if (mask & uint32_t(spv::ImageOperandsMask::Grad)) {
      // Grad uses two words.
      ++expected_num_image_operand_words;
    }

    if (expected_num_image_operand_words != num_words - word_index) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Number of image operand ids doesn't correspond to the bit "
                "mask";
    }
  } else if (num_words != word_index - 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Number of image operand ids doesn't correspond to the bit mask";
  }

  if (info.multisampled &
      (0 == (mask & uint32_t(spv::ImageOperandsMask::Sample)))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image Operand Sample is required for operation on "
              "multi-sampled image";
  }

  // After this point, only set bits in the image operands mask can cause
  // the module to be invalid.
  if (mask == 0) return SPV_SUCCESS;

  if (spvtools::utils::CountSetBits(
          mask & uint32_t(spv::ImageOperandsMask::Offset |
                          spv::ImageOperandsMask::ConstOffset |
                          spv::ImageOperandsMask::ConstOffsets |
                          spv::ImageOperandsMask::Offsets)) > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(4662)
           << "Image Operands Offset, ConstOffset, ConstOffsets, Offsets "
              "cannot be used together";
  }

  const bool is_implicit_lod = IsImplicitLod(opcode);
  const bool is_explicit_lod = IsExplicitLod(opcode);
  const bool is_valid_lod_operand = IsValidLodOperand(_, opcode);
  const bool is_valid_gather_lod_bias_amd = IsValidGatherLodBiasAMD(_, opcode);

  // The checks should be done in the order of definition of OperandImage.

  if (mask & uint32_t(spv::ImageOperandsMask::Bias)) {
    if (!is_implicit_lod && !is_valid_gather_lod_bias_amd) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Bias can only be used with ImplicitLod opcodes";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsFloatScalarType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand Bias to be float scalar";
    }

    if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
        info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Cube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Bias requires 'Dim' parameter to be 1D, 2D, 3D "
                "or Cube";
    }

    // Multisampled is already checked.
  }

  if (mask & uint32_t(spv::ImageOperandsMask::Lod)) {
    if (!is_valid_lod_operand && opcode != spv::Op::OpImageFetch &&
        opcode != spv::Op::OpImageSparseFetch &&
        !is_valid_gather_lod_bias_amd) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Lod can only be used with ExplicitLod opcodes "
             << "and OpImageFetch";
    }

    if (mask & uint32_t(spv::ImageOperandsMask::Grad)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand bits Lod and Grad cannot be set at the same "
                "time";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (is_explicit_lod || is_valid_gather_lod_bias_amd) {
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

    if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
        info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Cube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand Lod requires 'Dim' parameter to be 1D, 2D, 3D "
                "or Cube";
    }

    // Multisampled is already checked.
  }

  if (mask & uint32_t(spv::ImageOperandsMask::Grad)) {
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

    // Multisampled is already checked.
  }

  if (mask & uint32_t(spv::ImageOperandsMask::ConstOffset)) {
    if (info.dim == spv::Dim::Cube) {
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

  if (mask & uint32_t(spv::ImageOperandsMask::Offset)) {
    if (info.dim == spv::Dim::Cube) {
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

    if (!_.options()->before_hlsl_legalization &&
        spvIsVulkanEnv(_.context()->target_env)) {
      if (opcode != spv::Op::OpImageGather &&
          opcode != spv::Op::OpImageDrefGather &&
          opcode != spv::Op::OpImageSparseGather &&
          opcode != spv::Op::OpImageSparseDrefGather) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(4663)
               << "Image Operand Offset can only be used with "
                  "OpImage*Gather operations";
      }
    }
  }

  if (mask & uint32_t(spv::ImageOperandsMask::ConstOffsets)) {
    if (opcode != spv::Op::OpImageGather &&
        opcode != spv::Op::OpImageDrefGather &&
        opcode != spv::Op::OpImageSparseGather &&
        opcode != spv::Op::OpImageSparseDrefGather) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand ConstOffsets can only be used with "
                "OpImageGather and OpImageDrefGather";
    }

    if (info.dim == spv::Dim::Cube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand ConstOffsets cannot be used with Cube Image "
                "'Dim'";
    }

    const uint32_t id = inst->word(word_index++);
    const uint32_t type_id = _.GetTypeId(id);
    const Instruction* type_inst = _.FindDef(type_id);
    assert(type_inst);

    if (type_inst->opcode() != spv::Op::OpTypeArray) {
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
             << "Expected Image Operand ConstOffsets array components to be "
                "int vectors of size 2";
    }

    if (!spvOpcodeIsConstant(_.GetIdOpcode(id))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand ConstOffsets to be a const object";
    }
  }

  if (mask & uint32_t(spv::ImageOperandsMask::Sample)) {
    if (opcode != spv::Op::OpImageFetch && opcode != spv::Op::OpImageRead &&
        opcode != spv::Op::OpImageWrite &&
        opcode != spv::Op::OpImageSparseFetch &&
        opcode != spv::Op::OpImageSparseRead) {
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

  if (mask & uint32_t(spv::ImageOperandsMask::MinLod)) {
    if (!is_implicit_lod && !(mask & uint32_t(spv::ImageOperandsMask::Grad))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod can only be used with ImplicitLod "
             << "opcodes or together with Image Operand Grad";
    }

    const uint32_t type_id = _.GetTypeId(inst->word(word_index++));
    if (!_.IsFloatScalarType(type_id)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image Operand MinLod to be float scalar";
    }

    if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
        info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Cube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod requires 'Dim' parameter to be 1D, 2D, "
                "3D or Cube";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MinLod requires 'MS' parameter to be 0";
    }
  }

  if (mask & uint32_t(spv::ImageOperandsMask::MakeTexelAvailableKHR)) {
    // Checked elsewhere: capability and memory model are correct.
    if (opcode != spv::Op::OpImageWrite) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelAvailableKHR can only be used with Op"
             << spvOpcodeString(spv::Op::OpImageWrite) << ": Op"
             << spvOpcodeString(opcode);
    }

    if (!(mask & uint32_t(spv::ImageOperandsMask::NonPrivateTexelKHR))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelAvailableKHR requires "
                "NonPrivateTexelKHR is also specified: Op"
             << spvOpcodeString(opcode);
    }

    const auto available_scope = inst->word(word_index++);
    if (auto error = ValidateMemoryScope(_, inst, available_scope))
      return error;
  }

  if (mask & uint32_t(spv::ImageOperandsMask::MakeTexelVisibleKHR)) {
    // Checked elsewhere: capability and memory model are correct.
    if (opcode != spv::Op::OpImageRead &&
        opcode != spv::Op::OpImageSparseRead) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelVisibleKHR can only be used with Op"
             << spvOpcodeString(spv::Op::OpImageRead) << " or Op"
             << spvOpcodeString(spv::Op::OpImageSparseRead) << ": Op"
             << spvOpcodeString(opcode);
    }

    if (!(mask & uint32_t(spv::ImageOperandsMask::NonPrivateTexelKHR))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Operand MakeTexelVisibleKHR requires NonPrivateTexelKHR "
                "is also specified: Op"
             << spvOpcodeString(opcode);
    }

    const auto visible_scope = inst->word(word_index++);
    if (auto error = ValidateMemoryScope(_, inst, visible_scope)) return error;
  }

  if (mask & uint32_t(spv::ImageOperandsMask::SignExtend)) {
    // Checked elsewhere: SPIR-V 1.4 version or later.

    // "The texel value is converted to the target value via sign extension.
    // Only valid when the texel type is a scalar or vector of integer type."
    //
    // We don't have enough information to know what the texel type is.
    // In OpenCL, knowledge is deferred until runtime: the image SampledType is
    // void, and the Format is Unknown.
    // In Vulkan, the texel type is only known in all cases by the pipeline
    // setup.
  }

  if (mask & uint32_t(spv::ImageOperandsMask::ZeroExtend)) {
    // Checked elsewhere: SPIR-V 1.4 version or later.

    // "The texel value is converted to the target value via zero extension.
    // Only valid when the texel type is a scalar or vector of integer type."
    //
    // We don't have enough information to know what the texel type is.
    // In OpenCL, knowledge is deferred until runtime: the image SampledType is
    // void, and the Format is Unknown.
    // In Vulkan, the texel type is only known in all cases by the pipeline
    // setup.
  }

  if (mask & uint32_t(spv::ImageOperandsMask::Offsets)) {
    // TODO: add validation
  }

  if (mask & uint32_t(spv::ImageOperandsMask::Nontemporal)) {
    // Checked elsewhere: SPIR-V 1.6 version or later.
  }

  return SPV_SUCCESS;
}

// Validate OpImage*Proj* instructions
spv_result_t ValidateImageProj(ValidationState_t& _, const Instruction* inst,
                               const ImageTypeInfo& info) {
  if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
      info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Rect) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Dim' parameter to be 1D, 2D, 3D or Rect";
  }

  if (info.multisampled != 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'MS' parameter to be 0";
  }

  if (info.arrayed != 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'arrayed' parameter to be 0";
  }

  return SPV_SUCCESS;
}

// Validate OpImage*Read and OpImage*Write instructions
spv_result_t ValidateImageReadWrite(ValidationState_t& _,
                                    const Instruction* inst,
                                    const ImageTypeInfo& info) {
  if (info.sampled == 2) {
    if (info.dim == spv::Dim::Dim1D &&
        !_.HasCapability(spv::Capability::Image1D)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability Image1D is required to access storage image";
    } else if (info.dim == spv::Dim::Rect &&
               !_.HasCapability(spv::Capability::ImageRect)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability ImageRect is required to access storage image";
    } else if (info.dim == spv::Dim::Buffer &&
               !_.HasCapability(spv::Capability::ImageBuffer)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability ImageBuffer is required to access storage image";
    } else if (info.dim == spv::Dim::Cube && info.arrayed == 1 &&
               !_.HasCapability(spv::Capability::ImageCubeArray)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability ImageCubeArray is required to access "
             << "storage image";
    }

    if (info.multisampled == 1 &&
        !_.HasCapability(spv::Capability::ImageMSArray)) {
#if 0
      // TODO(atgoo@github.com) The description of this rule in the spec
      // is unclear and Glslang doesn't declare ImageMSArray. Need to clarify
      // and reenable.
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
          << "Capability ImageMSArray is required to access storage "
          << "image";
#endif
    }
  } else if (info.sampled != 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Sampled' parameter to be 0 or 2";
  }

  return SPV_SUCCESS;
}

// Returns true if opcode is *ImageSparse*, false otherwise.
bool IsSparse(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseFetch:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
    case spv::Op::OpImageSparseTexelsResident:
    case spv::Op::OpImageSparseRead: {
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
  const spv::Op opcode = inst->opcode();

  if (IsSparse(opcode)) {
    const Instruction* const type_inst = _.FindDef(inst->type_id());
    assert(type_inst);

    if (!type_inst || type_inst->opcode() != spv::Op::OpTypeStruct) {
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
const char* GetActualResultTypeStr(spv::Op opcode) {
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

  if (_.IsIntScalarType(info.sampled_type) &&
      (64 == _.GetBitWidth(info.sampled_type)) &&
      !_.HasCapability(spv::Capability::Int64ImageEXT)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Capability Int64ImageEXT is required when using Sampled Type of "
              "64-bit int";
  }

  const auto target_env = _.context()->target_env;
  if (spvIsVulkanEnv(target_env)) {
    if ((!_.IsFloatScalarType(info.sampled_type) &&
         !_.IsIntScalarType(info.sampled_type)) ||
        ((32 != _.GetBitWidth(info.sampled_type)) &&
         (64 != _.GetBitWidth(info.sampled_type))) ||
        ((64 == _.GetBitWidth(info.sampled_type)) &&
         _.IsFloatScalarType(info.sampled_type))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4656)
             << "Expected Sampled Type to be a 32-bit int, 64-bit int or "
                "32-bit float scalar type for Vulkan environment";
    }
  } else if (spvIsOpenCLEnv(target_env)) {
    if (!_.IsVoidType(info.sampled_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Sampled Type must be OpTypeVoid in the OpenCL environment.";
    }
  } else {
    const spv::Op sampled_type_opcode = _.GetIdOpcode(info.sampled_type);
    if (sampled_type_opcode != spv::Op::OpTypeVoid &&
        sampled_type_opcode != spv::Op::OpTypeInt &&
        sampled_type_opcode != spv::Op::OpTypeFloat) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Sampled Type to be either void or"
             << " numerical scalar type";
    }
  }

  // Universal checks on image type operands
  // Dim and Format and Access Qualifier are checked elsewhere.

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

  if (info.dim == spv::Dim::SubpassData) {
    if (info.sampled != 2) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(6214) << "Dim SubpassData requires Sampled to be 2";
    }

    if (info.format != spv::ImageFormat::Unknown) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Dim SubpassData requires format Unknown";
    }
  } else {
    if (info.multisampled && (info.sampled == 2) &&
        !_.HasCapability(spv::Capability::StorageImageMultisample)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability StorageImageMultisample is required when using "
                "multisampled storage image";
    }
  }

  if (spvIsOpenCLEnv(target_env)) {
    if ((info.arrayed == 1) && (info.dim != spv::Dim::Dim1D) &&
        (info.dim != spv::Dim::Dim2D)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "In the OpenCL environment, Arrayed may only be set to 1 "
             << "when Dim is either 1D or 2D.";
    }

    if (info.multisampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "MS must be 0 in the OpenCL environment.";
    }

    if (info.sampled != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Sampled must be 0 in the OpenCL environment.";
    }

    if (info.access_qualifier == spv::AccessQualifier::Max) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "In the OpenCL environment, the optional Access Qualifier"
             << " must be present.";
    }
  }

  if (spvIsVulkanEnv(target_env)) {
    if (info.sampled == 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4657)
             << "Sampled must be 1 or 2 in the Vulkan environment.";
    }

    if (info.dim == spv::Dim::SubpassData && info.arrayed != 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(6214) << "Dim SubpassData requires Arrayed to be 0";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateTypeSampledImage(ValidationState_t& _,
                                      const Instruction* inst) {
  const uint32_t image_type = inst->word(2);
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }
  // OpenCL requires Sampled=0, checked elsewhere.
  // Vulkan uses the Sampled=1 case.
  if ((info.sampled != 0) && (info.sampled != 1)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(4657)
           << "Sampled image type requires an image type with \"Sampled\" "
              "operand set to 0 or 1";
  }

  // This covers both OpTypeSampledImage and OpSampledImage.
  if (_.version() >= SPV_SPIRV_VERSION_WORD(1, 6) &&
      info.dim == spv::Dim::Buffer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "In SPIR-V 1.6 or later, sampled image dimension must not be "
              "Buffer";
  }

  return SPV_SUCCESS;
}

bool IsAllowedSampledImageOperand(spv::Op opcode, ValidationState_t& _) {
  switch (opcode) {
    case spv::Op::OpSampledImage:
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImage:
    case spv::Op::OpImageQueryLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
    case spv::Op::OpCopyObject:
      return true;
    case spv::Op::OpStore:
      if (_.HasCapability(spv::Capability::BindlessTextureNV)) return true;
      return false;
    default:
      return false;
  }
}

spv_result_t ValidateSampledImage(ValidationState_t& _,
                                  const Instruction* inst) {
  if (_.GetIdOpcode(inst->type_id()) != spv::Op::OpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypeSampledImage.";
  }

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
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
             << _.VkErrorID(6671)
             << "Expected Image 'Sampled' parameter to be 1 for Vulkan "
                "environment.";
    }
  } else {
    if (info.sampled != 0 && info.sampled != 1) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled' parameter to be 0 or 1";
    }
  }

  if (info.dim == spv::Dim::SubpassData) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image 'Dim' parameter to be not SubpassData.";
  }

  if (_.GetIdOpcode(_.GetOperandTypeId(inst, 3)) != spv::Op::OpTypeSampler) {
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
                  "Type <id> "
               << _.getIdName(inst->id())
               << " has a consumer in a different basic "
                  "block. The consumer instruction <id> is "
               << _.getIdName(consumer_instr->id()) << ".";
      }

      if (consumer_opcode == spv::Op::OpPhi ||
          consumer_opcode == spv::Op::OpSelect) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Result <id> from OpSampledImage instruction must not appear "
                  "as "
                  "operands of Op"
               << spvOpcodeString(static_cast<spv::Op>(consumer_opcode)) << "."
               << " Found result <id> " << _.getIdName(inst->id())
               << " as an operand of <id> " << _.getIdName(consumer_instr->id())
               << ".";
      }

      if (!IsAllowedSampledImageOperand(consumer_opcode, _)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Result <id> from OpSampledImage instruction must not appear "
                  "as operand for Op"
               << spvOpcodeString(static_cast<spv::Op>(consumer_opcode))
               << ", since it is not specified as taking an "
               << "OpTypeSampledImage."
               << " Found result <id> " << _.getIdName(inst->id())
               << " as an operand of <id> " << _.getIdName(consumer_instr->id())
               << ".";
      }
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageTexelPointer(ValidationState_t& _,
                                       const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  if (result_type->opcode() != spv::Op::OpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer";
  }

  const auto storage_class = result_type->GetOperandAs<spv::StorageClass>(1);
  if (storage_class != spv::StorageClass::Image) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer whose Storage Class "
              "operand is Image";
  }

  const auto ptr_type = result_type->GetOperandAs<uint32_t>(2);
  const auto ptr_opcode = _.GetIdOpcode(ptr_type);
  if (ptr_opcode != spv::Op::OpTypeInt && ptr_opcode != spv::Op::OpTypeFloat &&
      ptr_opcode != spv::Op::OpTypeVoid) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypePointer whose Type operand "
              "must be a scalar numerical type or OpTypeVoid";
  }

  const auto image_ptr = _.FindDef(_.GetOperandTypeId(inst, 2));
  if (!image_ptr || image_ptr->opcode() != spv::Op::OpTypePointer) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be OpTypePointer";
  }

  const auto image_type = image_ptr->GetOperandAs<uint32_t>(2);
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
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

  if (info.dim == spv::Dim::SubpassData) {
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
      case spv::Dim::Dim1D:
        expected_coord_size = 2;
        break;
      case spv::Dim::Cube:
      case spv::Dim::Dim2D:
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

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if ((info.format != spv::ImageFormat::R64i) &&
        (info.format != spv::ImageFormat::R64ui) &&
        (info.format != spv::ImageFormat::R32f) &&
        (info.format != spv::ImageFormat::R32i) &&
        (info.format != spv::ImageFormat::R32ui)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4658)
             << "Expected the Image Format in Image to be R64i, R64ui, R32f, "
                "R32i, or R32ui for Vulkan environment";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateImageLod(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (IsProj(opcode)) {
    if (spv_result_t result = ValidateImageProj(_, inst, info)) return result;
  }

  if (info.multisampled) {
    // When using image operands, the Sample image operand is required if and
    // only if the image is multisampled (MS=1). The Sample image operand is
    // only allowed for fetch, read, and write.
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Sampling operation is invalid for multisample image";
  }

  if (_.GetIdOpcode(info.sampled_type) != spv::Op::OpTypeVoid) {
    const uint32_t texel_component_type =
        _.GetComponentType(actual_result_type);
    if (texel_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if ((opcode == spv::Op::OpImageSampleExplicitLod ||
       opcode == spv::Op::OpImageSparseSampleExplicitLod) &&
      _.HasCapability(spv::Capability::Kernel)) {
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

  const uint32_t mask = inst->words().size() <= 5 ? 0 : inst->word(5);

  if (mask & uint32_t(spv::ImageOperandsMask::ConstOffset)) {
    if (spvIsOpenCLEnv(_.context()->target_env)) {
      if (opcode == spv::Op::OpImageSampleExplicitLod) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "ConstOffset image operand not allowed "
               << "in the OpenCL environment.";
      }
    }
  }

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

// Validates anything OpImage*Dref* instruction
spv_result_t ValidateImageDref(ValidationState_t& _, const Instruction* inst,
                               const ImageTypeInfo& info) {
  const uint32_t dref_type = _.GetOperandTypeId(inst, 4);
  if (!_.IsFloatScalarType(dref_type) || _.GetBitWidth(dref_type) != 32) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Dref to be of 32-bit float type";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (info.dim == spv::Dim::Dim3D) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4777)
             << "In Vulkan, OpImage*Dref* instructions must not use images "
                "with a 3D Dim";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateImageDrefLod(ValidationState_t& _,
                                  const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (IsProj(opcode)) {
    if (spv_result_t result = ValidateImageProj(_, inst, info)) return result;
  }

  if (info.multisampled) {
    // When using image operands, the Sample image operand is required if and
    // only if the image is multisampled (MS=1). The Sample image operand is
    // only allowed for fetch, read, and write.
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Dref sampling operation is invalid for multisample image";
  }

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

  if (spv_result_t result = ValidateImageDref(_, inst, info)) return result;

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 7))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageFetch(ValidationState_t& _, const Instruction* inst) {
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type)) {
    return error;
  }

  const spv::Op opcode = inst->opcode();
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (_.GetIdOpcode(info.sampled_type) != spv::Op::OpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (info.dim == spv::Dim::Cube) {
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

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageGather(ValidationState_t& _,
                                 const Instruction* inst) {
  uint32_t actual_result_type = 0;
  if (spv_result_t error = GetActualResultType(_, inst, &actual_result_type))
    return error;

  const spv::Op opcode = inst->opcode();
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Sampled Image to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.multisampled) {
    // When using image operands, the Sample image operand is required if and
    // only if the image is multisampled (MS=1). The Sample image operand is
    // only allowed for fetch, read, and write.
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Gather operation is invalid for multisample image";
  }

  if (opcode == spv::Op::OpImageDrefGather ||
      opcode == spv::Op::OpImageSparseDrefGather ||
      _.GetIdOpcode(info.sampled_type) != spv::Op::OpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (info.dim != spv::Dim::Dim2D && info.dim != spv::Dim::Cube &&
      info.dim != spv::Dim::Rect) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(4777)
           << "Expected Image 'Dim' to be 2D, Cube, or Rect";
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

  if (opcode == spv::Op::OpImageGather ||
      opcode == spv::Op::OpImageSparseGather) {
    const uint32_t component = inst->GetOperandAs<uint32_t>(4);
    const uint32_t component_index_type = _.GetTypeId(component);
    if (!_.IsIntScalarType(component_index_type) ||
        _.GetBitWidth(component_index_type) != 32) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Component to be 32-bit int scalar";
    }
    if (spvIsVulkanEnv(_.context()->target_env)) {
      if (!spvOpcodeIsConstant(_.GetIdOpcode(component))) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(4664)
               << "Expected Component Operand to be a const object for Vulkan "
                  "environment";
      }
    }
  } else {
    assert(opcode == spv::Op::OpImageDrefGather ||
           opcode == spv::Op::OpImageSparseDrefGather);
    if (spv_result_t result = ValidateImageDref(_, inst, info)) return result;
  }

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 7))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageRead(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
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

  const auto target_env = _.context()->target_env;
  // Vulkan requires the result to be a 4-element int or float
  // vector.
  if (spvIsVulkanEnv(target_env)) {
    if (_.GetDimension(actual_result_type) != 4) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4780) << "Expected "
             << GetActualResultTypeStr(opcode) << " to have 4 components";
    }
  }  // Check OpenCL below, after we get the image info.

  const uint32_t image_type = _.GetOperandTypeId(inst, 2);
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (spvIsOpenCLEnv(target_env)) {
    // In OpenCL, a read from a depth image returns a scalar float. In other
    // cases, the result is always a 4-element vector.
    // https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_data_format_for_reading_and_writing_images
    // https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#image-read-and-write-functions
    // The builtins for reading depth images are:
    //   float read_imagef(aQual image2d_depth_t image, int2 coord)
    //   float read_imagef(aQual image2d_array_depth_t image, int4 coord)
    if (info.depth) {
      if (!_.IsFloatScalarType(actual_result_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected " << GetActualResultTypeStr(opcode)
               << " from a depth image read to result in a scalar float value";
      }
    } else {
      if (_.GetDimension(actual_result_type) != 4) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected " << GetActualResultTypeStr(opcode)
               << " to have 4 components";
      }
    }

    const uint32_t mask = inst->words().size() <= 5 ? 0 : inst->word(5);
    if (mask & uint32_t(spv::ImageOperandsMask::ConstOffset)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ConstOffset image operand not allowed "
             << "in the OpenCL environment.";
    }
  }

  if (info.dim == spv::Dim::SubpassData) {
    if (opcode == spv::Op::OpImageSparseRead) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image Dim SubpassData cannot be used with ImageSparseRead";
    }

    _.function(inst->function()->id())
        ->RegisterExecutionModelLimitation(
            spv::ExecutionModel::Fragment,
            std::string("Dim SubpassData requires Fragment execution model: ") +
                spvOpcodeString(opcode));
  }

  if (_.GetIdOpcode(info.sampled_type) != spv::Op::OpTypeVoid) {
    const uint32_t result_component_type =
        _.GetComponentType(actual_result_type);
    if (result_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as "
             << GetActualResultTypeStr(opcode) << " components";
    }
  }

  if (spv_result_t result = ValidateImageReadWrite(_, inst, info))
    return result;

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

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (info.format == spv::ImageFormat::Unknown &&
        info.dim != spv::Dim::SubpassData &&
        !_.HasCapability(spv::Capability::StorageImageReadWithoutFormat)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability StorageImageReadWithoutFormat is required to "
             << "read storage image";
    }
  }

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 6))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImageWrite(ValidationState_t& _, const Instruction* inst) {
  const uint32_t image_type = _.GetOperandTypeId(inst, 0);
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.dim == spv::Dim::SubpassData) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image 'Dim' cannot be SubpassData";
  }

  if (spv_result_t result = ValidateImageReadWrite(_, inst, info))
    return result;

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

  // because it needs to match with 'Sampled Type' the Texel can't be a boolean
  const uint32_t texel_type = _.GetOperandTypeId(inst, 2);
  if (!_.IsIntScalarOrVectorType(texel_type) &&
      !_.IsFloatScalarOrVectorType(texel_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Texel to be int or float vector or scalar";
  }

  if (_.GetIdOpcode(info.sampled_type) != spv::Op::OpTypeVoid) {
    const uint32_t texel_component_type = _.GetComponentType(texel_type);
    if (texel_component_type != info.sampled_type) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Expected Image 'Sampled Type' to be the same as Texel "
             << "components";
    }
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (info.format == spv::ImageFormat::Unknown &&
        info.dim != spv::Dim::SubpassData &&
        !_.HasCapability(spv::Capability::StorageImageWriteWithoutFormat)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Capability StorageImageWriteWithoutFormat is required to "
                "write "
             << "to storage image";
    }
  }

  if (inst->words().size() > 4) {
    if (spvIsOpenCLEnv(_.context()->target_env)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Optional Image Operands are not allowed in the OpenCL "
             << "environment.";
    }
  }

  if (spv_result_t result =
          ValidateImageOperands(_, inst, info, /* word_index = */ 5))
    return result;

  return SPV_SUCCESS;
}

spv_result_t ValidateImage(ValidationState_t& _, const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  if (_.GetIdOpcode(result_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be OpTypeImage";
  }

  const uint32_t sampled_image_type = _.GetOperandTypeId(inst, 2);
  const Instruction* sampled_image_type_inst = _.FindDef(sampled_image_type);
  assert(sampled_image_type_inst);

  if (sampled_image_type_inst->opcode() != spv::Op::OpTypeSampledImage) {
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
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
    case spv::Dim::Dim1D:
      expected_num_components += 1;
      break;
    case spv::Dim::Dim2D:
    case spv::Dim::Cube:
      expected_num_components += 2;
      break;
    case spv::Dim::Dim3D:
      expected_num_components += 3;
      break;
    default:
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, 2D, 3D or Cube";
  }

  if (info.multisampled != 0) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Image 'MS' must be 0";
  }

  const auto target_env = _.context()->target_env;
  if (spvIsVulkanEnv(target_env)) {
    if (info.sampled != 1) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4659)
             << "OpImageQuerySizeLod must only consume an \"Image\" operand "
                "whose type has its \"Sampled\" operand set to 1";
    }
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
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
    case spv::Dim::Dim1D:
    case spv::Dim::Buffer:
      expected_num_components += 1;
      break;
    case spv::Dim::Dim2D:
    case spv::Dim::Cube:
    case spv::Dim::Rect:
      expected_num_components += 2;
      break;
    case spv::Dim::Dim3D:
      expected_num_components += 3;
      break;
    default:
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, Buffer, 2D, Cube, 3D or Rect";
  }

  if (info.dim == spv::Dim::Dim1D || info.dim == spv::Dim::Dim2D ||
      info.dim == spv::Dim::Dim3D || info.dim == spv::Dim::Cube) {
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

  if (_.GetIdOpcode(_.GetOperandTypeId(inst, 2)) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected operand to be of type OpTypeImage";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateImageQueryLod(ValidationState_t& _,
                                   const Instruction* inst) {
  _.function(inst->function()->id())
      ->RegisterExecutionModelLimitation(
          [&](spv::ExecutionModel model, std::string* message) {
            if (model != spv::ExecutionModel::Fragment &&
                model != spv::ExecutionModel::GLCompute) {
              if (message) {
                *message = std::string(
                    "OpImageQueryLod requires Fragment or GLCompute execution "
                    "model");
              }
              return false;
            }
            return true;
          });
  _.function(inst->function()->id())
      ->RegisterLimitation([](const ValidationState_t& state,
                              const Function* entry_point,
                              std::string* message) {
        const auto* models = state.GetExecutionModels(entry_point->id());
        const auto* modes = state.GetExecutionModes(entry_point->id());
        if (models->find(spv::ExecutionModel::GLCompute) != models->end() &&
            modes->find(spv::ExecutionMode::DerivativeGroupLinearNV) ==
                modes->end() &&
            modes->find(spv::ExecutionMode::DerivativeGroupQuadsNV) ==
                modes->end()) {
          if (message) {
            *message = std::string(
                "OpImageQueryLod requires DerivativeGroupQuadsNV "
                "or DerivativeGroupLinearNV execution mode for GLCompute "
                "execution model");
          }
          return false;
        }
        return true;
      });

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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeSampledImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image operand to be of type OpTypeSampledImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
      info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Cube) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Image 'Dim' must be 1D, 2D, 3D or Cube";
  }

  const uint32_t coord_type = _.GetOperandTypeId(inst, 3);
  if (_.HasCapability(spv::Capability::Kernel)) {
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

  // The operad is a sampled image.
  // The sampled image type is already checked to be parameterized by an image
  // type with Sampled=0 or Sampled=1.  Vulkan bans Sampled=0, and so we have
  // Sampled=1.  So the validator already enforces Vulkan VUID 4659:
  //   OpImageQuerySizeLod must only consume an “Image” operand whose type has
  //   its "Sampled" operand set to 1
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
  if (_.GetIdOpcode(image_type) != spv::Op::OpTypeImage) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Image to be of type OpTypeImage";
  }

  ImageTypeInfo info;
  if (!GetImageTypeInfo(_, image_type, &info)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Corrupt image type definition";
  }

  const spv::Op opcode = inst->opcode();
  if (opcode == spv::Op::OpImageQueryLevels) {
    if (info.dim != spv::Dim::Dim1D && info.dim != spv::Dim::Dim2D &&
        info.dim != spv::Dim::Dim3D && info.dim != spv::Dim::Cube) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Image 'Dim' must be 1D, 2D, 3D or Cube";
    }
    const auto target_env = _.context()->target_env;
    if (spvIsVulkanEnv(target_env)) {
      if (info.sampled != 1) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(4659)
               << "OpImageQueryLevels must only consume an \"Image\" operand "
                  "whose type has its \"Sampled\" operand set to 1";
      }
    }
  } else {
    assert(opcode == spv::Op::OpImageQuerySamples);
    if (info.dim != spv::Dim::Dim2D) {
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
  const spv::Op opcode = inst->opcode();
  if (IsImplicitLod(opcode)) {
    _.function(inst->function()->id())
        ->RegisterExecutionModelLimitation([opcode](spv::ExecutionModel model,
                                                    std::string* message) {
          if (model != spv::ExecutionModel::Fragment &&
              model != spv::ExecutionModel::GLCompute) {
            if (message) {
              *message =
                  std::string(
                      "ImplicitLod instructions require Fragment or GLCompute "
                      "execution model: ") +
                  spvOpcodeString(opcode);
            }
            return false;
          }
          return true;
        });
    _.function(inst->function()->id())
        ->RegisterLimitation([opcode](const ValidationState_t& state,
                                      const Function* entry_point,
                                      std::string* message) {
          const auto* models = state.GetExecutionModels(entry_point->id());
          const auto* modes = state.GetExecutionModes(entry_point->id());
          if (models &&
              models->find(spv::ExecutionModel::GLCompute) != models->end() &&
              (!modes ||
               (modes->find(spv::ExecutionMode::DerivativeGroupLinearNV) ==
                    modes->end() &&
                modes->find(spv::ExecutionMode::DerivativeGroupQuadsNV) ==
                    modes->end()))) {
            if (message) {
              *message =
                  std::string(
                      "ImplicitLod instructions require DerivativeGroupQuadsNV "
                      "or DerivativeGroupLinearNV execution mode for GLCompute "
                      "execution model: ") +
                  spvOpcodeString(opcode);
            }
            return false;
          }
          return true;
        });
  }

  switch (opcode) {
    case spv::Op::OpTypeImage:
      return ValidateTypeImage(_, inst);
    case spv::Op::OpTypeSampledImage:
      return ValidateTypeSampledImage(_, inst);
    case spv::Op::OpSampledImage:
      return ValidateSampledImage(_, inst);
    case spv::Op::OpImageTexelPointer:
      return ValidateImageTexelPointer(_, inst);

    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
      return ValidateImageLod(_, inst);

    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
      return ValidateImageDrefLod(_, inst);

    case spv::Op::OpImageFetch:
    case spv::Op::OpImageSparseFetch:
      return ValidateImageFetch(_, inst);

    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
      return ValidateImageGather(_, inst);

    case spv::Op::OpImageRead:
    case spv::Op::OpImageSparseRead:
      return ValidateImageRead(_, inst);

    case spv::Op::OpImageWrite:
      return ValidateImageWrite(_, inst);

    case spv::Op::OpImage:
      return ValidateImage(_, inst);

    case spv::Op::OpImageQueryFormat:
    case spv::Op::OpImageQueryOrder:
      return ValidateImageQueryFormatOrOrder(_, inst);

    case spv::Op::OpImageQuerySizeLod:
      return ValidateImageQuerySizeLod(_, inst);
    case spv::Op::OpImageQuerySize:
      return ValidateImageQuerySize(_, inst);
    case spv::Op::OpImageQueryLod:
      return ValidateImageQueryLod(_, inst);

    case spv::Op::OpImageQueryLevels:
    case spv::Op::OpImageQuerySamples:
      return ValidateImageQueryLevelsOrSamples(_, inst);

    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
      return ValidateImageSparseLod(_, inst);

    case spv::Op::OpImageSparseTexelsResident:
      return ValidateImageSparseTexelsResident(_, inst);

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
