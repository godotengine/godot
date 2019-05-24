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

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

bool IsValidWebGPUDecoration(uint32_t decoration) {
  switch (decoration) {
    case SpvDecorationSpecId:
    case SpvDecorationBlock:
    case SpvDecorationRowMajor:
    case SpvDecorationColMajor:
    case SpvDecorationArrayStride:
    case SpvDecorationMatrixStride:
    case SpvDecorationBuiltIn:
    case SpvDecorationNoPerspective:
    case SpvDecorationFlat:
    case SpvDecorationCentroid:
    case SpvDecorationRestrict:
    case SpvDecorationAliased:
    case SpvDecorationNonWritable:
    case SpvDecorationNonReadable:
    case SpvDecorationUniform:
    case SpvDecorationLocation:
    case SpvDecorationComponent:
    case SpvDecorationIndex:
    case SpvDecorationBinding:
    case SpvDecorationDescriptorSet:
    case SpvDecorationOffset:
    case SpvDecorationNoContraction:
      return true;
    default:
      return false;
  }
}

std::string LogStringForDecoration(uint32_t decoration) {
  switch (decoration) {
    case SpvDecorationRelaxedPrecision:
      return "RelaxedPrecision";
    case SpvDecorationSpecId:
      return "SpecId";
    case SpvDecorationBlock:
      return "Block";
    case SpvDecorationBufferBlock:
      return "BufferBlock";
    case SpvDecorationRowMajor:
      return "RowMajor";
    case SpvDecorationColMajor:
      return "ColMajor";
    case SpvDecorationArrayStride:
      return "ArrayStride";
    case SpvDecorationMatrixStride:
      return "MatrixStride";
    case SpvDecorationGLSLShared:
      return "GLSLShared";
    case SpvDecorationGLSLPacked:
      return "GLSLPacked";
    case SpvDecorationCPacked:
      return "CPacked";
    case SpvDecorationBuiltIn:
      return "BuiltIn";
    case SpvDecorationNoPerspective:
      return "NoPerspective";
    case SpvDecorationFlat:
      return "Flat";
    case SpvDecorationPatch:
      return "Patch";
    case SpvDecorationCentroid:
      return "Centroid";
    case SpvDecorationSample:
      return "Sample";
    case SpvDecorationInvariant:
      return "Invariant";
    case SpvDecorationRestrict:
      return "Restrict";
    case SpvDecorationAliased:
      return "Aliased";
    case SpvDecorationVolatile:
      return "Volatile";
    case SpvDecorationConstant:
      return "Constant";
    case SpvDecorationCoherent:
      return "Coherent";
    case SpvDecorationNonWritable:
      return "NonWritable";
    case SpvDecorationNonReadable:
      return "NonReadable";
    case SpvDecorationUniform:
      return "Uniform";
    case SpvDecorationSaturatedConversion:
      return "SaturatedConversion";
    case SpvDecorationStream:
      return "Stream";
    case SpvDecorationLocation:
      return "Location";
    case SpvDecorationComponent:
      return "Component";
    case SpvDecorationIndex:
      return "Index";
    case SpvDecorationBinding:
      return "Binding";
    case SpvDecorationDescriptorSet:
      return "DescriptorSet";
    case SpvDecorationOffset:
      return "Offset";
    case SpvDecorationXfbBuffer:
      return "XfbBuffer";
    case SpvDecorationXfbStride:
      return "XfbStride";
    case SpvDecorationFuncParamAttr:
      return "FuncParamAttr";
    case SpvDecorationFPRoundingMode:
      return "FPRoundingMode";
    case SpvDecorationFPFastMathMode:
      return "FPFastMathMode";
    case SpvDecorationLinkageAttributes:
      return "LinkageAttributes";
    case SpvDecorationNoContraction:
      return "NoContraction";
    case SpvDecorationInputAttachmentIndex:
      return "InputAttachmentIndex";
    case SpvDecorationAlignment:
      return "Alignment";
    case SpvDecorationMaxByteOffset:
      return "MaxByteOffset";
    case SpvDecorationAlignmentId:
      return "AlignmentId";
    case SpvDecorationMaxByteOffsetId:
      return "MaxByteOffsetId";
    case SpvDecorationNoSignedWrap:
      return "NoSignedWrap";
    case SpvDecorationNoUnsignedWrap:
      return "NoUnsignedWrap";
    case SpvDecorationExplicitInterpAMD:
      return "ExplicitInterpAMD";
    case SpvDecorationOverrideCoverageNV:
      return "OverrideCoverageNV";
    case SpvDecorationPassthroughNV:
      return "PassthroughNV";
    case SpvDecorationViewportRelativeNV:
      return "ViewportRelativeNV";
    case SpvDecorationSecondaryViewportRelativeNV:
      return "SecondaryViewportRelativeNV";
    case SpvDecorationPerPrimitiveNV:
      return "PerPrimitiveNV";
    case SpvDecorationPerViewNV:
      return "PerViewNV";
    case SpvDecorationPerTaskNV:
      return "PerTaskNV";
    case SpvDecorationPerVertexNV:
      return "PerVertexNV";
    case SpvDecorationNonUniformEXT:
      return "NonUniformEXT";
    case SpvDecorationRestrictPointerEXT:
      return "RestrictPointerEXT";
    case SpvDecorationAliasedPointerEXT:
      return "AliasedPointerEXT";
    case SpvDecorationHlslCounterBufferGOOGLE:
      return "HlslCounterBufferGOOGLE";
    case SpvDecorationHlslSemanticGOOGLE:
      return "HlslSemanticGOOGLE";
    default:
      break;
  }
  return "Unknown";
}

spv_result_t ValidateDecorate(ValidationState_t& _, const Instruction* inst) {
  const auto decoration = inst->GetOperandAs<uint32_t>(1);
  if (decoration == SpvDecorationSpecId) {
    const auto target_id = inst->GetOperandAs<uint32_t>(0);
    const auto target = _.FindDef(target_id);
    if (!target || !spvOpcodeIsScalarSpecConstant(target->opcode())) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpDecorate SpecId decoration target <id> '"
             << _.getIdName(target_id)
             << "' is not a scalar specialization constant.";
    }
  }

  if (spvIsWebGPUEnv(_.context()->target_env) &&
      !IsValidWebGPUDecoration(decoration)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpDecorate decoration '" << LogStringForDecoration(decoration)
           << "' is not valid for the WebGPU execution environment.";
  }

  // TODO: Add validations for all decorations.
  return SPV_SUCCESS;
}

spv_result_t ValidateMemberDecorate(ValidationState_t& _,
                                    const Instruction* inst) {
  const auto struct_type_id = inst->GetOperandAs<uint32_t>(0);
  const auto struct_type = _.FindDef(struct_type_id);
  if (!struct_type || SpvOpTypeStruct != struct_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpMemberDecorate Structure type <id> '"
           << _.getIdName(struct_type_id) << "' is not a struct type.";
  }
  const auto member = inst->GetOperandAs<uint32_t>(1);
  const auto member_count =
      static_cast<uint32_t>(struct_type->words().size() - 2);
  if (member_count <= member) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Index " << member
           << " provided in OpMemberDecorate for struct <id> "
           << _.getIdName(struct_type_id)
           << " is out of bounds. The structure has " << member_count
           << " members. Largest valid index is " << member_count - 1 << ".";
  }

  const auto decoration = inst->GetOperandAs<uint32_t>(2);
  if (spvIsWebGPUEnv(_.context()->target_env) &&
      !IsValidWebGPUDecoration(decoration)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpMemberDecorate decoration  '" << _.getIdName(decoration)
           << "' is not valid for the WebGPU execution environment.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateDecorationGroup(ValidationState_t& _,
                                     const Instruction* inst) {
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "OpDecorationGroup is not allowed in the WebGPU execution "
           << "environment.";
  }

  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  const auto decoration_group = _.FindDef(decoration_group_id);
  for (auto pair : decoration_group->uses()) {
    auto use = pair.first;
    if (use->opcode() != SpvOpDecorate && use->opcode() != SpvOpGroupDecorate &&
        use->opcode() != SpvOpGroupMemberDecorate &&
        use->opcode() != SpvOpName) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Result id of OpDecorationGroup can only "
             << "be targeted by OpName, OpGroupDecorate, "
             << "OpDecorate, and OpGroupMemberDecorate";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupDecorate(ValidationState_t& _,
                                   const Instruction* inst) {
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "OpGroupDecorate is not allowed in the WebGPU execution "
           << "environment.";
  }

  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  auto decoration_group = _.FindDef(decoration_group_id);
  if (!decoration_group || SpvOpDecorationGroup != decoration_group->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpGroupDecorate Decoration group <id> '"
           << _.getIdName(decoration_group_id)
           << "' is not a decoration group.";
  }
  for (unsigned i = 1; i < inst->operands().size(); ++i) {
    auto target_id = inst->GetOperandAs<uint32_t>(i);
    auto target = _.FindDef(target_id);
    if (!target || target->opcode() == SpvOpDecorationGroup) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpGroupDecorate may not target OpDecorationGroup <id> '"
             << _.getIdName(target_id) << "'";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupMemberDecorate(ValidationState_t& _,
                                         const Instruction* inst) {
  if (spvIsWebGPUEnv(_.context()->target_env)) {
    return _.diag(SPV_ERROR_INVALID_BINARY, inst)
           << "OpGroupMemberDecorate is not allowed in the WebGPU execution "
           << "environment.";
  }

  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  const auto decoration_group = _.FindDef(decoration_group_id);
  if (!decoration_group || SpvOpDecorationGroup != decoration_group->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpGroupMemberDecorate Decoration group <id> '"
           << _.getIdName(decoration_group_id)
           << "' is not a decoration group.";
  }
  // Grammar checks ensures that the number of arguments to this instruction
  // is an odd number: 1 decoration group + (id,literal) pairs.
  for (size_t i = 1; i + 1 < inst->operands().size(); i += 2) {
    const uint32_t struct_id = inst->GetOperandAs<uint32_t>(i);
    const uint32_t index = inst->GetOperandAs<uint32_t>(i + 1);
    auto struct_instr = _.FindDef(struct_id);
    if (!struct_instr || SpvOpTypeStruct != struct_instr->opcode()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpGroupMemberDecorate Structure type <id> '"
             << _.getIdName(struct_id) << "' is not a struct type.";
    }
    const uint32_t num_struct_members =
        static_cast<uint32_t>(struct_instr->words().size() - 2);
    if (index >= num_struct_members) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Index " << index
             << " provided in OpGroupMemberDecorate for struct <id> "
             << _.getIdName(struct_id)
             << " is out of bounds. The structure has " << num_struct_members
             << " members. Largest valid index is " << num_struct_members - 1
             << ".";
    }
  }
  return SPV_SUCCESS;
}

// Registers necessary decoration(s) for the appropriate IDs based on the
// instruction.
spv_result_t RegisterDecorations(ValidationState_t& _,
                                 const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpDecorate: {
      const uint32_t target_id = inst->word(1);
      const SpvDecoration dec_type = static_cast<SpvDecoration>(inst->word(2));
      std::vector<uint32_t> dec_params;
      if (inst->words().size() > 3) {
        dec_params.insert(dec_params.end(), inst->words().begin() + 3,
                          inst->words().end());
      }
      _.RegisterDecorationForId(target_id, Decoration(dec_type, dec_params));
      break;
    }
    case SpvOpMemberDecorate: {
      const uint32_t struct_id = inst->word(1);
      const uint32_t index = inst->word(2);
      const SpvDecoration dec_type = static_cast<SpvDecoration>(inst->word(3));
      std::vector<uint32_t> dec_params;
      if (inst->words().size() > 4) {
        dec_params.insert(dec_params.end(), inst->words().begin() + 4,
                          inst->words().end());
      }
      _.RegisterDecorationForId(struct_id,
                                Decoration(dec_type, dec_params, index));
      break;
    }
    case SpvOpDecorationGroup: {
      // We don't need to do anything right now. Assigning decorations to groups
      // will be taken care of via OpGroupDecorate.
      break;
    }
    case SpvOpGroupDecorate: {
      // Word 1 is the group <id>. All subsequent words are target <id>s that
      // are going to be decorated with the decorations.
      const uint32_t decoration_group_id = inst->word(1);
      std::vector<Decoration>& group_decorations =
          _.id_decorations(decoration_group_id);
      for (size_t i = 2; i < inst->words().size(); ++i) {
        const uint32_t target_id = inst->word(i);
        _.RegisterDecorationsForId(target_id, group_decorations.begin(),
                                   group_decorations.end());
      }
      break;
    }
    case SpvOpGroupMemberDecorate: {
      // Word 1 is the Decoration Group <id> followed by (struct<id>,literal)
      // pairs. All decorations of the group should be applied to all the struct
      // members that are specified in the instructions.
      const uint32_t decoration_group_id = inst->word(1);
      std::vector<Decoration>& group_decorations =
          _.id_decorations(decoration_group_id);
      // Grammar checks ensures that the number of arguments to this instruction
      // is an odd number: 1 decoration group + (id,literal) pairs.
      for (size_t i = 2; i + 1 < inst->words().size(); i = i + 2) {
        const uint32_t struct_id = inst->word(i);
        const uint32_t index = inst->word(i + 1);
        // ID validation phase ensures this is in fact a struct instruction and
        // that the index is not out of bound.
        _.RegisterDecorationsForStructMember(struct_id, index,
                                             group_decorations.begin(),
                                             group_decorations.end());
      }
      break;
    }
    default:
      break;
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t AnnotationPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpDecorate:
      if (auto error = ValidateDecorate(_, inst)) return error;
      break;
    case SpvOpMemberDecorate:
      if (auto error = ValidateMemberDecorate(_, inst)) return error;
      break;
    case SpvOpDecorationGroup:
      if (auto error = ValidateDecorationGroup(_, inst)) return error;
      break;
    case SpvOpGroupDecorate:
      if (auto error = ValidateGroupDecorate(_, inst)) return error;
      break;
    case SpvOpGroupMemberDecorate:
      if (auto error = ValidateGroupMemberDecorate(_, inst)) return error;
      break;
    default:
      break;
  }

  // In order to validate decoration rules, we need to know all the decorations
  // that are applied to any given <id>.
  RegisterDecorations(_, inst);

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
