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

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if the decoration takes ID parameters.
// TODO(dneto): This can be generated from the grammar.
bool DecorationTakesIdParameters(spv::Decoration type) {
  switch (type) {
    case spv::Decoration::UniformId:
    case spv::Decoration::AlignmentId:
    case spv::Decoration::MaxByteOffsetId:
    case spv::Decoration::HlslCounterBufferGOOGLE:
      return true;
    default:
      break;
  }
  return false;
}

bool IsMemberDecorationOnly(spv::Decoration dec) {
  switch (dec) {
    case spv::Decoration::RowMajor:
    case spv::Decoration::ColMajor:
    case spv::Decoration::MatrixStride:
      // SPIR-V spec bug? Offset is generated on variables when dealing with
      // transform feedback.
      // case spv::Decoration::Offset:
      return true;
    default:
      break;
  }
  return false;
}

bool IsNotMemberDecoration(spv::Decoration dec) {
  switch (dec) {
    case spv::Decoration::SpecId:
    case spv::Decoration::Block:
    case spv::Decoration::BufferBlock:
    case spv::Decoration::ArrayStride:
    case spv::Decoration::GLSLShared:
    case spv::Decoration::GLSLPacked:
    case spv::Decoration::CPacked:
    // TODO: https://github.com/KhronosGroup/glslang/issues/703:
    // glslang applies Restrict to structure members.
    // case spv::Decoration::Restrict:
    case spv::Decoration::Aliased:
    case spv::Decoration::Constant:
    case spv::Decoration::Uniform:
    case spv::Decoration::UniformId:
    case spv::Decoration::SaturatedConversion:
    case spv::Decoration::Index:
    case spv::Decoration::Binding:
    case spv::Decoration::DescriptorSet:
    case spv::Decoration::FuncParamAttr:
    case spv::Decoration::FPRoundingMode:
    case spv::Decoration::FPFastMathMode:
    case spv::Decoration::LinkageAttributes:
    case spv::Decoration::NoContraction:
    case spv::Decoration::InputAttachmentIndex:
    case spv::Decoration::Alignment:
    case spv::Decoration::MaxByteOffset:
    case spv::Decoration::AlignmentId:
    case spv::Decoration::MaxByteOffsetId:
    case spv::Decoration::NoSignedWrap:
    case spv::Decoration::NoUnsignedWrap:
    case spv::Decoration::NonUniform:
    case spv::Decoration::RestrictPointer:
    case spv::Decoration::AliasedPointer:
    case spv::Decoration::CounterBuffer:
      return true;
    default:
      break;
  }
  return false;
}

spv_result_t ValidateDecorationTarget(ValidationState_t& _, spv::Decoration dec,
                                      const Instruction* inst,
                                      const Instruction* target) {
  auto fail = [&_, dec, inst, target](uint32_t vuid) -> DiagnosticStream {
    DiagnosticStream ds = std::move(
        _.diag(SPV_ERROR_INVALID_ID, inst)
        << _.VkErrorID(vuid) << _.SpvDecorationString(dec)
        << " decoration on target <id> " << _.getIdName(target->id()) << " ");
    return ds;
  };
  switch (dec) {
    case spv::Decoration::SpecId:
      if (!spvOpcodeIsScalarSpecConstant(target->opcode())) {
        return fail(0) << "must be a scalar specialization constant";
      }
      break;
    case spv::Decoration::Block:
    case spv::Decoration::BufferBlock:
    case spv::Decoration::GLSLShared:
    case spv::Decoration::GLSLPacked:
    case spv::Decoration::CPacked:
      if (target->opcode() != spv::Op::OpTypeStruct) {
        return fail(0) << "must be a structure type";
      }
      break;
    case spv::Decoration::ArrayStride:
      if (target->opcode() != spv::Op::OpTypeArray &&
          target->opcode() != spv::Op::OpTypeRuntimeArray &&
          target->opcode() != spv::Op::OpTypePointer) {
        return fail(0) << "must be an array or pointer type";
      }
      break;
    case spv::Decoration::BuiltIn:
      if (target->opcode() != spv::Op::OpVariable &&
          !spvOpcodeIsConstant(target->opcode())) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "BuiltIns can only target variables, structure members or "
                  "constants";
      }
      if (_.HasCapability(spv::Capability::Shader) &&
          inst->GetOperandAs<spv::BuiltIn>(2) == spv::BuiltIn::WorkgroupSize) {
        if (!spvOpcodeIsConstant(target->opcode())) {
          return fail(0) << "must be a constant for WorkgroupSize";
        }
      } else if (target->opcode() != spv::Op::OpVariable) {
        return fail(0) << "must be a variable";
      }
      break;
    case spv::Decoration::NoPerspective:
    case spv::Decoration::Flat:
    case spv::Decoration::Patch:
    case spv::Decoration::Centroid:
    case spv::Decoration::Sample:
    case spv::Decoration::Restrict:
    case spv::Decoration::Aliased:
    case spv::Decoration::Volatile:
    case spv::Decoration::Coherent:
    case spv::Decoration::NonWritable:
    case spv::Decoration::NonReadable:
    case spv::Decoration::XfbBuffer:
    case spv::Decoration::XfbStride:
    case spv::Decoration::Component:
    case spv::Decoration::Stream:
    case spv::Decoration::RestrictPointer:
    case spv::Decoration::AliasedPointer:
      if (target->opcode() != spv::Op::OpVariable &&
          target->opcode() != spv::Op::OpFunctionParameter) {
        return fail(0) << "must be a memory object declaration";
      }
      if (_.GetIdOpcode(target->type_id()) != spv::Op::OpTypePointer) {
        return fail(0) << "must be a pointer type";
      }
      break;
    case spv::Decoration::Invariant:
    case spv::Decoration::Constant:
    case spv::Decoration::Location:
    case spv::Decoration::Index:
    case spv::Decoration::Binding:
    case spv::Decoration::DescriptorSet:
    case spv::Decoration::InputAttachmentIndex:
      if (target->opcode() != spv::Op::OpVariable) {
        return fail(0) << "must be a variable";
      }
      break;
    default:
      break;
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    // The following were all checked as pointer types above.
    spv::StorageClass sc = spv::StorageClass::Uniform;
    const auto type = _.FindDef(target->type_id());
    if (type && type->operands().size() > 2) {
      sc = type->GetOperandAs<spv::StorageClass>(1);
    }
    switch (dec) {
      case spv::Decoration::Location:
      case spv::Decoration::Component:
        // Location is used for input, output and ray tracing stages.
        if (sc != spv::StorageClass::Input && sc != spv::StorageClass::Output &&
            sc != spv::StorageClass::RayPayloadKHR &&
            sc != spv::StorageClass::IncomingRayPayloadKHR &&
            sc != spv::StorageClass::HitAttributeKHR &&
            sc != spv::StorageClass::CallableDataKHR &&
            sc != spv::StorageClass::IncomingCallableDataKHR &&
            sc != spv::StorageClass::ShaderRecordBufferKHR &&
            sc != spv::StorageClass::HitObjectAttributeNV) {
          return _.diag(SPV_ERROR_INVALID_ID, target)
                 << _.VkErrorID(6672) << _.SpvDecorationString(dec)
                 << " decoration must not be applied to this storage class";
        }
        break;
      case spv::Decoration::Index:
        // Langauge from SPIR-V definition of Index
        if (sc != spv::StorageClass::Output) {
          return fail(0) << "must be in the Output storage class";
        }
        break;
      case spv::Decoration::Binding:
      case spv::Decoration::DescriptorSet:
        if (sc != spv::StorageClass::StorageBuffer &&
            sc != spv::StorageClass::Uniform &&
            sc != spv::StorageClass::UniformConstant) {
          return fail(6491) << "must be in the StorageBuffer, Uniform, or "
                               "UniformConstant storage class";
        }
        break;
      case spv::Decoration::InputAttachmentIndex:
        if (sc != spv::StorageClass::UniformConstant) {
          return fail(6678) << "must be in the UniformConstant storage class";
        }
        break;
      case spv::Decoration::Flat:
      case spv::Decoration::NoPerspective:
      case spv::Decoration::Centroid:
      case spv::Decoration::Sample:
        if (sc != spv::StorageClass::Input && sc != spv::StorageClass::Output) {
          return fail(4670) << "storage class must be Input or Output";
        }
        break;
      case spv::Decoration::PerVertexKHR:
        if (sc != spv::StorageClass::Input) {
          return fail(6777) << "storage class must be Input";
        }
        break;
      default:
        break;
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateDecorate(ValidationState_t& _, const Instruction* inst) {
  const auto decoration = inst->GetOperandAs<spv::Decoration>(1);
  const auto target_id = inst->GetOperandAs<uint32_t>(0);
  const auto target = _.FindDef(target_id);
  if (!target) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "target is not defined";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if ((decoration == spv::Decoration::GLSLShared) ||
        (decoration == spv::Decoration::GLSLPacked)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << _.VkErrorID(4669) << "OpDecorate decoration '"
             << _.SpvDecorationString(decoration)
             << "' is not valid for the Vulkan execution environment.";
    }
  }

  if (DecorationTakesIdParameters(decoration)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Decorations taking ID parameters may not be used with "
              "OpDecorateId";
  }

  if (target->opcode() != spv::Op::OpDecorationGroup) {
    if (IsMemberDecorationOnly(decoration)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << _.SpvDecorationString(decoration)
             << " can only be applied to structure members";
    }

    if (auto error = ValidateDecorationTarget(_, decoration, inst, target)) {
      return error;
    }
  }

  // TODO: Add validations for all decorations.
  return SPV_SUCCESS;
}

spv_result_t ValidateDecorateId(ValidationState_t& _, const Instruction* inst) {
  const auto decoration = inst->GetOperandAs<spv::Decoration>(1);
  if (!DecorationTakesIdParameters(decoration)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Decorations that don't take ID parameters may not be used with "
              "OpDecorateId";
  }

  // No member decorations take id parameters, so we don't bother checking if
  // we are using a member only decoration here.

  // TODO: Add validations for these decorations.
  // UniformId is covered elsewhere.
  return SPV_SUCCESS;
}

spv_result_t ValidateMemberDecorate(ValidationState_t& _,
                                    const Instruction* inst) {
  const auto struct_type_id = inst->GetOperandAs<uint32_t>(0);
  const auto struct_type = _.FindDef(struct_type_id);
  if (!struct_type || spv::Op::OpTypeStruct != struct_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpMemberDecorate Structure type <id> "
           << _.getIdName(struct_type_id) << " is not a struct type.";
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

  const auto decoration = inst->GetOperandAs<spv::Decoration>(2);
  if (IsNotMemberDecoration(decoration)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << _.SpvDecorationString(decoration)
           << " cannot be applied to structure members";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateDecorationGroup(ValidationState_t& _,
                                     const Instruction* inst) {
  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  const auto decoration_group = _.FindDef(decoration_group_id);
  for (auto pair : decoration_group->uses()) {
    auto use = pair.first;
    if (use->opcode() != spv::Op::OpDecorate &&
        use->opcode() != spv::Op::OpGroupDecorate &&
        use->opcode() != spv::Op::OpGroupMemberDecorate &&
        use->opcode() != spv::Op::OpName &&
        use->opcode() != spv::Op::OpDecorateId && !use->IsNonSemantic()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Result id of OpDecorationGroup can only "
             << "be targeted by OpName, OpGroupDecorate, "
             << "OpDecorate, OpDecorateId, and OpGroupMemberDecorate";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupDecorate(ValidationState_t& _,
                                   const Instruction* inst) {
  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  auto decoration_group = _.FindDef(decoration_group_id);
  if (!decoration_group ||
      spv::Op::OpDecorationGroup != decoration_group->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpGroupDecorate Decoration group <id> "
           << _.getIdName(decoration_group_id) << " is not a decoration group.";
  }
  for (unsigned i = 1; i < inst->operands().size(); ++i) {
    auto target_id = inst->GetOperandAs<uint32_t>(i);
    auto target = _.FindDef(target_id);
    if (!target || target->opcode() == spv::Op::OpDecorationGroup) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpGroupDecorate may not target OpDecorationGroup <id> "
             << _.getIdName(target_id);
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupMemberDecorate(ValidationState_t& _,
                                         const Instruction* inst) {
  const auto decoration_group_id = inst->GetOperandAs<uint32_t>(0);
  const auto decoration_group = _.FindDef(decoration_group_id);
  if (!decoration_group ||
      spv::Op::OpDecorationGroup != decoration_group->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpGroupMemberDecorate Decoration group <id> "
           << _.getIdName(decoration_group_id) << " is not a decoration group.";
  }
  // Grammar checks ensures that the number of arguments to this instruction
  // is an odd number: 1 decoration group + (id,literal) pairs.
  for (size_t i = 1; i + 1 < inst->operands().size(); i += 2) {
    const uint32_t struct_id = inst->GetOperandAs<uint32_t>(i);
    const uint32_t index = inst->GetOperandAs<uint32_t>(i + 1);
    auto struct_instr = _.FindDef(struct_id);
    if (!struct_instr || spv::Op::OpTypeStruct != struct_instr->opcode()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpGroupMemberDecorate Structure type <id> "
             << _.getIdName(struct_id) << " is not a struct type.";
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
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId: {
      const uint32_t target_id = inst->word(1);
      const spv::Decoration dec_type =
          static_cast<spv::Decoration>(inst->word(2));
      std::vector<uint32_t> dec_params;
      if (inst->words().size() > 3) {
        dec_params.insert(dec_params.end(), inst->words().begin() + 3,
                          inst->words().end());
      }
      _.RegisterDecorationForId(target_id, Decoration(dec_type, dec_params));
      break;
    }
    case spv::Op::OpMemberDecorate: {
      const uint32_t struct_id = inst->word(1);
      const uint32_t index = inst->word(2);
      const spv::Decoration dec_type =
          static_cast<spv::Decoration>(inst->word(3));
      std::vector<uint32_t> dec_params;
      if (inst->words().size() > 4) {
        dec_params.insert(dec_params.end(), inst->words().begin() + 4,
                          inst->words().end());
      }
      _.RegisterDecorationForId(struct_id,
                                Decoration(dec_type, dec_params, index));
      break;
    }
    case spv::Op::OpDecorationGroup: {
      // We don't need to do anything right now. Assigning decorations to groups
      // will be taken care of via OpGroupDecorate.
      break;
    }
    case spv::Op::OpGroupDecorate: {
      // Word 1 is the group <id>. All subsequent words are target <id>s that
      // are going to be decorated with the decorations.
      const uint32_t decoration_group_id = inst->word(1);
      std::set<Decoration>& group_decorations =
          _.id_decorations(decoration_group_id);
      for (size_t i = 2; i < inst->words().size(); ++i) {
        const uint32_t target_id = inst->word(i);
        _.RegisterDecorationsForId(target_id, group_decorations.begin(),
                                   group_decorations.end());
      }
      break;
    }
    case spv::Op::OpGroupMemberDecorate: {
      // Word 1 is the Decoration Group <id> followed by (struct<id>,literal)
      // pairs. All decorations of the group should be applied to all the struct
      // members that are specified in the instructions.
      const uint32_t decoration_group_id = inst->word(1);
      std::set<Decoration>& group_decorations =
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
    case spv::Op::OpDecorate:
      if (auto error = ValidateDecorate(_, inst)) return error;
      break;
    case spv::Op::OpDecorateId:
      if (auto error = ValidateDecorateId(_, inst)) return error;
      break;
    // TODO(dneto): spv::Op::OpDecorateStringGOOGLE
    // See https://github.com/KhronosGroup/SPIRV-Tools/issues/2253
    case spv::Op::OpMemberDecorate:
      if (auto error = ValidateMemberDecorate(_, inst)) return error;
      break;
    case spv::Op::OpDecorationGroup:
      if (auto error = ValidateDecorationGroup(_, inst)) return error;
      break;
    case spv::Op::OpGroupDecorate:
      if (auto error = ValidateGroupDecorate(_, inst)) return error;
      break;
    case spv::Op::OpGroupMemberDecorate:
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
