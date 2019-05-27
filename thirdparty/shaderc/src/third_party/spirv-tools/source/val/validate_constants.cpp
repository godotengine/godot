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
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateConstantBool(ValidationState_t& _,
                                  const Instruction* inst) {
  auto type = _.FindDef(inst->type_id());
  if (!type || type->opcode() != SpvOpTypeBool) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Op" << spvOpcodeString(inst->opcode()) << " Result Type <id> '"
           << _.getIdName(inst->type_id()) << "' is not a boolean type.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateConstantComposite(ValidationState_t& _,
                                       const Instruction* inst) {
  std::string opcode_name = std::string("Op") + spvOpcodeString(inst->opcode());

  const auto result_type = _.FindDef(inst->type_id());
  if (!result_type || !spvOpcodeIsComposite(result_type->opcode())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opcode_name << " Result Type <id> '"
           << _.getIdName(inst->type_id()) << "' is not a composite type.";
  }

  const auto constituent_count = inst->words().size() - 3;
  switch (result_type->opcode()) {
    case SpvOpTypeVector: {
      const auto component_count = result_type->GetOperandAs<uint32_t>(2);
      if (component_count != constituent_count) {
        // TODO: Output ID's on diagnostic
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name
               << " Constituent <id> count does not match "
                  "Result Type <id> '"
               << _.getIdName(result_type->id())
               << "'s vector component count.";
      }
      const auto component_type =
          _.FindDef(result_type->GetOperandAs<uint32_t>(1));
      if (!component_type) {
        return _.diag(SPV_ERROR_INVALID_ID, result_type)
               << "Component type is not defined.";
      }
      for (size_t constituent_index = 2;
           constituent_index < inst->operands().size(); constituent_index++) {
        const auto constituent_id =
            inst->GetOperandAs<uint32_t>(constituent_index);
        const auto constituent = _.FindDef(constituent_id);
        if (!constituent ||
            !spvOpcodeIsConstantOrUndef(constituent->opcode())) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' is not a constant or undef.";
        }
        const auto constituent_result_type = _.FindDef(constituent->type_id());
        if (!constituent_result_type ||
            component_type->opcode() != constituent_result_type->opcode()) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "'s type does not match Result Type <id> '"
                 << _.getIdName(result_type->id()) << "'s vector element type.";
        }
      }
    } break;
    case SpvOpTypeMatrix: {
      const auto column_count = result_type->GetOperandAs<uint32_t>(2);
      if (column_count != constituent_count) {
        // TODO: Output ID's on diagnostic
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name
               << " Constituent <id> count does not match "
                  "Result Type <id> '"
               << _.getIdName(result_type->id()) << "'s matrix column count.";
      }

      const auto column_type = _.FindDef(result_type->words()[2]);
      if (!column_type) {
        return _.diag(SPV_ERROR_INVALID_ID, result_type)
               << "Column type is not defined.";
      }
      const auto component_count = column_type->GetOperandAs<uint32_t>(2);
      const auto component_type =
          _.FindDef(column_type->GetOperandAs<uint32_t>(1));
      if (!component_type) {
        return _.diag(SPV_ERROR_INVALID_ID, column_type)
               << "Component type is not defined.";
      }

      for (size_t constituent_index = 2;
           constituent_index < inst->operands().size(); constituent_index++) {
        const auto constituent_id =
            inst->GetOperandAs<uint32_t>(constituent_index);
        const auto constituent = _.FindDef(constituent_id);
        if (!constituent ||
            !(SpvOpConstantComposite == constituent->opcode() ||
              SpvOpSpecConstantComposite == constituent->opcode() ||
              SpvOpUndef == constituent->opcode())) {
          // The message says "... or undef" because the spec does not say
          // undef is a constant.
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' is not a constant composite or undef.";
        }
        const auto vector = _.FindDef(constituent->type_id());
        if (!vector) {
          return _.diag(SPV_ERROR_INVALID_ID, constituent)
                 << "Result type is not defined.";
        }
        if (column_type->opcode() != vector->opcode()) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' type does not match Result Type <id> '"
                 << _.getIdName(result_type->id()) << "'s matrix column type.";
        }
        const auto vector_component_type =
            _.FindDef(vector->GetOperandAs<uint32_t>(1));
        if (component_type->id() != vector_component_type->id()) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' component type does not match Result Type <id> '"
                 << _.getIdName(result_type->id())
                 << "'s matrix column component type.";
        }
        if (component_count != vector->words()[3]) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' vector component count does not match Result Type <id> '"
                 << _.getIdName(result_type->id())
                 << "'s vector component count.";
        }
      }
    } break;
    case SpvOpTypeArray: {
      auto element_type = _.FindDef(result_type->GetOperandAs<uint32_t>(1));
      if (!element_type) {
        return _.diag(SPV_ERROR_INVALID_ID, result_type)
               << "Element type is not defined.";
      }
      const auto length = _.FindDef(result_type->GetOperandAs<uint32_t>(2));
      if (!length) {
        return _.diag(SPV_ERROR_INVALID_ID, result_type)
               << "Length is not defined.";
      }
      bool is_int32;
      bool is_const;
      uint32_t value;
      std::tie(is_int32, is_const, value) = _.EvalInt32IfConst(length->id());
      if (is_int32 && is_const && value != constituent_count) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name
               << " Constituent count does not match "
                  "Result Type <id> '"
               << _.getIdName(result_type->id()) << "'s array length.";
      }
      for (size_t constituent_index = 2;
           constituent_index < inst->operands().size(); constituent_index++) {
        const auto constituent_id =
            inst->GetOperandAs<uint32_t>(constituent_index);
        const auto constituent = _.FindDef(constituent_id);
        if (!constituent ||
            !spvOpcodeIsConstantOrUndef(constituent->opcode())) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' is not a constant or undef.";
        }
        const auto constituent_type = _.FindDef(constituent->type_id());
        if (!constituent_type) {
          return _.diag(SPV_ERROR_INVALID_ID, constituent)
                 << "Result type is not defined.";
        }
        if (element_type->id() != constituent_type->id()) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "'s type does not match Result Type <id> '"
                 << _.getIdName(result_type->id()) << "'s array element type.";
        }
      }
    } break;
    case SpvOpTypeStruct: {
      const auto member_count = result_type->words().size() - 2;
      if (member_count != constituent_count) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name << " Constituent <id> '"
               << _.getIdName(inst->type_id())
               << "' count does not match Result Type <id> '"
               << _.getIdName(result_type->id()) << "'s struct member count.";
      }
      for (uint32_t constituent_index = 2, member_index = 1;
           constituent_index < inst->operands().size();
           constituent_index++, member_index++) {
        const auto constituent_id =
            inst->GetOperandAs<uint32_t>(constituent_index);
        const auto constituent = _.FindDef(constituent_id);
        if (!constituent ||
            !spvOpcodeIsConstantOrUndef(constituent->opcode())) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' is not a constant or undef.";
        }
        const auto constituent_type = _.FindDef(constituent->type_id());
        if (!constituent_type) {
          return _.diag(SPV_ERROR_INVALID_ID, constituent)
                 << "Result type is not defined.";
        }

        const auto member_type_id =
            result_type->GetOperandAs<uint32_t>(member_index);
        const auto member_type = _.FindDef(member_type_id);
        if (!member_type || member_type->id() != constituent_type->id()) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << opcode_name << " Constituent <id> '"
                 << _.getIdName(constituent_id)
                 << "' type does not match the Result Type <id> '"
                 << _.getIdName(result_type->id()) << "'s member type.";
        }
      }
    } break;
    case SpvOpTypeCooperativeMatrixNV: {
      if (1 != constituent_count) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name << " Constituent <id> '"
               << _.getIdName(inst->type_id()) << "' count must be one.";
      }
      const auto constituent_id = inst->GetOperandAs<uint32_t>(2);
      const auto constituent = _.FindDef(constituent_id);
      if (!constituent || !spvOpcodeIsConstantOrUndef(constituent->opcode())) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name << " Constituent <id> '"
               << _.getIdName(constituent_id)
               << "' is not a constant or undef.";
      }
      const auto constituent_type = _.FindDef(constituent->type_id());
      if (!constituent_type) {
        return _.diag(SPV_ERROR_INVALID_ID, constituent)
               << "Result type is not defined.";
      }

      const auto component_type_id = result_type->GetOperandAs<uint32_t>(1);
      const auto component_type = _.FindDef(component_type_id);
      if (!component_type || component_type->id() != constituent_type->id()) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << opcode_name << " Constituent <id> '"
               << _.getIdName(constituent_id)
               << "' type does not match the Result Type <id> '"
               << _.getIdName(result_type->id()) << "'s component type.";
      }
    } break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateConstantSampler(ValidationState_t& _,
                                     const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  if (!result_type || result_type->opcode() != SpvOpTypeSampler) {
    return _.diag(SPV_ERROR_INVALID_ID, result_type)
           << "OpConstantSampler Result Type <id> '"
           << _.getIdName(inst->type_id()) << "' is not a sampler type.";
  }

  return SPV_SUCCESS;
}

// True if instruction defines a type that can have a null value, as defined by
// the SPIR-V spec.  Tracks composite-type components through module to check
// nullability transitively.
bool IsTypeNullable(const std::vector<uint32_t>& instruction,
                    const ValidationState_t& _) {
  uint16_t opcode;
  uint16_t word_count;
  spvOpcodeSplit(instruction[0], &word_count, &opcode);
  switch (static_cast<SpvOp>(opcode)) {
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypePointer:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
      return true;
    case SpvOpTypeArray:
    case SpvOpTypeMatrix:
    case SpvOpTypeCooperativeMatrixNV:
    case SpvOpTypeVector: {
      auto base_type = _.FindDef(instruction[2]);
      return base_type && IsTypeNullable(base_type->words(), _);
    }
    case SpvOpTypeStruct: {
      for (size_t elementIndex = 2; elementIndex < instruction.size();
           ++elementIndex) {
        auto element = _.FindDef(instruction[elementIndex]);
        if (!element || !IsTypeNullable(element->words(), _)) return false;
      }
      return true;
    }
    default:
      return false;
  }
}

spv_result_t ValidateConstantNull(ValidationState_t& _,
                                  const Instruction* inst) {
  const auto result_type = _.FindDef(inst->type_id());
  if (!result_type || !IsTypeNullable(result_type->words(), _)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpConstantNull Result Type <id> '"
           << _.getIdName(inst->type_id()) << "' cannot have a null value.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateSpecConstantOp(ValidationState_t& _,
                                    const Instruction* inst) {
  const auto op = inst->GetOperandAs<SpvOp>(2);

  // The binary parser already ensures that the op is valid for *some*
  // environment.  Here we check restrictions.
  switch (op) {
    case SpvOpQuantizeToF16:
      if (!_.HasCapability(SpvCapabilityShader)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Specialization constant operation " << spvOpcodeString(op)
               << " requires Shader capability";
      }
      break;

    case SpvOpUConvert:
      if (!_.features().uconvert_spec_constant_op &&
          !_.HasCapability(SpvCapabilityKernel)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "UConvert requires Kernel capability or extension "
                  "SPV_AMD_gpu_shader_int16";
      }
      break;

    case SpvOpConvertFToS:
    case SpvOpConvertSToF:
    case SpvOpConvertFToU:
    case SpvOpConvertUToF:
    case SpvOpConvertPtrToU:
    case SpvOpConvertUToPtr:
    case SpvOpGenericCastToPtr:
    case SpvOpPtrCastToGeneric:
    case SpvOpBitcast:
    case SpvOpFNegate:
    case SpvOpFAdd:
    case SpvOpFSub:
    case SpvOpFMul:
    case SpvOpFDiv:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpPtrAccessChain:
    case SpvOpInBoundsPtrAccessChain:
      if (!_.HasCapability(SpvCapabilityKernel)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "Specialization constant operation " << spvOpcodeString(op)
               << " requires Kernel capability";
      }
      break;

    default:
      break;
  }

  // TODO(dneto): Validate result type and arguments to the various operations.
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ConstantPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpConstantTrue:
    case SpvOpConstantFalse:
    case SpvOpSpecConstantTrue:
    case SpvOpSpecConstantFalse:
      if (auto error = ValidateConstantBool(_, inst)) return error;
      break;
    case SpvOpConstantComposite:
    case SpvOpSpecConstantComposite:
      if (auto error = ValidateConstantComposite(_, inst)) return error;
      break;
    case SpvOpConstantSampler:
      if (auto error = ValidateConstantSampler(_, inst)) return error;
      break;
    case SpvOpConstantNull:
      if (auto error = ValidateConstantNull(_, inst)) return error;
      break;
    case SpvOpSpecConstantOp:
      if (auto error = ValidateSpecConstantOp(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
