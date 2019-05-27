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

#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if |a| and |b| are instruction defining pointers that point to
// the same type.
bool ArePointersToSameType(val::Instruction* a, val::Instruction* b) {
  if (a->opcode() != SpvOpTypePointer || b->opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t a_type = a->GetOperandAs<uint32_t>(2);
  return a_type && (a_type == b->GetOperandAs<uint32_t>(2));
}

spv_result_t ValidateFunction(ValidationState_t& _, const Instruction* inst) {
  const auto function_type_id = inst->GetOperandAs<uint32_t>(3);
  const auto function_type = _.FindDef(function_type_id);
  if (!function_type || SpvOpTypeFunction != function_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunction Function Type <id> '" << _.getIdName(function_type_id)
           << "' is not a function type.";
  }

  const auto return_id = function_type->GetOperandAs<uint32_t>(1);
  if (return_id != inst->type_id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunction Result Type <id> '" << _.getIdName(inst->type_id())
           << "' does not match the Function Type's return type <id> '"
           << _.getIdName(return_id) << "'.";
  }

  const std::vector<SpvOp> acceptable = {
      SpvOpDecorate,
      SpvOpEnqueueKernel,
      SpvOpEntryPoint,
      SpvOpExecutionMode,
      SpvOpExecutionModeId,
      SpvOpFunctionCall,
      SpvOpGetKernelNDrangeSubGroupCount,
      SpvOpGetKernelNDrangeMaxSubGroupSize,
      SpvOpGetKernelWorkGroupSize,
      SpvOpGetKernelPreferredWorkGroupSizeMultiple,
      SpvOpGetKernelLocalSizeForSubgroupCount,
      SpvOpGetKernelMaxNumSubgroups,
      SpvOpName};
  for (auto& pair : inst->uses()) {
    const auto* use = pair.first;
    if (std::find(acceptable.begin(), acceptable.end(), use->opcode()) ==
        acceptable.end()) {
      return _.diag(SPV_ERROR_INVALID_ID, use)
             << "Invalid use of function result id " << _.getIdName(inst->id())
             << ".";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateFunctionParameter(ValidationState_t& _,
                                       const Instruction* inst) {
  // NOTE: Find OpFunction & ensure OpFunctionParameter is not out of place.
  size_t param_index = 0;
  size_t inst_num = inst->LineNum() - 1;
  if (inst_num == 0) {
    return _.diag(SPV_ERROR_INVALID_LAYOUT, inst)
           << "Function parameter cannot be the first instruction.";
  }

  auto func_inst = &_.ordered_instructions()[inst_num];
  while (--inst_num) {
    func_inst = &_.ordered_instructions()[inst_num];
    if (func_inst->opcode() == SpvOpFunction) {
      break;
    } else if (func_inst->opcode() == SpvOpFunctionParameter) {
      ++param_index;
    }
  }

  if (func_inst->opcode() != SpvOpFunction) {
    return _.diag(SPV_ERROR_INVALID_LAYOUT, inst)
           << "Function parameter must be preceded by a function.";
  }

  const auto function_type_id = func_inst->GetOperandAs<uint32_t>(3);
  const auto function_type = _.FindDef(function_type_id);
  if (!function_type) {
    return _.diag(SPV_ERROR_INVALID_ID, func_inst)
           << "Missing function type definition.";
  }
  if (param_index >= function_type->words().size() - 3) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Too many OpFunctionParameters for " << func_inst->id()
           << ": expected " << function_type->words().size() - 3
           << " based on the function's type";
  }

  const auto param_type =
      _.FindDef(function_type->GetOperandAs<uint32_t>(param_index + 2));
  if (!param_type || inst->type_id() != param_type->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunctionParameter Result Type <id> '"
           << _.getIdName(inst->type_id())
           << "' does not match the OpTypeFunction parameter "
              "type of the same index.";
  }

  // Validate that PhysicalStorageBufferEXT have one of Restrict, Aliased,
  // RestrictPointerEXT, or AliasedPointerEXT.
  auto param_nonarray_type_id = param_type->id();
  while (_.GetIdOpcode(param_nonarray_type_id) == SpvOpTypeArray) {
    param_nonarray_type_id =
        _.FindDef(param_nonarray_type_id)->GetOperandAs<uint32_t>(1u);
  }
  if (_.GetIdOpcode(param_nonarray_type_id) == SpvOpTypePointer) {
    auto param_nonarray_type = _.FindDef(param_nonarray_type_id);
    if (param_nonarray_type->GetOperandAs<uint32_t>(1u) ==
        SpvStorageClassPhysicalStorageBufferEXT) {
      // check for Aliased or Restrict
      const auto& decorations = _.id_decorations(inst->id());

      bool foundAliased = std::any_of(
          decorations.begin(), decorations.end(), [](const Decoration& d) {
            return SpvDecorationAliased == d.dec_type();
          });

      bool foundRestrict = std::any_of(
          decorations.begin(), decorations.end(), [](const Decoration& d) {
            return SpvDecorationRestrict == d.dec_type();
          });

      if (!foundAliased && !foundRestrict) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpFunctionParameter " << inst->id()
               << ": expected Aliased or Restrict for PhysicalStorageBufferEXT "
                  "pointer.";
      }
      if (foundAliased && foundRestrict) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpFunctionParameter " << inst->id()
               << ": can't specify both Aliased and Restrict for "
                  "PhysicalStorageBufferEXT pointer.";
      }
    } else {
      const auto pointee_type_id =
          param_nonarray_type->GetOperandAs<uint32_t>(2);
      const auto pointee_type = _.FindDef(pointee_type_id);
      if (SpvOpTypePointer == pointee_type->opcode() &&
          pointee_type->GetOperandAs<uint32_t>(1u) ==
              SpvStorageClassPhysicalStorageBufferEXT) {
        // check for AliasedPointerEXT/RestrictPointerEXT
        const auto& decorations = _.id_decorations(inst->id());

        bool foundAliased = std::any_of(
            decorations.begin(), decorations.end(), [](const Decoration& d) {
              return SpvDecorationAliasedPointerEXT == d.dec_type();
            });

        bool foundRestrict = std::any_of(
            decorations.begin(), decorations.end(), [](const Decoration& d) {
              return SpvDecorationRestrictPointerEXT == d.dec_type();
            });

        if (!foundAliased && !foundRestrict) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "OpFunctionParameter " << inst->id()
                 << ": expected AliasedPointerEXT or RestrictPointerEXT for "
                    "PhysicalStorageBufferEXT pointer.";
        }
        if (foundAliased && foundRestrict) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "OpFunctionParameter " << inst->id()
                 << ": can't specify both AliasedPointerEXT and "
                    "RestrictPointerEXT for PhysicalStorageBufferEXT pointer.";
        }
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateFunctionCall(ValidationState_t& _,
                                  const Instruction* inst) {
  const auto function_id = inst->GetOperandAs<uint32_t>(2);
  const auto function = _.FindDef(function_id);
  if (!function || SpvOpFunction != function->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunctionCall Function <id> '" << _.getIdName(function_id)
           << "' is not a function.";
  }

  auto return_type = _.FindDef(function->type_id());
  if (!return_type || return_type->id() != inst->type_id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunctionCall Result Type <id> '"
           << _.getIdName(inst->type_id())
           << "'s type does not match Function <id> '"
           << _.getIdName(return_type->id()) << "'s return type.";
  }

  const auto function_type_id = function->GetOperandAs<uint32_t>(3);
  const auto function_type = _.FindDef(function_type_id);
  if (!function_type || function_type->opcode() != SpvOpTypeFunction) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Missing function type definition.";
  }

  const auto function_call_arg_count = inst->words().size() - 4;
  const auto function_param_count = function_type->words().size() - 3;
  if (function_param_count != function_call_arg_count) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpFunctionCall Function <id>'s parameter count does not match "
              "the argument count.";
  }

  for (size_t argument_index = 3, param_index = 2;
       argument_index < inst->operands().size();
       argument_index++, param_index++) {
    const auto argument_id = inst->GetOperandAs<uint32_t>(argument_index);
    const auto argument = _.FindDef(argument_id);
    if (!argument) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Missing argument " << argument_index - 3 << " definition.";
    }

    const auto argument_type = _.FindDef(argument->type_id());
    if (!argument_type) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Missing argument " << argument_index - 3
             << " type definition.";
    }

    const auto parameter_type_id =
        function_type->GetOperandAs<uint32_t>(param_index);
    const auto parameter_type = _.FindDef(parameter_type_id);
    if (!parameter_type ||
        (argument_type->id() != parameter_type->id() &&
         !(_.options()->relax_logical_pointer &&
           ArePointersToSameType(argument_type, parameter_type)))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpFunctionCall Argument <id> '" << _.getIdName(argument_id)
             << "'s type does not match Function <id> '"
             << _.getIdName(parameter_type_id) << "'s parameter type.";
    }

    if (_.addressing_model() == SpvAddressingModelLogical) {
      if (parameter_type->opcode() == SpvOpTypePointer &&
          !_.options()->relax_logical_pointer) {
        SpvStorageClass sc = parameter_type->GetOperandAs<SpvStorageClass>(1u);
        // Validate which storage classes can be pointer operands.
        switch (sc) {
          case SpvStorageClassUniformConstant:
          case SpvStorageClassFunction:
          case SpvStorageClassPrivate:
          case SpvStorageClassWorkgroup:
          case SpvStorageClassAtomicCounter:
            // These are always allowed.
            break;
          case SpvStorageClassStorageBuffer:
            if (!_.features().variable_pointers_storage_buffer) {
              return _.diag(SPV_ERROR_INVALID_ID, inst)
                     << "StorageBuffer pointer operand "
                     << _.getIdName(argument_id)
                     << " requires a variable pointers capability";
            }
            break;
          default:
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "Invalid storage class for pointer operand "
                   << _.getIdName(argument_id);
        }

        // Validate memory object declaration requirements.
        if (argument->opcode() != SpvOpVariable &&
            argument->opcode() != SpvOpFunctionParameter) {
          const bool ssbo_vptr =
              _.features().variable_pointers_storage_buffer &&
              sc == SpvStorageClassStorageBuffer;
          const bool wg_vptr =
              _.features().variable_pointers && sc == SpvStorageClassWorkgroup;
          const bool uc_ptr = sc == SpvStorageClassUniformConstant;
          if (!ssbo_vptr && !wg_vptr && !uc_ptr) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "Pointer operand " << _.getIdName(argument_id)
                   << " must be a memory object declaration";
          }
        }
      }
    }
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t FunctionPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpFunction:
      if (auto error = ValidateFunction(_, inst)) return error;
      break;
    case SpvOpFunctionParameter:
      if (auto error = ValidateFunctionParameter(_, inst)) return error;
      break;
    case SpvOpFunctionCall:
      if (auto error = ValidateFunctionCall(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
