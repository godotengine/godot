// Copyright (c) 2018 Google LLC.
// Copyright (c) 2019 NVIDIA Corporation
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
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateUndef(ValidationState_t& _, const Instruction* inst) {
  if (_.IsVoidType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Cannot create undefined values with void type";
  }
  if (_.HasCapability(spv::Capability::Shader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id()) &&
      !_.IsPointerType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Cannot create undefined values with 8- or 16-bit types";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateShaderClock(ValidationState_t& _,
                                 const Instruction* inst) {
  const uint32_t scope = inst->GetOperandAs<uint32_t>(2);
  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);
  if (is_const_int32 && spv::Scope(value) != spv::Scope::Subgroup &&
      spv::Scope(value) != spv::Scope::Device) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(4652) << "Scope must be Subgroup or Device";
  }

  // Result Type must be a 64 - bit unsigned integer type or
  // a vector of two - components of 32 -
  // bit unsigned integer type
  const uint32_t result_type = inst->type_id();
  if (!_.IsUnsigned64BitHandle(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Expected Value to be a "
                                                   "vector of two components"
                                                   " of unsigned integer"
                                                   " or 64bit unsigned integer";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateAssumeTrue(ValidationState_t& _, const Instruction* inst) {
  const auto operand_type_id = _.GetOperandTypeId(inst, 0);
  if (!operand_type_id || !_.IsBoolScalarType(operand_type_id)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Value operand of OpAssumeTrueKHR must be a boolean scalar";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateExpect(ValidationState_t& _, const Instruction* inst) {
  const auto result_type = inst->type_id();
  if (!_.IsBoolScalarOrVectorType(result_type) &&
      !_.IsIntScalarOrVectorType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Result of OpExpectKHR must be a scalar or vector of integer "
              "type or boolean type";
  }

  if (_.GetOperandTypeId(inst, 2) != result_type) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Type of Value operand of OpExpectKHR does not match the result "
              "type ";
  }
  if (_.GetOperandTypeId(inst, 3) != result_type) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Type of ExpectedValue operand of OpExpectKHR does not match the "
              "result type ";
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t MiscPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpUndef:
      if (auto error = ValidateUndef(_, inst)) return error;
      break;
    default:
      break;
  }
  switch (inst->opcode()) {
    case spv::Op::OpBeginInvocationInterlockEXT:
    case spv::Op::OpEndInvocationInterlockEXT:
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              spv::ExecutionModel::Fragment,
              "OpBeginInvocationInterlockEXT/OpEndInvocationInterlockEXT "
              "require Fragment execution model");

      _.function(inst->function()->id())
          ->RegisterLimitation([](const ValidationState_t& state,
                                  const Function* entry_point,
                                  std::string* message) {
            const auto* execution_modes =
                state.GetExecutionModes(entry_point->id());

            auto find_interlock = [](const spv::ExecutionMode& mode) {
              switch (mode) {
                case spv::ExecutionMode::PixelInterlockOrderedEXT:
                case spv::ExecutionMode::PixelInterlockUnorderedEXT:
                case spv::ExecutionMode::SampleInterlockOrderedEXT:
                case spv::ExecutionMode::SampleInterlockUnorderedEXT:
                case spv::ExecutionMode::ShadingRateInterlockOrderedEXT:
                case spv::ExecutionMode::ShadingRateInterlockUnorderedEXT:
                  return true;
                default:
                  return false;
              }
            };

            bool found = false;
            if (execution_modes) {
              auto i = std::find_if(execution_modes->begin(),
                                    execution_modes->end(), find_interlock);
              found = (i != execution_modes->end());
            }

            if (!found) {
              *message =
                  "OpBeginInvocationInterlockEXT/OpEndInvocationInterlockEXT "
                  "require a fragment shader interlock execution mode.";
              return false;
            }
            return true;
          });
      break;
    case spv::Op::OpDemoteToHelperInvocationEXT:
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              spv::ExecutionModel::Fragment,
              "OpDemoteToHelperInvocationEXT requires Fragment execution "
              "model");
      break;
    case spv::Op::OpIsHelperInvocationEXT: {
      const uint32_t result_type = inst->type_id();
      _.function(inst->function()->id())
          ->RegisterExecutionModelLimitation(
              spv::ExecutionModel::Fragment,
              "OpIsHelperInvocationEXT requires Fragment execution model");
      if (!_.IsBoolScalarType(result_type))
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Expected bool scalar type as Result Type: "
               << spvOpcodeString(inst->opcode());
      break;
    }
    case spv::Op::OpReadClockKHR:
      if (auto error = ValidateShaderClock(_, inst)) {
        return error;
      }
      break;
    case spv::Op::OpAssumeTrueKHR:
      if (auto error = ValidateAssumeTrue(_, inst)) {
        return error;
      }
      break;
    case spv::Op::OpExpectKHR:
      if (auto error = ValidateExpect(_, inst)) {
        return error;
      }
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
