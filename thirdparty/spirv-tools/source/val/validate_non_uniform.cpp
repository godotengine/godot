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

// Validates correctness of barrier SPIR-V instructions.

#include "source/val/validate.h"

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateGroupNonUniformBallotBitCount(ValidationState_t& _,
                                                   const Instruction* inst) {
  // Scope is already checked by ValidateExecutionScope() above.

  const uint32_t result_type = inst->type_id();
  if (!_.IsUnsignedIntScalarType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be an unsigned integer type scalar.";
  }

  const auto value = inst->GetOperandAs<uint32_t>(4);
  const auto value_type = _.FindDef(value)->type_id();
  if (!_.IsUnsignedIntVectorType(value_type) ||
      _.GetDimension(value_type) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Expected Value to be a "
                                                   "vector of four components "
                                                   "of integer type scalar";
  }

  const auto group = inst->GetOperandAs<spv::GroupOperation>(3);
  if (spvIsVulkanEnv(_.context()->target_env)) {
    if ((group != spv::GroupOperation::Reduce) &&
        (group != spv::GroupOperation::InclusiveScan) &&
        (group != spv::GroupOperation::ExclusiveScan)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4685)
             << "In Vulkan: The OpGroupNonUniformBallotBitCount group "
                "operation must be only: Reduce, InclusiveScan, or "
                "ExclusiveScan.";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformRotateKHR(ValidationState_t& _,
                                              const Instruction* inst) {
  // Scope is already checked by ValidateExecutionScope() above.
  const uint32_t result_type = inst->type_id();
  if (!_.IsIntScalarOrVectorType(result_type) &&
      !_.IsFloatScalarOrVectorType(result_type) &&
      !_.IsBoolScalarOrVectorType(result_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Expected Result Type to be a scalar or vector of "
              "floating-point, integer or boolean type.";
  }

  const uint32_t value_type = _.GetTypeId(inst->GetOperandAs<uint32_t>(3));
  if (value_type != result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result Type must be the same as the type of Value.";
  }

  const uint32_t delta_type = _.GetTypeId(inst->GetOperandAs<uint32_t>(4));
  if (!_.IsUnsignedIntScalarType(delta_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Delta must be a scalar of integer type, whose Signedness "
              "operand is 0.";
  }

  if (inst->words().size() > 6) {
    const uint32_t cluster_size_op_id = inst->GetOperandAs<uint32_t>(5);
    const uint32_t cluster_size_type = _.GetTypeId(cluster_size_op_id);
    if (!_.IsUnsignedIntScalarType(cluster_size_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ClusterSize must be a scalar of integer type, whose "
                "Signedness operand is 0.";
    }

    uint64_t cluster_size;
    if (!_.GetConstantValUint64(cluster_size_op_id, &cluster_size)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ClusterSize must come from a constant instruction.";
    }

    if ((cluster_size == 0) || ((cluster_size & (cluster_size - 1)) != 0)) {
      return _.diag(SPV_WARNING, inst)
             << "Behavior is undefined unless ClusterSize is at least 1 and a "
                "power of 2.";
    }

    // TODO(kpet) Warn about undefined behavior when ClusterSize is greater than
    // the declared SubGroupSize
  }

  return SPV_SUCCESS;
}

}  // namespace

// Validates correctness of non-uniform group instructions.
spv_result_t NonUniformPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();

  if (spvOpcodeIsNonUniformGroupOperation(opcode)) {
    const uint32_t execution_scope = inst->word(3);
    if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
      return error;
    }
  }

  switch (opcode) {
    case spv::Op::OpGroupNonUniformBallotBitCount:
      return ValidateGroupNonUniformBallotBitCount(_, inst);
    case spv::Op::OpGroupNonUniformRotateKHR:
      return ValidateGroupNonUniformRotateKHR(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
