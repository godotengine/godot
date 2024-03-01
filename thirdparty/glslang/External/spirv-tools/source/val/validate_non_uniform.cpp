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

#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateGroupNonUniformElect(ValidationState_t& _,
                                          const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformAnyAll(ValidationState_t& _,
                                           const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar type";
  }

  if (!_.IsBoolScalarType(_.GetOperandTypeId(inst, 3))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Predicate must be a boolean scalar type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformAllEqual(ValidationState_t& _,
                                             const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar type";
  }

  const auto value_type = _.GetOperandTypeId(inst, 3);
  if (!_.IsFloatScalarOrVectorType(value_type) &&
      !_.IsIntScalarOrVectorType(value_type) &&
      !_.IsBoolScalarOrVectorType(value_type)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a scalar or vector of integer, floating-point, or "
              "boolean type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformBroadcastShuffle(ValidationState_t& _,
                                                     const Instruction* inst) {
  const auto type_id = inst->type_id();
  if (!_.IsFloatScalarOrVectorType(type_id) &&
      !_.IsIntScalarOrVectorType(type_id) &&
      !_.IsBoolScalarOrVectorType(type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a scalar or vector of integer, floating-point, "
              "or boolean type";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 3);
  if (value_type_id != type_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of Value must match the Result type";
  }

  const auto GetOperandName = [](const spv::Op opcode) {
    std::string operand;
    switch (opcode) {
      case spv::Op::OpGroupNonUniformBroadcast:
      case spv::Op::OpGroupNonUniformShuffle:
        operand = "Id";
        break;
      case spv::Op::OpGroupNonUniformShuffleXor:
        operand = "Mask";
        break;
      case spv::Op::OpGroupNonUniformQuadBroadcast:
        operand = "Index";
        break;
      case spv::Op::OpGroupNonUniformQuadSwap:
        operand = "Direction";
        break;
      case spv::Op::OpGroupNonUniformShuffleUp:
      case spv::Op::OpGroupNonUniformShuffleDown:
      default:
        operand = "Delta";
        break;
    }
    return operand;
  };

  const auto id_type_id = _.GetOperandTypeId(inst, 4);
  if (!_.IsUnsignedIntScalarType(id_type_id)) {
    std::string operand = GetOperandName(inst->opcode());
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << operand << " must be an unsigned integer scalar";
  }

  const bool should_be_constant =
      inst->opcode() == spv::Op::OpGroupNonUniformQuadSwap ||
      ((inst->opcode() == spv::Op::OpGroupNonUniformBroadcast ||
        inst->opcode() == spv::Op::OpGroupNonUniformQuadBroadcast) &&
       _.version() < SPV_SPIRV_VERSION_WORD(1, 5));
  if (should_be_constant) {
    const auto id_id = inst->GetOperandAs<uint32_t>(4);
    const auto id_op = _.GetIdOpcode(id_id);
    if (!spvOpcodeIsConstant(id_op)) {
      std::string operand = GetOperandName(inst->opcode());
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Before SPIR-V 1.5, " << operand
             << " must be a constant instruction";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformBroadcastFirst(ValidationState_t& _,
                                                   const Instruction* inst) {
  const auto type_id = inst->type_id();
  if (!_.IsFloatScalarOrVectorType(type_id) &&
      !_.IsIntScalarOrVectorType(type_id) &&
      !_.IsBoolScalarOrVectorType(type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a scalar or vector of integer, floating-point, "
              "or boolean type";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 3);
  if (value_type_id != type_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of Value must match the Result type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformBallot(ValidationState_t& _,
                                           const Instruction* inst) {
  if (!_.IsUnsignedIntVectorType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a 4-component unsigned integer vector";
  }

  if (_.GetDimension(inst->type_id()) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a 4-component unsigned integer vector";
  }

  const auto pred_type_id = _.GetOperandTypeId(inst, 3);
  if (!_.IsBoolScalarType(pred_type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Predicate must be a boolean scalar";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformInverseBallot(ValidationState_t& _,
                                                  const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 3);
  if (!_.IsUnsignedIntVectorType(value_type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  if (_.GetDimension(value_type_id) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformBallotBitExtract(ValidationState_t& _,
                                                     const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 3);
  if (!_.IsUnsignedIntVectorType(value_type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  if (_.GetDimension(value_type_id) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  const auto id_type_id = _.GetOperandTypeId(inst, 4);
  if (!_.IsUnsignedIntScalarType(id_type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Id must be an unsigned integer scalar";
  }

  return SPV_SUCCESS;
}

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

spv_result_t ValidateGroupNonUniformBallotFind(ValidationState_t& _,
                                               const Instruction* inst) {
  if (!_.IsUnsignedIntScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be an unsigned integer scalar";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 3);
  if (!_.IsUnsignedIntVectorType(value_type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  if (_.GetDimension(value_type_id) != 4) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Value must be a 4-component unsigned integer vector";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupNonUniformArithmetic(ValidationState_t& _,
                                               const Instruction* inst) {
  const bool is_unsigned = inst->opcode() == spv::Op::OpGroupNonUniformUMin ||
                           inst->opcode() == spv::Op::OpGroupNonUniformUMax;
  const bool is_float = inst->opcode() == spv::Op::OpGroupNonUniformFAdd ||
                        inst->opcode() == spv::Op::OpGroupNonUniformFMul ||
                        inst->opcode() == spv::Op::OpGroupNonUniformFMin ||
                        inst->opcode() == spv::Op::OpGroupNonUniformFMax;
  const bool is_bool = inst->opcode() == spv::Op::OpGroupNonUniformLogicalAnd ||
                       inst->opcode() == spv::Op::OpGroupNonUniformLogicalOr ||
                       inst->opcode() == spv::Op::OpGroupNonUniformLogicalXor;
  if (is_float) {
    if (!_.IsFloatScalarOrVectorType(inst->type_id())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Result must be a floating-point scalar or vector";
    }
  } else if (is_bool) {
    if (!_.IsBoolScalarOrVectorType(inst->type_id())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Result must be a boolean scalar or vector";
    }
  } else if (is_unsigned) {
    if (!_.IsUnsignedIntScalarOrVectorType(inst->type_id())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Result must be an unsigned integer scalar or vector";
    }
  } else if (!_.IsIntScalarOrVectorType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be an integer scalar or vector";
  }

  const auto value_type_id = _.GetOperandTypeId(inst, 4);
  if (value_type_id != inst->type_id()) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of Value must match the Result type";
  }

  const auto group_op = inst->GetOperandAs<spv::GroupOperation>(3);
  bool is_clustered_reduce = group_op == spv::GroupOperation::ClusteredReduce;
  bool is_partitioned_nv =
      group_op == spv::GroupOperation::PartitionedReduceNV ||
      group_op == spv::GroupOperation::PartitionedInclusiveScanNV ||
      group_op == spv::GroupOperation::PartitionedExclusiveScanNV;
  if (inst->operands().size() <= 5) {
    if (is_clustered_reduce) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ClusterSize must be present when Operation is ClusteredReduce";
    } else if (is_partitioned_nv) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Ballot must be present when Operation is PartitionedReduceNV, "
                "PartitionedInclusiveScanNV, or PartitionedExclusiveScanNV";
    }
  } else {
    const auto operand_id = inst->GetOperandAs<uint32_t>(5);
    const auto* operand = _.FindDef(operand_id);
    if (is_partitioned_nv) {
      if (!operand || !_.IsIntScalarOrVectorType(operand->type_id())) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ballot must be a 4-component integer vector";
      }

      if (_.GetDimension(operand->type_id()) != 4) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Ballot must be a 4-component integer vector";
      }
    } else {
      if (!operand || !_.IsUnsignedIntScalarType(operand->type_id())) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "ClusterSize must be an unsigned integer scalar";
      }

      if (!spvOpcodeIsConstant(operand->opcode())) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "ClusterSize must be a constant instruction";
      }
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
    const Instruction* cluster_size_inst = _.FindDef(cluster_size_op_id);
    const uint32_t cluster_size_type =
        cluster_size_inst ? cluster_size_inst->type_id() : 0;
    if (!_.IsUnsignedIntScalarType(cluster_size_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ClusterSize must be a scalar of integer type, whose "
                "Signedness operand is 0.";
    }

    if (!spvOpcodeIsConstant(cluster_size_inst->opcode())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "ClusterSize must come from a constant instruction.";
    }

    uint64_t cluster_size;
    const bool valid_const =
        _.EvalConstantValUint64(cluster_size_op_id, &cluster_size);
    if (valid_const &&
        ((cluster_size == 0) || ((cluster_size & (cluster_size - 1)) != 0))) {
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
    // OpGroupNonUniformQuadAllKHR and OpGroupNonUniformQuadAnyKHR don't have
    // scope paramter
    if ((opcode != spv::Op::OpGroupNonUniformQuadAllKHR) &&
        (opcode != spv::Op::OpGroupNonUniformQuadAnyKHR)) {
      const uint32_t execution_scope = inst->GetOperandAs<uint32_t>(2);
      if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
        return error;
      }
    }
  }

  switch (opcode) {
    case spv::Op::OpGroupNonUniformElect:
      return ValidateGroupNonUniformElect(_, inst);
    case spv::Op::OpGroupNonUniformAny:
    case spv::Op::OpGroupNonUniformAll:
      return ValidateGroupNonUniformAnyAll(_, inst);
    case spv::Op::OpGroupNonUniformAllEqual:
      return ValidateGroupNonUniformAllEqual(_, inst);
    case spv::Op::OpGroupNonUniformBroadcast:
    case spv::Op::OpGroupNonUniformShuffle:
    case spv::Op::OpGroupNonUniformShuffleXor:
    case spv::Op::OpGroupNonUniformShuffleUp:
    case spv::Op::OpGroupNonUniformShuffleDown:
    case spv::Op::OpGroupNonUniformQuadBroadcast:
    case spv::Op::OpGroupNonUniformQuadSwap:
      return ValidateGroupNonUniformBroadcastShuffle(_, inst);
    case spv::Op::OpGroupNonUniformBroadcastFirst:
      return ValidateGroupNonUniformBroadcastFirst(_, inst);
    case spv::Op::OpGroupNonUniformBallot:
      return ValidateGroupNonUniformBallot(_, inst);
    case spv::Op::OpGroupNonUniformInverseBallot:
      return ValidateGroupNonUniformInverseBallot(_, inst);
    case spv::Op::OpGroupNonUniformBallotBitExtract:
      return ValidateGroupNonUniformBallotBitExtract(_, inst);
    case spv::Op::OpGroupNonUniformBallotBitCount:
      return ValidateGroupNonUniformBallotBitCount(_, inst);
    case spv::Op::OpGroupNonUniformBallotFindLSB:
    case spv::Op::OpGroupNonUniformBallotFindMSB:
      return ValidateGroupNonUniformBallotFind(_, inst);
    case spv::Op::OpGroupNonUniformIAdd:
    case spv::Op::OpGroupNonUniformFAdd:
    case spv::Op::OpGroupNonUniformIMul:
    case spv::Op::OpGroupNonUniformFMul:
    case spv::Op::OpGroupNonUniformSMin:
    case spv::Op::OpGroupNonUniformUMin:
    case spv::Op::OpGroupNonUniformFMin:
    case spv::Op::OpGroupNonUniformSMax:
    case spv::Op::OpGroupNonUniformUMax:
    case spv::Op::OpGroupNonUniformFMax:
    case spv::Op::OpGroupNonUniformBitwiseAnd:
    case spv::Op::OpGroupNonUniformBitwiseOr:
    case spv::Op::OpGroupNonUniformBitwiseXor:
    case spv::Op::OpGroupNonUniformLogicalAnd:
    case spv::Op::OpGroupNonUniformLogicalOr:
    case spv::Op::OpGroupNonUniformLogicalXor:
      return ValidateGroupNonUniformArithmetic(_, inst);
    case spv::Op::OpGroupNonUniformRotateKHR:
      return ValidateGroupNonUniformRotateKHR(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
