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
  return SPV_SUCCESS;
}

}  // namespace

// Validates correctness of non-uniform group instructions.
spv_result_t NonUniformPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();

  if (spvOpcodeIsNonUniformGroupOperation(opcode)) {
    const uint32_t execution_scope = inst->word(3);
    if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
      return error;
    }
  }

  switch (opcode) {
    case SpvOpGroupNonUniformBallotBitCount:
      return ValidateGroupNonUniformBallotBitCount(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
