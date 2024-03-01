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

#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateMemberName(ValidationState_t& _, const Instruction* inst) {
  const auto type_id = inst->GetOperandAs<uint32_t>(0);
  const auto type = _.FindDef(type_id);
  if (!type || spv::Op::OpTypeStruct != type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpMemberName Type <id> " << _.getIdName(type_id)
           << " is not a struct type.";
  }
  const auto member_id = inst->GetOperandAs<uint32_t>(1);
  const auto member_count = (uint32_t)(type->words().size() - 2);
  if (member_count <= member_id) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpMemberName Member <id> " << _.getIdName(member_id)
           << " index is larger than Type <id> " << _.getIdName(type->id())
           << "s member count.";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateLine(ValidationState_t& _, const Instruction* inst) {
  const auto file_id = inst->GetOperandAs<uint32_t>(0);
  const auto file = _.FindDef(file_id);
  if (!file || spv::Op::OpString != file->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpLine Target <id> " << _.getIdName(file_id)
           << " is not an OpString.";
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t DebugPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpMemberName:
      if (auto error = ValidateMemberName(_, inst)) return error;
      break;
    case spv::Op::OpLine:
      if (auto error = ValidateLine(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
