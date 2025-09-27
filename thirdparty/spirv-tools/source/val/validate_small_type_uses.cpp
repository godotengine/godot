// Copyright (c) 2019 Google LLC.
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

#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ValidateSmallTypeUses(ValidationState_t& _,
                                   const Instruction* inst) {
  if (!_.HasCapability(spv::Capability::Shader) || inst->type_id() == 0 ||
      !_.ContainsLimitedUseIntOrFloatType(inst->type_id())) {
    return SPV_SUCCESS;
  }

  if (_.IsPointerType(inst->type_id())) return SPV_SUCCESS;

  // The validator should previously have checked ways to generate 8- or 16-bit
  // types. So we only need to considervalid paths from source to sink.
  // When restricted, uses of 8- or 16-bit types can only be stores,
  // width-only conversions, decorations and copy object.
  for (auto use : inst->uses()) {
    const auto* user = use.first;
    switch (user->opcode()) {
      case spv::Op::OpDecorate:
      case spv::Op::OpDecorateId:
      case spv::Op::OpCopyObject:
      case spv::Op::OpStore:
      case spv::Op::OpFConvert:
      case spv::Op::OpUConvert:
      case spv::Op::OpSConvert:
        break;
      default:
        return _.diag(SPV_ERROR_INVALID_ID, user)
               << "Invalid use of 8- or 16-bit result";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
