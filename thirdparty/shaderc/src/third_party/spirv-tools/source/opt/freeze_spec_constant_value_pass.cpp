// Copyright (c) 2016 Google Inc.
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

#include "source/opt/freeze_spec_constant_value_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status FreezeSpecConstantValuePass::Process() {
  bool modified = false;
  auto ctx = context();
  ctx->module()->ForEachInst([&modified, ctx](Instruction* inst) {
    switch (inst->opcode()) {
      case SpvOp::SpvOpSpecConstant:
        inst->SetOpcode(SpvOp::SpvOpConstant);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantTrue:
        inst->SetOpcode(SpvOp::SpvOpConstantTrue);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantFalse:
        inst->SetOpcode(SpvOp::SpvOpConstantFalse);
        modified = true;
        break;
      case SpvOp::SpvOpDecorate:
        if (inst->GetSingleWordInOperand(1) ==
            SpvDecoration::SpvDecorationSpecId) {
          ctx->KillInst(inst);
          modified = true;
        }
        break;
      default:
        break;
    }
  });
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
