// Copyright (c) 2023 LunarG Inc.
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

#include "source/opt/switch_descriptorset_pass.h"

#include "source/opt/ir_builder.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {

Pass::Status SwitchDescriptorSetPass::Process() {
  Status status = Status::SuccessWithoutChange;
  auto* deco_mgr = context()->get_decoration_mgr();

  for (Instruction& var : context()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable) {
      continue;
    }
    auto decos = deco_mgr->GetDecorationsFor(var.result_id(), false);
    for (const auto& deco : decos) {
      spv::Decoration d = spv::Decoration(deco->GetSingleWordInOperand(1u));
      if (d == spv::Decoration::DescriptorSet &&
          deco->GetSingleWordInOperand(2u) == ds_from_) {
        deco->SetInOperand(2u, {ds_to_});
        status = Status::SuccessWithChange;
        break;
      }
    }
  }
  return status;
}

}  // namespace opt
}  // namespace spvtools
