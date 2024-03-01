// Copyright (c) 2022 Google LLC
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

#include "source/opt/remove_dontinline_pass.h"

namespace spvtools {
namespace opt {

Pass::Status RemoveDontInline::Process() {
  bool modified = false;
  modified = ClearDontInlineFunctionControl();
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool RemoveDontInline::ClearDontInlineFunctionControl() {
  bool modified = false;
  for (auto& func : *get_module()) {
    ClearDontInlineFunctionControl(&func);
  }
  return modified;
}

bool RemoveDontInline::ClearDontInlineFunctionControl(Function* function) {
  constexpr uint32_t kFunctionControlInOperandIdx = 0;
  Instruction* function_inst = &function->DefInst();
  uint32_t function_control =
      function_inst->GetSingleWordInOperand(kFunctionControlInOperandIdx);

  if ((function_control & uint32_t(spv::FunctionControlMask::DontInline)) ==
      0) {
    return false;
  }
  function_control &= ~uint32_t(spv::FunctionControlMask::DontInline);
  function_inst->SetInOperand(kFunctionControlInOperandIdx, {function_control});
  return true;
}

}  // namespace opt
}  // namespace spvtools
