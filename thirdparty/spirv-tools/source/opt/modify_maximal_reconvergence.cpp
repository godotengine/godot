// Copyright (c) 2024 Google LLC
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

#include "modify_maximal_reconvergence.h"

#include "source/opt/ir_context.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status ModifyMaximalReconvergence::Process() {
  bool changed = false;
  if (add_) {
    changed = AddMaximalReconvergence();
  } else {
    changed = RemoveMaximalReconvergence();
  }
  return changed ? Pass::Status::SuccessWithChange
                 : Pass::Status::SuccessWithoutChange;
}

bool ModifyMaximalReconvergence::AddMaximalReconvergence() {
  bool changed = false;
  bool has_extension = false;
  bool has_shader =
      context()->get_feature_mgr()->HasCapability(spv::Capability::Shader);
  for (auto extension : context()->extensions()) {
    if (extension.GetOperand(0).AsString() == "SPV_KHR_maximal_reconvergence") {
      has_extension = true;
      break;
    }
  }

  std::unordered_set<uint32_t> entry_points_with_mode;
  for (auto mode : get_module()->execution_modes()) {
    if (spv::ExecutionMode(mode.GetSingleWordInOperand(1)) ==
        spv::ExecutionMode::MaximallyReconvergesKHR) {
      entry_points_with_mode.insert(mode.GetSingleWordInOperand(0));
    }
  }

  for (auto entry_point : get_module()->entry_points()) {
    const uint32_t id = entry_point.GetSingleWordInOperand(1);
    if (!entry_points_with_mode.count(id)) {
      changed = true;
      if (!has_extension) {
        context()->AddExtension("SPV_KHR_maximal_reconvergence");
        has_extension = true;
      }
      if (!has_shader) {
        context()->AddCapability(spv::Capability::Shader);
        has_shader = true;
      }
      context()->AddExecutionMode(MakeUnique<Instruction>(
          context(), spv::Op::OpExecutionMode, 0, 0,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {id}},
              {SPV_OPERAND_TYPE_EXECUTION_MODE,
               {static_cast<uint32_t>(
                   spv::ExecutionMode::MaximallyReconvergesKHR)}}}));
      entry_points_with_mode.insert(id);
    }
  }

  return changed;
}

bool ModifyMaximalReconvergence::RemoveMaximalReconvergence() {
  bool changed = false;
  std::vector<Instruction*> to_remove;
  Instruction* mode = &*get_module()->execution_mode_begin();
  while (mode) {
    if (mode->opcode() != spv::Op::OpExecutionMode &&
        mode->opcode() != spv::Op::OpExecutionModeId) {
      break;
    }
    if (spv::ExecutionMode(mode->GetSingleWordInOperand(1)) ==
        spv::ExecutionMode::MaximallyReconvergesKHR) {
      mode = context()->KillInst(mode);
      changed = true;
    } else {
      mode = mode->NextNode();
    }
  }

  changed |=
      context()->RemoveExtension(Extension::kSPV_KHR_maximal_reconvergence);
  return changed;
}
}  // namespace opt
}  // namespace spvtools
