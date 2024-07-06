// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_set_function_control.h"

namespace spvtools {
namespace fuzz {

TransformationSetFunctionControl::TransformationSetFunctionControl(
    protobufs::TransformationSetFunctionControl message)
    : message_(std::move(message)) {}

TransformationSetFunctionControl::TransformationSetFunctionControl(
    uint32_t function_id, uint32_t function_control) {
  message_.set_function_id(function_id);
  message_.set_function_control(function_control);
}

bool TransformationSetFunctionControl::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  opt::Instruction* function_def_instruction =
      FindFunctionDefInstruction(ir_context);
  if (!function_def_instruction) {
    // The given function id does not correspond to any function.
    return false;
  }
  uint32_t existing_function_control_mask =
      function_def_instruction->GetSingleWordInOperand(0);

  // Check (via an assertion) that function control mask doesn't have any bad
  // bits set.
  uint32_t acceptable_function_control_bits = uint32_t(
      spv::FunctionControlMask::Inline | spv::FunctionControlMask::DontInline |
      spv::FunctionControlMask::Pure | spv::FunctionControlMask::Const);
  // The following is to keep release-mode compilers happy as this variable is
  // only used in an assertion.
  (void)(acceptable_function_control_bits);
  assert(!(message_.function_control() & ~acceptable_function_control_bits) &&
         "Nonsensical loop control bits were found.");

  // Check (via an assertion) that function control mask does not have both
  // Inline and DontInline bits set.
  assert(!((message_.function_control() &
            (uint32_t)spv::FunctionControlMask::Inline) &&
           (message_.function_control() &
            (uint32_t)spv::FunctionControlMask::DontInline)) &&
         "It is not OK to set both the 'Inline' and 'DontInline' bits of a "
         "function control mask");

  // Check that Const and Pure are only present if they were present on the
  // original function
  for (auto mask_bit :
       {spv::FunctionControlMask::Pure, spv::FunctionControlMask::Const}) {
    if ((message_.function_control() & uint32_t(mask_bit)) &&
        !(existing_function_control_mask & uint32_t(mask_bit))) {
      return false;
    }
  }

  return true;
}

void TransformationSetFunctionControl::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  opt::Instruction* function_def_instruction =
      FindFunctionDefInstruction(ir_context);
  function_def_instruction->SetInOperand(0, {message_.function_control()});
}

protobufs::Transformation TransformationSetFunctionControl::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_set_function_control() = message_;
  return result;
}

opt::Instruction* TransformationSetFunctionControl ::FindFunctionDefInstruction(
    opt::IRContext* ir_context) const {
  // Look through all functions for a function whose defining instruction's
  // result id matches |message_.function_id|, returning the defining
  // instruction if found.
  for (auto& function : *ir_context->module()) {
    if (function.DefInst().result_id() == message_.function_id()) {
      return &function.DefInst();
    }
  }
  // A nullptr result indicates that no match was found.
  return nullptr;
}

std::unordered_set<uint32_t> TransformationSetFunctionControl::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
