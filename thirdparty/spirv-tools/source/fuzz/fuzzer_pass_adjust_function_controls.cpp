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

#include "source/fuzz/fuzzer_pass_adjust_function_controls.h"

#include "source/fuzz/transformation_set_function_control.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustFunctionControls::FuzzerPassAdjustFunctionControls(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAdjustFunctionControls::Apply() {
  // Consider every function in the module.
  for (auto& function : *GetIRContext()->module()) {
    // Randomly decide whether to adjust this function's controls.
    if (GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAdjustingFunctionControl())) {
      // Grab the function control mask for the function in its present form.
      uint32_t existing_function_control_mask =
          function.DefInst().GetSingleWordInOperand(0);

      // For the new mask, we first randomly select one of three basic masks:
      // None, Inline or DontInline.  These are always valid (and are mutually
      // exclusive).
      std::vector<spv::FunctionControlMask> basic_function_control_masks = {
          spv::FunctionControlMask::MaskNone, spv::FunctionControlMask::Inline,
          spv::FunctionControlMask::DontInline};
      uint32_t new_function_control_mask =
          uint32_t(basic_function_control_masks[GetFuzzerContext()->RandomIndex(
              basic_function_control_masks)]);

      // We now consider the Pure and Const mask bits.  If these are already
      // set on the function then it's OK to keep them, but also interesting
      // to consider dropping them, so we decide randomly in each case.
      for (auto mask_bit :
           {spv::FunctionControlMask::Pure, spv::FunctionControlMask::Const}) {
        if ((existing_function_control_mask & uint32_t(mask_bit)) &&
            GetFuzzerContext()->ChooseEven()) {
          new_function_control_mask |= uint32_t(mask_bit);
        }
      }

      // Create and add a transformation.
      TransformationSetFunctionControl transformation(
          function.DefInst().result_id(), new_function_control_mask);
      ApplyTransformation(transformation);
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
