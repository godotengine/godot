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

#include "source/fuzz/fuzzer_pass_adjust_loop_controls.h"

#include "source/fuzz/transformation_set_loop_control.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustLoopControls::FuzzerPassAdjustLoopControls(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAdjustLoopControls::Apply() {
  // Consider every merge instruction in the module (via looking through all
  // functions and blocks).
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (auto merge_inst = block.GetMergeInst()) {
        // Ignore the instruction if it is not a loop merge.
        if (merge_inst->opcode() != spv::Op::OpLoopMerge) {
          continue;
        }

        // Decide randomly whether to adjust this loop merge.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAdjustingLoopControl())) {
          continue;
        }

        uint32_t existing_mask = merge_inst->GetSingleWordOperand(
            TransformationSetLoopControl::kLoopControlMaskInOperandIndex);

        // First, set the new mask to one of None, Unroll or DontUnroll.
        std::vector<uint32_t> basic_masks = {
            uint32_t(spv::LoopControlMask::MaskNone),
            uint32_t(spv::LoopControlMask::Unroll),
            uint32_t(spv::LoopControlMask::DontUnroll)};
        uint32_t new_mask =
            basic_masks[GetFuzzerContext()->RandomIndex(basic_masks)];

        // For the loop controls that depend on guarantees about what the loop
        // does, check which of these were present in the existing mask and
        // randomly decide whether to keep them.  They are just hints, so
        // removing them should not change the semantics of the module.
        for (auto mask_bit : {spv::LoopControlMask::DependencyInfinite,
                              spv::LoopControlMask::DependencyLength,
                              spv::LoopControlMask::MinIterations,
                              spv::LoopControlMask::MaxIterations,
                              spv::LoopControlMask::IterationMultiple}) {
          if ((existing_mask & uint32_t(mask_bit)) &&
              GetFuzzerContext()->ChooseEven()) {
            // The mask bits we are considering are not available in all SPIR-V
            // versions.  However, we only include a mask bit if it was present
            // in the original loop control mask, and we work under the
            // assumption that we are transforming a valid module, thus we don't
            // need to actually check whether the SPIR-V version being used
            // supports these loop control mask bits.
            new_mask |= uint32_t(mask_bit);
          }
        }

        // We use 0 for peel count and partial count in the case that we choose
        // not to set these controls.
        uint32_t peel_count = 0;
        uint32_t partial_count = 0;

        // PeelCount and PartialCount are not compatible with DontUnroll, so
        // we check whether DontUnroll is set.
        if (!(new_mask & uint32_t(spv::LoopControlMask::DontUnroll))) {
          // If PeelCount is supported by this SPIR-V version, randomly choose
          // whether to set it.  If it was set in the original mask and is not
          // selected for setting here, that amounts to dropping it.
          if (TransformationSetLoopControl::PeelCountIsSupported(
                  GetIRContext()) &&
              GetFuzzerContext()->ChooseEven()) {
            new_mask |= uint32_t(spv::LoopControlMask::PeelCount);
            // The peel count is chosen randomly - if PeelCount was already set
            // this will overwrite whatever peel count was previously used.
            peel_count = GetFuzzerContext()->GetRandomLoopControlPeelCount();
          }
          // Similar, but for PartialCount.
          if (TransformationSetLoopControl::PartialCountIsSupported(
                  GetIRContext()) &&
              GetFuzzerContext()->ChooseEven()) {
            new_mask |= uint32_t(spv::LoopControlMask::PartialCount);
            partial_count =
                GetFuzzerContext()->GetRandomLoopControlPartialCount();
          }
        }

        // Apply the transformation and add it to the output transformation
        // sequence.
        TransformationSetLoopControl transformation(block.id(), new_mask,
                                                    peel_count, partial_count);
        ApplyTransformation(transformation);
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
