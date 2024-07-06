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

#include "source/fuzz/fuzzer_pass_adjust_memory_operands_masks.h"

#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_set_memory_operands_mask.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustMemoryOperandsMasks::FuzzerPassAdjustMemoryOperandsMasks(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAdjustMemoryOperandsMasks::Apply() {
  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Consider every instruction in this block, using an explicit iterator so
      // that when we find an instruction of interest we can search backwards to
      // create an id descriptor for it.
      for (auto inst_it = block.cbegin(); inst_it != block.cend(); ++inst_it) {
        if (!TransformationSetMemoryOperandsMask::IsMemoryAccess(*inst_it)) {
          // We are only interested in memory access instructions.
          continue;
        }

        std::vector<uint32_t> indices_of_available_masks_to_adjust;
        // All memory instructions have at least one memory operands mask.
        indices_of_available_masks_to_adjust.push_back(0);
        // From SPIR-V 1.4 onwards, OpCopyMemory and OpCopyMemorySized have a
        // second mask.
        switch (inst_it->opcode()) {
          case spv::Op::OpCopyMemory:
          case spv::Op::OpCopyMemorySized:
            if (TransformationSetMemoryOperandsMask::
                    MultipleMemoryOperandMasksAreSupported(GetIRContext())) {
              indices_of_available_masks_to_adjust.push_back(1);
            }
            break;
          default:
            break;
        }

        // Consider the available masks
        for (auto mask_index : indices_of_available_masks_to_adjust) {
          // Randomly decide whether to adjust this mask.
          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()
                      ->GetChanceOfAdjustingMemoryOperandsMask())) {
            continue;
          }
          // Get the existing mask, using None if there was no mask present at
          // all.
          auto existing_mask_in_operand_index =
              TransformationSetMemoryOperandsMask::GetInOperandIndexForMask(
                  *inst_it, mask_index);
          auto existing_mask =
              existing_mask_in_operand_index < inst_it->NumInOperands()
                  ? inst_it->GetSingleWordInOperand(
                        existing_mask_in_operand_index)
                  : static_cast<uint32_t>(spv::MemoryAccessMask::MaskNone);

          // There are two things we can do to a mask:
          // - add Volatile if not already present
          // - toggle Nontemporal
          // The following ensures that we do at least one of these
          bool add_volatile =
              !(existing_mask & uint32_t(spv::MemoryAccessMask::Volatile)) &&
              GetFuzzerContext()->ChooseEven();
          bool toggle_nontemporal =
              !add_volatile || GetFuzzerContext()->ChooseEven();

          // These bitwise operations use '|' to add Volatile if desired, and
          // '^' to toggle Nontemporal if desired.
          uint32_t new_mask =
              (existing_mask |
               (add_volatile ? uint32_t(spv::MemoryAccessMask::Volatile)
                             : uint32_t(spv::MemoryAccessMask::MaskNone))) ^
              (toggle_nontemporal ? uint32_t(spv::MemoryAccessMask::Nontemporal)
                                  : uint32_t(spv::MemoryAccessMask::MaskNone));

          TransformationSetMemoryOperandsMask transformation(
              MakeInstructionDescriptor(block, inst_it), new_mask, mask_index);
          ApplyTransformation(transformation);
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
