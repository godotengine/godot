// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/fuzzer_pass_replace_opselects_with_conditional_branches.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceOpSelectsWithConditionalBranches::
    FuzzerPassReplaceOpSelectsWithConditionalBranches(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceOpSelectsWithConditionalBranches::Apply() {
  // Keep track of the instructions that we want to replace. We need to collect
  // them in a vector, since it's not safe to modify the module while iterating
  // over it.
  std::vector<uint32_t> replaceable_opselect_instruction_ids;

  // Loop over all the instructions in the module.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We cannot split loop headers, so we don't need to consider instructions
      // in loop headers that are also merge blocks (since they would need to be
      // split).
      if (block.IsLoopHeader() &&
          GetIRContext()->GetStructuredCFGAnalysis()->IsMergeBlock(
              block.id())) {
        continue;
      }

      for (auto& instruction : block) {
        // We only care about OpSelect instructions.
        if (instruction.opcode() != spv::Op::OpSelect) {
          continue;
        }

        // Randomly choose whether to consider this instruction for replacement.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingOpselectWithConditionalBranch())) {
          continue;
        }

        // If the selector does not have scalar boolean type (i.e., it is a
        // boolean vector) then ignore this OpSelect.
        if (GetIRContext()
                ->get_def_use_mgr()
                ->GetDef(fuzzerutil::GetTypeId(
                    GetIRContext(), instruction.GetSingleWordInOperand(0)))
                ->opcode() != spv::Op::OpTypeBool) {
          continue;
        }

        // If the block is a loop header and we need to split it, the
        // transformation cannot be applied because loop headers cannot be
        // split. We can break out of this loop because the transformation can
        // only be applied to at most the first instruction in a loop header.
        if (block.IsLoopHeader() && InstructionNeedsSplitBefore(&instruction)) {
          break;
        }

        // If the instruction separates an OpSampledImage from its use, the
        // block cannot be split around it and the instruction cannot be
        // replaced.
        if (fuzzerutil::
                SplittingBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
                    &block, &instruction)) {
          continue;
        }

        // We can apply the transformation to this instruction.
        replaceable_opselect_instruction_ids.push_back(instruction.result_id());
      }
    }
  }

  // Apply the transformations, splitting the blocks containing the
  // instructions, if necessary.
  for (uint32_t instruction_id : replaceable_opselect_instruction_ids) {
    auto instruction =
        GetIRContext()->get_def_use_mgr()->GetDef(instruction_id);

    // If the instruction requires the block containing it to be split before
    // it, split the block.
    if (InstructionNeedsSplitBefore(instruction)) {
      ApplyTransformation(TransformationSplitBlock(
          MakeInstructionDescriptor(GetIRContext(), instruction),
          GetFuzzerContext()->GetFreshId()));
    }

    // Decide whether to have two branches or just one.
    bool two_branches = GetFuzzerContext()->ChoosePercentage(
        GetFuzzerContext()
            ->GetChanceOfAddingBothBranchesWhenReplacingOpSelect());

    // If there will be only one branch, decide whether it will be the true
    // branch or the false branch.
    bool true_branch_id_zero =
        !two_branches &&
        GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfAddingTrueBranchWhenReplacingOpSelect());
    bool false_branch_id_zero = !two_branches && !true_branch_id_zero;

    uint32_t true_branch_id =
        true_branch_id_zero ? 0 : GetFuzzerContext()->GetFreshId();
    uint32_t false_branch_id =
        false_branch_id_zero ? 0 : GetFuzzerContext()->GetFreshId();

    ApplyTransformation(TransformationReplaceOpSelectWithConditionalBranch(
        instruction_id, true_branch_id, false_branch_id));
  }
}

bool FuzzerPassReplaceOpSelectsWithConditionalBranches::
    InstructionNeedsSplitBefore(opt::Instruction* instruction) {
  assert(instruction && instruction->opcode() == spv::Op::OpSelect &&
         "The instruction must be OpSelect.");

  auto block = GetIRContext()->get_instr_block(instruction);
  assert(block && "The instruction must be contained in a block.");

  // We need to split the block if the instruction is not the first in its
  // block.
  if (instruction->unique_id() != block->begin()->unique_id()) {
    return true;
  }

  // We need to split the block if it is a merge block.
  if (GetIRContext()->GetStructuredCFGAnalysis()->IsMergeBlock(block->id())) {
    return true;
  }

  // We need to split the block if it has more than one predecessor.
  if (GetIRContext()->cfg()->preds(block->id()).size() != 1) {
    return true;
  }

  // We need to split the block if its predecessor is a header or it does not
  // branch unconditionally to the block.
  auto predecessor = GetIRContext()->get_instr_block(
      GetIRContext()->cfg()->preds(block->id())[0]);
  return predecessor->MergeBlockIdIfAny() ||
         predecessor->terminator()->opcode() != spv::Op::OpBranch;
}

}  // namespace fuzz
}  // namespace spvtools
