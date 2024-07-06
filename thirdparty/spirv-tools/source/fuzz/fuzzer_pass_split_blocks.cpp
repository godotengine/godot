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

#include "source/fuzz/fuzzer_pass_split_blocks.h"

#include <vector>

#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSplitBlocks::FuzzerPassSplitBlocks(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassSplitBlocks::Apply() {
  // Gather up pointers to all the blocks in the module.  We are then able to
  // iterate over these pointers and split the blocks to which they point;
  // we cannot safely split blocks while we iterate through the module.
  std::vector<opt::BasicBlock*> blocks;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      blocks.push_back(&block);
    }
  }

  // Now go through all the block pointers that were gathered.
  for (auto& block : blocks) {
    // Probabilistically decide whether to try to split this block.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfSplittingBlock())) {
      // We are not going to try to split this block.
      continue;
    }

    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2964): consider
    //  taking a simpler approach to identifying the instruction before which
    //  to split a block.

    // We are going to try to split this block.  We now need to choose where
    // to split it.  We describe the instruction before which we would like to
    // split a block via an InstructionDescriptor, details of which are
    // commented in the protobufs definition file.
    std::vector<protobufs::InstructionDescriptor> instruction_descriptors;

    // The initial base instruction is the block label.
    uint32_t base = block->id();

    // Counts the number of times we have seen each opcode since we reset the
    // base instruction.
    std::map<spv::Op, uint32_t> skip_count;

    // Consider every instruction in the block.  The label is excluded: it is
    // only necessary to consider it as a base in case the first instruction
    // in the block does not have a result id.
    for (auto& inst : *block) {
      if (inst.HasResultId()) {
        // In the case that the instruction has a result id, we use the
        // instruction as its own base, and clear the skip counts we have
        // collected.
        base = inst.result_id();
        skip_count.clear();
      }
      const spv::Op opcode = inst.opcode();
      instruction_descriptors.emplace_back(MakeInstructionDescriptor(
          base, opcode, skip_count.count(opcode) ? skip_count.at(opcode) : 0));
      if (!inst.HasResultId()) {
        skip_count[opcode] =
            skip_count.count(opcode) ? skip_count.at(opcode) + 1 : 1;
      }
    }
    // Having identified all the places we might be able to split the block,
    // we choose one of them.
    auto transformation = TransformationSplitBlock(
        instruction_descriptors[GetFuzzerContext()->RandomIndex(
            instruction_descriptors)],
        GetFuzzerContext()->GetFreshId());
    // If the position we have chosen turns out to be a valid place to split
    // the block, we apply the split. Otherwise the block just doesn't get
    // split.
    MaybeApplyTransformation(transformation);
  }
}

}  // namespace fuzz
}  // namespace spvtools
