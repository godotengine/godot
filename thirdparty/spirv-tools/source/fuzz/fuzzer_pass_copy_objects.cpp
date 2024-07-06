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

#include "source/fuzz/fuzzer_pass_copy_objects.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_add_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassCopyObjects::FuzzerPassCopyObjects(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassCopyObjects::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(
            inst_it->opcode() ==
                spv::Op(instruction_descriptor.target_instruction_opcode()) &&
            "The opcode of the instruction we might insert before must be "
            "the same as the opcode in the descriptor for the instruction");

        if (GetTransformationContext()->GetFactManager()->BlockIsDead(
                block->id())) {
          // Don't create synonyms in dead blocks.
          return;
        }

        // Check whether it is legitimate to insert a copy before this
        // instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpCopyObject,
                                                          inst_it)) {
          return;
        }

        // Randomly decide whether to try inserting an object copy here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfCopyingObject())) {
          return;
        }

        const auto relevant_instructions = FindAvailableInstructions(
            function, block, inst_it,
            [this](opt::IRContext* ir_context, opt::Instruction* inst) {
              return TransformationAddSynonym::IsInstructionValid(
                  ir_context, *GetTransformationContext(), inst,
                  protobufs::TransformationAddSynonym::COPY_OBJECT);
            });

        // At this point, |relevant_instructions| contains all the instructions
        // we might think of copying.
        if (relevant_instructions.empty()) {
          return;
        }

        // Choose a copyable instruction at random, and create and apply an
        // object copying transformation based on it.
        ApplyTransformation(TransformationAddSynonym(
            relevant_instructions[GetFuzzerContext()->RandomIndex(
                                      relevant_instructions)]
                ->result_id(),
            protobufs::TransformationAddSynonym::COPY_OBJECT,
            GetFuzzerContext()->GetFreshId(), instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
