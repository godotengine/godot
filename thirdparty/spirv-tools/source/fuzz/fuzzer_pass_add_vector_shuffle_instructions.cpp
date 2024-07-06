// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_add_vector_shuffle_instructions.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_vector_shuffle.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddVectorShuffleInstructions::FuzzerPassAddVectorShuffleInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddVectorShuffleInstructions::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(
            instruction_iterator->opcode() ==
                spv::Op(instruction_descriptor.target_instruction_opcode()) &&
            "The opcode of the instruction we might insert before must be "
            "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to try adding an OpVectorShuffle instruction.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingVectorShuffle())) {
          return;
        }

        // It must be valid to insert an OpVectorShuffle instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpVectorShuffle, instruction_iterator)) {
          return;
        }

        // Looks for vectors that we might consider to use as OpVectorShuffle
        // operands.
        std::vector<opt::Instruction*> vector_instructions =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }

                  if (!ir_context->get_type_mgr()
                           ->GetType(instruction->type_id())
                           ->AsVector()) {
                    return false;
                  }

                  if (!GetTransformationContext()
                           ->GetFactManager()
                           ->IdIsIrrelevant(instruction->result_id()) &&
                      !fuzzerutil::CanMakeSynonymOf(ir_context,
                                                    *GetTransformationContext(),
                                                    *instruction)) {
                    // If the id is irrelevant, we can use it since it will not
                    // participate in DataSynonym fact. Otherwise, we should be
                    // able to produce a synonym out of the id.
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // If there are no vector instructions, then return.
        if (vector_instructions.empty()) {
          return;
        }

        auto vector_1_instruction =
            vector_instructions[GetFuzzerContext()->RandomIndex(
                vector_instructions)];
        auto vector_1_type = GetIRContext()
                                 ->get_type_mgr()
                                 ->GetType(vector_1_instruction->type_id())
                                 ->AsVector();

        auto vector_2_instruction =
            GetFuzzerContext()->RemoveAtRandomIndex(&vector_instructions);
        auto vector_2_type = GetIRContext()
                                 ->get_type_mgr()
                                 ->GetType(vector_2_instruction->type_id())
                                 ->AsVector();

        // |vector_1| and |vector_2| must have the same element type as each
        // other. The loop is guaranteed to terminate because each iteration
        // removes on possible choice for |vector_2|, and there is at least one
        // choice that will cause the loop to exit - namely |vector_1|.
        while (vector_1_type->element_type() != vector_2_type->element_type()) {
          vector_2_instruction =
              GetFuzzerContext()->RemoveAtRandomIndex(&vector_instructions);
          vector_2_type = GetIRContext()
                              ->get_type_mgr()
                              ->GetType(vector_2_instruction->type_id())
                              ->AsVector();
        }

        // Gets components and creates the appropriate result vector type.
        std::vector<uint32_t> components =
            GetFuzzerContext()->GetRandomComponentsForVectorShuffle(
                vector_1_type->element_count() +
                vector_2_type->element_count());
        FindOrCreateVectorType(GetIRContext()->get_type_mgr()->GetId(
                                   vector_1_type->element_type()),
                               static_cast<uint32_t>(components.size()));

        // Applies the vector shuffle transformation.
        ApplyTransformation(TransformationVectorShuffle(
            instruction_descriptor, GetFuzzerContext()->GetFreshId(),
            vector_1_instruction->result_id(),
            vector_2_instruction->result_id(), components));
      });
}

}  // namespace fuzz
}  // namespace spvtools
