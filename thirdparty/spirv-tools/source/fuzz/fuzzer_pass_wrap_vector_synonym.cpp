// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/fuzzer_pass_wrap_vector_synonym.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_composite_construct.h"
#include "source/fuzz/transformation_wrap_vector_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassWrapVectorSynonym::FuzzerPassWrapVectorSynonym(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassWrapVectorSynonym::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {

        // Randomly decide whether to wrap it to a vector operation.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfWrappingVectorSynonym())) {
          return;
        }

        // The transformation is not applicable if the instruction has missing
        // result id, type id, or is not supported type.
        if (!TransformationWrapVectorSynonym::IsInstructionSupported(
                GetIRContext(), *instruction_iterator)) {
          return;
        }

        // It must be valid to insert an OpCompositeConstruct instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpCompositeConstruct, instruction_iterator)) {
          return;
        }

        // Get the scalar operands from the original instruction.
        opt::Instruction* operand1 = GetIRContext()->get_def_use_mgr()->GetDef(
            instruction_iterator->GetSingleWordInOperand(0));
        opt::Instruction* operand2 = GetIRContext()->get_def_use_mgr()->GetDef(
            instruction_iterator->GetSingleWordInOperand(1));

        // We need to be able to make a synonym of the scalar operation's result
        // id, as well as the operand ids (for example, they cannot be
        // irrelevant).
        if (!fuzzerutil::CanMakeSynonymOf(GetIRContext(),
                                          *GetTransformationContext(),
                                          *instruction_iterator)) {
          return;
        }
        if (!fuzzerutil::CanMakeSynonymOf(
                GetIRContext(), *GetTransformationContext(), *operand1)) {
          return;
        }
        if (!fuzzerutil::CanMakeSynonymOf(
                GetIRContext(), *GetTransformationContext(), *operand2)) {
          return;
        }

        // Get a random vector size from 2 to 4.
        uint32_t vector_size = GetFuzzerContext()->GetWidthOfWrappingVector();

        // Randomly choose a position that target ids should be placed at.
        // The position is in range [0, n - 1], where n is the size of the
        // vector.
        uint32_t position =
            GetFuzzerContext()->GetRandomIndexForWrappingVector(vector_size);

        // Stores the ids of scalar constants.
        std::vector<uint32_t> vec1_components;
        std::vector<uint32_t> vec2_components;

        // Populate components based on vector type and size.
        for (uint32_t i = 0; i < vector_size; ++i) {
          if (i == position) {
            vec1_components.emplace_back(operand1->result_id());
            vec2_components.emplace_back(operand2->result_id());
          } else {
            vec1_components.emplace_back(
                FindOrCreateZeroConstant(operand1->type_id(), true));
            vec2_components.emplace_back(
                FindOrCreateZeroConstant(operand2->type_id(), true));
          }
        }

        // Add two OpCompositeConstruct to the module with result id returned.
        // The added vectors may have different types, for instance if the
        // scalar instruction operates on integers with differing sign.

        // Add the first OpCompositeConstruct that wraps the id of the first
        // operand.
        uint32_t result_id1 = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationCompositeConstruct(
            FindOrCreateVectorType(operand1->type_id(), vector_size),
            vec1_components, instruction_descriptor, result_id1));

        // Add the second OpCompositeConstruct that wraps the id of the second
        // operand.
        uint32_t result_id2 = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationCompositeConstruct(
            FindOrCreateVectorType(operand2->type_id(), vector_size),
            vec2_components, instruction_descriptor, result_id2));

        // The result of the vector instruction that
        // TransformationWrapVectorSynonym will create should be a vector of the
        // right size, with the scalar instruction's result type as its element
        // type. This can be distinct from the types of the operands, if the
        // scalar instruction adds two signed integers and stores the result in
        // an unsigned id, for example. A transformation is applied to add the
        // right type to the module.
        FindOrCreateVectorType(instruction_iterator->type_id(), vector_size);

        // Apply transformation to do vector operation and add synonym between
        // the result vector id and the id of the original instruction.
        ApplyTransformation(TransformationWrapVectorSynonym(
            instruction_iterator->result_id(), result_id1, result_id2,
            GetFuzzerContext()->GetFreshId(), position));
      });
}

}  // namespace fuzz
}  // namespace spvtools
