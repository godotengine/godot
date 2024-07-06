// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_add_synonyms.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddSynonyms::FuzzerPassAddSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddSynonyms::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (GetTransformationContext()->GetFactManager()->BlockIsDead(
                block->id())) {
          // Don't create synonyms in dead blocks.
          return;
        }

        // Skip |inst_it| if we can't insert anything above it. OpIAdd is just
        // a representative of some instruction that might be produced by the
        // transformation.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpIAdd,
                                                          inst_it)) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingSynonyms())) {
          return;
        }

        auto synonym_type = GetFuzzerContext()->GetRandomSynonymType();

        // Select all instructions that can be used to create a synonym to.
        auto available_instructions = FindAvailableInstructions(
            function, block, inst_it,
            [synonym_type, this](opt::IRContext* ir_context,
                                 opt::Instruction* inst) {
              // Check that we can create a synonym to |inst| as described by
              // the |synonym_type| and insert it before |inst_it|.
              return TransformationAddSynonym::IsInstructionValid(
                  ir_context, *GetTransformationContext(), inst, synonym_type);
            });

        if (available_instructions.empty()) {
          return;
        }

        const auto* existing_synonym =
            available_instructions[GetFuzzerContext()->RandomIndex(
                available_instructions)];

        // Make sure the module contains all instructions required to apply the
        // transformation.
        switch (synonym_type) {
          case protobufs::TransformationAddSynonym::ADD_ZERO:
          case protobufs::TransformationAddSynonym::SUB_ZERO:
          case protobufs::TransformationAddSynonym::LOGICAL_OR:
          case protobufs::TransformationAddSynonym::BITWISE_OR:
          case protobufs::TransformationAddSynonym::BITWISE_XOR:
            // Create a zero constant to be used as an operand of the synonymous
            // instruction.
            FindOrCreateZeroConstant(existing_synonym->type_id(), false);
            break;
          case protobufs::TransformationAddSynonym::MUL_ONE:
          case protobufs::TransformationAddSynonym::LOGICAL_AND: {
            const auto* existing_synonym_type =
                GetIRContext()->get_type_mgr()->GetType(
                    existing_synonym->type_id());
            assert(existing_synonym_type && "Instruction has invalid type");

            if (const auto* vector = existing_synonym_type->AsVector()) {
              auto element_type_id =
                  GetIRContext()->get_type_mgr()->GetId(vector->element_type());
              assert(element_type_id && "Vector's element type is invalid");

              auto one_word = vector->element_type()->AsFloat()
                                  ? fuzzerutil::FloatToWord(1)
                                  : 1u;
              FindOrCreateCompositeConstant(
                  std::vector<uint32_t>(
                      vector->element_count(),
                      FindOrCreateConstant({one_word}, element_type_id, false)),
                  existing_synonym->type_id(), false);
            } else {
              FindOrCreateConstant(
                  {existing_synonym_type->AsFloat() ? fuzzerutil::FloatToWord(1)
                                                    : 1u},
                  existing_synonym->type_id(), false);
            }
          } break;
          default:
            // This assertion will fail if some SynonymType is missing from the
            // switch statement.
            assert(
                !TransformationAddSynonym::IsAdditionalConstantRequired(
                    synonym_type) &&
                "|synonym_type| requires an additional constant to be present "
                "in the module");
            break;
        }

        ApplyTransformation(TransformationAddSynonym(
            existing_synonym->result_id(), synonym_type,
            GetFuzzerContext()->GetFreshId(), instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
