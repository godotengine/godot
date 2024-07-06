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

#include "source/fuzz/fuzzer_pass_add_composite_extract.h"

#include "source/fuzz/available_instructions.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_composite_extract.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCompositeExtract::FuzzerPassAddCompositeExtract(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddCompositeExtract::Apply() {
  std::vector<const protobufs::DataDescriptor*> composite_synonyms;
  for (const auto* dd :
       GetTransformationContext()->GetFactManager()->GetAllSynonyms()) {
    // |dd| must describe a component of a composite.
    if (!dd->index().empty()) {
      composite_synonyms.push_back(dd);
    }
  }

  AvailableInstructions available_composites(
      GetIRContext(), [](opt::IRContext* ir_context, opt::Instruction* inst) {
        return inst->type_id() && inst->result_id() &&
               fuzzerutil::IsCompositeType(
                   ir_context->get_type_mgr()->GetType(inst->type_id()));
      });

  ForEachInstructionWithInstructionDescriptor(
      [this, &available_composites, &composite_synonyms](
          opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
          opt::BasicBlock::iterator inst_it,
          const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpCompositeExtract, inst_it)) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCompositeExtract())) {
          return;
        }

        std::vector<const protobufs::DataDescriptor*> available_synonyms;
        for (const auto* dd : composite_synonyms) {
          if (fuzzerutil::IdIsAvailableBeforeInstruction(
                  GetIRContext(), &*inst_it, dd->object())) {
            available_synonyms.push_back(dd);
          }
        }

        auto candidate_composites =
            available_composites.GetAvailableBeforeInstruction(&*inst_it);

        if (available_synonyms.empty() && candidate_composites.empty()) {
          return;
        }

        uint32_t composite_id = 0;
        std::vector<uint32_t> indices;

        if (available_synonyms.empty() || (!candidate_composites.empty() &&
                                           GetFuzzerContext()->ChooseEven())) {
          const auto* inst =
              candidate_composites[GetFuzzerContext()->RandomIndex(
                  candidate_composites)];
          composite_id = inst->result_id();

          auto type_id = inst->type_id();
          do {
            uint32_t number_of_members = 0;

            const auto* type_inst =
                GetIRContext()->get_def_use_mgr()->GetDef(type_id);
            assert(type_inst && "Composite instruction has invalid type id");

            switch (type_inst->opcode()) {
              case spv::Op::OpTypeArray:
                number_of_members =
                    fuzzerutil::GetArraySize(*type_inst, GetIRContext());
                break;
              case spv::Op::OpTypeVector:
              case spv::Op::OpTypeMatrix:
                number_of_members = type_inst->GetSingleWordInOperand(1);
                break;
              case spv::Op::OpTypeStruct:
                number_of_members = type_inst->NumInOperands();
                break;
              default:
                assert(false && "|type_inst| is not a composite");
                return;
            }

            if (number_of_members == 0) {
              return;
            }

            indices.push_back(
                GetFuzzerContext()->GetRandomCompositeExtractIndex(
                    number_of_members));

            switch (type_inst->opcode()) {
              case spv::Op::OpTypeArray:
              case spv::Op::OpTypeVector:
              case spv::Op::OpTypeMatrix:
                type_id = type_inst->GetSingleWordInOperand(0);
                break;
              case spv::Op::OpTypeStruct:
                type_id = type_inst->GetSingleWordInOperand(indices.back());
                break;
              default:
                assert(false && "|type_inst| is not a composite");
                return;
            }
          } while (fuzzerutil::IsCompositeType(
                       GetIRContext()->get_type_mgr()->GetType(type_id)) &&
                   GetFuzzerContext()->ChoosePercentage(
                       GetFuzzerContext()
                           ->GetChanceOfGoingDeeperToExtractComposite()));
        } else {
          const auto* dd = available_synonyms[GetFuzzerContext()->RandomIndex(
              available_synonyms)];

          composite_id = dd->object();
          indices.assign(dd->index().begin(), dd->index().end());
        }

        assert(composite_id != 0 && !indices.empty() &&
               "Composite object should have been chosen correctly");

        ApplyTransformation(TransformationCompositeExtract(
            instruction_descriptor, GetFuzzerContext()->GetFreshId(),
            composite_id, indices));
      });
}

}  // namespace fuzz
}  // namespace spvtools
