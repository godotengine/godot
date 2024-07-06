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

#include "source/fuzz/fuzzer_pass_mutate_pointers.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_mutate_pointer.h"

namespace spvtools {
namespace fuzz {

FuzzerPassMutatePointers::FuzzerPassMutatePointers(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassMutatePointers::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfMutatingPointer())) {
          return;
        }

        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLoad,
                                                          inst_it)) {
          return;
        }

        auto available_pointers = FindAvailableInstructions(
            function, block, inst_it,
            [](opt::IRContext* ir_context, opt::Instruction* inst) {
              return TransformationMutatePointer::IsValidPointerInstruction(
                  ir_context, *inst);
            });

        if (available_pointers.empty()) {
          return;
        }

        const auto* pointer_inst =
            available_pointers[GetFuzzerContext()->RandomIndex(
                available_pointers)];

        // Make sure there is an irrelevant constant in the module.
        FindOrCreateZeroConstant(fuzzerutil::GetPointeeTypeIdFromPointerType(
                                     GetIRContext(), pointer_inst->type_id()),
                                 true);

        ApplyTransformation(TransformationMutatePointer(
            pointer_inst->result_id(), GetFuzzerContext()->GetFreshId(),
            instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
