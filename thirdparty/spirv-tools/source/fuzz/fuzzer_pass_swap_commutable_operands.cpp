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

#include "source/fuzz/fuzzer_pass_swap_commutable_operands.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_swap_commutable_operands.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSwapCommutableOperands::FuzzerPassSwapCommutableOperands(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassSwapCommutableOperands::Apply() {
  auto context = GetIRContext();
  // Iterates over the module's instructions and checks whether it is
  // commutative. In this case, the transformation is probabilistically applied.
  context->module()->ForEachInst(
      [this, context](opt::Instruction* instruction) {
        if (spvOpcodeIsCommutativeBinaryOperator(instruction->opcode()) &&
            GetFuzzerContext()->ChooseEven()) {
          auto instructionDescriptor =
              MakeInstructionDescriptor(context, instruction);
          auto transformation =
              TransformationSwapCommutableOperands(instructionDescriptor);
          ApplyTransformation(transformation);
        }
      });
}

}  // namespace fuzz
}  // namespace spvtools
