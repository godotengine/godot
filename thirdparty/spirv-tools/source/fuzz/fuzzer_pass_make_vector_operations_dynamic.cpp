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

#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_make_vector_operation_dynamic.h"

namespace spvtools {
namespace fuzz {

FuzzerPassMakeVectorOperationsDynamic::FuzzerPassMakeVectorOperationsDynamic(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassMakeVectorOperationsDynamic::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // Randomly decide whether to try applying the transformation.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfMakingVectorOperationDynamic())) {
          continue;
        }

        // |instruction| must be a vector operation.
        if (!TransformationMakeVectorOperationDynamic::IsVectorOperation(
                GetIRContext(), &instruction)) {
          continue;
        }

        // Make sure |instruction| has only one indexing operand.
        assert(
            instruction.NumInOperands() ==
                (instruction.opcode() == spv::Op::OpCompositeExtract ? 2 : 3) &&
            "FuzzerPassMakeVectorOperationsDynamic: the composite "
            "instruction must have "
            "only one indexing operand.");

        // Applies the make vector operation dynamic transformation.
        ApplyTransformation(TransformationMakeVectorOperationDynamic(
            instruction.result_id(),
            FindOrCreateIntegerConstant(
                {instruction.GetSingleWordInOperand(
                    instruction.opcode() == spv::Op::OpCompositeExtract ? 1
                                                                        : 2)},
                32, GetFuzzerContext()->ChooseEven(), false)));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
