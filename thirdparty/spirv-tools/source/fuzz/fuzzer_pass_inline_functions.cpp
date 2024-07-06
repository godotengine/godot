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

#include "source/fuzz/fuzzer_pass_inline_functions.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_inline_function.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassInlineFunctions::FuzzerPassInlineFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassInlineFunctions::Apply() {
  // |function_call_instructions| are the instructions that will be inlined.
  // First, they will be collected and then do the inlining in another loop.
  // This avoids changing the module while it is being inspected.
  std::vector<opt::Instruction*> function_call_instructions;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfInliningFunction())) {
          continue;
        }

        // |instruction| must be suitable for inlining.
        if (!TransformationInlineFunction::IsSuitableForInlining(
                GetIRContext(), &instruction)) {
          continue;
        }

        function_call_instructions.push_back(&instruction);
      }
    }
  }

  // Once the function calls have been collected, it's time to actually create
  // and apply the inlining transformations.
  for (auto& function_call_instruction : function_call_instructions) {
    // If |function_call_instruction| is not the penultimate instruction in its
    // block or its block termination instruction is not OpBranch, then try to
    // split |function_call_block| such that the conditions are met.
    auto* function_call_block =
        GetIRContext()->get_instr_block(function_call_instruction);
    if ((function_call_instruction != &*--function_call_block->tail() ||
         function_call_block->terminator()->opcode() != spv::Op::OpBranch) &&
        !MaybeApplyTransformation(TransformationSplitBlock(
            MakeInstructionDescriptor(GetIRContext(),
                                      function_call_instruction->NextNode()),
            GetFuzzerContext()->GetFreshId()))) {
      continue;
    }

    auto* called_function = fuzzerutil::FindFunction(
        GetIRContext(), function_call_instruction->GetSingleWordInOperand(0));

    // Mapping the called function instructions.
    std::map<uint32_t, uint32_t> result_id_map;
    for (auto& called_function_block : *called_function) {
      // The called function entry block label will not be inlined.
      if (&called_function_block != &*called_function->entry()) {
        result_id_map[called_function_block.GetLabelInst()->result_id()] =
            GetFuzzerContext()->GetFreshId();
      }

      for (auto& instruction_to_inline : called_function_block) {
        // The instructions are mapped to fresh ids.
        if (instruction_to_inline.HasResultId()) {
          result_id_map[instruction_to_inline.result_id()] =
              GetFuzzerContext()->GetFreshId();
        }
      }
    }

    // Applies the inline function transformation.
    ApplyTransformation(TransformationInlineFunction(
        function_call_instruction->result_id(), result_id_map));
  }
}

}  // namespace fuzz
}  // namespace spvtools
