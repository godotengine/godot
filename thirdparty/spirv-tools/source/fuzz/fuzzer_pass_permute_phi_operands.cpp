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

#include "source/fuzz/fuzzer_pass_permute_phi_operands.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_permute_phi_operands.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermutePhiOperands::FuzzerPassPermutePhiOperands(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPermutePhiOperands::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& /*unused*/) {
        const auto& inst = *inst_it;

        if (inst.opcode() != spv::Op::OpPhi) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfPermutingPhiOperands())) {
          return;
        }

        // Create a vector of indices for each pair of operands in the |inst|.
        // OpPhi always has an even number of operands.
        std::vector<uint32_t> permutation(inst.NumInOperands() / 2);
        std::iota(permutation.begin(), permutation.end(), 0);
        GetFuzzerContext()->Shuffle(&permutation);

        ApplyTransformation(TransformationPermutePhiOperands(
            inst.result_id(), std::move(permutation)));
      });
}

}  // namespace fuzz
}  // namespace spvtools
