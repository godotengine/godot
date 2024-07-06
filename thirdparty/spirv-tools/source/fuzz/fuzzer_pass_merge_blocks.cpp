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

#include "source/fuzz/fuzzer_pass_merge_blocks.h"

#include <vector>

#include "source/fuzz/transformation_merge_blocks.h"

namespace spvtools {
namespace fuzz {

FuzzerPassMergeBlocks::FuzzerPassMergeBlocks(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassMergeBlocks::Apply() {
  // First we populate a sequence of transformations that we might consider
  // applying.
  std::vector<TransformationMergeBlocks> potential_transformations;
  // We do this by considering every block of every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We probabilistically decide to ignore some blocks.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfMergingBlocks())) {
        continue;
      }
      // For other blocks, we add a transformation to merge the block into its
      // predecessor if that transformation would be applicable.
      TransformationMergeBlocks transformation(block.id());
      if (transformation.IsApplicable(GetIRContext(),
                                      *GetTransformationContext())) {
        potential_transformations.push_back(transformation);
      }
    }
  }

  while (!potential_transformations.empty()) {
    auto transformation =
        GetFuzzerContext()->RemoveAtRandomIndex(&potential_transformations);
    MaybeApplyTransformation(transformation);
  }
}

}  // namespace fuzz
}  // namespace spvtools
