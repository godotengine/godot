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

#include "source/fuzz/fuzzer_pass_add_dead_blocks.h"

#include <algorithm>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_dead_block.h"

namespace spvtools {
namespace fuzz {

namespace {

const size_t kMaxTransformationsInOnePass = 100U;

}  // namespace

FuzzerPassAddDeadBlocks::FuzzerPassAddDeadBlocks(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddDeadBlocks::Apply() {
  // We iterate over all blocks in the module collecting up those at which we
  // might add a branch to a new dead block.  We then loop over all such
  // candidates and actually apply transformations.  This separation is to
  // avoid modifying the module as we traverse it.
  std::vector<TransformationAddDeadBlock> candidate_transformations;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfAddingDeadBlock())) {
        continue;
      }

      // Make sure the module contains a boolean constant equal to
      // |condition_value|.
      bool condition_value = GetFuzzerContext()->ChooseEven();
      FindOrCreateBoolConstant(condition_value, false);

      // We speculatively create a transformation, and then apply it (below) if
      // it turns out to be applicable.  This avoids duplicating the logic for
      // applicability checking.
      //
      // It means that fresh ids for transformations that turn out not to be
      // applicable end up being unused.
      candidate_transformations.emplace_back(TransformationAddDeadBlock(
          GetFuzzerContext()->GetFreshId(), block.id(), condition_value));
    }
  }
  // Applying transformations can be expensive as each transformation requires
  // dominator information and also invalidates dominator information. We thus
  // limit the number of transformations that one application of this fuzzer
  // pass can apply. We choose to do this after identifying all the
  // transformations that we *might* want to apply, rather than breaking the
  // above loops once the limit is reached, to avoid biasing towards
  // transformations that target early parts of the module.
  GetFuzzerContext()->Shuffle(&candidate_transformations);
  for (size_t i = 0; i < std::min(kMaxTransformationsInOnePass,
                                  candidate_transformations.size());
       i++) {
    MaybeApplyTransformation(candidate_transformations[i]);
  }
}

}  // namespace fuzz
}  // namespace spvtools
