// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/pass_management/repeated_pass_manager_looped_with_recommendations.h"

namespace spvtools {
namespace fuzz {

RepeatedPassManagerLoopedWithRecommendations::
    RepeatedPassManagerLoopedWithRecommendations(
        FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
        RepeatedPassRecommender* pass_recommender)
    : RepeatedPassManager(fuzzer_context, pass_instances),
      num_transformations_applied_before_last_pass_choice_(0),
      next_pass_index_(0) {
  auto& passes = GetPassInstances()->GetPasses();
  do {
    FuzzerPass* current_pass =
        passes[GetFuzzerContext()->RandomIndex(passes)].get();
    pass_loop_.push_back(current_pass);
    for (auto future_pass :
         pass_recommender->GetFuturePassRecommendations(*current_pass)) {
      recommended_pass_indices_.insert(
          static_cast<uint32_t>(pass_loop_.size()));
      pass_loop_.push_back(future_pass);
    }
  } while (fuzzer_context->ChoosePercentage(
      fuzzer_context->GetChanceOfAddingAnotherPassToPassLoop()));
}

RepeatedPassManagerLoopedWithRecommendations::
    ~RepeatedPassManagerLoopedWithRecommendations() = default;

FuzzerPass* RepeatedPassManagerLoopedWithRecommendations::ChoosePass(
    const protobufs::TransformationSequence& applied_transformations) {
  assert((next_pass_index_ > 0 ||
          recommended_pass_indices_.count(next_pass_index_) == 0) &&
         "The first pass in the loop should not be a recommendation.");
  assert(static_cast<uint32_t>(applied_transformations.transformation_size()) >=
             num_transformations_applied_before_last_pass_choice_ &&
         "The number of applied transformations should not decrease.");
  if (num_transformations_applied_before_last_pass_choice_ ==
      static_cast<uint32_t>(applied_transformations.transformation_size())) {
    // The last pass that was applied did not lead to any new transformations.
    // We thus do not want to apply recommendations based on it, so we skip on
    // to the next non-recommended pass.
    while (recommended_pass_indices_.count(next_pass_index_)) {
      next_pass_index_ =
          (next_pass_index_ + 1) % static_cast<uint32_t>(pass_loop_.size());
    }
  }
  auto result = pass_loop_[next_pass_index_];
  next_pass_index_ =
      (next_pass_index_ + 1) % static_cast<uint32_t>(pass_loop_.size());
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
