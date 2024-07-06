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

#include "source/fuzz/pass_management/repeated_pass_manager_random_with_recommendations.h"

namespace spvtools {
namespace fuzz {

RepeatedPassManagerRandomWithRecommendations::
    RepeatedPassManagerRandomWithRecommendations(
        FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
        RepeatedPassRecommender* pass_recommender)
    : RepeatedPassManager(fuzzer_context, pass_instances),
      pass_recommender_(pass_recommender),
      num_transformations_applied_before_last_pass_choice_(0),
      last_pass_choice_(nullptr) {}

RepeatedPassManagerRandomWithRecommendations::
    ~RepeatedPassManagerRandomWithRecommendations() = default;

FuzzerPass* RepeatedPassManagerRandomWithRecommendations::ChoosePass(
    const protobufs::TransformationSequence& applied_transformations) {
  assert(static_cast<uint32_t>(applied_transformations.transformation_size()) >=
             num_transformations_applied_before_last_pass_choice_ &&
         "The number of applied transformations should not decrease.");
  if (last_pass_choice_ != nullptr &&
      static_cast<uint32_t>(applied_transformations.transformation_size()) >
          num_transformations_applied_before_last_pass_choice_) {
    // The last pass had some effect, so we make future recommendations based on
    // it.
    for (auto future_pass :
         pass_recommender_->GetFuturePassRecommendations(*last_pass_choice_)) {
      recommended_passes_.push_back(future_pass);
    }
  }

  FuzzerPass* result;
  if (recommended_passes_.empty() || GetFuzzerContext()->ChooseEven()) {
    auto& passes = GetPassInstances()->GetPasses();
    result = passes[GetFuzzerContext()->RandomIndex(passes)].get();
  } else {
    result = recommended_passes_.front();
    recommended_passes_.pop_front();
  }
  assert(result != nullptr && "A pass must have been chosen.");
  last_pass_choice_ = result;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
