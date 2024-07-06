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

#ifndef SOURCE_FUZZ_REPEATED_PASS_MANAGER_RANDOM_WITH_RECOMMENDATIONS_H_
#define SOURCE_FUZZ_REPEATED_PASS_MANAGER_RANDOM_WITH_RECOMMENDATIONS_H_

#include <deque>

#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_recommender.h"

namespace spvtools {
namespace fuzz {

// This repeated pass manager uses a pass recommender to recommend future passes
// each time a fuzzer pass is run.  It keeps a queue of recommended passes.
//
// Each time a fuzzer pass is requested, the manager either selects an enabled
// fuzzer pass at random, or selects the pass at the front of the recommendation
// queue, removing it from the queue.  The decision of which of these pass
// selection methods to use is made randomly each time ChoosePass is called.
//
// Either way, recommended follow-on passes for the chosen pass are added to
// the recommendation queue.
class RepeatedPassManagerRandomWithRecommendations
    : public RepeatedPassManager {
 public:
  RepeatedPassManagerRandomWithRecommendations(
      FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
      RepeatedPassRecommender* pass_recommender);

  ~RepeatedPassManagerRandomWithRecommendations() override;

  FuzzerPass* ChoosePass(const protobufs::TransformationSequence&
                             applied_transformations) override;

 private:
  // The queue of passes that have been recommended based on previously-chosen
  // passes.
  std::deque<FuzzerPass*> recommended_passes_;

  // Used to recommend future passes.
  RepeatedPassRecommender* pass_recommender_;

  // Used to detect when chosen passes have had no effect, so that their
  // associated recommendations are skipped.
  uint32_t num_transformations_applied_before_last_pass_choice_;

  // The fuzzer pass returned last time ChoosePass() was called; nullptr if
  // ChoosePass() has not yet been called.
  FuzzerPass* last_pass_choice_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_MANAGER_RANDOM_WITH_RECOMMENDATIONS_H_
