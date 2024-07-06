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

#ifndef SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_H_
#define SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_H_

#include <vector>

#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_recommender.h"

namespace spvtools {
namespace fuzz {

// On construction, this pass manager creates a sequence of fuzzer passes which
// is not changed thereafter.  Passes from this sequence are served up in round
// robin fashion each time ChoosePass is invoked - i.e., the sequence is a "pass
// loop".
//
// The pass loop is constructed by repeatedly:
// - Randomly adding an enabled pass
// - Adding all recommended follow-on passes for this pass
// and probabilistically terminating this process.
class RepeatedPassManagerLoopedWithRecommendations
    : public RepeatedPassManager {
 public:
  RepeatedPassManagerLoopedWithRecommendations(
      FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
      RepeatedPassRecommender* pass_recommender);

  ~RepeatedPassManagerLoopedWithRecommendations() override;

  FuzzerPass* ChoosePass(const protobufs::TransformationSequence&
                             applied_transformations) override;

 private:
  // The loop of fuzzer passes to be applied, populated on construction.
  std::vector<FuzzerPass*> pass_loop_;

  // A set of indices into |pass_loop_| recording which passes are in the loop
  // because they are recommended based on previous passes in the loop.  This
  // allows these recommended passes to be skipped if the passes they are
  // meant to amplify had no effect.
  std::unordered_set<uint32_t> recommended_pass_indices_;

  // Used to detect when chosen passes have had no effect, so that their
  // associated recommendations are skipped.
  uint32_t num_transformations_applied_before_last_pass_choice_;

  // An index into |pass_loop_| specifying which pass should be served up next
  // time ChoosePass is invoked.
  uint32_t next_pass_index_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_H_
