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

#ifndef SOURCE_FUZZ_REPEATED_PASS_RECOMMENDER_H_
#define SOURCE_FUZZ_REPEATED_PASS_RECOMMENDER_H_

#include <vector>

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Interface for influencing interactions between repeated fuzzer passes, by
// allowing hints as to which passes are recommended to be run after one
// another.
class RepeatedPassRecommender {
 public:
  virtual ~RepeatedPassRecommender();

  // Given a reference to a repeated pass, |pass|, returns a sequence of
  // repeated pass instances that might be worth running soon after having
  // run |pass|.
  virtual std::vector<FuzzerPass*> GetFuturePassRecommendations(
      const FuzzerPass& pass) = 0;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_RECOMMENDER_H_
