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

#include "source/fuzz/pass_management/repeated_pass_manager.h"

#include "source/fuzz/pass_management/repeated_pass_manager_looped_with_recommendations.h"
#include "source/fuzz/pass_management/repeated_pass_manager_random_with_recommendations.h"
#include "source/fuzz/pass_management/repeated_pass_manager_simple.h"

namespace spvtools {
namespace fuzz {

RepeatedPassManager::RepeatedPassManager(FuzzerContext* fuzzer_context,
                                         RepeatedPassInstances* pass_instances)
    : fuzzer_context_(fuzzer_context), pass_instances_(pass_instances) {}

RepeatedPassManager::~RepeatedPassManager() = default;

std::unique_ptr<RepeatedPassManager> RepeatedPassManager::Create(
    RepeatedPassStrategy strategy, FuzzerContext* fuzzer_context,
    RepeatedPassInstances* pass_instances,
    RepeatedPassRecommender* pass_recommender) {
  switch (strategy) {
    case RepeatedPassStrategy::kSimple:
      return MakeUnique<RepeatedPassManagerSimple>(fuzzer_context,
                                                   pass_instances);
    case RepeatedPassStrategy::kLoopedWithRecommendations:
      return MakeUnique<RepeatedPassManagerLoopedWithRecommendations>(
          fuzzer_context, pass_instances, pass_recommender);
    case RepeatedPassStrategy::kRandomWithRecommendations:
      return MakeUnique<RepeatedPassManagerRandomWithRecommendations>(
          fuzzer_context, pass_instances, pass_recommender);
  }

  assert(false && "Unreachable");
  return nullptr;
}

}  // namespace fuzz
}  // namespace spvtools
