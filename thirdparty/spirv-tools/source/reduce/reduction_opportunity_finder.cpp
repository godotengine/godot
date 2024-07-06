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

#include "reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

std::vector<opt::Function*> ReductionOpportunityFinder::GetTargetFunctions(
    opt::IRContext* ir_context, uint32_t target_function) {
  std::vector<opt::Function*> result;
  for (auto& function : *ir_context->module()) {
    if (!target_function || function.result_id() == target_function) {
      result.push_back(&function);
    }
  }
  assert((!target_function || !result.empty()) &&
         "Requested target function must exist.");
  return result;
}

}  // namespace reduce
}  // namespace spvtools
