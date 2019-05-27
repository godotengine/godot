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

#include "source/reduce/remove_function_reduction_opportunity_finder.h"

#include "source/reduce/remove_function_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveFunctionReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  // Consider each function.
  for (auto& function : *context->module()) {
    if (context->get_def_use_mgr()->NumUses(function.result_id()) > 0) {
      // If the function is referenced, ignore it.
      continue;
    }
    result.push_back(
        MakeUnique<RemoveFunctionReductionOpportunity>(context, &function));
  }
  return result;
}

std::string RemoveFunctionReductionOpportunityFinder::GetName() const {
  return "RemoveFunctionReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
