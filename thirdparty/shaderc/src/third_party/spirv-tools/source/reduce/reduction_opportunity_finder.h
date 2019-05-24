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

#ifndef SOURCE_REDUCE_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// Abstract class for finding opportunities for reducing a SPIR-V module.
class ReductionOpportunityFinder {
 public:
  ReductionOpportunityFinder() = default;

  virtual ~ReductionOpportunityFinder() = default;

  // Finds and returns the reduction opportunities relevant to this pass that
  // could be applied to the given SPIR-V module.
  virtual std::vector<std::unique_ptr<ReductionOpportunity>>
  GetAvailableOpportunities(opt::IRContext* context) const = 0;

  // Provides a name for the finder.
  virtual std::string GetName() const = 0;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_OPPORTUNITY_FINDER_H_
