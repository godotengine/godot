// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_REDUCE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder of opportunities to replace id operands of instructions with ids of
// undef.
class OperandToUndefReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  OperandToUndefReductionOpportunityFinder() = default;

  ~OperandToUndefReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context) const final;

 private:
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_FINDER_H_
