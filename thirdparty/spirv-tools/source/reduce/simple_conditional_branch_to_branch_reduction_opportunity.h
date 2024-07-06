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

#ifndef SOURCE_REDUCE_SIMPLE_CONDITIONAL_BRANCH_TO_BRANCH_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_SIMPLE_CONDITIONAL_BRANCH_TO_BRANCH_REDUCTION_OPPORTUNITY_H_

#include "source/opt/instruction.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to change simple conditional branches (conditional branches
// with one target) to an OpBranch.
class SimpleConditionalBranchToBranchReductionOpportunity
    : public ReductionOpportunity {
 public:
  // Constructs an opportunity to simplify |conditional_branch_instruction|.
  explicit SimpleConditionalBranchToBranchReductionOpportunity(
      opt::Instruction* conditional_branch_instruction);

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::Instruction* conditional_branch_instruction_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_SIMPLE_CONDITIONAL_BRANCH_TO_BRANCH_REDUCTION_OPPORTUNITY_H_
