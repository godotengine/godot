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

#ifndef SOURCE_REDUCE_SIMPLIFY_CONDITIONAL_BRANCH_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_SIMPLIFY_CONDITIONAL_BRANCH_REDUCTION_OPPORTUNITY_H_

#include "source/opt/basic_block.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to simplify a conditional branch to a simple conditional
// branch (a conditional branch with one target).
class ConditionalBranchToSimpleConditionalBranchReductionOpportunity
    : public ReductionOpportunity {
 public:
  // Constructs an opportunity to simplify |conditional_branch_instruction|. If
  // |redirect_to_true| is true, the false target will be changed to also point
  // to the true target; otherwise, the true target will be changed to also
  // point to the false target.
  explicit ConditionalBranchToSimpleConditionalBranchReductionOpportunity(
      opt::IRContext* context, opt::Instruction* conditional_branch_instruction,
      bool redirect_to_true);

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::IRContext* context_;
  opt::Instruction* conditional_branch_instruction_;

  // If true, the false target will be changed to point to the true target;
  // otherwise, the true target will be changed to point to the false target.
  bool redirect_to_true_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_SIMPLIFY_CONDITIONAL_BRANCH_REDUCTION_OPPORTUNITY_H_
