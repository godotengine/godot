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

#ifndef SOURCE_REDUCE_REMOVE_FUNCTION_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_FUNCTION_REDUCTION_OPPORTUNITY_H_

#include "source/opt/function.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to remove an unreferenced function.
class RemoveFunctionReductionOpportunity : public ReductionOpportunity {
 public:
  // Creates an opportunity to remove |function| from the module represented by
  // |context|.
  RemoveFunctionReductionOpportunity(opt::IRContext* context,
                                     opt::Function* function)
      : context_(context), function_(function) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  // The IR context for the module under analysis.
  opt::IRContext* context_;

  // The function that can be removed.
  opt::Function* function_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  //   SOURCE_REDUCE_REMOVE_FUNCTION_REDUCTION_OPPORTUNITY_H_
