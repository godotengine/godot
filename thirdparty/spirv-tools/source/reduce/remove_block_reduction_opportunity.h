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

#ifndef SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_H_

#include "source/opt/basic_block.h"
#include "source/opt/function.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to remove an unreferenced block.
// See RemoveBlockReductionOpportunityFinder.
class RemoveBlockReductionOpportunity : public ReductionOpportunity {
 public:
  // Creates the opportunity to remove |block| in |function| in |context|.
  RemoveBlockReductionOpportunity(opt::IRContext* context,
                                  opt::Function* function,
                                  opt::BasicBlock* block);

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::IRContext* context_;
  opt::Function* function_;
  opt::BasicBlock* block_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_H_
