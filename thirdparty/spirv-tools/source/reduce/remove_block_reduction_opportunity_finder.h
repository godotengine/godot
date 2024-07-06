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

#ifndef SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/opt/function.h"
#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder of opportunities to remove a block. The optimizer can remove dead
// code. However, the reducer needs to be able to remove at a fine-grained
// level.
class RemoveBlockReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  RemoveBlockReductionOpportunityFinder() = default;

  ~RemoveBlockReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context, uint32_t target_function) const final;

 private:
  // Returns true if the block |bi| in function |function| is a valid
  // opportunity according to various restrictions.
  static bool IsBlockValidOpportunity(opt::IRContext* context,
                                      opt::Function* function,
                                      opt::Function::iterator* bi);

  // Returns true if the instructions (definitions) in block |bi| have no
  // references, except for references from inside the block itself.
  static bool BlockInstructionsHaveNoOutsideReferences(
      opt::IRContext* context, const opt::Function::iterator& bi);
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H_
