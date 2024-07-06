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

#ifndef SOURCE_REDUCE_REDUCTION_PASS_H_
#define SOURCE_REDUCE_REDUCTION_PASS_H_

#include <limits>

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_opportunity_finder.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace reduce {

// Abstract class representing a reduction pass, which can be repeatedly
// invoked to find and apply particular reduction opportunities to a SPIR-V
// binary.  In the spirit of delta debugging, a pass initially tries to apply
// large chunks of reduction opportunities, iterating through available
// opportunities at a given granularity.  When an iteration over available
// opportunities completes, the granularity is reduced and iteration starts
// again, until the minimum granularity is reached.
class ReductionPass {
 public:
  // Constructs a reduction pass with a given target environment, |target_env|,
  // and a given finder of reduction opportunities, |finder|.
  explicit ReductionPass(const spv_target_env target_env,
                         std::unique_ptr<ReductionOpportunityFinder> finder)
      : target_env_(target_env),
        finder_(std::move(finder)),
        index_(0),
        granularity_(std::numeric_limits<uint32_t>::max()) {}

  // Applies the reduction pass to the given binary by applying a "chunk" of
  // reduction opportunities. Returns the new binary if a chunk was applied; in
  // this case, before the next call the caller must invoke
  // NotifyInteresting(...) to indicate whether the new binary is interesting.
  // Returns an empty vector if there are no more chunks left to apply; in this
  // case, the index will be reset and the granularity lowered for the next
  // round.
  //
  // If |target_function| is non-zero, only reduction opportunities that
  // simplify the internals of the function with result id |target_function|
  // will be applied.
  std::vector<uint32_t> TryApplyReduction(const std::vector<uint32_t>& binary,
                                          uint32_t target_function);

  // Notifies the reduction pass whether the binary returned from
  // TryApplyReduction is interesting, so that the next call to
  // TryApplyReduction will avoid applying the same chunk of opportunities.
  void NotifyInteresting(bool interesting);

  // Sets a consumer to which relevant messages will be directed.
  void SetMessageConsumer(MessageConsumer consumer);

  // Returns true if the granularity with which reduction opportunities are
  // applied has reached a minimum.
  bool ReachedMinimumGranularity() const;

  // Returns the name associated with this reduction pass (based on its
  // associated finder).
  std::string GetName() const;

 private:
  const spv_target_env target_env_;
  const std::unique_ptr<ReductionOpportunityFinder> finder_;
  MessageConsumer consumer_;
  uint32_t index_;
  uint32_t granularity_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_PASS_H_
