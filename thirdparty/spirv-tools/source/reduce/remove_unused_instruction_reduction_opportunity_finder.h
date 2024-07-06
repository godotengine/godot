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

#ifndef SOURCE_REDUCE_REMOVE_UNREFERENCED_INSTRUCTION_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_REMOVE_UNREFERENCED_INSTRUCTION_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder for opportunities to remove non-control-flow instructions in blocks
// in cases where the instruction's id is either not referenced at all, or
// referenced only in a trivial manner (for example, we regard a struct type as
// unused if it is referenced only by struct layout decorations).  As well as
// making the module smaller, removing an instruction that references particular
// ids may create opportunities for subsequently removing the instructions that
// generated those ids.
class RemoveUnusedInstructionReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  explicit RemoveUnusedInstructionReductionOpportunityFinder(
      bool remove_constants_and_undefs);

  ~RemoveUnusedInstructionReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context, uint32_t target_function) const final;

 private:
  // Returns true if and only if the only uses of |inst| are by decorations that
  // relate intimately to the instruction (as opposed to decorations that could
  // be removed independently), or by interface ids in OpEntryPoint.
  bool OnlyReferencedByIntimateDecorationOrEntryPointInterface(
      opt::IRContext* context, const opt::Instruction& inst) const;

  // Returns true if and only if |inst| is a decoration instruction that can
  // legitimately be removed on its own (rather than one that has to be removed
  // simultaneously with other instructions).
  bool IsIndependentlyRemovableDecoration(const opt::Instruction& inst) const;

  bool remove_constants_and_undefs_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_UNREFERENCED_INSTRUCTION_REDUCTION_OPPORTUNITY_FINDER_H_
