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

#ifndef SOURCE_REDUCE_REMOVE_UNUSED_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_FINDER_H_
#define SOURCE_REDUCE_REMOVE_UNUSED_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_FINDER_H_

#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder for opportunities to remove struct members that are not explicitly
// used by extract, insert or access chain instructions.
class RemoveUnusedStructMemberReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  RemoveUnusedStructMemberReductionOpportunityFinder() = default;

  ~RemoveUnusedStructMemberReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context, uint32_t target_function) const final;

 private:
  // A helper method to update |unused_members_to_structs| by removing from it
  // all struct member accesses that take place in
  // |composite_access_instruction|.
  //
  // |composite_type_id| is the type of the root object indexed into by the
  // instruction.
  //
  // |first_index_in_operand| provides indicates where in the input operands the
  // sequence of indices begins.
  //
  // |literal_indices| indicates whether indices are literals (true) or ids
  // (false).
  void MarkAccessedMembersAsUsed(
      opt::IRContext* context, uint32_t composite_type_id,
      uint32_t first_index_in_operand, bool literal_indices,
      const opt::Instruction& composite_access_instruction,
      std::map<uint32_t, std::set<opt::Instruction*>>* unused_member_to_structs)
      const;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_UNUSED_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_FINDER_H_
