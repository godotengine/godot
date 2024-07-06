// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_REDUCE_REMOVE_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_H_

#include "source/reduce/reduction_opportunity.h"

#include "source/opt/instruction.h"

namespace spvtools {
namespace reduce {

// An opportunity for removing a member from a struct type, adjusting all uses
// of the struct accordingly.
class RemoveStructMemberReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs a reduction opportunity from the struct type |struct_type|, for
  // removal of member |member_index|.
  RemoveStructMemberReductionOpportunity(opt::Instruction* struct_type,
                                         uint32_t member_index)
      : struct_type_(struct_type),
        member_index_(member_index),
        original_number_of_members_(struct_type->NumInOperands()) {}

  // Opportunities to remove fields from a common struct type mutually
  // invalidate each other.  We guard against this by requiring that the struct
  // still has the number of members it had when the opportunity was created.
  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  // |composite_access_instruction| is an instruction that accesses a composite
  // id using either a series of literal indices (e.g. in the case of
  // OpCompositeInsert) or a series of index ids (e.g. in the case of
  // OpAccessChain).
  //
  // This function adjusts the indices that are used by
  // |composite_access_instruction| to that whenever an index is accessing a
  // member of |struct_type_|, it is decremented if the member is beyond
  // |member_index_|, to account for the removal of the |member_index_|-th
  // member.
  //
  // |composite_type_id| is the id of the composite type that the series of
  // indices is to be applied to.
  //
  // |first_index_input_operand| specifies the first input operand that is an
  // index.
  //
  // |literal_indices| specifies whether indices are given as literals (true),
  // or as ids (false).
  //
  // If id-based indexing is used, this function will add a constant for
  // |member_index_| - 1 to the module if needed.
  void AdjustAccessedIndices(
      uint32_t composite_type_id, uint32_t first_index_input_operand,
      bool literal_indices, opt::IRContext* context,
      opt::Instruction* composite_access_instruction) const;

  // The struct type from which a member is to be removed.
  opt::Instruction* struct_type_;

  uint32_t member_index_;

  uint32_t original_number_of_members_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  //   SOURCE_REDUCE_REMOVE_STRUCT_MEMBER_REDUCTION_OPPORTUNITY_H_
