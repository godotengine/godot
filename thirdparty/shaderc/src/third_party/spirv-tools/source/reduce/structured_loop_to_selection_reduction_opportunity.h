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

#ifndef SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_

#include "source/opt/def_use_manager.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/function.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to replace a structured loop with a selection.
class StructuredLoopToSelectionReductionOpportunity
    : public ReductionOpportunity {
 public:
  // Constructs an opportunity from a loop header block and the function that
  // encloses it.
  explicit StructuredLoopToSelectionReductionOpportunity(
      opt::IRContext* context, opt::BasicBlock* loop_construct_header,
      opt::Function* enclosing_function)
      : context_(context),
        loop_construct_header_(loop_construct_header),
        enclosing_function_(enclosing_function) {}

  // Returns true if the loop header is reachable.  A structured loop might
  // become unreachable as a result of turning another structured loop into
  // a selection.
  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  // Parameter |original_target_id| is the id of the loop's merge block or
  // continue target.  This method considers each edge of the form
  // b->original_target_id and transforms it into an edge of the form b->c,
  // where c is the merge block of the structured control flow construct that
  // most tightly contains b.
  void RedirectToClosestMergeBlock(uint32_t original_target_id);

  // |source_id|, |original_target_id| and |new_target_id| are required to all
  // be distinct, with a CFG edge existing from |source_id| to
  // |original_target_id|, and |original_target_id| being either the merge block
  // or continue target for the loop being operated on.
  // The method removes this edge and adds an edge from
  // |source_id| to |new_target_id|.  It takes care of fixing up any OpPhi
  // instructions associated with |original_target_id| and |new_target_id|.
  void RedirectEdge(uint32_t source_id, uint32_t original_target_id,
                    uint32_t new_target_id);

  // Removes any components of |to_block|'s phi instructions relating to
  // |from_id|.
  void AdaptPhiInstructionsForRemovedEdge(uint32_t from_id,
                                          opt::BasicBlock* to_block);

  // Adds components to |to_block|'s phi instructions to account for a new
  // incoming edge from |from_id|.
  void AdaptPhiInstructionsForAddedEdge(uint32_t from_id,
                                        opt::BasicBlock* to_block);

  // Turns the OpLoopMerge for the loop into OpSelectionMerge, and adapts the
  // following branch instruction accordingly.
  void ChangeLoopToSelection();

  // Fixes any scenarios where, due to CFG changes, ids have uses not dominated
  // by their definitions, by changing such uses to uses of OpUndef or of dummy
  // variables.
  void FixNonDominatedIdUses();

  // Returns true if and only if at least one of the following holds:
  // 1) |def| dominates |use|
  // 2) |def| is an OpVariable
  // 3) |use| is part of an OpPhi, with associated incoming block b, and |def|
  // dominates b.
  bool DefinitionSufficientlyDominatesUse(opt::Instruction* def,
                                          opt::Instruction* use,
                                          uint32_t use_index,
                                          opt::BasicBlock& def_block);

  // Checks whether the global value list has an OpVariable of the given pointer
  // type, adding one if not, and returns the id of such an OpVariable.
  //
  // TODO(2184): This will likely be used by other reduction passes, so should
  // be factored out in due course.
  uint32_t FindOrCreateGlobalVariable(uint32_t pointer_type_id);

  // Checks whether the enclosing function has an OpVariable of the given
  // pointer type, adding one if not, and returns the id of such an OpVariable.
  //
  // TODO(2184): This will likely be used by other reduction passes, so should
  // be factored out in due course.
  uint32_t FindOrCreateFunctionVariable(uint32_t pointer_type_id);

  opt::IRContext* context_;
  opt::BasicBlock* loop_construct_header_;
  opt::Function* enclosing_function_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_
