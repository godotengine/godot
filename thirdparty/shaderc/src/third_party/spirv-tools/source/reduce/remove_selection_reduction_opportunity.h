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

#ifndef SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_

#include "source/opt/basic_block.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity for removing a selection construct by simply removing the
// OpSelectionMerge instruction; thus, the selection must have already been
// simplified to a point where the instruction can be trivially removed.
class RemoveSelectionReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs a reduction opportunity from the selection header |block| in
  // |function|.
  RemoveSelectionReductionOpportunity(opt::BasicBlock* header_block)
      : header_block_(header_block) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  // The header block of the selection.
  opt::BasicBlock* header_block_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  //   SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_
