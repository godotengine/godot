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

#ifndef SOURCE_REDUCE_CHANGE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_CHANGE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_H_

#include "source/opt/instruction.h"
#include "source/reduce/reduction_opportunity.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace reduce {

// An opportunity to replace an id operand of an instruction with undef.
class ChangeOperandToUndefReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs the opportunity to replace operand |operand_index| of |inst|
  // with undef.
  ChangeOperandToUndefReductionOpportunity(opt::IRContext* context,
                                           opt::Instruction* inst,
                                           uint32_t operand_index)
      : context_(context),
        inst_(inst),
        operand_index_(operand_index),
        original_id_(inst->GetOperand(operand_index).words[0]) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::IRContext* context_;
  opt::Instruction* const inst_;
  const uint32_t operand_index_;
  const uint32_t original_id_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CHANGE_OPERAND_TO_UNDEF_REDUCTION_OPPORTUNITY_H_
