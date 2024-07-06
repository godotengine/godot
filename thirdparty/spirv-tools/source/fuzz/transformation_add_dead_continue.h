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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_CONTINUE_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_CONTINUE_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddDeadContinue : public Transformation {
 public:
  explicit TransformationAddDeadContinue(
      protobufs::TransformationAddDeadContinue message);

  TransformationAddDeadContinue(uint32_t from_block,
                                bool continue_condition_value,
                                std::vector<uint32_t> phi_id);

  // - |message_.from_block| must be the id of a block a in the given module.
  // - a must be contained in a loop with continue target b
  // - The continue target b must be dominated by the head of the loop in which
  //   it is contained
  // - b must not be the merge block of a selection construct
  // - if |message_.continue_condition_value| holds (does not hold) then
  //   OpConstantTrue (OpConstantFalse) must be present in the module
  // - |message_.phi_ids| must be a list of ids that are all available at
  //   |message_.from_block|
  // - a must end with an unconditional branch to some block c.
  // - replacing this branch with a conditional branch to b or c, with
  //   the boolean constant associated with |message_.continue_condition_value|
  //   as the condition, and the ids in |message_.phi_ids| used to extend any
  //   OpPhi instructions at b as a result of the edge from a, must maintain
  //   validity of the module.
  //   In particular, adding an edge from somewhere in the loop to the continue
  //   target must not prevent uses of ids in the continue target from being
  //   dominated by the definitions of those ids.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the terminator of a with a conditional branch to b or c.
  // The boolean constant associated with |message_.continue_condition_value| is
  // used as the condition, and the order of b and c is arranged such that
  // control is guaranteed to jump to c.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddDeadContinue message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_CONTINUE_H_
