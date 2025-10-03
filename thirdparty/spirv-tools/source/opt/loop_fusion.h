// Copyright (c) 2018 Google LLC.
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

#ifndef SOURCE_OPT_LOOP_FUSION_H_
#define SOURCE_OPT_LOOP_FUSION_H_

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_utils.h"
#include "source/opt/scalar_analysis.h"

namespace spvtools {
namespace opt {

class LoopFusion {
 public:
  LoopFusion(IRContext* context, Loop* loop_0, Loop* loop_1)
      : context_(context),
        loop_0_(loop_0),
        loop_1_(loop_1),
        containing_function_(loop_0->GetHeaderBlock()->GetParent()) {}

  // Checks if the |loop_0| and |loop_1| are compatible for fusion.
  // That means:
  //   * they both have one induction variable
  //   * they have the same upper and lower bounds
  //     - same initial value
  //     - same condition
  //   * they have the same update step
  //   * they are adjacent, with |loop_0| appearing before |loop_1|
  //   * there are no break/continue in either of them
  //   * they both have pre-header blocks (required for ScalarEvolutionAnalysis
  //     and dependence checking).
  bool AreCompatible();

  // Checks if compatible |loop_0| and |loop_1| are legal to fuse.
  // * fused loops do not have any dependencies with dependence distance greater
  //   than 0 that did not exist in the original loops.
  // * there are no function calls in the loops (could have side-effects)
  bool IsLegal();

  // Perform the actual fusion of |loop_0_| and |loop_1_|. The loops have to be
  // compatible and the fusion has to be legal.
  void Fuse();

 private:
  // Check that the initial values are the same.
  bool CheckInit();

  // Check that the conditions are the same.
  bool CheckCondition();

  // Check that the steps are the same.
  bool CheckStep();

  // Returns |true| if |instruction| is used in the continue or condition block
  // of |loop|.
  bool UsedInContinueOrConditionBlock(Instruction* instruction, Loop* loop);

  // Remove entries in |instructions| that are not used in the continue or
  // condition block of |loop|.
  void RemoveIfNotUsedContinueOrConditionBlock(
      std::vector<Instruction*>* instructions, Loop* loop);

  // Returns |true| if |instruction| is used in |loop|.
  bool IsUsedInLoop(Instruction* instruction, Loop* loop);

  // Returns |true| if |loop| has at least one barrier or function call.
  bool ContainsBarriersOrFunctionCalls(Loop* loop);

  // Get all instructions in the |loop| (except in the latch block) that have
  // the opcode |opcode|.
  std::pair<std::vector<Instruction*>, std::vector<Instruction*>>
  GetLoadsAndStoresInLoop(Loop* loop);

  // Given a vector of memory operations (OpLoad/OpStore), constructs a map from
  // variables to the loads/stores that those variables.
  std::map<Instruction*, std::vector<Instruction*>> LocationToMemOps(
      const std::vector<Instruction*>& mem_ops);

  IRContext* context_;

  // The original loops to be fused.
  Loop* loop_0_;
  Loop* loop_1_;

  // The function that contains |loop_0_| and |loop_1_|.
  Function* containing_function_ = nullptr;

  // The induction variables for |loop_0_| and |loop_1_|.
  Instruction* induction_0_ = nullptr;
  Instruction* induction_1_ = nullptr;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_FUSION_H_
