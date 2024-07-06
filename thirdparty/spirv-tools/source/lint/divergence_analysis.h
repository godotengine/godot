// Copyright (c) 2021 Google LLC.
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

#ifndef SOURCE_LINT_DIVERGENCE_ANALYSIS_H_
#define SOURCE_LINT_DIVERGENCE_ANALYSIS_H_

#include <cstdint>
#include <ostream>
#include <unordered_map>

#include "source/opt/basic_block.h"
#include "source/opt/control_dependence.h"
#include "source/opt/dataflow.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace lint {

// Computes the static divergence level for blocks (control flow) and values.
//
// A value is uniform if all threads that execute it are guaranteed to have the
// same value. Similarly, a value is partially uniform if this is true only
// within each derivative group. If neither apply, it is divergent.
//
// Control flow through a block is uniform if for any possible execution and
// point in time, all threads are executing it, or no threads are executing it.
// In particular, it is never possible for some threads to be inside the block
// and some threads not executing.
// TODO(kuhar): Clarify the difference between uniform, divergent, and
// partially-uniform execution in this analysis.
//
// Caveat:
// As we use control dependence to determine how divergence is propagated, this
// analysis can be overly permissive when the merge block for a conditional
// branch or switch is later than (strictly postdominates) the expected merge
// block, which is the immediate postdominator. However, this is not expected to
// be a problem in practice, given that SPIR-V is generally output by compilers
// and other automated tools, which would assign the earliest possible merge
// block, rather than written by hand.
// TODO(kuhar): Handle late merges.
class DivergenceAnalysis : public opt::ForwardDataFlowAnalysis {
 public:
  // The tightest (most uniform) level of divergence that can be determined
  // statically for a value or control flow for a block.
  //
  // The values are ordered such that A > B means that A is potentially more
  // divergent than B.
  // TODO(kuhar): Rename |PartiallyUniform' to something less confusing. For
  // example, the enum could be based on scopes.
  enum class DivergenceLevel {
    // The value or control flow is uniform across the entire invocation group.
    kUniform = 0,
    // The value or control flow is uniform across the derivative group, but not
    // the invocation group.
    kPartiallyUniform = 1,
    // The value or control flow is not statically uniform.
    kDivergent = 2,
  };

  DivergenceAnalysis(opt::IRContext& context)
      : ForwardDataFlowAnalysis(context, LabelPosition::kLabelsAtEnd) {}

  // Returns the divergence level for the given value (non-label instructions),
  // or control flow for the given block.
  DivergenceLevel GetDivergenceLevel(uint32_t id) {
    auto it = divergence_.find(id);
    if (it == divergence_.end()) {
      return DivergenceLevel::kUniform;
    }
    return it->second;
  }

  // Returns the divergence source for the given id. The following types of
  // divergence flows from A to B are possible:
  //
  // data -> data: A is used as an operand in the definition of B.
  // data -> control: B is control-dependent on a branch with condition A.
  // control -> data: B is a OpPhi instruction in which A is a block operand.
  // control -> control: B is control-dependent on A.
  uint32_t GetDivergenceSource(uint32_t id) {
    auto it = divergence_source_.find(id);
    if (it == divergence_source_.end()) {
      return 0;
    }
    return it->second;
  }

  // Returns the dependence source for the control dependence for the given id.
  // This only exists for data -> control edges.
  //
  // In other words, if block 2 is dependent on block 1 due to value 3 (e.g.
  // block 1 terminates with OpBranchConditional %3 %2 %4):
  // * GetDivergenceSource(2) = 3
  // * GetDivergenceDependenceSource(2) = 1
  //
  // Returns 0 if not applicable.
  uint32_t GetDivergenceDependenceSource(uint32_t id) {
    auto it = divergence_dependence_source_.find(id);
    if (it == divergence_dependence_source_.end()) {
      return 0;
    }
    return it->second;
  }

  void InitializeWorklist(opt::Function* function,
                          bool is_first_iteration) override {
    // Since |EnqueueSuccessors| is complete, we only need one pass.
    if (is_first_iteration) {
      Setup(function);
      opt::ForwardDataFlowAnalysis::InitializeWorklist(function, true);
    }
  }

  void EnqueueSuccessors(opt::Instruction* inst) override;

  VisitResult Visit(opt::Instruction* inst) override;

 private:
  VisitResult VisitBlock(uint32_t id);
  VisitResult VisitInstruction(opt::Instruction* inst);

  // Computes the divergence level for the result of the given instruction
  // based on the current state of the analysis. This is always an
  // underapproximation, which will be improved as the analysis proceeds.
  DivergenceLevel ComputeInstructionDivergence(opt::Instruction* inst);

  // Computes the divergence level for a variable, which is used for loads.
  DivergenceLevel ComputeVariableDivergence(opt::Instruction* var);

  // Initializes data structures for performing dataflow on the given function.
  void Setup(opt::Function* function);

  std::unordered_map<uint32_t, DivergenceLevel> divergence_;
  std::unordered_map<uint32_t, uint32_t> divergence_source_;
  std::unordered_map<uint32_t, uint32_t> divergence_dependence_source_;

  // Stores the result of following unconditional branches starting from the
  // given block. This is used to detect when reconvergence needs to be
  // accounted for.
  std::unordered_map<uint32_t, uint32_t> follow_unconditional_branches_;

  opt::ControlDependenceAnalysis cd_;
};

std::ostream& operator<<(std::ostream& os,
                         DivergenceAnalysis::DivergenceLevel level);

}  // namespace lint
}  // namespace spvtools

#endif  // SOURCE_LINT_DIVERGENCE_ANALYSIS_H_
