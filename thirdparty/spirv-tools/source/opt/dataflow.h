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

#ifndef SOURCE_OPT_DATAFLOW_H_
#define SOURCE_OPT_DATAFLOW_H_

#include <queue>
#include <unordered_map>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

// Generic data-flow analysis.
// Maintains a worklist of instructions to process and processes them in a
// specified order. See also ForwardDataFlowAnalysis, which is specialized for
// forward data-flow analysis.
class DataFlowAnalysis {
 public:
  // The result of a |Visit| operation on an instruction.
  // This is used to determine when analysis has reached a fixpoint.
  enum class VisitResult {
    // The analysis result for this instruction has changed.
    // This means that any instructions that depend on it (its successors) must
    // be recomputed.
    kResultChanged,
    // The analysis result for this instruction has not changed.
    // When all visit operations return |kResultFixed|, the analysis has reached
    // a fixpoint (converged).
    kResultFixed,
  };

  virtual ~DataFlowAnalysis() {}

  // Run this analysis on a given function.
  // For analyses which work interprocedurally, |function| may be ignored.
  void Run(Function* function);

 protected:
  DataFlowAnalysis(IRContext& context) : context_(context) {}

  // Initialize the worklist for a given function.
  // |is_first_iteration| is true on the first call to |Run| and false
  // afterwards. All subsequent runs are only necessary to check if the analysis
  // has converged; if |EnqueueSuccessors| is complete, |InitializeWorklist|
  // should do nothing after the first iteration.
  virtual void InitializeWorklist(Function* function,
                                  bool is_first_iteration) = 0;

  // Enqueues the successors (instructions which use the analysis result) of
  // |inst|. This is not required to be complete, but convergence is faster when
  // it is. This is called whenever |Visit| returns |kResultChanged|.
  virtual void EnqueueSuccessors(Instruction* inst) = 0;

  // Visits the given instruction, recomputing the analysis result. This is
  // called once per instruction queued in |InitializeWorklist| and afterward
  // when a predecessor is changed, through |EnqueueSuccessors|.
  virtual VisitResult Visit(Instruction* inst) = 0;

  // Enqueues the given instruction to be visited. Ignored if already in the
  // worklist.
  bool Enqueue(Instruction* inst);

  IRContext& context() { return context_; }

 private:
  // Runs one pass, calling |InitializeWorklist| and then iterating through the
  // worklist until all fixed.
  VisitResult RunOnce(Function* function, bool is_first_iteration);

  IRContext& context_;
  std::unordered_map<Instruction*, bool> on_worklist_;
  // The worklist, which contains the list of instructions to be visited.
  //
  // The choice of data structure was influenced by the data in "Iterative
  // Data-flow Analysis, Revisited" (Cooper et al, 2002).
  // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.1549&rep=rep1&type=pdf
  // The paper shows that the overall performance benefit of a priority queue
  // over a regular queue or stack is relatively small (or negative).
  //
  // A queue has the advantage that nodes are visited in the same order they are
  // enqueued, which relieves the analysis from inserting nodes "backwards", for
  // example in worklist initialization. Also, as the paper claims that sorting
  // successors does not improve runtime, we can use a single queue which is
  // modified during iteration.
  std::queue<Instruction*> worklist_;
};

// A generic data flow analysis, specialized for forward analysis.
class ForwardDataFlowAnalysis : public DataFlowAnalysis {
 public:
  // Indicates where labels should be in the worklist RPO ordering.
  enum class LabelPosition {
    // Labels should be placed at the beginning of their blocks.
    kLabelsAtBeginning,
    // Labels should be placed at the end of their blocks.
    kLabelsAtEnd,
    // Labels should not be in the worklist.
    kNoLabels,
    // Only labels should be placed in the worklist.
    kLabelsOnly,
  };

  ForwardDataFlowAnalysis(IRContext& context, LabelPosition label_position)
      : DataFlowAnalysis(context), label_position_(label_position) {}

 protected:
  // Initializes the worklist in reverse postorder, regardless of
  // |is_first_iteration|. Labels are placed according to the label position
  // specified in the constructor.
  void InitializeWorklist(Function* function, bool is_first_iteration) override;

  // Enqueues the users and block successors of the given instruction.
  // See |EnqueueUsers| and |EnqueueBlockSuccessors|.
  void EnqueueSuccessors(Instruction* inst) override {
    EnqueueUsers(inst);
    EnqueueBlockSuccessors(inst);
  }

  // Enqueues the users of the given instruction.
  void EnqueueUsers(Instruction* inst);

  // Enqueues the labels of the successors of the block corresponding to the
  // given label instruction. Does nothing for other instructions.
  void EnqueueBlockSuccessors(Instruction* inst);

 private:
  LabelPosition label_position_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DATAFLOW_H_
