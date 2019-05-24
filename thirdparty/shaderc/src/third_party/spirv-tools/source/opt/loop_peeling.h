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

#ifndef SOURCE_OPT_LOOP_PEELING_H_
#define SOURCE_OPT_LOOP_PEELING_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_utils.h"
#include "source/opt/pass.h"
#include "source/opt/scalar_analysis.h"

namespace spvtools {
namespace opt {

// Utility class to perform the peeling of a given loop.
// The loop peeling transformation make a certain amount of a loop iterations to
// be executed either before (peel before) or after (peel after) the transformed
// loop.
//
// For peeling cases the transformation does the following steps:
//   - It clones the loop and inserts the cloned loop before the original loop;
//   - It connects all iterating values of the cloned loop with the
//     corresponding original loop values so that the second loop starts with
//     the appropriate values.
//   - It inserts a new induction variable "i" is inserted into the cloned that
//     starts with the value 0 and increment by step of one.
//
// The last step is specific to each case:
//   - Peel before: the transformation is to peel the "N" first iterations.
//     The exit condition of the cloned loop is changed so that the loop
//     exits when "i < N" becomes false. The original loop is then protected to
//     only execute if there is any iteration left to do.
//   - Peel after: the transformation is to peel the "N" last iterations,
//     then the exit condition of the cloned loop is changed so that the loop
//     exits when "i + N < max_iteration" becomes false, where "max_iteration"
//     is the upper bound of the loop. The cloned loop is then protected to
//     only execute if there is any iteration left to do no covered by the
//     second.
//
// To be peelable:
//   - The loop must be in LCSSA form;
//   - The loop must not contain any breaks;
//   - The loop must not have any ambiguous iterators updates (see
//     "CanPeelLoop").
// The method "CanPeelLoop" checks that those constrained are met.
class LoopPeeling {
 public:
  // LoopPeeling constructor.
  // |loop| is the loop to peel.
  // |loop_iteration_count| is the instruction holding the |loop| iteration
  // count, must be invariant for |loop| and must be of an int 32 type (signed
  // or unsigned).
  // |canonical_induction_variable| is an induction variable that can be used to
  // count the number of iterations, must be of the same type as
  // |loop_iteration_count| and start at 0 and increase by step of one at each
  // iteration. The value nullptr is interpreted as no suitable variable exists
  // and one will be created.
  LoopPeeling(Loop* loop, Instruction* loop_iteration_count,
              Instruction* canonical_induction_variable = nullptr)
      : context_(loop->GetContext()),
        loop_utils_(loop->GetContext(), loop),
        loop_(loop),
        loop_iteration_count_(!loop->IsInsideLoop(loop_iteration_count)
                                  ? loop_iteration_count
                                  : nullptr),
        int_type_(nullptr),
        original_loop_canonical_induction_variable_(
            canonical_induction_variable),
        canonical_induction_variable_(nullptr) {
    if (loop_iteration_count_) {
      int_type_ = context_->get_type_mgr()
                      ->GetType(loop_iteration_count_->type_id())
                      ->AsInteger();
      if (canonical_induction_variable_) {
        assert(canonical_induction_variable_->type_id() ==
                   loop_iteration_count_->type_id() &&
               "loop_iteration_count and canonical_induction_variable do not "
               "have the same type");
      }
    }
    GetIteratingExitValues();
  }

  // Returns true if the loop can be peeled.
  // To be peelable, all operation involved in the update of the loop iterators
  // must not dominates the exit condition. This restriction is a work around to
  // not miss compile code like:
  //
  //   for (int i = 0; i + 1 < N; i++) {}
  //   for (int i = 0; ++i < N; i++) {}
  //
  // The increment will happen before the test on the exit condition leading to
  // very look-a-like code.
  //
  // This restriction will not apply if a loop rotate is applied before (i.e.
  // becomes a do-while loop).
  bool CanPeelLoop() const {
    CFG& cfg = *context_->cfg();

    if (!loop_iteration_count_) {
      return false;
    }
    if (!int_type_) {
      return false;
    }
    if (int_type_->width() != 32) {
      return false;
    }
    if (!loop_->IsLCSSA()) {
      return false;
    }
    if (!loop_->GetMergeBlock()) {
      return false;
    }
    if (cfg.preds(loop_->GetMergeBlock()->id()).size() != 1) {
      return false;
    }
    if (!IsConditionCheckSideEffectFree()) {
      return false;
    }

    return !std::any_of(exit_value_.cbegin(), exit_value_.cend(),
                        [](std::pair<uint32_t, Instruction*> it) {
                          return it.second == nullptr;
                        });
  }

  // Moves the execution of the |factor| first iterations of the loop into a
  // dedicated loop.
  void PeelBefore(uint32_t factor);

  // Moves the execution of the |factor| last iterations of the loop into a
  // dedicated loop.
  void PeelAfter(uint32_t factor);

  // Returns the cloned loop.
  Loop* GetClonedLoop() { return cloned_loop_; }
  // Returns the original loop.
  Loop* GetOriginalLoop() { return loop_; }

 private:
  IRContext* context_;
  LoopUtils loop_utils_;
  // The original loop.
  Loop* loop_;
  // The initial |loop_| upper bound.
  Instruction* loop_iteration_count_;
  // The int type to use for the canonical_induction_variable_.
  analysis::Integer* int_type_;
  // The cloned loop.
  Loop* cloned_loop_;
  // This is set to true when the exit and back-edge branch instruction is the
  // same.
  bool do_while_form_;
  // The canonical induction variable from the original loop if it exists.
  Instruction* original_loop_canonical_induction_variable_;
  // The canonical induction variable of the cloned loop. The induction variable
  // is initialized to 0 and incremented by step of 1.
  Instruction* canonical_induction_variable_;
  // Map between loop iterators and exit values. Loop iterators
  std::unordered_map<uint32_t, Instruction*> exit_value_;

  // Duplicate |loop_| and place the new loop before the cloned loop. Iterating
  // values from the cloned loop are then connected to the original loop as
  // initializer.
  void DuplicateAndConnectLoop(LoopUtils::LoopCloningResult* clone_results);

  // Insert the canonical induction variable into the first loop as a simplified
  // counter.
  void InsertCanonicalInductionVariable(
      LoopUtils::LoopCloningResult* clone_results);

  // Fixes the exit condition of the before loop. The function calls
  // |condition_builder| to get the condition to use in the conditional branch
  // of the loop exit. The loop will be exited if the condition evaluate to
  // true. |condition_builder| takes an Instruction* that represent the
  // insertion point.
  void FixExitCondition(
      const std::function<uint32_t(Instruction*)>& condition_builder);

  // Gathers all operations involved in the update of |iterator| into
  // |operations|.
  void GetIteratorUpdateOperations(
      const Loop* loop, Instruction* iterator,
      std::unordered_set<Instruction*>* operations);

  // Gathers exiting iterator values. The function builds a map between each
  // iterating value in the loop (a phi instruction in the loop header) and its
  // SSA value when it exit the loop. If no exit value can be accurately found,
  // it is map to nullptr (see comment on CanPeelLoop).
  void GetIteratingExitValues();

  // Returns true if a for-loop has no instruction with effects before the
  // condition check.
  bool IsConditionCheckSideEffectFree() const;

  // Creates a new basic block and insert it between |bb| and the predecessor of
  // |bb|.
  BasicBlock* CreateBlockBefore(BasicBlock* bb);

  // Inserts code to only execute |loop| only if the given |condition| is true.
  // |if_merge| is a suitable basic block to be used by the if condition as
  // merge block.
  // The function returns the if block protecting the loop.
  BasicBlock* ProtectLoop(Loop* loop, Instruction* condition,
                          BasicBlock* if_merge);
};

// Implements a loop peeling optimization.
// For each loop, the pass will try to peel it if there is conditions that
// are true for the "N" first or last iterations of the loop.
// To avoid code size explosion, too large loops will not be peeled.
class LoopPeelingPass : public Pass {
 public:
  // Describes the peeling direction.
  enum class PeelDirection {
    kNone,    // Cannot peel
    kBefore,  // Can peel before
    kAfter    // Can peel last
  };

  // Holds some statistics about peeled function.
  struct LoopPeelingStats {
    std::vector<std::tuple<const Loop*, PeelDirection, uint32_t>> peeled_loops_;
  };

  LoopPeelingPass(LoopPeelingStats* stats = nullptr) : stats_(stats) {}

  // Sets the loop peeling growth threshold. If the code size increase is above
  // |code_grow_threshold|, the loop will not be peeled. The code size is
  // measured in terms of SPIR-V instructions.
  static void SetLoopPeelingThreshold(size_t code_grow_threshold) {
    code_grow_threshold_ = code_grow_threshold;
  }

  // Returns the loop peeling code growth threshold.
  static size_t GetLoopPeelingThreshold() { return code_grow_threshold_; }

  const char* name() const override { return "loop-peeling"; }

  // Processes the given |module|. Returns Status::Failure if errors occur when
  // processing. Returns the corresponding Status::Success if processing is
  // succesful to indicate whether changes have been made to the modue.
  Pass::Status Process() override;

 private:
  // Describes the peeling direction.
  enum class CmpOperator {
    kLT,  // less than
    kGT,  // greater than
    kLE,  // less than or equal
    kGE,  // greater than or equal
  };

  class LoopPeelingInfo {
   public:
    using Direction = std::pair<PeelDirection, uint32_t>;

    LoopPeelingInfo(Loop* loop, size_t loop_max_iterations,
                    ScalarEvolutionAnalysis* scev_analysis)
        : context_(loop->GetContext()),
          loop_(loop),
          scev_analysis_(scev_analysis),
          loop_max_iterations_(loop_max_iterations) {}

    // Returns by how much and to which direction a loop should be peeled to
    // make the conditional branch of the basic block |bb| an unconditional
    // branch. If |bb|'s terminator is not a conditional branch or the condition
    // is not workable then it returns PeelDirection::kNone and a 0 factor.
    Direction GetPeelingInfo(BasicBlock* bb) const;

   private:
    // Returns the id of the loop invariant operand of the conditional
    // expression |condition|. It returns if no operand is invariant.
    uint32_t GetFirstLoopInvariantOperand(Instruction* condition) const;
    // Returns the id of the non loop invariant operand of the conditional
    // expression |condition|. It returns if all operands are invariant.
    uint32_t GetFirstNonLoopInvariantOperand(Instruction* condition) const;

    // Returns the value of |rec| at the first loop iteration.
    SExpression GetValueAtFirstIteration(SERecurrentNode* rec) const;
    // Returns the value of |rec| at the given |iteration|.
    SExpression GetValueAtIteration(SERecurrentNode* rec,
                                    int64_t iteration) const;
    // Returns the value of |rec| at the last loop iteration.
    SExpression GetValueAtLastIteration(SERecurrentNode* rec) const;

    bool EvalOperator(CmpOperator cmp_op, SExpression lhs, SExpression rhs,
                      bool* result) const;

    Direction HandleEquality(SExpression lhs, SExpression rhs) const;
    Direction HandleInequality(CmpOperator cmp_op, SExpression lhs,
                               SERecurrentNode* rhs) const;

    static Direction GetNoneDirection() {
      return Direction{LoopPeelingPass::PeelDirection::kNone, 0};
    }
    IRContext* context_;
    Loop* loop_;
    ScalarEvolutionAnalysis* scev_analysis_;
    size_t loop_max_iterations_;
  };
  // Peel profitable loops in |f|.
  bool ProcessFunction(Function* f);
  // Peel |loop| if profitable.
  std::pair<bool, Loop*> ProcessLoop(Loop* loop, CodeMetrics* loop_size);

  static size_t code_grow_threshold_;
  LoopPeelingStats* stats_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_PEELING_H_
