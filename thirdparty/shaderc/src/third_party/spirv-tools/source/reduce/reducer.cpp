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

#include "source/reduce/reducer.h"

#include <cassert>
#include <sstream>

#include "source/reduce/merge_blocks_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_const_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_dominating_id_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_undef_reduction_opportunity_finder.h"
#include "source/reduce/remove_block_reduction_opportunity_finder.h"
#include "source/reduce/remove_function_reduction_opportunity_finder.h"
#include "source/reduce/remove_opname_instruction_reduction_opportunity_finder.h"
#include "source/reduce/remove_selection_reduction_opportunity_finder.h"
#include "source/reduce/remove_unreferenced_instruction_reduction_opportunity_finder.h"
#include "source/reduce/structured_loop_to_selection_reduction_opportunity_finder.h"
#include "source/spirv_reducer_options.h"

namespace spvtools {
namespace reduce {

struct Reducer::Impl {
  explicit Impl(spv_target_env env) : target_env(env) {}

  bool ReachedStepLimit(uint32_t current_step,
                        spv_const_reducer_options options);

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
  InterestingnessFunction interestingness_function;
  std::vector<std::unique_ptr<ReductionPass>> passes;
};

Reducer::Reducer(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Reducer::~Reducer() = default;

void Reducer::SetMessageConsumer(MessageConsumer c) {
  for (auto& pass : impl_->passes) {
    pass->SetMessageConsumer(c);
  }
  impl_->consumer = std::move(c);
}

void Reducer::SetInterestingnessFunction(
    Reducer::InterestingnessFunction interestingness_function) {
  impl_->interestingness_function = std::move(interestingness_function);
}

Reducer::ReductionResultStatus Reducer::Run(
    std::vector<uint32_t>&& binary_in, std::vector<uint32_t>* binary_out,
    spv_const_reducer_options options,
    spv_validator_options validator_options) const {
  std::vector<uint32_t> current_binary(std::move(binary_in));

  spvtools::SpirvTools tools(impl_->target_env);
  assert(tools.IsValid() && "Failed to create SPIRV-Tools interface");

  // Keeps track of how many reduction attempts have been tried.  Reduction
  // bails out if this reaches a given limit.
  uint32_t reductions_applied = 0;

  // Initial state should be valid.
  if (!tools.Validate(&current_binary[0], current_binary.size(),
                      validator_options)) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial binary is invalid; stopping.");
    return Reducer::ReductionResultStatus::kInitialStateInvalid;
  }

  // Initial state should be interesting.
  if (!impl_->interestingness_function(current_binary, reductions_applied)) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial state was not interesting; stopping.");
    return Reducer::ReductionResultStatus::kInitialStateNotInteresting;
  }

  // Determines whether, on completing one round of reduction passes, it is
  // worthwhile trying a further round.
  bool another_round_worthwhile = true;

  // Apply round after round of reduction passes until we hit the reduction
  // step limit, or deem that another round is not going to be worthwhile.
  while (!impl_->ReachedStepLimit(reductions_applied, options) &&
         another_round_worthwhile) {
    // At the start of a round of reduction passes, assume another round will
    // not be worthwhile unless we find evidence to the contrary.
    another_round_worthwhile = false;

    // Iterate through the available passes
    for (auto& pass : impl_->passes) {
      // If this pass hasn't reached its minimum granularity then it's
      // worth eventually doing another round of reductions, in order to
      // try this pass at a finer granularity.
      another_round_worthwhile |= !pass->ReachedMinimumGranularity();

      // Keep applying this pass at its current granularity until it stops
      // working or we hit the reduction step limit.
      impl_->consumer(SPV_MSG_INFO, nullptr, {},
                      ("Trying pass " + pass->GetName() + ".").c_str());
      do {
        auto maybe_result = pass->TryApplyReduction(current_binary);
        if (maybe_result.empty()) {
          // For this round, the pass has no more opportunities (chunks) to
          // apply, so move on to the next pass.
          impl_->consumer(
              SPV_MSG_INFO, nullptr, {},
              ("Pass " + pass->GetName() + " did not make a reduction step.")
                  .c_str());
          break;
        }
        bool interesting = false;
        std::stringstream stringstream;
        reductions_applied++;
        stringstream << "Pass " << pass->GetName() << " made reduction step "
                     << reductions_applied << ".";
        impl_->consumer(SPV_MSG_INFO, nullptr, {},
                        (stringstream.str().c_str()));
        if (!tools.Validate(&maybe_result[0], maybe_result.size(),
                            validator_options)) {
          // The reduction step went wrong and an invalid binary was produced.
          // By design, this shouldn't happen; this is a safeguard to stop an
          // invalid binary from being regarded as interesting.
          impl_->consumer(SPV_MSG_INFO, nullptr, {},
                          "Reduction step produced an invalid binary.");
          if (options->fail_on_validation_error) {
            return Reducer::ReductionResultStatus::kStateInvalid;
          }
        } else if (impl_->interestingness_function(maybe_result,
                                                   reductions_applied)) {
          // Success!  The binary produced by this reduction step is
          // interesting, so make it the binary of interest henceforth, and
          // note that it's worth doing another round of reduction passes.
          impl_->consumer(SPV_MSG_INFO, nullptr, {},
                          "Reduction step succeeded.");
          current_binary = std::move(maybe_result);
          interesting = true;
          another_round_worthwhile = true;
        }
        // We must call this before the next call to TryApplyReduction.
        pass->NotifyInteresting(interesting);
        // Bail out if the reduction step limit has been reached.
      } while (!impl_->ReachedStepLimit(reductions_applied, options));
    }
  }

  *binary_out = std::move(current_binary);

  // Report whether reduction completed, or bailed out early due to reaching
  // the step limit.
  if (impl_->ReachedStepLimit(reductions_applied, options)) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Reached reduction step limit; stopping.");
    return Reducer::ReductionResultStatus::kReachedStepLimit;
  }
  impl_->consumer(SPV_MSG_INFO, nullptr, {}, "No more to reduce; stopping.");
  return Reducer::ReductionResultStatus::kComplete;
}

void Reducer::AddDefaultReductionPasses() {
  AddReductionPass(spvtools::MakeUnique<
                   RemoveOpNameInstructionReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<OperandToUndefReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<OperandToConstReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<OperandToDominatingIdReductionOpportunityFinder>());
  AddReductionPass(spvtools::MakeUnique<
                   RemoveUnreferencedInstructionReductionOpportunityFinder>());
  AddReductionPass(spvtools::MakeUnique<
                   StructuredLoopToSelectionReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<MergeBlocksReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<RemoveFunctionReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<RemoveBlockReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<RemoveSelectionReductionOpportunityFinder>());
}

void Reducer::AddReductionPass(
    std::unique_ptr<ReductionOpportunityFinder>&& finder) {
  impl_->passes.push_back(spvtools::MakeUnique<ReductionPass>(
      impl_->target_env, std::move(finder)));
}

bool Reducer::Impl::ReachedStepLimit(uint32_t current_step,
                                     spv_const_reducer_options options) {
  return current_step >= options->step_limit;
}

}  // namespace reduce
}  // namespace spvtools
