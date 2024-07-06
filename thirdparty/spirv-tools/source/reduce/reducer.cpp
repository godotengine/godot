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

#include "source/reduce/conditional_branch_to_simple_conditional_branch_opportunity_finder.h"
#include "source/reduce/merge_blocks_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_const_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_dominating_id_reduction_opportunity_finder.h"
#include "source/reduce/operand_to_undef_reduction_opportunity_finder.h"
#include "source/reduce/remove_block_reduction_opportunity_finder.h"
#include "source/reduce/remove_function_reduction_opportunity_finder.h"
#include "source/reduce/remove_selection_reduction_opportunity_finder.h"
#include "source/reduce/remove_unused_instruction_reduction_opportunity_finder.h"
#include "source/reduce/remove_unused_struct_member_reduction_opportunity_finder.h"
#include "source/reduce/simple_conditional_branch_to_branch_opportunity_finder.h"
#include "source/reduce/structured_construct_to_block_reduction_opportunity_finder.h"
#include "source/reduce/structured_loop_to_selection_reduction_opportunity_finder.h"
#include "source/spirv_reducer_options.h"

namespace spvtools {
namespace reduce {

Reducer::Reducer(spv_target_env target_env) : target_env_(target_env) {}

Reducer::~Reducer() = default;

void Reducer::SetMessageConsumer(MessageConsumer c) {
  for (auto& pass : passes_) {
    pass->SetMessageConsumer(c);
  }
  for (auto& pass : cleanup_passes_) {
    pass->SetMessageConsumer(c);
  }
  consumer_ = std::move(c);
}

void Reducer::SetInterestingnessFunction(
    Reducer::InterestingnessFunction interestingness_function) {
  interestingness_function_ = std::move(interestingness_function);
}

Reducer::ReductionResultStatus Reducer::Run(
    const std::vector<uint32_t>& binary_in, std::vector<uint32_t>* binary_out,
    spv_const_reducer_options options,
    spv_validator_options validator_options) {
  std::vector<uint32_t> current_binary(binary_in);

  spvtools::SpirvTools tools(target_env_);
  assert(tools.IsValid() && "Failed to create SPIRV-Tools interface");

  // Keeps track of how many reduction attempts have been tried.  Reduction
  // bails out if this reaches a given limit.
  uint32_t reductions_applied = 0;

  // Initial state should be valid.
  if (!tools.Validate(&current_binary[0], current_binary.size(),
                      validator_options)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Initial binary is invalid; stopping.");
    return Reducer::ReductionResultStatus::kInitialStateInvalid;
  }

  // Initial state should be interesting.
  if (!interestingness_function_(current_binary, reductions_applied)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Initial state was not interesting; stopping.");
    return Reducer::ReductionResultStatus::kInitialStateNotInteresting;
  }

  Reducer::ReductionResultStatus result =
      RunPasses(&passes_, options, validator_options, tools, &current_binary,
                &reductions_applied);

  if (result == Reducer::ReductionResultStatus::kComplete) {
    // Cleanup passes.
    result = RunPasses(&cleanup_passes_, options, validator_options, tools,
                       &current_binary, &reductions_applied);
  }

  if (result == Reducer::ReductionResultStatus::kComplete) {
    consumer_(SPV_MSG_INFO, nullptr, {}, "No more to reduce; stopping.");
  }

  // Even if the reduction has failed by this point (e.g. due to producing an
  // invalid binary), we still update the output binary for better debugging.
  *binary_out = std::move(current_binary);

  return result;
}

void Reducer::AddDefaultReductionPasses() {
  AddReductionPass(
      spvtools::MakeUnique<RemoveUnusedInstructionReductionOpportunityFinder>(
          false));
  AddReductionPass(
      spvtools::MakeUnique<OperandToUndefReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<OperandToConstReductionOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<OperandToDominatingIdReductionOpportunityFinder>());
  AddReductionPass(spvtools::MakeUnique<
                   StructuredConstructToBlockReductionOpportunityFinder>());
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
  AddReductionPass(
      spvtools::MakeUnique<
          ConditionalBranchToSimpleConditionalBranchOpportunityFinder>());
  AddReductionPass(
      spvtools::MakeUnique<SimpleConditionalBranchToBranchOpportunityFinder>());
  AddReductionPass(spvtools::MakeUnique<
                   RemoveUnusedStructMemberReductionOpportunityFinder>());

  // Cleanup passes.

  AddCleanupReductionPass(
      spvtools::MakeUnique<RemoveUnusedInstructionReductionOpportunityFinder>(
          true));
}

void Reducer::AddReductionPass(
    std::unique_ptr<ReductionOpportunityFinder> finder) {
  passes_.push_back(
      spvtools::MakeUnique<ReductionPass>(target_env_, std::move(finder)));
}

void Reducer::AddCleanupReductionPass(
    std::unique_ptr<ReductionOpportunityFinder> finder) {
  cleanup_passes_.push_back(
      spvtools::MakeUnique<ReductionPass>(target_env_, std::move(finder)));
}

bool Reducer::ReachedStepLimit(uint32_t current_step,
                               spv_const_reducer_options options) {
  return current_step >= options->step_limit;
}

Reducer::ReductionResultStatus Reducer::RunPasses(
    std::vector<std::unique_ptr<ReductionPass>>* passes,
    spv_const_reducer_options options, spv_validator_options validator_options,
    const SpirvTools& tools, std::vector<uint32_t>* current_binary,
    uint32_t* const reductions_applied) {
  // Determines whether, on completing one round of reduction passes, it is
  // worthwhile trying a further round.
  bool another_round_worthwhile = true;

  // Apply round after round of reduction passes until we hit the reduction
  // step limit, or deem that another round is not going to be worthwhile.
  while (!ReachedStepLimit(*reductions_applied, options) &&
         another_round_worthwhile) {
    // At the start of a round of reduction passes, assume another round will
    // not be worthwhile unless we find evidence to the contrary.
    another_round_worthwhile = false;

    // Iterate through the available passes.
    for (auto& pass : *passes) {
      // If this pass hasn't reached its minimum granularity then it's
      // worth eventually doing another round of reductions, in order to
      // try this pass at a finer granularity.
      another_round_worthwhile |= !pass->ReachedMinimumGranularity();

      // Keep applying this pass at its current granularity until it stops
      // working or we hit the reduction step limit.
      consumer_(SPV_MSG_INFO, nullptr, {},
                ("Trying pass " + pass->GetName() + ".").c_str());
      do {
        auto maybe_result =
            pass->TryApplyReduction(*current_binary, options->target_function);
        if (maybe_result.empty()) {
          // For this round, the pass has no more opportunities (chunks) to
          // apply, so move on to the next pass.
          consumer_(
              SPV_MSG_INFO, nullptr, {},
              ("Pass " + pass->GetName() + " did not make a reduction step.")
                  .c_str());
          break;
        }
        bool interesting = false;
        std::stringstream stringstream;
        (*reductions_applied)++;
        stringstream << "Pass " << pass->GetName() << " made reduction step "
                     << *reductions_applied << ".";
        consumer_(SPV_MSG_INFO, nullptr, {}, (stringstream.str().c_str()));
        if (!tools.Validate(&maybe_result[0], maybe_result.size(),
                            validator_options)) {
          // The reduction step went wrong and an invalid binary was produced.
          // By design, this shouldn't happen; this is a safeguard to stop an
          // invalid binary from being regarded as interesting.
          consumer_(SPV_MSG_INFO, nullptr, {},
                    "Reduction step produced an invalid binary.");
          if (options->fail_on_validation_error) {
            // In this mode, we fail, so we update the current binary so it is
            // output for debugging.
            *current_binary = std::move(maybe_result);
            return Reducer::ReductionResultStatus::kStateInvalid;
          }
        } else if (interestingness_function_(maybe_result,
                                             *reductions_applied)) {
          // Success!  The binary produced by this reduction step is
          // interesting, so make it the binary of interest henceforth, and
          // note that it's worth doing another round of reduction passes.
          consumer_(SPV_MSG_INFO, nullptr, {}, "Reduction step succeeded.");
          *current_binary = std::move(maybe_result);
          interesting = true;
          another_round_worthwhile = true;
        }
        // We must call this before the next call to TryApplyReduction.
        pass->NotifyInteresting(interesting);
        // Bail out if the reduction step limit has been reached.
      } while (!ReachedStepLimit(*reductions_applied, options));
    }
  }

  // Report whether reduction completed, or bailed out early due to reaching
  // the step limit.
  if (ReachedStepLimit(*reductions_applied, options)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Reached reduction step limit; stopping.");
    return Reducer::ReductionResultStatus::kReachedStepLimit;
  }

  // The passes completed successfully, although we may still run more passes.
  return Reducer::ReductionResultStatus::kComplete;
}

}  // namespace reduce
}  // namespace spvtools
