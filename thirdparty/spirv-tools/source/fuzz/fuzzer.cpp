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

#include "source/fuzz/fuzzer.h"

#include <cassert>
#include <memory>
#include <numeric>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass_add_access_chains.h"
#include "source/fuzz/fuzzer_pass_add_bit_instruction_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_composite_extract.h"
#include "source/fuzz/fuzzer_pass_add_composite_inserts.h"
#include "source/fuzz/fuzzer_pass_add_composite_types.h"
#include "source/fuzz/fuzzer_pass_add_copy_memory.h"
#include "source/fuzz/fuzzer_pass_add_dead_blocks.h"
#include "source/fuzz/fuzzer_pass_add_dead_breaks.h"
#include "source/fuzz/fuzzer_pass_add_dead_continues.h"
#include "source/fuzz/fuzzer_pass_add_equation_instructions.h"
#include "source/fuzz/fuzzer_pass_add_function_calls.h"
#include "source/fuzz/fuzzer_pass_add_global_variables.h"
#include "source/fuzz/fuzzer_pass_add_image_sample_unused_components.h"
#include "source/fuzz/fuzzer_pass_add_loads.h"
#include "source/fuzz/fuzzer_pass_add_local_variables.h"
#include "source/fuzz/fuzzer_pass_add_loop_preheaders.h"
#include "source/fuzz/fuzzer_pass_add_loops_to_create_int_constant_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_no_contraction_decorations.h"
#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_parameters.h"
#include "source/fuzz/fuzzer_pass_add_relaxed_decorations.h"
#include "source/fuzz/fuzzer_pass_add_stores.h"
#include "source/fuzz/fuzzer_pass_add_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_vector_shuffle_instructions.h"
#include "source/fuzz/fuzzer_pass_adjust_branch_weights.h"
#include "source/fuzz/fuzzer_pass_adjust_function_controls.h"
#include "source/fuzz/fuzzer_pass_adjust_loop_controls.h"
#include "source/fuzz/fuzzer_pass_adjust_memory_operands_masks.h"
#include "source/fuzz/fuzzer_pass_adjust_selection_controls.h"
#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"
#include "source/fuzz/fuzzer_pass_construct_composites.h"
#include "source/fuzz/fuzzer_pass_copy_objects.h"
#include "source/fuzz/fuzzer_pass_donate_modules.h"
#include "source/fuzz/fuzzer_pass_duplicate_regions_with_selections.h"
#include "source/fuzz/fuzzer_pass_expand_vector_reductions.h"
#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_inline_functions.h"
#include "source/fuzz/fuzzer_pass_interchange_signedness_of_integer_operands.h"
#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"
#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"
#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"
#include "source/fuzz/fuzzer_pass_merge_blocks.h"
#include "source/fuzz/fuzzer_pass_merge_function_returns.h"
#include "source/fuzz/fuzzer_pass_mutate_pointers.h"
#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"
#include "source/fuzz/fuzzer_pass_outline_functions.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_pass_permute_function_variables.h"
#include "source/fuzz/fuzzer_pass_permute_instructions.h"
#include "source/fuzz/fuzzer_pass_permute_phi_operands.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_down.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_up.h"
#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"
#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"
#include "source/fuzz/fuzzer_pass_replace_branches_from_dead_blocks_with_exits.h"
#include "source/fuzz/fuzzer_pass_replace_copy_memories_with_loads_stores.h"
#include "source/fuzz/fuzzer_pass_replace_copy_objects_with_stores_loads.h"
#include "source/fuzz/fuzzer_pass_replace_irrelevant_ids.h"
#include "source/fuzz/fuzzer_pass_replace_linear_algebra_instructions.h"
#include "source/fuzz/fuzzer_pass_replace_loads_stores_with_copy_memories.h"
#include "source/fuzz/fuzzer_pass_replace_opphi_ids_from_dead_predecessors.h"
#include "source/fuzz/fuzzer_pass_replace_opselects_with_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_replace_parameter_with_global.h"
#include "source/fuzz/fuzzer_pass_replace_params_with_struct.h"
#include "source/fuzz/fuzzer_pass_split_blocks.h"
#include "source/fuzz/fuzzer_pass_swap_commutable_operands.h"
#include "source/fuzz/fuzzer_pass_swap_conditional_branch_operands.h"
#include "source/fuzz/fuzzer_pass_swap_functions.h"
#include "source/fuzz/fuzzer_pass_toggle_access_chain_instruction.h"
#include "source/fuzz/fuzzer_pass_wrap_regions_in_selections.h"
#include "source/fuzz/fuzzer_pass_wrap_vector_synonym.h"
#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_recommender_standard.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/build_module.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

Fuzzer::Fuzzer(std::unique_ptr<opt::IRContext> ir_context,
               std::unique_ptr<TransformationContext> transformation_context,
               std::unique_ptr<FuzzerContext> fuzzer_context,
               MessageConsumer consumer,
               const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers,
               bool enable_all_passes,
               RepeatedPassStrategy repeated_pass_strategy,
               bool validate_after_each_fuzzer_pass,
               spv_validator_options validator_options,
               bool ignore_inapplicable_transformations /* = true */)
    : consumer_(std::move(consumer)),
      enable_all_passes_(enable_all_passes),
      validate_after_each_fuzzer_pass_(validate_after_each_fuzzer_pass),
      validator_options_(validator_options),
      num_repeated_passes_applied_(0),
      is_valid_(true),
      ir_context_(std::move(ir_context)),
      transformation_context_(std::move(transformation_context)),
      fuzzer_context_(std::move(fuzzer_context)),
      transformation_sequence_out_(),
      pass_instances_(),
      repeated_pass_recommender_(nullptr),
      repeated_pass_manager_(nullptr),
      final_passes_(),
      ignore_inapplicable_transformations_(
          ignore_inapplicable_transformations) {
  assert(ir_context_ && "IRContext is not initialized");
  assert(fuzzer_context_ && "FuzzerContext is not initialized");
  assert(transformation_context_ && "TransformationContext is not initialized");
  assert(fuzzerutil::IsValidAndWellFormed(ir_context_.get(), validator_options_,
                                          consumer_) &&
         "IRContext is invalid");

  // The following passes are likely to be very useful: many other passes
  // introduce synonyms, irrelevant ids and constants that these passes can work
  // with.  We thus enable them with high probability.
  MaybeAddRepeatedPass<FuzzerPassObfuscateConstants>(90, &pass_instances_);
  MaybeAddRepeatedPass<FuzzerPassApplyIdSynonyms>(90, &pass_instances_);
  MaybeAddRepeatedPass<FuzzerPassReplaceIrrelevantIds>(90, &pass_instances_);

  do {
    // Each call to MaybeAddRepeatedPass randomly decides whether the given pass
    // should be enabled, and adds an instance of the pass to |pass_instances|
    // if it is enabled.
    MaybeAddRepeatedPass<FuzzerPassAddAccessChains>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddBitInstructionSynonyms>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeExtract>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeInserts>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeTypes>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddCopyMemory>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBlocks>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBreaks>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddDeadContinues>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddEquationInstructions>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddFunctionCalls>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddGlobalVariables>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddImageSampleUnusedComponents>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddLoads>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddLocalVariables>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddLoopPreheaders>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddLoopsToCreateIntConstantSynonyms>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddOpPhiSynonyms>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddParameters>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddRelaxedDecorations>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddStores>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddSynonyms>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassAddVectorShuffleInstructions>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassConstructComposites>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassCopyObjects>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassDonateModules>(&pass_instances_,
                                                  donor_suppliers);
    MaybeAddRepeatedPass<FuzzerPassDuplicateRegionsWithSelections>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassExpandVectorReductions>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassFlattenConditionalBranches>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassInlineFunctions>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassInvertComparisonOperators>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassMakeVectorOperationsDynamic>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassMergeBlocks>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassMergeFunctionReturns>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassMutatePointers>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassOutlineFunctions>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPermuteBlocks>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPermuteFunctionParameters>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPermuteInstructions>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPropagateInstructionsDown>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPropagateInstructionsUp>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassPushIdsThroughVariables>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceAddsSubsMulsWithCarryingExtended>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceBranchesFromDeadBlocksWithExits>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyMemoriesWithLoadsStores>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyObjectsWithStoresLoads>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceLoadsStoresWithCopyMemories>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceParameterWithGlobal>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceLinearAlgebraInstructions>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpPhiIdsFromDeadPredecessors>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpSelectsWithConditionalBranches>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassReplaceParamsWithStruct>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassSplitBlocks>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassSwapBranchConditionalOperands>(
        &pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassWrapRegionsInSelections>(&pass_instances_);
    MaybeAddRepeatedPass<FuzzerPassWrapVectorSynonym>(&pass_instances_);
    // There is a theoretical possibility that no pass instances were created
    // until now; loop again if so.
  } while (pass_instances_.GetPasses().empty());

  repeated_pass_recommender_ = MakeUnique<RepeatedPassRecommenderStandard>(
      &pass_instances_, fuzzer_context_.get());
  repeated_pass_manager_ = RepeatedPassManager::Create(
      repeated_pass_strategy, fuzzer_context_.get(), &pass_instances_,
      repeated_pass_recommender_.get());

  MaybeAddFinalPass<FuzzerPassAdjustBranchWeights>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassAdjustFunctionControls>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassAdjustLoopControls>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassAdjustMemoryOperandsMasks>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassAdjustSelectionControls>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassAddNoContractionDecorations>(&final_passes_);
  if (!fuzzer_context_->IsWgslCompatible()) {
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/4214):
    //  this is disabled temporarily due to some issues in the Tint compiler.
    //  Enable it back when the issues are resolved.
    MaybeAddFinalPass<FuzzerPassInterchangeSignednessOfIntegerOperands>(
        &final_passes_);
  }
  MaybeAddFinalPass<FuzzerPassInterchangeZeroLikeConstants>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassPermuteFunctionVariables>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassPermutePhiOperands>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassSwapCommutableOperands>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassSwapFunctions>(&final_passes_);
  MaybeAddFinalPass<FuzzerPassToggleAccessChainInstruction>(&final_passes_);
}

Fuzzer::~Fuzzer() = default;

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddRepeatedPass(uint32_t percentage_chance_of_adding_pass,
                                  RepeatedPassInstances* pass_instances,
                                  Args&&... extra_args) {
  if (enable_all_passes_ ||
      fuzzer_context_->ChoosePercentage(percentage_chance_of_adding_pass)) {
    pass_instances->SetPass(MakeUnique<FuzzerPassT>(
        ir_context_.get(), transformation_context_.get(), fuzzer_context_.get(),
        &transformation_sequence_out_, ignore_inapplicable_transformations_,
        std::forward<Args>(extra_args)...));
  }
}

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddFinalPass(std::vector<std::unique_ptr<FuzzerPass>>* passes,
                               Args&&... extra_args) {
  if (enable_all_passes_ || fuzzer_context_->ChooseEven()) {
    passes->push_back(MakeUnique<FuzzerPassT>(
        ir_context_.get(), transformation_context_.get(), fuzzer_context_.get(),
        &transformation_sequence_out_, ignore_inapplicable_transformations_,
        std::forward<Args>(extra_args)...));
  }
}

bool Fuzzer::ApplyPassAndCheckValidity(FuzzerPass* pass) const {
  pass->Apply();
  return !validate_after_each_fuzzer_pass_ ||
         fuzzerutil::IsValidAndWellFormed(ir_context_.get(), validator_options_,
                                          consumer_);
}

opt::IRContext* Fuzzer::GetIRContext() { return ir_context_.get(); }

const protobufs::TransformationSequence& Fuzzer::GetTransformationSequence()
    const {
  return transformation_sequence_out_;
}

Fuzzer::Result Fuzzer::Run(uint32_t num_of_transformations_to_apply) {
  assert(is_valid_ && "The module was invalidated during the previous fuzzing");

  const auto initial_num_of_transformations =
      static_cast<uint32_t>(transformation_sequence_out_.transformation_size());

  auto status = Status::kComplete;
  do {
    if (!ApplyPassAndCheckValidity(
            repeated_pass_manager_->ChoosePass(transformation_sequence_out_))) {
      status = Status::kFuzzerPassLedToInvalidModule;
      break;
    }

    // Check that the module is small enough.
    if (ir_context_->module()->id_bound() >=
        fuzzer_context_->GetIdBoundLimit()) {
      status = Status::kModuleTooBig;
      break;
    }

    auto transformations_applied_so_far = static_cast<uint32_t>(
        transformation_sequence_out_.transformation_size());
    assert(transformations_applied_so_far >= initial_num_of_transformations &&
           "Number of transformations cannot decrease");

    // Check if we've already applied the maximum number of transformations.
    if (transformations_applied_so_far >=
        fuzzer_context_->GetTransformationLimit()) {
      status = Status::kTransformationLimitReached;
      break;
    }

    // Check that we've not got stuck (this can happen if the only available
    // fuzzer passes are not able to apply any transformations, or can only
    // apply very few transformations).
    if (num_repeated_passes_applied_ >=
        fuzzer_context_->GetTransformationLimit()) {
      status = Status::kFuzzerStuck;
      break;
    }

    // Check whether we've exceeded the number of transformations we can apply
    // in a single call to this method.
    if (num_of_transformations_to_apply != 0 &&
        transformations_applied_so_far - initial_num_of_transformations >=
            num_of_transformations_to_apply) {
      status = Status::kComplete;
      break;
    }

  } while (ShouldContinueRepeatedPasses(num_of_transformations_to_apply == 0));

  if (status != Status::kFuzzerPassLedToInvalidModule) {
    // We apply this transformations despite the fact that we might exceed
    // |num_of_transformations_to_apply|. This is not a problem for us since
    // these fuzzer passes are relatively simple yet might trigger some bugs.
    for (auto& pass : final_passes_) {
      if (!ApplyPassAndCheckValidity(pass.get())) {
        status = Status::kFuzzerPassLedToInvalidModule;
        break;
      }
    }
  }

  is_valid_ = status != Status::kFuzzerPassLedToInvalidModule;
  return {status, static_cast<uint32_t>(
                      transformation_sequence_out_.transformation_size()) !=
                      initial_num_of_transformations};
}

bool Fuzzer::ShouldContinueRepeatedPasses(
    bool continue_fuzzing_probabilistically) {
  if (continue_fuzzing_probabilistically) {
    // If we have applied T transformations so far, and the limit on the number
    // of transformations to apply is L (where T < L), the chance that we will
    // continue fuzzing is:
    //
    //     1 - T/(2*L)
    //
    // That is, the chance of continuing decreases as more transformations are
    // applied.  Using 2*L instead of L increases the number of transformations
    // that are applied on average.
    auto transformations_applied_so_far = static_cast<uint32_t>(
        transformation_sequence_out_.transformation_size());
    auto chance_of_continuing = static_cast<uint32_t>(
        100.0 *
        (1.0 - (static_cast<double>(transformations_applied_so_far) /
                (2.0 * static_cast<double>(
                           fuzzer_context_->GetTransformationLimit())))));
    if (!fuzzer_context_->ChoosePercentage(chance_of_continuing)) {
      // We have probabilistically decided to stop.
      return false;
    }
  }
  // Continue fuzzing!
  num_repeated_passes_applied_++;
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
