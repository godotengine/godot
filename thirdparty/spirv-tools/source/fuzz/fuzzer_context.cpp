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

#include "source/fuzz/fuzzer_context.h"

#include <cmath>

namespace spvtools {
namespace fuzz {

namespace {

// An offset between the module's id bound and the minimum fresh id.
//
// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541): consider
//  the case where the maximum id bound is reached.
const uint32_t kIdBoundGap = 100;

// Limits to help control the overall fuzzing process and rein in individual
// fuzzer passes.
const uint32_t kIdBoundLimit = 50000;
const uint32_t kTransformationLimit = 2000;

// Default <minimum, maximum> pairs of probabilities for applying various
// transformations. All values are percentages. Keep them in alphabetical order.
const std::pair<uint32_t, uint32_t>
    kChanceOfAcceptingRepeatedPassRecommendation = {50, 80};
const std::pair<uint32_t, uint32_t> kChanceOfAddingAccessChain = {5, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingAnotherPassToPassLoop = {50,
                                                                            90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingAnotherStructField = {20,
                                                                         90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingArrayOrStructType = {20, 90};
const std::pair<uint32_t, uint32_t> KChanceOfAddingAtomicLoad = {30, 90};
const std::pair<uint32_t, uint32_t> KChanceOfAddingAtomicStore = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingBitInstructionSynonym = {5,
                                                                            20};
const std::pair<uint32_t, uint32_t>
    kChanceOfAddingBothBranchesWhenReplacingOpSelect = {40, 60};
const std::pair<uint32_t, uint32_t> kChanceOfAddingCompositeExtract = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingCompositeInsert = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingCopyMemory = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingDeadBlock = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingDeadBreak = {5, 80};
const std::pair<uint32_t, uint32_t> kChanceOfAddingDeadContinue = {5, 80};
const std::pair<uint32_t, uint32_t> kChanceOfAddingEquationInstruction = {5,
                                                                          90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingGlobalVariable = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingImageSampleUnusedComponents =
    {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingLoad = {5, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingLocalVariable = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingLoopPreheader = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingMatrixType = {20, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAddingNoContractionDecoration = {
    5, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAddingOpPhiSynonym = {5, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAddingParameters = {5, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAddingRelaxedDecoration = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAddingStore = {5, 50};
const std::pair<uint32_t, uint32_t> kChanceOfAddingSynonyms = {20, 50};
const std::pair<uint32_t, uint32_t>
    kChanceOfAddingTrueBranchWhenReplacingOpSelect = {40, 60};
const std::pair<uint32_t, uint32_t> kChanceOfAddingVectorType = {20, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAddingVectorShuffle = {20, 70};
const std::pair<uint32_t, uint32_t> kChanceOfAdjustingBranchWeights = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAdjustingFunctionControl = {20,
                                                                         70};
const std::pair<uint32_t, uint32_t> kChanceOfAdjustingLoopControl = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfAdjustingMemoryOperandsMask = {20,
                                                                            90};
const std::pair<uint32_t, uint32_t> kChanceOfAdjustingSelectionControl = {20,
                                                                          90};
const std::pair<uint32_t, uint32_t> kChanceOfCallingFunction = {1, 10};
const std::pair<uint32_t, uint32_t> kChanceOfChoosingStructTypeVsArrayType = {
    20, 80};
const std::pair<uint32_t, uint32_t> kChanceOfChoosingWorkgroupStorageClass = {
    50, 50};
const std::pair<uint32_t, uint32_t> kChanceOfConstructingComposite = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfCopyingObject = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfCreatingIntSynonymsUsingLoops = {
    5, 10};
const std::pair<uint32_t, uint32_t> kChanceOfDonatingAdditionalModule = {5, 50};
const std::pair<uint32_t, uint32_t> kChanceOfDuplicatingRegionWithSelection = {
    20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfExpandingVectorReduction = {20,
                                                                         90};
const std::pair<uint32_t, uint32_t> kChanceOfFlatteningConditionalBranch = {45,
                                                                            95};
const std::pair<uint32_t, uint32_t> kChanceOfGoingDeeperToExtractComposite = {
    30, 70};
const std::pair<uint32_t, uint32_t> kChanceOfGoingDeeperToInsertInComposite = {
    30, 70};
const std::pair<uint32_t, uint32_t> kChanceOfGoingDeeperWhenMakingAccessChain =
    {50, 95};
const std::pair<uint32_t, uint32_t>
    kChanceOfHavingTwoBlocksInLoopToCreateIntSynonym = {50, 80};
const std::pair<uint32_t, uint32_t> kChanceOfInliningFunction = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfInterchangingZeroLikeConstants = {
    10, 90};
const std::pair<uint32_t, uint32_t>
    kChanceOfInterchangingSignednessOfIntegerOperands = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfInvertingComparisonOperators = {
    20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfMakingDonorLivesafe = {40, 60};
const std::pair<uint32_t, uint32_t> kChanceOfMakingVectorOperationDynamic = {
    20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfMergingBlocks = {20, 95};
const std::pair<uint32_t, uint32_t> kChanceOfMergingFunctionReturns = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfMovingBlockDown = {20, 50};
const std::pair<uint32_t, uint32_t> kChanceOfMutatingPointer = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfObfuscatingConstant = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfOutliningFunction = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfPermutingFunctionVariables = {30,
                                                                           90};
const std::pair<uint32_t, uint32_t> kChanceOfPermutingInstructions = {20, 70};
const std::pair<uint32_t, uint32_t> kChanceOfPermutingParameters = {30, 90};
const std::pair<uint32_t, uint32_t> kChanceOfPermutingPhiOperands = {30, 90};
const std::pair<uint32_t, uint32_t> kChanceOfPropagatingInstructionsDown = {20,
                                                                            70};
const std::pair<uint32_t, uint32_t> kChanceOfPropagatingInstructionsUp = {20,
                                                                          70};
const std::pair<uint32_t, uint32_t> kChanceOfPushingIdThroughVariable = {5, 50};
const std::pair<uint32_t, uint32_t>
    kChanceOfReplacingAddSubMulWithCarryingExtended = {20, 70};
const std::pair<uint32_t, uint32_t>
    kChanceOfReplacingBranchFromDeadBlockWithExit = {10, 65};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingCopyMemoryWithLoadStore =
    {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingCopyObjectWithStoreLoad =
    {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingIdWithSynonym = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingIrrelevantId = {35, 95};
const std::pair<uint32_t, uint32_t>
    kChanceOfReplacingLinearAlgebraInstructions = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingLoadStoreWithCopyMemory =
    {20, 90};
const std::pair<uint32_t, uint32_t>
    kChanceOfReplacingOpPhiIdFromDeadPredecessor = {20, 90};
const std::pair<uint32_t, uint32_t>
    kChanceOfReplacingOpSelectWithConditionalBranch = {20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingParametersWithGlobals = {
    30, 70};
const std::pair<uint32_t, uint32_t> kChanceOfReplacingParametersWithStruct = {
    20, 40};
const std::pair<uint32_t, uint32_t> kChanceOfSplittingBlock = {40, 95};
const std::pair<uint32_t, uint32_t>
    kChanceOfSwappingAnotherPairOfFunctionVariables = {30, 90};
const std::pair<uint32_t, uint32_t> kChanceOfSwappingConditionalBranchOperands =
    {10, 70};
const std::pair<uint32_t, uint32_t> kChanceOfSwappingFunctions = {10, 90};
const std::pair<uint32_t, uint32_t> kChanceOfTogglingAccessChainInstruction = {
    20, 90};
const std::pair<uint32_t, uint32_t> kChanceOfWrappingRegionInSelection = {70,
                                                                          90};
const std::pair<uint32_t, uint32_t> kChanceOfWrappingVectorSynonym = {10, 90};

// Default limits for various quantities that are chosen during fuzzing.
// Keep them in alphabetical order.
const uint32_t kDefaultMaxEquivalenceClassSizeForDataSynonymFactClosure = 1000;
const uint32_t kDefaultMaxLoopControlPartialCount = 100;
const uint32_t kDefaultMaxLoopControlPeelCount = 100;
const uint32_t kDefaultMaxLoopLimit = 20;
const uint32_t kDefaultMaxNewArraySizeLimit = 100;
// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3424):
//  think whether there is a better limit on the maximum number of parameters.
const uint32_t kDefaultMaxNumberOfFunctionParameters = 128;
const uint32_t kDefaultMaxNumberOfNewParameters = 15;
const uint32_t kGetDefaultMaxNumberOfParametersReplacedWithStruct = 5;

// Default functions for controlling how deep to go during recursive
// generation/transformation. Keep them in alphabetical order.

const std::function<bool(uint32_t, RandomGenerator*)>
    kDefaultGoDeeperInConstantObfuscation =
        [](uint32_t current_depth, RandomGenerator* random_generator) -> bool {
  double chance = 1.0 / std::pow(3.0, static_cast<float>(current_depth + 1));
  return random_generator->RandomDouble() < chance;
};

}  // namespace

FuzzerContext::FuzzerContext(std::unique_ptr<RandomGenerator> random_generator,
                             uint32_t min_fresh_id, bool is_wgsl_compatible)
    : random_generator_(std::move(random_generator)),
      next_fresh_id_(min_fresh_id),
      is_wgsl_compatible_(is_wgsl_compatible),
      max_equivalence_class_size_for_data_synonym_fact_closure_(
          kDefaultMaxEquivalenceClassSizeForDataSynonymFactClosure),
      max_loop_control_partial_count_(kDefaultMaxLoopControlPartialCount),
      max_loop_control_peel_count_(kDefaultMaxLoopControlPeelCount),
      max_loop_limit_(kDefaultMaxLoopLimit),
      max_new_array_size_limit_(kDefaultMaxNewArraySizeLimit),
      max_number_of_function_parameters_(kDefaultMaxNumberOfFunctionParameters),
      max_number_of_new_parameters_(kDefaultMaxNumberOfNewParameters),
      max_number_of_parameters_replaced_with_struct_(
          kGetDefaultMaxNumberOfParametersReplacedWithStruct),
      go_deeper_in_constant_obfuscation_(
          kDefaultGoDeeperInConstantObfuscation) {
  chance_of_accepting_repeated_pass_recommendation_ =
      ChooseBetweenMinAndMax(kChanceOfAcceptingRepeatedPassRecommendation);
  chance_of_adding_access_chain_ =
      ChooseBetweenMinAndMax(kChanceOfAddingAccessChain);
  chance_of_adding_another_pass_to_pass_loop_ =
      ChooseBetweenMinAndMax(kChanceOfAddingAnotherPassToPassLoop);
  chance_of_adding_another_struct_field_ =
      ChooseBetweenMinAndMax(kChanceOfAddingAnotherStructField);
  chance_of_adding_array_or_struct_type_ =
      ChooseBetweenMinAndMax(kChanceOfAddingArrayOrStructType);
  chance_of_adding_atomic_load_ =
      ChooseBetweenMinAndMax(KChanceOfAddingAtomicLoad);
  chance_of_adding_atomic_store_ =
      ChooseBetweenMinAndMax(KChanceOfAddingAtomicStore);
  chance_of_adding_bit_instruction_synonym_ =
      ChooseBetweenMinAndMax(kChanceOfAddingBitInstructionSynonym);
  chance_of_adding_both_branches_when_replacing_opselect_ =
      ChooseBetweenMinAndMax(kChanceOfAddingBothBranchesWhenReplacingOpSelect);
  chance_of_adding_composite_extract_ =
      ChooseBetweenMinAndMax(kChanceOfAddingCompositeExtract);
  chance_of_adding_composite_insert_ =
      ChooseBetweenMinAndMax(kChanceOfAddingCompositeInsert);
  chance_of_adding_copy_memory_ =
      ChooseBetweenMinAndMax(kChanceOfAddingCopyMemory);
  chance_of_adding_dead_block_ =
      ChooseBetweenMinAndMax(kChanceOfAddingDeadBlock);
  chance_of_adding_dead_break_ =
      ChooseBetweenMinAndMax(kChanceOfAddingDeadBreak);
  chance_of_adding_dead_continue_ =
      ChooseBetweenMinAndMax(kChanceOfAddingDeadContinue);
  chance_of_adding_equation_instruction_ =
      ChooseBetweenMinAndMax(kChanceOfAddingEquationInstruction);
  chance_of_adding_global_variable_ =
      ChooseBetweenMinAndMax(kChanceOfAddingGlobalVariable);
  chance_of_adding_load_ = ChooseBetweenMinAndMax(kChanceOfAddingLoad);
  chance_of_adding_loop_preheader_ =
      ChooseBetweenMinAndMax(kChanceOfAddingLoopPreheader);
  chance_of_adding_image_sample_unused_components_ =
      ChooseBetweenMinAndMax(kChanceOfAddingImageSampleUnusedComponents);
  chance_of_adding_local_variable_ =
      ChooseBetweenMinAndMax(kChanceOfAddingLocalVariable);
  chance_of_adding_matrix_type_ =
      ChooseBetweenMinAndMax(kChanceOfAddingMatrixType);
  chance_of_adding_no_contraction_decoration_ =
      ChooseBetweenMinAndMax(kChanceOfAddingNoContractionDecoration);
  chance_of_adding_opphi_synonym_ =
      ChooseBetweenMinAndMax(kChanceOfAddingOpPhiSynonym);
  chance_of_adding_parameters =
      ChooseBetweenMinAndMax(kChanceOfAddingParameters);
  chance_of_adding_relaxed_decoration_ =
      ChooseBetweenMinAndMax(kChanceOfAddingRelaxedDecoration);
  chance_of_adding_store_ = ChooseBetweenMinAndMax(kChanceOfAddingStore);
  chance_of_adding_true_branch_when_replacing_opselect_ =
      ChooseBetweenMinAndMax(kChanceOfAddingTrueBranchWhenReplacingOpSelect);
  chance_of_adding_vector_shuffle_ =
      ChooseBetweenMinAndMax(kChanceOfAddingVectorShuffle);
  chance_of_adding_vector_type_ =
      ChooseBetweenMinAndMax(kChanceOfAddingVectorType);
  chance_of_adjusting_branch_weights_ =
      ChooseBetweenMinAndMax(kChanceOfAdjustingBranchWeights);
  chance_of_adjusting_function_control_ =
      ChooseBetweenMinAndMax(kChanceOfAdjustingFunctionControl);
  chance_of_adding_synonyms_ = ChooseBetweenMinAndMax(kChanceOfAddingSynonyms);
  chance_of_adjusting_loop_control_ =
      ChooseBetweenMinAndMax(kChanceOfAdjustingLoopControl);
  chance_of_adjusting_memory_operands_mask_ =
      ChooseBetweenMinAndMax(kChanceOfAdjustingMemoryOperandsMask);
  chance_of_adjusting_selection_control_ =
      ChooseBetweenMinAndMax(kChanceOfAdjustingSelectionControl);
  chance_of_calling_function_ =
      ChooseBetweenMinAndMax(kChanceOfCallingFunction);
  chance_of_choosing_struct_type_vs_array_type_ =
      ChooseBetweenMinAndMax(kChanceOfChoosingStructTypeVsArrayType);
  chance_of_choosing_workgroup_storage_class_ =
      ChooseBetweenMinAndMax(kChanceOfChoosingWorkgroupStorageClass);
  chance_of_constructing_composite_ =
      ChooseBetweenMinAndMax(kChanceOfConstructingComposite);
  chance_of_copying_object_ = ChooseBetweenMinAndMax(kChanceOfCopyingObject);
  chance_of_creating_int_synonyms_using_loops_ =
      ChooseBetweenMinAndMax(kChanceOfCreatingIntSynonymsUsingLoops);
  chance_of_donating_additional_module_ =
      ChooseBetweenMinAndMax(kChanceOfDonatingAdditionalModule);
  chance_of_duplicating_region_with_selection_ =
      ChooseBetweenMinAndMax(kChanceOfDuplicatingRegionWithSelection);
  chance_of_expanding_vector_reduction_ =
      ChooseBetweenMinAndMax(kChanceOfExpandingVectorReduction);
  chance_of_flattening_conditional_branch_ =
      ChooseBetweenMinAndMax(kChanceOfFlatteningConditionalBranch);
  chance_of_going_deeper_to_extract_composite_ =
      ChooseBetweenMinAndMax(kChanceOfGoingDeeperToExtractComposite);
  chance_of_going_deeper_to_insert_in_composite_ =
      ChooseBetweenMinAndMax(kChanceOfGoingDeeperToInsertInComposite);
  chance_of_going_deeper_when_making_access_chain_ =
      ChooseBetweenMinAndMax(kChanceOfGoingDeeperWhenMakingAccessChain);
  chance_of_having_two_blocks_in_loop_to_create_int_synonym_ =
      ChooseBetweenMinAndMax(kChanceOfHavingTwoBlocksInLoopToCreateIntSynonym);
  chance_of_inlining_function_ =
      ChooseBetweenMinAndMax(kChanceOfInliningFunction);
  chance_of_interchanging_signedness_of_integer_operands_ =
      ChooseBetweenMinAndMax(kChanceOfInterchangingSignednessOfIntegerOperands);
  chance_of_interchanging_zero_like_constants_ =
      ChooseBetweenMinAndMax(kChanceOfInterchangingZeroLikeConstants);
  chance_of_inverting_comparison_operators_ =
      ChooseBetweenMinAndMax(kChanceOfInvertingComparisonOperators);
  chance_of_making_donor_livesafe_ =
      ChooseBetweenMinAndMax(kChanceOfMakingDonorLivesafe);
  chance_of_making_vector_operation_dynamic_ =
      ChooseBetweenMinAndMax(kChanceOfMakingVectorOperationDynamic);
  chance_of_merging_blocks_ = ChooseBetweenMinAndMax(kChanceOfMergingBlocks);
  chance_of_merging_function_returns_ =
      ChooseBetweenMinAndMax(kChanceOfMergingFunctionReturns);
  chance_of_moving_block_down_ =
      ChooseBetweenMinAndMax(kChanceOfMovingBlockDown);
  chance_of_mutating_pointer_ =
      ChooseBetweenMinAndMax(kChanceOfMutatingPointer);
  chance_of_obfuscating_constant_ =
      ChooseBetweenMinAndMax(kChanceOfObfuscatingConstant);
  chance_of_outlining_function_ =
      ChooseBetweenMinAndMax(kChanceOfOutliningFunction);
  chance_of_permuting_function_variables_ =
      ChooseBetweenMinAndMax(kChanceOfPermutingFunctionVariables);
  chance_of_permuting_instructions_ =
      ChooseBetweenMinAndMax(kChanceOfPermutingInstructions);
  chance_of_permuting_parameters_ =
      ChooseBetweenMinAndMax(kChanceOfPermutingParameters);
  chance_of_permuting_phi_operands_ =
      ChooseBetweenMinAndMax(kChanceOfPermutingPhiOperands);
  chance_of_propagating_instructions_down_ =
      ChooseBetweenMinAndMax(kChanceOfPropagatingInstructionsDown);
  chance_of_propagating_instructions_up_ =
      ChooseBetweenMinAndMax(kChanceOfPropagatingInstructionsUp);
  chance_of_pushing_id_through_variable_ =
      ChooseBetweenMinAndMax(kChanceOfPushingIdThroughVariable);
  chance_of_replacing_add_sub_mul_with_carrying_extended_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingAddSubMulWithCarryingExtended);
  chance_of_replacing_branch_from_dead_block_with_exit_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingBranchFromDeadBlockWithExit);
  chance_of_replacing_copy_memory_with_load_store_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingCopyMemoryWithLoadStore);
  chance_of_replacing_copyobject_with_store_load_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingCopyObjectWithStoreLoad);
  chance_of_replacing_id_with_synonym_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingIdWithSynonym);
  chance_of_replacing_irrelevant_id_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingIrrelevantId);
  chance_of_replacing_linear_algebra_instructions_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingLinearAlgebraInstructions);
  chance_of_replacing_load_store_with_copy_memory_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingLoadStoreWithCopyMemory);
  chance_of_replacing_opphi_id_from_dead_predecessor_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingOpPhiIdFromDeadPredecessor);
  chance_of_replacing_opselect_with_conditional_branch_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingOpSelectWithConditionalBranch);
  chance_of_replacing_parameters_with_globals_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingParametersWithGlobals);
  chance_of_replacing_parameters_with_struct_ =
      ChooseBetweenMinAndMax(kChanceOfReplacingParametersWithStruct);
  chance_of_splitting_block_ = ChooseBetweenMinAndMax(kChanceOfSplittingBlock);
  chance_of_swapping_another_pair_of_function_variables_ =
      ChooseBetweenMinAndMax(kChanceOfSwappingAnotherPairOfFunctionVariables);
  chance_of_swapping_conditional_branch_operands_ =
      ChooseBetweenMinAndMax(kChanceOfSwappingConditionalBranchOperands);
  chance_of_swapping_functions_ =
      ChooseBetweenMinAndMax(kChanceOfSwappingFunctions);
  chance_of_toggling_access_chain_instruction_ =
      ChooseBetweenMinAndMax(kChanceOfTogglingAccessChainInstruction);
  chance_of_wrapping_region_in_selection_ =
      ChooseBetweenMinAndMax(kChanceOfWrappingRegionInSelection);
  chance_of_wrapping_vector_synonym_ =
      ChooseBetweenMinAndMax(kChanceOfWrappingVectorSynonym);
}

FuzzerContext::~FuzzerContext() = default;

uint32_t FuzzerContext::GetFreshId() { return next_fresh_id_++; }

std::vector<uint32_t> FuzzerContext::GetFreshIds(const uint32_t count) {
  std::vector<uint32_t> fresh_ids(count);

  for (uint32_t& fresh_id : fresh_ids) {
    fresh_id = next_fresh_id_++;
  }

  return fresh_ids;
}

bool FuzzerContext::ChooseEven() { return random_generator_->RandomBool(); }

bool FuzzerContext::ChoosePercentage(uint32_t percentage_chance) {
  assert(percentage_chance <= 100);
  return random_generator_->RandomPercentage() < percentage_chance;
}

uint32_t FuzzerContext::ChooseBetweenMinAndMax(
    const std::pair<uint32_t, uint32_t>& min_max) {
  assert(min_max.first <= min_max.second);
  return min_max.first +
         random_generator_->RandomUint32(min_max.second - min_max.first + 1);
}

protobufs::TransformationAddSynonym::SynonymType
FuzzerContext::GetRandomSynonymType() {
  // value_count method is guaranteed to return a value greater than 0.
  auto result_index = ChooseBetweenMinAndMax(
      {0, static_cast<uint32_t>(
              protobufs::TransformationAddSynonym::SynonymType_descriptor()
                  ->value_count() -
              1)});
  auto result = protobufs::TransformationAddSynonym::SynonymType_descriptor()
                    ->value(result_index)
                    ->number();
  assert(protobufs::TransformationAddSynonym::SynonymType_IsValid(result) &&
         "|result| is not a value of SynonymType");
  return static_cast<protobufs::TransformationAddSynonym::SynonymType>(result);
}

uint32_t FuzzerContext::GetIdBoundLimit() const { return kIdBoundLimit; }

uint32_t FuzzerContext::GetTransformationLimit() const {
  return kTransformationLimit;
}

uint32_t FuzzerContext::GetMinFreshId(opt::IRContext* ir_context) {
  return ir_context->module()->id_bound() + kIdBoundGap;
}

}  // namespace fuzz
}  // namespace spvtools
