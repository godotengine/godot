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

#ifndef SOURCE_FUZZ_FUZZER_CONTEXT_H_
#define SOURCE_FUZZ_FUZZER_CONTEXT_H_

#include <functional>
#include <utility>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/random_generator.h"
#include "source/opt/function.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Encapsulates all parameters that control the fuzzing process, such as the
// source of randomness and the probabilities with which transformations are
// applied.
class FuzzerContext {
 public:
  // Constructs a fuzzer context with a given random generator and the minimum
  // value that can be used for fresh ids.
  FuzzerContext(std::unique_ptr<RandomGenerator> random_generator,
                uint32_t min_fresh_id, bool is_wgsl_compatible);

  ~FuzzerContext();

  // Returns a random boolean.
  bool ChooseEven();

  // Returns true if and only if a randomly-chosen integer in the range [0, 100]
  // is less than |percentage_chance|.
  bool ChoosePercentage(uint32_t percentage_chance);

  // Returns a random index into |sequence|, which is expected to have a 'size'
  // method, and which must be non-empty.  Typically 'HasSizeMethod' will be an
  // std::vector.
  template <typename HasSizeMethod>
  uint32_t RandomIndex(const HasSizeMethod& sequence) const {
    assert(sequence.size() > 0);
    return random_generator_->RandomUint32(
        static_cast<uint32_t>(sequence.size()));
  }

  // Selects a random index into |sequence|, removes the element at that index
  // and returns it.
  template <typename T>
  T RemoveAtRandomIndex(std::vector<T>* sequence) const {
    uint32_t index = RandomIndex(*sequence);
    T result = sequence->at(index);
    sequence->erase(sequence->begin() + index);
    return result;
  }

  // Randomly shuffles a |sequence| between |lo| and |hi| indices inclusively.
  // |lo| and |hi| must be valid indices to the |sequence|.
  template <typename T>
  void Shuffle(std::vector<T>* sequence, size_t lo, size_t hi) const {
    auto& array = *sequence;

    if (array.empty()) {
      return;
    }

    assert(lo <= hi && hi < array.size() && "lo and/or hi indices are invalid");

    // i > lo to account for potential infinite loop when lo == 0
    for (size_t i = hi; i > lo; --i) {
      auto index =
          random_generator_->RandomUint32(static_cast<uint32_t>(i - lo + 1));

      if (lo + index != i) {
        // Introduce std::swap to the scope but don't use it
        // directly since there might be a better overload
        using std::swap;
        swap(array[lo + index], array[i]);
      }
    }
  }

  // Randomly shuffles a |sequence|.
  template <typename T>
  void Shuffle(std::vector<T>* sequence) const {
    if (!sequence->empty()) {
      Shuffle(sequence, 0, sequence->size() - 1);
    }
  }

  // Yields an id that is guaranteed not to be used in the module being fuzzed,
  // or to have been issued before.
  uint32_t GetFreshId();

  // Returns a vector of |count| fresh ids.
  std::vector<uint32_t> GetFreshIds(uint32_t count);

  // A suggested limit on the id bound for the module being fuzzed.  This is
  // useful for deciding when to stop the overall fuzzing process.  Furthermore,
  // fuzzer passes that run the risk of spiralling out of control can
  // periodically check this limit and terminate early if it has been reached.
  uint32_t GetIdBoundLimit() const;

  // A suggested limit on the number of transformations that should be applied.
  // Also useful to control the overall fuzzing process and rein in individual
  // fuzzer passes.
  uint32_t GetTransformationLimit() const;

  // Returns the minimum fresh id that can be used given the |ir_context|.
  static uint32_t GetMinFreshId(opt::IRContext* ir_context);

  // Returns true if all transformations should be compatible with WGSL.
  bool IsWgslCompatible() const {
    return is_wgsl_compatible_;
  }

  // Probabilities associated with applying various transformations.
  // Keep them in alphabetical order.
  uint32_t GetChanceOfAcceptingRepeatedPassRecommendation() const {
    return chance_of_accepting_repeated_pass_recommendation_;
  }
  uint32_t GetChanceOfAddingAccessChain() const {
    return chance_of_adding_access_chain_;
  }
  uint32_t GetChanceOfAddingAnotherPassToPassLoop() const {
    return chance_of_adding_another_pass_to_pass_loop_;
  }
  uint32_t GetChanceOfAddingAnotherStructField() const {
    return chance_of_adding_another_struct_field_;
  }
  uint32_t GetChanceOfAddingArrayOrStructType() const {
    return chance_of_adding_array_or_struct_type_;
  }
  uint32_t GetChanceOfAddingAtomicLoad() const {
    return chance_of_adding_atomic_load_;
  }
  uint32_t GetChanceOfAddingAtomicStore() const {
    return chance_of_adding_atomic_store_;
  }
  uint32_t GetChanceOfAddingBitInstructionSynonym() const {
    return chance_of_adding_bit_instruction_synonym_;
  }
  uint32_t GetChanceOfAddingBothBranchesWhenReplacingOpSelect() const {
    return chance_of_adding_both_branches_when_replacing_opselect_;
  }
  uint32_t GetChanceOfAddingCompositeExtract() const {
    return chance_of_adding_composite_extract_;
  }
  uint32_t GetChanceOfAddingCompositeInsert() const {
    return chance_of_adding_composite_insert_;
  }
  uint32_t GetChanceOfAddingCopyMemory() const {
    return chance_of_adding_copy_memory_;
  }
  uint32_t GetChanceOfAddingDeadBlock() const {
    return chance_of_adding_dead_block_;
  }
  uint32_t GetChanceOfAddingDeadBreak() const {
    return chance_of_adding_dead_break_;
  }
  uint32_t GetChanceOfAddingDeadContinue() const {
    return chance_of_adding_dead_continue_;
  }
  uint32_t GetChanceOfAddingEquationInstruction() const {
    return chance_of_adding_equation_instruction_;
  }
  uint32_t GetChanceOfAddingGlobalVariable() const {
    return chance_of_adding_global_variable_;
  }
  uint32_t GetChanceOfAddingImageSampleUnusedComponents() const {
    return chance_of_adding_image_sample_unused_components_;
  }
  uint32_t GetChanceOfAddingLoad() const { return chance_of_adding_load_; }
  uint32_t GetChanceOfAddingLocalVariable() const {
    return chance_of_adding_local_variable_;
  }
  uint32_t GetChanceOfAddingLoopPreheader() const {
    return chance_of_adding_loop_preheader_;
  }
  uint32_t GetChanceOfAddingMatrixType() const {
    return chance_of_adding_matrix_type_;
  }
  uint32_t GetChanceOfAddingNoContractionDecoration() const {
    return chance_of_adding_no_contraction_decoration_;
  }
  uint32_t GetChanceOfAddingOpPhiSynonym() const {
    return chance_of_adding_opphi_synonym_;
  }
  uint32_t GetChanceOfAddingParameters() const {
    return chance_of_adding_parameters;
  }
  uint32_t GetChanceOfAddingRelaxedDecoration() const {
    return chance_of_adding_relaxed_decoration_;
  }
  uint32_t GetChanceOfAddingStore() const { return chance_of_adding_store_; }
  uint32_t GetChanceOfAddingSynonyms() const {
    return chance_of_adding_synonyms_;
  }
  uint32_t GetChanceOfAddingTrueBranchWhenReplacingOpSelect() const {
    return chance_of_adding_true_branch_when_replacing_opselect_;
  }
  uint32_t GetChanceOfAddingVectorShuffle() const {
    return chance_of_adding_vector_shuffle_;
  }
  uint32_t GetChanceOfAddingVectorType() const {
    return chance_of_adding_vector_type_;
  }
  uint32_t GetChanceOfAdjustingBranchWeights() const {
    return chance_of_adjusting_branch_weights_;
  }
  uint32_t GetChanceOfAdjustingFunctionControl() const {
    return chance_of_adjusting_function_control_;
  }
  uint32_t GetChanceOfAdjustingLoopControl() const {
    return chance_of_adjusting_loop_control_;
  }
  uint32_t GetChanceOfAdjustingMemoryOperandsMask() const {
    return chance_of_adjusting_memory_operands_mask_;
  }
  uint32_t GetChanceOfAdjustingSelectionControl() const {
    return chance_of_adjusting_selection_control_;
  }
  uint32_t GetChanceOfCallingFunction() const {
    return chance_of_calling_function_;
  }
  uint32_t GetChanceOfChoosingStructTypeVsArrayType() const {
    return chance_of_choosing_struct_type_vs_array_type_;
  }
  uint32_t GetChanceOfChoosingWorkgroupStorageClass() const {
    return chance_of_choosing_workgroup_storage_class_;
  }
  uint32_t GetChanceOfConstructingComposite() const {
    return chance_of_constructing_composite_;
  }
  uint32_t GetChanceOfCopyingObject() const {
    return chance_of_copying_object_;
  }
  uint32_t GetChanceOfCreatingIntSynonymsUsingLoops() const {
    return chance_of_creating_int_synonyms_using_loops_;
  }
  uint32_t GetChanceOfDonatingAdditionalModule() const {
    return chance_of_donating_additional_module_;
  }
  uint32_t GetChanceOfDuplicatingRegionWithSelection() const {
    return chance_of_duplicating_region_with_selection_;
  }
  uint32_t GetChanceOfExpandingVectorReduction() const {
    return chance_of_expanding_vector_reduction_;
  }
  uint32_t GetChanceOfFlatteningConditionalBranch() const {
    return chance_of_flattening_conditional_branch_;
  }
  uint32_t GetChanceOfGoingDeeperToExtractComposite() const {
    return chance_of_going_deeper_to_extract_composite_;
  }
  uint32_t GetChanceOfGoingDeeperToInsertInComposite() const {
    return chance_of_going_deeper_to_insert_in_composite_;
  }
  uint32_t GetChanceOfGoingDeeperWhenMakingAccessChain() const {
    return chance_of_going_deeper_when_making_access_chain_;
  }
  uint32_t GetChanceOfHavingTwoBlocksInLoopToCreateIntSynonym() const {
    return chance_of_having_two_blocks_in_loop_to_create_int_synonym_;
  }
  uint32_t GetChanceOfInliningFunction() const {
    return chance_of_inlining_function_;
  }
  uint32_t GetChanceOfInterchangingSignednessOfIntegerOperands() const {
    return chance_of_interchanging_signedness_of_integer_operands_;
  }
  uint32_t GetChanceOfInterchangingZeroLikeConstants() const {
    return chance_of_interchanging_zero_like_constants_;
  }
  uint32_t GetChanceOfInvertingComparisonOperators() const {
    return chance_of_inverting_comparison_operators_;
  }
  uint32_t ChanceOfMakingDonorLivesafe() const {
    return chance_of_making_donor_livesafe_;
  }
  uint32_t GetChanceOfMakingVectorOperationDynamic() const {
    return chance_of_making_vector_operation_dynamic_;
  }
  uint32_t GetChanceOfMergingBlocks() const {
    return chance_of_merging_blocks_;
  }
  uint32_t GetChanceOfMergingFunctionReturns() const {
    return chance_of_merging_function_returns_;
  }
  uint32_t GetChanceOfMovingBlockDown() const {
    return chance_of_moving_block_down_;
  }
  uint32_t GetChanceOfMutatingPointer() const {
    return chance_of_mutating_pointer_;
  }
  uint32_t GetChanceOfObfuscatingConstant() const {
    return chance_of_obfuscating_constant_;
  }
  uint32_t GetChanceOfOutliningFunction() const {
    return chance_of_outlining_function_;
  }
  uint32_t GetChanceOfPermutingFunctionVariables() const {
    return chance_of_permuting_function_variables_;
  }
  uint32_t GetChanceOfPermutingInstructions() const {
    return chance_of_permuting_instructions_;
  }
  uint32_t GetChanceOfPermutingParameters() const {
    return chance_of_permuting_parameters_;
  }
  uint32_t GetChanceOfPermutingPhiOperands() const {
    return chance_of_permuting_phi_operands_;
  }
  uint32_t GetChanceOfPropagatingInstructionsDown() const {
    return chance_of_propagating_instructions_down_;
  }
  uint32_t GetChanceOfPropagatingInstructionsUp() const {
    return chance_of_propagating_instructions_up_;
  }
  uint32_t GetChanceOfPushingIdThroughVariable() const {
    return chance_of_pushing_id_through_variable_;
  }
  uint32_t GetChanceOfReplacingAddSubMulWithCarryingExtended() const {
    return chance_of_replacing_add_sub_mul_with_carrying_extended_;
  }
  uint32_t GetChanceOfReplacingBranchFromDeadBlockWithExit() const {
    return chance_of_replacing_branch_from_dead_block_with_exit_;
  }
  uint32_t GetChanceOfReplacingCopyMemoryWithLoadStore() const {
    return chance_of_replacing_copy_memory_with_load_store_;
  }
  uint32_t GetChanceOfReplacingCopyObjectWithStoreLoad() const {
    return chance_of_replacing_copyobject_with_store_load_;
  }
  uint32_t GetChanceOfReplacingIdWithSynonym() const {
    return chance_of_replacing_id_with_synonym_;
  }
  uint32_t GetChanceOfReplacingIrrelevantId() const {
    return chance_of_replacing_irrelevant_id_;
  }
  uint32_t GetChanceOfReplacingLinearAlgebraInstructions() const {
    return chance_of_replacing_linear_algebra_instructions_;
  }
  uint32_t GetChanceOfReplacingLoadStoreWithCopyMemory() const {
    return chance_of_replacing_load_store_with_copy_memory_;
  }
  uint32_t GetChanceOfReplacingOpPhiIdFromDeadPredecessor() const {
    return chance_of_replacing_opphi_id_from_dead_predecessor_;
  }
  uint32_t GetChanceOfReplacingOpselectWithConditionalBranch() const {
    return chance_of_replacing_opselect_with_conditional_branch_;
  }
  uint32_t GetChanceOfReplacingParametersWithGlobals() const {
    return chance_of_replacing_parameters_with_globals_;
  }
  uint32_t GetChanceOfReplacingParametersWithStruct() const {
    return chance_of_replacing_parameters_with_struct_;
  }
  uint32_t GetChanceOfSplittingBlock() const {
    return chance_of_splitting_block_;
  }
  uint32_t GetChanceOfSwappingAnotherPairOfFunctionVariables() const {
    return chance_of_swapping_another_pair_of_function_variables_;
  }
  uint32_t GetChanceOfSwappingConditionalBranchOperands() const {
    return chance_of_swapping_conditional_branch_operands_;
  }

  uint32_t GetChanceOfSwappingFunctions() const {
    return chance_of_swapping_functions_;
  }

  uint32_t GetChanceOfTogglingAccessChainInstruction() const {
    return chance_of_toggling_access_chain_instruction_;
  }
  uint32_t GetChanceOfWrappingRegionInSelection() const {
    return chance_of_wrapping_region_in_selection_;
  }

  uint32_t GetChanceOfWrappingVectorSynonym() const {
    return chance_of_wrapping_vector_synonym_;
  }

  // Other functions to control transformations. Keep them in alphabetical
  // order.
  uint32_t GetMaximumEquivalenceClassSizeForDataSynonymFactClosure() const {
    return max_equivalence_class_size_for_data_synonym_fact_closure_;
  }
  uint32_t GetMaximumNumberOfFunctionParameters() const {
    return max_number_of_function_parameters_;
  }
  uint32_t GetMaximumNumberOfParametersReplacedWithStruct() const {
    return max_number_of_parameters_replaced_with_struct_;
  }
  std::pair<uint32_t, uint32_t> GetRandomBranchWeights() {
    std::pair<uint32_t, uint32_t> branch_weights = {0, 0};

    while (branch_weights.first == 0 && branch_weights.second == 0) {
      // Using INT32_MAX to do not overflow UINT32_MAX when the branch weights
      // are added together.
      branch_weights.first = random_generator_->RandomUint32(INT32_MAX);
      branch_weights.second = random_generator_->RandomUint32(INT32_MAX);
    }

    return branch_weights;
  }
  std::vector<uint32_t> GetRandomComponentsForVectorShuffle(
      uint32_t max_component_index) {
    // Component count must be in range [2, 4].
    std::vector<uint32_t> components(random_generator_->RandomUint32(2) + 2);

    for (uint32_t& component : components) {
      component = random_generator_->RandomUint32(max_component_index);
    }

    return components;
  }
  uint32_t GetRandomCompositeExtractIndex(uint32_t number_of_members) {
    assert(number_of_members > 0 && "Composite object must have some members");
    return ChooseBetweenMinAndMax({0, number_of_members - 1});
  }
  uint32_t GetRandomIndexForAccessChain(uint32_t composite_size_bound) {
    return random_generator_->RandomUint32(composite_size_bound);
  }
  uint32_t GetRandomIndexForCompositeInsert(uint32_t number_of_components) {
    return random_generator_->RandomUint32(number_of_components);
  }
  uint32_t GetRandomIndexForWrappingVector(uint32_t vector_width) {
    return random_generator_->RandomUint32(vector_width);
  }
  int64_t GetRandomValueForStepConstantInLoop() {
    return random_generator_->RandomUint64(UINT64_MAX);
  }
  uint32_t GetRandomLoopControlPartialCount() {
    return random_generator_->RandomUint32(max_loop_control_partial_count_);
  }
  uint32_t GetRandomLoopControlPeelCount() {
    return random_generator_->RandomUint32(max_loop_control_peel_count_);
  }
  uint32_t GetRandomLoopLimit() {
    return random_generator_->RandomUint32(max_loop_limit_);
  }
  uint32_t GetRandomNumberOfLoopIterations(uint32_t max_num_iterations) {
    return ChooseBetweenMinAndMax({1, max_num_iterations});
  }
  uint32_t GetRandomNumberOfNewParameters(uint32_t num_of_params) {
    assert(num_of_params < GetMaximumNumberOfFunctionParameters());
    return ChooseBetweenMinAndMax(
        {1, std::min(max_number_of_new_parameters_,
                     GetMaximumNumberOfFunctionParameters() - num_of_params)});
  }
  uint32_t GetRandomNumberOfParametersReplacedWithStruct(uint32_t num_params) {
    assert(num_params != 0 && "A function must have parameters to replace");
    return ChooseBetweenMinAndMax(
        {1, std::min(num_params,
                     GetMaximumNumberOfParametersReplacedWithStruct())});
  }
  uint32_t GetRandomSizeForNewArray() {
    // Ensure that the array size is non-zero.
    return random_generator_->RandomUint32(max_new_array_size_limit_ - 1) + 1;
  }
  protobufs::TransformationAddSynonym::SynonymType GetRandomSynonymType();
  uint32_t GetRandomUnusedComponentCountForImageSample(
      uint32_t max_unused_component_count) {
    // Ensure that the number of unused components is non-zero.
    return random_generator_->RandomUint32(max_unused_component_count) + 1;
  }
  uint32_t GetWidthOfWrappingVector() {
    return 2 + random_generator_->RandomUint32(3);
  }
  bool GoDeeperInConstantObfuscation(uint32_t depth) {
    return go_deeper_in_constant_obfuscation_(depth, random_generator_.get());
  }

 private:
  // The source of randomness.
  std::unique_ptr<RandomGenerator> random_generator_;
  // The next fresh id to be issued.
  uint32_t next_fresh_id_;

  // True if all transformations should be compatible with WGSL spec.
  bool is_wgsl_compatible_;

  // Probabilities associated with applying various transformations.
  // Keep them in alphabetical order.
  uint32_t chance_of_accepting_repeated_pass_recommendation_;
  uint32_t chance_of_adding_access_chain_;
  uint32_t chance_of_adding_another_pass_to_pass_loop_;
  uint32_t chance_of_adding_another_struct_field_;
  uint32_t chance_of_adding_array_or_struct_type_;
  uint32_t chance_of_adding_atomic_load_;
  uint32_t chance_of_adding_atomic_store_;
  uint32_t chance_of_adding_bit_instruction_synonym_;
  uint32_t chance_of_adding_both_branches_when_replacing_opselect_;
  uint32_t chance_of_adding_composite_extract_;
  uint32_t chance_of_adding_composite_insert_;
  uint32_t chance_of_adding_copy_memory_;
  uint32_t chance_of_adding_dead_block_;
  uint32_t chance_of_adding_dead_break_;
  uint32_t chance_of_adding_dead_continue_;
  uint32_t chance_of_adding_equation_instruction_;
  uint32_t chance_of_adding_global_variable_;
  uint32_t chance_of_adding_image_sample_unused_components_;
  uint32_t chance_of_adding_load_;
  uint32_t chance_of_adding_local_variable_;
  uint32_t chance_of_adding_loop_preheader_;
  uint32_t chance_of_adding_matrix_type_;
  uint32_t chance_of_adding_no_contraction_decoration_;
  uint32_t chance_of_adding_opphi_synonym_;
  uint32_t chance_of_adding_parameters;
  uint32_t chance_of_adding_relaxed_decoration_;
  uint32_t chance_of_adding_store_;
  uint32_t chance_of_adding_synonyms_;
  uint32_t chance_of_adding_true_branch_when_replacing_opselect_;
  uint32_t chance_of_adding_vector_shuffle_;
  uint32_t chance_of_adding_vector_type_;
  uint32_t chance_of_adjusting_branch_weights_;
  uint32_t chance_of_adjusting_function_control_;
  uint32_t chance_of_adjusting_loop_control_;
  uint32_t chance_of_adjusting_memory_operands_mask_;
  uint32_t chance_of_adjusting_selection_control_;
  uint32_t chance_of_calling_function_;
  uint32_t chance_of_choosing_struct_type_vs_array_type_;
  uint32_t chance_of_choosing_workgroup_storage_class_;
  uint32_t chance_of_constructing_composite_;
  uint32_t chance_of_copying_object_;
  uint32_t chance_of_creating_int_synonyms_using_loops_;
  uint32_t chance_of_donating_additional_module_;
  uint32_t chance_of_duplicating_region_with_selection_;
  uint32_t chance_of_expanding_vector_reduction_;
  uint32_t chance_of_flattening_conditional_branch_;
  uint32_t chance_of_going_deeper_to_extract_composite_;
  uint32_t chance_of_going_deeper_to_insert_in_composite_;
  uint32_t chance_of_going_deeper_when_making_access_chain_;
  uint32_t chance_of_having_two_blocks_in_loop_to_create_int_synonym_;
  uint32_t chance_of_inlining_function_;
  uint32_t chance_of_interchanging_signedness_of_integer_operands_;
  uint32_t chance_of_interchanging_zero_like_constants_;
  uint32_t chance_of_inverting_comparison_operators_;
  uint32_t chance_of_making_donor_livesafe_;
  uint32_t chance_of_making_vector_operation_dynamic_;
  uint32_t chance_of_merging_blocks_;
  uint32_t chance_of_merging_function_returns_;
  uint32_t chance_of_moving_block_down_;
  uint32_t chance_of_mutating_pointer_;
  uint32_t chance_of_obfuscating_constant_;
  uint32_t chance_of_outlining_function_;
  uint32_t chance_of_permuting_function_variables_;
  uint32_t chance_of_permuting_instructions_;
  uint32_t chance_of_permuting_parameters_;
  uint32_t chance_of_permuting_phi_operands_;
  uint32_t chance_of_propagating_instructions_down_;
  uint32_t chance_of_propagating_instructions_up_;
  uint32_t chance_of_pushing_id_through_variable_;
  uint32_t chance_of_replacing_add_sub_mul_with_carrying_extended_;
  uint32_t chance_of_replacing_branch_from_dead_block_with_exit_;
  uint32_t chance_of_replacing_copy_memory_with_load_store_;
  uint32_t chance_of_replacing_copyobject_with_store_load_;
  uint32_t chance_of_replacing_id_with_synonym_;
  uint32_t chance_of_replacing_irrelevant_id_;
  uint32_t chance_of_replacing_linear_algebra_instructions_;
  uint32_t chance_of_replacing_load_store_with_copy_memory_;
  uint32_t chance_of_replacing_opphi_id_from_dead_predecessor_;
  uint32_t chance_of_replacing_opselect_with_conditional_branch_;
  uint32_t chance_of_replacing_parameters_with_globals_;
  uint32_t chance_of_replacing_parameters_with_struct_;
  uint32_t chance_of_splitting_block_;
  uint32_t chance_of_swapping_another_pair_of_function_variables_;
  uint32_t chance_of_swapping_conditional_branch_operands_;
  uint32_t chance_of_swapping_functions_;
  uint32_t chance_of_toggling_access_chain_instruction_;
  uint32_t chance_of_wrapping_region_in_selection_;
  uint32_t chance_of_wrapping_vector_synonym_;

  // Limits associated with various quantities for which random values are
  // chosen during fuzzing.
  // Keep them in alphabetical order.
  uint32_t max_equivalence_class_size_for_data_synonym_fact_closure_;
  uint32_t max_loop_control_partial_count_;
  uint32_t max_loop_control_peel_count_;
  uint32_t max_loop_limit_;
  uint32_t max_new_array_size_limit_;
  uint32_t max_number_of_function_parameters_;
  uint32_t max_number_of_new_parameters_;
  uint32_t max_number_of_parameters_replaced_with_struct_;

  // Functions to determine with what probability to go deeper when generating
  // or mutating constructs recursively.
  const std::function<bool(uint32_t, RandomGenerator*)>&
      go_deeper_in_constant_obfuscation_;

  // Requires |min_max.first| <= |min_max.second|, and returns a value in the
  // range [ |min_max.first|, |min_max.second| ]
  uint32_t ChooseBetweenMinAndMax(const std::pair<uint32_t, uint32_t>& min_max);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_CONTEXT_H_
