// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_REPEATED_PASS_INSTANCES_H_
#define SOURCE_FUZZ_REPEATED_PASS_INSTANCES_H_

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
#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_parameters.h"
#include "source/fuzz/fuzzer_pass_add_relaxed_decorations.h"
#include "source/fuzz/fuzzer_pass_add_stores.h"
#include "source/fuzz/fuzzer_pass_add_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_vector_shuffle_instructions.h"
#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"
#include "source/fuzz/fuzzer_pass_construct_composites.h"
#include "source/fuzz/fuzzer_pass_copy_objects.h"
#include "source/fuzz/fuzzer_pass_donate_modules.h"
#include "source/fuzz/fuzzer_pass_duplicate_regions_with_selections.h"
#include "source/fuzz/fuzzer_pass_expand_vector_reductions.h"
#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_inline_functions.h"
#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"
#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"
#include "source/fuzz/fuzzer_pass_merge_blocks.h"
#include "source/fuzz/fuzzer_pass_merge_function_returns.h"
#include "source/fuzz/fuzzer_pass_mutate_pointers.h"
#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"
#include "source/fuzz/fuzzer_pass_outline_functions.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_pass_permute_instructions.h"
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
#include "source/fuzz/fuzzer_pass_swap_conditional_branch_operands.h"
#include "source/fuzz/fuzzer_pass_wrap_regions_in_selections.h"
#include "source/fuzz/fuzzer_pass_wrap_vector_synonym.h"

namespace spvtools {
namespace fuzz {

// This class has a distinct member for each repeated fuzzer pass (i.e., a
// fuzzer pass that it makes sense to run multiple times).  If a member is null
// then we do not have an instance of that fuzzer pass, i.e. it is disabled.
// The class also provides access to the set of passes that are enabled.
class RepeatedPassInstances {
// This macro should be invoked below for every repeated fuzzer pass.  If a
// repeated fuzzer pass is called FuzzerPassFoo then the macro invocation:
//
//    REPEATED_PASS_INSTANCE(Foo);
//
// should be used.  This adds a private member of type FuzzerPassFoo*, and
// provides the following public methods:
//
// // Requires that SetPass has not been called previously with FuzzerPassFoo.
// // Adds |pass| to the set of known pass instances.
// void SetPass(std::unique_ptr<FuzzerPassFoo> pass);
//
// // Returns a pointer to a pass instance of type FuzzerPassFoo that was
// // previously registered via SetPass(), or nullptr if no such instance was
// // registered
// FuzzerPassFoo* GetFoo();
#define REPEATED_PASS_INSTANCE(NAME)                                     \
 public:                                                                 \
  FuzzerPass##NAME* Get##NAME() const { return NAME##_; }                \
  void SetPass(std::unique_ptr<FuzzerPass##NAME> pass) {                 \
    assert(NAME##_ == nullptr && "Attempt to set pass multiple times."); \
    NAME##_ = pass.get();                                                \
    passes_.push_back(std::move(pass));                                  \
  }                                                                      \
                                                                         \
 private:                                                                \
  FuzzerPass##NAME* NAME##_ = nullptr

  REPEATED_PASS_INSTANCE(AddAccessChains);
  REPEATED_PASS_INSTANCE(AddBitInstructionSynonyms);
  REPEATED_PASS_INSTANCE(AddCompositeExtract);
  REPEATED_PASS_INSTANCE(AddCompositeInserts);
  REPEATED_PASS_INSTANCE(AddCompositeTypes);
  REPEATED_PASS_INSTANCE(AddCopyMemory);
  REPEATED_PASS_INSTANCE(AddDeadBlocks);
  REPEATED_PASS_INSTANCE(AddDeadBreaks);
  REPEATED_PASS_INSTANCE(AddDeadContinues);
  REPEATED_PASS_INSTANCE(AddEquationInstructions);
  REPEATED_PASS_INSTANCE(AddFunctionCalls);
  REPEATED_PASS_INSTANCE(AddGlobalVariables);
  REPEATED_PASS_INSTANCE(AddImageSampleUnusedComponents);
  REPEATED_PASS_INSTANCE(AddLoads);
  REPEATED_PASS_INSTANCE(AddLocalVariables);
  REPEATED_PASS_INSTANCE(AddLoopPreheaders);
  REPEATED_PASS_INSTANCE(AddLoopsToCreateIntConstantSynonyms);
  REPEATED_PASS_INSTANCE(AddOpPhiSynonyms);
  REPEATED_PASS_INSTANCE(AddParameters);
  REPEATED_PASS_INSTANCE(AddRelaxedDecorations);
  REPEATED_PASS_INSTANCE(AddStores);
  REPEATED_PASS_INSTANCE(AddSynonyms);
  REPEATED_PASS_INSTANCE(AddVectorShuffleInstructions);
  REPEATED_PASS_INSTANCE(ApplyIdSynonyms);
  REPEATED_PASS_INSTANCE(ConstructComposites);
  REPEATED_PASS_INSTANCE(CopyObjects);
  REPEATED_PASS_INSTANCE(DonateModules);
  REPEATED_PASS_INSTANCE(DuplicateRegionsWithSelections);
  REPEATED_PASS_INSTANCE(ExpandVectorReductions);
  REPEATED_PASS_INSTANCE(FlattenConditionalBranches);
  REPEATED_PASS_INSTANCE(InlineFunctions);
  REPEATED_PASS_INSTANCE(InvertComparisonOperators);
  REPEATED_PASS_INSTANCE(MakeVectorOperationsDynamic);
  REPEATED_PASS_INSTANCE(MergeBlocks);
  REPEATED_PASS_INSTANCE(MergeFunctionReturns);
  REPEATED_PASS_INSTANCE(MutatePointers);
  REPEATED_PASS_INSTANCE(ObfuscateConstants);
  REPEATED_PASS_INSTANCE(OutlineFunctions);
  REPEATED_PASS_INSTANCE(PermuteBlocks);
  REPEATED_PASS_INSTANCE(PermuteFunctionParameters);
  REPEATED_PASS_INSTANCE(PermuteInstructions);
  REPEATED_PASS_INSTANCE(PropagateInstructionsDown);
  REPEATED_PASS_INSTANCE(PropagateInstructionsUp);
  REPEATED_PASS_INSTANCE(PushIdsThroughVariables);
  REPEATED_PASS_INSTANCE(ReplaceAddsSubsMulsWithCarryingExtended);
  REPEATED_PASS_INSTANCE(ReplaceBranchesFromDeadBlocksWithExits);
  REPEATED_PASS_INSTANCE(ReplaceCopyMemoriesWithLoadsStores);
  REPEATED_PASS_INSTANCE(ReplaceCopyObjectsWithStoresLoads);
  REPEATED_PASS_INSTANCE(ReplaceLoadsStoresWithCopyMemories);
  REPEATED_PASS_INSTANCE(ReplaceIrrelevantIds);
  REPEATED_PASS_INSTANCE(ReplaceOpPhiIdsFromDeadPredecessors);
  REPEATED_PASS_INSTANCE(ReplaceOpSelectsWithConditionalBranches);
  REPEATED_PASS_INSTANCE(ReplaceParameterWithGlobal);
  REPEATED_PASS_INSTANCE(ReplaceLinearAlgebraInstructions);
  REPEATED_PASS_INSTANCE(ReplaceParamsWithStruct);
  REPEATED_PASS_INSTANCE(SplitBlocks);
  REPEATED_PASS_INSTANCE(SwapBranchConditionalOperands);
  REPEATED_PASS_INSTANCE(WrapRegionsInSelections);
  REPEATED_PASS_INSTANCE(WrapVectorSynonym);
#undef REPEATED_PASS_INSTANCE

 public:
  // Yields the sequence of fuzzer pass instances that have been registered.
  const std::vector<std::unique_ptr<FuzzerPass>>& GetPasses() const {
    return passes_;
  }

 private:
  // The distinct fuzzer pass instances that have been registered via SetPass().
  std::vector<std::unique_ptr<FuzzerPass>> passes_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_INSTANCES_H_
