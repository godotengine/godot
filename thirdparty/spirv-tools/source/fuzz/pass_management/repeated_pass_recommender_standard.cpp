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

#include "source/fuzz/pass_management/repeated_pass_recommender_standard.h"

#include <numeric>

namespace spvtools {
namespace fuzz {

RepeatedPassRecommenderStandard::RepeatedPassRecommenderStandard(
    RepeatedPassInstances* pass_instances, FuzzerContext* fuzzer_context)
    : pass_instances_(pass_instances), fuzzer_context_(fuzzer_context) {}

RepeatedPassRecommenderStandard::~RepeatedPassRecommenderStandard() = default;

std::vector<FuzzerPass*>
RepeatedPassRecommenderStandard::GetFuturePassRecommendations(
    const FuzzerPass& pass) {
  if (&pass == pass_instances_->GetAddAccessChains()) {
    // - Adding access chains means there is more scope for loading and storing
    // - It could be worth making more access chains from the recently-added
    //   access chains
    return RandomOrderAndNonNull({pass_instances_->GetAddLoads(),
                                  pass_instances_->GetAddStores(),
                                  pass_instances_->GetAddAccessChains()});
  }
  if (&pass == pass_instances_->GetAddBitInstructionSynonyms()) {
    // - Adding bit instruction synonyms creates opportunities to apply synonyms
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetAddCompositeExtract()) {
    // - This transformation can introduce synonyms to the fact manager.
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetAddCompositeInserts()) {
    // - Having added inserts we will have more vectors, so there is scope for
    //   vector shuffling
    // - Adding inserts creates synonyms, which we should try to use
    // - Vector inserts can be made dynamic
    return RandomOrderAndNonNull(
        {pass_instances_->GetAddVectorShuffleInstructions(),
         pass_instances_->GetApplyIdSynonyms(),
         pass_instances_->GetMakeVectorOperationsDynamic()});
  }
  if (&pass == pass_instances_->GetAddCompositeTypes()) {
    // - More composite types gives more scope for constructing composites
    return RandomOrderAndNonNull({pass_instances_->GetConstructComposites()});
  }
  if (&pass == pass_instances_->GetAddCopyMemory()) {
    // - Recently-added copy memories could be replace with load-store pairs
    return RandomOrderAndNonNull(
        {pass_instances_->GetReplaceCopyMemoriesWithLoadsStores()});
  }
  if (&pass == pass_instances_->GetAddDeadBlocks()) {
    // - Dead blocks are great for adding function calls
    // - Dead blocks are also great for adding loads and stores
    // - The guard associated with a dead block can be obfuscated
    // - Branches from dead blocks may be replaced with exits
    return RandomOrderAndNonNull(
        {pass_instances_->GetAddFunctionCalls(), pass_instances_->GetAddLoads(),
         pass_instances_->GetAddStores(),
         pass_instances_->GetObfuscateConstants(),
         pass_instances_->GetReplaceBranchesFromDeadBlocksWithExits()});
  }
  if (&pass == pass_instances_->GetAddDeadBreaks()) {
    // - The guard of the dead break is a good candidate for obfuscation
    return RandomOrderAndNonNull({pass_instances_->GetObfuscateConstants()});
  }
  if (&pass == pass_instances_->GetAddDeadContinues()) {
    // - The guard of the dead continue is a good candidate for obfuscation
    return RandomOrderAndNonNull({pass_instances_->GetObfuscateConstants()});
  }
  if (&pass == pass_instances_->GetAddEquationInstructions()) {
    // - Equation instructions can create synonyms, which we can apply
    // - Equation instructions collaborate with one another to make synonyms, so
    //   having added some it is worth adding more
    return RandomOrderAndNonNull(
        {pass_instances_->GetApplyIdSynonyms(),
         pass_instances_->GetAddEquationInstructions()});
  }
  if (&pass == pass_instances_->GetAddFunctionCalls()) {
    // - Called functions can be inlined
    // - Irrelevant ids are created, so they can be replaced
    return RandomOrderAndNonNull({pass_instances_->GetInlineFunctions(),
                                  pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetAddGlobalVariables()) {
    // - New globals provide new possibilities for making access chains
    // - We can load from and store to new globals
    return RandomOrderAndNonNull({pass_instances_->GetAddAccessChains(),
                                  pass_instances_->GetAddLoads(),
                                  pass_instances_->GetAddStores()});
  }
  if (&pass == pass_instances_->GetAddImageSampleUnusedComponents()) {
    // - This introduces an unused component whose id is irrelevant and can be
    //   replaced
    return RandomOrderAndNonNull({pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetAddLoads()) {
    // - Loads might end up with corresponding stores, so that pairs can be
    //   replaced with memory copies
    return RandomOrderAndNonNull(
        {pass_instances_->GetReplaceLoadsStoresWithCopyMemories()});
  }
  if (&pass == pass_instances_->GetAddLocalVariables()) {
    // - New locals provide new possibilities for making access chains
    // - We can load from and store to new locals
    return RandomOrderAndNonNull({pass_instances_->GetAddAccessChains(),
                                  pass_instances_->GetAddLoads(),
                                  pass_instances_->GetAddStores()});
  }
  if (&pass == pass_instances_->GetAddLoopPreheaders()) {
    // - The loop preheader provides more scope for duplicating regions and
    //   outlining functions.
    return RandomOrderAndNonNull(
        {pass_instances_->GetDuplicateRegionsWithSelections(),
         pass_instances_->GetOutlineFunctions(),
         pass_instances_->GetWrapRegionsInSelections()});
  }
  if (&pass == pass_instances_->GetAddLoopsToCreateIntConstantSynonyms()) {
    // - New synonyms can be applied
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetAddOpPhiSynonyms()) {
    // - New synonyms can be applied
    // - If OpPhi synonyms are introduced for blocks with dead predecessors, the
    //   values consumed from dead predecessors can be replaced
    return RandomOrderAndNonNull(
        {pass_instances_->GetApplyIdSynonyms(),
         pass_instances_->GetReplaceOpPhiIdsFromDeadPredecessors()});
  }
  if (&pass == pass_instances_->GetAddParameters()) {
    // - We might be able to create interesting synonyms of new parameters.
    // - This introduces irrelevant ids, which can be replaced
    return RandomOrderAndNonNull({pass_instances_->GetAddSynonyms(),
                                  pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetAddRelaxedDecorations()) {
    // - No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetAddStores()) {
    // - Stores might end up with corresponding loads, so that pairs can be
    //   replaced with memory copies
    return RandomOrderAndNonNull(
        {pass_instances_->GetReplaceLoadsStoresWithCopyMemories()});
  }
  if (&pass == pass_instances_->GetAddSynonyms()) {
    // - New synonyms can be applied
    // - Synonym instructions use constants, which can be obfuscated
    // - Synonym instructions use irrelevant ids, which can be replaced
    // - Synonym instructions introduce addition/subtraction, which can be
    //   replaced with carrying/extended versions
    return RandomOrderAndNonNull(
        {pass_instances_->GetApplyIdSynonyms(),
         pass_instances_->GetObfuscateConstants(),
         pass_instances_->GetReplaceAddsSubsMulsWithCarryingExtended(),
         pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetAddVectorShuffleInstructions()) {
    // - Vector shuffles create synonyms that can be applied
    // - TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3806) Extract
    //    from composites.
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetApplyIdSynonyms()) {
    // - No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetConstructComposites()) {
    // - TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3806): Extract
    //    from composites.
    return RandomOrderAndNonNull({});
  }
  if (&pass == pass_instances_->GetCopyObjects()) {
    // - Object copies create synonyms that can be applied
    // - OpCopyObject can be replaced with a store/load pair
    return RandomOrderAndNonNull(
        {pass_instances_->GetApplyIdSynonyms(),
         pass_instances_->GetReplaceCopyObjectsWithStoresLoads()});
  }
  if (&pass == pass_instances_->GetDonateModules()) {
    // - New functions in the module can be called
    // - Donated dead functions produce irrelevant ids, which can be replaced
    // - Donated functions are good candidates for having their returns merged
    // - Donated dead functions may allow branches to be replaced with exits
    return RandomOrderAndNonNull(
        {pass_instances_->GetAddFunctionCalls(),
         pass_instances_->GetReplaceIrrelevantIds(),
         pass_instances_->GetMergeFunctionReturns(),
         pass_instances_->GetReplaceBranchesFromDeadBlocksWithExits()});
  }
  if (&pass == pass_instances_->GetDuplicateRegionsWithSelections()) {
    // - Parts of duplicated regions can be outlined
    return RandomOrderAndNonNull({pass_instances_->GetOutlineFunctions()});
  }
  if (&pass == pass_instances_->GetExpandVectorReductions()) {
    // - Adding OpAny and OpAll synonyms creates opportunities to apply synonyms
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetFlattenConditionalBranches()) {
    // - Parts of flattened selections can be outlined
    // - The flattening transformation introduces constants and irrelevant ids
    //   for enclosing hard-to-flatten operations; these can be obfuscated or
    //   replaced
    return RandomOrderAndNonNull({pass_instances_->GetObfuscateConstants(),
                                  pass_instances_->GetOutlineFunctions(),
                                  pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetInlineFunctions()) {
    // - Parts of inlined functions can be outlined again
    return RandomOrderAndNonNull({pass_instances_->GetOutlineFunctions()});
  }
  if (&pass == pass_instances_->GetInvertComparisonOperators()) {
    // - No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetMakeVectorOperationsDynamic()) {
    // - No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetMergeBlocks()) {
    // - Having merged some blocks it may be interesting to split them in a
    //   different way
    return RandomOrderAndNonNull({pass_instances_->GetSplitBlocks()});
  }
  if (&pass == pass_instances_->GetMergeFunctionReturns()) {
    // - Functions without early returns are more likely to be able to be
    //   inlined.
    return RandomOrderAndNonNull({pass_instances_->GetInlineFunctions()});
  }
  if (&pass == pass_instances_->GetMutatePointers()) {
    // - This creates irrelevant ids, which can be replaced
    return RandomOrderAndNonNull({pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetObfuscateConstants()) {
    // - No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetOutlineFunctions()) {
    // - This creates more functions, which can be called
    // - Inlining the function for the region that was outlined might also be
    //   fruitful; it will be inlined in a different form
    return RandomOrderAndNonNull({pass_instances_->GetAddFunctionCalls(),
                                  pass_instances_->GetInlineFunctions()});
  }
  if (&pass == pass_instances_->GetPermuteBlocks()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetPermuteFunctionParameters()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetPermuteInstructions()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetPropagateInstructionsDown()) {
    // - This fuzzer pass might create new synonyms that can later be applied.
    // - This fuzzer pass might create irrelevant ids that can later be
    //   replaced.
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms(),
                                  pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetPropagateInstructionsUp()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetPushIdsThroughVariables()) {
    // - This pass creates synonyms, so it is worth applying them
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms()});
  }
  if (&pass == pass_instances_->GetReplaceAddsSubsMulsWithCarryingExtended()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceBranchesFromDeadBlocksWithExits()) {
    // - Changing a branch to OpReturnValue introduces an irrelevant id, which
    //   can be replaced
    return RandomOrderAndNonNull({pass_instances_->GetReplaceIrrelevantIds()});
  }
  if (&pass == pass_instances_->GetReplaceCopyMemoriesWithLoadsStores()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceCopyObjectsWithStoresLoads()) {
    // - We may end up with load/store pairs that could be used to create memory
    //   copies
    return RandomOrderAndNonNull(
        {pass_instances_->GetReplaceLoadsStoresWithCopyMemories()});
  }
  if (&pass == pass_instances_->GetReplaceIrrelevantIds()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceLinearAlgebraInstructions()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceLoadsStoresWithCopyMemories()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceOpPhiIdsFromDeadPredecessors()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceOpSelectsWithConditionalBranches()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceParameterWithGlobal()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetReplaceParamsWithStruct()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetSplitBlocks()) {
    // - More blocks means more chances for adding dead breaks/continues, and
    //   for adding dead blocks
    return RandomOrderAndNonNull({pass_instances_->GetAddDeadBreaks(),
                                  pass_instances_->GetAddDeadContinues(),
                                  pass_instances_->GetAddDeadBlocks()});
  }
  if (&pass == pass_instances_->GetSwapBranchConditionalOperands()) {
    // No obvious follow-on passes
    return {};
  }
  if (&pass == pass_instances_->GetWrapRegionsInSelections()) {
    // - This pass uses an irrelevant boolean constant - we can replace it with
    //   something more interesting.
    // - We can obfuscate that very constant as well.
    // - We can flatten created selection construct.
    return RandomOrderAndNonNull(
        {pass_instances_->GetObfuscateConstants(),
         pass_instances_->GetReplaceIrrelevantIds(),
         pass_instances_->GetFlattenConditionalBranches()});
  }
  if (&pass == pass_instances_->GetWrapVectorSynonym()) {
    // This transformation introduces synonym facts and irrelevant ids.
    return RandomOrderAndNonNull({pass_instances_->GetApplyIdSynonyms(),
                                  pass_instances_->GetReplaceIrrelevantIds()});
  }

  assert(false && "Unreachable: every fuzzer pass should be dealt with.");
  return {};
}

std::vector<FuzzerPass*> RepeatedPassRecommenderStandard::RandomOrderAndNonNull(
    const std::vector<FuzzerPass*>& passes) {
  std::vector<uint32_t> indices(passes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<FuzzerPass*> result;
  while (!indices.empty()) {
    FuzzerPass* maybe_pass =
        passes[fuzzer_context_->RemoveAtRandomIndex(&indices)];
    if (maybe_pass != nullptr &&
        fuzzer_context_->ChoosePercentage(
            fuzzer_context_
                ->GetChanceOfAcceptingRepeatedPassRecommendation())) {
      result.push_back(maybe_pass);
    }
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
