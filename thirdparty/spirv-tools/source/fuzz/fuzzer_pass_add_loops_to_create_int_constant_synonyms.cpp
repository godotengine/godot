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

#include "source/fuzz/fuzzer_pass_add_loops_to_create_int_constant_synonyms.h"

#include "source/fuzz/call_graph.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_loop_to_create_int_constant_synonym.h"

namespace spvtools {
namespace fuzz {
namespace {
uint32_t kMaxNestingDepth = 4;
}  // namespace

FuzzerPassAddLoopsToCreateIntConstantSynonyms::
    FuzzerPassAddLoopsToCreateIntConstantSynonyms(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddLoopsToCreateIntConstantSynonyms::Apply() {
  std::vector<uint32_t> constants;

  // Choose the constants for which to create synonyms.
  for (auto constant_def : GetIRContext()->GetConstants()) {
    // Randomly decide whether to consider this constant.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfCreatingIntSynonymsUsingLoops())) {
      continue;
    }

    auto constant = GetIRContext()->get_constant_mgr()->FindDeclaredConstant(
        constant_def->result_id());

    // We do not consider irrelevant constants
    if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
            constant_def->result_id())) {
      continue;
    }

    // We only consider integer constants (scalar or vector).
    if (!constant->AsIntConstant() &&
        !(constant->AsVectorConstant() &&
          constant->AsVectorConstant()->component_type()->AsInteger())) {
      continue;
    }

    constants.push_back(constant_def->result_id());
  }

  std::vector<uint32_t> blocks;

  // Get a list of all the blocks before which we can add a loop creating a new
  // synonym. We cannot apply the transformation while iterating over the
  // module, because we are going to add new blocks.
  for (auto& function : *GetIRContext()->module()) {
    // Consider all non-dead blocks reachable from the first block of the
    // function.
    GetIRContext()->cfg()->ForEachBlockInPostOrder(
        &*function.begin(), [this, &blocks](opt::BasicBlock* block) {
          if (!GetTransformationContext()->GetFactManager()->BlockIsDead(
                  block->id())) {
            blocks.push_back(block->id());
          }
        });
  }

  // Make sure that the module has an OpTypeBool instruction, and 32-bit signed
  // integer constants 0 and 1, adding them if necessary.
  FindOrCreateBoolType();
  FindOrCreateIntegerConstant({0}, 32, true, false);
  FindOrCreateIntegerConstant({1}, 32, true, false);

  // Compute the call graph. We can use this for any further computation, since
  // we are not adding or removing functions or function calls.
  auto call_graph = CallGraph(GetIRContext());

  // Consider each constant and each block.
  for (uint32_t constant_id : constants) {
    // Choose one of the blocks.
    uint32_t block_id = blocks[GetFuzzerContext()->RandomIndex(blocks)];

    // Adjust the block so that the transformation can be applied.
    auto block = GetIRContext()->get_instr_block(block_id);

    // If the block is a loop header, add a simple preheader. We can do this
    // because we have excluded all the non-reachable headers.
    if (block->IsLoopHeader()) {
      block = GetOrCreateSimpleLoopPreheader(block->id());
      block_id = block->id();
    }

    assert(!block->IsLoopHeader() &&
           "The block cannot be a loop header at this point.");

    // If the block is a merge block, a continue block or it does not have
    // exactly 1 predecessor, split it after any OpPhi or OpVariable
    // instructions.
    if (GetIRContext()->GetStructuredCFGAnalysis()->IsMergeBlock(block->id()) ||
        GetIRContext()->GetStructuredCFGAnalysis()->IsContinueBlock(
            block->id()) ||
        GetIRContext()->cfg()->preds(block->id()).size() != 1) {
      block = SplitBlockAfterOpPhiOrOpVariable(block->id());
      block_id = block->id();
    }

    // Randomly decide the values for the number of iterations and the step
    // value, and compute the initial value accordingly.

    // The maximum number of iterations depends on the maximum possible loop
    // nesting depth of the block, computed interprocedurally, i.e. also
    // considering the possibility that the enclosing function is called inside
    // a loop. It is:
    // - 1 if the nesting depth is >= kMaxNestingDepth
    // - 2^(kMaxNestingDepth - nesting_depth) otherwise
    uint32_t max_nesting_depth =
        call_graph.GetMaxCallNestingDepth(block->GetParent()->result_id()) +
        GetIRContext()->GetStructuredCFGAnalysis()->LoopNestingDepth(
            block->id());
    uint32_t num_iterations =
        max_nesting_depth >= kMaxNestingDepth
            ? 1
            : GetFuzzerContext()->GetRandomNumberOfLoopIterations(
                  1u << (kMaxNestingDepth - max_nesting_depth));

    // Find or create the corresponding constant containing the number of
    // iterations.
    uint32_t num_iterations_id =
        FindOrCreateIntegerConstant({num_iterations}, 32, true, false);

    // Find the other constants.
    // We use 64-bit values and then use the bits that we need. We find the
    // step value (S) randomly and then compute the initial value (I) using
    // the equation I = C + S*N.
    uint32_t initial_value_id = 0;
    uint32_t step_value_id = 0;

    // Get the content of the existing constant.
    const auto constant =
        GetIRContext()->get_constant_mgr()->FindDeclaredConstant(constant_id);
    const auto constant_type_id =
        GetIRContext()->get_def_use_mgr()->GetDef(constant_id)->type_id();

    if (constant->AsIntConstant()) {
      // The constant is a scalar integer.

      std::tie(initial_value_id, step_value_id) =
          FindSuitableStepAndInitialValueConstants(
              constant->GetZeroExtendedValue(),
              constant->type()->AsInteger()->width(),
              constant->type()->AsInteger()->IsSigned(), num_iterations);
    } else {
      // The constant is a vector of integers.
      assert(constant->AsVectorConstant() &&
             constant->AsVectorConstant()->component_type()->AsInteger() &&
             "If the program got here, the constant should be a vector of "
             "integers.");

      // Find a constant for each component of the initial value and the step
      // values.
      std::vector<uint32_t> initial_value_component_ids;
      std::vector<uint32_t> step_value_component_ids;

      // Get the value, width and signedness of the components.
      std::vector<uint64_t> component_values;
      for (auto component : constant->AsVectorConstant()->GetComponents()) {
        component_values.push_back(component->GetZeroExtendedValue());
      }
      uint32_t bit_width =
          constant->AsVectorConstant()->component_type()->AsInteger()->width();
      uint32_t is_signed = constant->AsVectorConstant()
                               ->component_type()
                               ->AsInteger()
                               ->IsSigned();

      for (uint64_t component_val : component_values) {
        uint32_t initial_val_id;
        uint32_t step_val_id;
        std::tie(initial_val_id, step_val_id) =
            FindSuitableStepAndInitialValueConstants(component_val, bit_width,
                                                     is_signed, num_iterations);
        initial_value_component_ids.push_back(initial_val_id);
        step_value_component_ids.push_back(step_val_id);
      }

      // Find or create the vector constants.
      initial_value_id = FindOrCreateCompositeConstant(
          initial_value_component_ids, constant_type_id, false);
      step_value_id = FindOrCreateCompositeConstant(step_value_component_ids,
                                                    constant_type_id, false);
    }

    assert(initial_value_id && step_value_id &&
           "|initial_value_id| and |step_value_id| should have been defined.");

    // Randomly decide whether to have two blocks (or just one) in the new
    // loop.
    uint32_t additional_block_id =
        GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfHavingTwoBlocksInLoopToCreateIntSynonym())
            ? GetFuzzerContext()->GetFreshId()
            : 0;

    // Add the loop and create the synonym.
    ApplyTransformation(TransformationAddLoopToCreateIntConstantSynonym(
        constant_id, initial_value_id, step_value_id, num_iterations_id,
        block_id, GetFuzzerContext()->GetFreshId(),
        GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
        GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
        GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
        additional_block_id));
  }
}

std::pair<uint32_t, uint32_t> FuzzerPassAddLoopsToCreateIntConstantSynonyms::
    FindSuitableStepAndInitialValueConstants(uint64_t constant_val,
                                             uint32_t bit_width, bool is_signed,
                                             uint32_t num_iterations) {
  // Choose the step value randomly and compute the initial value accordingly.
  // The result of |initial_value| could overflow, but this is OK, since
  // the transformation takes overflows into consideration (the equation still
  // holds as long as the last |bit_width| bits of C and of (I-S*N) match).
  uint64_t step_value =
      GetFuzzerContext()->GetRandomValueForStepConstantInLoop();
  uint64_t initial_value = constant_val + step_value * num_iterations;

  uint32_t initial_val_id = FindOrCreateIntegerConstant(
      fuzzerutil::IntToWords(initial_value, bit_width, is_signed), bit_width,
      is_signed, false);

  uint32_t step_val_id = FindOrCreateIntegerConstant(
      fuzzerutil::IntToWords(step_value, bit_width, is_signed), bit_width,
      is_signed, false);

  return {initial_val_id, step_val_id};
}

}  // namespace fuzz
}  // namespace spvtools