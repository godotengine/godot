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

#include "source/fuzz/added_function_reducer.h"

#include "source/fuzz/instruction_message.h"
#include "source/fuzz/replayer.h"
#include "source/fuzz/transformation_add_function.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/reduce/reducer.h"

namespace spvtools {
namespace fuzz {

AddedFunctionReducer::AddedFunctionReducer(
    spv_target_env target_env, MessageConsumer consumer,
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const protobufs::TransformationSequence& transformation_sequence_in,
    uint32_t index_of_add_function_transformation,
    const Shrinker::InterestingnessFunction& shrinker_interestingness_function,
    bool validate_during_replay, spv_validator_options validator_options,
    uint32_t shrinker_step_limit, uint32_t num_existing_shrink_attempts)
    : target_env_(target_env),
      consumer_(std::move(consumer)),
      binary_in_(binary_in),
      initial_facts_(initial_facts),
      transformation_sequence_in_(transformation_sequence_in),
      index_of_add_function_transformation_(
          index_of_add_function_transformation),
      shrinker_interestingness_function_(shrinker_interestingness_function),
      validate_during_replay_(validate_during_replay),
      validator_options_(validator_options),
      shrinker_step_limit_(shrinker_step_limit),
      num_existing_shrink_attempts_(num_existing_shrink_attempts),
      num_reducer_interestingness_function_invocations_(0) {}

AddedFunctionReducer::~AddedFunctionReducer() = default;

AddedFunctionReducer::AddedFunctionReducerResult AddedFunctionReducer::Run() {
  // Replay all transformations before the AddFunction transformation, then
  // add the raw function associated with the AddFunction transformation.
  std::vector<uint32_t> binary_to_reduce;
  std::unordered_set<uint32_t> irrelevant_pointee_global_variables;
  ReplayPrefixAndAddFunction(&binary_to_reduce,
                             &irrelevant_pointee_global_variables);

  // Set up spirv-reduce to use our very specific interestingness function.
  reduce::Reducer reducer(target_env_);
  reducer.SetMessageConsumer(consumer_);
  reducer.AddDefaultReductionPasses();
  reducer.SetInterestingnessFunction(
      [this, &irrelevant_pointee_global_variables](
          const std::vector<uint32_t>& binary_under_reduction,
          uint32_t /*unused*/) {
        return InterestingnessFunctionForReducingAddedFunction(
            binary_under_reduction, irrelevant_pointee_global_variables);
      });

  // Instruct spirv-reduce to only target the function with the id associated
  // with the AddFunction transformation that we care about.
  spvtools::ReducerOptions reducer_options;
  reducer_options.set_target_function(GetAddedFunctionId());
  // Bound the number of reduction steps that spirv-reduce can make according
  // to the overall shrinker step limit and the number of shrink attempts that
  // have already been tried.
  assert(shrinker_step_limit_ > num_existing_shrink_attempts_ &&
         "The added function reducer should not have been invoked.");
  reducer_options.set_step_limit(shrinker_step_limit_ -
                                 num_existing_shrink_attempts_);

  // Run spirv-reduce.
  std::vector<uint32_t> reduced_binary;
  auto reducer_result =
      reducer.Run(std::move(binary_to_reduce), &reduced_binary, reducer_options,
                  validator_options_);
  if (reducer_result != reduce::Reducer::kComplete &&
      reducer_result != reduce::Reducer::kReachedStepLimit) {
    return {AddedFunctionReducerResultStatus::kReductionFailed,
            std::vector<uint32_t>(), protobufs::TransformationSequence(), 0};
  }

  // Provide the outer shrinker with an adapted sequence of transformations in
  // which the AddFunction transformation of interest has been simplified to use
  // the version of the added function that appears in |reduced_binary|.
  std::vector<uint32_t> binary_out;
  protobufs::TransformationSequence transformation_sequence_out;
  ReplayAdaptedTransformations(reduced_binary, &binary_out,
                               &transformation_sequence_out);
  // We subtract 1 from |num_reducer_interestingness_function_invocations_| to
  // account for the fact that spirv-reduce invokes its interestingness test
  // once before reduction commences in order to check that the initial module
  // is interesting.
  assert(num_reducer_interestingness_function_invocations_ > 0 &&
         "At a minimum spirv-reduce should have invoked its interestingness "
         "test once.");
  return {AddedFunctionReducerResultStatus::kComplete, std::move(binary_out),
          std::move(transformation_sequence_out),
          num_reducer_interestingness_function_invocations_ - 1};
}

bool AddedFunctionReducer::InterestingnessFunctionForReducingAddedFunction(
    const std::vector<uint32_t>& binary_under_reduction,
    const std::unordered_set<uint32_t>& irrelevant_pointee_global_variables) {
  uint32_t counter_for_shrinker_interestingness_function =
      num_existing_shrink_attempts_ +
      num_reducer_interestingness_function_invocations_;
  num_reducer_interestingness_function_invocations_++;

  // The reduced version of the added function must be limited to accessing
  // global variables appearing in |irrelevant_pointee_global_variables|.  This
  // is to guard against the possibility of spirv-reduce changing a reference
  // to an irrelevant global to a reference to a regular global variable, which
  // could cause the added function to change the semantics of the original
  // module.
  auto ir_context =
      BuildModule(target_env_, consumer_, binary_under_reduction.data(),
                  binary_under_reduction.size());
  assert(ir_context != nullptr && "The binary should be parsable.");
  for (auto& type_or_value : ir_context->module()->types_values()) {
    if (type_or_value.opcode() != spv::Op::OpVariable) {
      continue;
    }
    if (irrelevant_pointee_global_variables.count(type_or_value.result_id())) {
      continue;
    }
    if (!ir_context->get_def_use_mgr()->WhileEachUse(
            &type_or_value,
            [this, &ir_context](opt::Instruction* user,
                                uint32_t /*unused*/) -> bool {
              auto block = ir_context->get_instr_block(user);
              if (block != nullptr &&
                  block->GetParent()->result_id() == GetAddedFunctionId()) {
                return false;
              }
              return true;
            })) {
      return false;
    }
  }

  // For the binary to be deemed interesting, it must be possible to
  // successfully apply all the transformations, with the transformation at
  // index |index_of_add_function_transformation_| simplified to use the version
  // of the added function from |binary_under_reduction|.
  //
  // This might not be the case: spirv-reduce might have removed a chunk of the
  // added function on which future transformations depend.
  //
  // This is an optimization: the assumption is that having already shrunk the
  // transformation sequence down to minimal form, all transformations have a
  // role to play, and it's almost certainly a waste of time to invoke the
  // shrinker's interestingness function if we have eliminated transformations
  // that the shrinker previously tried to -- but could not -- eliminate.
  std::vector<uint32_t> binary_out;
  protobufs::TransformationSequence modified_transformations;
  ReplayAdaptedTransformations(binary_under_reduction, &binary_out,
                               &modified_transformations);
  if (transformation_sequence_in_.transformation_size() !=
      modified_transformations.transformation_size()) {
    return false;
  }

  // The resulting binary must be deemed interesting according to the shrinker's
  // interestingness function.
  return shrinker_interestingness_function_(
      binary_out, counter_for_shrinker_interestingness_function);
}

void AddedFunctionReducer::ReplayPrefixAndAddFunction(
    std::vector<uint32_t>* binary_out,
    std::unordered_set<uint32_t>* irrelevant_pointee_global_variables) const {
  assert(transformation_sequence_in_
             .transformation(index_of_add_function_transformation_)
             .has_add_function() &&
         "A TransformationAddFunction is required at the given index.");

  auto replay_result = Replayer(target_env_, consumer_, binary_in_,
                                initial_facts_, transformation_sequence_in_,
                                index_of_add_function_transformation_,
                                validate_during_replay_, validator_options_)
                           .Run();
  assert(replay_result.status == Replayer::ReplayerResultStatus::kComplete &&
         "Replay should succeed");
  assert(static_cast<uint32_t>(
             replay_result.applied_transformations.transformation_size()) ==
             index_of_add_function_transformation_ &&
         "All requested transformations should have applied.");

  auto* ir_context = replay_result.transformed_module.get();

  for (auto& type_or_value : ir_context->module()->types_values()) {
    if (type_or_value.opcode() != spv::Op::OpVariable) {
      continue;
    }
    if (replay_result.transformation_context->GetFactManager()
            ->PointeeValueIsIrrelevant(type_or_value.result_id())) {
      irrelevant_pointee_global_variables->insert(type_or_value.result_id());
    }
  }

  // Add the function associated with the transformation at
  // |index_of_add_function_transformation| to the module.  By construction this
  // should succeed.
  const protobufs::TransformationAddFunction&
      transformation_add_function_message =
          transformation_sequence_in_
              .transformation(index_of_add_function_transformation_)
              .add_function();
  bool success = TransformationAddFunction(transformation_add_function_message)
                     .TryToAddFunction(ir_context);
  (void)success;  // Keep release mode compilers happy.
  assert(success && "Addition of the function should have succeeded.");

  // Get the binary representation of the module with this function added.
  ir_context->module()->ToBinary(binary_out, false);
}

void AddedFunctionReducer::ReplayAdaptedTransformations(
    const std::vector<uint32_t>& binary_under_reduction,
    std::vector<uint32_t>* binary_out,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  assert(index_of_add_function_transformation_ <
             static_cast<uint32_t>(
                 transformation_sequence_in_.transformation_size()) &&
         "The relevant add function transformation must be present.");
  std::unique_ptr<opt::IRContext> ir_context_under_reduction =
      BuildModule(target_env_, consumer_, binary_under_reduction.data(),
                  binary_under_reduction.size());
  assert(ir_context_under_reduction && "Error building module.");

  protobufs::TransformationSequence modified_transformations;
  for (uint32_t i = 0;
       i <
       static_cast<uint32_t>(transformation_sequence_in_.transformation_size());
       i++) {
    if (i == index_of_add_function_transformation_) {
      protobufs::TransformationAddFunction modified_add_function =
          transformation_sequence_in_
              .transformation(index_of_add_function_transformation_)
              .add_function();
      assert(GetAddedFunctionId() ==
                 modified_add_function.instruction(0).result_id() &&
             "Unexpected result id for added function.");
      modified_add_function.clear_instruction();
      for (auto& function : *ir_context_under_reduction->module()) {
        if (function.result_id() != GetAddedFunctionId()) {
          continue;
        }
        function.ForEachInst(
            [&modified_add_function](const opt::Instruction* instruction) {
              *modified_add_function.add_instruction() =
                  MakeInstructionMessage(instruction);
            });
      }
      assert(modified_add_function.instruction_size() > 0 &&
             "Some instructions for the added function should remain.");
      *modified_transformations.add_transformation()->mutable_add_function() =
          modified_add_function;
    } else {
      *modified_transformations.add_transformation() =
          transformation_sequence_in_.transformation(i);
    }
  }
  assert(
      transformation_sequence_in_.transformation_size() ==
          modified_transformations.transformation_size() &&
      "The original and modified transformations should have the same size.");
  auto replay_result = Replayer(target_env_, consumer_, binary_in_,
                                initial_facts_, modified_transformations,
                                modified_transformations.transformation_size(),
                                validate_during_replay_, validator_options_)
                           .Run();
  assert(replay_result.status == Replayer::ReplayerResultStatus::kComplete &&
         "Replay should succeed.");
  replay_result.transformed_module->module()->ToBinary(binary_out, false);
  *transformation_sequence_out =
      std::move(replay_result.applied_transformations);
}

uint32_t AddedFunctionReducer::GetAddedFunctionId() const {
  return transformation_sequence_in_
      .transformation(index_of_add_function_transformation_)
      .add_function()
      .instruction(0)
      .result_id();
}

}  // namespace fuzz
}  // namespace spvtools
