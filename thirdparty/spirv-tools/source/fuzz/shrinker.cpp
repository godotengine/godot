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

#include "source/fuzz/shrinker.h"

#include <sstream>

#include "source/fuzz/added_function_reducer.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/replayer.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

namespace {

// A helper to get the size of a protobuf transformation sequence in a less
// verbose manner.
uint32_t NumRemainingTransformations(
    const protobufs::TransformationSequence& transformation_sequence) {
  return static_cast<uint32_t>(transformation_sequence.transformation_size());
}

// A helper to return a transformation sequence identical to |transformations|,
// except that a chunk of size |chunk_size| starting from |chunk_index| x
// |chunk_size| is removed (or as many transformations as available if the whole
// chunk is not).
protobufs::TransformationSequence RemoveChunk(
    const protobufs::TransformationSequence& transformations,
    uint32_t chunk_index, uint32_t chunk_size) {
  uint32_t lower = chunk_index * chunk_size;
  uint32_t upper = std::min((chunk_index + 1) * chunk_size,
                            NumRemainingTransformations(transformations));
  assert(lower < upper);
  assert(upper <= NumRemainingTransformations(transformations));
  protobufs::TransformationSequence result;
  for (uint32_t j = 0; j < NumRemainingTransformations(transformations); j++) {
    if (j >= lower && j < upper) {
      continue;
    }
    protobufs::Transformation transformation =
        transformations.transformation()[j];
    *result.mutable_transformation()->Add() = transformation;
  }
  return result;
}

}  // namespace

Shrinker::Shrinker(
    spv_target_env target_env, MessageConsumer consumer,
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const protobufs::TransformationSequence& transformation_sequence_in,
    const InterestingnessFunction& interestingness_function,
    uint32_t step_limit, bool validate_during_replay,
    spv_validator_options validator_options)
    : target_env_(target_env),
      consumer_(std::move(consumer)),
      binary_in_(binary_in),
      initial_facts_(initial_facts),
      transformation_sequence_in_(transformation_sequence_in),
      interestingness_function_(interestingness_function),
      step_limit_(step_limit),
      validate_during_replay_(validate_during_replay),
      validator_options_(validator_options) {}

Shrinker::~Shrinker() = default;

Shrinker::ShrinkerResult Shrinker::Run() {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  SpirvTools tools(target_env_);
  if (!tools.IsValid()) {
    consumer_(SPV_MSG_ERROR, nullptr, {},
              "Failed to create SPIRV-Tools interface; stopping.");
    return {Shrinker::ShrinkerResultStatus::kFailedToCreateSpirvToolsInterface,
            std::vector<uint32_t>(), protobufs::TransformationSequence()};
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in_[0], binary_in_.size(), validator_options_)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Initial binary is invalid; stopping.");
    return {Shrinker::ShrinkerResultStatus::kInitialBinaryInvalid,
            std::vector<uint32_t>(), protobufs::TransformationSequence()};
  }

  // Run a replay of the initial transformation sequence to check that it
  // succeeds.
  auto initial_replay_result =
      Replayer(target_env_, consumer_, binary_in_, initial_facts_,
               transformation_sequence_in_,
               static_cast<uint32_t>(
                   transformation_sequence_in_.transformation_size()),
               validate_during_replay_, validator_options_)
          .Run();
  if (initial_replay_result.status !=
      Replayer::ReplayerResultStatus::kComplete) {
    return {ShrinkerResultStatus::kReplayFailed, std::vector<uint32_t>(),
            protobufs::TransformationSequence()};
  }
  // Get the binary that results from running these transformations, and the
  // subsequence of the initial transformations that actually apply (in
  // principle this could be a strict subsequence).
  std::vector<uint32_t> current_best_binary;
  initial_replay_result.transformed_module->module()->ToBinary(
      &current_best_binary, false);
  protobufs::TransformationSequence current_best_transformations =
      std::move(initial_replay_result.applied_transformations);

  // Check that the binary produced by applying the initial transformations is
  // indeed interesting.
  if (!interestingness_function_(current_best_binary, 0)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Initial binary is not interesting; stopping.");
    return {ShrinkerResultStatus::kInitialBinaryNotInteresting,
            std::vector<uint32_t>(), protobufs::TransformationSequence()};
  }

  uint32_t attempt = 0;  // Keeps track of the number of shrink attempts that
                         // have been tried, whether successful or not.

  uint32_t chunk_size =
      std::max(1u, NumRemainingTransformations(current_best_transformations) /
                       2);  // The number of contiguous transformations that the
                            // shrinker will try to remove in one go; starts
                            // high and decreases during the shrinking process.

  // Keep shrinking until we:
  // - reach the step limit,
  // - run out of transformations to remove, or
  // - cannot make the chunk size any smaller.
  while (attempt < step_limit_ &&
         !current_best_transformations.transformation().empty() &&
         chunk_size > 0) {
    bool progress_this_round =
        false;  // Used to decide whether to make the chunk size with which we
                // remove transformations smaller.  If we managed to remove at
                // least one chunk of transformations at a particular chunk
                // size, we set this flag so that we do not yet decrease the
                // chunk size.

    assert(chunk_size <=
               NumRemainingTransformations(current_best_transformations) &&
           "Chunk size should never exceed the number of transformations that "
           "remain.");

    // The number of chunks is the ceiling of (#remaining_transformations /
    // chunk_size).
    const uint32_t num_chunks =
        (NumRemainingTransformations(current_best_transformations) +
         chunk_size - 1) /
        chunk_size;
    assert(num_chunks >= 1 && "There should be at least one chunk.");
    assert(num_chunks * chunk_size >=
               NumRemainingTransformations(current_best_transformations) &&
           "All transformations should be in some chunk.");

    // We go through the transformations in reverse, in chunks of size
    // |chunk_size|, using |chunk_index| to track which chunk to try removing
    // next.  The loop exits early if we reach the shrinking step limit.
    for (int chunk_index = num_chunks - 1;
         attempt < step_limit_ && chunk_index >= 0; chunk_index--) {
      // Remove a chunk of transformations according to the current index and
      // chunk size.
      auto transformations_with_chunk_removed =
          RemoveChunk(current_best_transformations,
                      static_cast<uint32_t>(chunk_index), chunk_size);

      // Replay the smaller sequence of transformations to get a next binary and
      // transformation sequence. Note that the transformations arising from
      // replay might be even smaller than the transformations with the chunk
      // removed, because removing those transformations might make further
      // transformations inapplicable.
      auto replay_result =
          Replayer(
              target_env_, consumer_, binary_in_, initial_facts_,
              transformations_with_chunk_removed,
              static_cast<uint32_t>(
                  transformations_with_chunk_removed.transformation_size()),
              validate_during_replay_, validator_options_)
              .Run();
      if (replay_result.status != Replayer::ReplayerResultStatus::kComplete) {
        // Replay should not fail; if it does, we need to abort shrinking.
        return {ShrinkerResultStatus::kReplayFailed, std::vector<uint32_t>(),
                protobufs::TransformationSequence()};
      }

      assert(
          NumRemainingTransformations(replay_result.applied_transformations) >=
              chunk_index * chunk_size &&
          "Removing this chunk of transformations should not have an effect "
          "on earlier chunks.");

      std::vector<uint32_t> transformed_binary;
      replay_result.transformed_module->module()->ToBinary(&transformed_binary,
                                                           false);
      if (interestingness_function_(transformed_binary, attempt)) {
        // If the binary arising from the smaller transformation sequence is
        // interesting, this becomes our current best binary and transformation
        // sequence.
        current_best_binary = std::move(transformed_binary);
        current_best_transformations =
            std::move(replay_result.applied_transformations);
        progress_this_round = true;
      }
      // Either way, this was a shrink attempt, so increment our count of shrink
      // attempts.
      attempt++;
    }
    if (!progress_this_round) {
      // If we didn't manage to remove any chunks at this chunk size, try a
      // smaller chunk size.
      chunk_size /= 2;
    }
    // Decrease the chunk size until it becomes no larger than the number of
    // remaining transformations.
    while (chunk_size >
           NumRemainingTransformations(current_best_transformations)) {
      chunk_size /= 2;
    }
  }

  // We now use spirv-reduce to minimise the functions associated with any
  // AddFunction transformations that remain.
  //
  // Consider every remaining transformation.
  for (uint32_t transformation_index = 0;
       attempt < step_limit_ &&
       transformation_index <
           static_cast<uint32_t>(
               current_best_transformations.transformation_size());
       transformation_index++) {
    // Skip all transformations apart from TransformationAddFunction.
    if (!current_best_transformations.transformation(transformation_index)
             .has_add_function()) {
      continue;
    }
    // Invoke spirv-reduce on the function encoded in this AddFunction
    // transformation.  The details of this are rather involved, and so are
    // encapsulated in a separate class.
    auto added_function_reducer_result =
        AddedFunctionReducer(target_env_, consumer_, binary_in_, initial_facts_,
                             current_best_transformations, transformation_index,
                             interestingness_function_, validate_during_replay_,
                             validator_options_, step_limit_, attempt)
            .Run();
    // Reducing the added function should succeed.  If it doesn't, we report
    // a shrinking error.
    if (added_function_reducer_result.status !=
        AddedFunctionReducer::AddedFunctionReducerResultStatus::kComplete) {
      return {ShrinkerResultStatus::kAddedFunctionReductionFailed,
              std::vector<uint32_t>(), protobufs::TransformationSequence()};
    }
    assert(current_best_transformations.transformation_size() ==
               added_function_reducer_result.applied_transformations
                   .transformation_size() &&
           "The number of transformations should not have changed.");
    current_best_binary =
        std::move(added_function_reducer_result.transformed_binary);
    current_best_transformations =
        std::move(added_function_reducer_result.applied_transformations);
    // The added function reducer reports how many reduction attempts
    // spirv-reduce took when reducing the function.  We regard each of these
    // as a shrinker attempt.
    attempt += added_function_reducer_result.num_reduction_attempts;
  }

  // Indicate whether shrinking completed or was truncated due to reaching the
  // step limit.
  //
  // Either way, the output from the shrinker is the best binary we saw, and the
  // transformations that led to it.
  assert(attempt <= step_limit_);
  if (attempt == step_limit_) {
    std::stringstream strstream;
    strstream << "Shrinking did not complete; step limit " << step_limit_
              << " was reached.";
    consumer_(SPV_MSG_WARNING, nullptr, {}, strstream.str().c_str());
    return {Shrinker::ShrinkerResultStatus::kStepLimitReached,
            std::move(current_best_binary),
            std::move(current_best_transformations)};
  }
  return {Shrinker::ShrinkerResultStatus::kComplete,
          std::move(current_best_binary),
          std::move(current_best_transformations)};
}

uint32_t Shrinker::GetIdBound(const std::vector<uint32_t>& binary) const {
  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context =
      BuildModule(target_env_, consumer_, binary.data(), binary.size());
  assert(ir_context && "Error building module.");
  return ir_context->module()->id_bound();
}

}  // namespace fuzz
}  // namespace spvtools
