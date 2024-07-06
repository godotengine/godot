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

#ifndef SOURCE_FUZZ_ADDED_FUNCTION_REDUCER_H_
#define SOURCE_FUZZ_ADDED_FUNCTION_REDUCER_H_

#include <unordered_set>
#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/shrinker.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// An auxiliary class used by Shrinker, this class takes care of using
// spirv-reduce to reduce the body of a function encoded in an AddFunction
// transformation, in case a smaller, simpler function can be added instead.
class AddedFunctionReducer {
 public:
  // Possible statuses that can result from running the shrinker.
  enum class AddedFunctionReducerResultStatus {
    kComplete,
    kReductionFailed,
  };

  struct AddedFunctionReducerResult {
    AddedFunctionReducerResultStatus status;
    std::vector<uint32_t> transformed_binary;
    protobufs::TransformationSequence applied_transformations;
    uint32_t num_reduction_attempts;
  };

  AddedFunctionReducer(
      spv_target_env target_env, MessageConsumer consumer,
      const std::vector<uint32_t>& binary_in,
      const protobufs::FactSequence& initial_facts,
      const protobufs::TransformationSequence& transformation_sequence_in,
      uint32_t index_of_add_function_transformation,
      const Shrinker::InterestingnessFunction&
          shrinker_interestingness_function,
      bool validate_during_replay, spv_validator_options validator_options,
      uint32_t shrinker_step_limit, uint32_t num_existing_shrink_attempts);

  // Disables copy/move constructor/assignment operations.
  AddedFunctionReducer(const AddedFunctionReducer&) = delete;
  AddedFunctionReducer(AddedFunctionReducer&&) = delete;
  AddedFunctionReducer& operator=(const AddedFunctionReducer&) = delete;
  AddedFunctionReducer& operator=(AddedFunctionReducer&&) = delete;

  ~AddedFunctionReducer();

  // Invokes spirv-reduce on the function in the AddFunction transformation
  // identified by |index_of_add_function_transformation|.  Returns a sequence
  // of transformations identical to |transformation_sequence_in|, except that
  // the AddFunction transformation at |index_of_add_function_transformation|
  // might have been simplified.  The binary associated with applying the
  // resulting sequence of transformations to |binary_in| is also returned, as
  // well as the number of reduction steps that spirv-reduce made.
  //
  // On failure, an empty transformation sequence and binary are returned,
  // with a placeholder value of 0 for the number of reduction attempts.
  AddedFunctionReducerResult Run();

 private:
  // Yields, via |binary_out|, the binary obtained by applying transformations
  // [0, |index_of_added_function_| - 1] from |transformations_in_| to
  // |binary_in_|, and then adding the raw function encoded in
  // |transformations_in_[index_of_added_function_]| (without adapting that
  // function to make it livesafe).  This function has |added_function_id_| as
  // its result id.
  //
  // The ids associated with all global variables in |binary_out| that had the
  // "irrelevant pointee value" fact are also returned via
  // |irrelevant_pointee_global_variables|.
  //
  // The point of this function is that spirv-reduce can subsequently be applied
  // to function |added_function_id_| in |binary_out|.  By construction,
  // |added_function_id_| should originally manipulate globals for which
  // "irrelevant pointee value" facts hold.  The set
  // |irrelevant_pointee_global_variables| can be used to force spirv-reduce
  // to preserve this, to avoid the reduced function ending up manipulating
  // other global variables of the SPIR-V module, potentially changing their
  // value and thus changing the semantics of the module.
  void ReplayPrefixAndAddFunction(
      std::vector<uint32_t>* binary_out,
      std::unordered_set<uint32_t>* irrelevant_pointee_global_variables) const;

  // This is the interestingness function that will be used by spirv-reduce
  // when shrinking the added function.
  //
  // For |binary_under_reduction| to be deemed interesting, the following
  // conditions must hold:
  // - The function with id |added_function_id_| in |binary_under_reduction|
  //   must only reference global variables in
  //   |irrelevant_pointee_global_variables|.  This avoids the reduced function
  //   changing the semantics of the original SPIR-V module.
  // - It must be possible to successfully replay the transformations in
  //   |transformation_sequence_in_|, adapted so that the function added by the
  //   transformation at |index_of_add_function_transformation_| is replaced by
  //   the function with id |added_function_id_| in |binary_under_reduction|,
  //   to |binary_in| (starting with initial facts |initial_facts_|).
  // - All the transformations in this sequence must be successfully applied
  //   during replay.
  // - The resulting binary must be interesting according to
  //   |shrinker_interestingness_function_|.
  bool InterestingnessFunctionForReducingAddedFunction(
      const std::vector<uint32_t>& binary_under_reduction,
      const std::unordered_set<uint32_t>& irrelevant_pointee_global_variables);

  // Starting with |binary_in_| and |initial_facts_|, the transformations in
  // |transformation_sequence_in_| are replayed.  However, the transformation
  // at index |index_of_add_function_transformation_| of
  // |transformation_sequence_in_| -- which is guaranteed to be an AddFunction
  // transformation -- is adapted so that the function to be added is replaced
  // with the function in |binary_under_reduction| with id |added_function_id_|.
  //
  // The binary resulting from this replay is returned via |binary_out|, and the
  // adapted transformation sequence via |transformation_sequence_out|.
  void ReplayAdaptedTransformations(
      const std::vector<uint32_t>& binary_under_reduction,
      std::vector<uint32_t>* binary_out,
      protobufs::TransformationSequence* transformation_sequence_out) const;

  // Returns the id of the function to be added by the AddFunction
  // transformation at
  // |transformation_sequence_in_[index_of_add_function_transformation_]|.
  uint32_t GetAddedFunctionId() const;

  // Target environment.
  const spv_target_env target_env_;

  // Message consumer.
  MessageConsumer consumer_;

  // The initial binary to which transformations are applied -- i.e., the
  // binary to which spirv-fuzz originally applied transformations.
  const std::vector<uint32_t>& binary_in_;

  // Initial facts about |binary_in_|.
  const protobufs::FactSequence& initial_facts_;

  // A set of transformations that can be successfully applied to |binary_in_|.
  const protobufs::TransformationSequence& transformation_sequence_in_;

  // An index into |transformation_sequence_in_| referring to an AddFunction
  // transformation.  This is the transformation to be simplified using
  // spirv-reduce.
  const uint32_t index_of_add_function_transformation_;

  // The interestingness function that has been provided to guide the
  // overall shrinking process.  The AddFunction transformation being simplified
  // by this class should still -- when applied in conjunction with the other
  // transformations in |transformation_sequence_in_| -- lead to a binary that
  // is deemed interesting by this function.
  const Shrinker::InterestingnessFunction& shrinker_interestingness_function_;

  // Determines whether to check for validity during the replaying of
  // transformations.
  const bool validate_during_replay_;

  // Options to control validation.
  spv_validator_options validator_options_;

  // The step limit associated with the overall shrinking process.
  const uint32_t shrinker_step_limit_;

  // The number of shrink attempts that had been applied prior to invoking this
  // AddedFunctionReducer instance.
  const uint32_t num_existing_shrink_attempts_;

  // Tracks the number of attempts that spirv-reduce has invoked its
  // interestingness function, which it does once at the start of reduction,
  // and then once more each time it makes a reduction step.
  uint32_t num_reducer_interestingness_function_invocations_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_ADDED_FUNCTION_REDUCER_H_
