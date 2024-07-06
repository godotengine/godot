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

#ifndef SOURCE_FUZZ_FUZZER_H_
#define SOURCE_FUZZ_FUZZER_H_

#include <memory>
#include <utility>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/pass_management/repeated_pass_instances.h"
#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_recommender.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/random_generator.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Transforms a SPIR-V module into a semantically equivalent SPIR-V module by
// running a number of randomized fuzzer passes.
class Fuzzer {
 public:
  // Possible statuses that can result from running the fuzzer.
  enum class Status {
    kComplete,
    kModuleTooBig,
    kTransformationLimitReached,
    kFuzzerStuck,
    kFuzzerPassLedToInvalidModule,
  };

  struct Result {
    // Status of the fuzzing session.
    Status status;

    // Equals to true if new transformations were applied during the previous
    // fuzzing session.
    bool is_changed;
  };

  Fuzzer(std::unique_ptr<opt::IRContext> ir_context,
         std::unique_ptr<TransformationContext> transformation_context,
         std::unique_ptr<FuzzerContext> fuzzer_context,
         MessageConsumer consumer,
         const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers,
         bool enable_all_passes, RepeatedPassStrategy repeated_pass_strategy,
         bool validate_after_each_fuzzer_pass,
         spv_validator_options validator_options,
         bool ignore_inapplicable_transformations = true);

  // Disables copy/move constructor/assignment operations.
  Fuzzer(const Fuzzer&) = delete;
  Fuzzer(Fuzzer&&) = delete;
  Fuzzer& operator=(const Fuzzer&) = delete;
  Fuzzer& operator=(Fuzzer&&) = delete;

  ~Fuzzer();

  // Transforms |ir_context_| by running a number of randomized fuzzer passes.
  // Initial facts about the input binary and the context in which it will be
  // executed are provided with |transformation_context_|.
  // |num_of_transformations| is equal to the maximum number of transformations
  // applied in a single call to this method. This parameter is ignored if its
  // value is equal to 0. Because fuzzing cannot stop mid way through a fuzzer
  // pass, fuzzing will stop after the fuzzer pass that exceeds
  // |num_of_transformations| has completed, so that the total number of
  // transformations may be somewhat larger than this number.
  Result Run(uint32_t num_of_transformations_to_apply);

  // Returns the current IR context. It may be invalid if the Run method
  // returned Status::kFuzzerPassLedToInvalidModule previously.
  opt::IRContext* GetIRContext();

  // Returns the sequence of applied transformations.
  const protobufs::TransformationSequence& GetTransformationSequence() const;

 private:
  // A convenience method to add a repeated fuzzer pass to |pass_instances| with
  // probability |percentage_chance_of_adding_pass|%, or with probability 100%
  // if |enable_all_passes_| is true.
  //
  // All fuzzer passes take members |ir_context_|, |transformation_context_|,
  // |fuzzer_context_| and |transformation_sequence_out_| as parameters.  Extra
  // arguments can be provided via |extra_args|.
  template <typename FuzzerPassT, typename... Args>
  void MaybeAddRepeatedPass(uint32_t percentage_chance_of_adding_pass,
                            RepeatedPassInstances* pass_instances,
                            Args&&... extra_args);

  // The same as the above, with |percentage_chance_of_adding_pass| == 50%.
  template <typename FuzzerPassT, typename... Args>
  void MaybeAddRepeatedPass(RepeatedPassInstances* pass_instances,
                            Args&&... extra_args) {
    MaybeAddRepeatedPass<FuzzerPassT>(50, pass_instances,
                                      std::forward<Args>(extra_args)...);
  }

  // A convenience method to add a final fuzzer pass to |passes| with
  // probability 50%, or with probability 100% if |enable_all_passes_| is true.
  //
  // All fuzzer passes take members |ir_context_|, |transformation_context_|,
  // |fuzzer_context_| and |transformation_sequence_out_| as parameters.  Extra
  // arguments can be provided via |extra_args|.
  template <typename FuzzerPassT, typename... Args>
  void MaybeAddFinalPass(std::vector<std::unique_ptr<FuzzerPass>>* passes,
                         Args&&... extra_args);

  // Decides whether to apply more repeated passes. The probability decreases as
  // the number of transformations that have been applied increases.
  // The described probability is only applied if
  // |continue_fuzzing_probabilistically| is true.
  bool ShouldContinueRepeatedPasses(bool continue_fuzzing_probabilistically);

  // Applies |pass|, which must be a pass constructed with |ir_context|.
  // If |validate_after_each_fuzzer_pass_| is not set, true is always returned.
  // Otherwise, true is returned if and only if |ir_context| passes validation,
  // every block has its enclosing function as its parent, and every
  // instruction has a distinct unique id.
  bool ApplyPassAndCheckValidity(FuzzerPass* pass) const;

  // Message consumer that will be invoked once for each message communicated
  // from the library.
  const MessageConsumer consumer_;

  // Determines whether all passes should be enabled, vs. having passes be
  // probabilistically enabled.
  const bool enable_all_passes_;

  // Determines whether the validator should be invoked after every fuzzer pass.
  const bool validate_after_each_fuzzer_pass_;

  // Options to control validation.
  const spv_validator_options validator_options_;

  // The number of repeated fuzzer passes that have been applied is kept track
  // of, in order to enforce a hard limit on the number of times such passes
  // can be applied.
  uint32_t num_repeated_passes_applied_;

  // We use this to determine whether we can continue fuzzing incrementally
  // since the previous call to the Run method could've returned
  // kFuzzerPassLedToInvalidModule.
  bool is_valid_;

  // Intermediate representation for the module being fuzzed, which gets
  // mutated as fuzzing proceeds.
  std::unique_ptr<opt::IRContext> ir_context_;

  // Contextual information that is required in order to apply
  // transformations.
  std::unique_ptr<TransformationContext> transformation_context_;

  // Provides probabilities that control the fuzzing process.
  std::unique_ptr<FuzzerContext> fuzzer_context_;

  // The sequence of transformations that have been applied during fuzzing. It
  // is initially empty and grows as fuzzer passes are applied.
  protobufs::TransformationSequence transformation_sequence_out_;

  // This object contains instances of all fuzzer passes that will participate
  // in the fuzzing.
  RepeatedPassInstances pass_instances_;

  // This object defines the recommendation logic for fuzzer passes.
  std::unique_ptr<RepeatedPassRecommender> repeated_pass_recommender_;

  // This object manager a list of fuzzer pass and their available
  // recommendations.
  std::unique_ptr<RepeatedPassManager> repeated_pass_manager_;

  // Some passes that it does not make sense to apply repeatedly, as they do not
  // unlock other passes.
  std::vector<std::unique_ptr<FuzzerPass>> final_passes_;

  // When set, this flag causes inapplicable transformations that should be
  // applicable by construction to be ignored. This is useful when the fuzzer
  // is being deployed at scale to test a SPIR-V processing tool, and where it
  // is desirable to ignore bugs in the fuzzer itself.
  const bool ignore_inapplicable_transformations_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_H_
