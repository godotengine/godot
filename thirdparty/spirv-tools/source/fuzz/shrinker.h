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

#ifndef SOURCE_FUZZ_SHRINKER_H_
#define SOURCE_FUZZ_SHRINKER_H_

#include <memory>
#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Shrinks a sequence of transformations that lead to an interesting SPIR-V
// binary to yield a smaller sequence of transformations that still produce an
// interesting binary.
class Shrinker {
 public:
  // Possible statuses that can result from running the shrinker.
  enum class ShrinkerResultStatus {
    kComplete,
    kFailedToCreateSpirvToolsInterface,
    kInitialBinaryInvalid,
    kInitialBinaryNotInteresting,
    kReplayFailed,
    kStepLimitReached,
    kAddedFunctionReductionFailed,
  };

  struct ShrinkerResult {
    ShrinkerResultStatus status;
    std::vector<uint32_t> transformed_binary;
    protobufs::TransformationSequence applied_transformations;
  };

  // The type for a function that will take a binary, |binary|, and return true
  // if and only if the binary is deemed interesting. (The function also takes
  // an integer argument, |counter|, that will be incremented each time the
  // function is called; this is for debugging purposes).
  //
  // The notion of "interesting" depends on what properties of the binary or
  // tools that process the binary we are trying to maintain during shrinking.
  using InterestingnessFunction = std::function<bool(
      const std::vector<uint32_t>& binary, uint32_t counter)>;

  Shrinker(spv_target_env target_env, MessageConsumer consumer,
           const std::vector<uint32_t>& binary_in,
           const protobufs::FactSequence& initial_facts,
           const protobufs::TransformationSequence& transformation_sequence_in,
           const InterestingnessFunction& interestingness_function,
           uint32_t step_limit, bool validate_during_replay,
           spv_validator_options validator_options);

  // Disables copy/move constructor/assignment operations.
  Shrinker(const Shrinker&) = delete;
  Shrinker(Shrinker&&) = delete;
  Shrinker& operator=(const Shrinker&) = delete;
  Shrinker& operator=(Shrinker&&) = delete;

  ~Shrinker();

  // Requires that when |transformation_sequence_in_| is applied to |binary_in_|
  // with initial facts |initial_facts_|, the resulting binary is interesting
  // according to |interestingness_function_|.
  //
  // If shrinking succeeded -- possibly terminating early due to reaching the
  // shrinker's step limit -- an associated result status is returned together
  // with a subsequence of |transformation_sequence_in_| that, when applied
  // to |binary_in_| with initial facts |initial_facts_|, produces a binary
  // that is also interesting according to |interestingness_function_|; this
  // binary is also returned.
  //
  // If shrinking failed for some reason, an appropriate result status is
  // returned together with an empty binary and empty transformation sequence.
  ShrinkerResult Run();

 private:
  // Returns the id bound for the given SPIR-V binary, which is assumed to be
  // valid.
  uint32_t GetIdBound(const std::vector<uint32_t>& binary) const;

  // Target environment.
  const spv_target_env target_env_;

  // Message consumer that will be invoked once for each message communicated
  // from the library.
  MessageConsumer consumer_;

  // The binary to which transformations are to be applied.
  const std::vector<uint32_t>& binary_in_;

  // Initial facts known to hold in advance of applying any transformations.
  const protobufs::FactSequence& initial_facts_;

  // The series of transformations to be shrunk.
  const protobufs::TransformationSequence& transformation_sequence_in_;

  // Function that decides whether a given module is interesting.
  const InterestingnessFunction& interestingness_function_;

  // Step limit to decide when to terminate shrinking early.
  const uint32_t step_limit_;

  // Determines whether to check for validity during the replaying of
  // transformations.
  const bool validate_during_replay_;

  // Options to control validation.
  spv_validator_options validator_options_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_SHRINKER_H_
