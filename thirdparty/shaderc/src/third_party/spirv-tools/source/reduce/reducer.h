// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_REDUCE_REDUCER_H_
#define SOURCE_REDUCE_REDUCER_H_

#include <functional>
#include <string>

#include "source/reduce/reduction_pass.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace reduce {

// This class manages the process of applying a reduction -- parameterized by a
// number of reduction passes and an interestingness test, to a SPIR-V binary.
class Reducer {
 public:
  // Possible statuses that can result from running a reduction.
  enum ReductionResultStatus {
    kInitialStateNotInteresting,
    kReachedStepLimit,
    kComplete,
    kInitialStateInvalid,

    // Returned when the fail-on-validation-error option is set and a
    // reduction step yields a state that fails validation.
    kStateInvalid,
  };

  // The type for a function that will take a binary and return true if and
  // only if the binary is deemed interesting. (The function also takes an
  // integer argument that will be incremented each time the function is
  // called; this is for debugging purposes).
  //
  // The notion of "interesting" depends on what properties of the binary or
  // tools that process the binary we are trying to maintain during reduction.
  using InterestingnessFunction =
      std::function<bool(const std::vector<uint32_t>&, uint32_t)>;

  // Constructs an instance with the given target |env|, which is used to
  // decode the binary to be reduced later.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  //
  // The constructed instance also needs to have an interestingness function
  // set and some reduction passes added to it in order to be useful.
  explicit Reducer(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  Reducer(const Reducer&) = delete;
  Reducer(Reducer&&) = delete;
  Reducer& operator=(const Reducer&) = delete;
  Reducer& operator=(Reducer&&) = delete;

  // Destructs this instance.
  ~Reducer();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Sets the function that will be used to decide whether a reduced binary
  // turned out to be interesting.
  void SetInterestingnessFunction(
      InterestingnessFunction interestingness_function);

  // Adds all default reduction passes.
  void AddDefaultReductionPasses();

  // Adds a reduction pass based on the given finder to the sequence of passes
  // that will be iterated over.
  void AddReductionPass(std::unique_ptr<ReductionOpportunityFinder>&& finder);

  // Reduces the given SPIR-V module |binary_out|.
  // The reduced binary ends up in |binary_out|.
  // A status is returned.
  ReductionResultStatus Run(std::vector<uint32_t>&& binary_in,
                            std::vector<uint32_t>* binary_out,
                            spv_const_reducer_options options,
                            spv_validator_options validator_options) const;

 private:
  struct Impl;                  // Opaque struct for holding internal data.
  std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCER_H_
