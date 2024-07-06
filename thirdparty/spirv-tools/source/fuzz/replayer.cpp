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

#include "source/fuzz/replayer.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "source/fuzz/counter_overflow_id_source.h"
#include "source/fuzz/fact_manager/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/build_module.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

Replayer::Replayer(
    spv_target_env target_env, MessageConsumer consumer,
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const protobufs::TransformationSequence& transformation_sequence_in,
    uint32_t num_transformations_to_apply, bool validate_during_replay,
    spv_validator_options validator_options)
    : target_env_(target_env),
      consumer_(std::move(consumer)),
      binary_in_(binary_in),
      initial_facts_(initial_facts),
      transformation_sequence_in_(transformation_sequence_in),
      num_transformations_to_apply_(num_transformations_to_apply),
      validate_during_replay_(validate_during_replay),
      validator_options_(validator_options) {}

Replayer::~Replayer() = default;

Replayer::ReplayerResult Replayer::Run() {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (num_transformations_to_apply_ >
      static_cast<uint32_t>(
          transformation_sequence_in_.transformation_size())) {
    consumer_(SPV_MSG_ERROR, nullptr, {},
              "The number of transformations to be replayed must not "
              "exceed the size of the transformation sequence.");
    return {Replayer::ReplayerResultStatus::kTooManyTransformationsRequested,
            nullptr, nullptr, protobufs::TransformationSequence()};
  }

  spvtools::SpirvTools tools(target_env_);
  if (!tools.IsValid()) {
    consumer_(SPV_MSG_ERROR, nullptr, {},
              "Failed to create SPIRV-Tools interface; stopping.");
    return {Replayer::ReplayerResultStatus::kFailedToCreateSpirvToolsInterface,
            nullptr, nullptr, protobufs::TransformationSequence()};
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in_[0], binary_in_.size(), validator_options_)) {
    consumer_(SPV_MSG_INFO, nullptr, {},
              "Initial binary is invalid; stopping.");
    return {Replayer::ReplayerResultStatus::kInitialBinaryInvalid, nullptr,
            nullptr, protobufs::TransformationSequence()};
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context =
      BuildModule(target_env_, consumer_, binary_in_.data(), binary_in_.size());
  assert(ir_context);

  // For replay validation, we track the last valid SPIR-V binary that was
  // observed. Initially this is the input binary.
  std::vector<uint32_t> last_valid_binary;
  if (validate_during_replay_) {
    last_valid_binary = binary_in_;
  }

  // We find the smallest id that is (a) not in use by the original module, and
  // (b) not used by any transformation in the sequence to be replayed.  This
  // serves as a starting id from which to issue overflow ids if they are
  // required during replay.
  uint32_t first_overflow_id = ir_context->module()->id_bound();
  for (auto& transformation : transformation_sequence_in_.transformation()) {
    auto fresh_ids = Transformation::FromMessage(transformation)->GetFreshIds();
    if (!fresh_ids.empty()) {
      first_overflow_id =
          std::max(first_overflow_id,
                   *std::max_element(fresh_ids.begin(), fresh_ids.end()) + 1);
    }
  }

  std::unique_ptr<TransformationContext> transformation_context =
      MakeUnique<TransformationContext>(
          MakeUnique<FactManager>(ir_context.get()), validator_options_,
          MakeUnique<CounterOverflowIdSource>(first_overflow_id));
  transformation_context->GetFactManager()->AddInitialFacts(consumer_,
                                                            initial_facts_);

  // We track the largest id bound observed, to ensure that it only increases
  // as transformations are applied.
  uint32_t max_observed_id_bound = ir_context->module()->id_bound();
  (void)(max_observed_id_bound);  // Keep release-mode compilers happy.

  protobufs::TransformationSequence transformation_sequence_out;

  // Consider the transformation proto messages in turn.
  uint32_t counter = 0;
  for (auto& message : transformation_sequence_in_.transformation()) {
    if (counter >= num_transformations_to_apply_) {
      break;
    }
    counter++;

    auto transformation = Transformation::FromMessage(message);

    // Check whether the transformation can be applied.
    if (transformation->IsApplicable(ir_context.get(),
                                     *transformation_context)) {
      // The transformation is applicable, so apply it, and copy it to the
      // sequence of transformations that were applied.
      transformation->Apply(ir_context.get(), transformation_context.get());
      *transformation_sequence_out.add_transformation() = message;

      assert(ir_context->module()->id_bound() >= max_observed_id_bound &&
             "The module's id bound should only increase due to applying "
             "transformations.");
      max_observed_id_bound = ir_context->module()->id_bound();

      if (validate_during_replay_) {
        std::vector<uint32_t> binary_to_validate;
        ir_context->module()->ToBinary(&binary_to_validate, false);

        // Check whether the latest transformation led to a valid binary.
        if (!tools.Validate(&binary_to_validate[0], binary_to_validate.size(),
                            validator_options_)) {
          consumer_(SPV_MSG_INFO, nullptr, {},
                    "Binary became invalid during replay (set a "
                    "breakpoint to inspect); stopping.");
          return {Replayer::ReplayerResultStatus::kReplayValidationFailure,
                  nullptr, nullptr, protobufs::TransformationSequence()};
        }

        // The binary was valid, so it becomes the latest valid binary.
        last_valid_binary = std::move(binary_to_validate);
      }
    }
  }

  return {Replayer::ReplayerResultStatus::kComplete, std::move(ir_context),
          std::move(transformation_context),
          std::move(transformation_sequence_out)};
}

}  // namespace fuzz
}  // namespace spvtools
