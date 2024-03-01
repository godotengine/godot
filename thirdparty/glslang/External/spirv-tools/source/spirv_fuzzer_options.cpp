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

#include "source/spirv_fuzzer_options.h"

namespace {
// The default maximum number of steps for the reducer to run before giving up.
const uint32_t kDefaultStepLimit = 250;
}  // namespace

spv_fuzzer_options_t::spv_fuzzer_options_t()
    : has_random_seed(false),
      random_seed(0),
      replay_range(0),
      replay_validation_enabled(false),
      shrinker_step_limit(kDefaultStepLimit),
      fuzzer_pass_validation_enabled(false),
      all_passes_enabled(false) {}

SPIRV_TOOLS_EXPORT spv_fuzzer_options spvFuzzerOptionsCreate() {
  return new spv_fuzzer_options_t();
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsDestroy(spv_fuzzer_options options) {
  delete options;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableReplayValidation(
    spv_fuzzer_options options) {
  options->replay_validation_enabled = true;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetRandomSeed(
    spv_fuzzer_options options, uint32_t seed) {
  options->has_random_seed = true;
  options->random_seed = seed;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetReplayRange(
    spv_fuzzer_options options, int32_t replay_range) {
  options->replay_range = replay_range;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetShrinkerStepLimit(
    spv_fuzzer_options options, uint32_t shrinker_step_limit) {
  options->shrinker_step_limit = shrinker_step_limit;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableFuzzerPassValidation(
    spv_fuzzer_options options) {
  options->fuzzer_pass_validation_enabled = true;
}

SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableAllPasses(
    spv_fuzzer_options options) {
  options->all_passes_enabled = true;
}
