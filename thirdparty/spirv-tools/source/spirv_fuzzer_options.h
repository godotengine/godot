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

#ifndef SOURCE_SPIRV_FUZZER_OPTIONS_H_
#define SOURCE_SPIRV_FUZZER_OPTIONS_H_

#include "spirv-tools/libspirv.h"

#include <string>
#include <utility>

// Manages command line options passed to the SPIR-V Fuzzer. New struct
// members may be added for any new option.
struct spv_fuzzer_options_t {
  spv_fuzzer_options_t();

  // See spvFuzzerOptionsSetRandomSeed.
  bool has_random_seed;
  uint32_t random_seed;

  // See spvFuzzerOptionsSetReplayRange.
  int32_t replay_range;

  // See spvFuzzerOptionsEnableReplayValidation.
  bool replay_validation_enabled;

  // See spvFuzzerOptionsSetShrinkerStepLimit.
  uint32_t shrinker_step_limit;

  // See spvFuzzerOptionsValidateAfterEveryPass.
  bool fuzzer_pass_validation_enabled;

  // See spvFuzzerOptionsEnableAllPasses.
  bool all_passes_enabled;
};

#endif  // SOURCE_SPIRV_FUZZER_OPTIONS_H_
