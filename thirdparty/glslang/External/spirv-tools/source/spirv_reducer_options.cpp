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

#include <cassert>
#include <cstring>

#include "source/spirv_reducer_options.h"

namespace {
// The default maximum number of steps the reducer will take before giving up.
const uint32_t kDefaultStepLimit = 2500;
}  // namespace

spv_reducer_options_t::spv_reducer_options_t()
    : step_limit(kDefaultStepLimit),
      fail_on_validation_error(false),
      target_function(0) {}

SPIRV_TOOLS_EXPORT spv_reducer_options spvReducerOptionsCreate() {
  return new spv_reducer_options_t();
}

SPIRV_TOOLS_EXPORT void spvReducerOptionsDestroy(spv_reducer_options options) {
  delete options;
}

SPIRV_TOOLS_EXPORT void spvReducerOptionsSetStepLimit(
    spv_reducer_options options, uint32_t step_limit) {
  options->step_limit = step_limit;
}

SPIRV_TOOLS_EXPORT void spvReducerOptionsSetFailOnValidationError(
    spv_reducer_options options, bool fail_on_validation_error) {
  options->fail_on_validation_error = fail_on_validation_error;
}

SPIRV_TOOLS_EXPORT void spvReducerOptionsSetTargetFunction(
    spv_reducer_options options, uint32_t target_function) {
  options->target_function = target_function;
}
