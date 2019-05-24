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

#ifndef SOURCE_SPIRV_REDUCER_OPTIONS_H_
#define SOURCE_SPIRV_REDUCER_OPTIONS_H_

#include "spirv-tools/libspirv.h"

#include <string>
#include <utility>

// The default maximum number of steps for the reducer to run before giving up.
const uint32_t kDefaultStepLimit = 250;

// Manages command line options passed to the SPIR-V Reducer. New struct
// members may be added for any new option.
struct spv_reducer_options_t {
  spv_reducer_options_t()
      : step_limit(kDefaultStepLimit), fail_on_validation_error(false) {}

  // See spvReducerOptionsSetStepLimit.
  uint32_t step_limit;

  // See spvReducerOptionsSetFailOnValidationError.
  bool fail_on_validation_error;
};

#endif  // SOURCE_SPIRV_REDUCER_OPTIONS_H_
