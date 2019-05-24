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

#ifndef SOURCE_SPIRV_OPTIMIZER_OPTIONS_H_
#define SOURCE_SPIRV_OPTIMIZER_OPTIONS_H_

#include "source/spirv_validator_options.h"
#include "spirv-tools/libspirv.h"

// Manages command line options passed to the SPIR-V Validator. New struct
// members may be added for any new option.
struct spv_optimizer_options_t {
  spv_optimizer_options_t()
      : run_validator_(true),
        val_options_(),
        max_id_bound_(kDefaultMaxIdBound) {}

  // When true the validator will be run before optimizations are run.
  bool run_validator_;

  // Options to pass to the validator if it is run.
  spv_validator_options_t val_options_;

  // The maximum value the id bound for a module can have.  The Spir-V spec says
  // this value must be at least 0x3FFFFF, but implementations can allow for a
  // higher value.
  uint32_t max_id_bound_;
};
#endif  // SOURCE_SPIRV_OPTIMIZER_OPTIONS_H_
