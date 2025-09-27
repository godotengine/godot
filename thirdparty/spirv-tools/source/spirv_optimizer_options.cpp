// Copyright (c) 2017 Google Inc.
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

#include "source/spirv_optimizer_options.h"

SPIRV_TOOLS_EXPORT spv_optimizer_options spvOptimizerOptionsCreate(void) {
  return new spv_optimizer_options_t();
}

SPIRV_TOOLS_EXPORT void spvOptimizerOptionsDestroy(
    spv_optimizer_options options) {
  delete options;
}

SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetRunValidator(
    spv_optimizer_options options, bool val) {
  options->run_validator_ = val;
}

SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetValidatorOptions(
    spv_optimizer_options options, spv_validator_options val) {
  options->val_options_ = *val;
}
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetMaxIdBound(
    spv_optimizer_options options, uint32_t val) {
  options->max_id_bound_ = val;
}

SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetPreserveBindings(
    spv_optimizer_options options, bool val) {
  options->preserve_bindings_ = val;
}

SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetPreserveSpecConstants(
    spv_optimizer_options options, bool val) {
  options->preserve_spec_constants_ = val;
}
