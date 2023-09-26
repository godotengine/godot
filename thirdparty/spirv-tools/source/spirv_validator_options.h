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

#ifndef SOURCE_SPIRV_VALIDATOR_OPTIONS_H_
#define SOURCE_SPIRV_VALIDATOR_OPTIONS_H_

#include "spirv-tools/libspirv.h"

// Return true if the command line option for the validator limit is valid (Also
// returns the Enum for option in this case). Returns false otherwise.
bool spvParseUniversalLimitsOptions(const char* s, spv_validator_limit* limit);

// Default initialization of this structure is to the default Universal Limits
// described in the SPIR-V Spec.
struct validator_universal_limits_t {
  uint32_t max_struct_members{16383};
  uint32_t max_struct_depth{255};
  uint32_t max_local_variables{524287};
  uint32_t max_global_variables{65535};
  uint32_t max_switch_branches{16383};
  uint32_t max_function_args{255};
  uint32_t max_control_flow_nesting_depth{1023};
  uint32_t max_access_chain_indexes{255};
  uint32_t max_id_bound{0x3FFFFF};
};

// Manages command line options passed to the SPIR-V Validator. New struct
// members may be added for any new option.
struct spv_validator_options_t {
  spv_validator_options_t()
      : universal_limits_(),
        relax_struct_store(false),
        relax_logical_pointer(false),
        relax_block_layout(false),
        uniform_buffer_standard_layout(false),
        scalar_block_layout(false),
        workgroup_scalar_block_layout(false),
        skip_block_layout(false),
        allow_localsizeid(false),
        before_hlsl_legalization(false),
        use_friendly_names(true) {}

  validator_universal_limits_t universal_limits_;
  bool relax_struct_store;
  bool relax_logical_pointer;
  bool relax_block_layout;
  bool uniform_buffer_standard_layout;
  bool scalar_block_layout;
  bool workgroup_scalar_block_layout;
  bool skip_block_layout;
  bool allow_localsizeid;
  bool before_hlsl_legalization;
  bool use_friendly_names;
};

#endif  // SOURCE_SPIRV_VALIDATOR_OPTIONS_H_
