// Copyright 2018 The Shaderc Authors. All rights reserved.
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

#ifndef SHADERC_STATUS_H_
#define SHADERC_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

// Indicate the status of a compilation.
typedef enum {
  shaderc_compilation_status_success = 0,
  shaderc_compilation_status_invalid_stage,  // error stage deduction
  shaderc_compilation_status_compilation_error,
  shaderc_compilation_status_internal_error,  // unexpected failure
  shaderc_compilation_status_null_result_object,
  shaderc_compilation_status_invalid_assembly,
  shaderc_compilation_status_validation_error,
} shaderc_compilation_status;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // SHADERC_STATUS_H_
