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

#ifndef SHADERC_ENV_H_
#define SHADERC_ENV_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  shaderc_target_env_vulkan,  // create SPIR-V under Vulkan semantics
  shaderc_target_env_opengl,  // create SPIR-V under OpenGL semantics
  // NOTE: SPIR-V code generation is not supported for shaders under OpenGL
  // compatibility profile.
  shaderc_target_env_opengl_compat,  // create SPIR-V under OpenGL semantics,
                                     // including compatibility profile
                                     // functions
  shaderc_target_env_default = shaderc_target_env_vulkan
} shaderc_target_env;

typedef enum {
  // For Vulkan, use Vulkan's mapping of version numbers to integers.
  // See vulkan.h
  shaderc_env_version_vulkan_1_0 = (((uint32_t)1 << 22)),
  shaderc_env_version_vulkan_1_1 = (((uint32_t)1 << 22) | (1 << 12)),
  // For OpenGL, use the number from #version in shaders.
  // TODO(dneto): Currently no difference between OpenGL 4.5 and 4.6.
  // See glslang/Standalone/Standalone.cpp
  // TODO(dneto): Glslang doesn't accept a OpenGL client version of 460.
  shaderc_env_version_opengl_4_5 = 450,
} shaderc_env_version;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // SHADERC_ENV_H_
