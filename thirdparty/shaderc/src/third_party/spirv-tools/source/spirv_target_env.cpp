// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/spirv_target_env.h"

#include <cstring>

#include "source/spirv_constant.h"
#include "spirv-tools/libspirv.h"

const char* spvTargetEnvDescription(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
      return "SPIR-V 1.0";
    case SPV_ENV_VULKAN_1_0:
      return "SPIR-V 1.0 (under Vulkan 1.0 semantics)";
    case SPV_ENV_UNIVERSAL_1_1:
      return "SPIR-V 1.1";
    case SPV_ENV_OPENCL_1_2:
      return "SPIR-V 1.0 (under OpenCL 1.2 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
      return "SPIR-V 1.0 (under OpenCL 1.2 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_0:
      return "SPIR-V 1.0 (under OpenCL 2.0 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
      return "SPIR-V 1.0 (under OpenCL 2.0 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_1:
      return "SPIR-V 1.0 (under OpenCL 2.1 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
      return "SPIR-V 1.0 (under OpenCL 2.1 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_2:
      return "SPIR-V 1.2 (under OpenCL 2.2 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
      return "SPIR-V 1.2 (under OpenCL 2.2 Embedded Profile semantics)";
    case SPV_ENV_OPENGL_4_0:
      return "SPIR-V 1.0 (under OpenGL 4.0 semantics)";
    case SPV_ENV_OPENGL_4_1:
      return "SPIR-V 1.0 (under OpenGL 4.1 semantics)";
    case SPV_ENV_OPENGL_4_2:
      return "SPIR-V 1.0 (under OpenGL 4.2 semantics)";
    case SPV_ENV_OPENGL_4_3:
      return "SPIR-V 1.0 (under OpenGL 4.3 semantics)";
    case SPV_ENV_OPENGL_4_5:
      return "SPIR-V 1.0 (under OpenGL 4.5 semantics)";
    case SPV_ENV_UNIVERSAL_1_2:
      return "SPIR-V 1.2";
    case SPV_ENV_UNIVERSAL_1_3:
      return "SPIR-V 1.3";
    case SPV_ENV_VULKAN_1_1:
      return "SPIR-V 1.3 (under Vulkan 1.1 semantics)";
    case SPV_ENV_WEBGPU_0:
      return "SPIR-V 1.3 (under WIP WebGPU semantics)";
  }
  return "";
}

uint32_t spvVersionForTargetEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      return SPV_SPIRV_VERSION_WORD(1, 0);
    case SPV_ENV_UNIVERSAL_1_1:
      return SPV_SPIRV_VERSION_WORD(1, 1);
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
      return SPV_SPIRV_VERSION_WORD(1, 2);
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_WEBGPU_0:
      return SPV_SPIRV_VERSION_WORD(1, 3);
  }
  return SPV_SPIRV_VERSION_WORD(0, 0);
}

bool spvParseTargetEnv(const char* s, spv_target_env* env) {
  auto match = [s](const char* b) {
    return s && (0 == strncmp(s, b, strlen(b)));
  };
  if (match("vulkan1.0")) {
    if (env) *env = SPV_ENV_VULKAN_1_0;
    return true;
  } else if (match("vulkan1.1")) {
    if (env) *env = SPV_ENV_VULKAN_1_1;
    return true;
  } else if (match("spv1.0")) {
    if (env) *env = SPV_ENV_UNIVERSAL_1_0;
    return true;
  } else if (match("spv1.1")) {
    if (env) *env = SPV_ENV_UNIVERSAL_1_1;
    return true;
  } else if (match("spv1.2")) {
    if (env) *env = SPV_ENV_UNIVERSAL_1_2;
    return true;
  } else if (match("spv1.3")) {
    if (env) *env = SPV_ENV_UNIVERSAL_1_3;
    return true;
  } else if (match("opencl1.2embedded")) {
    if (env) *env = SPV_ENV_OPENCL_EMBEDDED_1_2;
    return true;
  } else if (match("opencl1.2")) {
    if (env) *env = SPV_ENV_OPENCL_1_2;
    return true;
  } else if (match("opencl2.0embedded")) {
    if (env) *env = SPV_ENV_OPENCL_EMBEDDED_2_0;
    return true;
  } else if (match("opencl2.0")) {
    if (env) *env = SPV_ENV_OPENCL_2_0;
    return true;
  } else if (match("opencl2.1embedded")) {
    if (env) *env = SPV_ENV_OPENCL_EMBEDDED_2_1;
    return true;
  } else if (match("opencl2.1")) {
    if (env) *env = SPV_ENV_OPENCL_2_1;
    return true;
  } else if (match("opencl2.2embedded")) {
    if (env) *env = SPV_ENV_OPENCL_EMBEDDED_2_2;
    return true;
  } else if (match("opencl2.2")) {
    if (env) *env = SPV_ENV_OPENCL_2_2;
    return true;
  } else if (match("opengl4.0")) {
    if (env) *env = SPV_ENV_OPENGL_4_0;
    return true;
  } else if (match("opengl4.1")) {
    if (env) *env = SPV_ENV_OPENGL_4_1;
    return true;
  } else if (match("opengl4.2")) {
    if (env) *env = SPV_ENV_OPENGL_4_2;
    return true;
  } else if (match("opengl4.3")) {
    if (env) *env = SPV_ENV_OPENGL_4_3;
    return true;
  } else if (match("opengl4.5")) {
    if (env) *env = SPV_ENV_OPENGL_4_5;
    return true;
  } else if (match("webgpu0")) {
    if (env) *env = SPV_ENV_WEBGPU_0;
    return true;
  } else {
    if (env) *env = SPV_ENV_UNIVERSAL_1_0;
    return false;
  }
}

bool spvIsVulkanEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_WEBGPU_0:
      return false;
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
      return true;
  }
  return false;
}

bool spvIsOpenCLEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_WEBGPU_0:
      return false;
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
      return true;
  }
  return false;
}

bool spvIsWebGPUEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
      return false;
    case SPV_ENV_WEBGPU_0:
      return true;
  }
  return false;
}

bool spvIsVulkanOrWebGPUEnv(spv_target_env env) {
  return spvIsVulkanEnv(env) || spvIsWebGPUEnv(env);
}

std::string spvLogStringForEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2: {
      return "OpenCL";
    }
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5: {
      return "OpenGL";
    }
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1: {
      return "Vulkan";
    }
    case SPV_ENV_WEBGPU_0: {
      return "WebGPU";
    }
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3: {
      return "Universal";
    }
  }
  return "Unknown";
}
