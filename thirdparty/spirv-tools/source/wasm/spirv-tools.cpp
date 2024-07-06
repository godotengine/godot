// Copyright (c) 2020 The Khronos Group Inc.
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

#include "spirv-tools/libspirv.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>
using namespace emscripten;

void print_msg_to_stderr (spv_message_level_t, const char*,
                          const spv_position_t&, const char* m) {
  std::cerr << "error: " << m << std::endl;
};

std::string dis(std::string const& buffer, uint32_t env, uint32_t options) {
  spvtools::SpirvTools core(static_cast<spv_target_env>(env));
  core.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  const uint32_t* ptr = reinterpret_cast<const uint32_t*>(buffer.data());
  spirv.assign(ptr, ptr + buffer.size() / 4);
  std::string disassembly;
  if (!core.Disassemble(spirv, &disassembly, options)) return "Error";
  return disassembly;
}

emscripten::val as(std::string const& source, uint32_t env, uint32_t options) {
  spvtools::SpirvTools core(static_cast<spv_target_env>(env));
  core.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  if (!core.Assemble(source, &spirv, options)) spirv.clear();
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(spirv.data());
  return emscripten::val(emscripten::typed_memory_view(spirv.size() * 4,
    ptr));
}

EMSCRIPTEN_BINDINGS(my_module) {
  function("dis", &dis);
  function("as", &as);
  
  constant("SPV_ENV_UNIVERSAL_1_0", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_0));
  constant("SPV_ENV_VULKAN_1_0", static_cast<uint32_t>(SPV_ENV_VULKAN_1_0));
  constant("SPV_ENV_UNIVERSAL_1_1", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_1));
  constant("SPV_ENV_OPENCL_2_1", static_cast<uint32_t>(SPV_ENV_OPENCL_2_1));
  constant("SPV_ENV_OPENCL_2_2", static_cast<uint32_t>(SPV_ENV_OPENCL_2_2));
  constant("SPV_ENV_OPENGL_4_0", static_cast<uint32_t>(SPV_ENV_OPENGL_4_0));
  constant("SPV_ENV_OPENGL_4_1", static_cast<uint32_t>(SPV_ENV_OPENGL_4_1));
  constant("SPV_ENV_OPENGL_4_2", static_cast<uint32_t>(SPV_ENV_OPENGL_4_2));
  constant("SPV_ENV_OPENGL_4_3", static_cast<uint32_t>(SPV_ENV_OPENGL_4_3));
  constant("SPV_ENV_OPENGL_4_5", static_cast<uint32_t>(SPV_ENV_OPENGL_4_5));
  constant("SPV_ENV_UNIVERSAL_1_2", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_2));
  constant("SPV_ENV_OPENCL_1_2", static_cast<uint32_t>(SPV_ENV_OPENCL_1_2));
  constant("SPV_ENV_OPENCL_EMBEDDED_1_2", static_cast<uint32_t>(SPV_ENV_OPENCL_EMBEDDED_1_2));
  constant("SPV_ENV_OPENCL_2_0", static_cast<uint32_t>(SPV_ENV_OPENCL_2_0));
  constant("SPV_ENV_OPENCL_EMBEDDED_2_0", static_cast<uint32_t>(SPV_ENV_OPENCL_EMBEDDED_2_0));
  constant("SPV_ENV_OPENCL_EMBEDDED_2_1", static_cast<uint32_t>(SPV_ENV_OPENCL_EMBEDDED_2_1));
  constant("SPV_ENV_OPENCL_EMBEDDED_2_2", static_cast<uint32_t>(SPV_ENV_OPENCL_EMBEDDED_2_2));
  constant("SPV_ENV_UNIVERSAL_1_3", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_3));
  constant("SPV_ENV_VULKAN_1_1", static_cast<uint32_t>(SPV_ENV_VULKAN_1_1));
  constant("SPV_ENV_WEBGPU_0", static_cast<uint32_t>(SPV_ENV_WEBGPU_0));
  constant("SPV_ENV_UNIVERSAL_1_4", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_4));
  constant("SPV_ENV_VULKAN_1_1_SPIRV_1_4", static_cast<uint32_t>(SPV_ENV_VULKAN_1_1_SPIRV_1_4));
  constant("SPV_ENV_UNIVERSAL_1_5", static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_5));
  constant("SPV_ENV_VULKAN_1_2", static_cast<uint32_t>(SPV_ENV_VULKAN_1_2));
  constant("SPV_ENV_UNIVERSAL_1_6",
           static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_6));

  constant("SPV_BINARY_TO_TEXT_OPTION_NONE", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NONE));
  constant("SPV_BINARY_TO_TEXT_OPTION_PRINT", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_PRINT));
  constant("SPV_BINARY_TO_TEXT_OPTION_COLOR", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
  constant("SPV_BINARY_TO_TEXT_OPTION_INDENT", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT));
  constant("SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET));
  constant("SPV_BINARY_TO_TEXT_OPTION_NO_HEADER", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));
  constant("SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));

  constant("SPV_TEXT_TO_BINARY_OPTION_NONE", static_cast<uint32_t>(SPV_TEXT_TO_BINARY_OPTION_NONE));
  constant("SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS", static_cast<uint32_t>(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
}
