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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "shaderc/shaderc.h"
#include "shaderc/spvc.h"

// Because we want to test this as a plain old C file, we cannot use
// gtest, so just run a simple smoke test.

int main() {
  const char *test_program =
      "               OpCapability Shader\n"
      "          %1 = OpExtInstImport \"GLSL.std.450\"\n"
      "               OpMemoryModel Logical GLSL450\n"
      "               OpEntryPoint Vertex %main \"main\" %outColor %vtxColor\n"
      "               OpSource ESSL 310\n"
      "               OpSourceExtension "
      "\"GL_GOOGLE_cpp_style_line_directive\"\n"
      "               OpSourceExtension \"GL_GOOGLE_include_directive\"\n"
      "               OpName %main \"main\"\n"
      "               OpName %outColor \"outColor\"\n"
      "               OpName %vtxColor \"vtxColor\"\n"
      "               OpDecorate %outColor Location 0\n"
      "               OpDecorate %vtxColor Location 0\n"
      "       %void = OpTypeVoid\n"
      "          %3 = OpTypeFunction %void\n"
      "      %float = OpTypeFloat 32\n"
      "    %v4float = OpTypeVector %float 4\n"
      "%_ptr_Output_v4float = OpTypePointer Output %v4float\n"
      "   %outColor = OpVariable %_ptr_Output_v4float Output\n"
      "%_ptr_Input_v4float = OpTypePointer Input %v4float\n"
      "   %vtxColor = OpVariable %_ptr_Input_v4float Input\n"
      "       %main = OpFunction %void None %3\n"
      "          %5 = OpLabel\n"
      "         %12 = OpLoad %v4float %vtxColor\n"
      "               OpStore %outColor %12\n"
      "               OpReturn\n"
      "               OpFunctionEnd\n";

  shaderc_spvc_compiler_t compiler;
  shaderc_spvc_compilation_result_t result;
  shaderc_spvc_compile_options_t options;
  int ret_code = 0;

  compiler = shaderc_spvc_compiler_initialize();
  options = shaderc_spvc_compile_options_initialize();

  shaderc_compiler_t shaderc;
  shaderc = shaderc_compiler_initialize();
  shaderc_compile_options_t opt = shaderc_compile_options_initialize();
  shaderc_compilation_result_t res = shaderc_assemble_into_spv(
      shaderc, test_program, strlen(test_program), opt);

  result = shaderc_spvc_compile_into_glsl(
      compiler, (const uint32_t *)shaderc_result_get_bytes(res),
      shaderc_result_get_length(res) / sizeof(uint32_t), options);
  assert(result);
  if (shaderc_spvc_result_get_status(result) == shaderc_compilation_status_success) {
    printf("success! %lu characters of glsl\n",
           (unsigned long)(strlen(shaderc_spvc_result_get_output(result))));
  } else {
    printf("failed to produce glsl\n");
    ret_code = -1;
  }
  shaderc_spvc_result_release(result);

  result = shaderc_spvc_compile_into_hlsl(
      compiler, (const uint32_t *)shaderc_result_get_bytes(res),
      shaderc_result_get_length(res) / sizeof(uint32_t), options);
  assert(result);
  if (shaderc_spvc_result_get_status(result) == shaderc_compilation_status_success) {
    printf("success! %lu characters of hlsl\n",
           (unsigned long)(strlen(shaderc_spvc_result_get_output(result))));
  } else {
    printf("failed to produce hlsl\n");
    ret_code = -1;
  }
  shaderc_spvc_result_release(result);

  result = shaderc_spvc_compile_into_msl(
      compiler, (const uint32_t *)shaderc_result_get_bytes(res),
      shaderc_result_get_length(res) / sizeof(uint32_t), options);
  assert(result);
  if (shaderc_spvc_result_get_status(result) ==
      shaderc_compilation_status_success) {
    printf("success! %lu characters of msl\n",
           (unsigned long)(strlen(shaderc_spvc_result_get_output(result))));
  } else {
    printf("failed to produce msl\n");
    ret_code = -1;
  }
  shaderc_spvc_result_release(result);

  shaderc_compile_options_release(opt);
  shaderc_result_release(res);
  shaderc_spvc_compile_options_release(options);
  shaderc_spvc_compiler_release(compiler);
  shaderc_compiler_release(shaderc);

  return ret_code;
}
