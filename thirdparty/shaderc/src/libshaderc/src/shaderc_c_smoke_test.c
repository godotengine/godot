// Copyright 2016 The Shaderc Authors. All rights reserved.
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

#include "shaderc/shaderc.h"
#include <assert.h>
#include <string.h>

// Because we want to test this as a plain old C file, we cannot use
// gtest, so just run a simple smoke test.

int main() {
  const char* test_program =
      "#version 310 es\n"
      "layout(location = 0) in highp vec4 vtxColor;\n"
      "layout(location = 0) out highp vec4 outColor;\n"
      "void main() {\n"
      "  outColor = vtxColor;"
      "}\n";
  shaderc_compiler_t compiler;
  shaderc_compilation_result_t result;
  shaderc_compile_options_t options;

  compiler = shaderc_compiler_initialize();
  options = shaderc_compile_options_initialize();
  shaderc_compile_options_add_macro_definition(options, "FOO", 3, "1", 1);
  result = shaderc_compile_into_spv(
      compiler, test_program, strlen(test_program),
      shaderc_glsl_fragment_shader, "a.glsl", "main", options);

  assert(result);

  if (shaderc_result_get_compilation_status(result) !=
      shaderc_compilation_status_success) {
    // Early exit on failure.
    return -1;
  }
  shaderc_result_release(result);
  shaderc_compile_options_release(options);
  shaderc_compiler_release(compiler);

  return 0;
}

