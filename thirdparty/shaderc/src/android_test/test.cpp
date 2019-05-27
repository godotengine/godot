// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "shaderc/shaderc.hpp"
#include <android_native_app_glue.h>

void android_main(struct android_app* state) {
  app_dummy();
  shaderc::Compiler compiler;
  const char* test_program = "void main() {}";
  compiler.CompileGlslToSpv(test_program, strlen(test_program),
                            shaderc_glsl_vertex_shader, "shader");
}
