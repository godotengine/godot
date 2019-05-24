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

#include <gtest/gtest.h>
#include <thread>

#include "common_shaders_for_test.h"
#include "shaderc/spvc.h"

namespace {

TEST(Init, MultipleCalls) {
  shaderc_spvc_compiler_t compiler1, compiler2, compiler3;
  EXPECT_NE(nullptr, compiler1 = shaderc_spvc_compiler_initialize());
  EXPECT_NE(nullptr, compiler2 = shaderc_spvc_compiler_initialize());
  EXPECT_NE(nullptr, compiler3 = shaderc_spvc_compiler_initialize());
  shaderc_spvc_compiler_release(compiler1);
  shaderc_spvc_compiler_release(compiler2);
  shaderc_spvc_compiler_release(compiler3);
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST(Init, MultipleThreadsCalling) {
  shaderc_spvc_compiler_t compiler1, compiler2, compiler3;
  std::thread t1(
      [&compiler1]() { compiler1 = shaderc_spvc_compiler_initialize(); });
  std::thread t2(
      [&compiler2]() { compiler2 = shaderc_spvc_compiler_initialize(); });
  std::thread t3(
      [&compiler3]() { compiler3 = shaderc_spvc_compiler_initialize(); });
  t1.join();
  t2.join();
  t3.join();
  EXPECT_NE(nullptr, compiler1);
  EXPECT_NE(nullptr, compiler2);
  EXPECT_NE(nullptr, compiler3);
  shaderc_spvc_compiler_release(compiler1);
  shaderc_spvc_compiler_release(compiler2);
  shaderc_spvc_compiler_release(compiler3);
}
#endif

TEST(Compile, Glsl) {
  shaderc_spvc_compiler_t compiler;
  shaderc_spvc_compile_options_t options;

  compiler = shaderc_spvc_compiler_initialize();
  options = shaderc_spvc_compile_options_initialize();

  shaderc_spvc_compilation_result_t result = shaderc_spvc_compile_into_glsl(
      compiler, kShader1, sizeof(kShader1) / sizeof(uint32_t), options);
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(shaderc_compilation_status_success,
            shaderc_spvc_result_get_status(result));

  shaderc_spvc_result_release(result);
  shaderc_spvc_compile_options_release(options);
  shaderc_spvc_compiler_release(compiler);
}

TEST(Compile, Hlsl) {
  shaderc_spvc_compiler_t compiler;
  shaderc_spvc_compile_options_t options;

  compiler = shaderc_spvc_compiler_initialize();
  options = shaderc_spvc_compile_options_initialize();

  shaderc_spvc_compilation_result_t result = shaderc_spvc_compile_into_hlsl(
      compiler, kShader1, sizeof(kShader1) / sizeof(uint32_t), options);
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(shaderc_compilation_status_success,
            shaderc_spvc_result_get_status(result));

  shaderc_spvc_result_release(result);
  shaderc_spvc_compile_options_release(options);
  shaderc_spvc_compiler_release(compiler);
}

TEST(Compile, Msl) {
  shaderc_spvc_compiler_t compiler;
  shaderc_spvc_compile_options_t options;

  compiler = shaderc_spvc_compiler_initialize();
  options = shaderc_spvc_compile_options_initialize();

  shaderc_spvc_compilation_result_t result = shaderc_spvc_compile_into_msl(
      compiler, kShader1, sizeof(kShader1) / sizeof(uint32_t), options);
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(shaderc_compilation_status_success,
            shaderc_spvc_result_get_status(result));

  shaderc_spvc_result_release(result);
  shaderc_spvc_compile_options_release(options);
  shaderc_spvc_compiler_release(compiler);
}

}  // anonymous namespace
