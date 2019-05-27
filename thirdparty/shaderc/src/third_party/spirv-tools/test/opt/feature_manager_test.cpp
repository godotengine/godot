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

#include <algorithm>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

using FeatureManagerTest = ::testing::Test;

TEST_F(FeatureManagerTest, MissingExtension) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_FALSE(context->get_feature_mgr()->HasExtension(
      Extension::kSPV_KHR_variable_pointers));
}

TEST_F(FeatureManagerTest, OneExtension) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpExtension "SPV_KHR_variable_pointers"
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_TRUE(context->get_feature_mgr()->HasExtension(
      Extension::kSPV_KHR_variable_pointers));
}

TEST_F(FeatureManagerTest, NotADifferentExtension) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpExtension "SPV_KHR_variable_pointers"
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_FALSE(context->get_feature_mgr()->HasExtension(
      Extension::kSPV_KHR_storage_buffer_storage_class));
}

TEST_F(FeatureManagerTest, TwoExtensions) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpExtension "SPV_KHR_variable_pointers"
OpExtension "SPV_KHR_storage_buffer_storage_class"
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_TRUE(context->get_feature_mgr()->HasExtension(
      Extension::kSPV_KHR_variable_pointers));
  EXPECT_TRUE(context->get_feature_mgr()->HasExtension(
      Extension::kSPV_KHR_storage_buffer_storage_class));
}

// Test capability checks.
TEST_F(FeatureManagerTest, ExplicitlyPresent1) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_TRUE(context->get_feature_mgr()->HasCapability(SpvCapabilityShader));
  EXPECT_FALSE(context->get_feature_mgr()->HasCapability(SpvCapabilityKernel));
}

TEST_F(FeatureManagerTest, ExplicitlyPresent2) {
  const std::string text = R"(
OpCapability Kernel
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  EXPECT_FALSE(context->get_feature_mgr()->HasCapability(SpvCapabilityShader));
  EXPECT_TRUE(context->get_feature_mgr()->HasCapability(SpvCapabilityKernel));
}

TEST_F(FeatureManagerTest, ImplicitlyPresent) {
  const std::string text = R"(
OpCapability Tessellation
OpMemoryModel Logical GLSL450
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  ASSERT_NE(context, nullptr);

  // Check multiple levels of indirection.  Tessellation implies Shader, which
  // implies Matrix.
  EXPECT_TRUE(
      context->get_feature_mgr()->HasCapability(SpvCapabilityTessellation));
  EXPECT_TRUE(context->get_feature_mgr()->HasCapability(SpvCapabilityShader));
  EXPECT_TRUE(context->get_feature_mgr()->HasCapability(SpvCapabilityMatrix));
  EXPECT_FALSE(context->get_feature_mgr()->HasCapability(SpvCapabilityKernel));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
