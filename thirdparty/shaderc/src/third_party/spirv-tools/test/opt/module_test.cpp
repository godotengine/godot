// Copyright (c) 2016 Google Inc.
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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.hpp"
#include "test/opt/module_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::Eq;
using spvtest::GetIdBound;

TEST(ModuleTest, SetIdBound) {
  Module m;
  // It's initialized to 0.
  EXPECT_EQ(0u, GetIdBound(m));

  m.SetIdBound(19);
  EXPECT_EQ(19u, GetIdBound(m));

  m.SetIdBound(102);
  EXPECT_EQ(102u, GetIdBound(m));
}

// Returns an IRContext owning the module formed by assembling the given text,
// then loading the result.
inline std::unique_ptr<IRContext> BuildModule(std::string text) {
  return spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                               SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
}

TEST(ModuleTest, ComputeIdBound) {
  // Emtpy module case.
  EXPECT_EQ(1u, BuildModule("")->module()->ComputeIdBound());
  // Sensitive to result id
  EXPECT_EQ(2u, BuildModule("%void = OpTypeVoid")->module()->ComputeIdBound());
  // Sensitive to type id
  EXPECT_EQ(1000u,
            BuildModule("%a = OpTypeArray !999 3")->module()->ComputeIdBound());
  // Sensitive to a regular Id parameter
  EXPECT_EQ(2000u,
            BuildModule("OpDecorate !1999 0")->module()->ComputeIdBound());
  // Sensitive to a scope Id parameter.
  EXPECT_EQ(3000u,
            BuildModule("%f = OpFunction %void None %fntype %a = OpLabel "
                        "OpMemoryBarrier !2999 %b\n")
                ->module()
                ->ComputeIdBound());
  // Sensitive to a semantics Id parameter
  EXPECT_EQ(4000u,
            BuildModule("%f = OpFunction %void None %fntype %a = OpLabel "
                        "OpMemoryBarrier %b !3999\n")
                ->module()
                ->ComputeIdBound());
}

TEST(ModuleTest, OstreamOperator) {
  const std::string text = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpName %7 "restrict"
OpDecorate %8 Restrict
%9 = OpTypeVoid
%10 = OpTypeInt 32 0
%11 = OpTypeStruct %10 %10
%12 = OpTypePointer Function %10
%13 = OpTypePointer Function %11
%14 = OpConstant %10 0
%15 = OpConstant %10 1
%7 = OpTypeFunction %9
%1 = OpFunction %9 None %7
%2 = OpLabel
%8 = OpVariable %13 Function
%3 = OpAccessChain %12 %8 %14
%4 = OpLoad %10 %3
%5 = OpAccessChain %12 %8 %15
%6 = OpLoad %10 %5
OpReturn
OpFunctionEnd)";

  std::string s;
  std::ostringstream str(s);
  str << *BuildModule(text)->module();
  EXPECT_EQ(text, str.str());
}

TEST(ModuleTest, OstreamOperatorInt64) {
  const std::string text = R"(OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
OpName %7 "restrict"
OpDecorate %5 Restrict
%9 = OpTypeVoid
%10 = OpTypeInt 64 0
%11 = OpTypeStruct %10 %10
%12 = OpTypePointer Function %10
%13 = OpTypePointer Function %11
%14 = OpConstant %10 0
%15 = OpConstant %10 1
%16 = OpConstant %10 4294967297
%7 = OpTypeFunction %9
%1 = OpFunction %9 None %7
%2 = OpLabel
%5 = OpVariable %12 Function
%6 = OpLoad %10 %5
OpSelectionMerge %3 None
OpSwitch %6 %3 4294967297 %4
%4 = OpLabel
OpBranch %3
%3 = OpLabel
OpReturn
OpFunctionEnd)";

  std::string s;
  std::ostringstream str(s);
  str << *BuildModule(text)->module();
  EXPECT_EQ(text, str.str());
}

TEST(ModuleTest, IdBoundTestAtLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context = BuildModule(text);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound);
  uint32_t next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, 0);
  EXPECT_EQ(current_bound, context->module()->id_bound());
  next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, 0);
}

TEST(ModuleTest, IdBoundTestBelowLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context = BuildModule(text);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound + 100);
  uint32_t next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, current_bound);
  EXPECT_EQ(current_bound + 1, context->module()->id_bound());
  next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, current_bound + 1);
}

TEST(ModuleTest, IdBoundTestNearLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context = BuildModule(text);
  uint32_t current_bound = context->module()->id_bound();
  context->set_max_id_bound(current_bound + 1);
  uint32_t next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, current_bound);
  EXPECT_EQ(current_bound + 1, context->module()->id_bound());
  next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, 0);
}

TEST(ModuleTest, IdBoundTestUIntMax) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4294967294 = OpLabel ; ID is UINT_MAX-1
OpReturn
OpFunctionEnd)";

  std::unique_ptr<IRContext> context = BuildModule(text);
  uint32_t current_bound = context->module()->id_bound();

  // Expecting |BuildModule| to preserve the numeric ids.
  EXPECT_EQ(current_bound, std::numeric_limits<uint32_t>::max());

  context->set_max_id_bound(current_bound);
  uint32_t next_id_bound = context->module()->TakeNextIdBound();
  EXPECT_EQ(next_id_bound, 0);
  EXPECT_EQ(current_bound, context->module()->id_bound());
}
}  // namespace
}  // namespace opt
}  // namespace spvtools
