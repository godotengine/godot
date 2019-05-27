// Copyright 2017 The Shaderc Authors. All rights reserved.
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

#include "resource_parse.h"

#include <gmock/gmock.h>

namespace {

using glslc::ParseResourceSettings;
using glslc::ResourceSetting;
using testing::Eq;

struct ResourceSettingsCase {
  std::string input;
  bool success;
  std::vector<ResourceSetting> settings;
  std::string message;
};

using ParseResourceSettingsTest = ::testing::TestWithParam<ResourceSettingsCase>;

TEST_P(ParseResourceSettingsTest, Sample) {
  std::vector<ResourceSetting> settings;
  std::string err;
  const bool succeeded = ParseResourceSettings(GetParam().input, &settings, &err);
  EXPECT_THAT(succeeded, Eq(GetParam().success));
  EXPECT_THAT(settings, Eq(GetParam().settings));
  EXPECT_THAT(err, Eq(GetParam().message));
}

INSTANTIATE_TEST_SUITE_P(ParseResources, ParseResourceSettingsTest,
  ::testing::ValuesIn(std::vector<ResourceSettingsCase>{
    {"", true, {}, ""},
    {"   \t \t \n ", true, {}, ""},
    {" blorp blam", false, {}, "invalid resource limit: blorp"},
    {"MaxLightsxyz", false, {}, "invalid resource limit: MaxLightsxyz"},
    {"MaxLights", false, {}, "missing value after limit: MaxLights"},
    {"MaxLights x", false, {}, "invalid integer: x"},
    {"MaxLights 99x", false, {}, "invalid integer: 99x"},
    {"MaxLights 12 blam", false, {}, "invalid resource limit: blam"},
    {"MaxLights 12", true, {{shaderc_limit_max_lights, 12}}, ""},
    // Test negative number
    {"MinProgramTexelOffset -9", true, {{shaderc_limit_min_program_texel_offset, -9}}, ""},
    // Test leading, intervening, and trailing whitespace
    {" \tMaxLights \n 12 \t ", true, {{shaderc_limit_max_lights, 12}}, ""},
    // Test more than one limit setting.
    {"MinProgramTexelOffset -10 MaxLights 4", true, {{shaderc_limit_min_program_texel_offset, -10}, {shaderc_limit_max_lights, 4}}, ""},
    // Check ignore cases.
    {"nonInductiveForLoops", false, {}, "missing value after limit: nonInductiveForLoops"},
    {"nonInductiveForLoops 1", true, {}, ""},
    {"whileLoops 1", true, {}, ""},
    {"doWhileLoops 1", true, {}, ""},
    {"generalUniformIndexing 1", true, {}, ""},
    {"generalAttributeMatrixVectorIndexing 1", true, {}, ""},
    {"generalVaryingIndexing 1", true, {}, ""},
    {"generalSamplerIndexing 1", true, {}, ""},
    {"generalVariableIndexing 1", true, {}, ""},
    {"generalConstantMatrixVectorIndexing 1", true, {}, ""},
    // Check an ignore case with a regular case
    {"whileLoops 1 MaxLights 99", true, {{shaderc_limit_max_lights, 99}}, ""},
  }));

}  // anonymous namespace
