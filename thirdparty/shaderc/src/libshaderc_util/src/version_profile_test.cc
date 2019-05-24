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

#include "libshaderc_util/version_profile.h"

#include "gmock/gmock.h"

namespace {

using shaderc_util::IsKnownVersion;
using shaderc_util::ParseVersionProfile;
using ::testing::Eq;
using ::testing::ValuesIn;


TEST(IsKnownVersionTest, Samples) {
  EXPECT_TRUE(IsKnownVersion(100));
  EXPECT_TRUE(IsKnownVersion(110));
  EXPECT_TRUE(IsKnownVersion(120));
  EXPECT_TRUE(IsKnownVersion(130));
  EXPECT_TRUE(IsKnownVersion(140));
  EXPECT_TRUE(IsKnownVersion(150));
  EXPECT_TRUE(IsKnownVersion(300));
  EXPECT_TRUE(IsKnownVersion(330));
  EXPECT_TRUE(IsKnownVersion(310));
  EXPECT_TRUE(IsKnownVersion(400));
  EXPECT_TRUE(IsKnownVersion(410));
  EXPECT_TRUE(IsKnownVersion(420));
  EXPECT_TRUE(IsKnownVersion(430));
  EXPECT_TRUE(IsKnownVersion(440));
  EXPECT_TRUE(IsKnownVersion(450));
  EXPECT_TRUE(IsKnownVersion(460));
  EXPECT_FALSE(IsKnownVersion(101));
  EXPECT_FALSE(IsKnownVersion(470));
}


struct ParseVersionProfileCase {
  std::string input;
  bool success;
  // The following are only used when success is true.
  int expected_version;
  EProfile expected_profile;
};

using ParseVersionProfileTest = ::testing::TestWithParam<ParseVersionProfileCase>;

TEST_P(ParseVersionProfileTest, Sample) {
  int version = 0;
  EProfile profile = EBadProfile;
  const bool result = ParseVersionProfile(GetParam().input, &version, &profile);
  EXPECT_THAT(result, GetParam().success);
  if (result) {
    EXPECT_THAT(version, GetParam().expected_version);
    EXPECT_THAT(profile, GetParam().expected_profile);
  }
}


// For OpenGL ES GLSL (ESSL) versions, see
// https://www.khronos.org/registry/OpenGL/index_e.php

INSTANTIATE_TEST_SUITE_P(OpenGLESCases, ParseVersionProfileTest,
                        ValuesIn(std::vector<ParseVersionProfileCase>{
                            {"100es", true, 100, EEsProfile},
                            {"300es", true, 300, EEsProfile},
                            {"310es", true, 310, EEsProfile},
                            {"320es", true, 320, EEsProfile},
                            {"99es", false, 0, EBadProfile},
                            {"500es", false, 0, EBadProfile},
                        }));

// For OpenGL GLSL versions, see
// https://www.khronos.org/registry/OpenGL/index_gl.php

INSTANTIATE_TEST_SUITE_P(OpenGLBlankCases, ParseVersionProfileTest,
                        ValuesIn(std::vector<ParseVersionProfileCase>{
                            {"110", true, 110, ENoProfile},
                            {"120", true, 120, ENoProfile},
                            {"130", true, 130, ENoProfile},
                            {"140", true, 140, ENoProfile},
                            {"150", true, 150, ENoProfile},
                            {"330", true, 330, ENoProfile},
                            {"400", true, 400, ENoProfile},
                            {"410", true, 410, ENoProfile},
                            {"420", true, 420, ENoProfile},
                            {"430", true, 430, ENoProfile},
                            {"440", true, 440, ENoProfile},
                            {"450", true, 450, ENoProfile},
                            {"460", true, 460, ENoProfile},
                            {"99", false, 0, EBadProfile},
                            {"500", false, 0, EBadProfile},
                        }));

INSTANTIATE_TEST_SUITE_P(OpenGLCoreCases, ParseVersionProfileTest,
                        ValuesIn(std::vector<ParseVersionProfileCase>{
                            {"320core", true, 320, ECoreProfile},
                            {"330core", true, 330, ECoreProfile},
                            {"400core", true, 400, ECoreProfile},
                            {"410core", true, 410, ECoreProfile},
                            {"420core", true, 420, ECoreProfile},
                            {"430core", true, 430, ECoreProfile},
                            {"440core", true, 440, ECoreProfile},
                            {"450core", true, 450, ECoreProfile},
                            {"460core", true, 460, ECoreProfile},
                        }));

INSTANTIATE_TEST_SUITE_P(
    OpenGLCompatibilityCases, ParseVersionProfileTest,
    ValuesIn(std::vector<ParseVersionProfileCase>{
        {"320compatibility", true, 320, ECompatibilityProfile},
        {"330compatibility", true, 330, ECompatibilityProfile},
        {"400compatibility", true, 400, ECompatibilityProfile},
        {"410compatibility", true, 410, ECompatibilityProfile},
        {"420compatibility", true, 420, ECompatibilityProfile},
        {"430compatibility", true, 430, ECompatibilityProfile},
        {"440compatibility", true, 440, ECompatibilityProfile},
        {"450compatibility", true, 450, ECompatibilityProfile},
        {"460compatibility", true, 460, ECompatibilityProfile},
    }));

}  // anonymous namespace
