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

// Validation tests for illegal instructions

#include <string>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::spvtest::MakeInstruction;
using ::testing::Eq;

using ReservedSamplingInstTest = RoundTripTest;

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjImplicitLod) {
  std::string input = "%2 = OpImageSparseSampleProjImplicitLod %1 %3 %4\n";
  EXPECT_THAT(
      CompiledInstructions(input, SPV_ENV_UNIVERSAL_1_0),
      Eq(MakeInstruction(SpvOpImageSparseSampleProjImplicitLod, {1, 2, 3, 4})));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjExplicitLod) {
  std::string input =
      "%2 = OpImageSparseSampleProjExplicitLod %1 %3 %4 Lod %5\n";
  EXPECT_THAT(CompiledInstructions(input, SPV_ENV_UNIVERSAL_1_0),
              Eq(MakeInstruction(SpvOpImageSparseSampleProjExplicitLod,
                                 {1, 2, 3, 4, SpvImageOperandsLodMask, 5})));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjDrefImplicitLod) {
  std::string input =
      "%2 = OpImageSparseSampleProjDrefImplicitLod %1 %3 %4 %5\n";
  EXPECT_THAT(CompiledInstructions(input, SPV_ENV_UNIVERSAL_1_0),
              Eq(MakeInstruction(SpvOpImageSparseSampleProjDrefImplicitLod,
                                 {1, 2, 3, 4, 5})));
}

TEST_F(ReservedSamplingInstTest, OpImageSparseSampleProjDrefExplicitLod) {
  std::string input =
      "%2 = OpImageSparseSampleProjDrefExplicitLod %1 %3 %4 %5 Lod %6\n";
  EXPECT_THAT(CompiledInstructions(input, SPV_ENV_UNIVERSAL_1_0),
              Eq(MakeInstruction(SpvOpImageSparseSampleProjDrefExplicitLod,
                                 {1, 2, 3, 4, 5, SpvImageOperandsLodMask, 6})));
}

}  // namespace
}  // namespace spvtools
