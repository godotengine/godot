// Copyright (c) 2018 Google LLC
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

#include <string>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using StripLineReflectInfoTest = PassTest<::testing::Test>;

TEST_F(StripLineReflectInfoTest, StripHlslSemantic) {
  // This is a non-sensical example, but exercises the instructions.
  std::string before = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_decorate_string"
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpMemoryModel Logical Simple
OpDecorateStringGOOGLE %float HlslSemanticGOOGLE "foobar"
OpDecorateStringGOOGLE %void HlslSemanticGOOGLE "my goodness"
%void = OpTypeVoid
%float = OpTypeFloat 32
)";
  std::string after = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical Simple
%void = OpTypeVoid
%float = OpTypeFloat 32
)";

  SinglePassRunAndCheck<StripReflectInfoPass>(before, after, false);
}

TEST_F(StripLineReflectInfoTest, StripHlslCounterBuffer) {
  std::string before = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpMemoryModel Logical Simple
OpDecorateId %void HlslCounterBufferGOOGLE %float
%void = OpTypeVoid
%float = OpTypeFloat 32
)";
  std::string after = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical Simple
%void = OpTypeVoid
%float = OpTypeFloat 32
)";

  SinglePassRunAndCheck<StripReflectInfoPass>(before, after, false);
}

TEST_F(StripLineReflectInfoTest, StripHlslSemanticOnMember) {
  // This is a non-sensical example, but exercises the instructions.
  std::string before = R"(OpCapability Shader
OpCapability Linkage
OpExtension "SPV_GOOGLE_decorate_string"
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpMemoryModel Logical Simple
OpMemberDecorateStringGOOGLE %struct 0 HlslSemanticGOOGLE "foobar"
%float = OpTypeFloat 32
%_struct_3 = OpTypeStruct %float
)";
  std::string after = R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical Simple
%float = OpTypeFloat 32
%_struct_3 = OpTypeStruct %float
)";

  SinglePassRunAndCheck<StripReflectInfoPass>(before, after, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
