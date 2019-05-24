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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateDerivatives = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "Fragment") {
  std::stringstream ss;
  ss << R"(
OpCapability Shader
OpCapability DerivativeControl
)";

  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\""
     << " %f32_var_input"
     << " %f32vec4_var_input"
     << "\n";
  if (execution_model == "Fragment") {
    ss << "OpExecutionMode %main OriginUpperLeft\n";
  }

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f32vec4 = OpTypeVector %f32 4

%f32_ptr_input = OpTypePointer Input %f32
%f32_var_input = OpVariable %f32_ptr_input Input

%f32vec4_ptr_input = OpTypePointer Input %f32vec4
%f32vec4_var_input = OpVariable %f32vec4_ptr_input Input

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_F(ValidateDerivatives, ScalarSuccess) {
  const std::string body = R"(
%f32_var = OpLoad %f32 %f32_var_input
%val1 = OpDPdx %f32 %f32_var
%val2 = OpDPdy %f32 %f32_var
%val3 = OpFwidth %f32 %f32_var
%val4 = OpDPdxFine %f32 %f32_var
%val5 = OpDPdyFine %f32 %f32_var
%val6 = OpFwidthFine %f32 %f32_var
%val7 = OpDPdxCoarse %f32 %f32_var
%val8 = OpDPdyCoarse %f32 %f32_var
%val9 = OpFwidthCoarse %f32 %f32_var
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateDerivatives, VectorSuccess) {
  const std::string body = R"(
%f32vec4_var = OpLoad %f32vec4 %f32vec4_var_input
%val1 = OpDPdx %f32vec4 %f32vec4_var
%val2 = OpDPdy %f32vec4 %f32vec4_var
%val3 = OpFwidth %f32vec4 %f32vec4_var
%val4 = OpDPdxFine %f32vec4 %f32vec4_var
%val5 = OpDPdyFine %f32vec4 %f32vec4_var
%val6 = OpFwidthFine %f32vec4 %f32vec4_var
%val7 = OpDPdxCoarse %f32vec4 %f32vec4_var
%val8 = OpDPdyCoarse %f32vec4 %f32vec4_var
%val9 = OpFwidthCoarse %f32vec4 %f32vec4_var
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateDerivatives, OpDPdxWrongResultType) {
  const std::string body = R"(
%f32_var = OpLoad %f32 %f32_var_input
%val1 = OpDPdx %u32 %f32vec4
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 10[%v4float] cannot "
                                               "be a type"));
}

TEST_F(ValidateDerivatives, OpDPdxWrongPType) {
  const std::string body = R"(
%f32vec4_var = OpLoad %f32vec4 %f32vec4_var_input
%val1 = OpDPdx %f32 %f32vec4_var
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected P type and Result Type to be the same: "
                        "DPdx"));
}

TEST_F(ValidateDerivatives, OpDPdxWrongExecutionModel) {
  const std::string body = R"(
%f32vec4_var = OpLoad %f32vec4 %f32vec4_var_input
%val1 = OpDPdx %f32vec4 %f32vec4_var
)";

  CompileSuccessfully(GenerateShaderCode(body, "", "Vertex").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Derivative instructions require Fragment execution model: DPdx"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
