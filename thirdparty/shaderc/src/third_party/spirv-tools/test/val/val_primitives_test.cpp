// Copyright (c) 2017 LunarG Inc.
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

using ValidatePrimitives = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions =
        "OpCapability GeometryStreams",
    const std::string& execution_model = "Geometry") {
  std::ostringstream ss;
  ss << capabilities_and_extensions << "\n";
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";
  if (execution_model == "Geometry") {
    ss << "OpExecutionMode %main InputPoints\n";
    ss << "OpExecutionMode %main OutputPoints\n";
  }

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u32vec4 = OpTypeVector %u32 4

%f32_0 = OpConstant %f32 0
%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

// Returns SPIR-V assembly fragment representing a function call,
// the end of the callee body, and the preamble and body of the called
// function with the given body, but missing the final return and
// function-end.  The result is of the form where it can be used in the
// |body| argument to GenerateShaderCode.
std::string CallAndCallee(const std::string& body) {
  std::ostringstream ss;
  ss << R"(
%dummy = OpFunctionCall %void %foo
OpReturn
OpFunctionEnd

%foo = OpFunction %void None %func
%foo_entry = OpLabel
)";

  ss << body;

  return ss.str();
}

// OpEmitVertex doesn't have any parameters, so other validation
// is handled by the binary parser, and generic dominance checks.
TEST_F(ValidatePrimitives, EmitVertexSuccess) {
  CompileSuccessfully(
      GenerateShaderCode("OpEmitVertex", "OpCapability Geometry"));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidatePrimitives, EmitVertexFailMissingCapability) {
  CompileSuccessfully(
      GenerateShaderCode("OpEmitVertex", "OpCapability Shader", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Opcode EmitVertex requires one of these capabilities: Geometry"));
}

TEST_F(ValidatePrimitives, EmitVertexFailWrongExecutionMode) {
  CompileSuccessfully(
      GenerateShaderCode("OpEmitVertex", "OpCapability Geometry", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("EmitVertex instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives, EmitVertexFailWrongExecutionModeNestedFunction) {
  CompileSuccessfully(GenerateShaderCode(CallAndCallee("OpEmitVertex"),
                                         "OpCapability Geometry", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("EmitVertex instructions require Geometry execution model"));
}

// OpEndPrimitive doesn't have any parameters, so other validation
// is handled by the binary parser, and generic dominance checks.
TEST_F(ValidatePrimitives, EndPrimitiveSuccess) {
  CompileSuccessfully(
      GenerateShaderCode("OpEndPrimitive", "OpCapability Geometry"));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidatePrimitives, EndPrimitiveFailMissingCapability) {
  CompileSuccessfully(
      GenerateShaderCode("OpEndPrimitive", "OpCapability Shader", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Opcode EndPrimitive requires one of these capabilities: Geometry"));
}

TEST_F(ValidatePrimitives, EndPrimitiveFailWrongExecutionMode) {
  CompileSuccessfully(
      GenerateShaderCode("OpEndPrimitive", "OpCapability Geometry", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("EndPrimitive instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives, EndPrimitiveFailWrongExecutionModeNestedFunction) {
  CompileSuccessfully(GenerateShaderCode(CallAndCallee("OpEndPrimitive"),
                                         "OpCapability Geometry", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("EndPrimitive instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives, EmitStreamVertexSuccess) {
  const std::string body = R"(
OpEmitStreamVertex %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidatePrimitives, EmitStreamVertexFailMissingCapability) {
  CompileSuccessfully(GenerateShaderCode("OpEmitStreamVertex %u32_0",
                                         "OpCapability Shader", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode EmitStreamVertex requires one of these "
                        "capabilities: GeometryStreams"));
}

TEST_F(ValidatePrimitives, EmitStreamVertexFailWrongExecutionMode) {
  CompileSuccessfully(GenerateShaderCode(
      "OpEmitStreamVertex %u32_0", "OpCapability GeometryStreams", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "EmitStreamVertex instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives,
       EmitStreamVertexFailWrongExecutionModeNestedFunction) {
  CompileSuccessfully(
      GenerateShaderCode(CallAndCallee("OpEmitStreamVertex %u32_0"),
                         "OpCapability GeometryStreams", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "EmitStreamVertex instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives, EmitStreamVertexNonInt) {
  const std::string body = R"(
OpEmitStreamVertex %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EmitStreamVertex: "
                        "expected Stream to be int scalar"));
}

TEST_F(ValidatePrimitives, EmitStreamVertexNonScalar) {
  const std::string body = R"(
OpEmitStreamVertex %u32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EmitStreamVertex: "
                        "expected Stream to be int scalar"));
}

TEST_F(ValidatePrimitives, EmitStreamVertexNonConstant) {
  const std::string body = R"(
%val1 = OpIAdd %u32 %u32_0 %u32_1
OpEmitStreamVertex %val1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EmitStreamVertex: "
                        "expected Stream to be constant instruction"));
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveSuccess) {
  const std::string body = R"(
OpEndStreamPrimitive %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveFailMissingCapability) {
  CompileSuccessfully(GenerateShaderCode("OpEndStreamPrimitive %u32_0",
                                         "OpCapability Shader", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode EndStreamPrimitive requires one of these "
                        "capabilities: GeometryStreams"));
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveFailWrongExecutionMode) {
  CompileSuccessfully(GenerateShaderCode(
      "OpEndStreamPrimitive %u32_0", "OpCapability GeometryStreams", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "EndStreamPrimitive instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives,
       EndStreamPrimitiveFailWrongExecutionModeNestedFunction) {
  CompileSuccessfully(
      GenerateShaderCode(CallAndCallee("OpEndStreamPrimitive %u32_0"),
                         "OpCapability GeometryStreams", "Vertex"));
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "EndStreamPrimitive instructions require Geometry execution model"));
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveNonInt) {
  const std::string body = R"(
OpEndStreamPrimitive %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EndStreamPrimitive: "
                        "expected Stream to be int scalar"));
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveNonScalar) {
  const std::string body = R"(
OpEndStreamPrimitive %u32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EndStreamPrimitive: "
                        "expected Stream to be int scalar"));
}

TEST_F(ValidatePrimitives, EndStreamPrimitiveNonConstant) {
  const std::string body = R"(
%val1 = OpIAdd %u32 %u32_0 %u32_1
OpEndStreamPrimitive %val1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("EndStreamPrimitive: "
                        "expected Stream to be constant instruction"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
