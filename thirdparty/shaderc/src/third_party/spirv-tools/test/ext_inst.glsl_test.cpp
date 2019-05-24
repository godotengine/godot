// Copyright (c) 2015-2016 The Khronos Group Inc.
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
#include <string>
#include <vector>

#include "source/latest_version_glsl_std_450_header.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

/// Context for an extended instruction.
///
/// Information about a GLSL extended instruction (including its opname, return
/// type, etc.) and related instructions used to generate the return type and
/// constant as the operands. Used in generating extended instruction tests.
struct ExtInstContext {
  const char* extInstOpName;
  const char* extInstOperandVars;
  /// The following fields are used to check the SPIR-V binary representation
  /// of this instruction.
  uint32_t extInstOpcode;  ///< Opcode value for this extended instruction.
  uint32_t extInstLength;  ///< Wordcount of this extended instruction.
  std::vector<uint32_t> extInstOperandIds;  ///< Ids for operands.
};

using ExtInstGLSLstd450RoundTripTest = ::testing::TestWithParam<ExtInstContext>;

TEST_P(ExtInstGLSLstd450RoundTripTest, ParameterizedExtInst) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  const std::string spirv = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Simple
OpEntryPoint Vertex %2 "main"
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%2 = OpFunction %3 None %5
%6 = OpLabel
%8 = OpExtInst %7 %1 )" + std::string(GetParam().extInstOpName) +
                            " " + GetParam().extInstOperandVars + R"(
OpReturn
OpFunctionEnd
)";
  const std::string spirv_header =
      R"(; SPIR-V
; Version: 1.0
; Generator: Khronos SPIR-V Tools Assembler; 0
; Bound: 9
; Schema: 0)";
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic;
  spv_result_t error = spvTextToBinary(context, spirv.c_str(), spirv.size(),
                                       &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error)
        << "Source was: " << std::endl
        << spirv << std::endl
        << "Test case for : " << GetParam().extInstOpName << std::endl;
  }

  // Check we do have the extended instruction's corresponding binary code in
  // the generated SPIR-V binary.
  std::vector<uint32_t> expected_contains(
      {12 /*OpExtInst*/ | GetParam().extInstLength << 16, 7 /*return type*/,
       8 /*result id*/, 1 /*glsl450 import*/, GetParam().extInstOpcode});
  for (uint32_t operand : GetParam().extInstOperandIds) {
    expected_contains.push_back(operand);
  }
  EXPECT_NE(binary->code + binary->wordCount,
            std::search(binary->code, binary->code + binary->wordCount,
                        expected_contains.begin(), expected_contains.end()))
      << "Cannot find\n"
      << spvtest::WordVector(expected_contains).str() << "in\n"
      << spvtest::WordVector(*binary).str();

  // Check round trip gives the same text.
  spv_text output_text = nullptr;
  error = spvBinaryToText(context, binary->code, binary->wordCount,
                          SPV_BINARY_TO_TEXT_OPTION_NONE, &output_text,
                          &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
  EXPECT_EQ(spirv_header + spirv, output_text->str);
  spvTextDestroy(output_text);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

INSTANTIATE_TEST_SUITE_P(
    ExtInstParameters, ExtInstGLSLstd450RoundTripTest,
    ::testing::ValuesIn(std::vector<ExtInstContext>({
        // We are only testing the correctness of encoding and decoding here.
        // Semantic correctness should be the responsibility of validator. So
        // some of the instructions below have incorrect operand and/or return
        // types, e.g, Modf, ModfStruct, etc.
        {"Round", "%5", 1, 6, {5}},
        {"RoundEven", "%5", 2, 6, {5}},
        {"Trunc", "%5", 3, 6, {5}},
        {"FAbs", "%5", 4, 6, {5}},
        {"SAbs", "%5", 5, 6, {5}},
        {"FSign", "%5", 6, 6, {5}},
        {"SSign", "%5", 7, 6, {5}},
        {"Floor", "%5", 8, 6, {5}},
        {"Ceil", "%5", 9, 6, {5}},
        {"Fract", "%5", 10, 6, {5}},
        {"Radians", "%5", 11, 6, {5}},
        {"Degrees", "%5", 12, 6, {5}},
        {"Sin", "%5", 13, 6, {5}},
        {"Cos", "%5", 14, 6, {5}},
        {"Tan", "%5", 15, 6, {5}},
        {"Asin", "%5", 16, 6, {5}},
        {"Acos", "%5", 17, 6, {5}},
        {"Atan", "%5", 18, 6, {5}},
        {"Sinh", "%5", 19, 6, {5}},
        {"Cosh", "%5", 20, 6, {5}},
        {"Tanh", "%5", 21, 6, {5}},
        {"Asinh", "%5", 22, 6, {5}},
        {"Acosh", "%5", 23, 6, {5}},
        {"Atanh", "%5", 24, 6, {5}},
        {"Atan2", "%5 %5", 25, 7, {5, 5}},
        {"Pow", "%5 %5", 26, 7, {5, 5}},
        {"Exp", "%5", 27, 6, {5}},
        {"Log", "%5", 28, 6, {5}},
        {"Exp2", "%5", 29, 6, {5}},
        {"Log2", "%5", 30, 6, {5}},
        {"Sqrt", "%5", 31, 6, {5}},
        {"InverseSqrt", "%5", 32, 6, {5}},
        {"Determinant", "%5", 33, 6, {5}},
        {"MatrixInverse", "%5", 34, 6, {5}},
        {"Modf", "%5 %5", 35, 7, {5, 5}},
        {"ModfStruct", "%5", 36, 6, {5}},
        {"FMin", "%5 %5", 37, 7, {5, 5}},
        {"UMin", "%5 %5", 38, 7, {5, 5}},
        {"SMin", "%5 %5", 39, 7, {5, 5}},
        {"FMax", "%5 %5", 40, 7, {5, 5}},
        {"UMax", "%5 %5", 41, 7, {5, 5}},
        {"SMax", "%5 %5", 42, 7, {5, 5}},
        {"FClamp", "%5 %5 %5", 43, 8, {5, 5, 5}},
        {"UClamp", "%5 %5 %5", 44, 8, {5, 5, 5}},
        {"SClamp", "%5 %5 %5", 45, 8, {5, 5, 5}},
        {"FMix", "%5 %5 %5", 46, 8, {5, 5, 5}},
        {"IMix", "%5 %5 %5", 47, 8, {5, 5, 5}},  // Bug 15452. Reserved.
        {"Step", "%5 %5", 48, 7, {5, 5}},
        {"SmoothStep", "%5 %5 %5", 49, 8, {5, 5, 5}},
        {"Fma", "%5 %5 %5", 50, 8, {5, 5, 5}},
        {"Frexp", "%5 %5", 51, 7, {5, 5}},
        {"FrexpStruct", "%5", 52, 6, {5}},
        {"Ldexp", "%5 %5", 53, 7, {5, 5}},
        {"PackSnorm4x8", "%5", 54, 6, {5}},
        {"PackUnorm4x8", "%5", 55, 6, {5}},
        {"PackSnorm2x16", "%5", 56, 6, {5}},
        {"PackUnorm2x16", "%5", 57, 6, {5}},
        {"PackHalf2x16", "%5", 58, 6, {5}},
        {"PackDouble2x32", "%5", 59, 6, {5}},
        {"UnpackSnorm2x16", "%5", 60, 6, {5}},
        {"UnpackUnorm2x16", "%5", 61, 6, {5}},
        {"UnpackHalf2x16", "%5", 62, 6, {5}},
        {"UnpackSnorm4x8", "%5", 63, 6, {5}},
        {"UnpackUnorm4x8", "%5", 64, 6, {5}},
        {"UnpackDouble2x32", "%5", 65, 6, {5}},
        {"Length", "%5", 66, 6, {5}},
        {"Distance", "%5 %5", 67, 7, {5, 5}},
        {"Cross", "%5 %5", 68, 7, {5, 5}},
        {"Normalize", "%5", 69, 6, {5}},
        // clang-format off
        {"FaceForward", "%5 %5 %5", 70, 8, {5, 5, 5}},
        // clang-format on
        {"Reflect", "%5 %5", 71, 7, {5, 5}},
        {"Refract", "%5 %5 %5", 72, 8, {5, 5, 5}},
        {"FindILsb", "%5", 73, 6, {5}},
        {"FindSMsb", "%5", 74, 6, {5}},
        {"FindUMsb", "%5", 75, 6, {5}},
        {"InterpolateAtCentroid", "%5", 76, 6, {5}},
        // clang-format off
        {"InterpolateAtSample", "%5 %5", 77, 7, {5, 5}},
        {"InterpolateAtOffset", "%5 %5", 78, 7, {5, 5}},
        // clang-format on
        {"NMin", "%5 %5", 79, 7, {5, 5}},
        {"NMax", "%5 %5", 80, 7, {5, 5}},
        {"NClamp", "%5 %5 %5", 81, 8, {5, 5, 5}},
    })));

}  // namespace
}  // namespace spvtools
