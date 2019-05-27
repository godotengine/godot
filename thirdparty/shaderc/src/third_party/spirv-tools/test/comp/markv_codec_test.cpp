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

// Tests for unique type declaration rules validator.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/comp/markv.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"
#include "tools/comp/markv_model_factory.h"

namespace spvtools {
namespace comp {
namespace {

using spvtest::ScopedContext;
using MarkvTest = ::testing::TestWithParam<MarkvModelType>;

void DiagnosticsMessageHandler(spv_message_level_t level, const char*,
                               const spv_position_t& position,
                               const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: " << position.index << ": " << message << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

// Compiles |code| to SPIR-V |words|.
void Compile(const std::string& code, std::vector<uint32_t>* words,
             uint32_t options = SPV_TEXT_TO_BINARY_OPTION_NONE,
             spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  spvtools::Context ctx(env);
  ctx.SetMessageConsumer(DiagnosticsMessageHandler);

  spv_binary spirv_binary;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinaryWithOptions(
                             ctx.CContext(), code.c_str(), code.size(), options,
                             &spirv_binary, nullptr));

  *words = std::vector<uint32_t>(spirv_binary->code,
                                 spirv_binary->code + spirv_binary->wordCount);

  spvBinaryDestroy(spirv_binary);
}

// Disassembles SPIR-V |words| to |out_text|.
void Disassemble(const std::vector<uint32_t>& words, std::string* out_text,
                 spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  spvtools::Context ctx(env);
  ctx.SetMessageConsumer(DiagnosticsMessageHandler);

  spv_text text = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryToText(ctx.CContext(), words.data(),
                                         words.size(), 0, &text, nullptr));
  assert(text);

  *out_text = std::string(text->str, text->length);
  spvTextDestroy(text);
}

// Encodes/decodes |original|, assembles/dissasembles |original|, then compares
// the results of the two operations.
void TestEncodeDecode(MarkvModelType model_type,
                      const std::string& original_text) {
  spvtools::Context ctx(SPV_ENV_UNIVERSAL_1_2);
  std::unique_ptr<MarkvModel> model = CreateMarkvModel(model_type);
  MarkvCodecOptions options;

  std::vector<uint32_t> expected_binary;
  Compile(original_text, &expected_binary);
  ASSERT_FALSE(expected_binary.empty());

  std::string expected_text;
  Disassemble(expected_binary, &expected_text);
  ASSERT_FALSE(expected_text.empty());

  std::vector<uint32_t> binary_to_encode;
  Compile(original_text, &binary_to_encode,
          SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_FALSE(binary_to_encode.empty());

  std::stringstream encoder_comments;
  const auto output_to_string_stream =
      [&encoder_comments](const std::string& str) { encoder_comments << str; };

  std::vector<uint8_t> markv;
  ASSERT_EQ(SPV_SUCCESS,
            SpirvToMarkv(ctx.CContext(), binary_to_encode, options, *model,
                         DiagnosticsMessageHandler, output_to_string_stream,
                         MarkvDebugConsumer(), &markv));
  ASSERT_FALSE(markv.empty());

  std::vector<uint32_t> decoded_binary;
  ASSERT_EQ(SPV_SUCCESS,
            MarkvToSpirv(ctx.CContext(), markv, options, *model,
                         DiagnosticsMessageHandler, MarkvLogConsumer(),
                         MarkvDebugConsumer(), &decoded_binary));
  ASSERT_FALSE(decoded_binary.empty());

  EXPECT_EQ(expected_binary, decoded_binary) << encoder_comments.str();

  std::string decoded_text;
  Disassemble(decoded_binary, &decoded_text);
  ASSERT_FALSE(decoded_text.empty());

  EXPECT_EQ(expected_text, decoded_text) << encoder_comments.str();
}

void TestEncodeDecodeShaderMainBody(MarkvModelType model_type,
                                    const std::string& body) {
  const std::string prefix =
      R"(
OpCapability Shader
OpCapability Int64
OpCapability Float64
%ext_inst = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f64 = OpTypeFloat 64
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1
%boolvec2 = OpTypeVector %bool 2
%s32vec2 = OpTypeVector %s32 2
%u32vec2 = OpTypeVector %u32 2
%f32vec2 = OpTypeVector %f32 2
%f64vec2 = OpTypeVector %f64 2
%boolvec3 = OpTypeVector %bool 3
%u32vec3 = OpTypeVector %u32 3
%s32vec3 = OpTypeVector %s32 3
%f32vec3 = OpTypeVector %f32 3
%f64vec3 = OpTypeVector %f64 3
%boolvec4 = OpTypeVector %bool 4
%u32vec4 = OpTypeVector %u32 4
%s32vec4 = OpTypeVector %s32 4
%f32vec4 = OpTypeVector %f32 4
%f64vec4 = OpTypeVector %f64 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4
%f32_pi = OpConstant %f32 3.14159

%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%s32_2 = OpConstant %s32 2
%s32_3 = OpConstant %s32 3
%s32_4 = OpConstant %s32 4
%s32_m1 = OpConstant %s32 -1

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec3_012 = OpConstantComposite %u32vec3 %u32_0 %u32_1 %u32_2
%u32vec3_123 = OpConstantComposite %u32vec3 %u32_1 %u32_2 %u32_3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3
%u32vec4_1234 = OpConstantComposite %u32vec4 %u32_1 %u32_2 %u32_3 %u32_4

%s32vec2_01 = OpConstantComposite %s32vec2 %s32_0 %s32_1
%s32vec2_12 = OpConstantComposite %s32vec2 %s32_1 %s32_2
%s32vec3_012 = OpConstantComposite %s32vec3 %s32_0 %s32_1 %s32_2
%s32vec3_123 = OpConstantComposite %s32vec3 %s32_1 %s32_2 %s32_3
%s32vec4_0123 = OpConstantComposite %s32vec4 %s32_0 %s32_1 %s32_2 %s32_3
%s32vec4_1234 = OpConstantComposite %s32vec4 %s32_1 %s32_2 %s32_3 %s32_4

%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string suffix =
      R"(
OpReturn
OpFunctionEnd)";

  TestEncodeDecode(model_type, prefix + body + suffix);
}

TEST_P(MarkvTest, U32Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%u32 = OpTypeInt 32 0
%100 = OpConstant %u32 0
%200 = OpConstant %u32 1
%300 = OpConstant %u32 4294967295
)");
}

TEST_P(MarkvTest, S32Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%s32 = OpTypeInt 32 1
%100 = OpConstant %s32 0
%200 = OpConstant %s32 1
%300 = OpConstant %s32 -1
%400 = OpConstant %s32 2147483647
%500 = OpConstant %s32 -2147483648
)");
}

TEST_P(MarkvTest, U64Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%u64 = OpTypeInt 64 0
%100 = OpConstant %u64 0
%200 = OpConstant %u64 1
%300 = OpConstant %u64 18446744073709551615
)");
}

TEST_P(MarkvTest, S64Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%s64 = OpTypeInt 64 1
%100 = OpConstant %s64 0
%200 = OpConstant %s64 1
%300 = OpConstant %s64 -1
%400 = OpConstant %s64 9223372036854775807
%500 = OpConstant %s64 -9223372036854775808
)");
}

TEST_P(MarkvTest, U16Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%u16 = OpTypeInt 16 0
%100 = OpConstant %u16 0
%200 = OpConstant %u16 1
%300 = OpConstant %u16 65535
)");
}

TEST_P(MarkvTest, S16Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%s16 = OpTypeInt 16 1
%100 = OpConstant %s16 0
%200 = OpConstant %s16 1
%300 = OpConstant %s16 -1
%400 = OpConstant %s16 32767
%500 = OpConstant %s16 -32768
)");
}

TEST_P(MarkvTest, F32Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%f32 = OpTypeFloat 32
%100 = OpConstant %f32 0
%200 = OpConstant %f32 1
%300 = OpConstant %f32 0.1
%400 = OpConstant %f32 -0.1
)");
}

TEST_P(MarkvTest, F64Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float64
OpMemoryModel Logical GLSL450
%f64 = OpTypeFloat 64
%100 = OpConstant %f64 0
%200 = OpConstant %f64 1
%300 = OpConstant %f64 0.1
%400 = OpConstant %f64 -0.1
)");
}

TEST_P(MarkvTest, F16Literal) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float16
OpMemoryModel Logical GLSL450
%f16 = OpTypeFloat 16
%100 = OpConstant %f16 0
%200 = OpConstant %f16 1
%300 = OpConstant %f16 0.1
%400 = OpConstant %f16 -0.1
)");
}

TEST_P(MarkvTest, StringLiteral) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpExtension "xxx"
OpExtension "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OpExtension ""
OpMemoryModel Logical GLSL450
)");
}

TEST_P(MarkvTest, WithFunction) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%100 = OpConstant %u32 1
%200 = OpConstant %u32 2
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%300 = OpIAdd %u32 %100 %200
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, WithMultipleFunctions) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%one = OpConstant %f32 1
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%f32_func = OpTypeFunction %f32 %f32
%sqr_plus_one = OpFunction %f32 None %f32_func
%x = OpFunctionParameter %f32
%100 = OpLabel
%x2 = OpFMul %f32 %x %x
%x2p1 = OpFunctionCall %f32 %plus_one %x2
OpReturnValue %x2p1
OpFunctionEnd
%plus_one = OpFunction %f32 None %f32_func
%y = OpFunctionParameter %f32
%200 = OpLabel
%yp1 = OpFAdd %f32 %y %one
OpReturnValue %yp1
OpFunctionEnd
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%1p1 = OpFunctionCall %f32 %sqr_plus_one %one
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, ForwardDeclaredId) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeInt 32 0
%3 = OpTypeVector %2 2
%4 = OpConstant %2 2
%5 = OpTypeArray %2 %4
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%1 = OpFunction %6 None %7
%8 = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, WithSwitch) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpCapability Int64
OpMemoryModel Physical32 OpenCL
%u64 = OpTypeInt 64 0
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%val = OpConstant %u64 1
%main = OpFunction %void None %void_func
%entry_main = OpLabel
OpSwitch %val %default 1 %case1 1000000000000 %case2
%case1 = OpLabel
OpNop
OpBranch %after_switch
%case2 = OpLabel
OpNop
OpBranch %after_switch
%default = OpLabel
OpNop
OpBranch %after_switch
%after_switch = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, WithLoop) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_main = OpLabel
OpLoopMerge %merge %continue DontUnroll|DependencyLength 10
OpBranch %begin_loop
%begin_loop = OpLabel
OpNop
OpBranch %continue
%continue = OpLabel
OpNop
OpBranch %begin_loop
%merge = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, WithDecorate) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 ArrayStride 4
OpDecorate %1 Uniform
%2 = OpTypeFloat 32
%1 = OpTypeRuntimeArray %2
)");
}

TEST_P(MarkvTest, WithExtInst) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
%opencl = OpExtInstImport "OpenCL.std"
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%100 = OpConstant %f32 1.1
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%200 = OpExtInst %f32 %opencl cos %100
OpReturn
OpFunctionEnd
)");
}

TEST_P(MarkvTest, F32Mul) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpFMul %f32 %f32_0 %f32_1
%val2 = OpFMul %f32 %f32_2 %f32_0
%val3 = OpFMul %f32 %f32_pi %f32_2
%val4 = OpFMul %f32 %f32_1 %f32_1
)");
}

TEST_P(MarkvTest, U32Mul) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpIMul %u32 %u32_0 %u32_1
%val2 = OpIMul %u32 %u32_2 %u32_0
%val3 = OpIMul %u32 %u32_3 %u32_2
%val4 = OpIMul %u32 %u32_1 %u32_1
)");
}

TEST_P(MarkvTest, S32Mul) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpIMul %s32 %s32_0 %s32_1
%val2 = OpIMul %s32 %s32_2 %s32_0
%val3 = OpIMul %s32 %s32_m1 %s32_2
%val4 = OpIMul %s32 %s32_1 %s32_1
)");
}

TEST_P(MarkvTest, F32Add) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpFAdd %f32 %f32_0 %f32_1
%val2 = OpFAdd %f32 %f32_2 %f32_0
%val3 = OpFAdd %f32 %f32_pi %f32_2
%val4 = OpFAdd %f32 %f32_1 %f32_1
)");
}

TEST_P(MarkvTest, U32Add) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpIAdd %u32 %u32_0 %u32_1
%val2 = OpIAdd %u32 %u32_2 %u32_0
%val3 = OpIAdd %u32 %u32_3 %u32_2
%val4 = OpIAdd %u32 %u32_1 %u32_1
)");
}

TEST_P(MarkvTest, S32Add) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%val1 = OpIAdd %s32 %s32_0 %s32_1
%val2 = OpIAdd %s32 %s32_2 %s32_0
%val3 = OpIAdd %s32 %s32_m1 %s32_2
%val4 = OpIAdd %s32 %s32_1 %s32_1
)");
}

TEST_P(MarkvTest, F32Dot) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%dot2_1 = OpDot %f32 %f32vec2_01 %f32vec2_12
%dot2_2 = OpDot %f32 %f32vec2_01 %f32vec2_01
%dot2_3 = OpDot %f32 %f32vec2_12 %f32vec2_12
%dot3_1 = OpDot %f32 %f32vec3_012 %f32vec3_123
%dot3_2 = OpDot %f32 %f32vec3_012 %f32vec3_012
%dot3_3 = OpDot %f32 %f32vec3_123 %f32vec3_123
%dot4_1 = OpDot %f32 %f32vec4_0123 %f32vec4_1234
%dot4_2 = OpDot %f32 %f32vec4_0123 %f32vec4_0123
%dot4_3 = OpDot %f32 %f32vec4_1234 %f32vec4_1234
)");
}

TEST_P(MarkvTest, F32VectorCompositeConstruct) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%cc1 = OpCompositeConstruct %f32vec4 %f32vec2_01 %f32vec2_12
%cc2 = OpCompositeConstruct %f32vec3 %f32vec2_01 %f32_2
%cc3 = OpCompositeConstruct %f32vec2 %f32_1 %f32_2
%cc4 = OpCompositeConstruct %f32vec4 %f32_1 %f32_2 %cc3
)");
}

TEST_P(MarkvTest, U32VectorCompositeConstruct) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%cc1 = OpCompositeConstruct %u32vec4 %u32vec2_01 %u32vec2_12
%cc2 = OpCompositeConstruct %u32vec3 %u32vec2_01 %u32_2
%cc3 = OpCompositeConstruct %u32vec2 %u32_1 %u32_2
%cc4 = OpCompositeConstruct %u32vec4 %u32_1 %u32_2 %cc3
)");
}

TEST_P(MarkvTest, S32VectorCompositeConstruct) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%cc1 = OpCompositeConstruct %u32vec4 %u32vec2_01 %u32vec2_12
%cc2 = OpCompositeConstruct %u32vec3 %u32vec2_01 %u32_2
%cc3 = OpCompositeConstruct %u32vec2 %u32_1 %u32_2
%cc4 = OpCompositeConstruct %u32vec4 %u32_1 %u32_2 %cc3
)");
}

TEST_P(MarkvTest, F32VectorCompositeExtract) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%f32vec4_3210 = OpCompositeConstruct %f32vec4 %f32_3 %f32_2 %f32_1 %f32_0
%f32vec3_013 = OpCompositeExtract %f32vec3 %f32vec4_0123 0 1 3
)");
}

TEST_P(MarkvTest, F32VectorComparison) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%f32vec4_3210 = OpCompositeConstruct %f32vec4 %f32_3 %f32_2 %f32_1 %f32_0
%c1 = OpFOrdEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c2 = OpFUnordEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c3 = OpFOrdNotEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c4 = OpFUnordNotEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c5 = OpFOrdLessThan %boolvec4 %f32vec4_0123 %f32vec4_3210
%c6 = OpFUnordLessThan %boolvec4 %f32vec4_0123 %f32vec4_3210
%c7 = OpFOrdGreaterThan %boolvec4 %f32vec4_0123 %f32vec4_3210
%c8 = OpFUnordGreaterThan %boolvec4 %f32vec4_0123 %f32vec4_3210
%c9 = OpFOrdLessThanEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c10 = OpFUnordLessThanEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c11 = OpFOrdGreaterThanEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
%c12 = OpFUnordGreaterThanEqual %boolvec4 %f32vec4_0123 %f32vec4_3210
)");
}

TEST_P(MarkvTest, VectorShuffle) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%f32vec4_3210 = OpCompositeConstruct %f32vec4 %f32_3 %f32_2 %f32_1 %f32_0
%sh1 = OpVectorShuffle %f32vec2 %f32vec4_0123 %f32vec4_3210 3 6
%sh2 = OpVectorShuffle %f32vec3 %f32vec2_01 %f32vec4_3210 0 3 4
)");
}

TEST_P(MarkvTest, VectorTimesScalar) {
  TestEncodeDecodeShaderMainBody(GetParam(), R"(
%f32vec4_3210 = OpCompositeConstruct %f32vec4 %f32_3 %f32_2 %f32_1 %f32_0
%res1 = OpVectorTimesScalar %f32vec4 %f32vec4_0123 %f32_2
%res2 = OpVectorTimesScalar %f32vec4 %f32vec4_3210 %f32_2
)");
}

TEST_P(MarkvTest, SpirvSpecSample) {
  TestEncodeDecode(GetParam(), R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %31 %33 %42 %57
               OpExecutionMode %4 OriginLowerLeft

; Debug information
               OpSource GLSL 450
               OpName %4 "main"
               OpName %9 "scale"
               OpName %17 "S"
               OpMemberName %17 0 "b"
               OpMemberName %17 1 "v"
               OpMemberName %17 2 "i"
               OpName %18 "blockName"
               OpMemberName %18 0 "s"
               OpMemberName %18 1 "cond"
               OpName %20 ""
               OpName %31 "color"
               OpName %33 "color1"
               OpName %42 "color2"
               OpName %48 "i"
               OpName %57 "multiplier"

; Annotations (non-debug)
               OpDecorate %15 ArrayStride 16
               OpMemberDecorate %17 0 Offset 0
               OpMemberDecorate %17 1 Offset 16
               OpMemberDecorate %17 2 Offset 96
               OpMemberDecorate %18 0 Offset 0
               OpMemberDecorate %18 1 Offset 112
               OpDecorate %18 Block
               OpDecorate %20 DescriptorSet 0
               OpDecorate %42 NoPerspective

; All types, variables, and constants
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2                      ; void ()
          %6 = OpTypeFloat 32                         ; 32-bit float
          %7 = OpTypeVector %6 4                      ; vec4
          %8 = OpTypePointer Function %7              ; function-local vec4*
         %10 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpConstantComposite %7 %10 %10 %11 %10 ; vec4(1.0, 1.0, 2.0, 1.0)
         %13 = OpTypeInt 32 0                         ; 32-bit int, sign-less
         %14 = OpConstant %13 5
         %15 = OpTypeArray %7 %14
         %16 = OpTypeInt 32 1
         %17 = OpTypeStruct %13 %15 %16
         %18 = OpTypeStruct %17 %13
         %19 = OpTypePointer Uniform %18
         %20 = OpVariable %19 Uniform
         %21 = OpConstant %16 1
         %22 = OpTypePointer Uniform %13
         %25 = OpTypeBool
         %26 = OpConstant %13 0
         %30 = OpTypePointer Output %7
         %31 = OpVariable %30 Output
         %32 = OpTypePointer Input %7
         %33 = OpVariable %32 Input
         %35 = OpConstant %16 0
         %36 = OpConstant %16 2
         %37 = OpTypePointer Uniform %7
         %42 = OpVariable %32 Input
         %47 = OpTypePointer Function %16
         %55 = OpConstant %16 4
         %57 = OpVariable %32 Input

; All functions
          %4 = OpFunction %2 None %3                  ; main()
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %48 = OpVariable %47 Function
               OpStore %9 %12
         %23 = OpAccessChain %22 %20 %21              ; location of cond
         %24 = OpLoad %13 %23                         ; load 32-bit int from cond
         %27 = OpINotEqual %25 %24 %26                ; convert to bool
               OpSelectionMerge %29 None              ; structured if
               OpBranchConditional %27 %28 %41        ; if cond
         %28 = OpLabel                                ; then
         %34 = OpLoad %7 %33
         %38 = OpAccessChain %37 %20 %35 %21 %36      ; s.v[2]
         %39 = OpLoad %7 %38
         %40 = OpFAdd %7 %34 %39
               OpStore %31 %40
               OpBranch %29
         %41 = OpLabel                                ; else
         %43 = OpLoad %7 %42
         %44 = OpExtInst %7 %1 Sqrt %43               ; extended instruction sqrt
         %45 = OpLoad %7 %9
         %46 = OpFMul %7 %44 %45
               OpStore %31 %46
               OpBranch %29
         %29 = OpLabel                                ; endif
               OpStore %48 %35
               OpBranch %49
         %49 = OpLabel
               OpLoopMerge %51 %52 None               ; structured loop
               OpBranch %53
         %53 = OpLabel
         %54 = OpLoad %16 %48
         %56 = OpSLessThan %25 %54 %55                ; i < 4 ?
               OpBranchConditional %56 %50 %51        ; body or break
         %50 = OpLabel                                ; body
         %58 = OpLoad %7 %57
         %59 = OpLoad %7 %31
         %60 = OpFMul %7 %59 %58
               OpStore %31 %60
               OpBranch %52
         %52 = OpLabel                                ; continue target
         %61 = OpLoad %16 %48
         %62 = OpIAdd %16 %61 %21                     ; ++i
               OpStore %48 %62
               OpBranch %49                           ; loop back
         %51 = OpLabel                                ; loop merge point
               OpReturn
               OpFunctionEnd
)");
}

TEST_P(MarkvTest, SampleFromDeadBranchEliminationTest) {
  TestEncodeDecode(GetParam(), R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%12 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%14 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%main = OpFunction %void None %5
%17 = OpLabel
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
%21 = OpPhi %v4float %12 %19 %14 %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)");
}

INSTANTIATE_TEST_SUITE_P(AllMarkvModels, MarkvTest,
                         ::testing::ValuesIn(std::vector<MarkvModelType>{
                             kMarkvModelShaderLite,
                             kMarkvModelShaderMid,
                             kMarkvModelShaderMax,
                         }));

}  // namespace
}  // namespace comp
}  // namespace spvtools
