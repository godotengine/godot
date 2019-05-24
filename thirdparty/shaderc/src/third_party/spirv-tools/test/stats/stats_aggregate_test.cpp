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

#include <string>
#include <unordered_map>

#include "test/test_fixture.h"
#include "test/unit_spirv.h"
#include "tools/stats/spirv_stats.h"

namespace spvtools {
namespace stats {
namespace {

using spvtest::ScopedContext;

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

// Calls AggregateStats for binary compiled from |code|.
void CompileAndAggregateStats(const std::string& code, SpirvStats* stats,
                              spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  spvtools::Context ctx(env);
  ctx.SetMessageConsumer(DiagnosticsMessageHandler);
  spv_binary binary;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(ctx.CContext(), code.c_str(),
                                         code.size(), &binary, nullptr));

  ASSERT_EQ(SPV_SUCCESS, AggregateStats(ctx.CContext(), binary->code,
                                        binary->wordCount, nullptr, stats));
  spvBinaryDestroy(binary);
}

TEST(AggregateStats, CapabilityHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(4u, stats.capability_hist.size());
  EXPECT_EQ(0u, stats.capability_hist.count(SpvCapabilityShader));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5u, stats.capability_hist.size());
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(5u, stats.capability_hist.size());
  EXPECT_EQ(1u, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(3u, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5u, stats.capability_hist.size());
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(2u, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(4u, stats.capability_hist.at(SpvCapabilityLinkage));
}

TEST(AggregateStats, ExtensionHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Physical32 OpenCL
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_viewport_array2"
OpExtension "greatest_extension_ever"
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1u, stats.extension_hist.size());
  EXPECT_EQ(0u, stats.extension_hist.count("SPV_NV_viewport_array2"));
  EXPECT_EQ(1u, stats.extension_hist.at("SPV_KHR_16bit_storage"));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(3u, stats.extension_hist.size());
  EXPECT_EQ(1u, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(1u, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(1u, stats.extension_hist.at("greatest_extension_ever"));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(3u, stats.extension_hist.size());
  EXPECT_EQ(1u, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(2u, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(1u, stats.extension_hist.at("greatest_extension_ever"));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(3u, stats.extension_hist.size());
  EXPECT_EQ(2u, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(2u, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(2u, stats.extension_hist.at("greatest_extension_ever"));
}

TEST(AggregateStats, VersionHistogram) {
  const std::string code1 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1u, stats.version_hist.size());
  EXPECT_EQ(1u, stats.version_hist.at(0x00010100));

  CompileAndAggregateStats(code1, &stats, SPV_ENV_UNIVERSAL_1_0);
  EXPECT_EQ(2u, stats.version_hist.size());
  EXPECT_EQ(1u, stats.version_hist.at(0x00010100));
  EXPECT_EQ(1u, stats.version_hist.at(0x00010000));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(2u, stats.version_hist.size());
  EXPECT_EQ(2u, stats.version_hist.at(0x00010100));
  EXPECT_EQ(1u, stats.version_hist.at(0x00010000));

  CompileAndAggregateStats(code1, &stats, SPV_ENV_UNIVERSAL_1_0);
  EXPECT_EQ(2u, stats.version_hist.size());
  EXPECT_EQ(2u, stats.version_hist.at(0x00010100));
  EXPECT_EQ(2u, stats.version_hist.at(0x00010000));
}

TEST(AggregateStats, GeneratorHistogram) {
  const std::string code1 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  const uint32_t kGeneratorKhronosAssembler = SPV_GENERATOR_KHRONOS_ASSEMBLER
                                              << 16;

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1u, stats.generator_hist.size());
  EXPECT_EQ(1u, stats.generator_hist.at(kGeneratorKhronosAssembler));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1u, stats.generator_hist.size());
  EXPECT_EQ(2u, stats.generator_hist.at(kGeneratorKhronosAssembler));
}

TEST(AggregateStats, OpcodeHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Int64
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%u64 = OpTypeInt 64 0
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_viewport_array2"
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(4u, stats.opcode_hist.size());
  EXPECT_EQ(4u, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(1u, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(1u, stats.opcode_hist.at(SpvOpTypeFloat));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5u, stats.opcode_hist.size());
  EXPECT_EQ(6u, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(1u, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(1u, stats.opcode_hist.at(SpvOpExtension));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(5u, stats.opcode_hist.size());
  EXPECT_EQ(10u, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(3u, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(4u, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(1u, stats.opcode_hist.at(SpvOpExtension));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5u, stats.opcode_hist.size());
  EXPECT_EQ(12u, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(4u, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(4u, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(2u, stats.opcode_hist.at(SpvOpExtension));
}

TEST(AggregateStats, OpcodeMarkovHistogram) {
  const std::string code1 = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_viewport_array2"
OpMemoryModel Logical GLSL450
)";

  const std::string code2 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Int64
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%u64 = OpTypeInt 64 0
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
)";

  SpirvStats stats;
  stats.opcode_markov_hist.resize(2);

  CompileAndAggregateStats(code1, &stats);
  ASSERT_EQ(2u, stats.opcode_markov_hist.size());
  EXPECT_EQ(2u, stats.opcode_markov_hist[0].size());
  EXPECT_EQ(2u, stats.opcode_markov_hist[0].at(SpvOpCapability).size());
  EXPECT_EQ(1u, stats.opcode_markov_hist[0].at(SpvOpExtension).size());
  EXPECT_EQ(
      1u, stats.opcode_markov_hist[0].at(SpvOpCapability).at(SpvOpCapability));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[0].at(SpvOpCapability).at(SpvOpExtension));
  EXPECT_EQ(
      1u, stats.opcode_markov_hist[0].at(SpvOpExtension).at(SpvOpMemoryModel));

  EXPECT_EQ(1u, stats.opcode_markov_hist[1].size());
  EXPECT_EQ(2u, stats.opcode_markov_hist[1].at(SpvOpCapability).size());
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpExtension));
  EXPECT_EQ(
      1u, stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpMemoryModel));

  CompileAndAggregateStats(code2, &stats);
  ASSERT_EQ(2u, stats.opcode_markov_hist.size());
  EXPECT_EQ(4u, stats.opcode_markov_hist[0].size());
  EXPECT_EQ(3u, stats.opcode_markov_hist[0].at(SpvOpCapability).size());
  EXPECT_EQ(1u, stats.opcode_markov_hist[0].at(SpvOpExtension).size());
  EXPECT_EQ(1u, stats.opcode_markov_hist[0].at(SpvOpMemoryModel).size());
  EXPECT_EQ(2u, stats.opcode_markov_hist[0].at(SpvOpTypeInt).size());
  EXPECT_EQ(
      4u, stats.opcode_markov_hist[0].at(SpvOpCapability).at(SpvOpCapability));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[0].at(SpvOpCapability).at(SpvOpExtension));
  EXPECT_EQ(
      1u, stats.opcode_markov_hist[0].at(SpvOpCapability).at(SpvOpMemoryModel));
  EXPECT_EQ(
      1u, stats.opcode_markov_hist[0].at(SpvOpExtension).at(SpvOpMemoryModel));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[0].at(SpvOpMemoryModel).at(SpvOpTypeInt));
  EXPECT_EQ(1u, stats.opcode_markov_hist[0].at(SpvOpTypeInt).at(SpvOpTypeInt));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[0].at(SpvOpTypeInt).at(SpvOpTypeFloat));

  EXPECT_EQ(3u, stats.opcode_markov_hist[1].size());
  EXPECT_EQ(4u, stats.opcode_markov_hist[1].at(SpvOpCapability).size());
  EXPECT_EQ(1u, stats.opcode_markov_hist[1].at(SpvOpMemoryModel).size());
  EXPECT_EQ(1u, stats.opcode_markov_hist[1].at(SpvOpTypeInt).size());
  EXPECT_EQ(
      2u, stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpCapability));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpExtension));
  EXPECT_EQ(
      2u, stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpMemoryModel));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[1].at(SpvOpCapability).at(SpvOpTypeInt));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[1].at(SpvOpMemoryModel).at(SpvOpTypeInt));
  EXPECT_EQ(1u,
            stats.opcode_markov_hist[1].at(SpvOpTypeInt).at(SpvOpTypeFloat));
}

TEST(AggregateStats, ConstantLiteralsHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpCapability Float64
OpCapability Int16
OpCapability Int64
OpMemoryModel Physical32 OpenCL
%u16 = OpTypeInt 16 0
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%1 = OpConstant %f32 0.1
%2 = OpConstant %f32 -2
%3 = OpConstant %f64 -2
%4 = OpConstant %u16 16
%5 = OpConstant %u16 2
%6 = OpConstant %u32 32
%7 = OpConstant %u64 64
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpCapability Int64
OpMemoryModel Logical GLSL450
%f32 = OpTypeFloat 32
%u16 = OpTypeInt 16 0
%s16 = OpTypeInt 16 1
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1
%1 = OpConstant %f32 0.1
%2 = OpConstant %f32 -2
%3 = OpConstant %u16 1
%4 = OpConstant %u16 16
%5 = OpConstant %u16 2
%6 = OpConstant %s16 -16
%7 = OpConstant %u32 32
%8 = OpConstant %s32 2
%9 = OpConstant %s32 -32
%10 = OpConstant %u64 64
%11 = OpConstant %s64 -64
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(2u, stats.f32_constant_hist.size());
  EXPECT_EQ(1u, stats.f64_constant_hist.size());
  EXPECT_EQ(1u, stats.f32_constant_hist.at(0.1f));
  EXPECT_EQ(1u, stats.f32_constant_hist.at(-2.f));
  EXPECT_EQ(1u, stats.f64_constant_hist.at(-2));

  EXPECT_EQ(2u, stats.u16_constant_hist.size());
  EXPECT_EQ(0u, stats.s16_constant_hist.size());
  EXPECT_EQ(1u, stats.u32_constant_hist.size());
  EXPECT_EQ(0u, stats.s32_constant_hist.size());
  EXPECT_EQ(1u, stats.u64_constant_hist.size());
  EXPECT_EQ(0u, stats.s64_constant_hist.size());
  EXPECT_EQ(1u, stats.u16_constant_hist.at(16));
  EXPECT_EQ(1u, stats.u16_constant_hist.at(2));
  EXPECT_EQ(1u, stats.u32_constant_hist.at(32));
  EXPECT_EQ(1u, stats.u64_constant_hist.at(64));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(2u, stats.f32_constant_hist.size());
  EXPECT_EQ(1u, stats.f64_constant_hist.size());
  EXPECT_EQ(2u, stats.f32_constant_hist.at(0.1f));
  EXPECT_EQ(2u, stats.f32_constant_hist.at(-2.f));
  EXPECT_EQ(1u, stats.f64_constant_hist.at(-2));

  EXPECT_EQ(3u, stats.u16_constant_hist.size());
  EXPECT_EQ(1u, stats.s16_constant_hist.size());
  EXPECT_EQ(1u, stats.u32_constant_hist.size());
  EXPECT_EQ(2u, stats.s32_constant_hist.size());
  EXPECT_EQ(1u, stats.u64_constant_hist.size());
  EXPECT_EQ(1u, stats.s64_constant_hist.size());
  EXPECT_EQ(2u, stats.u16_constant_hist.at(16));
  EXPECT_EQ(2u, stats.u16_constant_hist.at(2));
  EXPECT_EQ(1u, stats.u16_constant_hist.at(1));
  EXPECT_EQ(1u, stats.s16_constant_hist.at(-16));
  EXPECT_EQ(2u, stats.u32_constant_hist.at(32));
  EXPECT_EQ(1u, stats.s32_constant_hist.at(2));
  EXPECT_EQ(1u, stats.s32_constant_hist.at(-32));
  EXPECT_EQ(2u, stats.u64_constant_hist.at(64));
  EXPECT_EQ(1u, stats.s64_constant_hist.at(-64));
}

}  // namespace
}  // namespace stats
}  // namespace spvtools
