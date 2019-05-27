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

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "spirv-tools/optimizer.hpp"
#include "spirv/1.1/spirv.h"

namespace spvtools {
namespace {

using ::testing::ContainerEq;
using ::testing::HasSubstr;

// Return a string that contains the minimum instructions needed to form
// a valid module.  Other instructions can be appended to this string.
std::string Header() {
  return R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";
}

// When we assemble with a target environment of SPIR-V 1.1, we expect
// the following in the module header version word.
const uint32_t kExpectedSpvVersion = 0x10100;

TEST(CppInterface, SuccessfulRoundTrip) {
  const std::string input_text = "%2 = OpSizeOf %1 %3\n";
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);

  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(input_text, &binary));
  EXPECT_TRUE(binary.size() > 5u);
  EXPECT_EQ(SpvMagicNumber, binary[0]);
  EXPECT_EQ(kExpectedSpvVersion, binary[1]);

  // This cannot pass validation since %1 is not defined.
  t.SetMessageConsumer([](spv_message_level_t level, const char* source,
                          const spv_position_t& position, const char* message) {
    EXPECT_EQ(SPV_MSG_ERROR, level);
    EXPECT_STREQ("input", source);
    EXPECT_EQ(0u, position.line);
    EXPECT_EQ(0u, position.column);
    EXPECT_EQ(1u, position.index);
    EXPECT_STREQ("ID 1[%1] has not been defined\n  %2 = OpSizeOf %1 %3\n",
                 message);
  });
  EXPECT_FALSE(t.Validate(binary));

  std::string output_text;
  EXPECT_TRUE(t.Disassemble(binary, &output_text));
  EXPECT_EQ(input_text, output_text);
}

TEST(CppInterface, AssembleEmptyModule) {
  std::vector<uint32_t> binary(10, 42);
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  EXPECT_TRUE(t.Assemble("", &binary));
  // We only have the header.
  EXPECT_EQ(5u, binary.size());
  EXPECT_EQ(SpvMagicNumber, binary[0]);
  EXPECT_EQ(kExpectedSpvVersion, binary[1]);
}

TEST(CppInterface, AssembleOverloads) {
  const std::string input_text = "%2 = OpSizeOf %1 %3\n";
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  {
    std::vector<uint32_t> binary;
    EXPECT_TRUE(t.Assemble(input_text, &binary));
    EXPECT_TRUE(binary.size() > 5u);
    EXPECT_EQ(SpvMagicNumber, binary[0]);
    EXPECT_EQ(kExpectedSpvVersion, binary[1]);
  }
  {
    std::vector<uint32_t> binary;
    EXPECT_TRUE(t.Assemble(input_text.data(), input_text.size(), &binary));
    EXPECT_TRUE(binary.size() > 5u);
    EXPECT_EQ(SpvMagicNumber, binary[0]);
    EXPECT_EQ(kExpectedSpvVersion, binary[1]);
  }
  {  // Ignore the last newline.
    std::vector<uint32_t> binary;
    EXPECT_TRUE(t.Assemble(input_text.data(), input_text.size() - 1, &binary));
    EXPECT_TRUE(binary.size() > 5u);
    EXPECT_EQ(SpvMagicNumber, binary[0]);
    EXPECT_EQ(kExpectedSpvVersion, binary[1]);
  }
}

TEST(CppInterface, DisassembleEmptyModule) {
  std::string text(10, 'x');
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  int invocation_count = 0;
  t.SetMessageConsumer(
      [&invocation_count](spv_message_level_t level, const char* source,
                          const spv_position_t& position, const char* message) {
        ++invocation_count;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        EXPECT_STREQ("input", source);
        EXPECT_EQ(0u, position.line);
        EXPECT_EQ(0u, position.column);
        EXPECT_EQ(0u, position.index);
        EXPECT_STREQ("Missing module.", message);
      });
  EXPECT_FALSE(t.Disassemble({}, &text));
  EXPECT_EQ("xxxxxxxxxx", text);  // The original string is unmodified.
  EXPECT_EQ(1, invocation_count);
}

TEST(CppInterface, DisassembleOverloads) {
  const std::string input_text = "%2 = OpSizeOf %1 %3\n";
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);

  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(input_text, &binary));

  {
    std::string output_text;
    EXPECT_TRUE(t.Disassemble(binary, &output_text));
    EXPECT_EQ(input_text, output_text);
  }
  {
    std::string output_text;
    EXPECT_TRUE(t.Disassemble(binary.data(), binary.size(), &output_text));
    EXPECT_EQ(input_text, output_text);
  }
}

TEST(CppInterface, SuccessfulValidation) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  int invocation_count = 0;
  t.SetMessageConsumer([&invocation_count](spv_message_level_t, const char*,
                                           const spv_position_t&, const char*) {
    ++invocation_count;
  });

  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(Header(), &binary));
  EXPECT_TRUE(t.Validate(binary));
  EXPECT_EQ(0, invocation_count);
}

TEST(CppInterface, ValidateOverloads) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(Header(), &binary));

  { EXPECT_TRUE(t.Validate(binary)); }
  { EXPECT_TRUE(t.Validate(binary.data(), binary.size())); }
}

TEST(CppInterface, ValidateEmptyModule) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  int invocation_count = 0;
  t.SetMessageConsumer(
      [&invocation_count](spv_message_level_t level, const char* source,
                          const spv_position_t& position, const char* message) {
        ++invocation_count;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        EXPECT_STREQ("input", source);
        EXPECT_EQ(0u, position.line);
        EXPECT_EQ(0u, position.column);
        EXPECT_EQ(0u, position.index);
        EXPECT_STREQ("Invalid SPIR-V magic number.", message);
      });
  EXPECT_FALSE(t.Validate({}));
  EXPECT_EQ(1, invocation_count);
}

// Returns the assembly for a SPIR-V module with a struct declaration
// with the given number of members.
std::string MakeModuleHavingStruct(int num_members) {
  std::stringstream os;
  os << Header();
  os << R"(%1 = OpTypeInt 32 0
           %2 = OpTypeStruct)";
  for (int i = 0; i < num_members; i++) os << " %1";
  return os.str();
}

TEST(CppInterface, ValidateWithOptionsPass) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(MakeModuleHavingStruct(10), &binary));
  const ValidatorOptions opts;

  EXPECT_TRUE(t.Validate(binary.data(), binary.size(), opts));
}

TEST(CppInterface, ValidateWithOptionsFail) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble(MakeModuleHavingStruct(10), &binary));
  ValidatorOptions opts;
  opts.SetUniversalLimit(spv_validator_limit_max_struct_members, 9);
  std::stringstream os;
  t.SetMessageConsumer([&os](spv_message_level_t, const char*,
                             const spv_position_t&,
                             const char* message) { os << message; });

  EXPECT_FALSE(t.Validate(binary.data(), binary.size(), opts));
  EXPECT_THAT(
      os.str(),
      HasSubstr(
          "Number of OpTypeStruct members (10) has exceeded the limit (9)"));
}

// Checks that after running the given optimizer |opt| on the given |original|
// source code, we can get the given |optimized| source code.
void CheckOptimization(const std::string& original,
                       const std::string& optimized, const Optimizer& opt) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(t.Assemble(original, &original_binary));

  std::vector<uint32_t> optimized_binary;
  EXPECT_TRUE(opt.Run(original_binary.data(), original_binary.size(),
                      &optimized_binary));

  std::string optimized_text;
  EXPECT_TRUE(t.Disassemble(optimized_binary, &optimized_text));
  EXPECT_EQ(optimized, optimized_text);
}

TEST(CppInterface, OptimizeEmptyModule) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> binary;
  EXPECT_TRUE(t.Assemble("", &binary));

  Optimizer o(SPV_ENV_UNIVERSAL_1_1);
  o.RegisterPass(CreateStripDebugInfoPass());

  // Fails to validate.
  EXPECT_FALSE(o.Run(binary.data(), binary.size(), &binary));
}

TEST(CppInterface, OptimizeModifiedModule) {
  Optimizer o(SPV_ENV_UNIVERSAL_1_1);
  o.RegisterPass(CreateStripDebugInfoPass());
  CheckOptimization(Header() + "OpSource GLSL 450", Header(), o);
}

TEST(CppInterface, OptimizeMulitplePasses) {
  std::string original_text = Header() +
                              "OpSource GLSL 450 "
                              "OpDecorate %true SpecId 1 "
                              "%bool = OpTypeBool "
                              "%true = OpSpecConstantTrue %bool";

  Optimizer o(SPV_ENV_UNIVERSAL_1_1);
  o.RegisterPass(CreateStripDebugInfoPass())
      .RegisterPass(CreateFreezeSpecConstantValuePass());

  std::string expected_text = Header() +
                              "%bool = OpTypeBool\n"
                              "%true = OpConstantTrue %bool\n";

  CheckOptimization(original_text, expected_text, o);
}

TEST(CppInterface, OptimizeDoNothingWithPassToken) {
  CreateFreezeSpecConstantValuePass();
  auto token = CreateUnifyConstantPass();
}

TEST(CppInterface, OptimizeReassignPassToken) {
  auto token = CreateNullPass();
  token = CreateStripDebugInfoPass();

  CheckOptimization(
      Header() + "OpSource GLSL 450", Header(),
      Optimizer(SPV_ENV_UNIVERSAL_1_1).RegisterPass(std::move(token)));
}

TEST(CppInterface, OptimizeMoveConstructPassToken) {
  auto token1 = CreateStripDebugInfoPass();
  Optimizer::PassToken token2(std::move(token1));

  CheckOptimization(
      Header() + "OpSource GLSL 450", Header(),
      Optimizer(SPV_ENV_UNIVERSAL_1_1).RegisterPass(std::move(token2)));
}

TEST(CppInterface, OptimizeMoveAssignPassToken) {
  auto token1 = CreateStripDebugInfoPass();
  auto token2 = CreateNullPass();
  token2 = std::move(token1);

  CheckOptimization(
      Header() + "OpSource GLSL 450", Header(),
      Optimizer(SPV_ENV_UNIVERSAL_1_1).RegisterPass(std::move(token2)));
}

TEST(CppInterface, OptimizeSameAddressForOriginalOptimizedBinary) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::vector<uint32_t> binary;
  ASSERT_TRUE(t.Assemble(Header() + "OpSource GLSL 450", &binary));

  EXPECT_TRUE(Optimizer(SPV_ENV_UNIVERSAL_1_1)
                  .RegisterPass(CreateStripDebugInfoPass())
                  .Run(binary.data(), binary.size(), &binary));

  std::string optimized_text;
  EXPECT_TRUE(t.Disassemble(binary, &optimized_text));
  EXPECT_EQ(Header(), optimized_text);
}

// TODO(antiagainst): tests for SetMessageConsumer().

}  // namespace
}  // namespace spvtools
