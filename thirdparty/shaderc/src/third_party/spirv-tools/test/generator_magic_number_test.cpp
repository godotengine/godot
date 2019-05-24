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

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/opcode.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::spvtest::EnumCase;
using ::testing::Eq;
using GeneratorMagicNumberTest =
    ::testing::TestWithParam<EnumCase<spv_generator_t>>;

TEST_P(GeneratorMagicNumberTest, Single) {
  EXPECT_THAT(std::string(spvGeneratorStr(GetParam().value())),
              GetParam().name());
}

INSTANTIATE_TEST_SUITE_P(
    Registered, GeneratorMagicNumberTest,
    ::testing::ValuesIn(std::vector<EnumCase<spv_generator_t>>{
        {SPV_GENERATOR_KHRONOS, "Khronos"},
        {SPV_GENERATOR_LUNARG, "LunarG"},
        {SPV_GENERATOR_VALVE, "Valve"},
        {SPV_GENERATOR_CODEPLAY, "Codeplay"},
        {SPV_GENERATOR_NVIDIA, "NVIDIA"},
        {SPV_GENERATOR_ARM, "ARM"},
        {SPV_GENERATOR_KHRONOS_LLVM_TRANSLATOR,
         "Khronos LLVM/SPIR-V Translator"},
        {SPV_GENERATOR_KHRONOS_ASSEMBLER, "Khronos SPIR-V Tools Assembler"},
        {SPV_GENERATOR_KHRONOS_GLSLANG, "Khronos Glslang Reference Front End"},
    }));

INSTANTIATE_TEST_SUITE_P(
    Unregistered, GeneratorMagicNumberTest,
    ::testing::ValuesIn(std::vector<EnumCase<spv_generator_t>>{
        // We read registered entries from the SPIR-V XML Registry file
        // which can change over time.
        {spv_generator_t(1000), "Unknown"},
        {spv_generator_t(9999), "Unknown"},
    }));

}  // namespace
}  // namespace spvtools
