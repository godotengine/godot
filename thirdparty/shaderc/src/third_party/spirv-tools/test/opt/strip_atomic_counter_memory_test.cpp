// Copyright (c) 2019 Google Inc.
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

#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

typedef std::tuple<std::string, std::string> StripAtomicCounterMemoryParam;

using MemorySemanticsModified =
    PassTest<::testing::TestWithParam<StripAtomicCounterMemoryParam>>;
using NonMemorySemanticsUnmodifiedTest = PassTest<::testing::Test>;

void operator+=(std::vector<const char*>& lhs, const char* rhs) {
  lhs.push_back(rhs);
}

std::string GetConstDecl(std::string val) {
  std::string decl;
  decl += "%uint_" + val + " = OpConstant %uint " + val;
  return decl;
}

std::string GetUnchangedString(std::string(generate_inst)(std::string),
                               std::string val) {
  std::string decl = GetConstDecl(val);
  std::string inst = generate_inst(val);

  std::vector<const char*> result = {
      // clang-format off
              "OpCapability Shader",
              "OpCapability VulkanMemoryModelKHR",
              "OpExtension \"SPV_KHR_vulkan_memory_model\"",
              "OpMemoryModel Logical VulkanKHR",
              "OpEntryPoint Vertex %1 \"shader\"",
      "%uint = OpTypeInt 32 0",
"%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint",
         "%4 = OpVariable %_ptr_Workgroup_uint Workgroup",
    "%uint_0 = OpConstant %uint 0",
    "%uint_1 = OpConstant %uint 1",
      "%void = OpTypeVoid",
         "%8 = OpTypeFunction %void",
               decl.c_str(),
         "%1 = OpFunction %void None %8",
        "%10 = OpLabel",
               inst.c_str(),
              "OpReturn",
              "OpFunctionEnd"
      // clang-format on
  };
  return JoinAllInsts(result);
}

std::string GetChangedString(std::string(generate_inst)(std::string),
                             std::string orig, std::string changed) {
  std::string orig_decl = GetConstDecl(orig);
  std::string changed_decl = GetConstDecl(changed);
  std::string inst = generate_inst(changed);

  std::vector<const char*> result = {
      // clang-format off
              "OpCapability Shader",
              "OpCapability VulkanMemoryModelKHR",
              "OpExtension \"SPV_KHR_vulkan_memory_model\"",
              "OpMemoryModel Logical VulkanKHR",
              "OpEntryPoint Vertex %1 \"shader\"",
      "%uint = OpTypeInt 32 0",
"%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint",
         "%4 = OpVariable %_ptr_Workgroup_uint Workgroup",
    "%uint_0 = OpConstant %uint 0",
    "%uint_1 = OpConstant %uint 1",
      "%void = OpTypeVoid",
         "%8 = OpTypeFunction %void",
               orig_decl.c_str() };
  // clang-format on
  if (changed != "0") result += changed_decl.c_str();
  result += "%1 = OpFunction %void None %8";
  result += "%10 = OpLabel";
  result += inst.c_str();
  result += "OpReturn";
  result += "OpFunctionEnd";
  return JoinAllInsts(result);
}

std::tuple<std::string, std::string> GetInputAndExpected(
    std::string(generate_inst)(std::string),
    StripAtomicCounterMemoryParam param) {
  std::string orig = std::get<0>(param);
  std::string changed = std::get<1>(param);
  std::string input = GetUnchangedString(generate_inst, orig);
  std::string expected = orig == changed
                             ? GetUnchangedString(generate_inst, changed)
                             : GetChangedString(generate_inst, orig, changed);
  return std::make_tuple(input, expected);
}

std::string GetOpControlBarrierInst(std::string val) {
  return "OpControlBarrier %uint_1 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpControlBarrier) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpControlBarrierInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpMemoryBarrierInst(std::string val) {
  return "OpMemoryBarrier %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpMemoryBarrier) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpMemoryBarrierInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicLoadInst(std::string val) {
  return "%11 = OpAtomicLoad %uint %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpAtomicLoad) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicLoadInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicStoreInst(std::string val) {
  return "OpAtomicStore %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicStore) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicStoreInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicExchangeInst(std::string val) {
  return "%11 = OpAtomicExchange %uint %4 %uint_1 %uint_" + val + " %uint_0";
}

TEST_P(MemorySemanticsModified, OpAtomicExchange) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicExchangeInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicCompareExchangeInst(std::string val) {
  return "%11 = OpAtomicCompareExchange %uint %4 %uint_1 %uint_" + val +
         " %uint_" + val + " %uint_0 %uint_0";
}

TEST_P(MemorySemanticsModified, OpAtomicCompareExchange) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicCompareExchangeInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicCompareExchangeWeakInst(std::string val) {
  return "%11 = OpAtomicCompareExchangeWeak %uint %4 %uint_1 %uint_" + val +
         " %uint_" + val + " %uint_0 %uint_0";
}

TEST_P(MemorySemanticsModified, OpAtomicCompareExchangeWeak) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicCompareExchangeWeakInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicIIncrementInst(std::string val) {
  return "%11 = OpAtomicIIncrement %uint %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpAtomicIIncrement) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicIIncrementInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicIDecrementInst(std::string val) {
  return "%11 = OpAtomicIDecrement %uint %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpAtomicIDecrement) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicIDecrementInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicIAddInst(std::string val) {
  return "%11 = OpAtomicIAdd %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicIAdd) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicIAddInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicISubInst(std::string val) {
  return "%11 = OpAtomicISub %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicISub) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicISubInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicSMinInst(std::string val) {
  return "%11 = OpAtomicSMin %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicSMin) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicSMinInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicUMinInst(std::string val) {
  return "%11 = OpAtomicUMin %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicUMin) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicUMinInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicSMaxInst(std::string val) {
  return "%11 = OpAtomicSMax %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicSMax) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicSMaxInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicUMaxInst(std::string val) {
  return "%11 = OpAtomicUMax %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicUMax) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicUMaxInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicAndInst(std::string val) {
  return "%11 = OpAtomicAnd %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicAnd) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicAndInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicOrInst(std::string val) {
  return "%11 = OpAtomicOr %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicOr) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicOrInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicXorInst(std::string val) {
  return "%11 = OpAtomicXor %uint %4 %uint_1 %uint_" + val + " %uint_1";
}

TEST_P(MemorySemanticsModified, OpAtomicXor) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicXorInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicFlagTestAndSetInst(std::string val) {
  return "%11 = OpAtomicFlagTestAndSet %uint %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpAtomicFlagTestAndSet) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicFlagTestAndSetInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpAtomicFlagClearInst(std::string val) {
  return "OpAtomicFlagClear %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpAtomicFlagClear) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpAtomicFlagClearInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetOpMemoryNamedBarrierInst(std::string val) {
  return "OpMemoryNamedBarrier %4 %uint_1 %uint_" + val;
}

TEST_P(MemorySemanticsModified, OpMemoryNamedBarrier) {
  std::string input, expected;
  std::tie(input, expected) =
      GetInputAndExpected(GetOpMemoryNamedBarrierInst, GetParam());
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    StripAtomicCounterMemoryTest, MemorySemanticsModified,
    ::testing::ValuesIn(std::vector<StripAtomicCounterMemoryParam>({
       std::make_tuple("1024", "0"),
       std::make_tuple("5", "5"),
       std::make_tuple("1288", "264"),
       std::make_tuple("264", "264")
    })));
// clang-format on

std::string GetNoMemorySemanticsPresentInst(std::string val) {
  return "%11 = OpVariable %_ptr_Workgroup_uint Workgroup %uint_" + val;
}

TEST_F(NonMemorySemanticsUnmodifiedTest, NoMemorySemanticsPresent) {
  std::string input, expected;
  StripAtomicCounterMemoryParam param = std::make_tuple("1288", "1288");
  std::tie(input, expected) =
      GetInputAndExpected(GetNoMemorySemanticsPresentInst, param);
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

std::string GetMemorySemanticsPresentInst(std::string val) {
  return "%11 = OpAtomicIAdd %uint %4 %uint_1 %uint_" + val + " %uint_1288";
}

TEST_F(NonMemorySemanticsUnmodifiedTest, MemorySemanticsPresent) {
  std::string input, expected;
  StripAtomicCounterMemoryParam param = std::make_tuple("1288", "264");
  std::tie(input, expected) =
      GetInputAndExpected(GetMemorySemanticsPresentInst, param);
  SinglePassRunAndCheck<StripAtomicCounterMemoryPass>(input, expected,
                                                      /* skip_nop = */ false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
