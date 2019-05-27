// Copyright (c) 2018 Google LLC.
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
#include "source/opt/ir_builder.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_peeling.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

class PeelingPassTest : public PassTest<::testing::Test> {
 public:
  // Generic routine to run the loop peeling pass and check
  LoopPeelingPass::LoopPeelingStats AssembleAndRunPeelingTest(
      const std::string& text_head, const std::string& text_tail, SpvOp opcode,
      const std::string& res_id, const std::string& op1,
      const std::string& op2) {
    std::string opcode_str;
    switch (opcode) {
      case SpvOpSLessThan:
        opcode_str = "OpSLessThan";
        break;
      case SpvOpSGreaterThan:
        opcode_str = "OpSGreaterThan";
        break;
      case SpvOpSLessThanEqual:
        opcode_str = "OpSLessThanEqual";
        break;
      case SpvOpSGreaterThanEqual:
        opcode_str = "OpSGreaterThanEqual";
        break;
      case SpvOpIEqual:
        opcode_str = "OpIEqual";
        break;
      case SpvOpINotEqual:
        opcode_str = "OpINotEqual";
        break;
      default:
        assert(false && "Unhandled");
        break;
    }
    std::string test_cond =
        res_id + " = " + opcode_str + "  %bool " + op1 + " " + op2 + "\n";

    LoopPeelingPass::LoopPeelingStats stats;
    SinglePassRunAndDisassemble<LoopPeelingPass>(
        text_head + test_cond + text_tail, true, true, &stats);

    return stats;
  }

  // Generic routine to run the loop peeling pass and check
  LoopPeelingPass::LoopPeelingStats RunPeelingTest(
      const std::string& text_head, const std::string& text_tail, SpvOp opcode,
      const std::string& res_id, const std::string& op1, const std::string& op2,
      size_t nb_of_loops) {
    LoopPeelingPass::LoopPeelingStats stats = AssembleAndRunPeelingTest(
        text_head, text_tail, opcode, res_id, op1, op2);

    Function& f = *context()->module()->begin();
    LoopDescriptor& ld = *context()->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), nb_of_loops);

    return stats;
  }

  using PeelTraceType =
      std::vector<std::pair<LoopPeelingPass::PeelDirection, uint32_t>>;

  void BuildAndCheckTrace(const std::string& text_head,
                          const std::string& text_tail, SpvOp opcode,
                          const std::string& res_id, const std::string& op1,
                          const std::string& op2,
                          const PeelTraceType& expected_peel_trace,
                          size_t expected_nb_of_loops) {
    auto stats = RunPeelingTest(text_head, text_tail, opcode, res_id, op1, op2,
                                expected_nb_of_loops);

    EXPECT_EQ(stats.peeled_loops_.size(), expected_peel_trace.size());
    if (stats.peeled_loops_.size() != expected_peel_trace.size()) {
      return;
    }

    PeelTraceType::const_iterator expected_trace_it =
        expected_peel_trace.begin();
    decltype(stats.peeled_loops_)::const_iterator stats_it =
        stats.peeled_loops_.begin();

    while (expected_trace_it != expected_peel_trace.end()) {
      EXPECT_EQ(expected_trace_it->first, std::get<1>(*stats_it));
      EXPECT_EQ(expected_trace_it->second, std::get<2>(*stats_it));
      ++expected_trace_it;
      ++stats_it;
    }
  }
};

/*
Test are derivation of the following generated test from the following GLSL +
--eliminate-local-multi-store

#version 330 core
void main() {
  int a = 0;
  for(int i = 1; i < 10; i += 2) {
    if (i < 3) {
      a += 2;
    }
  }
}

The condition is interchanged to test < > <= >= == and peel before/after
opportunities.
*/
TEST_F(PeelingPassTest, PeelingPassBasic) {
  const std::string text_head = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %a "a"
               OpName %i "i"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
       %bool = OpTypeBool
     %int_20 = OpConstant %int 20
     %int_19 = OpConstant %int 19
     %int_18 = OpConstant %int 18
     %int_17 = OpConstant %int 17
     %int_16 = OpConstant %int 16
     %int_15 = OpConstant %int 15
     %int_14 = OpConstant %int 14
     %int_13 = OpConstant %int 13
     %int_12 = OpConstant %int 12
     %int_11 = OpConstant %int 11
     %int_10 = OpConstant %int 10
      %int_9 = OpConstant %int 9
      %int_8 = OpConstant %int 8
      %int_7 = OpConstant %int 7
      %int_6 = OpConstant %int 6
      %int_5 = OpConstant %int 5
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_2 = OpConstant %int 2
      %int_1 = OpConstant %int 1
      %int_0 = OpConstant %int 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_int Function
          %i = OpVariable %_ptr_Function_int Function
               OpStore %a %int_0
               OpStore %i %int_0
               OpBranch %11
         %11 = OpLabel
         %31 = OpPhi %int %int_0 %5 %33 %14
         %32 = OpPhi %int %int_1 %5 %30 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %bool %32 %int_20
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
  )";
  const std::string text_tail = R"(
               OpSelectionMerge %24 None
               OpBranchConditional %22 %23 %24
         %23 = OpLabel
         %27 = OpIAdd %int %31 %int_2
               OpStore %a %27
               OpBranch %24
         %24 = OpLabel
         %33 = OpPhi %int %31 %12 %27 %23
               OpBranch %14
         %14 = OpLabel
         %30 = OpIAdd %int %32 %int_2
               OpStore %i %30
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  auto run_test = [&text_head, &text_tail, this](SpvOp opcode,
                                                 const std::string& op1,
                                                 const std::string& op2) {
    auto stats =
        RunPeelingTest(text_head, text_tail, opcode, "%22", op1, op2, 2);

    EXPECT_EQ(stats.peeled_loops_.size(), 1u);
    if (stats.peeled_loops_.size() != 1u)
      return std::pair<LoopPeelingPass::PeelDirection, uint32_t>{
          LoopPeelingPass::PeelDirection::kNone, 0};

    return std::pair<LoopPeelingPass::PeelDirection, uint32_t>{
        std::get<1>(*stats.peeled_loops_.begin()),
        std::get<2>(*stats.peeled_loops_.begin())};
  };

  // Test LT
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv < 4");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%32", "%int_4");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 4 > iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%int_4", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before iv < 5");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%32", "%int_5");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 5 > iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%int_5", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv < 16");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%32", "%int_16");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 16 > iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%int_16", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after iv < 17");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%32", "%int_17");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 17 > iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%int_17", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Test GT
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv > 5");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%32", "%int_5");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 5 < iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%int_5", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before iv > 4");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%32", "%int_4");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 4 < iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%int_4", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv > 16");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%32", "%int_16");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 16 < iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%int_16", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after iv > 17");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThan, "%32", "%int_17");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 17 < iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThan, "%int_17", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Test LE
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv <= 4");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%32", "%int_4");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 4 => iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%int_4", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before iv <= 3");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%32", "%int_3");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 3 => iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%int_3", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv <= 16");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%32", "%int_16");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 16 => iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%int_16", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after iv <= 15");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%32", "%int_15");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 15 => iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%int_15", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Test GE
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv >= 5");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%32", "%int_5");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 35 >= iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%int_5", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before iv >= 4");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%32", "%int_4");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel before 4 <= iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%int_4", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv >= 17");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%32", "%int_17");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 17 <= iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%int_17", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after iv >= 16");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSGreaterThanEqual, "%32", "%int_16");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }
  {
    SCOPED_TRACE("Peel after 16 <= iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpSLessThanEqual, "%int_16", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 2u);
  }

  // Test EQ
  // Peel before by a factor of 1.
  {
    SCOPED_TRACE("Peel before iv == 1");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpIEqual, "%32", "%int_1");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 1u);
  }
  {
    SCOPED_TRACE("Peel before 1 == iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpIEqual, "%int_1", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 1u);
  }

  // Peel after by a factor of 1.
  {
    SCOPED_TRACE("Peel after iv == 19");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpIEqual, "%32", "%int_19");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 1u);
  }
  {
    SCOPED_TRACE("Peel after 19 == iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpIEqual, "%int_19", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 1u);
  }

  // Test NE
  // Peel before by a factor of 1.
  {
    SCOPED_TRACE("Peel before iv != 1");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpINotEqual, "%32", "%int_1");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 1u);
  }
  {
    SCOPED_TRACE("Peel before 1 != iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpINotEqual, "%int_1", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kBefore);
    EXPECT_EQ(peel_info.second, 1u);
  }

  // Peel after by a factor of 1.
  {
    SCOPED_TRACE("Peel after iv != 19");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpINotEqual, "%32", "%int_19");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 1u);
  }
  {
    SCOPED_TRACE("Peel after 19 != iv");

    std::pair<LoopPeelingPass::PeelDirection, uint32_t> peel_info =
        run_test(SpvOpINotEqual, "%int_19", "%32");
    EXPECT_EQ(peel_info.first, LoopPeelingPass::PeelDirection::kAfter);
    EXPECT_EQ(peel_info.second, 1u);
  }

  // No peel.
  {
    SCOPED_TRACE("No Peel: 20 => iv");

    auto stats = RunPeelingTest(text_head, text_tail, SpvOpSLessThanEqual,
                                "%22", "%int_20", "%32", 1);

    EXPECT_EQ(stats.peeled_loops_.size(), 0u);
  }
}

/*
Test are derivation of the following generated test from the following GLSL +
--eliminate-local-multi-store

#version 330 core
void main() {
  int a = 0;
  for(int i = 0; i < 10; ++i) {
    if (i < 3) {
      a += 2;
    }
    if (i < 1) {
      a += 2;
    }
  }
}

The condition is interchanged to test < > <= >= == and peel before/after
opportunities.
*/
TEST_F(PeelingPassTest, MultiplePeelingPass) {
  const std::string text_head = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %a "a"
               OpName %i "i"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
       %bool = OpTypeBool
     %int_10 = OpConstant %int 10
      %int_9 = OpConstant %int 9
      %int_8 = OpConstant %int 8
      %int_7 = OpConstant %int 7
      %int_6 = OpConstant %int 6
      %int_5 = OpConstant %int 5
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_2 = OpConstant %int 2
      %int_1 = OpConstant %int 1
      %int_0 = OpConstant %int 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_int Function
          %i = OpVariable %_ptr_Function_int Function
               OpStore %a %int_0
               OpStore %i %int_0
               OpBranch %11
         %11 = OpLabel
         %37 = OpPhi %int %int_0 %5 %40 %14
         %38 = OpPhi %int %int_0 %5 %36 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %bool %38 %int_10
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
  )";
  const std::string text_tail = R"(
               OpSelectionMerge %24 None
               OpBranchConditional %22 %23 %24
         %23 = OpLabel
         %27 = OpIAdd %int %37 %int_2
               OpStore %a %27
               OpBranch %24
         %24 = OpLabel
         %39 = OpPhi %int %37 %12 %27 %23
         %30 = OpSLessThan %bool %38 %int_1
               OpSelectionMerge %32 None
               OpBranchConditional %30 %31 %32
         %31 = OpLabel
         %34 = OpIAdd %int %39 %int_2
               OpStore %a %34
               OpBranch %32
         %32 = OpLabel
         %40 = OpPhi %int %39 %24 %34 %31
               OpBranch %14
         %14 = OpLabel
         %36 = OpIAdd %int %38 %int_1
               OpStore %i %36
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  auto run_test = [&text_head, &text_tail, this](
                      SpvOp opcode, const std::string& op1,
                      const std::string& op2,
                      const PeelTraceType& expected_peel_trace) {
    BuildAndCheckTrace(text_head, text_tail, opcode, "%22", op1, op2,
                       expected_peel_trace, expected_peel_trace.size() + 1);
  };

  // Test LT
  // Peel before by a factor of 3.
  {
    SCOPED_TRACE("Peel before iv < 3");

    run_test(SpvOpSLessThan, "%38", "%int_3",
             {{LoopPeelingPass::PeelDirection::kBefore, 3u}});
  }
  {
    SCOPED_TRACE("Peel before 3 > iv");

    run_test(SpvOpSGreaterThan, "%int_3", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 3u}});
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv < 8");

    run_test(SpvOpSLessThan, "%38", "%int_8",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }
  {
    SCOPED_TRACE("Peel after 8 > iv");

    run_test(SpvOpSGreaterThan, "%int_8", "%38",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }

  // Test GT
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv > 2");

    run_test(SpvOpSGreaterThan, "%38", "%int_2",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }
  {
    SCOPED_TRACE("Peel before 2 < iv");

    run_test(SpvOpSLessThan, "%int_2", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }

  // Peel after by a factor of 3.
  {
    SCOPED_TRACE("Peel after iv > 7");

    run_test(SpvOpSGreaterThan, "%38", "%int_7",
             {{LoopPeelingPass::PeelDirection::kAfter, 3u}});
  }
  {
    SCOPED_TRACE("Peel after 7 < iv");

    run_test(SpvOpSLessThan, "%int_7", "%38",
             {{LoopPeelingPass::PeelDirection::kAfter, 3u}});
  }

  // Test LE
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv <= 1");

    run_test(SpvOpSLessThanEqual, "%38", "%int_1",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }
  {
    SCOPED_TRACE("Peel before 1 => iv");

    run_test(SpvOpSGreaterThanEqual, "%int_1", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv <= 7");

    run_test(SpvOpSLessThanEqual, "%38", "%int_7",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }
  {
    SCOPED_TRACE("Peel after 7 => iv");

    run_test(SpvOpSGreaterThanEqual, "%int_7", "%38",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }

  // Test GE
  // Peel before by a factor of 2.
  {
    SCOPED_TRACE("Peel before iv >= 2");

    run_test(SpvOpSGreaterThanEqual, "%38", "%int_2",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }
  {
    SCOPED_TRACE("Peel before 2 <= iv");

    run_test(SpvOpSLessThanEqual, "%int_2", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 2u}});
  }

  // Peel after by a factor of 2.
  {
    SCOPED_TRACE("Peel after iv >= 8");

    run_test(SpvOpSGreaterThanEqual, "%38", "%int_8",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }
  {
    SCOPED_TRACE("Peel after 8 <= iv");

    run_test(SpvOpSLessThanEqual, "%int_8", "%38",
             {{LoopPeelingPass::PeelDirection::kAfter, 2u}});
  }
  // Test EQ
  // Peel before by a factor of 1.
  {
    SCOPED_TRACE("Peel before iv == 0");

    run_test(SpvOpIEqual, "%38", "%int_0",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }
  {
    SCOPED_TRACE("Peel before 0 == iv");

    run_test(SpvOpIEqual, "%int_0", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }

  // Peel after by a factor of 1.
  {
    SCOPED_TRACE("Peel after iv == 9");

    run_test(SpvOpIEqual, "%38", "%int_9",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }
  {
    SCOPED_TRACE("Peel after 9 == iv");

    run_test(SpvOpIEqual, "%int_9", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }

  // Test NE
  // Peel before by a factor of 1.
  {
    SCOPED_TRACE("Peel before iv != 0");

    run_test(SpvOpINotEqual, "%38", "%int_0",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }
  {
    SCOPED_TRACE("Peel before 0 != iv");

    run_test(SpvOpINotEqual, "%int_0", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }

  // Peel after by a factor of 1.
  {
    SCOPED_TRACE("Peel after iv != 9");

    run_test(SpvOpINotEqual, "%38", "%int_9",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }
  {
    SCOPED_TRACE("Peel after 9 != iv");

    run_test(SpvOpINotEqual, "%int_9", "%38",
             {{LoopPeelingPass::PeelDirection::kBefore, 1u}});
  }
}

/*
Test are derivation of the following generated test from the following GLSL +
--eliminate-local-multi-store

#version 330 core
void main() {
  int a = 0;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (i < 3) {
        a += 2;
      }
    }
  }
}
*/
TEST_F(PeelingPassTest, PeelingNestedPass) {
  const std::string text_head = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %a "a"
               OpName %i "i"
               OpName %j "j"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_7 = OpConstant %int 7
      %int_3 = OpConstant %int 3
      %int_2 = OpConstant %int 2
      %int_1 = OpConstant %int 1
         %43 = OpUndef %int
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_int Function
          %i = OpVariable %_ptr_Function_int Function
          %j = OpVariable %_ptr_Function_int Function
               OpStore %a %int_0
               OpStore %i %int_0
               OpBranch %11
         %11 = OpLabel
         %41 = OpPhi %int %int_0 %5 %45 %14
         %42 = OpPhi %int %int_0 %5 %40 %14
         %44 = OpPhi %int %43 %5 %46 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %bool %42 %int_10
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
               OpStore %j %int_0
               OpBranch %21
         %21 = OpLabel
         %45 = OpPhi %int %41 %12 %47 %24
         %46 = OpPhi %int %int_0 %12 %38 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %27 = OpSLessThan %bool %46 %int_10
               OpBranchConditional %27 %22 %23
         %22 = OpLabel
  )";

  const std::string text_tail = R"(
               OpSelectionMerge %32 None
               OpBranchConditional %30 %31 %32
         %31 = OpLabel
         %35 = OpIAdd %int %45 %int_2
               OpStore %a %35
               OpBranch %32
         %32 = OpLabel
         %47 = OpPhi %int %45 %22 %35 %31
               OpBranch %24
         %24 = OpLabel
         %38 = OpIAdd %int %46 %int_1
               OpStore %j %38
               OpBranch %21
         %23 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %40 = OpIAdd %int %42 %int_1
               OpStore %i %40
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  auto run_test =
      [&text_head, &text_tail, this](
          SpvOp opcode, const std::string& op1, const std::string& op2,
          const PeelTraceType& expected_peel_trace, size_t nb_of_loops) {
        BuildAndCheckTrace(text_head, text_tail, opcode, "%30", op1, op2,
                           expected_peel_trace, nb_of_loops);
      };

  // Peeling outer before by a factor of 3.
  {
    SCOPED_TRACE("Peel before iv_i < 3");

    // Expect peel before by a factor of 3 and 4 loops at the end.
    run_test(SpvOpSLessThan, "%42", "%int_3",
             {{LoopPeelingPass::PeelDirection::kBefore, 3u}}, 4);
  }
  // Peeling outer loop after by a factor of 3.
  {
    SCOPED_TRACE("Peel after iv_i < 7");

    // Expect peel after by a factor of 3 and 4 loops at the end.
    run_test(SpvOpSLessThan, "%42", "%int_7",
             {{LoopPeelingPass::PeelDirection::kAfter, 3u}}, 4);
  }

  // Peeling inner loop before by a factor of 3.
  {
    SCOPED_TRACE("Peel before iv_j < 3");

    // Expect peel before by a factor of 3 and 3 loops at the end.
    run_test(SpvOpSLessThan, "%46", "%int_3",
             {{LoopPeelingPass::PeelDirection::kBefore, 3u}}, 3);
  }
  // Peeling inner loop after by a factor of 3.
  {
    SCOPED_TRACE("Peel after iv_j < 7");

    // Expect peel after by a factor of 3 and 3 loops at the end.
    run_test(SpvOpSLessThan, "%46", "%int_7",
             {{LoopPeelingPass::PeelDirection::kAfter, 3u}}, 3);
  }

  // Not unworkable condition.
  {
    SCOPED_TRACE("No peel");

    // Expect no peeling and 2 loops at the end.
    run_test(SpvOpSLessThan, "%46", "%42", {}, 2);
  }

  // Could do a peeling of 3, but the goes over the threshold.
  {
    SCOPED_TRACE("Over threshold");

    size_t current_threshold = LoopPeelingPass::GetLoopPeelingThreshold();
    LoopPeelingPass::SetLoopPeelingThreshold(1u);
    // Expect no peeling and 2 loops at the end.
    run_test(SpvOpSLessThan, "%46", "%int_7", {}, 2);
    LoopPeelingPass::SetLoopPeelingThreshold(current_threshold);
  }
}
/*
Test are derivation of the following generated test from the following GLSL +
--eliminate-local-multi-store

#version 330 core
void main() {
  int a = 0;
  for (int i = 0, j = 0; i < 10; j++, i++) {
    if (i < j) {
      a += 2;
    }
  }
}
*/
TEST_F(PeelingPassTest, PeelingNoChanges) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %a "a"
               OpName %i "i"
               OpName %j "j"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_2 = OpConstant %int 2
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_int Function
          %i = OpVariable %_ptr_Function_int Function
          %j = OpVariable %_ptr_Function_int Function
               OpStore %a %int_0
               OpStore %i %int_0
               OpStore %j %int_0
               OpBranch %12
         %12 = OpLabel
         %34 = OpPhi %int %int_0 %5 %37 %15
         %35 = OpPhi %int %int_0 %5 %33 %15
         %36 = OpPhi %int %int_0 %5 %31 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %bool %35 %int_10
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %23 = OpSLessThan %bool %35 %36
               OpSelectionMerge %25 None
               OpBranchConditional %23 %24 %25
         %24 = OpLabel
         %28 = OpIAdd %int %34 %int_2
               OpStore %a %28
               OpBranch %25
         %25 = OpLabel
         %37 = OpPhi %int %34 %13 %28 %24
               OpBranch %15
         %15 = OpLabel
         %31 = OpIAdd %int %36 %int_1
               OpStore %j %31
         %33 = OpIAdd %int %35 %int_1
               OpStore %i %33
               OpBranch %12
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  {
    auto result =
        SinglePassRunAndDisassemble<LoopPeelingPass>(text, true, false);

    EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
