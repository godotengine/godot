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

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using StrengthReductionBasicTest = PassTest<::testing::Test>;

// Test to make sure we replace 5*8.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy8) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%uint = OpTypeInt 32 0",
     "%uint_5 = OpConstant %uint 5",
     "%uint_8 = OpConstant %uint 8",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %uint %uint_5 %uint_8",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(HasSubstr("OpIMul")));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %uint %uint_5 %uint_3"));
}

// TODO(dneto): Add Effcee as required dependency, and make this unconditional.
// Test to make sure we replace 16*5
// Also demonstrate use of Effcee matching.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy16) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
; We know disassembly will produce %uint here, but
;  CHECK: %uint = OpTypeInt 32 0
;  CHECK-DAG: [[five:%[a-zA-Z_\d]+]] = OpConstant %uint 5

; We have RE2 regular expressions, so \w matches [_a-zA-Z0-9].
; This shows the preferred pattern for matching SPIR-V identifiers.
; (We could have cheated in this case since we know the disassembler will
; generate the 'nice' name of "%uint_4".
;  CHECK-DAG: [[four:%\w+]] = OpConstant %uint 4
       %uint = OpTypeInt 32 0
     %uint_5 = OpConstant %uint 5
    %uint_16 = OpConstant %uint 16
       %main = OpFunction %void None %4
; CHECK: OpLabel
          %8 = OpLabel
; CHECK-NEXT: OpShiftLeftLogical %uint [[five]] [[four]]
; The multiplication disappears.
; CHECK-NOT: OpIMul
          %9 = OpIMul %uint %uint_16 %uint_5
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd)";

  SinglePassRunAndMatch<StrengthReductionPass>(text, false);
}

// Test to make sure we replace a multiple of 32 and 4.
TEST_F(StrengthReductionBasicTest, BasicTwoPowersOf2) {
  // In this case, we have two powers of 2.  Need to make sure we replace only
  // one of them for the bit shift.
  // clang-format off
  const std::string text = R"(
          OpCapability Shader
     %1 = OpExtInstImport "GLSL.std.450"
          OpMemoryModel Logical GLSL450
          OpEntryPoint Vertex %main "main"
          OpName %main "main"
  %void = OpTypeVoid
     %4 = OpTypeFunction %void
   %int = OpTypeInt 32 1
%int_32 = OpConstant %int 32
 %int_4 = OpConstant %int 4
  %main = OpFunction %void None %4
     %8 = OpLabel
     %9 = OpIMul %int %int_32 %int_4
          OpReturn
          OpFunctionEnd
)";
  // clang-format on
  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(HasSubstr("OpIMul")));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %int %int_4 %uint_5"));
}

// Test to make sure we don't replace 0*5.
TEST_F(StrengthReductionBasicTest, BasicDontReplace0) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
      "%int_0 = OpConstant %int 0",
      "%int_5 = OpConstant %int 5",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %int %int_0 %int_5",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// Test to make sure we do not replace a multiple of 5 and 7.
TEST_F(StrengthReductionBasicTest, BasicNoChange) {
  const std::vector<const char*> text = {
      // clang-format off
             "OpCapability Shader",
        "%1 = OpExtInstImport \"GLSL.std.450\"",
             "OpMemoryModel Logical GLSL450",
             "OpEntryPoint Vertex %2 \"main\"",
             "OpName %2 \"main\"",
        "%3 = OpTypeVoid",
        "%4 = OpTypeFunction %3",
        "%5 = OpTypeInt 32 1",
        "%6 = OpTypeInt 32 0",
        "%7 = OpConstant %5 5",
        "%8 = OpConstant %5 7",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %5 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// Test to make sure constants and types are reused and not duplicated.
TEST_F(StrengthReductionBasicTest, NoDuplicateConstantsAndTypes) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%uint = OpTypeInt 32 0",
     "%uint_8 = OpConstant %uint 8",
     "%uint_3 = OpConstant %uint 3",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %uint %uint_8 %uint_3",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output,
              Not(MatchesRegex(".*OpConstant %uint 3.*OpConstant %uint 3.*")));
  EXPECT_THAT(output, Not(MatchesRegex(".*OpTypeInt 32 0.*OpTypeInt 32 0.*")));
}

// Test to make sure we generate the constants only once
TEST_F(StrengthReductionBasicTest, BasicCreateOneConst) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%uint = OpTypeInt 32 0",
     "%uint_5 = OpConstant %uint 5",
     "%uint_9 = OpConstant %uint 9",
   "%uint_128 = OpConstant %uint 128",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %uint %uint_5 %uint_128",
         "%10 = OpIMul %uint %uint_9 %uint_128",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(HasSubstr("OpIMul")));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %uint %uint_5 %uint_7"));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %uint %uint_9 %uint_7"));
}

// Test to make sure we generate the instructions in the correct position and
// that the uses get replaced as well.  Here we check that the use in the return
// is replaced, we also check that we can replace two OpIMuls when one feeds the
// other.
TEST_F(StrengthReductionBasicTest, BasicCheckPositionAndReplacement) {
  // This is just the preamble to set up the test.
  const std::vector<const char*> common_text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpName %main \"main\"",
               "OpName %foo_i1_ \"foo(i1;\"",
               "OpName %n \"n\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
               "OpName %param \"param\"",
               "OpDecorate %gl_FragColor Location 0",
       "%void = OpTypeVoid",
          "%3 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
"%_ptr_Function_int = OpTypePointer Function %int",
          "%8 = OpTypeFunction %int %_ptr_Function_int",
    "%int_256 = OpConstant %int 256",
      "%int_2 = OpConstant %int 2",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_1 = OpConstant %float 1",
     "%int_10 = OpConstant %int 10",
  "%float_0_375 = OpConstant %float 0.375",
  "%float_0_75 = OpConstant %float 0.75",
       "%uint = OpTypeInt 32 0",
     "%uint_8 = OpConstant %uint 8",
     "%uint_1 = OpConstant %uint 1",
       "%main = OpFunction %void None %3",
          "%5 = OpLabel",
      "%param = OpVariable %_ptr_Function_int Function",
               "OpStore %param %int_10",
         "%26 = OpFunctionCall %int %foo_i1_ %param",
         "%27 = OpConvertSToF %float %26",
         "%28 = OpFDiv %float %float_1 %27",
         "%31 = OpCompositeConstruct %v4float %28 %float_0_375 %float_0_75 %float_1",
               "OpStore %gl_FragColor %31",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  // This is the real test.  The two OpIMul should be replaced.  The expected
  // output is in |foo_after|.
  const std::vector<const char*> foo_before = {
      // clang-format off
    "%foo_i1_ = OpFunction %int None %8",
          "%n = OpFunctionParameter %_ptr_Function_int",
         "%11 = OpLabel",
         "%12 = OpLoad %int %n",
         "%14 = OpIMul %int %12 %int_256",
         "%16 = OpIMul %int %14 %int_2",
               "OpReturnValue %16",
               "OpFunctionEnd",

      // clang-format on
  };

  const std::vector<const char*> foo_after = {
      // clang-format off
    "%foo_i1_ = OpFunction %int None %8",
          "%n = OpFunctionParameter %_ptr_Function_int",
         "%11 = OpLabel",
         "%12 = OpLoad %int %n",
         "%33 = OpShiftLeftLogical %int %12 %uint_8",
         "%34 = OpShiftLeftLogical %int %33 %uint_1",
               "OpReturnValue %34",
               "OpFunctionEnd",
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<StrengthReductionPass>(
      JoinAllInsts(Concat(common_text, foo_before)),
      JoinAllInsts(Concat(common_text, foo_after)),
      /* skip_nop = */ true, /* do_validate = */ true);
}

// Test that, when the result of an OpIMul instruction has more than 1 use, and
// the instruction is replaced, all of the uses of the results are replace with
// the new result.
TEST_F(StrengthReductionBasicTest, BasicTestMultipleReplacements) {
  // This is just the preamble to set up the test.
  const std::vector<const char*> common_text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpName %main \"main\"",
               "OpName %foo_i1_ \"foo(i1;\"",
               "OpName %n \"n\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
               "OpName %param \"param\"",
               "OpDecorate %gl_FragColor Location 0",
       "%void = OpTypeVoid",
          "%3 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
"%_ptr_Function_int = OpTypePointer Function %int",
          "%8 = OpTypeFunction %int %_ptr_Function_int",
    "%int_256 = OpConstant %int 256",
      "%int_2 = OpConstant %int 2",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_1 = OpConstant %float 1",
     "%int_10 = OpConstant %int 10",
  "%float_0_375 = OpConstant %float 0.375",
  "%float_0_75 = OpConstant %float 0.75",
       "%uint = OpTypeInt 32 0",
     "%uint_8 = OpConstant %uint 8",
     "%uint_1 = OpConstant %uint 1",
       "%main = OpFunction %void None %3",
          "%5 = OpLabel",
      "%param = OpVariable %_ptr_Function_int Function",
               "OpStore %param %int_10",
         "%26 = OpFunctionCall %int %foo_i1_ %param",
         "%27 = OpConvertSToF %float %26",
         "%28 = OpFDiv %float %float_1 %27",
         "%31 = OpCompositeConstruct %v4float %28 %float_0_375 %float_0_75 %float_1",
               "OpStore %gl_FragColor %31",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  // This is the real test.  The two OpIMul instructions should be replaced.  In
  // particular, we want to be sure that both uses of %16 are changed to use the
  // new result.
  const std::vector<const char*> foo_before = {
      // clang-format off
    "%foo_i1_ = OpFunction %int None %8",
          "%n = OpFunctionParameter %_ptr_Function_int",
         "%11 = OpLabel",
         "%12 = OpLoad %int %n",
         "%14 = OpIMul %int %12 %int_256",
         "%16 = OpIMul %int %14 %int_2",
         "%17 = OpIAdd %int %14 %16",
               "OpReturnValue %17",
               "OpFunctionEnd",

      // clang-format on
  };

  const std::vector<const char*> foo_after = {
      // clang-format off
    "%foo_i1_ = OpFunction %int None %8",
          "%n = OpFunctionParameter %_ptr_Function_int",
         "%11 = OpLabel",
         "%12 = OpLoad %int %n",
         "%34 = OpShiftLeftLogical %int %12 %uint_8",
         "%35 = OpShiftLeftLogical %int %34 %uint_1",
         "%17 = OpIAdd %int %34 %35",
               "OpReturnValue %17",
               "OpFunctionEnd",
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<StrengthReductionPass>(
      JoinAllInsts(Concat(common_text, foo_before)),
      JoinAllInsts(Concat(common_text, foo_after)),
      /* skip_nop = */ true, /* do_validate = */ true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
