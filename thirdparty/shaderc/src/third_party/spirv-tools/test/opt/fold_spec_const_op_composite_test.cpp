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

#include <sstream>
#include <string>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using FoldSpecConstantOpAndCompositePassBasicTest = PassTest<::testing::Test>;

TEST_F(FoldSpecConstantOpAndCompositePassBasicTest, Empty) {
  SinglePassRunAndCheck<FoldSpecConstantOpAndCompositePass>(
      "", "", /* skip_nop = */ true);
}

// A test of the basic functionality of FoldSpecConstantOpAndCompositePass.
// A spec constant defined with an integer addition operation should be folded
// to a normal constant with fixed value.
TEST_F(FoldSpecConstantOpAndCompositePassBasicTest, Basic) {
  AssemblyBuilder builder;
  builder.AppendTypesConstantsGlobals({
      // clang-format off
        "%int = OpTypeInt 32 1",
        "%frozen_spec_const_int = OpConstant %int 1",
        "%const_int = OpConstant %int 2",
        // Folding target:
        "%spec_add = OpSpecConstantOp %int IAdd %frozen_spec_const_int %const_int",
      // clang-format on
  });

  std::vector<const char*> expected = {
      // clang-format off
                    "OpCapability Shader",
                    "OpCapability Float64",
               "%1 = OpExtInstImport \"GLSL.std.450\"",
                    "OpMemoryModel Logical GLSL450",
                    "OpEntryPoint Vertex %main \"main\"",
                    "OpName %void \"void\"",
                    "OpName %main_func_type \"main_func_type\"",
                    "OpName %main \"main\"",
                    "OpName %main_func_entry_block \"main_func_entry_block\"",
                    "OpName %int \"int\"",
                    "OpName %frozen_spec_const_int \"frozen_spec_const_int\"",
                    "OpName %const_int \"const_int\"",
                    "OpName %spec_add \"spec_add\"",
            "%void = OpTypeVoid",
  "%main_func_type = OpTypeFunction %void",
             "%int = OpTypeInt 32 1",
"%frozen_spec_const_int = OpConstant %int 1",
       "%const_int = OpConstant %int 2",
        // The SpecConstantOp IAdd instruction should be replace by OpConstant
        // instruction:
        "%spec_add = OpConstant %int 3",
            "%main = OpFunction %void None %main_func_type",
"%main_func_entry_block = OpLabel",
                    "OpReturn",
                    "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<FoldSpecConstantOpAndCompositePass>(
      builder.GetCode(), JoinAllInsts(expected), /* skip_nop = */ true);
}

// A test of skipping folding an instruction when the instruction result type
// has decorations.
TEST_F(FoldSpecConstantOpAndCompositePassBasicTest,
       SkipWhenTypeHasDecorations) {
  AssemblyBuilder builder;
  builder
      .AppendAnnotations({
          // clang-format off
          "OpDecorate %int RelaxedPrecision",
          // clang-format on
      })
      .AppendTypesConstantsGlobals({
          // clang-format off
          "%int = OpTypeInt 32 1",
          "%frozen_spec_const_int = OpConstant %int 1",
          "%const_int = OpConstant %int 2",
          // The following spec constant should not be folded as the result type
          // has relaxed precision decoration.
          "%spec_add = OpSpecConstantOp %int IAdd %frozen_spec_const_int %const_int",
          // clang-format on
      });

  SinglePassRunAndCheck<FoldSpecConstantOpAndCompositePass>(
      builder.GetCode(), builder.GetCode(), /* skip_nop = */ true);
}

// All types and some common constants that are potentially required in
// FoldSpecConstantOpAndCompositeTest.
std::vector<std::string> CommonTypesAndConstants() {
  return std::vector<std::string>{
      // clang-format off
      // scalar types
      "%bool = OpTypeBool",
      "%uint = OpTypeInt 32 0",
      "%int = OpTypeInt 32 1",
      "%float = OpTypeFloat 32",
      "%double = OpTypeFloat 64",
      // vector types
      "%v2bool = OpTypeVector %bool 2",
      "%v2uint = OpTypeVector %uint 2",
      "%v2int = OpTypeVector %int 2",
      "%v3int = OpTypeVector %int 3",
      "%v4int = OpTypeVector %int 4",
      "%v2float = OpTypeVector %float 2",
      "%v2double = OpTypeVector %double 2",
      // variable pointer types
      "%_pf_bool = OpTypePointer Function %bool",
      "%_pf_uint = OpTypePointer Function %uint",
      "%_pf_int = OpTypePointer Function %int",
      "%_pf_float = OpTypePointer Function %float",
      "%_pf_double = OpTypePointer Function %double",
      "%_pf_v2int = OpTypePointer Function %v2int",
      "%_pf_v2float = OpTypePointer Function %v2float",
      "%_pf_v2double = OpTypePointer Function %v2double",
      // struct types
      "%inner_struct = OpTypeStruct %bool %int %float",
      "%outer_struct = OpTypeStruct %inner_struct %int",
      "%flat_struct = OpTypeStruct %bool %int %float",

      // common constants
      // scalar constants:
      "%bool_true = OpConstantTrue %bool",
      "%bool_false = OpConstantFalse %bool",
      "%bool_null = OpConstantNull %bool",
      "%signed_zero = OpConstant %int 0",
      "%unsigned_zero = OpConstant %uint 0",
      "%signed_one = OpConstant %int 1",
      "%unsigned_one = OpConstant %uint 1",
      "%signed_two = OpConstant %int 2",
      "%unsigned_two = OpConstant %uint 2",
      "%signed_three = OpConstant %int 3",
      "%unsigned_three = OpConstant %uint 3",
      "%signed_null = OpConstantNull %int",
      "%unsigned_null = OpConstantNull %uint",
      // vector constants:
      "%bool_true_vec = OpConstantComposite %v2bool %bool_true %bool_true",
      "%bool_false_vec = OpConstantComposite %v2bool %bool_false %bool_false",
      "%bool_null_vec = OpConstantNull %v2bool",
      "%signed_zero_vec = OpConstantComposite %v2int %signed_zero %signed_zero",
      "%unsigned_zero_vec = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
      "%signed_one_vec = OpConstantComposite %v2int %signed_one %signed_one",
      "%unsigned_one_vec = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
      "%signed_two_vec = OpConstantComposite %v2int %signed_two %signed_two",
      "%unsigned_two_vec = OpConstantComposite %v2uint %unsigned_two %unsigned_two",
      "%signed_three_vec = OpConstantComposite %v2int %signed_three %signed_three",
      "%unsigned_three_vec = OpConstantComposite %v2uint %unsigned_three %unsigned_three",
      "%signed_null_vec = OpConstantNull %v2int",
      "%unsigned_null_vec = OpConstantNull %v2uint",
      "%v4int_0_1_2_3 = OpConstantComposite %v4int %signed_zero %signed_one %signed_two %signed_three",
      // clang-format on
  };
}

// A helper function to strip OpName instructions from the given string of
// disassembly code. Returns the string with all OpName instruction stripped.
std::string StripOpNameInstructions(const std::string& str) {
  std::stringstream ss(str);
  std::ostringstream oss;
  std::string inst_str;
  while (std::getline(ss, inst_str, '\n')) {
    if (inst_str.find("OpName %") == std::string::npos) {
      oss << inst_str << '\n';
    }
  }
  return oss.str();
}

struct FoldSpecConstantOpAndCompositePassTestCase {
  // Original constants with unfolded spec constants.
  std::vector<std::string> original;
  // Expected cosntants after folding.
  std::vector<std::string> expected;
};

using FoldSpecConstantOpAndCompositePassTest = PassTest<
    ::testing::TestWithParam<FoldSpecConstantOpAndCompositePassTestCase>>;

TEST_P(FoldSpecConstantOpAndCompositePassTest, ParamTestCase) {
  AssemblyBuilder test_code_builder, expected_code_builder;
  const auto& tc = GetParam();
  test_code_builder.AppendTypesConstantsGlobals(CommonTypesAndConstants());
  test_code_builder.AppendTypesConstantsGlobals(tc.original);
  expected_code_builder.AppendTypesConstantsGlobals(CommonTypesAndConstants());
  expected_code_builder.AppendTypesConstantsGlobals(tc.expected);
  const std::string original = test_code_builder.GetCode();
  const std::string expected = expected_code_builder.GetCode();

  // Run the optimization and get the result code in disassembly.
  std::string optimized;
  auto status = Pass::Status::SuccessWithoutChange;
  std::tie(optimized, status) =
      SinglePassRunAndDisassemble<FoldSpecConstantOpAndCompositePass>(
          original, /* skip_nop = */ true, /* do_validation = */ false);

  // Check the optimized code, but ignore the OpName instructions.
  EXPECT_NE(Pass::Status::Failure, status);
  EXPECT_EQ(
      StripOpNameInstructions(expected) == StripOpNameInstructions(original),
      status == Pass::Status::SuccessWithoutChange);
  EXPECT_EQ(StripOpNameInstructions(expected),
            StripOpNameInstructions(optimized));
}

// Tests that OpSpecConstantComposite opcodes are replace with
// OpConstantComposite correctly.
INSTANTIATE_TEST_SUITE_P(
    Composite, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // normal vector
            {
              // original
              {
                "%spec_v2bool = OpSpecConstantComposite %v2bool %bool_true %bool_false",
                "%spec_v2uint = OpSpecConstantComposite %v2uint %unsigned_one %unsigned_one",
                "%spec_v2int_a = OpSpecConstantComposite %v2int %signed_one %signed_two",
                // Spec constants whose value can not be fully resolved should
                // not be processed.
                "%spec_int = OpSpecConstant %int 99",
                "%spec_v2int_b = OpSpecConstantComposite %v2int %signed_one %spec_int",
              },
              // expected
              {
                "%spec_v2bool = OpConstantComposite %v2bool %bool_true %bool_false",
                "%spec_v2uint = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
                "%spec_v2int_a = OpConstantComposite %v2int %signed_one %signed_two",
                "%spec_int = OpSpecConstant %int 99",
                "%spec_v2int_b = OpSpecConstantComposite %v2int %signed_one %spec_int",
              },
            },
            // vector with null constants
            {
              // original
              {
                "%null_bool = OpConstantNull %bool",
                "%null_int = OpConstantNull %int",
                "%spec_v2bool = OpSpecConstantComposite %v2bool %null_bool %null_bool",
                "%spec_v3int = OpSpecConstantComposite %v3int %null_int %null_int %null_int",
                "%spec_v4int = OpSpecConstantComposite %v4int %null_int %null_int %null_int %null_int",
              },
              // expected
              {
                "%null_bool = OpConstantNull %bool",
                "%null_int = OpConstantNull %int",
                "%spec_v2bool = OpConstantComposite %v2bool %null_bool %null_bool",
                "%spec_v3int = OpConstantComposite %v3int %null_int %null_int %null_int",
                "%spec_v4int = OpConstantComposite %v4int %null_int %null_int %null_int %null_int",
              },
            },
            // flat struct
            {
              // original
              {
                "%float_1 = OpConstant %float 1",
                "%flat_1 = OpSpecConstantComposite %flat_struct %bool_true %signed_null %float_1",
                // following struct should not be folded as the value of
                // %spec_float is not determined.
                "%spec_float = OpSpecConstant %float 1",
                "%flat_2 = OpSpecConstantComposite %flat_struct %bool_true %signed_one %spec_float",
              },
              // expected
              {
                "%float_1 = OpConstant %float 1",
                "%flat_1 = OpConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%spec_float = OpSpecConstant %float 1",
                "%flat_2 = OpSpecConstantComposite %flat_struct %bool_true %signed_one %spec_float",
              }
            },
            // nested struct
            {
              // original
              {
                "%float_1 = OpConstant %float 1",
                "%inner_1 = OpSpecConstantComposite %inner_struct %bool_true %signed_null %float_1",
                "%outer_1 = OpSpecConstantComposite %outer_struct %inner_1 %signed_one",
                // following structs should not be folded as the value of
                // %spec_float is not determined.
                "%spec_float = OpSpecConstant %float 1",
                "%inner_2 = OpSpecConstantComposite %inner_struct %bool_true %signed_null %spec_float",
                "%outer_2 = OpSpecConstantComposite %outer_struct %inner_2 %signed_one",
              },
              // expected
              {
                "%float_1 = OpConstant %float 1",
                "%inner_1 = OpConstantComposite %inner_struct %bool_true %signed_null %float_1",
                "%outer_1 = OpConstantComposite %outer_struct %inner_1 %signed_one",
                "%spec_float = OpSpecConstant %float 1",
                "%inner_2 = OpSpecConstantComposite %inner_struct %bool_true %signed_null %spec_float",
                "%outer_2 = OpSpecConstantComposite %outer_struct %inner_2 %signed_one",
              }
            },
            // composite constants touched by OpUndef should be skipped
            {
              // original
              {
                "%undef = OpUndef %float",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_one %undef",
                "%outer = OpSpecConstantComposite %outer_struct %inner %signed_one",
              },
              // expected
              {
                "%undef = OpUndef %float",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_one %undef",
                "%outer = OpSpecConstantComposite %outer_struct %inner %signed_one",
              },
            }
        // clang-format on
    })));

// Tests for operations that resulting in different types.
INSTANTIATE_TEST_SUITE_P(
    Cast, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(
        std::vector<FoldSpecConstantOpAndCompositePassTestCase>({
            // clang-format off
            // int -> bool scalar
            {
              // original
              {
                "%spec_bool_t = OpSpecConstantOp %bool INotEqual %signed_three %signed_zero",
                "%spec_bool_f = OpSpecConstantOp %bool INotEqual %signed_zero %signed_zero",
                "%spec_bool_from_null = OpSpecConstantOp %bool INotEqual %signed_null %signed_zero",
              },
              // expected
              {
                "%spec_bool_t = OpConstantTrue %bool",
                "%spec_bool_f = OpConstantFalse %bool",
                "%spec_bool_from_null = OpConstantFalse %bool",
              },
            },

            // uint -> bool scalar
            {
              // original
              {
                "%spec_bool_t = OpSpecConstantOp %bool INotEqual %unsigned_three %unsigned_zero",
                "%spec_bool_f = OpSpecConstantOp %bool INotEqual %unsigned_zero %unsigned_zero",
                "%spec_bool_from_null = OpSpecConstantOp %bool INotEqual %unsigned_null %unsigned_zero",
              },
              // expected
              {
                "%spec_bool_t = OpConstantTrue %bool",
                "%spec_bool_f = OpConstantFalse %bool",
                "%spec_bool_from_null = OpConstantFalse %bool",
              },
            },

            // bool -> int scalar
            {
              // original
              {
                "%spec_int_one = OpSpecConstantOp %int Select %bool_true %signed_one %signed_zero",
                "%spec_int_zero = OpSpecConstantOp %int Select %bool_false %signed_one %signed_zero",
                "%spec_int_from_null = OpSpecConstantOp %int Select %bool_null %signed_one %signed_zero",
              },
              // expected
              {
                "%spec_int_one = OpConstant %int 1",
                "%spec_int_zero = OpConstant %int 0",
                "%spec_int_from_null = OpConstant %int 0",
              },
            },

            // uint -> int scalar
            {
              // original
              {
                "%spec_int_one = OpSpecConstantOp %int IAdd %unsigned_one %signed_zero",
                "%spec_int_zero = OpSpecConstantOp %int IAdd %unsigned_zero %signed_zero",
                "%spec_int_from_null = OpSpecConstantOp %int IAdd %unsigned_null %unsigned_zero",
              },
              // expected
              {
                "%spec_int_one = OpConstant %int 1",
                "%spec_int_zero = OpConstant %int 0",
                "%spec_int_from_null = OpConstant %int 0",
              },
            },

            // bool -> uint scalar
            {
              // original
              {
                "%spec_uint_one = OpSpecConstantOp %uint Select %bool_true %unsigned_one %unsigned_zero",
                "%spec_uint_zero = OpSpecConstantOp %uint Select %bool_false %unsigned_one %unsigned_zero",
                "%spec_uint_from_null = OpSpecConstantOp %uint Select %bool_null %unsigned_one %unsigned_zero",
              },
              // expected
              {
                "%spec_uint_one = OpConstant %uint 1",
                "%spec_uint_zero = OpConstant %uint 0",
                "%spec_uint_from_null = OpConstant %uint 0",
              },
            },

            // int -> uint scalar
            {
              // original
              {
                "%spec_uint_one = OpSpecConstantOp %uint IAdd %signed_one %unsigned_zero",
                "%spec_uint_zero = OpSpecConstantOp %uint IAdd %signed_zero %unsigned_zero",
                "%spec_uint_from_null = OpSpecConstantOp %uint IAdd %signed_null %unsigned_zero",
              },
              // expected
              {
                "%spec_uint_one = OpConstant %uint 1",
                "%spec_uint_zero = OpConstant %uint 0",
                "%spec_uint_from_null = OpConstant %uint 0",
              },
            },

            // int -> bool vector
            {
              // original
              {
                "%spec_bool_t_vec = OpSpecConstantOp %v2bool INotEqual %signed_three_vec %signed_zero_vec",
                "%spec_bool_f_vec = OpSpecConstantOp %v2bool INotEqual %signed_zero_vec %signed_zero_vec",
                "%spec_bool_from_null = OpSpecConstantOp %v2bool INotEqual %signed_null_vec %signed_zero_vec",
              },
              // expected
              {
                "%true = OpConstantTrue %bool",
                "%true_0 = OpConstantTrue %bool",
                "%spec_bool_t_vec = OpConstantComposite %v2bool %bool_true %bool_true",
                "%false = OpConstantFalse %bool",
                "%false_0 = OpConstantFalse %bool",
                "%spec_bool_f_vec = OpConstantComposite %v2bool %bool_false %bool_false",
                "%false_1 = OpConstantFalse %bool",
                "%false_2 = OpConstantFalse %bool",
                "%spec_bool_from_null = OpConstantComposite %v2bool %bool_false %bool_false",
              },
            },

            // uint -> bool vector
            {
              // original
              {
                "%spec_bool_t_vec = OpSpecConstantOp %v2bool INotEqual %unsigned_three_vec %unsigned_zero_vec",
                "%spec_bool_f_vec = OpSpecConstantOp %v2bool INotEqual %unsigned_zero_vec %unsigned_zero_vec",
                "%spec_bool_from_null = OpSpecConstantOp %v2bool INotEqual %unsigned_null_vec %unsigned_zero_vec",
              },
              // expected
              {
                "%true = OpConstantTrue %bool",
                "%true_0 = OpConstantTrue %bool",
                "%spec_bool_t_vec = OpConstantComposite %v2bool %bool_true %bool_true",
                "%false = OpConstantFalse %bool",
                "%false_0 = OpConstantFalse %bool",
                "%spec_bool_f_vec = OpConstantComposite %v2bool %bool_false %bool_false",
                "%false_1 = OpConstantFalse %bool",
                "%false_2 = OpConstantFalse %bool",
                "%spec_bool_from_null = OpConstantComposite %v2bool %bool_false %bool_false",
              },
            },

            // bool -> int vector
            {
                // original
              {
                "%spec_int_one_vec = OpSpecConstantOp %v2int Select %bool_true_vec %signed_one_vec %signed_zero_vec",
                "%spec_int_zero_vec = OpSpecConstantOp %v2int Select %bool_false_vec %signed_one_vec %signed_zero_vec",
                "%spec_int_from_null = OpSpecConstantOp %v2int Select %bool_null_vec %signed_one_vec %signed_zero_vec",
              },
              // expected
              {
                "%int_1 = OpConstant %int 1",
                "%int_1_0 = OpConstant %int 1",
                "%spec_int_one_vec = OpConstantComposite %v2int %signed_one %signed_one",
                "%int_0 = OpConstant %int 0",
                "%int_0_0 = OpConstant %int 0",
                "%spec_int_zero_vec = OpConstantComposite %v2int %signed_zero %signed_zero",
                "%int_0_1 = OpConstant %int 0",
                "%int_0_2 = OpConstant %int 0",
                "%spec_int_from_null = OpConstantComposite %v2int %signed_zero %signed_zero",
              },
            },

            // uint -> int vector
            {
              // original
              {
                "%spec_int_one_vec = OpSpecConstantOp %v2int IAdd %unsigned_one_vec %signed_zero_vec",
                "%spec_int_zero_vec = OpSpecConstantOp %v2int IAdd %unsigned_zero_vec %signed_zero_vec",
                "%spec_int_from_null = OpSpecConstantOp %v2int IAdd %unsigned_null_vec %signed_zero_vec",
              },
              // expected
              {
                "%int_1 = OpConstant %int 1",
                "%int_1_0 = OpConstant %int 1",
                "%spec_int_one_vec = OpConstantComposite %v2int %signed_one %signed_one",
                "%int_0 = OpConstant %int 0",
                "%int_0_0 = OpConstant %int 0",
                "%spec_int_zero_vec = OpConstantComposite %v2int %signed_zero %signed_zero",
                "%int_0_1 = OpConstant %int 0",
                "%int_0_2 = OpConstant %int 0",
                "%spec_int_from_null = OpConstantComposite %v2int %signed_zero %signed_zero",
              },
            },

            // bool -> uint vector
            {
              // original
              {
                "%spec_uint_one_vec = OpSpecConstantOp %v2uint Select %bool_true_vec %unsigned_one_vec %unsigned_zero_vec",
                "%spec_uint_zero_vec = OpSpecConstantOp %v2uint Select %bool_false_vec %unsigned_one_vec %unsigned_zero_vec",
                "%spec_uint_from_null = OpSpecConstantOp %v2uint Select %bool_null_vec %unsigned_one_vec %unsigned_zero_vec",
              },
              // expected
              {
                "%uint_1 = OpConstant %uint 1",
                "%uint_1_0 = OpConstant %uint 1",
                "%spec_uint_one_vec = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
                "%uint_0 = OpConstant %uint 0",
                "%uint_0_0 = OpConstant %uint 0",
                "%spec_uint_zero_vec = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
                "%uint_0_1 = OpConstant %uint 0",
                "%uint_0_2 = OpConstant %uint 0",
                "%spec_uint_from_null = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
              },
            },

            // int -> uint vector
            {
              // original
              {
                "%spec_uint_one_vec = OpSpecConstantOp %v2uint IAdd %signed_one_vec %unsigned_zero_vec",
                "%spec_uint_zero_vec = OpSpecConstantOp %v2uint IAdd %signed_zero_vec %unsigned_zero_vec",
                "%spec_uint_from_null = OpSpecConstantOp %v2uint IAdd %signed_null_vec %unsigned_zero_vec",
              },
              // expected
              {
                "%uint_1 = OpConstant %uint 1",
                "%uint_1_0 = OpConstant %uint 1",
                "%spec_uint_one_vec = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
                "%uint_0 = OpConstant %uint 0",
                "%uint_0_0 = OpConstant %uint 0",
                "%spec_uint_zero_vec = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
                "%uint_0_1 = OpConstant %uint 0",
                "%uint_0_2 = OpConstant %uint 0",
                "%spec_uint_from_null = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
              },
            },
            // clang-format on
        })));

// Tests about boolean scalar logical operations and comparison operations with
// scalar int/uint type.
INSTANTIATE_TEST_SUITE_P(
    Logical, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // scalar integer comparison
            {
              // original
              {
                "%int_minus_1 = OpConstant %int -1",

                "%slt_0_1 = OpSpecConstantOp %bool SLessThan %signed_zero %signed_one",
                "%sgt_0_1 = OpSpecConstantOp %bool SGreaterThan %signed_zero %signed_one",
                "%sle_2_2 = OpSpecConstantOp %bool SLessThanEqual %signed_two %signed_two",
                "%sge_2_1 = OpSpecConstantOp %bool SGreaterThanEqual %signed_two %signed_one",
                "%sge_2_null = OpSpecConstantOp %bool SGreaterThanEqual %signed_two %signed_null",
                "%sge_minus_1_null = OpSpecConstantOp %bool SGreaterThanEqual %int_minus_1 %signed_null",

                "%ult_0_1 = OpSpecConstantOp %bool ULessThan %unsigned_zero %unsigned_one",
                "%ugt_0_1 = OpSpecConstantOp %bool UGreaterThan %unsigned_zero %unsigned_one",
                "%ule_2_3 = OpSpecConstantOp %bool ULessThanEqual %unsigned_two %unsigned_three",
                "%uge_1_1 = OpSpecConstantOp %bool UGreaterThanEqual %unsigned_one %unsigned_one",
                "%uge_2_null = OpSpecConstantOp %bool UGreaterThanEqual %unsigned_two %unsigned_null",
                "%uge_minus_1_null = OpSpecConstantOp %bool UGreaterThanEqual %int_minus_1 %unsigned_null",
              },
              // expected
              {
                "%int_minus_1 = OpConstant %int -1",

                "%slt_0_1 = OpConstantTrue %bool",
                "%sgt_0_1 = OpConstantFalse %bool",
                "%sle_2_2 = OpConstantTrue %bool",
                "%sge_2_1 = OpConstantTrue %bool",
                "%sge_2_null = OpConstantTrue %bool",
                "%sge_minus_1_null = OpConstantFalse %bool",

                "%ult_0_1 = OpConstantTrue %bool",
                "%ugt_0_1 = OpConstantFalse %bool",
                "%ule_2_3 = OpConstantTrue %bool",
                "%uge_1_1 = OpConstantTrue %bool",
                "%uge_2_null = OpConstantTrue %bool",
                "%uge_minus_1_null = OpConstantTrue %bool",
              },
            },
            // Logical and, or, xor.
            {
              // original
              {
                "%logical_or = OpSpecConstantOp %bool LogicalOr %bool_true %bool_false",
                "%logical_and = OpSpecConstantOp %bool LogicalAnd %bool_true %bool_false",
                "%logical_not = OpSpecConstantOp %bool LogicalNot %bool_true",
                "%logical_eq = OpSpecConstantOp %bool LogicalEqual %bool_true %bool_true",
                "%logical_neq = OpSpecConstantOp %bool LogicalNotEqual %bool_true %bool_true",
                "%logical_and_null = OpSpecConstantOp %bool LogicalAnd %bool_true %bool_null",
              },
              // expected
              {
                "%logical_or = OpConstantTrue %bool",
                "%logical_and = OpConstantFalse %bool",
                "%logical_not = OpConstantFalse %bool",
                "%logical_eq = OpConstantTrue %bool",
                "%logical_neq = OpConstantFalse %bool",
                "%logical_and_null = OpConstantFalse %bool",
              },
            },
        // clang-format on
    })));

// Tests about arithmetic operations for scalar int and uint types.
INSTANTIATE_TEST_SUITE_P(
    ScalarArithmetic, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // scalar integer negate
            {
              // original
              {
                "%int_minus_1 = OpSpecConstantOp %int SNegate %signed_one",
                "%int_minus_2 = OpSpecConstantOp %int SNegate %signed_two",
                "%int_neg_null = OpSpecConstantOp %int SNegate %signed_null",
                "%int_max = OpConstant %int 2147483647",
                "%int_neg_max = OpSpecConstantOp %int SNegate %int_max",
              },
              // expected
              {
                "%int_minus_1 = OpConstant %int -1",
                "%int_minus_2 = OpConstant %int -2",
                "%int_neg_null = OpConstant %int 0",
                "%int_max = OpConstant %int 2147483647",
                "%int_neg_max = OpConstant %int -2147483647",
              },
            },
            // scalar integer not
            {
              // original
              {
                "%uint_4294967294 = OpSpecConstantOp %uint Not %unsigned_one",
                "%uint_4294967293 = OpSpecConstantOp %uint Not %unsigned_two",
                "%uint_neg_null = OpSpecConstantOp %uint Not %unsigned_null",
              },
              // expected
              {
                "%uint_4294967294 = OpConstant %uint 4294967294",
                "%uint_4294967293 = OpConstant %uint 4294967293",
                "%uint_neg_null = OpConstant %uint 4294967295",
              },
            },
            // scalar integer add, sub, mul, div
            {
              // original
              {
                "%signed_max = OpConstant %int 2147483647",
                "%signed_min = OpConstant %int -2147483648",

                "%spec_int_iadd = OpSpecConstantOp %int IAdd %signed_three %signed_two",
                "%spec_int_isub = OpSpecConstantOp %int ISub %signed_one %spec_int_iadd",
                "%spec_int_sdiv = OpSpecConstantOp %int SDiv %spec_int_isub %signed_two",
                "%spec_int_imul = OpSpecConstantOp %int IMul %spec_int_sdiv %signed_three",
                "%spec_int_iadd_null = OpSpecConstantOp %int IAdd %spec_int_imul %signed_null",
                "%spec_int_imul_null = OpSpecConstantOp %int IMul %spec_int_iadd_null %signed_null",
                "%spec_int_iadd_overflow = OpSpecConstantOp %int IAdd %signed_max %signed_three",
                "%spec_int_isub_overflow = OpSpecConstantOp %int ISub %signed_min %signed_three",

                "%spec_uint_iadd = OpSpecConstantOp %uint IAdd %unsigned_three %unsigned_two",
                "%spec_uint_isub = OpSpecConstantOp %uint ISub %unsigned_one %spec_uint_iadd",
                "%spec_uint_udiv = OpSpecConstantOp %uint UDiv %spec_uint_isub %unsigned_three",
                "%spec_uint_imul = OpSpecConstantOp %uint IMul %spec_uint_udiv %unsigned_two",
                "%spec_uint_isub_null = OpSpecConstantOp %uint ISub %spec_uint_imul %signed_null",
              },
              // expected
              {
                "%signed_max = OpConstant %int 2147483647",
                "%signed_min = OpConstant %int -2147483648",

                "%spec_int_iadd = OpConstant %int 5",
                "%spec_int_isub = OpConstant %int -4",
                "%spec_int_sdiv = OpConstant %int -2",
                "%spec_int_imul = OpConstant %int -6",
                "%spec_int_iadd_null = OpConstant %int -6",
                "%spec_int_imul_null = OpConstant %int 0",
                "%spec_int_iadd_overflow = OpConstant %int -2147483646",
                "%spec_int_isub_overflow = OpConstant %int 2147483645",

                "%spec_uint_iadd = OpConstant %uint 5",
                "%spec_uint_isub = OpConstant %uint 4294967292",
                "%spec_uint_udiv = OpConstant %uint 1431655764",
                "%spec_uint_imul = OpConstant %uint 2863311528",
                "%spec_uint_isub_null = OpConstant %uint 2863311528",
              },
            },
            // scalar integer rem, mod
            {
              // original
              {
                // common constants
                "%int_7 = OpConstant %int 7",
                "%uint_7 = OpConstant %uint 7",
                "%int_minus_7 = OpConstant %int -7",
                "%int_minus_3 = OpConstant %int -3",

                // srem
                "%7_srem_3 = OpSpecConstantOp %int SRem %int_7 %signed_three",
                "%minus_7_srem_3 = OpSpecConstantOp %int SRem %int_minus_7 %signed_three",
                "%7_srem_minus_3 = OpSpecConstantOp %int SRem %int_7 %int_minus_3",
                "%minus_7_srem_minus_3 = OpSpecConstantOp %int SRem %int_minus_7 %int_minus_3",
                // smod
                "%7_smod_3 = OpSpecConstantOp %int SMod %int_7 %signed_three",
                "%minus_7_smod_3 = OpSpecConstantOp %int SMod %int_minus_7 %signed_three",
                "%7_smod_minus_3 = OpSpecConstantOp %int SMod %int_7 %int_minus_3",
                "%minus_7_smod_minus_3 = OpSpecConstantOp %int SMod %int_minus_7 %int_minus_3",
                // umod
                "%7_umod_3 = OpSpecConstantOp %uint UMod %uint_7 %unsigned_three",
                // null constant
                "%null_srem_3 = OpSpecConstantOp %int SRem %signed_null %signed_three",
                "%null_smod_3 = OpSpecConstantOp %int SMod %signed_null %signed_three",
                "%null_umod_3 = OpSpecConstantOp %uint UMod %unsigned_null %unsigned_three",
              },
              // expected
              {
                // common constants
                "%int_7 = OpConstant %int 7",
                "%uint_7 = OpConstant %uint 7",
                "%int_minus_7 = OpConstant %int -7",
                "%int_minus_3 = OpConstant %int -3",

                // srem
                "%7_srem_3 = OpConstant %int 1",
                "%minus_7_srem_3 = OpConstant %int -1",
                "%7_srem_minus_3 = OpConstant %int 1",
                "%minus_7_srem_minus_3 = OpConstant %int -1",
                // smod
                "%7_smod_3 = OpConstant %int 1",
                "%minus_7_smod_3 = OpConstant %int 2",
                "%7_smod_minus_3 = OpConstant %int -2",
                "%minus_7_smod_minus_3 = OpConstant %int -1",
                // umod
                "%7_umod_3 = OpConstant %uint 1",
                // null constant
                "%null_srem_3 = OpConstant %int 0",
                "%null_smod_3 = OpConstant %int 0",
                "%null_umod_3 = OpConstant %uint 0",
              },
            },
            // scalar integer bitwise and shift
            {
              // original
              {
                // bitwise
                "%xor_1_3 = OpSpecConstantOp %int BitwiseXor %signed_one %signed_three",
                "%and_1_2 = OpSpecConstantOp %int BitwiseAnd %signed_one %xor_1_3",
                "%or_1_2 = OpSpecConstantOp %int BitwiseOr %signed_one %xor_1_3",
                "%xor_3_null = OpSpecConstantOp %int BitwiseXor %or_1_2 %signed_null",

                // shift
                "%unsigned_31 = OpConstant %uint 31",
                "%unsigned_left_shift_max = OpSpecConstantOp %uint ShiftLeftLogical %unsigned_one %unsigned_31",
                "%unsigned_right_shift_logical = OpSpecConstantOp %uint ShiftRightLogical %unsigned_left_shift_max %unsigned_31",
                "%signed_right_shift_arithmetic = OpSpecConstantOp %int ShiftRightArithmetic %unsigned_left_shift_max %unsigned_31",
                "%left_shift_null_31 = OpSpecConstantOp %uint ShiftLeftLogical %unsigned_null %unsigned_31",
                "%right_shift_31_null = OpSpecConstantOp %uint ShiftRightLogical %unsigned_31 %unsigned_null",
              },
              // expected
              {
                "%xor_1_3 = OpConstant %int 2",
                "%and_1_2 = OpConstant %int 0",
                "%or_1_2 = OpConstant %int 3",
                "%xor_3_null = OpConstant %int 3",

                "%unsigned_31 = OpConstant %uint 31",
                "%unsigned_left_shift_max = OpConstant %uint 2147483648",
                "%unsigned_right_shift_logical = OpConstant %uint 1",
                "%signed_right_shift_arithmetic = OpConstant %int -1",
                "%left_shift_null_31 = OpConstant %uint 0",
                "%right_shift_31_null = OpConstant %uint 31",
              },
            },
            // Skip folding if any operands have undetermined value.
            {
              // original
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_iadd = OpSpecConstantOp %int IAdd %signed_three %spec_int",
              },
              // expected
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_iadd = OpSpecConstantOp %int IAdd %signed_three %spec_int",
              },
            },
        // clang-format on
    })));

// Tests about arithmetic operations for vector int and uint types.
INSTANTIATE_TEST_SUITE_P(
    VectorArithmetic, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // vector integer negate
            {
              // original
              {
                "%v2int_minus_1 = OpSpecConstantOp %v2int SNegate %signed_one_vec",
                "%v2int_minus_2 = OpSpecConstantOp %v2int SNegate %signed_two_vec",
                "%v2int_neg_null = OpSpecConstantOp %v2int SNegate %signed_null_vec",
              },
              // expected
              {
                "%int_n1 = OpConstant %int -1",
                "%int_n1_0 = OpConstant %int -1",
                "%v2int_minus_1 = OpConstantComposite %v2int %int_n1 %int_n1",
                "%int_n2 = OpConstant %int -2",
                "%int_n2_0 = OpConstant %int -2",
                "%v2int_minus_2 = OpConstantComposite %v2int %int_n2 %int_n2",
                "%int_0 = OpConstant %int 0",
                "%int_0_0 = OpConstant %int 0",
                "%v2int_neg_null = OpConstantComposite %v2int %signed_zero %signed_zero",
              },
            },
            // vector integer (including null vetors) add, sub, div, mul
            {
              // original
              {
                "%spec_v2int_iadd = OpSpecConstantOp %v2int IAdd %signed_three_vec %signed_two_vec",
                "%spec_v2int_isub = OpSpecConstantOp %v2int ISub %signed_one_vec %spec_v2int_iadd",
                "%spec_v2int_sdiv = OpSpecConstantOp %v2int SDiv %spec_v2int_isub %signed_two_vec",
                "%spec_v2int_imul = OpSpecConstantOp %v2int IMul %spec_v2int_sdiv %signed_three_vec",
                "%spec_v2int_iadd_null = OpSpecConstantOp %v2int IAdd %spec_v2int_imul %signed_null_vec",

                "%spec_v2uint_iadd = OpSpecConstantOp %v2uint IAdd %unsigned_three_vec %unsigned_two_vec",
                "%spec_v2uint_isub = OpSpecConstantOp %v2uint ISub %unsigned_one_vec %spec_v2uint_iadd",
                "%spec_v2uint_udiv = OpSpecConstantOp %v2uint UDiv %spec_v2uint_isub %unsigned_three_vec",
                "%spec_v2uint_imul = OpSpecConstantOp %v2uint IMul %spec_v2uint_udiv %unsigned_two_vec",
                "%spec_v2uint_isub_null = OpSpecConstantOp %v2uint ISub %spec_v2uint_imul %signed_null_vec",
              },
              // expected
              {
                "%int_5 = OpConstant %int 5",
                "%int_5_0 = OpConstant %int 5",
                "%spec_v2int_iadd = OpConstantComposite %v2int %int_5 %int_5",
                "%int_n4 = OpConstant %int -4",
                "%int_n4_0 = OpConstant %int -4",
                "%spec_v2int_isub = OpConstantComposite %v2int %int_n4 %int_n4",
                "%int_n2 = OpConstant %int -2",
                "%int_n2_0 = OpConstant %int -2",
                "%spec_v2int_sdiv = OpConstantComposite %v2int %int_n2 %int_n2",
                "%int_n6 = OpConstant %int -6",
                "%int_n6_0 = OpConstant %int -6",
                "%spec_v2int_imul = OpConstantComposite %v2int %int_n6 %int_n6",
                "%int_n6_1 = OpConstant %int -6",
                "%int_n6_2 = OpConstant %int -6",
                "%spec_v2int_iadd_null = OpConstantComposite %v2int %int_n6 %int_n6",

                "%uint_5 = OpConstant %uint 5",
                "%uint_5_0 = OpConstant %uint 5",
                "%spec_v2uint_iadd = OpConstantComposite %v2uint %uint_5 %uint_5",
                "%uint_4294967292 = OpConstant %uint 4294967292",
                "%uint_4294967292_0 = OpConstant %uint 4294967292",
                "%spec_v2uint_isub = OpConstantComposite %v2uint %uint_4294967292 %uint_4294967292",
                "%uint_1431655764 = OpConstant %uint 1431655764",
                "%uint_1431655764_0 = OpConstant %uint 1431655764",
                "%spec_v2uint_udiv = OpConstantComposite %v2uint %uint_1431655764 %uint_1431655764",
                "%uint_2863311528 = OpConstant %uint 2863311528",
                "%uint_2863311528_0 = OpConstant %uint 2863311528",
                "%spec_v2uint_imul = OpConstantComposite %v2uint %uint_2863311528 %uint_2863311528",
                "%uint_2863311528_1 = OpConstant %uint 2863311528",
                "%uint_2863311528_2 = OpConstant %uint 2863311528",
                "%spec_v2uint_isub_null = OpConstantComposite %v2uint %uint_2863311528 %uint_2863311528",
              },
            },
            // vector integer rem, mod
            {
              // original
              {
                // common constants
                "%int_7 = OpConstant %int 7",
                "%v2int_7 = OpConstantComposite %v2int %int_7 %int_7",
                "%uint_7 = OpConstant %uint 7",
                "%v2uint_7 = OpConstantComposite %v2uint %uint_7 %uint_7",
                "%int_minus_7 = OpConstant %int -7",
                "%v2int_minus_7 = OpConstantComposite %v2int %int_minus_7 %int_minus_7",
                "%int_minus_3 = OpConstant %int -3",
                "%v2int_minus_3 = OpConstantComposite %v2int %int_minus_3 %int_minus_3",

                // srem
                "%7_srem_3 = OpSpecConstantOp %v2int SRem %v2int_7 %signed_three_vec",
                "%minus_7_srem_3 = OpSpecConstantOp %v2int SRem %v2int_minus_7 %signed_three_vec",
                "%7_srem_minus_3 = OpSpecConstantOp %v2int SRem %v2int_7 %v2int_minus_3",
                "%minus_7_srem_minus_3 = OpSpecConstantOp %v2int SRem %v2int_minus_7 %v2int_minus_3",
                // smod
                "%7_smod_3 = OpSpecConstantOp %v2int SMod %v2int_7 %signed_three_vec",
                "%minus_7_smod_3 = OpSpecConstantOp %v2int SMod %v2int_minus_7 %signed_three_vec",
                "%7_smod_minus_3 = OpSpecConstantOp %v2int SMod %v2int_7 %v2int_minus_3",
                "%minus_7_smod_minus_3 = OpSpecConstantOp %v2int SMod %v2int_minus_7 %v2int_minus_3",
                // umod
                "%7_umod_3 = OpSpecConstantOp %v2uint UMod %v2uint_7 %unsigned_three_vec",
              },
              // expected
              {
                // common constants
                "%int_7 = OpConstant %int 7",
                "%v2int_7 = OpConstantComposite %v2int %int_7 %int_7",
                "%uint_7 = OpConstant %uint 7",
                "%v2uint_7 = OpConstantComposite %v2uint %uint_7 %uint_7",
                "%int_minus_7 = OpConstant %int -7",
                "%v2int_minus_7 = OpConstantComposite %v2int %int_minus_7 %int_minus_7",
                "%int_minus_3 = OpConstant %int -3",
                "%v2int_minus_3 = OpConstantComposite %v2int %int_minus_3 %int_minus_3",

                // srem
                "%int_1 = OpConstant %int 1",
                "%int_1_0 = OpConstant %int 1",
                "%7_srem_3 = OpConstantComposite %v2int %signed_one %signed_one",
                "%int_n1 = OpConstant %int -1",
                "%int_n1_0 = OpConstant %int -1",
                "%minus_7_srem_3 = OpConstantComposite %v2int %int_n1 %int_n1",
                "%int_1_1 = OpConstant %int 1",
                "%int_1_2 = OpConstant %int 1",
                "%7_srem_minus_3 = OpConstantComposite %v2int %signed_one %signed_one",
                "%int_n1_1 = OpConstant %int -1",
                "%int_n1_2 = OpConstant %int -1",
                "%minus_7_srem_minus_3 = OpConstantComposite %v2int %int_n1 %int_n1",
                // smod
                "%int_1_3 = OpConstant %int 1",
                "%int_1_4 = OpConstant %int 1",
                "%7_smod_3 = OpConstantComposite %v2int %signed_one %signed_one",
                "%int_2 = OpConstant %int 2",
                "%int_2_0 = OpConstant %int 2",
                "%minus_7_smod_3 = OpConstantComposite %v2int %signed_two %signed_two",
                "%int_n2 = OpConstant %int -2",
                "%int_n2_0 = OpConstant %int -2",
                "%7_smod_minus_3 = OpConstantComposite %v2int %int_n2 %int_n2",
                "%int_n1_3 = OpConstant %int -1",
                "%int_n1_4 = OpConstant %int -1",
                "%minus_7_smod_minus_3 = OpConstantComposite %v2int %int_n1 %int_n1",
                // umod
                "%uint_1 = OpConstant %uint 1",
                "%uint_1_0 = OpConstant %uint 1",
                "%7_umod_3 = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
              },
            },
            // vector integer bitwise, shift
            {
              // original
              {
                "%xor_1_3 = OpSpecConstantOp %v2int BitwiseXor %signed_one_vec %signed_three_vec",
                "%and_1_2 = OpSpecConstantOp %v2int BitwiseAnd %signed_one_vec %xor_1_3",
                "%or_1_2 = OpSpecConstantOp %v2int BitwiseOr %signed_one_vec %xor_1_3",

                "%unsigned_31 = OpConstant %uint 31",
                "%v2unsigned_31 = OpConstantComposite %v2uint %unsigned_31 %unsigned_31",
                "%unsigned_left_shift_max = OpSpecConstantOp %v2uint ShiftLeftLogical %unsigned_one_vec %v2unsigned_31",
                "%unsigned_right_shift_logical = OpSpecConstantOp %v2uint ShiftRightLogical %unsigned_left_shift_max %v2unsigned_31",
                "%signed_right_shift_arithmetic = OpSpecConstantOp %v2int ShiftRightArithmetic %unsigned_left_shift_max %v2unsigned_31",
              },
              // expected
              {
                "%int_2 = OpConstant %int 2",
                "%int_2_0 = OpConstant %int 2",
                "%xor_1_3 = OpConstantComposite %v2int %signed_two %signed_two",
                "%int_0 = OpConstant %int 0",
                "%int_0_0 = OpConstant %int 0",
                "%and_1_2 = OpConstantComposite %v2int %signed_zero %signed_zero",
                "%int_3 = OpConstant %int 3",
                "%int_3_0 = OpConstant %int 3",
                "%or_1_2 = OpConstantComposite %v2int %signed_three %signed_three",

                "%unsigned_31 = OpConstant %uint 31",
                "%v2unsigned_31 = OpConstantComposite %v2uint %unsigned_31 %unsigned_31",
                "%uint_2147483648 = OpConstant %uint 2147483648",
                "%uint_2147483648_0 = OpConstant %uint 2147483648",
                "%unsigned_left_shift_max = OpConstantComposite %v2uint %uint_2147483648 %uint_2147483648",
                "%uint_1 = OpConstant %uint 1",
                "%uint_1_0 = OpConstant %uint 1",
                "%unsigned_right_shift_logical = OpConstantComposite %v2uint %unsigned_one %unsigned_one",
                "%int_n1 = OpConstant %int -1",
                "%int_n1_0 = OpConstant %int -1",
                "%signed_right_shift_arithmetic = OpConstantComposite %v2int %int_n1 %int_n1",
              },
            },
            // Skip folding if any vector operands or components of the operands
            // have undetermined value.
            {
              // original
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_vec = OpSpecConstantComposite %v2int %signed_zero %spec_int",
                "%spec_iadd = OpSpecConstantOp %v2int IAdd %signed_three_vec %spec_vec",
              },
              // expected
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_vec = OpSpecConstantComposite %v2int %signed_zero %spec_int",
                "%spec_iadd = OpSpecConstantOp %v2int IAdd %signed_three_vec %spec_vec",
              },
            },
            // Skip folding if any vector operands are defined by OpUndef
            {
              // original
              {
                "%undef = OpUndef %int",
                "%vec = OpConstantComposite %v2int %undef %signed_one",
                "%spec_iadd = OpSpecConstantOp %v2int IAdd %signed_three_vec %vec",
              },
              // expected
              {
                "%undef = OpUndef %int",
                "%vec = OpConstantComposite %v2int %undef %signed_one",
                "%spec_iadd = OpSpecConstantOp %v2int IAdd %signed_three_vec %vec",
              },
            },
        // clang-format on
    })));

// Tests for SpecConstantOp CompositeExtract instruction
INSTANTIATE_TEST_SUITE_P(
    CompositeExtract, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // normal vector
            {
              // original
              {
                "%r = OpSpecConstantOp %int CompositeExtract %signed_three_vec 0",
                "%x = OpSpecConstantOp %int CompositeExtract %v4int_0_1_2_3 0",
                "%y = OpSpecConstantOp %int CompositeExtract %v4int_0_1_2_3 1",
                "%z = OpSpecConstantOp %int CompositeExtract %v4int_0_1_2_3 2",
                "%w = OpSpecConstantOp %int CompositeExtract %v4int_0_1_2_3 3",
              },
              // expected
              {
                "%r = OpConstant %int 3",
                "%x = OpConstant %int 0",
                "%y = OpConstant %int 1",
                "%z = OpConstant %int 2",
                "%w = OpConstant %int 3",
              },
            },
            // null vector
            {
              // original
              {
                "%x = OpSpecConstantOp %int CompositeExtract %signed_null_vec 0",
                "%y = OpSpecConstantOp %int CompositeExtract %signed_null_vec 1",
                "%null_v4int = OpConstantNull %v4int",
                "%z = OpSpecConstantOp %int CompositeExtract %signed_null_vec 2",
              },
              // expected
              {
                "%x = OpConstantNull %int",
                "%y = OpConstantNull %int",
                "%null_v4int = OpConstantNull %v4int",
                "%z = OpConstantNull %int",
              }
            },
            // normal flat struct
            {
              // original
              {
                "%float_1 = OpConstant %float 1",
                "%flat_1 = OpConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%extract_bool = OpSpecConstantOp %bool CompositeExtract %flat_1 0",
                "%extract_int = OpSpecConstantOp %int CompositeExtract %flat_1 1",
                "%extract_float_1 = OpSpecConstantOp %float CompositeExtract %flat_1 2",
                // foldable composite constants built with OpSpecConstantComposite
                // should also be processed.
                "%flat_2 = OpSpecConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%extract_float_2 = OpSpecConstantOp %float CompositeExtract %flat_2 2",
              },
              // expected
              {
                "%float_1 = OpConstant %float 1",
                "%flat_1 = OpConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%extract_bool = OpConstantTrue %bool",
                "%extract_int = OpConstantNull %int",
                "%extract_float_1 = OpConstant %float 1",
                "%flat_2 = OpConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%extract_float_2 = OpConstant %float 1",
              },
            },
            // null flat struct
            {
              // original
              {
                "%flat = OpConstantNull %flat_struct",
                "%extract_bool = OpSpecConstantOp %bool CompositeExtract %flat 0",
                "%extract_int = OpSpecConstantOp %int CompositeExtract %flat 1",
                "%extract_float = OpSpecConstantOp %float CompositeExtract %flat 2",
              },
              // expected
              {
                "%flat = OpConstantNull %flat_struct",
                "%extract_bool = OpConstantNull %bool",
                "%extract_int = OpConstantNull %int",
                "%extract_float = OpConstantNull %float",
              },
            },
            // normal nested struct
            {
              // original
              {
                "%float_1 = OpConstant %float 1",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_null %float_1",
                "%outer = OpConstantComposite %outer_struct %inner %signed_one",
                "%extract_inner = OpSpecConstantOp %inner_struct CompositeExtract %outer 0",
                "%extract_int = OpSpecConstantOp %int CompositeExtract %outer 1",
                "%extract_inner_float = OpSpecConstantOp %int CompositeExtract %outer 0 2",
              },
              // expected
              {
                "%float_1 = OpConstant %float 1",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_null %float_1",
                "%outer = OpConstantComposite %outer_struct %inner %signed_one",
                "%extract_inner = OpConstantComposite %flat_struct %bool_true %signed_null %float_1",
                "%extract_int = OpConstant %int 1",
                "%extract_inner_float = OpConstant %float 1",
              },
            },
            // null nested struct
            {
              // original
              {
                "%outer = OpConstantNull %outer_struct",
                "%extract_inner = OpSpecConstantOp %inner_struct CompositeExtract %outer 0",
                "%extract_int = OpSpecConstantOp %int CompositeExtract %outer 1",
                "%extract_inner_float = OpSpecConstantOp %float CompositeExtract %outer 0 2",
              },
              // expected
              {
                "%outer = OpConstantNull %outer_struct",
                "%extract_inner = OpConstantNull %inner_struct",
                "%extract_int = OpConstantNull %int",
                "%extract_inner_float = OpConstantNull %float",
              },
            },
            // skip folding if the any composite constant's value are not fully
            // determined, even though the extracting target might have
            // determined value.
            {
              // original
              {
                "%float_1 = OpConstant %float 1",
                "%spec_float = OpSpecConstant %float 1",
                "%spec_inner = OpSpecConstantComposite %inner_struct %bool_true %signed_null %spec_float",
                "%spec_outer = OpSpecConstantComposite %outer_struct %spec_inner %signed_one",
                "%spec_vec = OpSpecConstantComposite %v2float %spec_float %float_1",
                "%extract_inner = OpSpecConstantOp %int CompositeExtract %spec_inner 1",
                "%extract_outer = OpSpecConstantOp %int CompositeExtract %spec_outer 1",
                "%extract_vec = OpSpecConstantOp %float CompositeExtract %spec_vec 1",
              },
              // expected
              {
                "%float_1 = OpConstant %float 1",
                "%spec_float = OpSpecConstant %float 1",
                "%spec_inner = OpSpecConstantComposite %inner_struct %bool_true %signed_null %spec_float",
                "%spec_outer = OpSpecConstantComposite %outer_struct %spec_inner %signed_one",
                "%spec_vec = OpSpecConstantComposite %v2float %spec_float %float_1",
                "%extract_inner = OpSpecConstantOp %int CompositeExtract %spec_inner 1",
                "%extract_outer = OpSpecConstantOp %int CompositeExtract %spec_outer 1",
                "%extract_vec = OpSpecConstantOp %float CompositeExtract %spec_vec 1",
              },
            },
            // skip if the composite constant depends on the result of OpUndef,
            // even though the composite extract target element does not depends
            // on the OpUndef.
            {
              // original
              {
                "%undef = OpUndef %float",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_one %undef",
                "%outer = OpConstantComposite %outer_struct %inner %signed_one",
                "%extract_inner = OpSpecConstantOp %int CompositeExtract %inner 1",
                "%extract_outer = OpSpecConstantOp %int CompositeExtract %outer 1",
              },
              // expected
              {
                "%undef = OpUndef %float",
                "%inner = OpConstantComposite %inner_struct %bool_true %signed_one %undef",
                "%outer = OpConstantComposite %outer_struct %inner %signed_one",
                "%extract_inner = OpSpecConstantOp %int CompositeExtract %inner 1",
                "%extract_outer = OpSpecConstantOp %int CompositeExtract %outer 1",
              },
            },
            // TODO(qining): Add tests for Array and other composite type constants.
        // clang-format on
    })));

// Tests the swizzle operations for spec const vectors.
INSTANTIATE_TEST_SUITE_P(
    VectorShuffle, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
            // normal vector
            {
              // original
              {
                "%xy = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 0 1",
                "%yz = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 1 2",
                "%zw = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 2 3",
                "%wx = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 3 0",
                "%xx = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 0 0",
                "%yyy = OpSpecConstantOp %v3int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 1 1 1",
                "%wwww = OpSpecConstantOp %v4int VectorShuffle %v4int_0_1_2_3 %v4int_0_1_2_3 2 2 2 2",
              },
              // expected
              {
                "%xy = OpConstantComposite %v2int %signed_zero %signed_one",
                "%yz = OpConstantComposite %v2int %signed_one %signed_two",
                "%zw = OpConstantComposite %v2int %signed_two %signed_three",
                "%wx = OpConstantComposite %v2int %signed_three %signed_zero",
                "%xx = OpConstantComposite %v2int %signed_zero %signed_zero",
                "%yyy = OpConstantComposite %v3int %signed_one %signed_one %signed_one",
                "%wwww = OpConstantComposite %v4int %signed_two %signed_two %signed_two %signed_two",
              },
            },
            // null vector
            {
              // original
              {
                "%a = OpSpecConstantOp %v2int VectorShuffle %signed_null_vec %v4int_0_1_2_3 0 1",
                "%b = OpSpecConstantOp %v2int VectorShuffle %signed_null_vec %v4int_0_1_2_3 2 3",
                "%c = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %signed_null_vec 3 4",
                "%d = OpSpecConstantOp %v2int VectorShuffle %signed_null_vec %signed_null_vec 1 2",
              },
              // expected
              {
                "%60 = OpConstantNull %int",
                "%a = OpConstantComposite %v2int %signed_null %signed_null",
                "%62 = OpConstantNull %int",
                "%b = OpConstantComposite %v2int %signed_zero %signed_one",
                "%64 = OpConstantNull %int",
                "%c = OpConstantComposite %v2int %signed_three %signed_null",
                "%66 = OpConstantNull %int",
                "%d = OpConstantComposite %v2int %signed_null %signed_null",
              }
            },
            // skip if any of the components of the vector operands do not have
            // determined value, even though the result vector might not be
            // built with those undermined values.
            {
              // original
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_ivec = OpSpecConstantComposite %v2int %signed_null %spec_int",
                "%a = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %spec_ivec 0 1",
                "%b = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %spec_ivec 3 4",
              },
              // expected
              {
                "%spec_int = OpSpecConstant %int 1",
                "%spec_ivec = OpSpecConstantComposite %v2int %signed_null %spec_int",
                "%a = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %spec_ivec 0 1",
                "%b = OpSpecConstantOp %v2int VectorShuffle %v4int_0_1_2_3 %spec_ivec 3 4",
              },
            },
            // Skip if any components of the two vector operands depend on
            // the result of OpUndef. Even though the selected components do
            // not depend on the OpUndef result.
            {
              // original
              {
                "%undef = OpUndef %int",
                "%vec_1 = OpConstantComposite %v2int %undef %signed_one",
                "%dep = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 0 3",
                "%not_dep_element = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 1 3",
                "%no_dep_vector = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 2 3",
              },
              // expected
              {
                "%undef = OpUndef %int",
                "%vec_1 = OpConstantComposite %v2int %undef %signed_one",
                "%dep = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 0 3",
                "%not_dep_element = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 1 3",
                "%no_dep_vector = OpSpecConstantOp %v2int VectorShuffle %vec_1 %signed_three_vec 2 3",
              },
            },
        // clang-format on
    })));

// Test with long use-def chain.
INSTANTIATE_TEST_SUITE_P(
    LongDefUseChain, FoldSpecConstantOpAndCompositePassTest,
    ::testing::ValuesIn(std::vector<
                        FoldSpecConstantOpAndCompositePassTestCase>({
        // clang-format off
        // Long Def-Use chain with binary operations.
        {
            // original
            {
              "%array_size = OpConstant %int 4",
              "%type_arr_int_4 = OpTypeArray %int %array_size",
              "%spec_int_0 = OpConstant %int 100",
              "%spec_int_1 = OpConstant %int 1",
              "%spec_int_2 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_1",
              "%spec_int_3 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_2",
              "%spec_int_4 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_3",
              "%spec_int_5 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_4",
              "%spec_int_6 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_5",
              "%spec_int_7 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_6",
              "%spec_int_8 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_7",
              "%spec_int_9 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_8",
              "%spec_int_10 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_9",
              "%spec_int_11 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_10",
              "%spec_int_12 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_11",
              "%spec_int_13 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_12",
              "%spec_int_14 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_13",
              "%spec_int_15 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_14",
              "%spec_int_16 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_15",
              "%spec_int_17 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_16",
              "%spec_int_18 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_17",
              "%spec_int_19 = OpSpecConstantOp %int IAdd %spec_int_0 %spec_int_18",
              "%spec_int_20 = OpSpecConstantOp %int ISub %spec_int_0 %spec_int_19",
              "%used_vec_a = OpSpecConstantComposite %v2int %spec_int_18 %spec_int_19",
              "%used_vec_b = OpSpecConstantOp %v2int IMul %used_vec_a %used_vec_a",
              "%spec_int_21 = OpSpecConstantOp %int CompositeExtract %used_vec_b 0",
              "%array = OpConstantComposite %type_arr_int_4 %spec_int_20 %spec_int_20 %spec_int_21 %spec_int_21",
              // Spec constants whose values can not be fully resolved should
              // not be processed.
              "%spec_int_22 = OpSpecConstant %int 123",
              "%spec_int_23 = OpSpecConstantOp %int IAdd %spec_int_22 %signed_one",
            },
            // expected
            {
              "%array_size = OpConstant %int 4",
              "%type_arr_int_4 = OpTypeArray %int %array_size",
              "%spec_int_0 = OpConstant %int 100",
              "%spec_int_1 = OpConstant %int 1",
              "%spec_int_2 = OpConstant %int 101",
              "%spec_int_3 = OpConstant %int -1",
              "%spec_int_4 = OpConstant %int 99",
              "%spec_int_5 = OpConstant %int 1",
              "%spec_int_6 = OpConstant %int 101",
              "%spec_int_7 = OpConstant %int -1",
              "%spec_int_8 = OpConstant %int 99",
              "%spec_int_9 = OpConstant %int 1",
              "%spec_int_10 = OpConstant %int 101",
              "%spec_int_11 = OpConstant %int -1",
              "%spec_int_12 = OpConstant %int 99",
              "%spec_int_13 = OpConstant %int 1",
              "%spec_int_14 = OpConstant %int 101",
              "%spec_int_15 = OpConstant %int -1",
              "%spec_int_16 = OpConstant %int 101",
              "%spec_int_17 = OpConstant %int 201",
              "%spec_int_18 = OpConstant %int -101",
              "%spec_int_19 = OpConstant %int -1",
              "%spec_int_20 = OpConstant %int 101",
              "%used_vec_a = OpConstantComposite %v2int %spec_int_18 %spec_int_19",
              "%int_10201 = OpConstant %int 10201",
              "%int_1 = OpConstant %int 1",
              "%used_vec_b = OpConstantComposite %v2int %int_10201 %signed_one",
              "%spec_int_21 = OpConstant %int 10201",
              "%array = OpConstantComposite %type_arr_int_4 %spec_int_20 %spec_int_20 %spec_int_21 %spec_int_21",
              "%spec_int_22 = OpSpecConstant %int 123",
              "%spec_int_23 = OpSpecConstantOp %int IAdd %spec_int_22 %signed_one",
            },
        },
        // Long Def-Use chain with swizzle
        })));

}  // namespace
}  // namespace opt
}  // namespace spvtools
