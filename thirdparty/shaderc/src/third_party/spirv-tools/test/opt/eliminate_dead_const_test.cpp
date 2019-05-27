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

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using EliminateDeadConstantBasicTest = PassTest<::testing::Test>;

TEST_F(EliminateDeadConstantBasicTest, BasicAllDeadConstants) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%bool = OpTypeBool",
       "%true = OpConstantTrue %bool",
      "%false = OpConstantFalse %bool",
        "%int = OpTypeInt 32 1",
          "%9 = OpConstant %int 1",
       "%uint = OpTypeInt 32 0",
         "%11 = OpConstant %uint 2",
      "%float = OpTypeFloat 32",
         "%13 = OpConstant %float 3.1415",
     "%double = OpTypeFloat 64",
         "%15 = OpConstant %double 3.14159265358979",
       "%main = OpFunction %void None %4",
         "%16 = OpLabel",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  // None of the above constants is ever used, so all of them should be
  // eliminated.
  const char* const_decl_opcodes[] = {
      " OpConstantTrue ",
      " OpConstantFalse ",
      " OpConstant ",
  };
  // Skip lines that have any one of const_decl_opcodes.
  const std::string expected_disassembly =
      SelectiveJoin(text, [&const_decl_opcodes](const char* line) {
        return std::any_of(
            std::begin(const_decl_opcodes), std::end(const_decl_opcodes),
            [&line](const char* const_decl_op) {
              return std::string(line).find(const_decl_op) != std::string::npos;
            });
      });

  SinglePassRunAndCheck<EliminateDeadConstantPass>(
      JoinAllInsts(text), expected_disassembly, /* skip_nop = */ true);
}

TEST_F(EliminateDeadConstantBasicTest, BasicNoneDeadConstants) {
  const std::vector<const char*> text = {
      // clang-format off
                "OpCapability Shader",
                "OpCapability Float64",
           "%1 = OpExtInstImport \"GLSL.std.450\"",
                "OpMemoryModel Logical GLSL450",
                "OpEntryPoint Vertex %main \"main\"",
                "OpName %main \"main\"",
                "OpName %btv \"btv\"",
                "OpName %bfv \"bfv\"",
                "OpName %iv \"iv\"",
                "OpName %uv \"uv\"",
                "OpName %fv \"fv\"",
                "OpName %dv \"dv\"",
        "%void = OpTypeVoid",
          "%10 = OpTypeFunction %void",
        "%bool = OpTypeBool",
 "%_ptr_Function_bool = OpTypePointer Function %bool",
        "%true = OpConstantTrue %bool",
       "%false = OpConstantFalse %bool",
         "%int = OpTypeInt 32 1",
 "%_ptr_Function_int = OpTypePointer Function %int",
       "%int_1 = OpConstant %int 1",
        "%uint = OpTypeInt 32 0",
 "%_ptr_Function_uint = OpTypePointer Function %uint",
      "%uint_2 = OpConstant %uint 2",
       "%float = OpTypeFloat 32",
 "%_ptr_Function_float = OpTypePointer Function %float",
  "%float_3_1415 = OpConstant %float 3.1415",
      "%double = OpTypeFloat 64",
 "%_ptr_Function_double = OpTypePointer Function %double",
 "%double_3_14159265358979 = OpConstant %double 3.14159265358979",
        "%main = OpFunction %void None %10",
          "%27 = OpLabel",
         "%btv = OpVariable %_ptr_Function_bool Function",
         "%bfv = OpVariable %_ptr_Function_bool Function",
          "%iv = OpVariable %_ptr_Function_int Function",
          "%uv = OpVariable %_ptr_Function_uint Function",
          "%fv = OpVariable %_ptr_Function_float Function",
          "%dv = OpVariable %_ptr_Function_double Function",
                "OpStore %btv %true",
                "OpStore %bfv %false",
                "OpStore %iv %int_1",
                "OpStore %uv %uint_2",
                "OpStore %fv %float_3_1415",
                "OpStore %dv %double_3_14159265358979",
                "OpReturn",
                "OpFunctionEnd",
      // clang-format on
  };
  // All constants are used, so none of them should be eliminated.
  SinglePassRunAndCheck<EliminateDeadConstantPass>(
      JoinAllInsts(text), JoinAllInsts(text), /* skip_nop = */ true);
}

struct EliminateDeadConstantTestCase {
  // Type declarations and constants that should be kept.
  std::vector<std::string> used_consts;
  // Instructions that refer to constants, this is added to create uses for
  // some constants so they won't be treated as dead constants.
  std::vector<std::string> main_insts;
  // Dead constants that should be removed.
  std::vector<std::string> dead_consts;
};

// All types that are potentially required in EliminateDeadConstantTest.
const std::vector<std::string> CommonTypes = {
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
    "%v3float = OpTypeVector %float 3",
    "%v2double = OpTypeVector %double 2",
    // variable pointer types
    "%_pf_bool = OpTypePointer Function %bool",
    "%_pf_uint = OpTypePointer Function %uint",
    "%_pf_int = OpTypePointer Function %int",
    "%_pf_float = OpTypePointer Function %float",
    "%_pf_double = OpTypePointer Function %double",
    "%_pf_v2int = OpTypePointer Function %v2int",
    "%_pf_v3int = OpTypePointer Function %v3int",
    "%_pf_v2float = OpTypePointer Function %v2float",
    "%_pf_v3float = OpTypePointer Function %v3float",
    "%_pf_v2double = OpTypePointer Function %v2double",
    // struct types
    "%inner_struct = OpTypeStruct %bool %int %float %double",
    "%outer_struct = OpTypeStruct %inner_struct %int %double",
    "%flat_struct = OpTypeStruct %bool %int %float %double",
    // clang-format on
};

using EliminateDeadConstantTest =
    PassTest<::testing::TestWithParam<EliminateDeadConstantTestCase>>;

TEST_P(EliminateDeadConstantTest, Custom) {
  auto& tc = GetParam();
  AssemblyBuilder builder;
  builder.AppendTypesConstantsGlobals(CommonTypes)
      .AppendTypesConstantsGlobals(tc.used_consts)
      .AppendInMain(tc.main_insts);
  const std::string expected = builder.GetCode();
  builder.AppendTypesConstantsGlobals(tc.dead_consts);
  const std::string assembly_with_dead_const = builder.GetCode();
  SinglePassRunAndCheck<EliminateDeadConstantPass>(
      assembly_with_dead_const, expected, /*  skip_nop = */ true);
}

INSTANTIATE_TEST_SUITE_P(
    ScalarTypeConstants, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // Scalar type constants, one dead constant and one used constant.
        {
            /* .used_consts = */
            {
                "%used_const_int = OpConstant %int 1",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Function",
                "OpStore %int_var %used_const_int",
            },
            /* .dead_consts = */
            {
                "%dead_const_int = OpConstant %int 1",
            },
        },
        {
            /* .used_consts = */
            {
                "%used_const_uint = OpConstant %uint 1",
            },
            /* .main_insts = */
            {
                "%uint_var = OpVariable %_pf_uint Function",
                "OpStore %uint_var %used_const_uint",
            },
            /* .dead_consts = */
            {
                "%dead_const_uint = OpConstant %uint 1",
            },
        },
        {
            /* .used_consts = */
            {
                "%used_const_float = OpConstant %float 3.1415",
            },
            /* .main_insts = */
            {
                "%float_var = OpVariable %_pf_float Function",
                "OpStore %float_var %used_const_float",
            },
            /* .dead_consts = */
            {
                "%dead_const_float = OpConstant %float 3.1415",
            },
        },
        {
            /* .used_consts = */
            {
                "%used_const_double = OpConstant %double 3.141592653",
            },
            /* .main_insts = */
            {
                "%double_var = OpVariable %_pf_double Function",
                "OpStore %double_var %used_const_double",
            },
            /* .dead_consts = */
            {
                "%dead_const_double = OpConstant %double 3.141592653",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    VectorTypeConstants, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // Tests eliminating dead constant type ivec2. One dead constant vector
        // and one used constant vector, each built from its own group of
        // scalar constants.
        {
            /* .used_consts = */
            {
                "%used_int_x = OpConstant %int 1",
                "%used_int_y = OpConstant %int 2",
                "%used_v2int = OpConstantComposite %v2int %used_int_x %used_int_y",
            },
            /* .main_insts = */
            {
                "%v2int_var = OpVariable %_pf_v2int Function",
                "OpStore %v2int_var %used_v2int",
            },
            /* .dead_consts = */
            {
                "%dead_int_x = OpConstant %int 1",
                "%dead_int_y = OpConstant %int 2",
                "%dead_v2int = OpConstantComposite %v2int %dead_int_x %dead_int_y",
            },
        },
        // Tests eliminating dead constant ivec2. One dead constant vector and
        // one used constant vector. But both built from a same group of
        // scalar constants.
        {
            /* .used_consts = */
            {
                "%used_int_x = OpConstant %int 1",
                "%used_int_y = OpConstant %int 2",
                "%used_int_z = OpConstant %int 3",
                "%used_v3int = OpConstantComposite %v3int %used_int_x %used_int_y %used_int_z",
            },
            /* .main_insts = */
            {
                "%v3int_var = OpVariable %_pf_v3int Function",
                "OpStore %v3int_var %used_v3int",
            },
            /* .dead_consts = */
            {
                "%dead_v3int = OpConstantComposite %v3int %used_int_x %used_int_y %used_int_z",
            },
        },
        // Tests eliminating dead cosntant vec2. One dead constant vector and
        // one used constant vector. Each built from its own group of scalar
        // constants.
        {
            /* .used_consts = */
            {
                "%used_float_x = OpConstant %float 3.1415",
                "%used_float_y = OpConstant %float 4.25",
                "%used_v2float = OpConstantComposite %v2float %used_float_x %used_float_y",
            },
            /* .main_insts = */
            {
                "%v2float_var = OpVariable %_pf_v2float Function",
                "OpStore %v2float_var %used_v2float",
            },
            /* .dead_consts = */
            {
                "%dead_float_x = OpConstant %float 3.1415",
                "%dead_float_y = OpConstant %float 4.25",
                "%dead_v2float = OpConstantComposite %v2float %dead_float_x %dead_float_y",
            },
        },
        // Tests eliminating dead cosntant vec2. One dead constant vector and
        // one used constant vector. Both built from a same group of scalar
        // constants.
        {
            /* .used_consts = */
            {
                "%used_float_x = OpConstant %float 3.1415",
                "%used_float_y = OpConstant %float 4.25",
                "%used_float_z = OpConstant %float 4.75",
                "%used_v3float = OpConstantComposite %v3float %used_float_x %used_float_y %used_float_z",
            },
            /* .main_insts = */
            {
                "%v3float_var = OpVariable %_pf_v3float Function",
                "OpStore %v3float_var %used_v3float",
            },
            /* .dead_consts = */
            {
                "%dead_v3float = OpConstantComposite %v3float %used_float_x %used_float_y %used_float_z",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    StructTypeConstants, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // A plain struct type dead constants. All of its components are dead
        // constants too.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
                "%dead_bool = OpConstantTrue %bool",
                "%dead_int = OpConstant %int 1",
                "%dead_float = OpConstant %float 2.5",
                "%dead_double = OpConstant %double 3.14159265358979",
                "%dead_struct = OpConstantComposite %flat_struct %dead_bool %dead_int %dead_float %dead_double",
            },
        },
        // A plain struct type dead constants. Some of its components are dead
        // constants while others are not.
        {
            /* .used_consts = */
            {
                "%used_int = OpConstant %int 1",
                "%used_double = OpConstant %double 3.14159265358979",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Function",
                "OpStore %int_var %used_int",
                "%double_var = OpVariable %_pf_double Function",
                "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
                "%dead_bool = OpConstantTrue %bool",
                "%dead_float = OpConstant %float 2.5",
                "%dead_struct = OpConstantComposite %flat_struct %dead_bool %used_int %dead_float %used_double",
            },
        },
        // A nesting struct type dead constants. All components of both outer
        // and inner structs are dead and should be removed after dead constant
        // elimination.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
                "%dead_bool = OpConstantTrue %bool",
                "%dead_int = OpConstant %int 1",
                "%dead_float = OpConstant %float 2.5",
                "%dead_double = OpConstant %double 3.1415926535",
                "%dead_inner_struct = OpConstantComposite %inner_struct %dead_bool %dead_int %dead_float %dead_double",
                "%dead_int2 = OpConstant %int 2",
                "%dead_double2 = OpConstant %double 1.428571428514",
                "%dead_outer_struct = OpConstantComposite %outer_struct %dead_inner_struct %dead_int2 %dead_double2",
            },
        },
        // A nesting struct type dead constants. Some of its components are
        // dead constants while others are not.
        {
            /* .used_consts = */
            {
                "%used_int = OpConstant %int 1",
                "%used_double = OpConstant %double 3.14159265358979",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Function",
                "OpStore %int_var %used_int",
                "%double_var = OpVariable %_pf_double Function",
                "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
                "%dead_bool = OpConstantTrue %bool",
                "%dead_float = OpConstant %float 2.5",
                "%dead_inner_struct = OpConstantComposite %inner_struct %dead_bool %used_int %dead_float %used_double",
                "%dead_int = OpConstant %int 2",
                "%dead_outer_struct = OpConstantComposite %outer_struct %dead_inner_struct %dead_int %used_double",
            },
        },
        // A nesting struct case. The inner struct is used while the outer struct is not
        {
          /* .used_const = */
          {
            "%used_bool = OpConstantTrue %bool",
            "%used_int = OpConstant %int 1",
            "%used_float = OpConstant %float 1.25",
            "%used_double = OpConstant %double 1.23456789012345",
            "%used_inner_struct = OpConstantComposite %inner_struct %used_bool %used_int %used_float %used_double",
          },
          /* .main_insts = */
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "%bool_from_inner_struct = OpCompositeExtract %bool %used_inner_struct 0",
            "OpStore %bool_var %bool_from_inner_struct",
          },
          /* .dead_consts = */
          {
            "%dead_int = OpConstant %int 2",
            "%dead_outer_struct = OpConstantComposite %outer_struct %used_inner_struct %dead_int %used_double"
          },
        },
        // A nesting struct case. The outer struct is used, so the inner struct should not
        // be removed even though it is not used anywhere.
        {
          /* .used_const = */
          {
            "%used_bool = OpConstantTrue %bool",
            "%used_int = OpConstant %int 1",
            "%used_float = OpConstant %float 1.25",
            "%used_double = OpConstant %double 1.23456789012345",
            "%used_inner_struct = OpConstantComposite %inner_struct %used_bool %used_int %used_float %used_double",
            "%used_outer_struct = OpConstantComposite %outer_struct %used_inner_struct %used_int %used_double"
          },
          /* .main_insts = */
          {
            "%int_var = OpVariable %_pf_int Function",
            "%int_from_outer_struct = OpCompositeExtract %int %used_outer_struct 1",
            "OpStore %int_var %int_from_outer_struct",
          },
          /* .dead_consts = */ {},
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    ScalarTypeSpecConstants, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // All scalar type spec constants.
        {
            /* .used_consts = */
            {
                "%used_bool = OpSpecConstantTrue %bool",
                "%used_uint = OpSpecConstant %uint 2",
                "%used_int = OpSpecConstant %int 2",
                "%used_float = OpSpecConstant %float 2.5",
                "%used_double = OpSpecConstant %double 1.42857142851",
            },
            /* .main_insts = */
            {
                "%bool_var = OpVariable %_pf_bool Function",
                "%uint_var = OpVariable %_pf_uint Function",
                "%int_var = OpVariable %_pf_int Function",
                "%float_var = OpVariable %_pf_float Function",
                "%double_var = OpVariable %_pf_double Function",
                "OpStore %bool_var %used_bool", "OpStore %uint_var %used_uint",
                "OpStore %int_var %used_int", "OpStore %float_var %used_float",
                "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
                "%dead_bool = OpSpecConstantTrue %bool",
                "%dead_uint = OpSpecConstant %uint 2",
                "%dead_int = OpSpecConstant %int 2",
                "%dead_float = OpSpecConstant %float 2.5",
                "%dead_double = OpSpecConstant %double 1.42857142851",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    VectorTypeSpecConstants, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // Bool vector type spec constants. One vector has all component dead,
        // another vector has one dead boolean and one used boolean.
        {
            /* .used_consts = */
            {
                "%used_bool = OpSpecConstantTrue %bool",
            },
            /* .main_insts = */
            {
                "%bool_var = OpVariable %_pf_bool Function",
                "OpStore %bool_var %used_bool",
            },
            /* .dead_consts = */
            {
                "%dead_bool = OpSpecConstantFalse %bool",
                "%dead_bool_vec1 = OpSpecConstantComposite %v2bool %dead_bool %dead_bool",
                "%dead_bool_vec2 = OpSpecConstantComposite %v2bool %dead_bool %used_bool",
            },
        },

        // Uint vector type spec constants. One vector has all component dead,
        // another vector has one dead unsigend integer and one used unsigned
        // integer.
        {
            /* .used_consts = */
            {
                "%used_uint = OpSpecConstant %uint 3",
            },
            /* .main_insts = */
            {
                "%uint_var = OpVariable %_pf_uint Function",
                "OpStore %uint_var %used_uint",
            },
            /* .dead_consts = */
            {
                "%dead_uint = OpSpecConstant %uint 1",
                "%dead_uint_vec1 = OpSpecConstantComposite %v2uint %dead_uint %dead_uint",
                "%dead_uint_vec2 = OpSpecConstantComposite %v2uint %dead_uint %used_uint",
            },
        },

        // Int vector type spec constants. One vector has all component dead,
        // another vector has one dead integer and one used integer.
        {
            /* .used_consts = */
            {
                "%used_int = OpSpecConstant %int 3",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Function",
                "OpStore %int_var %used_int",
            },
            /* .dead_consts = */
            {
                "%dead_int = OpSpecConstant %int 1",
                "%dead_int_vec1 = OpSpecConstantComposite %v2int %dead_int %dead_int",
                "%dead_int_vec2 = OpSpecConstantComposite %v2int %dead_int %used_int",
            },
        },

        // Int vector type spec constants built with both spec constants and
        // front-end constants.
        {
            /* .used_consts = */
            {
                "%used_spec_int = OpSpecConstant %int 3",
                "%used_front_end_int = OpConstant %int 3",
            },
            /* .main_insts = */
            {
                "%int_var1 = OpVariable %_pf_int Function",
                "OpStore %int_var1 %used_spec_int",
                "%int_var2 = OpVariable %_pf_int Function",
                "OpStore %int_var2 %used_front_end_int",
            },
            /* .dead_consts = */
            {
                "%dead_spec_int = OpSpecConstant %int 1",
                "%dead_front_end_int = OpConstant %int 1",
                // Dead front-end and dead spec constants
                "%dead_int_vec1 = OpSpecConstantComposite %v2int %dead_spec_int %dead_front_end_int",
                // Used front-end and dead spec constants
                "%dead_int_vec2 = OpSpecConstantComposite %v2int %dead_spec_int %used_front_end_int",
                // Dead front-end and used spec constants
                "%dead_int_vec3 = OpSpecConstantComposite %v2int %dead_front_end_int %used_spec_int",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    SpecConstantOp, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // Cast operations: uint <-> int <-> bool
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
                // Assistant constants, only used in dead spec constant
                // operations.
                "%signed_zero = OpConstant %int 0",
                "%signed_zero_vec = OpConstantComposite %v2int %signed_zero %signed_zero",
                "%unsigned_zero = OpConstant %uint 0",
                "%unsigned_zero_vec = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
                "%signed_one = OpConstant %int 1",
                "%signed_one_vec = OpConstantComposite %v2int %signed_one %signed_one",
                "%unsigned_one = OpConstant %uint 1",
                "%unsigned_one_vec = OpConstantComposite %v2uint %unsigned_one %unsigned_one",

                // Spec constants that support casting to each other.
                "%dead_bool = OpSpecConstantTrue %bool",
                "%dead_uint = OpSpecConstant %uint 1",
                "%dead_int = OpSpecConstant %int 2",
                "%dead_bool_vec = OpSpecConstantComposite %v2bool %dead_bool %dead_bool",
                "%dead_uint_vec = OpSpecConstantComposite %v2uint %dead_uint %dead_uint",
                "%dead_int_vec = OpSpecConstantComposite %v2int %dead_int %dead_int",

                // Scalar cast to boolean spec constant.
                "%int_to_bool = OpSpecConstantOp %bool INotEqual %dead_int %signed_zero",
                "%uint_to_bool = OpSpecConstantOp %bool INotEqual %dead_uint %unsigned_zero",

                // Vector cast to boolean spec constant.
                "%int_to_bool_vec = OpSpecConstantOp %v2bool INotEqual %dead_int_vec %signed_zero_vec",
                "%uint_to_bool_vec = OpSpecConstantOp %v2bool INotEqual %dead_uint_vec %unsigned_zero_vec",

                // Scalar cast to int spec constant.
                "%bool_to_int = OpSpecConstantOp %int Select %dead_bool %signed_one %signed_zero",
                "%uint_to_int = OpSpecConstantOp %uint IAdd %dead_uint %unsigned_zero",

                // Vector cast to int spec constant.
                "%bool_to_int_vec = OpSpecConstantOp %v2int Select %dead_bool_vec %signed_one_vec %signed_zero_vec",
                "%uint_to_int_vec = OpSpecConstantOp %v2uint IAdd %dead_uint_vec %unsigned_zero_vec",

                // Scalar cast to uint spec constant.
                "%bool_to_uint = OpSpecConstantOp %uint Select %dead_bool %unsigned_one %unsigned_zero",
                "%int_to_uint_vec = OpSpecConstantOp %uint IAdd %dead_int %signed_zero",

                // Vector cast to uint spec constant.
                "%bool_to_uint_vec = OpSpecConstantOp %v2uint Select %dead_bool_vec %unsigned_one_vec %unsigned_zero_vec",
                "%int_to_uint = OpSpecConstantOp %v2uint IAdd %dead_int_vec %signed_zero_vec",
            },
        },

        // Add, sub, mul, div, rem.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
                "%dead_spec_int_a = OpSpecConstant %int 1",
                "%dead_spec_int_a_vec = OpSpecConstantComposite %v2int %dead_spec_int_a %dead_spec_int_a",

                "%dead_spec_int_b = OpSpecConstant %int 2",
                "%dead_spec_int_b_vec = OpSpecConstantComposite %v2int %dead_spec_int_b %dead_spec_int_b",

                "%dead_const_int_c = OpConstant %int 3",
                "%dead_const_int_c_vec = OpConstantComposite %v2int %dead_const_int_c %dead_const_int_c",

                // Add
                "%add_a_b = OpSpecConstantOp %int IAdd %dead_spec_int_a %dead_spec_int_b",
                "%add_a_b_vec = OpSpecConstantOp %v2int IAdd %dead_spec_int_a_vec %dead_spec_int_b_vec",

                // Sub
                "%sub_a_b = OpSpecConstantOp %int ISub %dead_spec_int_a %dead_spec_int_b",
                "%sub_a_b_vec = OpSpecConstantOp %v2int ISub %dead_spec_int_a_vec %dead_spec_int_b_vec",

                // Mul
                "%mul_a_b = OpSpecConstantOp %int IMul %dead_spec_int_a %dead_spec_int_b",
                "%mul_a_b_vec = OpSpecConstantOp %v2int IMul %dead_spec_int_a_vec %dead_spec_int_b_vec",

                // Div
                "%div_a_b = OpSpecConstantOp %int SDiv %dead_spec_int_a %dead_spec_int_b",
                "%div_a_b_vec = OpSpecConstantOp %v2int SDiv %dead_spec_int_a_vec %dead_spec_int_b_vec",

                // Bitwise Xor
                "%xor_a_b = OpSpecConstantOp %int BitwiseXor %dead_spec_int_a %dead_spec_int_b",
                "%xor_a_b_vec = OpSpecConstantOp %v2int BitwiseXor %dead_spec_int_a_vec %dead_spec_int_b_vec",

                // Scalar Comparison
                "%less_a_b = OpSpecConstantOp %bool SLessThan %dead_spec_int_a %dead_spec_int_b",
            },
        },

        // Vectors without used swizzles should be removed.
        {
            /* .used_consts = */
            {
                "%used_int = OpConstant %int 3",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Function",
                "OpStore %int_var %used_int",
            },
            /* .dead_consts = */
            {
                "%dead_int = OpConstant %int 3",

                "%dead_spec_int_a = OpSpecConstant %int 1",
                "%vec_a = OpSpecConstantComposite %v4int %dead_spec_int_a %dead_spec_int_a %dead_int %dead_int",

                "%dead_spec_int_b = OpSpecConstant %int 2",
                "%vec_b = OpSpecConstantComposite %v4int %dead_spec_int_b %dead_spec_int_b %used_int %used_int",

                // Extract scalar
                "%a_x = OpSpecConstantOp %int CompositeExtract %vec_a 0",
                "%b_x = OpSpecConstantOp %int CompositeExtract %vec_b 0",

                // Extract vector
                "%a_xy = OpSpecConstantOp %v2int VectorShuffle %vec_a %vec_a 0 1",
                "%b_xy = OpSpecConstantOp %v2int VectorShuffle %vec_b %vec_b 0 1",
            },
        },
        // Vectors with used swizzles should not be removed.
        {
            /* .used_consts = */
            {
                "%used_int = OpConstant %int 3",
                "%used_spec_int_a = OpSpecConstant %int 1",
                "%used_spec_int_b = OpSpecConstant %int 2",
                // Create vectors
                "%vec_a = OpSpecConstantComposite %v4int %used_spec_int_a %used_spec_int_a %used_int %used_int",
                "%vec_b = OpSpecConstantComposite %v4int %used_spec_int_b %used_spec_int_b %used_int %used_int",
                // Extract vector
                "%a_xy = OpSpecConstantOp %v2int VectorShuffle %vec_a %vec_a 0 1",
                "%b_xy = OpSpecConstantOp %v2int VectorShuffle %vec_b %vec_b 0 1",
            },
            /* .main_insts = */
            {
                "%v2int_var_a = OpVariable %_pf_v2int Function",
                "%v2int_var_b = OpVariable %_pf_v2int Function",
                "OpStore %v2int_var_a %a_xy",
                "OpStore %v2int_var_b %b_xy",
            },
            /* .dead_consts = */ {},
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    LongDefUseChain, EliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<EliminateDeadConstantTestCase>({
        // clang-format off
        // Long Def-Use chain with binary operations.
        {
            /* .used_consts = */
            {
              "%array_size = OpConstant %int 4",
              "%type_arr_int_4 = OpTypeArray %int %array_size",
              "%used_int_0 = OpConstant %int 100",
              "%used_int_1 = OpConstant %int 1",
              "%used_int_2 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_1",
              "%used_int_3 = OpSpecConstantOp %int ISub %used_int_0 %used_int_2",
              "%used_int_4 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_3",
              "%used_int_5 = OpSpecConstantOp %int ISub %used_int_0 %used_int_4",
              "%used_int_6 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_5",
              "%used_int_7 = OpSpecConstantOp %int ISub %used_int_0 %used_int_6",
              "%used_int_8 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_7",
              "%used_int_9 = OpSpecConstantOp %int ISub %used_int_0 %used_int_8",
              "%used_int_10 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_9",
              "%used_int_11 = OpSpecConstantOp %int ISub %used_int_0 %used_int_10",
              "%used_int_12 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_11",
              "%used_int_13 = OpSpecConstantOp %int ISub %used_int_0 %used_int_12",
              "%used_int_14 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_13",
              "%used_int_15 = OpSpecConstantOp %int ISub %used_int_0 %used_int_14",
              "%used_int_16 = OpSpecConstantOp %int ISub %used_int_0 %used_int_15",
              "%used_int_17 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_16",
              "%used_int_18 = OpSpecConstantOp %int ISub %used_int_0 %used_int_17",
              "%used_int_19 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_18",
              "%used_int_20 = OpSpecConstantOp %int ISub %used_int_0 %used_int_19",
              "%used_vec_a = OpSpecConstantComposite %v2int %used_int_18 %used_int_19",
              "%used_vec_b = OpSpecConstantOp %v2int IMul %used_vec_a %used_vec_a",
              "%used_int_21 = OpSpecConstantOp %int CompositeExtract %used_vec_b 0",
              "%used_array = OpConstantComposite %type_arr_int_4 %used_int_20 %used_int_20 %used_int_21 %used_int_21",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Function",
              "%used_array_2 = OpCompositeExtract %int %used_array 2",
              "OpStore %int_var %used_array_2",
            },
            /* .dead_consts = */
            {
              "%dead_int_1 = OpConstant %int 2",
              "%dead_int_2 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_1",
              "%dead_int_3 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_2",
              "%dead_int_4 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_3",
              "%dead_int_5 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_4",
              "%dead_int_6 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_5",
              "%dead_int_7 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_6",
              "%dead_int_8 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_7",
              "%dead_int_9 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_8",
              "%dead_int_10 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_9",
              "%dead_int_11 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_10",
              "%dead_int_12 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_11",
              "%dead_int_13 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_12",
              "%dead_int_14 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_13",
              "%dead_int_15 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_14",
              "%dead_int_16 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_15",
              "%dead_int_17 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_16",
              "%dead_int_18 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_17",
              "%dead_int_19 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_18",
              "%dead_int_20 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_19",
              "%dead_vec_a = OpSpecConstantComposite %v2int %dead_int_18 %dead_int_19",
              "%dead_vec_b = OpSpecConstantOp %v2int IMul %dead_vec_a %dead_vec_a",
              "%dead_int_21 = OpSpecConstantOp %int CompositeExtract %dead_vec_b 0",
              "%dead_array = OpConstantComposite %type_arr_int_4 %dead_int_20 %used_int_20 %dead_int_19 %used_int_19",
            },
        },
        // Long Def-Use chain with swizzle
        // clang-format on
    })));

}  // namespace
}  // namespace opt
}  // namespace spvtools
