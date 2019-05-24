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
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

// Returns the types defining instructions commonly used in many tests.
std::vector<std::string> CommonTypes() {
  return std::vector<std::string>{
      // clang-format off
    // scalar types
    "%bool = OpTypeBool",
    "%uint = OpTypeInt 32 0",
    "%int = OpTypeInt 32 1",
    "%uint64 = OpTypeInt 64 0",
    "%int64 = OpTypeInt 64 1",
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
    // struct types
    "%inner_struct = OpTypeStruct %bool %float",
    "%outer_struct = OpTypeStruct %inner_struct %int %double",
    "%flat_struct = OpTypeStruct %bool %int %float %double",
    // variable pointer types
    "%_pf_bool = OpTypePointer Function %bool",
    "%_pf_uint = OpTypePointer Function %uint",
    "%_pf_int = OpTypePointer Function %int",
    "%_pf_uint64 = OpTypePointer Function %uint64",
    "%_pf_int64 = OpTypePointer Function %int64",
    "%_pf_float = OpTypePointer Function %float",
    "%_pf_double = OpTypePointer Function %double",
    "%_pf_v2int = OpTypePointer Function %v2int",
    "%_pf_v3int = OpTypePointer Function %v3int",
    "%_pf_v4int = OpTypePointer Function %v4int",
    "%_pf_v2float = OpTypePointer Function %v2float",
    "%_pf_v3float = OpTypePointer Function %v3float",
    "%_pf_v2double = OpTypePointer Function %v2double",
    "%_pf_inner_struct = OpTypePointer Function %inner_struct",
    "%_pf_outer_struct = OpTypePointer Function %outer_struct",
    "%_pf_flat_struct = OpTypePointer Function %flat_struct",
      // clang-format on
  };
}

// A helper function to strip OpName instructions from the given string of
// disassembly code and put those debug instructions to a set. Returns the
// string with all OpName instruction stripped and a set of OpName
// instructions.
std::tuple<std::string, std::unordered_set<std::string>>
StripOpNameInstructionsToSet(const std::string& str) {
  std::stringstream ss(str);
  std::ostringstream oss;
  std::string inst_str;
  std::unordered_set<std::string> opname_instructions;
  while (std::getline(ss, inst_str, '\n')) {
    if (inst_str.find("OpName %") == std::string::npos) {
      oss << inst_str << '\n';
    } else {
      opname_instructions.insert(inst_str);
    }
  }
  return std::make_tuple(oss.str(), std::move(opname_instructions));
}

// The test fixture for all tests of UnifyConstantPass. This fixture defines
// the rule of checking: all the optimized code should be exactly the same as
// the expected code, except the OpName instructions, which can be different in
// order.
template <typename T>
class UnifyConstantTest : public PassTest<T> {
 protected:
  // Runs UnifyConstantPass on the code built from the given |test_builder|,
  // and checks whether the optimization result matches with the code built
  // from |expected_builder|.
  void Check(const AssemblyBuilder& expected_builder,
             const AssemblyBuilder& test_builder) {
    // unoptimized code
    const std::string original_before_strip = test_builder.GetCode();
    std::string original_without_opnames;
    std::unordered_set<std::string> original_opnames;
    std::tie(original_without_opnames, original_opnames) =
        StripOpNameInstructionsToSet(original_before_strip);

    // expected code
    std::string expected_without_opnames;
    std::unordered_set<std::string> expected_opnames;
    std::tie(expected_without_opnames, expected_opnames) =
        StripOpNameInstructionsToSet(expected_builder.GetCode());

    // optimized code
    std::string optimized_before_strip;
    auto status = Pass::Status::SuccessWithoutChange;
    std::tie(optimized_before_strip, status) =
        this->template SinglePassRunAndDisassemble<UnifyConstantPass>(
            test_builder.GetCode(),
            /* skip_nop = */ true, /* do_validation = */ false);
    std::string optimized_without_opnames;
    std::unordered_set<std::string> optimized_opnames;
    std::tie(optimized_without_opnames, optimized_opnames) =
        StripOpNameInstructionsToSet(optimized_before_strip);

    // Flag "status" should be returned correctly.
    EXPECT_NE(Pass::Status::Failure, status);
    EXPECT_EQ(expected_without_opnames == original_without_opnames,
              status == Pass::Status::SuccessWithoutChange);
    // Code except OpName instructions should be exactly the same.
    EXPECT_EQ(expected_without_opnames, optimized_without_opnames);
    // OpName instructions can be in different order, but the content must be
    // the same.
    EXPECT_EQ(expected_opnames, optimized_opnames);
  }
};

using UnifyFrontEndConstantSingleTest =
    UnifyConstantTest<PassTest<::testing::Test>>;

TEST_F(UnifyFrontEndConstantSingleTest, Basic) {
  AssemblyBuilder test_builder;
  AssemblyBuilder expected_builder;

  test_builder
      .AppendTypesConstantsGlobals({
          "%uint = OpTypeInt 32 0", "%_pf_uint = OpTypePointer Function %uint",
          "%unsigned_1 = OpConstant %uint 1",
          "%unsigned_1_duplicate = OpConstant %uint 1",  // duplicated constant
      })
      .AppendInMain({
          "%uint_var = OpVariable %_pf_uint Function",
          "OpStore %uint_var %unsigned_1_duplicate",
      });

  expected_builder
      .AppendTypesConstantsGlobals({
          "%uint = OpTypeInt 32 0",
          "%_pf_uint = OpTypePointer Function %uint",
          "%unsigned_1 = OpConstant %uint 1",
      })
      .AppendInMain({
          "%uint_var = OpVariable %_pf_uint Function",
          "OpStore %uint_var %unsigned_1",
      })
      .AppendNames({
          "OpName %unsigned_1 \"unsigned_1_duplicate\"",  // the OpName
                                                          // instruction of the
                                                          // removed duplicated
                                                          // constant won't be
                                                          // erased.
      });
  Check(expected_builder, test_builder);
}

TEST_F(UnifyFrontEndConstantSingleTest, SkipWhenResultIdHasDecorations) {
  AssemblyBuilder test_builder;
  AssemblyBuilder expected_builder;

  test_builder
      .AppendAnnotations({
          // So far we don't have valid decorations for constants. This is
          // preparing for the future updates of SPIR-V.
          // TODO(qining): change to a valid decoration once they are available.
          "OpDecorate %f_1 RelaxedPrecision",
          "OpDecorate %f_2_dup RelaxedPrecision",
      })
      .AppendTypesConstantsGlobals({
          // clang-format off
          "%float = OpTypeFloat 32",
          "%_pf_float = OpTypePointer Function %float",
          "%f_1 = OpConstant %float 1",
          // %f_1 has decoration, so %f_1 will not be used to replace %f_1_dup.
          "%f_1_dup = OpConstant %float 1",
          "%f_2 = OpConstant %float 2",
          // %_2_dup has decoration, so %f_2 will not replace %f_2_dup.
          "%f_2_dup = OpConstant %float 2",
          // no decoration for %f_3 or %f_3_dup, %f_3_dup should be replaced.
          "%f_3 = OpConstant %float 3",
          "%f_3_dup = OpConstant %float 3",
          // clang-format on
      })
      .AppendInMain({
          // clang-format off
          "%f_var = OpVariable %_pf_float Function",
          "OpStore %f_var %f_1_dup",
          "OpStore %f_var %f_2_dup",
          "OpStore %f_var %f_3_dup",
          // clang-format on
      });

  expected_builder
      .AppendAnnotations({
          "OpDecorate %f_1 RelaxedPrecision",
          "OpDecorate %f_2_dup RelaxedPrecision",
      })
      .AppendTypesConstantsGlobals({
          // clang-format off
          "%float = OpTypeFloat 32",
          "%_pf_float = OpTypePointer Function %float",
          "%f_1 = OpConstant %float 1",
          "%f_1_dup = OpConstant %float 1",
          "%f_2 = OpConstant %float 2",
          "%f_2_dup = OpConstant %float 2",
          "%f_3 = OpConstant %float 3",
          // clang-format on
      })
      .AppendInMain({
          // clang-format off
          "%f_var = OpVariable %_pf_float Function",
          "OpStore %f_var %f_1_dup",
          "OpStore %f_var %f_2_dup",
          "OpStore %f_var %f_3",
          // clang-format on
      })
      .AppendNames({
          "OpName %f_3 \"f_3_dup\"",
      });

  Check(expected_builder, test_builder);
}

TEST_F(UnifyFrontEndConstantSingleTest, UnifyWithDecorationOnTypes) {
  AssemblyBuilder test_builder;
  AssemblyBuilder expected_builder;

  test_builder
      .AppendAnnotations({
          "OpMemberDecorate %flat_d 1 RelaxedPrecision",
      })
      .AppendTypesConstantsGlobals({
          // clang-format off
          "%int = OpTypeInt 32 1",
          "%float = OpTypeFloat 32",
          "%flat = OpTypeStruct %int %float",
          "%_pf_flat = OpTypePointer Function %flat",
          // decorated flat struct
          "%flat_d = OpTypeStruct %int %float",
          "%_pf_flat_d = OpTypePointer Function %flat_d",
          // perserved contants. %flat_1 and %flat_d has same members, but
          // their type are different in decorations, so they should not be
          // used to replace each other.
          "%int_1 = OpConstant %int 1",
          "%float_1 = OpConstant %float 1",
          "%flat_1 = OpConstantComposite %flat %int_1 %float_1",
          "%flat_d_1 = OpConstantComposite %flat_d %int_1 %float_1",
          // duplicated constants.
          "%flat_1_dup = OpConstantComposite %flat %int_1 %float_1",
          "%flat_d_1_dup = OpConstantComposite %flat_d %int_1 %float_1",
          // clang-format on
      })
      .AppendInMain({
          "%flat_var = OpVariable %_pf_flat Function",
          "OpStore %flat_var %flat_1_dup",
          "%flat_d_var = OpVariable %_pf_flat_d Function",
          "OpStore %flat_d_var %flat_d_1_dup",
      });

  expected_builder
      .AppendAnnotations({
          "OpMemberDecorate %flat_d 1 RelaxedPrecision",
      })
      .AppendTypesConstantsGlobals({
          // clang-format off
          "%int = OpTypeInt 32 1",
          "%float = OpTypeFloat 32",
          "%flat = OpTypeStruct %int %float",
          "%_pf_flat = OpTypePointer Function %flat",
          // decorated flat struct
          "%flat_d = OpTypeStruct %int %float",
          "%_pf_flat_d = OpTypePointer Function %flat_d",
          "%int_1 = OpConstant %int 1",
          "%float_1 = OpConstant %float 1",
          "%flat_1 = OpConstantComposite %flat %int_1 %float_1",
          "%flat_d_1 = OpConstantComposite %flat_d %int_1 %float_1",
          // clang-format on
      })
      .AppendInMain({
          "%flat_var = OpVariable %_pf_flat Function",
          "OpStore %flat_var %flat_1",
          "%flat_d_var = OpVariable %_pf_flat_d Function",
          "OpStore %flat_d_var %flat_d_1",
      })
      .AppendNames({
          "OpName %flat_1 \"flat_1_dup\"",
          "OpName %flat_d_1 \"flat_d_1_dup\"",
      });

  Check(expected_builder, test_builder);
}

struct UnifyConstantTestCase {
  // preserved constants.
  std::vector<std::string> preserved_consts;
  // expected uses of the preserved constants.
  std::vector<std::string> use_preserved_consts;
  // duplicated constants of the preserved constants.
  std::vector<std::string> duplicate_consts;
  // uses of the duplicated constants, expected to be updated to use the
  // preserved constants.
  std::vector<std::string> use_duplicate_consts;
  // The updated OpName instructions that originally refer to duplicated
  // constants.
  std::vector<std::string> remapped_names;
};

using UnifyFrontEndConstantParamTest = UnifyConstantTest<
    PassTest<::testing::TestWithParam<UnifyConstantTestCase>>>;

TEST_P(UnifyFrontEndConstantParamTest, TestCase) {
  auto& tc = GetParam();
  AssemblyBuilder test_builder;
  AssemblyBuilder expected_builder;
  test_builder.AppendTypesConstantsGlobals(CommonTypes());
  expected_builder.AppendTypesConstantsGlobals(CommonTypes());

  test_builder.AppendTypesConstantsGlobals(tc.preserved_consts)
      .AppendTypesConstantsGlobals(tc.duplicate_consts)
      .AppendInMain(tc.use_duplicate_consts);

  // Duplicated constants are killed in the expected output, and the debug
  // instructions attached to those duplicated instructions will be migrated to
  // the corresponding preserved constants.
  expected_builder.AppendTypesConstantsGlobals(tc.preserved_consts)
      .AppendInMain(tc.use_preserved_consts)
      .AppendNames(tc.remapped_names);

  Check(expected_builder, test_builder);
}

INSTANTIATE_TEST_SUITE_P(
    Case, UnifyFrontEndConstantParamTest,
    ::
        testing::
            ValuesIn(
                std::
                    vector<UnifyConstantTestCase>(
                        {
                            // clang-format off
        // basic tests for scalar constants
        {
          // preserved constants
          {
            "%bool_true = OpConstantTrue %bool",
            "%signed_1 = OpConstant %int 1",
            "%signed_minus_1 = OpConstant %int64 -1",
            "%unsigned_max = OpConstant %uint64 18446744073709551615",
            "%float_1 = OpConstant %float 1",
            "%double_1 = OpConstant %double 1",
          },
          // use preserved constants in main
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %bool_true",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_1",
            "%int64_var = OpVariable %_pf_int64 Function",
            "OpStore %int64_var %signed_minus_1",
            "%uint64_var = OpVariable %_pf_uint64 Function",
            "OpStore %uint64_var %unsigned_max",
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_1",
            "%double_var = OpVariable %_pf_double Function",
            "OpStore %double_var %double_1",
          },
          // duplicated constants
          {
            "%bool_true_duplicate = OpConstantTrue %bool",
            "%signed_1_duplicate = OpConstant %int 1",
            "%signed_minus_1_duplicate = OpConstant %int64 -1",
            "%unsigned_max_duplicate = OpConstant %uint64 18446744073709551615",
            "%float_1_duplicate = OpConstant %float 1",
            "%double_1_duplicate = OpConstant %double 1",
          },
          // use duplicated constants in main
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %bool_true_duplicate",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_1_duplicate",
            "%int64_var = OpVariable %_pf_int64 Function",
            "OpStore %int64_var %signed_minus_1_duplicate",
            "%uint64_var = OpVariable %_pf_uint64 Function",
            "OpStore %uint64_var %unsigned_max_duplicate",
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_1_duplicate",
            "%double_var = OpVariable %_pf_double Function",
            "OpStore %double_var %double_1_duplicate",
          },
          // remapped names
          {
            "OpName %bool_true \"bool_true_duplicate\"",
            "OpName %signed_1 \"signed_1_duplicate\"",
            "OpName %signed_minus_1 \"signed_minus_1_duplicate\"",
            "OpName %unsigned_max \"unsigned_max_duplicate\"",
            "OpName %float_1 \"float_1_duplicate\"",
            "OpName %double_1 \"double_1_duplicate\"",
          },
        },
        // NaN in different bit patterns should not be unified, but the ones
        // using same bit pattern should be unified.
        {
          // preserved constants
          {
            "%float_nan_1 = OpConstant %float 0x1.8p+128", // !2143289344, 7FC00000
            "%float_nan_2 = OpConstant %float 0x1.800002p+128",// !2143289345 7FC00001
          },
          // use preserved constants in main
          {
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_nan_1",
            "OpStore %float_var %float_nan_2",
          },
          // duplicated constants
          {
            "%float_nan_1_duplicate = OpConstant %float 0x1.8p+128", // !2143289344, 7FC00000
            "%float_nan_2_duplicate = OpConstant %float 0x1.800002p+128",// !2143289345, 7FC00001
          },
          // use duplicated constants in main
          {
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_nan_1_duplicate",
            "OpStore %float_var %float_nan_2_duplicate",
          },
          // remapped names
          {
            "OpName %float_nan_1 \"float_nan_1_duplicate\"",
            "OpName %float_nan_2 \"float_nan_2_duplicate\"",
          },
        },
        // null values
        {
          // preserved constants
          {
            "%bool_null = OpConstantNull %bool",
            "%signed_null = OpConstantNull %int",
            "%signed_64_null = OpConstantNull %int64",
            "%float_null = OpConstantNull %float",
            "%double_null = OpConstantNull %double",
            // zero-valued constants will not be unified with the equivalent
            // null constants.
            "%signed_zero = OpConstant %int 0",
          },
          // use preserved constants in main
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %bool_null",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_null",
            "%int64_var = OpVariable %_pf_int64 Function",
            "OpStore %int64_var %signed_64_null",
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_null",
            "%double_var = OpVariable %_pf_double Function",
            "OpStore %double_var %double_null",
          },
          // duplicated constants
          {
            "%bool_null_duplicate = OpConstantNull %bool",
            "%signed_null_duplicate = OpConstantNull %int",
            "%signed_64_null_duplicate = OpConstantNull %int64",
            "%float_null_duplicate = OpConstantNull %float",
            "%double_null_duplicate = OpConstantNull %double",
          },
          // use duplicated constants in main
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %bool_null_duplicate",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_null_duplicate",
            "%int64_var = OpVariable %_pf_int64 Function",
            "OpStore %int64_var %signed_64_null_duplicate",
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %float_null_duplicate",
            "%double_var = OpVariable %_pf_double Function",
            "OpStore %double_var %double_null_duplicate",
          },
          // remapped names
          {
            "OpName %bool_null \"bool_null_duplicate\"",
            "OpName %signed_null \"signed_null_duplicate\"",
            "OpName %signed_64_null \"signed_64_null_duplicate\"",
            "OpName %float_null \"float_null_duplicate\"",
            "OpName %double_null \"double_null_duplicate\"",
          },
        },
        // constant sampler
        {
          // preserved constants
          {
            "%sampler = OpTypeSampler",
            "%_pf_sampler = OpTypePointer Function %sampler",
            "%sampler_1 = OpConstantSampler %sampler Repeat 0 Linear",
          },
          // use preserved constants in main
          {
            "%sampler_var = OpVariable %_pf_sampler Function",
            "OpStore %sampler_var %sampler_1",
          },
          // duplicated constants
          {
            "%sampler_1_duplicate = OpConstantSampler %sampler Repeat 0 Linear",
          },
          // use duplicated constants in main
          {
            "%sampler_var = OpVariable %_pf_sampler Function",
            "OpStore %sampler_var %sampler_1_duplicate",
          },
          // remapped names
          {
            "OpName %sampler_1 \"sampler_1_duplicate\"",
          },
        },
        // duplicate vector built from same ids.
        {
          // preserved constants
          {
            "%signed_1 = OpConstant %int 1",
            "%signed_2 = OpConstant %int 2",
            "%signed_3 = OpConstant %int 3",
            "%signed_4 = OpConstant %int 4",
            "%vec = OpConstantComposite %v4int %signed_1 %signed_2 %signed_3 %signed_4",
          },
          // use preserved constants in main
          {
            "%vec_var = OpVariable %_pf_v4int Function",
            "OpStore %vec_var %vec",
          },
          // duplicated constants
          {
            "%vec_duplicate = OpConstantComposite %v4int %signed_1 %signed_2 %signed_3 %signed_4",
          },
          // use duplicated constants in main
          {
            "%vec_var = OpVariable %_pf_v4int Function",
            "OpStore %vec_var %vec_duplicate",
          },
          // remapped names
          {
            "OpName %vec \"vec_duplicate\"",
          }
        },
        // duplicate vector built from duplicated ids.
        {
          // preserved constants
          {
            "%signed_1 = OpConstant %int 1",
            "%signed_2 = OpConstant %int 2",
            "%signed_3 = OpConstant %int 3",
            "%signed_4 = OpConstant %int 4",
            "%vec = OpConstantComposite %v4int %signed_1 %signed_2 %signed_3 %signed_4",
          },
          // use preserved constants in main
          {
            "%vec_var = OpVariable %_pf_v4int Function",
            "OpStore %vec_var %vec",
          },
          // duplicated constants
          {
            "%signed_3_duplicate = OpConstant %int 3",
            "%signed_4_duplicate = OpConstant %int 4",
            "%vec_duplicate = OpConstantComposite %v4int %signed_1 %signed_2 %signed_3_duplicate %signed_4_duplicate",
          },
          // use duplicated constants in main
          {
            "%vec_var = OpVariable %_pf_v4int Function",
            "OpStore %vec_var %vec_duplicate",
          },
          // remapped names
          {
            "OpName %signed_3 \"signed_3_duplicate\"",
            "OpName %signed_4 \"signed_4_duplicate\"",
            "OpName %vec \"vec_duplicate\"",
          },
        },
        // flat struct
        {
          // preserved constants
          {
            "%bool_true = OpConstantTrue %bool",
            "%signed_1 = OpConstant %int 1",
            "%float_1 = OpConstant %float 1",
            "%double_1 = OpConstant %double 1",
            "%s = OpConstantComposite %flat_struct %bool_true %signed_1 %float_1 %double_1",
          },
          // use preserved constants in main
          {
            "%s_var = OpVariable %_pf_flat_struct Function",
            "OpStore %s_var %s",
          },
          // duplicated constants
          {
            "%float_1_duplicate = OpConstant %float 1",
            "%double_1_duplicate = OpConstant %double 1",
            "%s_duplicate = OpConstantComposite %flat_struct %bool_true %signed_1 %float_1_duplicate %double_1_duplicate",
          },
          // use duplicated constants in main
          {
            "%s_var = OpVariable %_pf_flat_struct Function",
            "OpStore %s_var %s_duplicate",
          },
          // remapped names
          {
            "OpName %float_1 \"float_1_duplicate\"",
            "OpName %double_1 \"double_1_duplicate\"",
            "OpName %s \"s_duplicate\"",
          },
        },
        // nested struct
        {
          // preserved constants
          {
            "%bool_true = OpConstantTrue %bool",
            "%signed_1 = OpConstant %int 1",
            "%float_1 = OpConstant %float 1",
            "%double_1 = OpConstant %double 1",
            "%inner = OpConstantComposite %inner_struct %bool_true %float_1",
            "%outer = OpConstantComposite %outer_struct %inner %signed_1 %double_1",
          },
          // use preserved constants in main
          {
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %outer",
          },
          // duplicated constants
          {
            "%float_1_duplicate = OpConstant %float 1",
            "%double_1_duplicate = OpConstant %double 1",
            "%inner_duplicate = OpConstantComposite %inner_struct %bool_true %float_1_duplicate",
            "%outer_duplicate = OpConstantComposite %outer_struct %inner_duplicate %signed_1 %double_1_duplicate",
          },
          // use duplicated constants in main
          {
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %outer_duplicate",
          },
          // remapped names
          {
            "OpName %float_1 \"float_1_duplicate\"",
            "OpName %double_1 \"double_1_duplicate\"",
            "OpName %inner \"inner_duplicate\"",
            "OpName %outer \"outer_duplicate\"",
          },
        },
        // composite type null constants. Null constants and zero-valued
        // constants should not be used to replace each other.
        {
          // preserved constants
          {
            "%bool_zero = OpConstantFalse %bool",
            "%float_zero = OpConstant %float 0",
            "%int_null = OpConstantNull %int",
            "%double_null = OpConstantNull %double",
            // inner_struct type null constant.
            "%null_inner = OpConstantNull %inner_struct",
            // zero-valued composite constant built from zero-valued constant
            // component. inner_zero should not be replace by null_inner.
            "%inner_zero = OpConstantComposite %inner_struct %bool_zero %float_zero",
            // zero-valued composite contant built from zero-valued constants
            // and null constants.
            "%outer_zero = OpConstantComposite %outer_struct %inner_zero %int_null %double_null",
            // outer_struct type null constant, it should not be replaced by
            // outer_zero.
            "%null_outer = OpConstantNull %outer_struct",
          },
          // use preserved constants in main
          {
            "%inner_var = OpVariable %_pf_inner_struct Function",
            "OpStore %inner_var %inner_zero",
            "OpStore %inner_var %null_inner",
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %outer_zero",
            "OpStore %outer_var %null_outer",
          },
          // duplicated constants
          {
            "%null_inner_dup = OpConstantNull %inner_struct",
            "%null_outer_dup = OpConstantNull %outer_struct",
            "%inner_zero_dup = OpConstantComposite %inner_struct %bool_zero %float_zero",
            "%outer_zero_dup = OpConstantComposite %outer_struct %inner_zero_dup %int_null %double_null",
          },
          // use duplicated constants in main
          {
            "%inner_var = OpVariable %_pf_inner_struct Function",
            "OpStore %inner_var %inner_zero_dup",
            "OpStore %inner_var %null_inner_dup",
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %outer_zero_dup",
            "OpStore %outer_var %null_outer_dup",
          },
          // remapped names
          {
            "OpName %null_inner \"null_inner_dup\"",
            "OpName %null_outer \"null_outer_dup\"",
            "OpName %inner_zero \"inner_zero_dup\"",
            "OpName %outer_zero \"outer_zero_dup\"",
          },
        },
        // Spec Constants with SpecId decoration should be skipped.
        {
          // preserved constants
          {
            // Assembly builder will add OpDecorate SpecId instruction for the
            // following spec constant instructions automatically.
            "%spec_bool_1 = OpSpecConstantTrue %bool",
            "%spec_bool_2 = OpSpecConstantTrue %bool",
            "%spec_int_1 = OpSpecConstant %int 1",
            "%spec_int_2 = OpSpecConstant %int 1",
          },
          // use preserved constants in main
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %spec_bool_1",
            "OpStore %bool_var %spec_bool_2",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %spec_int_1",
            "OpStore %int_var %spec_int_2",
          },
          // duplicated constants. No duplicated instruction to remove in this
          // case.
          {},
          // use duplicated constants in main. Same as the above 'use preserved
          // constants in main' defined above, as no instruction should be
          // removed in this case.
          {
            "%bool_var = OpVariable %_pf_bool Function",
            "OpStore %bool_var %spec_bool_1",
            "OpStore %bool_var %spec_bool_2",
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %spec_int_1",
            "OpStore %int_var %spec_int_2",
          },
          // remapped names. No duplicated instruction removed, so this is
          // empty.
          {}
        },
        // spec constant composite
        {
          // preserved constants
          {
            "%spec_bool_true = OpSpecConstantTrue %bool",
            "%spec_signed_1 = OpSpecConstant %int 1",
            "%float_1 = OpConstant %float 1",
            "%double_1 = OpConstant %double 1",
            "%spec_inner = OpSpecConstantComposite %inner_struct %spec_bool_true %float_1",
            "%spec_outer = OpSpecConstantComposite %outer_struct %spec_inner %spec_signed_1 %double_1",
            "%spec_vec2 = OpSpecConstantComposite %v2float %float_1 %float_1",
          },
          // use preserved constants in main
          {
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %spec_outer",
            "%v2float_var = OpVariable %_pf_v2float Function",
            "OpStore %v2float_var %spec_vec2",
          },
          // duplicated constants
          {
            "%float_1_duplicate = OpConstant %float 1",
            "%double_1_duplicate = OpConstant %double 1",
            "%spec_inner_duplicate = OpSpecConstantComposite %inner_struct %spec_bool_true %float_1_duplicate",
            "%spec_outer_duplicate = OpSpecConstantComposite %outer_struct %spec_inner_duplicate %spec_signed_1 %double_1_duplicate",
            "%spec_vec2_duplicate = OpSpecConstantComposite %v2float %float_1 %float_1_duplicate",
          },
          // use duplicated constants in main
          {
            "%outer_var = OpVariable %_pf_outer_struct Function",
            "OpStore %outer_var %spec_outer_duplicate",
            "%v2float_var = OpVariable %_pf_v2float Function",
            "OpStore %v2float_var %spec_vec2_duplicate",
          },
          // remapped names
          {
            "OpName %float_1 \"float_1_duplicate\"",
            "OpName %double_1 \"double_1_duplicate\"",
            "OpName %spec_inner \"spec_inner_duplicate\"",
            "OpName %spec_outer \"spec_outer_duplicate\"",
            "OpName %spec_vec2 \"spec_vec2_duplicate\"",
          },
        },
        // spec constant op with int scalar
        {
          // preserved constants
          {
            "%spec_signed_1 = OpSpecConstant %int 1",
            "%spec_signed_2 = OpSpecConstant %int 2",
            "%spec_signed_add = OpSpecConstantOp %int IAdd %spec_signed_1 %spec_signed_2",
          },
          // use preserved constants in main
          {
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %spec_signed_add",
          },
          // duplicated constants
          {
            "%spec_signed_add_duplicate = OpSpecConstantOp %int IAdd %spec_signed_1 %spec_signed_2",
          },
          // use duplicated contants in main
          {
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %spec_signed_add_duplicate",
          },
          // remapped names
          {
            "OpName %spec_signed_add \"spec_signed_add_duplicate\"",
          },
        },
        // spec constant op composite extract
        {
          // preserved constants
          {
            "%float_1 = OpConstant %float 1",
            "%spec_vec2 = OpSpecConstantComposite %v2float %float_1 %float_1",
            "%spec_extract = OpSpecConstantOp %float CompositeExtract %spec_vec2 1",
          },
          // use preserved constants in main
          {
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %spec_extract",
          },
          // duplicated constants
          {
            "%spec_extract_duplicate = OpSpecConstantOp %float CompositeExtract %spec_vec2 1",
          },
          // use duplicated constants in main
          {
            "%float_var = OpVariable %_pf_float Function",
            "OpStore %float_var %spec_extract_duplicate",
          },
          // remapped names
          {
            "OpName %spec_extract \"spec_extract_duplicate\"",
          },
        },
        // spec constant op vector shuffle
        {
          // preserved constants
          {
            "%float_1 = OpConstant %float 1",
            "%float_2 = OpConstant %float 2",
            "%spec_vec2_1 = OpSpecConstantComposite %v2float %float_1 %float_1",
            "%spec_vec2_2 = OpSpecConstantComposite %v2float %float_2 %float_2",
            "%spec_vector_shuffle = OpSpecConstantOp %v2float VectorShuffle %spec_vec2_1 %spec_vec2_2 1 2",
          },
          // use preserved constants in main
          {
            "%v2float_var = OpVariable %_pf_v2float Function",
            "OpStore %v2float_var %spec_vector_shuffle",
          },
          // duplicated constants
          {
            "%spec_vector_shuffle_duplicate = OpSpecConstantOp %v2float VectorShuffle %spec_vec2_1 %spec_vec2_2 1 2",
          },
          // use duplicated constants in main
          {
            "%v2float_var = OpVariable %_pf_v2float Function",
            "OpStore %v2float_var %spec_vector_shuffle_duplicate",
          },
          // remapped names
          {
            "OpName %spec_vector_shuffle \"spec_vector_shuffle_duplicate\"",
          },
        },
        // long dependency chain
        {
          // preserved constants
          {
            "%array_size = OpConstant %int 4",
            "%type_arr_int_4 = OpTypeArray %int %array_size",
            "%signed_0 = OpConstant %int 100",
            "%signed_1 = OpConstant %int 1",
            "%signed_2 = OpSpecConstantOp %int IAdd %signed_0 %signed_1",
            "%signed_3 = OpSpecConstantOp %int ISub %signed_0 %signed_2",
            "%signed_4 = OpSpecConstantOp %int IAdd %signed_0 %signed_3",
            "%signed_5 = OpSpecConstantOp %int ISub %signed_0 %signed_4",
            "%signed_6 = OpSpecConstantOp %int IAdd %signed_0 %signed_5",
            "%signed_7 = OpSpecConstantOp %int ISub %signed_0 %signed_6",
            "%signed_8 = OpSpecConstantOp %int IAdd %signed_0 %signed_7",
            "%signed_9 = OpSpecConstantOp %int ISub %signed_0 %signed_8",
            "%signed_10 = OpSpecConstantOp %int IAdd %signed_0 %signed_9",
            "%signed_11 = OpSpecConstantOp %int ISub %signed_0 %signed_10",
            "%signed_12 = OpSpecConstantOp %int IAdd %signed_0 %signed_11",
            "%signed_13 = OpSpecConstantOp %int ISub %signed_0 %signed_12",
            "%signed_14 = OpSpecConstantOp %int IAdd %signed_0 %signed_13",
            "%signed_15 = OpSpecConstantOp %int ISub %signed_0 %signed_14",
            "%signed_16 = OpSpecConstantOp %int ISub %signed_0 %signed_15",
            "%signed_17 = OpSpecConstantOp %int IAdd %signed_0 %signed_16",
            "%signed_18 = OpSpecConstantOp %int ISub %signed_0 %signed_17",
            "%signed_19 = OpSpecConstantOp %int IAdd %signed_0 %signed_18",
            "%signed_20 = OpSpecConstantOp %int ISub %signed_0 %signed_19",
            "%signed_vec_a = OpSpecConstantComposite %v2int %signed_18 %signed_19",
            "%signed_vec_b = OpSpecConstantOp %v2int IMul %signed_vec_a %signed_vec_a",
            "%signed_21 = OpSpecConstantOp %int CompositeExtract %signed_vec_b 0",
            "%signed_array = OpConstantComposite %type_arr_int_4 %signed_20 %signed_20 %signed_21 %signed_21",
            "%signed_22 = OpSpecConstantOp %int CompositeExtract %signed_array 0",
          },
          // use preserved constants in main
          {
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_22",
          },
          // duplicated constants
          {
            "%signed_0_dup = OpConstant %int 100",
            "%signed_1_dup = OpConstant %int 1",
            "%signed_2_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_1_dup",
            "%signed_3_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_2_dup",
            "%signed_4_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_3_dup",
            "%signed_5_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_4_dup",
            "%signed_6_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_5_dup",
            "%signed_7_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_6_dup",
            "%signed_8_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_7_dup",
            "%signed_9_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_8_dup",
            "%signed_10_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_9_dup",
            "%signed_11_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_10_dup",
            "%signed_12_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_11_dup",
            "%signed_13_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_12_dup",
            "%signed_14_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_13_dup",
            "%signed_15_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_14_dup",
            "%signed_16_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_15_dup",
            "%signed_17_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_16_dup",
            "%signed_18_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_17_dup",
            "%signed_19_dup = OpSpecConstantOp %int IAdd %signed_0_dup %signed_18_dup",
            "%signed_20_dup = OpSpecConstantOp %int ISub %signed_0_dup %signed_19_dup",
            "%signed_vec_a_dup = OpSpecConstantComposite %v2int %signed_18_dup %signed_19_dup",
            "%signed_vec_b_dup = OpSpecConstantOp %v2int IMul %signed_vec_a_dup %signed_vec_a_dup",
            "%signed_21_dup = OpSpecConstantOp %int CompositeExtract %signed_vec_b_dup 0",
            "%signed_array_dup = OpConstantComposite %type_arr_int_4 %signed_20_dup %signed_20_dup %signed_21_dup %signed_21_dup",
            "%signed_22_dup = OpSpecConstantOp %int CompositeExtract %signed_array_dup 0",
          },
          // use duplicated constants in main
          {
            "%int_var = OpVariable %_pf_int Function",
            "OpStore %int_var %signed_22_dup",
          },
          // remapped names
          {
            "OpName %signed_0 \"signed_0_dup\"",
            "OpName %signed_1 \"signed_1_dup\"",
            "OpName %signed_2 \"signed_2_dup\"",
            "OpName %signed_3 \"signed_3_dup\"",
            "OpName %signed_4 \"signed_4_dup\"",
            "OpName %signed_5 \"signed_5_dup\"",
            "OpName %signed_6 \"signed_6_dup\"",
            "OpName %signed_7 \"signed_7_dup\"",
            "OpName %signed_8 \"signed_8_dup\"",
            "OpName %signed_9 \"signed_9_dup\"",
            "OpName %signed_10 \"signed_10_dup\"",
            "OpName %signed_11 \"signed_11_dup\"",
            "OpName %signed_12 \"signed_12_dup\"",
            "OpName %signed_13 \"signed_13_dup\"",
            "OpName %signed_14 \"signed_14_dup\"",
            "OpName %signed_15 \"signed_15_dup\"",
            "OpName %signed_16 \"signed_16_dup\"",
            "OpName %signed_17 \"signed_17_dup\"",
            "OpName %signed_18 \"signed_18_dup\"",
            "OpName %signed_19 \"signed_19_dup\"",
            "OpName %signed_20 \"signed_20_dup\"",
            "OpName %signed_vec_a \"signed_vec_a_dup\"",
            "OpName %signed_vec_b \"signed_vec_b_dup\"",
            "OpName %signed_21 \"signed_21_dup\"",
            "OpName %signed_array \"signed_array_dup\"",
            "OpName %signed_22 \"signed_22_dup\"",
          },
        },
                            // clang-format on
                        })));

}  // namespace
}  // namespace opt
}  // namespace spvtools
