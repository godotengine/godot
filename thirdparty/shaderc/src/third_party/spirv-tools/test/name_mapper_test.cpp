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
#include <vector>

#include "gmock/gmock.h"
#include "source/name_mapper.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::ScopedContext;
using ::testing::Eq;

TEST(TrivialNameTest, Samples) {
  auto mapper = GetTrivialNameMapper();
  EXPECT_EQ(mapper(1), "1");
  EXPECT_EQ(mapper(1999), "1999");
  EXPECT_EQ(mapper(1024), "1024");
}

// A test case for the name mappers that actually look at an assembled module.
struct NameIdCase {
  std::string assembly;  // Input assembly text
  uint32_t id;
  std::string expected_name;
};

using FriendlyNameTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<NameIdCase>>;

TEST_P(FriendlyNameTest, SingleMapping) {
  ScopedContext context(SPV_ENV_UNIVERSAL_1_1);
  auto words = CompileSuccessfully(GetParam().assembly, SPV_ENV_UNIVERSAL_1_1);
  auto friendly_mapper =
      FriendlyNameMapper(context.context, words.data(), words.size());
  NameMapper mapper = friendly_mapper.GetNameMapper();
  EXPECT_THAT(mapper(GetParam().id), Eq(GetParam().expected_name))
      << GetParam().assembly << std::endl
      << " for id " << GetParam().id;
}

INSTANTIATE_TEST_SUITE_P(ScalarType, FriendlyNameTest,
                         ::testing::ValuesIn(std::vector<NameIdCase>{
                             {"%1 = OpTypeVoid", 1, "void"},
                             {"%1 = OpTypeBool", 1, "bool"},
                             {"%1 = OpTypeInt 8 0", 1, "uchar"},
                             {"%1 = OpTypeInt 8 1", 1, "char"},
                             {"%1 = OpTypeInt 16 0", 1, "ushort"},
                             {"%1 = OpTypeInt 16 1", 1, "short"},
                             {"%1 = OpTypeInt 32 0", 1, "uint"},
                             {"%1 = OpTypeInt 32 1", 1, "int"},
                             {"%1 = OpTypeInt 64 0", 1, "ulong"},
                             {"%1 = OpTypeInt 64 1", 1, "long"},
                             {"%1 = OpTypeInt 1 0", 1, "u1"},
                             {"%1 = OpTypeInt 1 1", 1, "i1"},
                             {"%1 = OpTypeInt 33 0", 1, "u33"},
                             {"%1 = OpTypeInt 33 1", 1, "i33"},

                             {"%1 = OpTypeFloat 16", 1, "half"},
                             {"%1 = OpTypeFloat 32", 1, "float"},
                             {"%1 = OpTypeFloat 64", 1, "double"},
                             {"%1 = OpTypeFloat 10", 1, "fp10"},
                             {"%1 = OpTypeFloat 55", 1, "fp55"},
                         }));

INSTANTIATE_TEST_SUITE_P(
    VectorType, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeBool %2 = OpTypeVector %1 1", 2, "v1bool"},
        {"%1 = OpTypeBool %2 = OpTypeVector %1 2", 2, "v2bool"},
        {"%1 = OpTypeBool %2 = OpTypeVector %1 3", 2, "v3bool"},
        {"%1 = OpTypeBool %2 = OpTypeVector %1 4", 2, "v4bool"},

        {"%1 = OpTypeInt 8 0 %2 = OpTypeVector %1 2", 2, "v2uchar"},
        {"%1 = OpTypeInt 16 1 %2 = OpTypeVector %1 3", 2, "v3short"},
        {"%1 = OpTypeInt 32 0 %2 = OpTypeVector %1 4", 2, "v4uint"},
        {"%1 = OpTypeInt 64 1 %2 = OpTypeVector %1 3", 2, "v3long"},
        {"%1 = OpTypeInt 20 0 %2 = OpTypeVector %1 4", 2, "v4u20"},
        {"%1 = OpTypeInt 21 1 %2 = OpTypeVector %1 3", 2, "v3i21"},

        {"%1 = OpTypeFloat 32 %2 = OpTypeVector %1 2", 2, "v2float"},
        // OpName overrides the element name.
        {"OpName %1 \"time\" %1 = OpTypeFloat 32 %2 = OpTypeVector %1 2", 2,
         "v2time"},
    }));

INSTANTIATE_TEST_SUITE_P(
    MatrixType, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeBool %2 = OpTypeVector %1 2 %3 = OpTypeMatrix %2 2", 3,
         "mat2v2bool"},
        {"%1 = OpTypeFloat 32 %2 = OpTypeVector %1 2 %3 = OpTypeMatrix %2 3", 3,
         "mat3v2float"},
        {"%1 = OpTypeFloat 32 %2 = OpTypeVector %1 2 %3 = OpTypeMatrix %2 4", 3,
         "mat4v2float"},
        {"OpName %1 \"time\" %1 = OpTypeFloat 32 %2 = OpTypeVector %1 2 %3 = "
         "OpTypeMatrix %2 4",
         3, "mat4v2time"},
        {"OpName %2 \"lat_long\" %1 = OpTypeFloat 32 %2 = OpTypeVector %1 2 %3 "
         "= OpTypeMatrix %2 4",
         3, "mat4lat_long"},
    }));

INSTANTIATE_TEST_SUITE_P(
    OpName, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"OpName %1 \"abcdefg\"", 1, "abcdefg"},
        {"OpName %1 \"Hello world!\"", 1, "Hello_world_"},
        {"OpName %1 \"0123456789\"", 1, "0123456789"},
        {"OpName %1 \"_\"", 1, "_"},
        // An empty string is not valid for SPIR-V assembly IDs.
        {"OpName %1 \"\"", 1, "_"},
        // Test uniqueness when presented with things mapping to "_"
        {"OpName %1 \"\" OpName %2 \"\"", 1, "_"},
        {"OpName %1 \"\" OpName %2 \"\"", 2, "__0"},
        {"OpName %1 \"\" OpName %2 \"\" OpName %3 \"_\"", 3, "__1"},
        // Test uniqueness of names that are forced to be
        // numbers.
        {"OpName %1 \"2\" OpName %2 \"2\"", 1, "2"},
        {"OpName %1 \"2\" OpName %2 \"2\"", 2, "2_0"},
        // Test uniqueness in the face of forward references
        // for Ids that don't already have friendly names.
        // In particular, the first OpDecorate assigns the name, and
        // the second one can't override it.
        {"OpDecorate %1 Volatile OpDecorate %1 Restrict", 1, "1"},
        // But a forced name can override the name that
        // would have been assigned via the OpDecorate
        // forward reference.
        {"OpName %1 \"mememe\" OpDecorate %1 Volatile OpDecorate %1 Restrict",
         1, "mememe"},
        // OpName can override other inferences.  We assume valid instruction
        // ordering, where OpName precedes type definitions.
        {"OpName %1 \"myfloat\" %1 = OpTypeFloat 32", 1, "myfloat"},
    }));

INSTANTIATE_TEST_SUITE_P(
    UniquenessHeuristic, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeVoid %2 = OpTypeVoid %3 = OpTypeVoid", 1, "void"},
        {"%1 = OpTypeVoid %2 = OpTypeVoid %3 = OpTypeVoid", 2, "void_0"},
        {"%1 = OpTypeVoid %2 = OpTypeVoid %3 = OpTypeVoid", 3, "void_1"},
    }));

INSTANTIATE_TEST_SUITE_P(Arrays, FriendlyNameTest,
                         ::testing::ValuesIn(std::vector<NameIdCase>{
                             {"OpName %2 \"FortyTwo\" %1 = OpTypeFloat 32 "
                              "%2 = OpConstant %1 42 %3 = OpTypeArray %1 %2",
                              3, "_arr_float_FortyTwo"},
                             {"%1 = OpTypeInt 32 0 "
                              "%2 = OpTypeRuntimeArray %1",
                              2, "_runtimearr_uint"},
                         }));

INSTANTIATE_TEST_SUITE_P(Structs, FriendlyNameTest,
                         ::testing::ValuesIn(std::vector<NameIdCase>{
                             {"%1 = OpTypeBool "
                              "%2 = OpTypeStruct %1 %1 %1",
                              2, "_struct_2"},
                             {"%1 = OpTypeBool "
                              "%2 = OpTypeStruct %1 %1 %1 "
                              "%3 = OpTypeStruct %2 %2",
                              3, "_struct_3"},
                         }));

INSTANTIATE_TEST_SUITE_P(
    Pointer, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeFloat 32 %2 = OpTypePointer Workgroup %1", 2,
         "_ptr_Workgroup_float"},
        {"%1 = OpTypeBool %2 = OpTypePointer Private %1", 2,
         "_ptr_Private_bool"},
        // OpTypeForwardPointer doesn't force generation of the name for its
        // target type.
        {"%1 = OpTypeBool OpTypeForwardPointer %2 Private %2 = OpTypePointer "
         "Private %1",
         2, "_ptr_Private_bool"},
    }));

INSTANTIATE_TEST_SUITE_P(ExoticTypes, FriendlyNameTest,
                         ::testing::ValuesIn(std::vector<NameIdCase>{
                             {"%1 = OpTypeEvent", 1, "Event"},
                             {"%1 = OpTypeDeviceEvent", 1, "DeviceEvent"},
                             {"%1 = OpTypeReserveId", 1, "ReserveId"},
                             {"%1 = OpTypeQueue", 1, "Queue"},
                             {"%1 = OpTypeOpaque \"hello world!\"", 1,
                              "Opaque_hello_world_"},
                             {"%1 = OpTypePipe ReadOnly", 1, "PipeReadOnly"},
                             {"%1 = OpTypePipe WriteOnly", 1, "PipeWriteOnly"},
                             {"%1 = OpTypePipe ReadWrite", 1, "PipeReadWrite"},
                             {"%1 = OpTypePipeStorage", 1, "PipeStorage"},
                             {"%1 = OpTypeNamedBarrier", 1, "NamedBarrier"},
                         }));

// Makes a test case for a BuiltIn variable declaration.
NameIdCase BuiltInCase(std::string assembly_name, std::string expected) {
  return NameIdCase{std::string("OpDecorate %1 BuiltIn ") + assembly_name +
                        " %1 = OpVariable %2 Input",
                    1, expected};
}

// Makes a test case for a BuiltIn variable declaration.  In this overload,
// the expected result is the same as the assembly name.
NameIdCase BuiltInCase(std::string assembly_name) {
  return BuiltInCase(assembly_name, assembly_name);
}

// Makes a test case for a BuiltIn variable declaration.  In this overload,
// the expected result is the same as the assembly name, but with a "gl_"
// prefix.
NameIdCase BuiltInGLCase(std::string assembly_name) {
  return BuiltInCase(assembly_name, std::string("gl_") + assembly_name);
}

INSTANTIATE_TEST_SUITE_P(
    BuiltIns, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        BuiltInGLCase("Position"),
        BuiltInGLCase("PointSize"),
        BuiltInGLCase("ClipDistance"),
        BuiltInGLCase("CullDistance"),
        BuiltInCase("VertexId", "gl_VertexID"),
        BuiltInCase("InstanceId", "gl_InstanceID"),
        BuiltInCase("PrimitiveId", "gl_PrimitiveID"),
        BuiltInCase("InvocationId", "gl_InvocationID"),
        BuiltInGLCase("Layer"),
        BuiltInGLCase("ViewportIndex"),
        BuiltInGLCase("TessLevelOuter"),
        BuiltInGLCase("TessLevelInner"),
        BuiltInGLCase("TessCoord"),
        BuiltInGLCase("PatchVertices"),
        BuiltInGLCase("FragCoord"),
        BuiltInGLCase("PointCoord"),
        BuiltInGLCase("FrontFacing"),
        BuiltInCase("SampleId", "gl_SampleID"),
        BuiltInGLCase("SamplePosition"),
        BuiltInGLCase("SampleMask"),
        BuiltInGLCase("FragDepth"),
        BuiltInGLCase("HelperInvocation"),
        BuiltInCase("NumWorkgroups", "gl_NumWorkGroups"),
        BuiltInCase("WorkgroupSize", "gl_WorkGroupSize"),
        BuiltInCase("WorkgroupId", "gl_WorkGroupID"),
        BuiltInCase("LocalInvocationId", "gl_LocalInvocationID"),
        BuiltInCase("GlobalInvocationId", "gl_GlobalInvocationID"),
        BuiltInGLCase("LocalInvocationIndex"),
        BuiltInCase("WorkDim"),
        BuiltInCase("GlobalSize"),
        BuiltInCase("EnqueuedWorkgroupSize"),
        BuiltInCase("GlobalOffset"),
        BuiltInCase("GlobalLinearId"),
        BuiltInCase("SubgroupSize"),
        BuiltInCase("SubgroupMaxSize"),
        BuiltInCase("NumSubgroups"),
        BuiltInCase("NumEnqueuedSubgroups"),
        BuiltInCase("SubgroupId"),
        BuiltInCase("SubgroupLocalInvocationId"),
        BuiltInGLCase("VertexIndex"),
        BuiltInGLCase("InstanceIndex"),
        BuiltInCase("SubgroupEqMaskKHR"),
        BuiltInCase("SubgroupGeMaskKHR"),
        BuiltInCase("SubgroupGtMaskKHR"),
        BuiltInCase("SubgroupLeMaskKHR"),
        BuiltInCase("SubgroupLtMaskKHR"),
    }));

INSTANTIATE_TEST_SUITE_P(DebugNameOverridesBuiltin, FriendlyNameTest,
                         ::testing::ValuesIn(std::vector<NameIdCase>{
                             {"OpName %1 \"foo\" OpDecorate %1 BuiltIn WorkDim "
                              "%1 = OpVariable %2 Input",
                              1, "foo"}}));

INSTANTIATE_TEST_SUITE_P(
    SimpleIntegralConstants, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeInt 32 0 %2 = OpConstant %1 0", 2, "uint_0"},
        {"%1 = OpTypeInt 32 0 %2 = OpConstant %1 1", 2, "uint_1"},
        {"%1 = OpTypeInt 32 0 %2 = OpConstant %1 2", 2, "uint_2"},
        {"%1 = OpTypeInt 32 0 %2 = OpConstant %1 9", 2, "uint_9"},
        {"%1 = OpTypeInt 32 0 %2 = OpConstant %1 42", 2, "uint_42"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 0", 2, "int_0"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 1", 2, "int_1"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 2", 2, "int_2"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 9", 2, "int_9"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 42", 2, "int_42"},
        {"%1 = OpTypeInt 32 1 %2 = OpConstant %1 -42", 2, "int_n42"},
        // Exotic bit widths
        {"%1 = OpTypeInt 33 0 %2 = OpConstant %1 0", 2, "u33_0"},
        {"%1 = OpTypeInt 33 1 %2 = OpConstant %1 10", 2, "i33_10"},
        {"%1 = OpTypeInt 33 1 %2 = OpConstant %1 -19", 2, "i33_n19"},
    }));

INSTANTIATE_TEST_SUITE_P(
    SimpleFloatConstants, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeFloat 16\n%2 = OpConstant %1 0x1.ff4p+16", 2,
         "half_0x1_ff4p_16"},
        {"%1 = OpTypeFloat 16\n%2 = OpConstant %1 -0x1.d2cp-10", 2,
         "half_n0x1_d2cpn10"},
        // 32-bit floats
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 -3.125", 2, "float_n3_125"},
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1.8p+128", 2,
         "float_0x1_8p_128"},  // NaN
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1.0002p+128", 2,
         "float_n0x1_0002p_128"},  // NaN
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1p+128", 2,
         "float_0x1p_128"},  // Inf
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1p+128", 2,
         "float_n0x1p_128"},  // -Inf
                              // 64-bit floats
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 -3.125", 2, "double_n3_125"},
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.ffffffffffffap-1023", 2,
         "double_0x1_ffffffffffffapn1023"},  // small normal
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.ffffffffffffap-1023", 2,
         "double_n0x1_ffffffffffffapn1023"},
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.8p+1024", 2,
         "double_0x1_8p_1024"},  // NaN
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.0002p+1024", 2,
         "double_n0x1_0002p_1024"},  // NaN
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1p+1024", 2,
         "double_0x1p_1024"},  // Inf
        {"%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1p+1024", 2,
         "double_n0x1p_1024"},  // -Inf
    }));

INSTANTIATE_TEST_SUITE_P(
    BooleanConstants, FriendlyNameTest,
    ::testing::ValuesIn(std::vector<NameIdCase>{
        {"%1 = OpTypeBool\n%2 = OpConstantTrue %1", 2, "true"},
        {"%1 = OpTypeBool\n%2 = OpConstantFalse %1", 2, "false"},
    }));

}  // namespace
}  // namespace spvtools
