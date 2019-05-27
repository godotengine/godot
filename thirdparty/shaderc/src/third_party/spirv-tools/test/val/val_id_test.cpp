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

#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

// NOTE: The tests in this file are ONLY testing ID usage, there for the input
// SPIR-V does not follow the logical layout rules from the spec in all cases in
// order to makes the tests smaller. Validation of the whole module is handled
// in stages, ID validation is only one of these stages. All validation stages
// are stand alone.

namespace spvtools {
namespace val {
namespace {

using spvtest::ScopedContext;
using ::testing::HasSubstr;
using ::testing::ValuesIn;

using ValidateIdWithMessage = spvtest::ValidateBase<bool>;

std::string kOpCapabilitySetupWithoutVector16 = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Addresses
     OpCapability Int8
     OpCapability Int16
     OpCapability Int64
     OpCapability Float64
     OpCapability LiteralSampler
     OpCapability Pipes
     OpCapability DeviceEnqueue
)";

std::string kOpCapabilitySetup = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Addresses
     OpCapability Int8
     OpCapability Int16
     OpCapability Int64
     OpCapability Float64
     OpCapability LiteralSampler
     OpCapability Pipes
     OpCapability DeviceEnqueue
     OpCapability Vector16
)";

std::string kOpVariablePtrSetUp = R"(
     OpCapability VariablePointers
     OpExtension "SPV_KHR_variable_pointers"
)";

std::string kGLSL450MemoryModel =
    kOpCapabilitySetup + kOpVariablePtrSetUp + R"(
     OpMemoryModel Logical GLSL450
)";

std::string kGLSL450MemoryModelWithoutVector16 =
    kOpCapabilitySetupWithoutVector16 + kOpVariablePtrSetUp + R"(
     OpMemoryModel Logical GLSL450
)";

std::string kNoKernelGLSL450MemoryModel = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability Addresses
     OpCapability Int8
     OpCapability Int16
     OpCapability Int64
     OpCapability Float64
     OpMemoryModel Logical GLSL450
)";

std::string kOpenCLMemoryModel32 = R"(
     OpCapability Addresses
     OpCapability Linkage
     OpCapability Kernel
%1 = OpExtInstImport "OpenCL.std"
     OpMemoryModel Physical32 OpenCL
)";

std::string kOpenCLMemoryModel64 = R"(
     OpCapability Addresses
     OpCapability Linkage
     OpCapability Kernel
     OpCapability Int64
%1 = OpExtInstImport "OpenCL.std"
     OpMemoryModel Physical64 OpenCL
)";

std::string sampledImageSetup = R"(
                    %void = OpTypeVoid
            %typeFuncVoid = OpTypeFunction %void
                   %float = OpTypeFloat 32
                 %v4float = OpTypeVector %float 4
              %image_type = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_img = OpTypePointer UniformConstant %image_type
                     %tex = OpVariable %_ptr_UniformConstant_img UniformConstant
            %sampler_type = OpTypeSampler
%_ptr_UniformConstant_sam = OpTypePointer UniformConstant %sampler_type
                       %s = OpVariable %_ptr_UniformConstant_sam UniformConstant
      %sampled_image_type = OpTypeSampledImage %image_type
                 %v2float = OpTypeVector %float 2
                 %float_1 = OpConstant %float 1
                 %float_2 = OpConstant %float 2
           %const_vec_1_1 = OpConstantComposite %v2float %float_1 %float_1
           %const_vec_2_2 = OpConstantComposite %v2float %float_2 %float_2
               %bool_type = OpTypeBool
               %spec_true = OpSpecConstantTrue %bool_type
                    %main = OpFunction %void None %typeFuncVoid
                 %label_1 = OpLabel
              %image_inst = OpLoad %image_type %tex
            %sampler_inst = OpLoad %sampler_type %s
)";

std::string BranchConditionalSetup = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 140
               OpName %main "main"

             ; type definitions
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4

             ; constants
       %true = OpConstantTrue %bool
         %i0 = OpConstant %int 0
         %i1 = OpConstant %int 1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1


             ; main function header
       %void = OpTypeVoid
   %voidfunc = OpTypeFunction %void
       %main = OpFunction %void None %voidfunc
      %lmain = OpLabel
)";

std::string BranchConditionalTail = R"(
   %target_t = OpLabel
               OpNop
               OpBranch %end
   %target_f = OpLabel
               OpNop
               OpBranch %end

        %end = OpLabel

               OpReturn
               OpFunctionEnd
)";

// TODO: OpUndef

TEST_F(ValidateIdWithMessage, OpName) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpName %2 "name"
%1 = OpTypeInt 32 0
%2 = OpTypePointer UniformConstant %1
%3 = OpVariable %2 UniformConstant)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpMemberNameGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberName %2 0 "foo"
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpMemberNameTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberName %1 0 "foo"
%1 = OpTypeInt 32 0)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpMemberName Type <id> '1[%uint]' is not a struct type."));
}
TEST_F(ValidateIdWithMessage, OpMemberNameMemberBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberName %1 1 "foo"
%2 = OpTypeInt 32 0
%1 = OpTypeStruct %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpMemberName Member <id> '1[%_struct_1]' index is larger "
                "than Type <id> '1[%_struct_1]'s member count."));
}

TEST_F(ValidateIdWithMessage, OpLineGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpString "/path/to/source.file"
     OpLine %1 0 0
%2 = OpTypeInt 32 0
%3 = OpTypePointer Input %2
%4 = OpVariable %3 Input)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpLineFileBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
  %1 = OpTypeInt 32 0
     OpLine %1 0 0
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpLine Target <id> '1[%uint]' is not an OpString."));
}

TEST_F(ValidateIdWithMessage, OpDecorateGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpDecorate %2 GLSLShared
%1 = OpTypeInt 64 0
%2 = OpTypeStruct %1 %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpDecorateBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpDecorate %1 GLSLShared)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("forward referenced IDs have not been defined"));
}

TEST_F(ValidateIdWithMessage, OpMemberDecorateGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %2 0 RelaxedPrecision
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpMemberDecorateBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 RelaxedPrecision
%1 = OpTypeInt 32 0)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpMemberDecorate Structure type <id> '1[%uint]' is "
                        "not a struct type."));
}
TEST_F(ValidateIdWithMessage, OpMemberDecorateMemberBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 3 RelaxedPrecision
%int = OpTypeInt 32 0
%1 = OpTypeStruct %int %int)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Index 3 provided in OpMemberDecorate for struct <id> "
                        "1[%_struct_1] is out of bounds. The structure has 2 "
                        "members. Largest valid index is 1."));
}

TEST_F(ValidateIdWithMessage, OpGroupDecorateGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpDecorationGroup
     OpDecorate %1 RelaxedPrecision
     OpDecorate %1 GLSLShared
     OpGroupDecorate %1 %3 %4
%2 = OpTypeInt 32 0
%3 = OpConstant %2 42
%4 = OpConstant %2 23)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpDecorationGroupBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpDecorationGroup
     OpDecorate %1 RelaxedPrecision
     OpDecorate %1 GLSLShared
     OpMemberDecorate %1 0 Constant
    )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result id of OpDecorationGroup can only "
                        "be targeted by OpName, OpGroupDecorate, "
                        "OpDecorate, and OpGroupMemberDecorate"));
}
TEST_F(ValidateIdWithMessage, OpGroupDecorateDecorationGroupBad) {
  std::string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    %1 = OpExtInstImport "GLSL.std.450"
    OpMemoryModel Logical GLSL450
    OpGroupDecorate %1 %2 %3
%2 = OpTypeInt 32 0
%3 = OpConstant %2 42)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpGroupDecorate Decoration group <id> '1[%1]' is not "
                        "a decoration group."));
}
TEST_F(ValidateIdWithMessage, OpGroupDecorateTargetBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpDecorationGroup
     OpDecorate %1 RelaxedPrecision
     OpDecorate %1 GLSLShared
     OpGroupDecorate %1 %3
%2 = OpTypeInt 32 0)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("forward referenced IDs have not been defined"));
}
TEST_F(ValidateIdWithMessage, OpGroupMemberDecorateDecorationGroupBad) {
  std::string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    %1 = OpExtInstImport "GLSL.std.450"
    OpMemoryModel Logical GLSL450
    OpGroupMemberDecorate %1 %2 0
%2 = OpTypeInt 32 0)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpGroupMemberDecorate Decoration group <id> '1[%1]' "
                        "is not a decoration group."));
}
TEST_F(ValidateIdWithMessage, OpGroupMemberDecorateIdNotStructBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     %1 = OpDecorationGroup
     OpGroupMemberDecorate %1 %2 0
%2 = OpTypeInt 32 0)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpGroupMemberDecorate Structure type <id> '2[%uint]' "
                        "is not a struct type."));
}
TEST_F(ValidateIdWithMessage, OpGroupMemberDecorateIndexOutOfBoundBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
  OpDecorate %1 Offset 0
  %1 = OpDecorationGroup
  OpGroupMemberDecorate %1 %struct 3
%float  = OpTypeFloat 32
%struct = OpTypeStruct %float %float %float
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Index 3 provided in OpGroupMemberDecorate for struct "
                        "<id> 2[%_struct_2] is out of bounds. The structure "
                        "has 3 members. Largest valid index is 2."));
}

// TODO: OpExtInst

TEST_F(ValidateIdWithMessage, OpEntryPointGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %3 ""
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpEntryPointFunctionBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %1 ""
%1 = OpTypeVoid)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpEntryPoint Entry Point <id> '1[%void]' is not a "
                        "function."));
}
TEST_F(ValidateIdWithMessage, OpEntryPointParameterCountBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %3 ""
%1 = OpTypeVoid
%2 = OpTypeFunction %1 %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpEntryPoint Entry Point <id> '1[%1]'s function "
                        "parameter count is not zero"));
}
TEST_F(ValidateIdWithMessage, OpEntryPointReturnTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %3 ""
%1 = OpTypeInt 32 0
%ret = OpConstant %1 0
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturnValue %ret
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpEntryPoint Entry Point <id> '1[%1]'s function "
                        "return type is not void."));
}

TEST_F(ValidateIdWithMessage, OpEntryPointInterfaceIsNotVariableTypeBad) {
  std::string spirv = R"(
               OpCapability Shader
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %ptr_builtin_1
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpMemberDecorate %struct_1 0 BuiltIn InvocationId
      %int = OpTypeInt 32 1
     %void = OpTypeVoid
     %func = OpTypeFunction %void
 %struct_1 = OpTypeStruct %int
%ptr_builtin_1 = OpTypePointer Input %struct_1
       %main = OpFunction %void None %func
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Interfaces passed to OpEntryPoint must be of type "
                        "OpTypeVariable. Found OpTypePointer."));
}

TEST_F(ValidateIdWithMessage, OpEntryPointInterfaceStorageClassBad) {
  std::string spirv = R"(
               OpCapability Shader
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %in_1
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpMemberDecorate %struct_1 0 BuiltIn InvocationId
      %int = OpTypeInt 32 1
     %void = OpTypeVoid
     %func = OpTypeFunction %void
 %struct_1 = OpTypeStruct %int
%ptr_builtin_1 = OpTypePointer Uniform %struct_1
       %in_1 = OpVariable %ptr_builtin_1 Uniform
       %main = OpFunction %void None %func
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpEntryPoint interfaces must be OpVariables with "
                        "Storage Class of Input(1) or Output(3). Found Storage "
                        "Class 2 for Entry Point id 1."));
}

TEST_F(ValidateIdWithMessage, OpExecutionModeGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpExecutionModeEntryPointMissing) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpExecutionMode Entry Point <id> '1[%1]' is not the "
                        "Entry Point operand of an OpEntryPoint."));
}

TEST_F(ValidateIdWithMessage, OpExecutionModeEntryPointBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpEntryPoint GLCompute %3 "" %a
     OpExecutionMode %a LocalSize 1 1 1
%void = OpTypeVoid
%ptr = OpTypePointer Input %void
%a = OpVariable %ptr Input
%2 = OpTypeFunction %void
%3 = OpFunction %void None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpExecutionMode Entry Point <id> '2[%2]' is not the "
                        "Entry Point operand of an OpEntryPoint."));
}

TEST_F(ValidateIdWithMessage, OpTypeVectorFloat) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeVectorInt) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeVector %1 4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeVectorUInt) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 64 0
%2 = OpTypeVector %1 4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeVectorBool) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeBool
%2 = OpTypeVector %1 4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeVectorComponentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypePointer UniformConstant %1
%3 = OpTypeVector %2 4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeVector Component Type <id> "
                "'2[%_ptr_UniformConstant_float]' is not a scalar type."));
}

TEST_F(ValidateIdWithMessage, OpTypeVectorColumnCountLessThanTwoBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Illegal number of components (1) for TypeVector\n  %v1float = "
                "OpTypeVector %float 1\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeVectorColumnCountGreaterThanFourBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Illegal number of components (5) for TypeVector\n  %v5float = "
                "OpTypeVector %float 5\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeVectorColumnCountEightWithoutVector16Bad) {
  std::string spirv = kGLSL450MemoryModelWithoutVector16 + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 8)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Having 8 components for TypeVector requires the Vector16 "
                "capability\n  %v8float = OpTypeVector %float 8\n"));
}

TEST_F(ValidateIdWithMessage,
       OpTypeVectorColumnCountSixteenWithoutVector16Bad) {
  std::string spirv = kGLSL450MemoryModelWithoutVector16 + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 16)";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Having 16 components for TypeVector requires the Vector16 "
                "capability\n  %v16float = OpTypeVector %float 16\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeVectorColumnCountOfEightWithVector16Good) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 8)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage,
       OpTypeVectorColumnCountOfSixteenWithVector16Good) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 16)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeMatrixGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 2
%3 = OpTypeMatrix %2 3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeMatrixColumnTypeNonVectorBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeMatrix %1 3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("olumns in a matrix must be of type vector.\n  %mat3float = "
                "OpTypeMatrix %float 3\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeMatrixVectorTypeNonFloatBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 16 0
%2 = OpTypeVector %1 2
%3 = OpTypeMatrix %2 2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Matrix types can only be parameterized with floating-point "
                "types.\n  %mat2v2ushort = OpTypeMatrix %v2ushort 2\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeMatrixColumnCountLessThanTwoBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 2
%3 = OpTypeMatrix %2 1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Matrix types can only be parameterized as having only 2, 3, "
                "or 4 columns.\n  %mat1v2float = OpTypeMatrix %v2float 1\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeMatrixColumnCountGreaterThanFourBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 2
%3 = OpTypeMatrix %2 8)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Matrix types can only be parameterized as having only 2, 3, "
                "or 4 columns.\n  %mat8v2float = OpTypeMatrix %v2float 8\n"));
}

TEST_F(ValidateIdWithMessage, OpTypeSamplerGood) {
  // In Rev31, OpTypeSampler takes no arguments.
  std::string spirv = kGLSL450MemoryModel + R"(
%s = OpTypeSampler)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeArrayGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1
%3 = OpTypeArray %1 %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeArrayElementTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1
%3 = OpTypeArray %2 %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeArray Element Type <id> '2[%uint_1]' is not a "
                        "type."));
}

// Signed or unsigned.
enum Signed { kSigned, kUnsigned };

// Creates an assembly snippet declaring OpTypeArray with the given length.
std::string MakeArrayLength(const std::string& len, Signed isSigned,
                            int width) {
  std::ostringstream ss;
  ss << R"(
    OpCapability Shader
    OpCapability Linkage
    OpCapability Int16
    OpCapability Int64
  )";
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << " %t = OpTypeInt " << width << (isSigned == kSigned ? " 1" : " 0");
  ss << " %l = OpConstant %t " << len;
  ss << " %a = OpTypeArray %t %l";
  return ss.str();
}

// Tests OpTypeArray.  Parameter is the width (in bits) of the array-length's
// type.
class OpTypeArrayLengthTest
    : public spvtest::TextToBinaryTestBase<::testing::TestWithParam<int>> {
 protected:
  OpTypeArrayLengthTest()
      : position_(spv_position_t{0, 0, 0}),
        diagnostic_(spvDiagnosticCreate(&position_, "")) {}

  ~OpTypeArrayLengthTest() { spvDiagnosticDestroy(diagnostic_); }

  // Runs spvValidate() on v, printing any errors via spvDiagnosticPrint().
  spv_result_t Val(const SpirvVector& v, const std::string& expected_err = "") {
    spv_const_binary_t cbinary{v.data(), v.size()};
    spvDiagnosticDestroy(diagnostic_);
    diagnostic_ = nullptr;
    const auto status =
        spvValidate(ScopedContext().context, &cbinary, &diagnostic_);
    if (status != SPV_SUCCESS) {
      spvDiagnosticPrint(diagnostic_);
      EXPECT_THAT(std::string(diagnostic_->error),
                  testing::ContainsRegex(expected_err));
    }
    return status;
  }

 private:
  spv_position_t position_;  // For creating diagnostic_.
  spv_diagnostic diagnostic_;
};

TEST_P(OpTypeArrayLengthTest, LengthPositive) {
  const int width = GetParam();
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("1", kSigned, width))));
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("1", kUnsigned, width))));
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("2", kSigned, width))));
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("2", kUnsigned, width))));
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("55", kSigned, width))));
  EXPECT_EQ(SPV_SUCCESS,
            Val(CompileSuccessfully(MakeArrayLength("55", kUnsigned, width))));
  const std::string fpad(width / 4 - 1, 'F');
  EXPECT_EQ(
      SPV_SUCCESS,
      Val(CompileSuccessfully(MakeArrayLength("0x7" + fpad, kSigned, width))));
  EXPECT_EQ(SPV_SUCCESS, Val(CompileSuccessfully(
                             MakeArrayLength("0xF" + fpad, kUnsigned, width))));
}

TEST_P(OpTypeArrayLengthTest, LengthZero) {
  const int width = GetParam();
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength("0", kSigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength("0", kUnsigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
}

TEST_P(OpTypeArrayLengthTest, LengthNegative) {
  const int width = GetParam();
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength("-1", kSigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength("-2", kSigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength("-123", kSigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
  const std::string neg_max = "0x8" + std::string(width / 4 - 1, '0');
  EXPECT_EQ(SPV_ERROR_INVALID_ID,
            Val(CompileSuccessfully(MakeArrayLength(neg_max, kSigned, width)),
                "OpTypeArray Length <id> '2\\[%.*\\]' default value must be at "
                "least 1."));
}

// The only valid widths for integers are 8, 16, 32, and 64.
// Since the Int8 capability requires the Kernel capability, and the Kernel
// capability prohibits usage of signed integers, we can skip 8-bit integers
// here since the purpose of these tests is to check the validity of
// OpTypeArray, not OpTypeInt.
INSTANTIATE_TEST_SUITE_P(Widths, OpTypeArrayLengthTest,
                         ValuesIn(std::vector<int>{16, 32, 64}));

TEST_F(ValidateIdWithMessage, OpTypeArrayLengthNull) {
  std::string spirv = kGLSL450MemoryModel + R"(
%i32 = OpTypeInt 32 0
%len = OpConstantNull %i32
%ary = OpTypeArray %i32 %len)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpTypeArray Length <id> '2[%2]' default value must be at least 1."));
}

TEST_F(ValidateIdWithMessage, OpTypeArrayLengthSpecConst) {
  std::string spirv = kGLSL450MemoryModel + R"(
%i32 = OpTypeInt 32 0
%len = OpSpecConstant %i32 2
%ary = OpTypeArray %i32 %len)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeArrayLengthSpecConstOp) {
  std::string spirv = kGLSL450MemoryModel + R"(
%i32 = OpTypeInt 32 0
%c1 = OpConstant %i32 1
%c2 = OpConstant %i32 2
%len = OpSpecConstantOp %i32 IAdd %c1 %c2
%ary = OpTypeArray %i32 %len)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpTypeRuntimeArrayGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeRuntimeArray %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpTypeRuntimeArrayBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypeRuntimeArray %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeRuntimeArray Element Type <id> '2[%uint_0]' is not a "
                "type."));
}
// TODO: Object of this type can only be created with OpVariable using the
// Unifrom Storage Class

TEST_F(ValidateIdWithMessage, OpTypeStructGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFloat 64
%3 = OpTypePointer Input %1
%4 = OpTypeStruct %1 %2 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpTypeStructMemberTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFloat 64
%3 = OpConstant %2 0.0
%4 = OpTypeStruct %1 %2 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeStruct Member Type <id> '3[%double_0]' is not "
                        "a type."));
}

TEST_F(ValidateIdWithMessage, OpTypePointerGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Input %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpTypePointerBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypePointer Input %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypePointer Type <id> '2[%uint_0]' is not a "
                        "type."));
}

TEST_F(ValidateIdWithMessage, OpTypeFunctionGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpTypeFunctionReturnTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypeFunction %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeFunction Return Type <id> '2[%uint_0]' is not "
                        "a type."));
}
TEST_F(ValidateIdWithMessage, OpTypeFunctionParameterBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1 %2 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTypeFunction Parameter Type <id> '3[%uint_0]' is not a "
                "type."));
}

TEST_F(ValidateIdWithMessage, OpTypeFunctionParameterTypeVoidBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%4 = OpTypeFunction %1 %2 %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeFunction Parameter Type <id> '1[%void]' cannot "
                        "be OpTypeVoid."));
}

TEST_F(ValidateIdWithMessage, OpTypePipeGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 16
%3 = OpTypePipe ReadOnly)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpConstantTrueGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeBool
%2 = OpConstantTrue %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantTrueBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpConstantTrue %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantTrue Result Type <id> '1[%void]' is not a boolean "
                "type."));
}

TEST_F(ValidateIdWithMessage, OpConstantFalseGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeBool
%2 = OpConstantTrue %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantFalseBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpConstantFalse %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantFalse Result Type <id> '1[%void]' is not a boolean "
                "type."));
}

TEST_F(ValidateIdWithMessage, OpConstantGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpConstant !1 !0)";
  // The expected failure code is implementation dependent (currently
  // INVALID_BINARY because the binary parser catches these cases) and may
  // change over time, but this must always fail.
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpConstantCompositeVectorGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpConstant %1 3.14
%4 = OpConstantComposite %2 %3 %3 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeVectorWithUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpConstant %1 3.14
%9 = OpUndef %1
%4 = OpConstantComposite %2 %3 %3 %3 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeVectorResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpConstant %1 3.14
%4 = OpConstantComposite %1 %3 %3 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantComposite Result Type <id> '1[%float]' is not a "
                "composite type."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeVectorConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%4 = OpTypeInt 32 0
%3 = OpConstant %1 3.14
%5 = OpConstant %4 42 ; bad type for constant value
%6 = OpConstantComposite %2 %3 %5 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantComposite Constituent <id> '5[%uint_42]'s type "
                "does not match Result Type <id> '2[%v4float]'s vector "
                "element type."));
}
TEST_F(ValidateIdWithMessage,
       OpConstantCompositeVectorConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%4 = OpTypeInt 32 0
%3 = OpConstant %1 3.14
%5 = OpUndef %4 ; bad type for undef value
%6 = OpConstantComposite %2 %3 %5 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantComposite Constituent <id> '5[%5]'s type does not "
                "match Result Type <id> '2[%v4float]'s vector element type."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeMatrixGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpConstantComposite %2 %5 %5 %5 %4
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeMatrixUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpUndef %2
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeMatrixConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
%11 = OpTypeVector %1 3
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpConstantComposite %11 %5 %5 %5
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> '10[%10]' vector "
                        "component count does not match Result Type <id> "
                        "'4[%mat4v4float]'s vector component count."));
}
TEST_F(ValidateIdWithMessage,
       OpConstantCompositeMatrixConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
%11 = OpTypeVector %1 3
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpUndef %11
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> '10[%10]' vector "
                        "component count does not match Result Type <id> "
                        "'4[%mat4v4float]'s vector component count."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpConstantComposite %3 %2 %2 %2 %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayWithUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%9 = OpUndef %1
%3 = OpTypeArray %1 %2
%4 = OpConstantComposite %3 %2 %2 %2 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayConstConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpConstantComposite %3 %2 %2 %2 %1)";  // Uses a type as operand
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 1[%uint] cannot be a "
                                               "type"));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayConstConstituentBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpTypePointer Uniform %1
%5 = OpVariable %4 Uniform
%6 = OpConstantComposite %3 %2 %2 %2 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> '5[%5]' is not a "
                        "constant or undef."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%5 = OpTypeFloat 32
%6 = OpConstant %5 3.14 ; bad type for const value
%4 = OpConstantComposite %3 %2 %2 %2 %6)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> "
                        "'5[%float_3_1400001]'s type does not match Result "
                        "Type <id> '3[%_arr_uint_uint_4]'s array element "
                        "type."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeArrayConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%5 = OpTypeFloat 32
%6 = OpUndef %5 ; bad type for undef
%4 = OpConstantComposite %3 %2 %2 %2 %6)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> "
                        "'5[%5]'s type does not match Result "
                        "Type <id> '3[%_arr_uint_uint_4]'s array element "
                        "type."));
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeStructGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpConstant %2 4300000000
%6 = OpConstantComposite %3 %4 %4 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeStructUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpUndef %2
%6 = OpConstantComposite %3 %4 %4 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantCompositeStructMemberTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpConstant %2 4300000000
%6 = OpConstantComposite %3 %4 %5 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> "
                        "'5[%ulong_4300000000]' type does not match the "
                        "Result Type <id> '3[%_struct_3]'s member type."));
}

TEST_F(ValidateIdWithMessage, OpConstantCompositeStructMemberUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpUndef %2
%6 = OpConstantComposite %3 %4 %5 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantComposite Constituent <id> '5[%5]' type "
                        "does not match the Result Type <id> '3[%_struct_3]'s "
                        "member type."));
}

TEST_F(ValidateIdWithMessage, OpConstantSamplerGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%float = OpTypeFloat 32
%samplerType = OpTypeSampler
%3 = OpConstantSampler %samplerType ClampToEdge 0 Nearest)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpConstantSamplerResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpConstantSampler %1 Clamp 0 Nearest)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpConstantSampler Result Type <id> '1[%float]' is not a sampler "
          "type."));
}

TEST_F(ValidateIdWithMessage, OpConstantNullGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeBool
 %2 = OpConstantNull %1
 %3 = OpTypeInt 32 0
 %4 = OpConstantNull %3
 %5 = OpTypeFloat 32
 %6 = OpConstantNull %5
 %7 = OpTypePointer UniformConstant %3
 %8 = OpConstantNull %7
 %9 = OpTypeEvent
%10 = OpConstantNull %9
%11 = OpTypeDeviceEvent
%12 = OpConstantNull %11
%13 = OpTypeReserveId
%14 = OpConstantNull %13
%15 = OpTypeQueue
%16 = OpConstantNull %15
%17 = OpTypeVector %5 2
%18 = OpConstantNull %17
%19 = OpTypeMatrix %17 2
%20 = OpConstantNull %19
%25 = OpConstant %3 8
%21 = OpTypeArray %3 %25
%22 = OpConstantNull %21
%23 = OpTypeStruct %3 %5 %1
%24 = OpConstantNull %23
%26 = OpTypeArray %17 %25
%27 = OpConstantNull %26
%28 = OpTypeStruct %7 %26 %26 %1
%29 = OpConstantNull %28
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpConstantNullBasicBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpConstantNull %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpConstantNull Result Type <id> '1[%void]' cannot have a null "
                "value."));
}

TEST_F(ValidateIdWithMessage, OpConstantNullArrayBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%2 = OpTypeInt 32 0
%3 = OpTypeSampler
%4 = OpConstant %2 4
%5 = OpTypeArray %3 %4
%6 = OpConstantNull %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpConstantNull Result Type <id> '4[%_arr_2_uint_4]' cannot have a "
          "null value."));
}

TEST_F(ValidateIdWithMessage, OpConstantNullStructBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%2 = OpTypeSampler
%3 = OpTypeStruct %2 %2
%4 = OpConstantNull %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantNull Result Type <id> '2[%_struct_2]' "
                        "cannot have a null value."));
}

TEST_F(ValidateIdWithMessage, OpConstantNullRuntimeArrayBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%bool = OpTypeBool
%array = OpTypeRuntimeArray %bool
%null = OpConstantNull %array)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpConstantNull Result Type <id> '2[%_runtimearr_bool]' cannot have "
          "a null value."));
}

TEST_F(ValidateIdWithMessage, OpSpecConstantTrueGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeBool
%2 = OpSpecConstantTrue %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpSpecConstantTrueBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpSpecConstantTrue %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Specialization constant must be a boolean type."));
}

TEST_F(ValidateIdWithMessage, OpSpecConstantFalseGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeBool
%2 = OpSpecConstantFalse %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpSpecConstantFalseBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpSpecConstantFalse %1)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Specialization constant must be a boolean type."));
}

TEST_F(ValidateIdWithMessage, OpSpecConstantGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpSpecConstant %1 42)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpSpecConstantBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpSpecConstant !1 !4)";
  // The expected failure code is implementation dependent (currently
  // INVALID_BINARY because the binary parser catches these cases) and may
  // change over time, but this must always fail.
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Type Id 1 is not a scalar numeric type"));
}

// Valid: SpecConstantComposite specializes to a vector.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeVectorGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpSpecConstant %1 3.14
%4 = OpConstant %1 3.14
%5 = OpSpecConstantComposite %2 %3 %3 %4 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Valid: Vector of floats and Undefs.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeVectorWithUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpSpecConstant %1 3.14
%5 = OpConstant %1 3.14
%9 = OpUndef %1
%4 = OpSpecConstantComposite %2 %3 %5 %3 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: result type is float.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeVectorResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpSpecConstant %1 3.14
%4 = OpSpecConstantComposite %1 %3 %3 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("is not a composite type"));
}

// Invalid: Vector contains a mix of Int and Float.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeVectorConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%4 = OpTypeInt 32 0
%3 = OpSpecConstant %1 3.14
%5 = OpConstant %4 42 ; bad type for constant value
%6 = OpSpecConstantComposite %2 %3 %5 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> "
                        "'5[%uint_42]'s type does not match Result Type <id> "
                        "'2[%v4float]'s vector element type."));
}

// Invalid: Constituent is not a constant
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeVectorConstituentNotConstantBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpTypeInt 32 0
%4 = OpSpecConstant %1 3.14
%5 = OpTypePointer Uniform %1
%6 = OpVariable %5 Uniform
%7 = OpSpecConstantComposite %2 %6 %4 %4 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '6[%6]' is "
                        "not a constant or undef."));
}

// Invalid: Vector contains a mix of Undef-int and Float.
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeVectorConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%4 = OpTypeInt 32 0
%3 = OpSpecConstant %1 3.14
%5 = OpUndef %4 ; bad type for undef value
%6 = OpSpecConstantComposite %2 %3 %5 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]'s "
                        "type does not match Result Type <id> '2[%v4float]'s "
                        "vector element type."));
}

// Invalid: Vector expects 3 components, but 4 specified.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeVectorNumComponentsBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 3
%3 = OpConstant %1 3.14
%5 = OpSpecConstant %1 4.0
%6 = OpSpecConstantComposite %2 %3 %5 %3 %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> count does "
                        "not match Result Type <id> '2[%v3float]'s vector "
                        "component count."));
}

// Valid: 4x4 matrix of floats
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpSpecConstant %1 0.0
 %6 = OpSpecConstantComposite %2 %4 %5 %5 %5
 %7 = OpSpecConstantComposite %2 %5 %4 %5 %5
 %8 = OpSpecConstantComposite %2 %5 %5 %4 %5
 %9 = OpSpecConstantComposite %2 %5 %5 %5 %4
%10 = OpSpecConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Valid: Matrix in which one column is Undef
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpSpecConstant %1 0.0
 %6 = OpSpecConstantComposite %2 %4 %5 %5 %5
 %7 = OpSpecConstantComposite %2 %5 %4 %5 %5
 %8 = OpSpecConstantComposite %2 %5 %5 %4 %5
 %9 = OpUndef %2
%10 = OpSpecConstantComposite %3 %6 %7 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Matrix in which the sizes of column vectors are not equal.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeVector %1 3
 %4 = OpTypeMatrix %2 4
 %5 = OpSpecConstant %1 1.0
 %6 = OpConstant %1 0.0
 %7 = OpSpecConstantComposite %2 %5 %6 %6 %6
 %8 = OpSpecConstantComposite %2 %6 %5 %6 %6
 %9 = OpSpecConstantComposite %2 %6 %6 %5 %6
 %10 = OpSpecConstantComposite %3 %6 %6 %6
%11 = OpSpecConstantComposite %4 %7 %8 %9 %10)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '10[%10]' "
                        "vector component count does not match Result Type "
                        "<id> '4[%mat4v4float]'s vector component count."));
}

// Invalid: Matrix type expects 4 columns but only 3 specified.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixNumColsBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpSpecConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpSpecConstantComposite %2 %4 %5 %5 %5
 %7 = OpSpecConstantComposite %2 %5 %4 %5 %5
 %8 = OpSpecConstantComposite %2 %5 %5 %4 %5
%10 = OpSpecConstantComposite %3 %6 %7 %8)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpSpecConstantComposite Constituent <id> count does "
                "not match Result Type <id> '3[%mat4v4float]'s matrix column "
                "count."));
}

// Invalid: Composite contains a non-const/undef component
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeMatrixConstituentNotConstBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpConstant %1 0.0
 %3 = OpTypeVector %1 4
 %4 = OpTypeMatrix %3 4
 %5 = OpSpecConstantComposite %3 %2 %2 %2 %2
 %6 = OpTypePointer Uniform %1
 %7 = OpVariable %6 Uniform
 %8 = OpSpecConstantComposite %4 %5 %5 %5 %7)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '7[%7]' is "
                        "not a constant composite or undef."));
}

// Invalid: Composite contains a column that is *not* a vector (it's an array)
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixColTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeInt 32 0
 %3 = OpSpecConstant %2 4
 %4 = OpConstant %1 0.0
 %5 = OpTypeVector %1 4
 %6 = OpTypeArray %2 %3
 %7 = OpTypeMatrix %5 4
 %8  = OpSpecConstantComposite %6 %3 %3 %3 %3
 %9  = OpSpecConstantComposite %5 %4 %4 %4 %4
 %10 = OpSpecConstantComposite %7 %9 %9 %9 %8)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '8[%8]' type "
                        "does not match Result Type <id> '7[%mat4v4float]'s "
                        "matrix column type."));
}

// Invalid: Matrix with an Undef column of the wrong size.
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeMatrixConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeVector %1 3
 %4 = OpTypeMatrix %2 4
 %5 = OpSpecConstant %1 1.0
 %6 = OpSpecConstant %1 0.0
 %7 = OpSpecConstantComposite %2 %5 %6 %6 %6
 %8 = OpSpecConstantComposite %2 %6 %5 %6 %6
 %9 = OpSpecConstantComposite %2 %6 %6 %5 %6
 %10 = OpUndef %3
 %11 = OpSpecConstantComposite %4 %7 %8 %9 %10)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '10[%10]' "
                        "vector component count does not match Result Type "
                        "<id> '4[%mat4v4float]'s vector component count."));
}

// Invalid: Matrix in which some columns are Int and some are Float.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeMatrixColumnTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeInt 32 0
 %2 = OpTypeFloat 32
 %3 = OpTypeVector %1 2
 %4 = OpTypeVector %2 2
 %5 = OpTypeMatrix %4 2
 %6 = OpSpecConstant %1 42
 %7 = OpConstant %2 3.14
 %8 = OpSpecConstantComposite %3 %6 %6
 %9 = OpSpecConstantComposite %4 %7 %7
%10 = OpSpecConstantComposite %5 %8 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '8[%8]' "
                        "component type does not match Result Type <id> "
                        "'5[%mat2v2float]'s matrix column component type."));
}

// Valid: Array of integers
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeArrayGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpSpecConstant %1 4
%5 = OpConstant %1 5
%3 = OpTypeArray %1 %2
%6 = OpTypeArray %1 %5
%4 = OpSpecConstantComposite %3 %2 %2 %2 %2
%7 = OpSpecConstantComposite %3 %5 %5 %5 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Expecting an array of 4 components, but 3 specified.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeArrayNumComponentsBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpSpecConstantComposite %3 %2 %2 %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent count does not "
                        "match Result Type <id> '3[%_arr_uint_uint_4]'s array "
                        "length."));
}

// Valid: Array of Integers and Undef-int
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeArrayWithUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpSpecConstant %1 4
%9 = OpUndef %1
%3 = OpTypeArray %1 %2
%4 = OpSpecConstantComposite %3 %2 %2 %2 %9)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Array uses a type as operand.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeArrayConstConstituentBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpTypePointer Uniform %1
%5 = OpVariable %4 Uniform
%6 = OpSpecConstantComposite %3 %2 %2 %2 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]' is "
                        "not a constant or undef."));
}

// Invalid: Array has a mix of Int and Float components.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeArrayConstituentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpTypeFloat 32
%5 = OpSpecConstant %4 3.14 ; bad type for const value
%6 = OpSpecConstantComposite %3 %2 %2 %2 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]'s "
                        "type does not match Result Type <id> "
                        "'3[%_arr_uint_uint_4]'s array element type."));
}

// Invalid: Array has a mix of Int and Undef-float.
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeArrayConstituentUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpSpecConstant %1 4
%3 = OpTypeArray %1 %2
%5 = OpTypeFloat 32
%6 = OpUndef %5 ; bad type for undef
%4 = OpSpecConstantComposite %3 %2 %2 %2 %6)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]'s "
                        "type does not match Result Type <id> "
                        "'3[%_arr_uint_2]'s array element type."));
}

// Valid: Struct of {Int32,Int32,Int64}.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeStructGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpSpecConstant %2 4300000000
%6 = OpSpecConstantComposite %3 %4 %4 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: missing one int32 struct member.
TEST_F(ValidateIdWithMessage,
       OpSpecConstantCompositeStructMissingComponentBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%3 = OpTypeStruct %1 %1 %1
%4 = OpConstant %1 42
%5 = OpSpecConstant %1 430
%6 = OpSpecConstantComposite %3 %4 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> "
                        "'2[%_struct_2]' count does not match Result Type "
                        "<id> '2[%_struct_2]'s struct member count."));
}

// Valid: Struct uses Undef-int64.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeStructUndefGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpSpecConstant %1 42
%5 = OpUndef %2
%6 = OpSpecConstantComposite %3 %4 %4 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Composite contains non-const/undef component.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeStructNonConstBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpSpecConstant %1 42
%5 = OpUndef %2
%6 = OpTypePointer Uniform %1
%7 = OpVariable %6 Uniform
%8 = OpSpecConstantComposite %3 %4 %7 %5)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '7[%7]' is "
                        "not a constant or undef."));
}

// Invalid: Struct component type does not match expected specialization type.
// Second component was expected to be Int32, but got Int64.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeStructMemberTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpSpecConstant %2 4300000000
%6 = OpSpecConstantComposite %3 %4 %5 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]' type "
                        "does not match the Result Type <id> '3[%_struct_3]'s "
                        "member type."));
}

// Invalid: Undef-int64 used when Int32 was expected.
TEST_F(ValidateIdWithMessage, OpSpecConstantCompositeStructMemberUndefTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 0
%3 = OpTypeStruct %1 %1 %2
%4 = OpSpecConstant %1 42
%5 = OpUndef %2
%6 = OpSpecConstantComposite %3 %4 %5 %4)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantComposite Constituent <id> '5[%5]' type "
                        "does not match the Result Type <id> '3[%_struct_3]'s "
                        "member type."));
}

// TODO: OpSpecConstantOp

TEST_F(ValidateIdWithMessage, OpVariableGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Input %1
%3 = OpVariable %2 Input)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpVariableInitializerConstantGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Input %1
%3 = OpConstant %1 42
%4 = OpVariable %2 Input %3)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpVariableInitializerGlobalVariableGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Uniform %1
%3 = OpVariable %2 Uniform
%4 = OpTypePointer Private %2 ; pointer to pointer
%5 = OpVariable %4 Private %3
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
// TODO: Positive test OpVariable with OpConstantNull of OpTypePointer
TEST_F(ValidateIdWithMessage, OpVariableResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpVariable %1 Input)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVariable Result Type <id> '1[%uint]' is not a pointer "
                "type."));
}
TEST_F(ValidateIdWithMessage, OpVariableInitializerIsTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Input %1
%3 = OpVariable %2 Input %2)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 2[%_ptr_Input_uint] "
                                               "cannot be a type"));
}

TEST_F(ValidateIdWithMessage, OpVariableInitializerIsFunctionVarBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ptrint = OpTypePointer Function %int
%ptrptrint = OpTypePointer Function %ptrint
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%var = OpVariable %ptrint Function
%varinit = OpVariable %ptrptrint Function %var ; Can't initialize function variable.
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpVariable Initializer <id> '8[%8]' is not a constant "
                        "or module-scope variable"));
}

TEST_F(ValidateIdWithMessage, OpVariableInitializerIsModuleVarGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ptrint = OpTypePointer Uniform %int
%mvar = OpVariable %ptrint Uniform
%ptrptrint = OpTypePointer Function %ptrint
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%goodvar = OpVariable %ptrptrint Function %mvar ; This is ok
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariableContainsBoolBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%bool = OpTypeBool
%int = OpTypeInt 32 0
%block = OpTypeStruct %bool %int
%_ptr_Uniform_block = OpTypePointer Uniform %block
%var = OpVariable %_ptr_Uniform_block Uniform
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%load = OpLoad %block %var
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("If OpTypeBool is stored in conjunction with OpVariable"
                        ", it can only be used with non-externally visible "
                        "shader Storage Classes: Workgroup, CrossWorkgroup, "
                        "Private, and Function"));
}

TEST_F(ValidateIdWithMessage, OpVariableContainsBoolPointerGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%bool = OpTypeBool
%boolptr = OpTypePointer Uniform %bool
%int = OpTypeInt 32 0
%block = OpTypeStruct %boolptr %int
%_ptr_Uniform_block = OpTypePointer Uniform %block
%var = OpVariable %_ptr_Uniform_block Uniform
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%load = OpLoad %block %var
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariableContainsBuiltinBoolGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpMemberDecorate %input 0 BuiltIn FrontFacing
%bool = OpTypeBool
%input = OpTypeStruct %bool
%_ptr_input = OpTypePointer Input %input
%var = OpVariable %_ptr_input Input
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%load = OpLoad %input %var
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariableContainsRayPayloadBoolGood) {
  std::string spirv = R"(
OpCapability RayTracingNV
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_ray_tracing"
OpMemoryModel Logical GLSL450
%bool = OpTypeBool
%PerRayData = OpTypeStruct %bool
%_ptr_PerRayData = OpTypePointer RayPayloadNV %PerRayData
%var = OpVariable %_ptr_PerRayData RayPayloadNV
%void = OpTypeVoid
%fnty = OpTypeFunction %void
%main = OpFunction %void None %fnty
%entry = OpLabel
%load = OpLoad %PerRayData %var
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariablePointerNoVariablePointersBad) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%_ptr_workgroup_int = OpTypePointer Workgroup %int
%_ptr_function_ptr = OpTypePointer Function %_ptr_workgroup_int
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%var = OpVariable %_ptr_function_ptr Function
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "In Logical addressing, variables may not allocate a pointer type"));
}

TEST_F(ValidateIdWithMessage,
       OpVariablePointerNoVariablePointersRelaxedLogicalGood) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%_ptr_workgroup_int = OpTypePointer Workgroup %int
%_ptr_function_ptr = OpTypePointer Function %_ptr_workgroup_int
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%var = OpVariable %_ptr_function_ptr Function
OpReturn
OpFunctionEnd
)";

  auto options = getValidatorOptions();
  options->relax_logical_pointer = true;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpFunctionWithNonMemoryObject) {
  // DXC generates code that looks like when given something like:
  //   T t;
  //   t.s.fn_1();
  // This needs to be accepted before legalization takes place, so we
  // will include it with the relaxed logical pointer.

  const std::string spirv = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
               OpSource HLSL 600
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
  %_struct_5 = OpTypeStruct
  %_struct_6 = OpTypeStruct %_struct_5
%_ptr_Function__struct_6 = OpTypePointer Function %_struct_6
%_ptr_Function__struct_5 = OpTypePointer Function %_struct_5
         %23 = OpTypeFunction %void %_ptr_Function__struct_5
          %1 = OpFunction %void None %9
         %10 = OpLabel
         %11 = OpVariable %_ptr_Function__struct_6 Function
         %20 = OpAccessChain %_ptr_Function__struct_5 %11 %int_0
         %21 = OpFunctionCall %void %12 %20
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %void None %23
         %13 = OpFunctionParameter %_ptr_Function__struct_5
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto options = getValidatorOptions();
  options->relax_logical_pointer = true;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage,
       OpVariablePointerVariablePointersStorageBufferGood) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointersStorageBuffer
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%_ptr_workgroup_int = OpTypePointer Workgroup %int
%_ptr_function_ptr = OpTypePointer Function %_ptr_workgroup_int
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%var = OpVariable %_ptr_function_ptr Function
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariablePointerVariablePointersGood) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%_ptr_workgroup_int = OpTypePointer Workgroup %int
%_ptr_function_ptr = OpTypePointer Function %_ptr_workgroup_int
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%var = OpVariable %_ptr_function_ptr Function
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVariablePointerVariablePointersBad) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%_ptr_workgroup_int = OpTypePointer Workgroup %int
%_ptr_uniform_ptr = OpTypePointer Uniform %_ptr_workgroup_int
%var = OpVariable %_ptr_uniform_ptr Uniform
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("In Logical addressing with variable pointers, "
                        "variables that allocate pointers must be in Function "
                        "or Private storage classes"));
}

TEST_F(ValidateIdWithMessage, OpLoadGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypeFunction %1
 %5 = OpVariable %3 UniformConstant
 %6 = OpFunction %1 None %4
 %7 = OpLabel
 %8 = OpLoad %2 %5
 %9 = OpReturn
%10 = OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// TODO: Add tests that exercise VariablePointersStorageBuffer instead of
// VariablePointers.
void createVariablePointerSpirvProgram(std::ostringstream* spirv,
                                       std::string result_strategy,
                                       bool use_varptr_cap,
                                       bool add_helper_function) {
  *spirv << "OpCapability Shader ";
  if (use_varptr_cap) {
    *spirv << "OpCapability VariablePointers ";
    *spirv << "OpExtension \"SPV_KHR_variable_pointers\" ";
  }
  *spirv << "OpExtension \"SPV_KHR_storage_buffer_storage_class\" ";
  *spirv << R"(
    OpMemoryModel Logical GLSL450
    OpEntryPoint GLCompute %main "main"
    %void      = OpTypeVoid
    %voidf     = OpTypeFunction %void
    %bool      = OpTypeBool
    %i32       = OpTypeInt 32 1
    %f32       = OpTypeFloat 32
    %f32ptr    = OpTypePointer StorageBuffer %f32
    %i         = OpConstant %i32 1
    %zero      = OpConstant %i32 0
    %float_1   = OpConstant %f32 1.0
    %ptr1      = OpVariable %f32ptr StorageBuffer
    %ptr2      = OpVariable %f32ptr StorageBuffer
  )";
  if (add_helper_function) {
    *spirv << R"(
      ; ////////////////////////////////////////////////////////////
      ;;;; Function that returns a pointer
      ; ////////////////////////////////////////////////////////////
      %selector_func_type  = OpTypeFunction %f32ptr %bool %f32ptr %f32ptr
      %choose_input_func   = OpFunction %f32ptr None %selector_func_type
      %is_neg_param        = OpFunctionParameter %bool
      %first_ptr_param     = OpFunctionParameter %f32ptr
      %second_ptr_param    = OpFunctionParameter %f32ptr
      %selector_func_begin = OpLabel
      %result_ptr          = OpSelect %f32ptr %is_neg_param %first_ptr_param %second_ptr_param
      OpReturnValue %result_ptr
      OpFunctionEnd
    )";
  }
  *spirv << R"(
    %main      = OpFunction %void None %voidf
    %label     = OpLabel
  )";
  *spirv << result_strategy;
  *spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
}

// With the VariablePointer Capability, OpLoad should allow loading a
// VaiablePointer. In this test the variable pointer is obtained by an OpSelect
TEST_F(ValidateIdWithMessage, OpLoadVarPtrOpSelectGood) {
  std::string result_strategy = R"(
    %isneg     = OpSLessThan %bool %i %zero
    %varptr    = OpSelect %f32ptr %isneg %ptr1 %ptr2
    %result    = OpLoad %f32 %varptr
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    true /* Add VariablePointers Capability? */,
                                    false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Without the VariablePointers Capability, OpLoad will not allow loading
// through a variable pointer.
// Disabled since using OpSelect with pointers without VariablePointers will
// fail LogicalsPass.
TEST_F(ValidateIdWithMessage, DISABLED_OpLoadVarPtrOpSelectBad) {
  std::string result_strategy = R"(
    %isneg     = OpSLessThan %bool %i %zero
    %varptr    = OpSelect %f32ptr %isneg %ptr1 %ptr2
    %result    = OpLoad %f32 %varptr
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    false /* Add VariablePointers Capability?*/,
                                    false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("is not a logical pointer."));
}

// With the VariablePointer Capability, OpLoad should allow loading a
// VaiablePointer. In this test the variable pointer is obtained by an OpPhi
TEST_F(ValidateIdWithMessage, OpLoadVarPtrOpPhiGood) {
  std::string result_strategy = R"(
    %is_neg      = OpSLessThan %bool %i %zero
    OpSelectionMerge %end_label None
    OpBranchConditional %is_neg %take_ptr_1 %take_ptr_2
    %take_ptr_1 = OpLabel
    OpBranch      %end_label
    %take_ptr_2 = OpLabel
    OpBranch      %end_label
    %end_label  = OpLabel
    %varptr     = OpPhi %f32ptr %ptr1 %take_ptr_1 %ptr2 %take_ptr_2
    %result     = OpLoad %f32 %varptr
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    true /* Add VariablePointers Capability?*/,
                                    false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Without the VariablePointers Capability, OpPhi can have a pointer result
// type.
TEST_F(ValidateIdWithMessage, OpPhiBad) {
  std::string result_strategy = R"(
    %is_neg      = OpSLessThan %bool %i %zero
    OpSelectionMerge %end_label None
    OpBranchConditional %is_neg %take_ptr_1 %take_ptr_2
    %take_ptr_1 = OpLabel
    OpBranch      %end_label
    %take_ptr_2 = OpLabel
    OpBranch      %end_label
    %end_label  = OpLabel
    %varptr     = OpPhi %f32ptr %ptr1 %take_ptr_1 %ptr2 %take_ptr_2
    %result     = OpLoad %f32 %varptr
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    false /* Add VariablePointers Capability?*/,
                                    false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Using pointers with OpPhi requires capability "
                        "VariablePointers or VariablePointersStorageBuffer"));
}

// With the VariablePointer Capability, OpLoad should allow loading through a
// VaiablePointer. In this test the variable pointer is obtained from an
// OpFunctionCall (return value from a function)
TEST_F(ValidateIdWithMessage, OpLoadVarPtrOpFunctionCallGood) {
  std::ostringstream spirv;
  std::string result_strategy = R"(
    %isneg     = OpSLessThan %bool %i %zero
    %varptr    = OpFunctionCall %f32ptr %choose_input_func %isneg %ptr1 %ptr2
    %result    = OpLoad %f32 %varptr
  )";

  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    true /* Add VariablePointers Capability?*/,
                                    true /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpLoadResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpVariable %3 UniformConstant
%6 = OpFunction %1 None %4
%7 = OpLabel
%8 = OpLoad %3 %5
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpLoad Result Type <id> "
                        "'3[%_ptr_UniformConstant_uint]' does not match "
                        "Pointer <id> '5[%5]'s type."));
}

TEST_F(ValidateIdWithMessage, OpLoadPointerBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpFunction %1 None %4
%6 = OpLabel
%7 = OpLoad %2 %8
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  // Prove that SSA checks trigger for a bad Id value.
  // The next test case show the not-a-logical-pointer case.
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ID 8[%8] has not been "
                                               "defined"));
}

// Disabled as bitcasting type to object is now not valid.
TEST_F(ValidateIdWithMessage, DISABLED_OpLoadLogicalPointerBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFloat 32
%4 = OpTypePointer UniformConstant %2
%5 = OpTypePointer UniformConstant %3
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
%9 = OpBitcast %5 %4 ; Not valid in logical addressing
%10 = OpLoad %3 %9 ; Should trigger message
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  // Once we start checking bitcasts, we might catch that
  // as the error first, instead of catching it here.
  // I don't know if it's possible to generate a bad case
  // if/when the validator is complete.
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpLoad Pointer <id> '9' is not a logical pointer."));
}

TEST_F(ValidateIdWithMessage, OpStoreGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 Uniform
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %5
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpStorePointerBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 UniformConstant
%7 = OpConstant %2 0
%8 = OpFunction %1 None %4
%9 = OpLabel
     OpStore %7 %5
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Pointer <id> '7[%uint_0]' is not a logical "
                        "pointer."));
}

// Disabled as bitcasting type to object is now not valid.
TEST_F(ValidateIdWithMessage, DISABLED_OpStoreLogicalPointerBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFloat 32
%4 = OpTypePointer UniformConstant %2
%5 = OpTypePointer UniformConstant %3
%6 = OpTypeFunction %1
%7 = OpConstantNull %5
%8 = OpFunction %1 None %6
%9 = OpLabel
%10 = OpBitcast %5 %4 ; Not valid in logical addressing
%11 = OpStore %10 %7 ; Should trigger message
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Pointer <id> '10' is not a logical pointer."));
}

// Without the VariablePointer Capability, OpStore should may not store
// through a variable pointer.
// Disabled since using OpSelect with pointers without VariablePointers will
// fail LogicalsPass.
TEST_F(ValidateIdWithMessage, DISABLED_OpStoreVarPtrBad) {
  std::string result_strategy = R"(
    %isneg     = OpSLessThan %bool %i %zero
    %varptr    = OpSelect %f32ptr %isneg %ptr1 %ptr2
                 OpStore %varptr %float_1
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(
      &spirv, result_strategy, false /* Add VariablePointers Capability? */,
      false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("is not a logical pointer."));
}

// With the VariablePointer Capability, OpStore should allow storing through a
// variable pointer.
TEST_F(ValidateIdWithMessage, OpStoreVarPtrGood) {
  std::string result_strategy = R"(
    %isneg     = OpSLessThan %bool %i %zero
    %varptr    = OpSelect %f32ptr %isneg %ptr1 %ptr2
                 OpStore %varptr %float_1
  )";

  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv, result_strategy,
                                    true /* Add VariablePointers Capability? */,
                                    false /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpStoreObjectGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 Uniform
%7 = OpFunction %1 None %4
%8 = OpLabel
%9 = OpUndef %1
     OpStore %6 %9
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Object <id> '9[%9]'s type is void."));
}
TEST_F(ValidateIdWithMessage, OpStoreTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%9 = OpTypeFloat 32
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1
%5 = OpConstant %9 3.14
%6 = OpVariable %3 Uniform
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %5
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Pointer <id> '7[%7]'s type does not match "
                        "Object <id> '6[%float_3_1400001]'s type."));
}

// The next series of test check test a relaxation of the rules for stores to
// structs.  The first test checks that we get a failure when the option is not
// set to relax the rule.
// TODO: Add tests for layout compatible arrays and matricies when the validator
//       relaxes the rules for them as well.  Also need test to check for layout
//       decorations specific to those types.
TEST_F(ValidateIdWithMessage, OpStoreTypeBadStruct) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 0
     OpMemberDecorate %1 1 Offset 4
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 4
%3 = OpTypeVoid
%4 = OpTypeFloat 32
%1 = OpTypeStruct %4 %4
%5 = OpTypePointer Uniform %1
%2 = OpTypeStruct %4 %4
%6 = OpTypeFunction %3
%7 = OpConstant %4 3.14
%8 = OpVariable %5 Uniform
%9 = OpFunction %3 None %6
%10 = OpLabel
%11 = OpCompositeConstruct %2 %7 %7
      OpStore %8 %11
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Pointer <id> '8[%8]'s type does not match "
                        "Object <id> '11[%11]'s type."));
}

// Same code as the last test.  The difference is that we relax the rule.
// Because the structs %3 and %5 are defined the same way.
TEST_F(ValidateIdWithMessage, OpStoreTypeRelaxedStruct) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 0
     OpMemberDecorate %1 1 Offset 4
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 4
%3 = OpTypeVoid
%4 = OpTypeFloat 32
%1 = OpTypeStruct %4 %4
%5 = OpTypePointer Uniform %1
%2 = OpTypeStruct %4 %4
%6 = OpTypeFunction %3
%7 = OpConstant %4 3.14
%8 = OpVariable %5 Uniform
%9 = OpFunction %3 None %6
%10 = OpLabel
%11 = OpCompositeConstruct %2 %7 %7
      OpStore %8 %11
      OpReturn
      OpFunctionEnd)";
  spvValidatorOptionsSetRelaxStoreStruct(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Same code as the last test excect for an extra decoration on one of the
// members. With the relaxed rules, the code is still valid.
TEST_F(ValidateIdWithMessage, OpStoreTypeRelaxedStructWithExtraDecoration) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 0
     OpMemberDecorate %1 1 Offset 4
     OpMemberDecorate %1 0 RelaxedPrecision
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 4
%3 = OpTypeVoid
%4 = OpTypeFloat 32
%1 = OpTypeStruct %4 %4
%5 = OpTypePointer Uniform %1
%2 = OpTypeStruct %4 %4
%6 = OpTypeFunction %3
%7 = OpConstant %4 3.14
%8 = OpVariable %5 Uniform
%9 = OpFunction %3 None %6
%10 = OpLabel
%11 = OpCompositeConstruct %2 %7 %7
      OpStore %8 %11
      OpReturn
      OpFunctionEnd)";
  spvValidatorOptionsSetRelaxStoreStruct(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// This test check that we recursively traverse the struct to check if they are
// interchangable.
TEST_F(ValidateIdWithMessage, OpStoreTypeRelaxedNestedStruct) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 0
     OpMemberDecorate %1 1 Offset 4
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 8
     OpMemberDecorate %3 0 Offset 0
     OpMemberDecorate %3 1 Offset 4
     OpMemberDecorate %4 0 Offset 0
     OpMemberDecorate %4 1 Offset 8
%5 = OpTypeVoid
%6 = OpTypeInt 32 0
%7 = OpTypeFloat 32
%1 = OpTypeStruct %7 %6
%2 = OpTypeStruct %1 %1
%8 = OpTypePointer Uniform %2
%3 = OpTypeStruct %7 %6
%4 = OpTypeStruct %3 %3
%9 = OpTypeFunction %5
%10 = OpConstant %6 7
%11 = OpConstant %7 3.14
%12 = OpConstantComposite %3 %11 %10
%13 = OpVariable %8 Uniform
%14 = OpFunction %5 None %9
%15 = OpLabel
%16 = OpCompositeConstruct %4 %12 %12
      OpStore %13 %16
      OpReturn
      OpFunctionEnd)";
  spvValidatorOptionsSetRelaxStoreStruct(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// This test check that the even with the relaxed rules an error is identified
// if the members of the struct are in a different order.
TEST_F(ValidateIdWithMessage, OpStoreTypeBadRelaxedStruct1) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 0
     OpMemberDecorate %1 1 Offset 4
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 8
     OpMemberDecorate %3 0 Offset 0
     OpMemberDecorate %3 1 Offset 4
     OpMemberDecorate %4 0 Offset 0
     OpMemberDecorate %4 1 Offset 8
%5 = OpTypeVoid
%6 = OpTypeInt 32 0
%7 = OpTypeFloat 32
%1 = OpTypeStruct %6 %7
%2 = OpTypeStruct %1 %1
%8 = OpTypePointer Uniform %2
%3 = OpTypeStruct %7 %6
%4 = OpTypeStruct %3 %3
%9 = OpTypeFunction %5
%10 = OpConstant %6 7
%11 = OpConstant %7 3.14
%12 = OpConstantComposite %3 %11 %10
%13 = OpVariable %8 Uniform
%14 = OpFunction %5 None %9
%15 = OpLabel
%16 = OpCompositeConstruct %4 %12 %12
      OpStore %13 %16
      OpReturn
      OpFunctionEnd)";
  spvValidatorOptionsSetRelaxStoreStruct(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpStore Pointer <id> '13[%13]'s layout does not match Object "
                "<id> '16[%16]'s layout."));
}

// This test check that the even with the relaxed rules an error is identified
// if the members of the struct are at different offsets.
TEST_F(ValidateIdWithMessage, OpStoreTypeBadRelaxedStruct2) {
  std::string spirv = kGLSL450MemoryModel + R"(
     OpMemberDecorate %1 0 Offset 4
     OpMemberDecorate %1 1 Offset 0
     OpMemberDecorate %2 0 Offset 0
     OpMemberDecorate %2 1 Offset 8
     OpMemberDecorate %3 0 Offset 0
     OpMemberDecorate %3 1 Offset 4
     OpMemberDecorate %4 0 Offset 0
     OpMemberDecorate %4 1 Offset 8
%5 = OpTypeVoid
%6 = OpTypeInt 32 0
%7 = OpTypeFloat 32
%1 = OpTypeStruct %7 %6
%2 = OpTypeStruct %1 %1
%8 = OpTypePointer Uniform %2
%3 = OpTypeStruct %7 %6
%4 = OpTypeStruct %3 %3
%9 = OpTypeFunction %5
%10 = OpConstant %6 7
%11 = OpConstant %7 3.14
%12 = OpConstantComposite %3 %11 %10
%13 = OpVariable %8 Uniform
%14 = OpFunction %5 None %9
%15 = OpLabel
%16 = OpCompositeConstruct %4 %12 %12
      OpStore %13 %16
      OpReturn
      OpFunctionEnd)";
  spvValidatorOptionsSetRelaxStoreStruct(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpStore Pointer <id> '13[%13]'s layout does not match Object "
                "<id> '16[%16]'s layout."));
}

TEST_F(ValidateIdWithMessage, OpStoreTypeRelaxedLogicalPointerReturnPointer) {
  const std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 1
%2 = OpTypePointer Function %1
%3 = OpTypeFunction %2 %2
%4 = OpFunction %2 None %3
%5 = OpFunctionParameter %2
%6 = OpLabel
     OpReturnValue %5
     OpFunctionEnd)";

  spvValidatorOptionsSetRelaxLogicalPointer(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpStoreTypeRelaxedLogicalPointerAllocPointer) {
  const std::string spirv = R"(
      OpCapability Shader
      OpCapability Linkage
      OpMemoryModel Logical GLSL450
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 1
 %3 = OpTypeFunction %1          ; void(void)
 %4 = OpTypePointer Uniform %2   ; int*
 %5 = OpTypePointer Private %4   ; int** (Private)
 %6 = OpTypePointer Function %4  ; int** (Function)
 %7 = OpVariable %5 Private
 %8 = OpFunction %1 None %3
 %9 = OpLabel
%10 = OpVariable %6 Function
      OpReturn
      OpFunctionEnd)";

  spvValidatorOptionsSetRelaxLogicalPointer(options_, true);
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpStoreVoid) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1
%6 = OpVariable %3 Uniform
%7 = OpFunction %1 None %4
%8 = OpLabel
%9 = OpFunctionCall %1 %7
     OpStore %6 %9
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpStore Object <id> '8[%8]'s type is void."));
}

TEST_F(ValidateIdWithMessage, OpStoreLabel) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1
%6 = OpVariable %3 Uniform
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %8
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 7[%7] requires a type"));
}

// TODO: enable when this bug is fixed:
// https://cvs.khronos.org/bugzilla/show_bug.cgi?id=15404
TEST_F(ValidateIdWithMessage, DISABLED_OpStoreFunction) {
  std::string spirv = kGLSL450MemoryModel + R"(
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %2
%5 = OpConstant %2 123
%6 = OpVariable %3 UniformConstant
%7 = OpFunction %2 None %4
%8 = OpLabel
     OpStore %6 %7
     OpReturnValue %5
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpStoreBuiltin) {
  std::string spirv = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 1 1 1
               OpSource GLSL 450
               OpName %main "main"

               OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId

        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input

       %zero = OpConstant %uint 0
 %v3uint_000 = OpConstantComposite %v3uint %zero %zero %zero

       %void = OpTypeVoid
   %voidfunc = OpTypeFunction %void
       %main = OpFunction %void None %voidfunc
      %lmain = OpLabel

               OpStore %gl_GlobalInvocationID %v3uint_000

               OpReturn
               OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("storage class is read-only"));
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpConstant %2 42
 %5 = OpVariable %3 UniformConstant %4
 %6 = OpTypePointer Function %2
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %6 Function
      OpCopyMemory %10 %5 None
      OpReturn
      OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryNonPointerTarget) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1 %2 %3
%5 = OpFunction %1 None %4
%6 = OpFunctionParameter %2
%7 = OpFunctionParameter %3
%8 = OpLabel
OpCopyMemory %6 %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Target operand <id> '6[%6]' is not a pointer."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryNonPointerSource) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpTypeFunction %1 %2 %3
%5 = OpFunction %1 None %4
%6 = OpFunctionParameter %2
%7 = OpFunctionParameter %3
%8 = OpLabel
OpCopyMemory %7 %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Source operand <id> '6[%6]' is not a pointer."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpConstant %2 42
 %5 = OpVariable %3 UniformConstant %4
%11 = OpTypeFloat 32
 %6 = OpTypePointer Function %11
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %6 Function
      OpCopyMemory %10 %5 None
      OpReturn
      OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Target <id> '5[%5]'s type does not match "
                        "Source <id> '2[%uint]'s type."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryVoidTarget) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %1
%4 = OpTypePointer Uniform %2
%5 = OpTypeFunction %1 %3 %4
%6 = OpFunction %1 None %5
%7 = OpFunctionParameter %3
%8 = OpFunctionParameter %4
%9 = OpLabel
OpCopyMemory %7 %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Target operand <id> '7[%7]' cannot be a void "
                        "pointer."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemoryVoidSource) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %1
%4 = OpTypePointer Uniform %2
%5 = OpTypeFunction %1 %3 %4
%6 = OpFunction %1 None %5
%7 = OpFunctionParameter %3
%8 = OpFunctionParameter %4
%9 = OpLabel
OpCopyMemory %8 %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Source operand <id> '7[%7]' cannot be a void "
                        "pointer."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %5 None
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpCopyMemorySizedTargetBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypePointer Function %2
%5 = OpConstant %2 4
%6 = OpVariable %3 UniformConstant %5
%7 = OpTypeFunction %1
%8 = OpFunction %1 None %7
%9 = OpLabel
     OpCopyMemorySized %5 %5 %5 None
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Target operand <id> '5[%uint_4]' is not a pointer."));
}
TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSourceBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypePointer Function %2
%5 = OpConstant %2 4
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
%9 = OpVariable %4 Function
     OpCopyMemorySized %9 %5 %5 None
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Source operand <id> '5[%uint_4]' is not a pointer."));
}
TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %6 None
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Size operand <id> '6[%6]' must be a scalar integer type."));
}
TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
%11 = OpTypeFloat 32
%12 = OpConstant %11 1.0
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %12 None
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Size operand <id> '9[%float_1]' must be a scalar integer "
                "type."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeConstantNull) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstantNull %2
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Size operand <id> '3[%3]' cannot be a constant "
                        "zero."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeConstantZero) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Size operand <id> '3[%uint_0]' cannot be a constant "
                        "zero."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeConstantZero64) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 64 0
%3 = OpConstant %2 0
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Size operand <id> '3[%ulong_0]' cannot be a constant "
                        "zero."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeConstantNegative) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpConstant %2 -1
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Size operand <id> '3[%int_n1]' cannot have the sign bit set "
                "to 1."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeConstantNegative64) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 64 1
%3 = OpConstant %2 -1
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Size operand <id> '3[%long_n1]' cannot have the sign bit set "
                "to 1."));
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeUnsignedNegative) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 2147483648
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpCopyMemorySizedSizeUnsignedNegative64) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 64 0
%3 = OpConstant %2 9223372036854775808
%4 = OpTypePointer Uniform %2
%5 = OpTypeFloat 32
%6 = OpTypePointer UniformConstant %5
%7 = OpTypeFunction %1 %4 %6
%8 = OpFunction %1 None %7
%9 = OpFunctionParameter %4
%10 = OpFunctionParameter %6
%11 = OpLabel
OpCopyMemorySized %9 %10 %3
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

const char kDeeplyNestedStructureSetup[] = R"(
%void = OpTypeVoid
%void_f  = OpTypeFunction %void
%int = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%mat4x3 = OpTypeMatrix %v3float 4
%_ptr_Private_mat4x3 = OpTypePointer Private %mat4x3
%_ptr_Private_float = OpTypePointer Private %float
%my_matrix = OpVariable %_ptr_Private_mat4x3 Private
%my_float_var = OpVariable %_ptr_Private_float Private
%_ptr_Function_float = OpTypePointer Function %float
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%int_5 = OpConstant %int 5

; Making the following nested structures.
;
; struct S {
;   bool b;
;   vec4 v[5];
;   int i;
;   mat4x3 m[5];
; }
; uniform blockName {
;   S s;
;   bool cond;
;   RunTimeArray arr;
; }

%f32arr = OpTypeRuntimeArray %float
%v4float = OpTypeVector %float 4
%array5_mat4x3 = OpTypeArray %mat4x3 %int_5
%array5_vec4 = OpTypeArray %v4float %int_5
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_vec4 = OpTypePointer Function %v4float
%_ptr_Uniform_vec4 = OpTypePointer Uniform %v4float
%struct_s = OpTypeStruct %int %array5_vec4 %int %array5_mat4x3
%struct_blockName = OpTypeStruct %struct_s %int %f32arr
%_ptr_Uniform_blockName = OpTypePointer Uniform %struct_blockName
%_ptr_Uniform_struct_s = OpTypePointer Uniform %struct_s
%_ptr_Uniform_array5_mat4x3 = OpTypePointer Uniform %array5_mat4x3
%_ptr_Uniform_mat4x3 = OpTypePointer Uniform %mat4x3
%_ptr_Uniform_v3float = OpTypePointer Uniform %v3float
%blockName_var = OpVariable %_ptr_Uniform_blockName Uniform
%spec_int = OpSpecConstant %int 2
%float_0 = OpConstant %float 0
%func = OpFunction %void None %void_f
%my_label = OpLabel
)";

// In what follows, Access Chain Instruction refers to one of the following:
// OpAccessChain, OpInBoundsAccessChain, OpPtrAccessChain, and
// OpInBoundsPtrAccessChain
using AccessChainInstructionTest = spvtest::ValidateBase<std::string>;

// Determines whether the access chain instruction requires the 'element id'
// argument.
bool AccessChainRequiresElemId(const std::string& instr) {
  return (instr == "OpPtrAccessChain" || instr == "OpInBoundsPtrAccessChain");
}

// Valid: Access a float in a matrix using an access chain instruction.
TEST_P(AccessChainInstructionTest, AccessChainGood) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup +
                      "%float_entry = " + instr +
                      R"( %_ptr_Private_float %my_matrix )" + elem +
                      R"(%int_0 %int_1
              OpReturn
              OpFunctionEnd
          )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid. The result type of an access chain instruction must be a pointer.
TEST_P(AccessChainInstructionTest, AccessChainResultTypeBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%float_entry = )" +
                      instr +
                      R"( %float %my_matrix )" + elem +
                      R"(%int_0 %int_1
OpReturn
OpFunctionEnd
  )";

  const std::string expected_err = "The Result Type of " + instr +
                                   " <id> '36[%36]' must be "
                                   "OpTypePointer. Found OpTypeFloat.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid. The base type of an access chain instruction must be a pointer.
TEST_P(AccessChainInstructionTest, AccessChainBaseTypeVoidBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%float_entry = )" +
                      instr + " %_ptr_Private_float %void " + elem +
                      R"(%int_0 %int_1
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 1[%void] cannot be a "
                                               "type"));
}

// Invalid. The base type of an access chain instruction must be a pointer.
TEST_P(AccessChainInstructionTest, AccessChainBaseTypeNonPtrVariableBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_float %_ptr_Private_float )" +
                      elem +
                      R"(%int_0 %int_1
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 8[%_ptr_Private_float] cannot be a type"));
}

// Invalid: The storage class of Base and Result do not match.
TEST_P(AccessChainInstructionTest,
       AccessChainResultAndBaseStorageClassDoesntMatchBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Function_float %my_matrix )" + elem +
                      R"(%int_0 %int_1
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err =
      "The result pointer storage class and base pointer storage class in " +
      instr + " do not match.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid. The base type of an access chain instruction must point to a
// composite object.
TEST_P(AccessChainInstructionTest,
       AccessChainBasePtrNotPointingToCompositeBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_float %my_float_var )" + elem +
                      R"(%int_0
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = instr +
                                   " reached non-composite type while "
                                   "indexes still remain to be traversed.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Valid. No Indexes were passed to the access chain instruction. The Result
// Type is the same as the Base type.
TEST_P(AccessChainInstructionTest, AccessChainNoIndexesGood) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_float %my_float_var )" + elem +
                      R"(
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid. No Indexes were passed to the access chain instruction, but the
// Result Type is different from the Base type.
TEST_P(AccessChainInstructionTest, AccessChainNoIndexesBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_mat4x3 %my_float_var )" + elem +
                      R"(
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("result type (OpTypeMatrix) does not match the type that "
                "results from indexing into the base <id> (OpTypeFloat)."));
}

// Valid: 255 indexes passed to the access chain instruction. Limit is 255.
TEST_P(AccessChainInstructionTest, AccessChainTooManyIndexesGood) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? " %int_0 " : "";
  int depth = 255;
  std::string header = kGLSL450MemoryModel + kDeeplyNestedStructureSetup;
  header.erase(header.find("%func"));
  std::ostringstream spirv;
  spirv << header << "\n";

  // Build nested structures. Struct 'i' contains struct 'i-1'
  spirv << "%s_depth_1 = OpTypeStruct %float\n";
  for (int i = 2; i <= depth; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %s_depth_" << i - 1 << "\n";
  }

  // Define Pointer and Variable to use for the AccessChain instruction.
  spirv << "%_ptr_Uniform_deep_struct = OpTypePointer Uniform %s_depth_"
        << depth << "\n";
  spirv << "%deep_var = OpVariable %_ptr_Uniform_deep_struct Uniform\n";

  // Function Start
  spirv << R"(
  %func = OpFunction %void None %void_f
  %my_label = OpLabel
  )";

  // AccessChain with 'n' indexes (n = depth)
  spirv << "%entry = " << instr << " %_ptr_Uniform_float %deep_var" << elem;
  for (int i = 0; i < depth; ++i) {
    spirv << " %int_0";
  }

  // Function end
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: 256 indexes passed to the access chain instruction. Limit is 255.
TEST_P(AccessChainInstructionTest, AccessChainTooManyIndexesBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? " %int_0 " : "";
  std::ostringstream spirv;
  spirv << kGLSL450MemoryModel << kDeeplyNestedStructureSetup;
  spirv << "%entry = " << instr << " %_ptr_Private_float %my_matrix" << elem;
  for (int i = 0; i < 256; ++i) {
    spirv << " %int_0";
  }
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  const std::string expected_err = "The number of indexes in " + instr +
                                   " may not exceed 255. Found 256 indexes.";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Valid: 10 indexes passed to the access chain instruction. (Custom limit: 10)
TEST_P(AccessChainInstructionTest, CustomizedAccessChainTooManyIndexesGood) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? " %int_0 " : "";
  int depth = 10;
  std::string header = kGLSL450MemoryModel + kDeeplyNestedStructureSetup;
  header.erase(header.find("%func"));
  std::ostringstream spirv;
  spirv << header << "\n";

  // Build nested structures. Struct 'i' contains struct 'i-1'
  spirv << "%s_depth_1 = OpTypeStruct %float\n";
  for (int i = 2; i <= depth; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %s_depth_" << i - 1 << "\n";
  }

  // Define Pointer and Variable to use for the AccessChain instruction.
  spirv << "%_ptr_Uniform_deep_struct = OpTypePointer Uniform %s_depth_"
        << depth << "\n";
  spirv << "%deep_var = OpVariable %_ptr_Uniform_deep_struct Uniform\n";

  // Function Start
  spirv << R"(
  %func = OpFunction %void None %void_f
  %my_label = OpLabel
  )";

  // AccessChain with 'n' indexes (n = depth)
  spirv << "%entry = " << instr << " %_ptr_Uniform_float %deep_var" << elem;
  for (int i = 0; i < depth; ++i) {
    spirv << " %int_0";
  }

  // Function end
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_access_chain_indexes, 10u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: 11 indexes passed to the access chain instruction. Custom Limit:10
TEST_P(AccessChainInstructionTest, CustomizedAccessChainTooManyIndexesBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? " %int_0 " : "";
  std::ostringstream spirv;
  spirv << kGLSL450MemoryModel << kDeeplyNestedStructureSetup;
  spirv << "%entry = " << instr << " %_ptr_Private_float %my_matrix" << elem;
  for (int i = 0; i < 11; ++i) {
    spirv << " %int_0";
  }
  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";
  const std::string expected_err = "The number of indexes in " + instr +
                                   " may not exceed 10. Found 11 indexes.";
  spvValidatorOptionsSetUniversalLimit(
      options_, spv_validator_limit_max_access_chain_indexes, 10u);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: Index passed to the access chain instruction is float (must be
// integer).
TEST_P(AccessChainInstructionTest, AccessChainUndefinedIndexBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_float %my_matrix )" + elem +
                      R"(%float_0 %int_1
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err =
      "Indexes passed to " + instr + " must be of type integer.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: The index argument that indexes into a struct must be of type
// OpConstant.
TEST_P(AccessChainInstructionTest, AccessChainStructIndexNotConstantBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%f = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_0 %spec_int %int_2
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err =
      "The <id> passed to " + instr +
      " to index into a structure must be an OpConstant.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: Indexing up to a vec4 granularity, but result type expected float.
TEST_P(AccessChainInstructionTest,
       AccessChainStructResultTypeDoesntMatchIndexedTypeBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_0 %int_1 %int_2
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = instr +
                                   " result type (OpTypeFloat) does not match "
                                   "the type that results from indexing into "
                                   "the base <id> (OpTypeVector).";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: Reach non-composite type (bool) when unused indexes remain.
TEST_P(AccessChainInstructionTest, AccessChainStructTooManyIndexesBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_0 %int_2 %int_2
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = instr +
                                   " reached non-composite type while "
                                   "indexes still remain to be traversed.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: Trying to find index 3 of the struct that has only 3 members.
TEST_P(AccessChainInstructionTest, AccessChainStructIndexOutOfBoundBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_3 %int_2 %int_2
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = "Index is out of bounds: " + instr +
                                   " can not find index 3 into the structure "
                                   "<id> '25[%_struct_25]'. This structure "
                                   "has 3 members. Largest valid index is 2.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Valid: Tests that we can index into Struct, Array, Matrix, and Vector!
TEST_P(AccessChainInstructionTest, AccessChainIndexIntoAllTypesGood) {
  // indexes that we are passing are: 0, 3, 1, 2, 0
  // 0 will select the struct_s within the base struct (blockName)
  // 3 will select the Array that contains 5 matrices
  // 1 will select the Matrix that is at index 1 of the array
  // 2 will select the column (which is a vector) within the matrix at index 2
  // 0 will select the element at the index 0 of the vector. (which is a float).
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::ostringstream spirv;
  spirv << kGLSL450MemoryModel << kDeeplyNestedStructureSetup << std::endl;
  spirv << "%ss = " << instr << " %_ptr_Uniform_struct_s %blockName_var "
        << elem << "%int_0" << std::endl;
  spirv << "%sa = " << instr << " %_ptr_Uniform_array5_mat4x3 %blockName_var "
        << elem << "%int_0 %int_3" << std::endl;
  spirv << "%sm = " << instr << " %_ptr_Uniform_mat4x3 %blockName_var " << elem
        << "%int_0 %int_3 %int_1" << std::endl;
  spirv << "%sc = " << instr << " %_ptr_Uniform_v3float %blockName_var " << elem
        << "%int_0 %int_3 %int_1 %int_2" << std::endl;
  spirv << "%entry = " << instr << " %_ptr_Uniform_float %blockName_var "
        << elem << "%int_0 %int_3 %int_1 %int_2 %int_0" << std::endl;
  spirv << R"(
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Valid: Access an element of OpTypeRuntimeArray.
TEST_P(AccessChainInstructionTest, AccessChainIndexIntoRuntimeArrayGood) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%runtime_arr_entry = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_2 %int_0
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Unused index when accessing OpTypeRuntimeArray.
TEST_P(AccessChainInstructionTest, AccessChainIndexIntoRuntimeArrayBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%runtime_arr_entry = )" +
                      instr + R"( %_ptr_Uniform_float %blockName_var )" + elem +
                      R"(%int_2 %int_0 %int_1
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err =
      instr +
      " reached non-composite type while indexes still remain to be traversed.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: Reached scalar type before arguments to the access chain instruction
// finished.
TEST_P(AccessChainInstructionTest, AccessChainMatrixMoreArgsThanNeededBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_float %my_matrix )" + elem +
                      R"(%int_0 %int_1 %int_0
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = instr +
                                   " reached non-composite type while "
                                   "indexes still remain to be traversed.";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Invalid: The result type and the type indexed into do not match.
TEST_P(AccessChainInstructionTest,
       AccessChainResultTypeDoesntMatchIndexedTypeBad) {
  const std::string instr = GetParam();
  const std::string elem = AccessChainRequiresElemId(instr) ? "%int_0 " : "";
  std::string spirv = kGLSL450MemoryModel + kDeeplyNestedStructureSetup + R"(
%entry = )" +
                      instr + R"( %_ptr_Private_mat4x3 %my_matrix )" + elem +
                      R"(%int_0 %int_1
OpReturn
OpFunctionEnd
  )";
  const std::string expected_err = instr +
                                   " result type (OpTypeMatrix) does not match "
                                   "the type that results from indexing into "
                                   "the base <id> (OpTypeFloat).";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected_err));
}

// Run tests for Access Chain Instructions.
INSTANTIATE_TEST_SUITE_P(
    CheckAccessChainInstructions, AccessChainInstructionTest,
    ::testing::Values("OpAccessChain", "OpInBoundsAccessChain",
                      "OpPtrAccessChain", "OpInBoundsPtrAccessChain"));

// TODO: OpArrayLength
// TODO: OpImagePointer
// TODO: OpGenericPtrMemSemantics

TEST_F(ValidateIdWithMessage, OpFunctionGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2 %2
%4 = OpFunction %1 None %3
%5 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpFunctionResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 42
%4 = OpTypeFunction %1 %2 %2
%5 = OpFunction %2 None %4
%6 = OpLabel
     OpReturnValue %3
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpFunction Result Type <id> '2[%uint]' does not "
                        "match the Function Type's return type <id> "
                        "'1[%void]'."));
}
TEST_F(ValidateIdWithMessage, OpReturnValueTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFloat 32
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpFunction %1 None %4
%6 = OpLabel
     OpReturnValue %3
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpReturnValue Value <id> '3[%float_0]'s type does "
                        "not match OpFunction's return type."));
}
TEST_F(ValidateIdWithMessage, OpFunctionFunctionTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%4 = OpFunction %1 None %2
%5 = OpLabel
     OpReturn
OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpFunction Function Type <id> '2[%uint]' is not a function "
                "type."));
}

TEST_F(ValidateIdWithMessage, OpFunctionUseBad) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeFloat 32
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturnValue %3
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid use of function result id 3[%3]."));
}

TEST_F(ValidateIdWithMessage, OpFunctionParameterGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2
%4 = OpFunction %1 None %3
%5 = OpFunctionParameter %2
%6 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpFunctionParameterMultipleGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2 %2
%4 = OpFunction %1 None %3
%5 = OpFunctionParameter %2
%6 = OpFunctionParameter %2
%7 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpFunctionParameterResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2
%4 = OpFunction %1 None %3
%5 = OpFunctionParameter %1
%6 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpFunctionParameter Result Type <id> '1[%void]' does not "
                "match the OpTypeFunction parameter type of the same index."));
}

TEST_F(ValidateIdWithMessage, OpFunctionCallGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
     OpReturnValue %7
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
%12 = OpFunctionCall %2 %6 %5
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpFunctionCallResultTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpIAdd %2 %7 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
%12 = OpFunctionCall %1 %6 %5
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpFunctionCall Result Type <id> '1[%void]'s type "
                        "does not match Function <id> '2[%uint]'s return "
                        "type."));
}
TEST_F(ValidateIdWithMessage, OpFunctionCallFunctionBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%10 = OpFunction %1 None %4
%11 = OpLabel
%12 = OpFunctionCall %2 %5 %5
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpFunctionCall Function <id> '5[%uint_42]' is not a "
                        "function."));
}
TEST_F(ValidateIdWithMessage, OpFunctionCallArgumentTypeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42

%13 = OpTypeFloat 32
%14 = OpConstant %13 3.14

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpIAdd %2 %7 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
%12 = OpFunctionCall %2 %6 %14
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpFunctionCall Argument <id> '7[%float_3_1400001]'s "
                        "type does not match Function <id> '2[%uint]'s "
                        "parameter type."));
}

// Valid: OpSampledImage result <id> is used in the same block by
// OpImageSampleImplictLod
TEST_F(ValidateIdWithMessage, OpSampledImageGood) {
  std::string spirv = kGLSL450MemoryModel + sampledImageSetup + R"(
%smpld_img = OpSampledImage %sampled_image_type %image_inst %sampler_inst
%si_lod    = OpImageSampleImplicitLod %v4float %smpld_img %const_vec_1_1
    OpReturn
    OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: OpSampledImage result <id> is defined in one block and used in a
// different block.
TEST_F(ValidateIdWithMessage, OpSampledImageUsedInDifferentBlockBad) {
  std::string spirv = kGLSL450MemoryModel + sampledImageSetup + R"(
%smpld_img = OpSampledImage %sampled_image_type %image_inst %sampler_inst
OpBranch %label_2
%label_2 = OpLabel
%si_lod  = OpImageSampleImplicitLod %v4float %smpld_img %const_vec_1_1
OpReturn
OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("All OpSampledImage instructions must be in the same block in "
                "which their Result <id> are consumed. OpSampledImage Result "
                "Type <id> '23[%23]' has a consumer in a different basic "
                "block. The consumer instruction <id> is '25[%25]'."));
}

// Invalid: OpSampledImage result <id> is used by OpSelect
// Note: According to the Spec, OpSelect parameters must be either a scalar or a
// vector. Therefore, OpTypeSampledImage is an illegal parameter for OpSelect.
// However, the OpSelect validation does not catch this today. Therefore, it is
// caught by the OpSampledImage validation. If the OpSelect validation code is
// updated, the error message for this test may change.
//
// Disabled since OpSelect catches this now.
TEST_F(ValidateIdWithMessage, DISABLED_OpSampledImageUsedInOpSelectBad) {
  std::string spirv = kGLSL450MemoryModel + sampledImageSetup + R"(
%smpld_img  = OpSampledImage %sampled_image_type %image_inst %sampler_inst
%select_img = OpSelect %sampled_image_type %spec_true %smpld_img %smpld_img
OpReturn
OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result <id> from OpSampledImage instruction must not "
                        "appear as operands of OpSelect. Found result <id> "
                        "'23' as an operand of <id> '24'."));
}

// Valid: Get a float in a matrix using CompositeExtract.
// Valid: Insert float into a matrix using CompositeInsert.
TEST_F(ValidateIdWithMessage, CompositeExtractInsertGood) {
  std::ostringstream spirv;
  spirv << kGLSL450MemoryModel << kDeeplyNestedStructureSetup << std::endl;
  spirv << "%matrix = OpLoad %mat4x3 %my_matrix" << std::endl;
  spirv << "%float_entry = OpCompositeExtract  %float %matrix 0 1" << std::endl;

  // To test CompositeInsert, insert the object back in after extraction.
  spirv << "%new_composite = OpCompositeInsert %mat4x3 %float_entry %matrix 0 1"
        << std::endl;
  spirv << R"(OpReturn
              OpFunctionEnd)";
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

#if 0
TEST_F(ValidateIdWithMessage, OpFunctionCallArgumentCountBar) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpLoad %2 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %2 %6 %5
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
#endif

// TODO: The many things that changed with how images are used.
// TODO: OpTextureSample
// TODO: OpTextureSampleDref
// TODO: OpTextureSampleLod
// TODO: OpTextureSampleProj
// TODO: OpTextureSampleGrad
// TODO: OpTextureSampleOffset
// TODO: OpTextureSampleProjLod
// TODO: OpTextureSampleProjGrad
// TODO: OpTextureSampleLodOffset
// TODO: OpTextureSampleProjOffset
// TODO: OpTextureSampleGradOffset
// TODO: OpTextureSampleProjLodOffset
// TODO: OpTextureSampleProjGradOffset
// TODO: OpTextureFetchTexelLod
// TODO: OpTextureFetchTexelOffset
// TODO: OpTextureFetchSample
// TODO: OpTextureFetchTexel
// TODO: OpTextureGather
// TODO: OpTextureGatherOffset
// TODO: OpTextureGatherOffsets
// TODO: OpTextureQuerySizeLod
// TODO: OpTextureQuerySize
// TODO: OpTextureQueryLevels
// TODO: OpTextureQuerySamples
// TODO: OpConvertUToF
// TODO: OpConvertFToS
// TODO: OpConvertSToF
// TODO: OpConvertUToF
// TODO: OpUConvert
// TODO: OpSConvert
// TODO: OpFConvert
// TODO: OpConvertPtrToU
// TODO: OpConvertUToPtr
// TODO: OpPtrCastToGeneric
// TODO: OpGenericCastToPtr
// TODO: OpBitcast
// TODO: OpGenericCastToPtrExplicit
// TODO: OpSatConvertSToU
// TODO: OpSatConvertUToS
// TODO: OpVectorExtractDynamic
// TODO: OpVectorInsertDynamic

TEST_F(ValidateIdWithMessage, OpVectorShuffleIntGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec3 = OpTypeVector %int 3
%ivec4 = OpTypeVector %int 4
%ptr_ivec3 = OpTypePointer Function %ivec3
%undef = OpUndef %ivec4
%int_42 = OpConstant %int 42
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%1 = OpConstantComposite %ivec3 %int_42 %int_0 %int_2
%2 = OpTypeFunction %ivec3
%3 = OpFunction %ivec3 None %2
%4 = OpLabel
%var = OpVariable %ptr_ivec3 Function %1
%5 = OpLoad %ivec3 %var
%6 = OpVectorShuffle %ivec3 %5 %undef 2 1 0
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleFloatGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec3 = OpTypeVector %float 3
%vec4 = OpTypeVector %float 4
%ptr_vec2 = OpTypePointer Function %vec2
%ptr_vec3 = OpTypePointer Function %vec3
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %vec2 %float_2 %float_1
%2 = OpConstantComposite %vec3 %float_1 %float_2 %float_2
%3 = OpTypeFunction %vec4
%4 = OpFunction %vec4 None %3
%5 = OpLabel
%var = OpVariable %ptr_vec2 Function %1
%var2 = OpVariable %ptr_vec3 Function %2
%6 = OpLoad %vec2 %var
%7 = OpLoad %vec3 %var2
%8 = OpVectorShuffle %vec4 %6 %7 4 3 1 0xffffffff
     OpReturnValue %8
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleScalarResultType) {
  std::string spirv = kGLSL450MemoryModel + R"(
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%ptr_vec2 = OpTypePointer Function %vec2
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %vec2 %float_2 %float_1
%2 = OpTypeFunction %float
%3 = OpFunction %float None %2
%4 = OpLabel
%var = OpVariable %ptr_vec2 Function %1
%5 = OpLoad %vec2 %var
%6 = OpVectorShuffle %float %5 %5 0
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Result Type of OpVectorShuffle must be OpTypeVector."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleComponentCount) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec3 = OpTypeVector %int 3
%ptr_ivec3 = OpTypePointer Function %ivec3
%int_42 = OpConstant %int 42
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%1 = OpConstantComposite %ivec3 %int_42 %int_0 %int_2
%2 = OpTypeFunction %ivec3
%3 = OpFunction %ivec3 None %2
%4 = OpLabel
%var = OpVariable %ptr_ivec3 Function %1
%5 = OpLoad %ivec3 %var
%6 = OpVectorShuffle %ivec3 %5 %5 0 1
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpVectorShuffle component literals count does not match "
                "Result Type <id> '2[%v3uint]'s vector component count."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleVector1Type) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec2 = OpTypeVector %int 2
%ptr_int = OpTypePointer Function %int
%undef = OpUndef %ivec2
%int_42 = OpConstant %int 42
%2 = OpTypeFunction %ivec2
%3 = OpFunction %ivec2 None %2
%4 = OpLabel
%var = OpVariable %ptr_int Function %int_42
%5 = OpLoad %int %var
%6 = OpVectorShuffle %ivec2 %5 %undef 0 0
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The type of Vector 1 must be OpTypeVector."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleVector2Type) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec2 = OpTypeVector %int 2
%ptr_ivec2 = OpTypePointer Function %ivec2
%undef = OpUndef %int
%int_42 = OpConstant %int 42
%1 = OpConstantComposite %ivec2 %int_42 %int_42
%2 = OpTypeFunction %ivec2
%3 = OpFunction %ivec2 None %2
%4 = OpLabel
%var = OpVariable %ptr_ivec2 Function %1
%5 = OpLoad %ivec2 %var
%6 = OpVectorShuffle %ivec2 %5 %undef 0 1
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The type of Vector 2 must be OpTypeVector."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleVector1ComponentType) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec3 = OpTypeVector %int 3
%ptr_ivec3 = OpTypePointer Function %ivec3
%int_42 = OpConstant %int 42
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%float = OpTypeFloat 32
%vec3 = OpTypeVector %float 3
%vec4 = OpTypeVector %float 4
%ptr_vec3 = OpTypePointer Function %vec3
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %ivec3 %int_42 %int_0 %int_2
%2 = OpConstantComposite %vec3 %float_1 %float_2 %float_2
%3 = OpTypeFunction %vec4
%4 = OpFunction %vec4 None %3
%5 = OpLabel
%var = OpVariable %ptr_ivec3 Function %1
%var2 = OpVariable %ptr_vec3 Function %2
%6 = OpLoad %ivec3 %var
%7 = OpLoad %vec3 %var2
%8 = OpVectorShuffle %vec4 %6 %7 4 3 1 0
     OpReturnValue %8
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Component Type of Vector 1 must be the same as "
                        "ResultType."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleVector2ComponentType) {
  std::string spirv = kGLSL450MemoryModel + R"(
%int = OpTypeInt 32 0
%ivec3 = OpTypeVector %int 3
%ptr_ivec3 = OpTypePointer Function %ivec3
%int_42 = OpConstant %int 42
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%float = OpTypeFloat 32
%vec3 = OpTypeVector %float 3
%vec4 = OpTypeVector %float 4
%ptr_vec3 = OpTypePointer Function %vec3
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %ivec3 %int_42 %int_0 %int_2
%2 = OpConstantComposite %vec3 %float_1 %float_2 %float_2
%3 = OpTypeFunction %vec4
%4 = OpFunction %vec4 None %3
%5 = OpLabel
%var = OpVariable %ptr_ivec3 Function %1
%var2 = OpVariable %ptr_vec3 Function %2
%6 = OpLoad %vec3 %var2
%7 = OpLoad %ivec3 %var
%8 = OpVectorShuffle %vec4 %6 %7 4 3 1 0
     OpReturnValue %8
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The Component Type of Vector 2 must be the same as "
                        "ResultType."));
}

TEST_F(ValidateIdWithMessage, OpVectorShuffleLiterals) {
  std::string spirv = kGLSL450MemoryModel + R"(
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec3 = OpTypeVector %float 3
%vec4 = OpTypeVector %float 4
%ptr_vec2 = OpTypePointer Function %vec2
%ptr_vec3 = OpTypePointer Function %vec3
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %vec2 %float_2 %float_1
%2 = OpConstantComposite %vec3 %float_1 %float_2 %float_2
%3 = OpTypeFunction %vec4
%4 = OpFunction %vec4 None %3
%5 = OpLabel
%var = OpVariable %ptr_vec2 Function %1
%var2 = OpVariable %ptr_vec3 Function %2
%6 = OpLoad %vec2 %var
%7 = OpLoad %vec3 %var2
%8 = OpVectorShuffle %vec4 %6 %7 0 8 2 6
     OpReturnValue %8
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Component index 8 is out of bounds for combined (Vector1 + Vector2) "
          "size of 5."));
}

TEST_F(ValidateIdWithMessage, WebGPUOpVectorShuffle0xFFFFFFFFLiteralBad) {
  std::string spirv = R"(
    OpCapability Shader
    OpCapability VulkanMemoryModelKHR
    OpExtension "SPV_KHR_vulkan_memory_model"
    OpMemoryModel Logical VulkanKHR
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec3 = OpTypeVector %float 3
%vec4 = OpTypeVector %float 4
%ptr_vec2 = OpTypePointer Function %vec2
%ptr_vec3 = OpTypePointer Function %vec3
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%1 = OpConstantComposite %vec2 %float_2 %float_1
%2 = OpConstantComposite %vec3 %float_1 %float_2 %float_2
%3 = OpTypeFunction %vec4
%4 = OpFunction %vec4 None %3
%5 = OpLabel
%var = OpVariable %ptr_vec2 Function %1
%var2 = OpVariable %ptr_vec3 Function %2
%6 = OpLoad %vec2 %var
%7 = OpLoad %vec3 %var2
%8 = OpVectorShuffle %vec4 %6 %7 4 3 1 0xffffffff
     OpReturnValue %8
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Component literal at operand 3 cannot be 0xFFFFFFFF in"
                        " WebGPU execution environment."));
}

// TODO: OpCompositeConstruct
// TODO: OpCompositeExtract
// TODO: OpCompositeInsert
// TODO: OpCopyObject
// TODO: OpTranspose
// TODO: OpSNegate
// TODO: OpFNegate
// TODO: OpNot
// TODO: OpIAdd
// TODO: OpFAdd
// TODO: OpISub
// TODO: OpFSub
// TODO: OpIMul
// TODO: OpFMul
// TODO: OpUDiv
// TODO: OpSDiv
// TODO: OpFDiv
// TODO: OpUMod
// TODO: OpSRem
// TODO: OpSMod
// TODO: OpFRem
// TODO: OpFMod
// TODO: OpVectorTimesScalar
// TODO: OpMatrixTimesScalar
// TODO: OpVectorTimesMatrix
// TODO: OpMatrixTimesVector
// TODO: OpMatrixTimesMatrix
// TODO: OpOuterProduct
// TODO: OpDot
// TODO: OpShiftRightLogical
// TODO: OpShiftRightArithmetic
// TODO: OpShiftLeftLogical
// TODO: OpBitwiseOr
// TODO: OpBitwiseXor
// TODO: OpBitwiseAnd
// TODO: OpAny
// TODO: OpAll
// TODO: OpIsNan
// TODO: OpIsInf
// TODO: OpIsFinite
// TODO: OpIsNormal
// TODO: OpSignBitSet
// TODO: OpLessOrGreater
// TODO: OpOrdered
// TODO: OpUnordered
// TODO: OpLogicalOr
// TODO: OpLogicalXor
// TODO: OpLogicalAnd
// TODO: OpSelect
// TODO: OpIEqual
// TODO: OpFOrdEqual
// TODO: OpFUnordEqual
// TODO: OpINotEqual
// TODO: OpFOrdNotEqual
// TODO: OpFUnordNotEqual
// TODO: OpULessThan
// TODO: OpSLessThan
// TODO: OpFOrdLessThan
// TODO: OpFUnordLessThan
// TODO: OpUGreaterThan
// TODO: OpSGreaterThan
// TODO: OpFOrdGreaterThan
// TODO: OpFUnordGreaterThan
// TODO: OpULessThanEqual
// TODO: OpSLessThanEqual
// TODO: OpFOrdLessThanEqual
// TODO: OpFUnordLessThanEqual
// TODO: OpUGreaterThanEqual
// TODO: OpSGreaterThanEqual
// TODO: OpFOrdGreaterThanEqual
// TODO: OpFUnordGreaterThanEqual
// TODO: OpDPdx
// TODO: OpDPdy
// TODO: OpFWidth
// TODO: OpDPdxFine
// TODO: OpDPdyFine
// TODO: OpFwidthFine
// TODO: OpDPdxCoarse
// TODO: OpDPdyCoarse
// TODO: OpFwidthCoarse
// TODO: OpLoopMerge
// TODO: OpSelectionMerge
// TODO: OpBranch

TEST_F(ValidateIdWithMessage, OpPhiNotAType) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %3 %3 %7
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ID 3[%true] is not a type "
                                               "id"));
}

TEST_F(ValidateIdWithMessage, OpPhiSamePredecessor) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranchConditional %3 %8 %8
%8 = OpLabel
%9 = OpPhi %2 %3 %7
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpPhiOddArgumentNumber) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %2 %3
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi does not have an equal number of incoming "
                        "values and basic blocks."));
}

TEST_F(ValidateIdWithMessage, OpPhiTooFewPredecessors) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranch %8
%8 = OpLabel
%9 = OpPhi %2
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi's number of incoming blocks (0) does not match "
                        "block's predecessor count (1)."));
}

TEST_F(ValidateIdWithMessage, OpPhiTooManyPredecessors) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranch %8
%9 = OpLabel
OpReturn
%8 = OpLabel
%10 = OpPhi %2 %3 %7 %3 %9
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi's number of incoming blocks (2) does not match "
                        "block's predecessor count (1)."));
}

TEST_F(ValidateIdWithMessage, OpPhiMismatchedTypes) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeInt 32 0
%6 = OpConstant %5 0
%7 = OpTypeFunction %4
%8 = OpFunction %4 None %7
%9 = OpLabel
OpBranchConditional %3 %10 %11
%11 = OpLabel
OpBranch %10
%10 = OpLabel
%12 = OpPhi %2 %3 %9 %6 %11
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi's result type <id> 2[%bool] does not match "
                        "incoming value <id> 6[%uint_0] type <id> "
                        "5[%uint]."));
}

TEST_F(ValidateIdWithMessage, OpPhiPredecessorNotABlock) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranchConditional %3 %8 %9
%9 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %8
%8 = OpLabel
%10 = OpPhi %2 %3 %7 %3 %3
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi's incoming basic block <id> 3[%true] is not an "
                        "OpLabel."));
}

TEST_F(ValidateIdWithMessage, OpPhiNotAPredecessor) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
%2 = OpTypeBool
%3 = OpConstantTrue %2
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
OpBranchConditional %3 %8 %9
%9 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %8
%8 = OpLabel
%10 = OpPhi %2 %3 %7 %3 %9
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpPhi's incoming basic block <id> 9[%9] is not a "
                        "predecessor of <id> 8[%8]."));
}

TEST_F(ValidateIdWithMessage, OpBranchConditionalGood) {
  std::string spirv = BranchConditionalSetup + R"(
    %branch_cond = OpINotEqual %bool %i0 %i1
                   OpSelectionMerge %end None
                   OpBranchConditional %branch_cond %target_t %target_f
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateIdWithMessage, OpBranchConditionalWithWeightsGood) {
  std::string spirv = BranchConditionalSetup + R"(
    %branch_cond = OpINotEqual %bool %i0 %i1
                   OpSelectionMerge %end None
                   OpBranchConditional %branch_cond %target_t %target_f 1 1
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_CondIsScalarInt) {
  std::string spirv = BranchConditionalSetup + R"(
                   OpSelectionMerge %end None
                   OpBranchConditional %i0 %target_t %target_f
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Condition operand for OpBranchConditional must be of boolean type"));
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_TrueTargetIsNotLabel) {
  std::string spirv = BranchConditionalSetup + R"(
                   OpSelectionMerge %end None
                   OpBranchConditional %true %i0 %target_f
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The 'True Label' operand for OpBranchConditional must "
                        "be the ID of an OpLabel instruction"));
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_FalseTargetIsNotLabel) {
  std::string spirv = BranchConditionalSetup + R"(
                   OpSelectionMerge %end None
                   OpBranchConditional %true %target_t %i0
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The 'False Label' operand for OpBranchConditional "
                        "must be the ID of an OpLabel instruction"));
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_NotEnoughWeights) {
  std::string spirv = BranchConditionalSetup + R"(
    %branch_cond = OpINotEqual %bool %i0 %i1
                   OpSelectionMerge %end None
                   OpBranchConditional %branch_cond %target_t %target_f 1
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpBranchConditional requires either 3 or 5 parameters"));
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_TooManyWeights) {
  std::string spirv = BranchConditionalSetup + R"(
    %branch_cond = OpINotEqual %bool %i0 %i1
                   OpSelectionMerge %end None
                   OpBranchConditional %branch_cond %target_t %target_f 1 2 3
  )" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpBranchConditional requires either 3 or 5 parameters"));
}

TEST_F(ValidateIdWithMessage, OpBranchConditional_ConditionIsAType) {
  std::string spirv = BranchConditionalSetup + R"(
OpBranchConditional %bool %target_t %target_f
)" + BranchConditionalTail;

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 3[%bool] cannot be a "
                                               "type"));
}

// TODO: OpSwitch

TEST_F(ValidateIdWithMessage, OpReturnValueConstantGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2
%4 = OpConstant %2 42
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %4
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpReturnValueVariableGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0 ;10
%3 = OpTypeFunction %2
%8 = OpTypePointer Function %2 ;18
%4 = OpConstant %2 42 ;22
%5 = OpFunction %2 None %3 ;27
%6 = OpLabel ;29
%7 = OpVariable %8 Function %4 ;34
%9 = OpLoad %2 %7
     OpReturnValue %9 ;36
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpReturnValueExpressionGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2
%4 = OpConstant %2 42
%5 = OpFunction %2 None %3
%6 = OpLabel
%7 = OpIAdd %2 %4 %4
     OpReturnValue %7
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpReturnValueIsType) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %1
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 1[%void] cannot be a "
                                               "type"));
}

TEST_F(ValidateIdWithMessage, OpReturnValueIsLabel) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %6
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 5[%5] requires a type"));
}

TEST_F(ValidateIdWithMessage, OpReturnValueIsVoid) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1
%5 = OpFunction %1 None %3
%6 = OpLabel
%7 = OpFunctionCall %1 %5
     OpReturnValue %7
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpReturnValue value's type <id> '1[%void]' is missing or "
                "void."));
}

TEST_F(ValidateIdWithMessage, OpReturnValueIsVariableInPhysical) {
  // It's valid to return a pointer in a physical addressing model.
  std::string spirv = kOpCapabilitySetup + R"(
     OpMemoryModel Physical32 OpenCL
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Function %2
%4 = OpTypeFunction %3
%5 = OpFunction %3 None %4
%6 = OpLabel
%7 = OpVariable %3 Function
     OpReturnValue %7
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpReturnValueIsVariableInLogical) {
  // It's invalid to return a pointer in a physical addressing model.
  std::string spirv = kOpCapabilitySetup + R"(
     OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Function %2
%4 = OpTypeFunction %3
%5 = OpFunction %3 None %4
%6 = OpLabel
%7 = OpVariable %3 Function
     OpReturnValue %7
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpReturnValue value's type <id> "
                        "'3[%_ptr_Function_uint]' is a pointer, which is "
                        "invalid in the Logical addressing model."));
}

// With the VariablePointer Capability, the return value of a function is
// allowed to be a pointer.
TEST_F(ValidateIdWithMessage, OpReturnValueVarPtrGood) {
  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv,
                                    "" /* Instructions to add to "main" */,
                                    true /* Add VariablePointers Capability?*/,
                                    true /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Without the VariablePointer Capability, the return value of a function is
// *not* allowed to be a pointer.
// Disabled since using OpSelect with pointers without VariablePointers will
// fail LogicalsPass.
TEST_F(ValidateIdWithMessage, DISABLED_OpReturnValueVarPtrBad) {
  std::ostringstream spirv;
  createVariablePointerSpirvProgram(&spirv,
                                    "" /* Instructions to add to "main" */,
                                    false /* Add VariablePointers Capability?*/,
                                    true /* Use Helper Function? */);
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpReturnValue value's type <id> '7' is a pointer, "
                        "which is invalid in the Logical addressing model."));
}

// TODO: enable when this bug is fixed:
// https://cvs.khronos.org/bugzilla/show_bug.cgi?id=15404
TEST_F(ValidateIdWithMessage, DISABLED_OpReturnValueIsFunction) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %5
     OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, UndefinedTypeId) {
  std::string spirv = kGLSL450MemoryModel + R"(
%s = OpTypeStruct %i32
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Forward reference operands in an OpTypeStruct must "
                        "first be declared using OpTypeForwardPointer."));
}

TEST_F(ValidateIdWithMessage, UndefinedIdScope) {
  std::string spirv = kGLSL450MemoryModel + R"(
%u32    = OpTypeInt 32 0
%memsem = OpConstant %u32 0
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%f      = OpFunction %void None %void_f
%l      = OpLabel
          OpMemoryBarrier %undef %memsem
          OpReturn
          OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ID 7[%7] has not been "
                                               "defined"));
}

TEST_F(ValidateIdWithMessage, UndefinedIdMemSem) {
  std::string spirv = kGLSL450MemoryModel + R"(
%u32    = OpTypeInt 32 0
%scope  = OpConstant %u32 0
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%f      = OpFunction %void None %void_f
%l      = OpLabel
          OpMemoryBarrier %scope %undef
          OpReturn
          OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ID 7[%7] has not been "
                                               "defined"));
}

TEST_F(ValidateIdWithMessage,
       KernelOpEntryPointAndOpInBoundsPtrAccessChainGood) {
  std::string spirv = kOpenCLMemoryModel32 + R"(
      OpEntryPoint Kernel %2 "simple_kernel"
      OpSource OpenCL_C 200000
      OpDecorate %3 BuiltIn GlobalInvocationId
      OpDecorate %3 Constant
      OpDecorate %4 FuncParamAttr NoCapture
      OpDecorate %3 LinkageAttributes "__spirv_GlobalInvocationId" Import
 %5 = OpTypeInt 32 0
 %6 = OpTypeVector %5 3
 %7 = OpTypePointer UniformConstant %6
 %3 = OpVariable %7 UniformConstant
 %8 = OpTypeVoid
 %9 = OpTypeStruct %5
%10 = OpTypePointer CrossWorkgroup %9
%11 = OpTypeFunction %8 %10
%12 = OpConstant %5 0
%13 = OpTypePointer CrossWorkgroup %5
%14 = OpConstant %5 42
 %2 = OpFunction %8 None %11
 %4 = OpFunctionParameter %10
%15 = OpLabel
%16 = OpLoad %6 %3 Aligned 0
%17 = OpCompositeExtract %5 %16 0
%18 = OpInBoundsPtrAccessChain %13 %4 %17 %12
      OpStore %18 %14 Aligned 4
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpPtrAccessChainGood) {
  std::string spirv = kOpenCLMemoryModel64 + R"(
      OpEntryPoint Kernel %2 "another_kernel"
      OpSource OpenCL_C 200000
      OpDecorate %3 BuiltIn GlobalInvocationId
      OpDecorate %3 Constant
      OpDecorate %4 FuncParamAttr NoCapture
      OpDecorate %3 LinkageAttributes "__spirv_GlobalInvocationId" Import
 %5 = OpTypeInt 64 0
 %6 = OpTypeVector %5 3
 %7 = OpTypePointer UniformConstant %6
 %3 = OpVariable %7 UniformConstant
 %8 = OpTypeVoid
 %9 = OpTypeInt 32 0
%10 = OpTypeStruct %9
%11 = OpTypePointer CrossWorkgroup %10
%12 = OpTypeFunction %8 %11
%13 = OpConstant %5 4294967295
%14 = OpConstant %9 0
%15 = OpTypePointer CrossWorkgroup %9
%16 = OpConstant %9 42
 %2 = OpFunction %8 None %12
 %4 = OpFunctionParameter %11
%17 = OpLabel
%18 = OpLoad %6 %3 Aligned 0
%19 = OpCompositeExtract %5 %18 0
%20 = OpBitwiseAnd %5 %19 %13
%21 = OpPtrAccessChain %15 %4 %20 %14
      OpStore %21 %16 Aligned 4
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, StgBufOpPtrAccessChainGood) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpCapability VariablePointersStorageBuffer
     OpExtension "SPV_KHR_variable_pointers"
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
%int = OpTypeInt 32 0
%int_2 = OpConstant %int 2
%int_4 = OpConstant %int 4
%struct = OpTypeStruct %int
%array = OpTypeArray %struct %int_4
%ptr = OpTypePointer StorageBuffer %array
%var = OpVariable %ptr StorageBuffer
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
%5 = OpPtrAccessChain %ptr %var %int_2
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIdWithMessage, OpLoadBitcastPointerGood) {
  std::string spirv = kOpenCLMemoryModel64 + R"(
%2  = OpTypeVoid
%3  = OpTypeInt 32 0
%4  = OpTypeFloat 32
%5  = OpTypePointer UniformConstant %3
%6  = OpTypePointer UniformConstant %4
%7  = OpVariable %5 UniformConstant
%8  = OpTypeFunction %2
%9  = OpFunction %2 None %8
%10 = OpLabel
%11 = OpBitcast %6 %7
%12 = OpLoad %4 %11
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpLoadBitcastNonPointerBad) {
  std::string spirv = kOpenCLMemoryModel64 + R"(
%2  = OpTypeVoid
%3  = OpTypeInt 32 0
%4  = OpTypeFloat 32
%5  = OpTypePointer UniformConstant %3
%6  = OpTypeFunction %2
%7  = OpVariable %5 UniformConstant
%8  = OpFunction %2 None %6
%9  = OpLabel
%10 = OpLoad %3 %7
%11 = OpBitcast %4 %10
%12 = OpLoad %3 %11
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpLoad type for pointer <id> '11[%11]' is not a pointer "
                "type."));
}
TEST_F(ValidateIdWithMessage, OpStoreBitcastPointerGood) {
  std::string spirv = kOpenCLMemoryModel64 + R"(
%2  = OpTypeVoid
%3  = OpTypeInt 32 0
%4  = OpTypeFloat 32
%5  = OpTypePointer Function %3
%6  = OpTypePointer Function %4
%7  = OpTypeFunction %2
%8  = OpConstant %3 42
%9  = OpFunction %2 None %7
%10 = OpLabel
%11 = OpVariable %6 Function
%12 = OpBitcast %5 %11
      OpStore %12 %8
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}
TEST_F(ValidateIdWithMessage, OpStoreBitcastNonPointerBad) {
  std::string spirv = kOpenCLMemoryModel64 + R"(
%2  = OpTypeVoid
%3  = OpTypeInt 32 0
%4  = OpTypeFloat 32
%5  = OpTypePointer Function %4
%6  = OpTypeFunction %2
%7  = OpConstant %4 42
%8  = OpFunction %2 None %6
%9  = OpLabel
%10 = OpVariable %5 Function
%11 = OpBitcast %3 %7
      OpStore %11 %7
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpStore type for pointer <id> '11[%11]' is not a pointer "
                "type."));
}

// Result <id> resulting from an instruction within a function may not be used
// outside that function.
TEST_F(ValidateIdWithMessage, ResultIdUsedOutsideOfFunctionBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpTypePointer Function %3
%5 = OpFunction %1 None %2
%6 = OpLabel
%7 = OpVariable %4 Function
OpReturn
OpFunctionEnd
%8 = OpFunction %1 None %2
%9 = OpLabel
%10 = OpLoad %3 %7
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "ID 7[%7] defined in block 6[%6] does not dominate its use in block "
          "9[%9]"));
}

TEST_F(ValidateIdWithMessage, SpecIdTargetNotSpecializationConstant) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpDecorate %1 SpecId 200
%void = OpTypeVoid
%2 = OpTypeFunction %void
%int = OpTypeInt 32 0
%1 = OpConstant %int 3
%main = OpFunction %void None %2
%4 = OpLabel
OpReturnValue %1
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpDecorate SpecId decoration target <id> "
                        "'1[%uint_3]' is not a scalar specialization "
                        "constant."));
}

TEST_F(ValidateIdWithMessage, SpecIdTargetOpSpecConstantOpBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpDecorate %1 SpecId 200
%void = OpTypeVoid
%2 = OpTypeFunction %void
%int = OpTypeInt 32 0
%3 = OpConstant %int 1
%4 = OpConstant %int 2
%1 = OpSpecConstantOp %int IAdd %3 %4
%main = OpFunction %void None %2
%6 = OpLabel
OpReturnValue %3
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpDecorate SpecId decoration target <id> '1[%1]' is "
                        "not a scalar specialization constant."));
}

TEST_F(ValidateIdWithMessage, SpecIdTargetOpSpecConstantCompositeBad) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpDecorate %1 SpecId 200
%void = OpTypeVoid
%2 = OpTypeFunction %void
%int = OpTypeInt 32 0
%3 = OpConstant %int 1
%1 = OpSpecConstantComposite %int
%main = OpFunction %void None %2
%4 = OpLabel
OpReturnValue %3
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpDecorate SpecId decoration target <id> '1[%1]' is "
                        "not a scalar specialization constant."));
}

TEST_F(ValidateIdWithMessage, SpecIdTargetGood) {
  std::string spirv = kGLSL450MemoryModel + R"(
OpDecorate %3 SpecId 200
OpDecorate %4 SpecId 201
OpDecorate %5 SpecId 202
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%int = OpTypeInt 32 0
%bool = OpTypeBool
%3 = OpSpecConstant %int 3
%4 = OpSpecConstantTrue %bool
%5 = OpSpecConstantFalse %bool
%main = OpFunction %1 None %2
%6 = OpLabel
OpReturn
OpFunctionEnd
  )";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateIdWithMessage, CorrectErrorForShuffle) {
  std::string spirv = kGLSL450MemoryModel + R"(
   %uint = OpTypeInt 32 0
  %float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%v2float = OpTypeVector %float 2
   %void = OpTypeVoid
    %548 = OpTypeFunction %void
     %CS = OpFunction %void None %548
    %550 = OpLabel
   %6275 = OpUndef %v2float
   %6280 = OpUndef %v2float
   %6282 = OpVectorShuffle %v4float %6275 %6280 0 1 4 5
           OpReturn
           OpFunctionEnd
  )";

  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Component index 4 is out of bounds for combined (Vector1 + Vector2) "
          "size of 4."));
  EXPECT_EQ(25, getErrorPosition().index);
}

TEST_F(ValidateIdWithMessage, VoidStructMember) {
  const std::string spirv = kGLSL450MemoryModel + R"(
%void = OpTypeVoid
%struct = OpTypeStruct %void
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Structures cannot contain a void type."));
}

TEST_F(ValidateIdWithMessage, TypeFunctionBadUse) {
  std::string spirv = kGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypePointer Function %2
%4 = OpFunction %1 None %2
%5 = OpLabel
     OpReturn
     OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid use of function type result id 2[%2]."));
}

TEST_F(ValidateIdWithMessage, BadTypeId) {
  std::string spirv = kGLSL450MemoryModel + R"(
          %1 = OpTypeVoid
          %2 = OpTypeFunction %1
          %3 = OpTypeFloat 32
          %4 = OpConstant %3 0
          %5 = OpFunction %1 None %2
          %6 = OpLabel
          %7 = OpUndef %4
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ID 4[%float_0] is not a type "
                                               "id"));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelLoadMakePointerVisibleGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypeFunction %1
%6 = OpConstant %2 2
%7 = OpFunction %1 None %5
%8 = OpLabel
%9 = OpLoad %2 %4 NonPrivatePointerKHR|MakePointerVisibleKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelLoadMakePointerVisibleMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypeFunction %1
%6 = OpConstant %2 2
%7 = OpFunction %1 None %5
%8 = OpLabel
%9 = OpLoad %2 %4 MakePointerVisibleKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerVisibleKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelLoadNonPrivatePointerBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Private %2
%4 = OpVariable %3 Private
%5 = OpTypeFunction %1
%6 = OpConstant %2 2
%7 = OpFunction %1 None %5
%8 = OpLabel
%9 = OpLoad %2 %4 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelLoadMakePointerAvailableCannotBeUsed) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypeFunction %1
%6 = OpConstant %2 2
%7 = OpFunction %1 None %5
%8 = OpLabel
%9 = OpLoad %2 %4 NonPrivatePointerKHR|MakePointerAvailableKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("MakePointerAvailableKHR cannot be used with OpLoad"));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelStoreMakePointerAvailableGood) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpVariable %3 Uniform
%5 = OpTypeFunction %1
%6 = OpConstant %2 5
%7 = OpFunction %1 None %5
%8 = OpLabel
OpStore %4 %6 NonPrivatePointerKHR|MakePointerAvailableKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelStoreMakePointerAvailableMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpVariable %3 Uniform
%5 = OpTypeFunction %1
%6 = OpConstant %2 5
%7 = OpFunction %1 None %5
%8 = OpLabel
OpStore %4 %6 MakePointerAvailableKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerAvailableKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelStoreNonPrivatePointerBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Output %2
%4 = OpVariable %3 Output
%5 = OpTypeFunction %1
%6 = OpConstant %2 5
%7 = OpFunction %1 None %5
%8 = OpLabel
OpStore %4 %6 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelStoreMakePointerVisibleCannotBeUsed) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpCapability Linkage
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Uniform %2
%4 = OpVariable %3 Uniform
%5 = OpTypeFunction %1
%6 = OpConstant %2 5
%7 = OpFunction %1 None %5
%8 = OpLabel
OpStore %4 %6 NonPrivatePointerKHR|MakePointerVisibleKHR %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("MakePointerVisibleKHR cannot be used with OpStore."));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelCopyMemoryAvailable) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 NonPrivatePointerKHR|MakePointerAvailableKHR %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelCopyMemoryVisible) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 NonPrivatePointerKHR|MakePointerVisibleKHR %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelCopyMemoryAvailableAndVisible) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 NonPrivatePointerKHR|MakePointerAvailableKHR|MakePointerVisibleKHR %7 %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemoryAvailableMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 MakePointerAvailableKHR %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerAvailableKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemoryVisibleMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 MakePointerVisibleKHR %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerVisibleKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemoryAvailableBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Output %2
%4 = OpVariable %3 Output
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemoryVisibleBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Input %2
%6 = OpVariable %5 Input
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemory %4 %6 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelCopyMemorySizedAvailable) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 NonPrivatePointerKHR|MakePointerAvailableKHR %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, VulkanMemoryModelCopyMemorySizedVisible) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 NonPrivatePointerKHR|MakePointerVisibleKHR %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemorySizedAvailableAndVisible) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 NonPrivatePointerKHR|MakePointerAvailableKHR|MakePointerVisibleKHR %7 %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemorySizedAvailableMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 MakePointerAvailableKHR %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerAvailableKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemorySizedVisibleMissingNonPrivatePointer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 MakePointerVisibleKHR %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR must be specified if "
                        "MakePointerVisibleKHR is specified."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemorySizedAvailableBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Output %2
%4 = OpVariable %3 Output
%5 = OpTypePointer Uniform %2
%6 = OpVariable %5 Uniform
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage,
       VulkanMemoryModelCopyMemorySizedVisibleBadStorageClass) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer Workgroup %2
%4 = OpVariable %3 Workgroup
%5 = OpTypePointer Input %2
%6 = OpVariable %5 Input
%7 = OpConstant %2 2
%8 = OpConstant %2 5
%9 = OpTypeFunction %1
%10 = OpFunction %1 None %9
%11 = OpLabel
OpCopyMemorySized %4 %6 %7 NonPrivatePointerKHR
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("NonPrivatePointerKHR requires a pointer in Uniform, "
                        "Workgroup, CrossWorkgroup, Generic, Image or "
                        "StorageBuffer storage classes."));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock1) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpReturn
%7 = OpLabel
%8 = OpFunctionCall %3 %9
OpUnreachable
OpFunctionEnd
%9 = OpFunction %3 None %4
%10 = OpLabel
OpReturnValue %8
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 8[%8] defined in block 7[%7] does not dominate its "
                        "use in block 10[%10]\n  %10 = OpLabel"));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock2) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpReturn
%7 = OpLabel
%8 = OpFunctionCall %3 %9
OpUnreachable
OpFunctionEnd
%9 = OpFunction %3 None %4
%10 = OpLabel
OpReturnValue %8
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 8[%8] defined in block 7[%7] does not dominate its "
                        "use in block 10[%10]\n  %10 = OpLabel"));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock3) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpReturn
%7 = OpLabel
%8 = OpFunctionCall %3 %9
OpReturn
OpFunctionEnd
%9 = OpFunction %3 None %4
%10 = OpLabel
OpReturnValue %8
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 8[%8] defined in block 7[%7] does not dominate its "
                        "use in block 10[%10]\n  %10 = OpLabel"));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock4) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpReturn
%7 = OpLabel
%8 = OpUndef %3
%9 = OpCopyObject %3 %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock5) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpReturn
%7 = OpLabel
%8 = OpUndef %3
OpBranch %9
%9 = OpLabel
%10 = OpCopyObject %3 %8
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, IdDefInUnreachableBlock6) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
OpBranch %7
%8 = OpLabel
%9 = OpUndef %3
OpBranch %7
%7 = OpLabel
%10 = OpCopyObject %3 %9
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("ID 9[%9] defined in block 8[%8] does not dominate its "
                        "use in block 7[%7]\n  %7 = OpLabel"));
}

TEST_F(ValidateIdWithMessage, ReachableDefUnreachableUse) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeFunction %3
%5 = OpFunction %1 None %2
%6 = OpLabel
%7 = OpUndef %3
OpReturn
%8 = OpLabel
%9 = OpCopyObject %3 %7
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateIdWithMessage, UnreachableDefUsedInPhi) {
  const std::string spirv = kNoKernelGLSL450MemoryModel + R"(
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %bool = OpTypeBool
          %6 = OpTypeFunction %float
          %1 = OpFunction %void None %3
          %7 = OpLabel
          %8 = OpUndef %bool
               OpSelectionMerge %9 None
               OpBranchConditional %8 %10 %9
         %10 = OpLabel
         %11 = OpUndef %float
               OpBranch %9
         %12 = OpLabel
         %13 = OpUndef %float
               OpUnreachable
          %9 = OpLabel
         %14 = OpPhi %float %11 %10 %13 %7
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("In OpPhi instruction 14[%14], ID 13[%13] definition does not "
                "dominate its parent 7[%7]\n  %14 = OpPhi %float %11 %10 %13 "
                "%7"));
}

TEST_F(ValidateIdWithMessage, OpTypeForwardPointerNotAPointerType) {
  std::string spirv = R"(
     OpCapability GenericPointer
     OpCapability VariablePointersStorageBuffer
     OpMemoryModel Logical GLSL450
     OpEntryPoint Fragment %1 "main"
     OpExecutionMode %1 OriginLowerLeft
     OpTypeForwardPointer %2 CrossWorkgroup
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 DontInline %3
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Pointer type in OpTypeForwardPointer is not a pointer "
                        "type.\n  OpTypeForwardPointer %void CrossWorkgroup"));
}

TEST_F(ValidateIdWithMessage, OpTypeForwardPointerWrongStorageClass) {
  std::string spirv = R"(
     OpCapability GenericPointer
     OpCapability VariablePointersStorageBuffer
     OpMemoryModel Logical GLSL450
     OpEntryPoint Fragment %1 "main"
     OpExecutionMode %1 OriginLowerLeft
     OpTypeForwardPointer %2 CrossWorkgroup
%int = OpTypeInt 32 1
%2 = OpTypePointer Function %int
%void = OpTypeVoid
%3 = OpTypeFunction %void
%1 = OpFunction %void None %3
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Storage class in OpTypeForwardPointer does not match the "
                "pointer definition.\n  OpTypeForwardPointer "
                "%_ptr_Function_int CrossWorkgroup"));
}
}  // namespace
}  // namespace val
}  // namespace spvtools
