// Copyright (c) 2019 Google LLC.
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
#include <tuple>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Combine;
using ::testing::HasSubstr;
using ::testing::Values;

using ValidateFunctionCall = spvtest::ValidateBase<std::string>;

std::string GenerateShader(const std::string& storage_class,
                           const std::string& capabilities,
                           const std::string& extensions) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability AtomicStorage
)" + capabilities + R"(
OpExtension "SPV_KHR_storage_buffer_storage_class"
)" +
                      extensions + R"(
OpMemoryModel Logical GLSL450
OpName %var "var"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr = OpTypePointer )" + storage_class + R"( %int
%caller_ty = OpTypeFunction %void
%callee_ty = OpTypeFunction %void %ptr
)";

  if (storage_class != "Function") {
    spirv += "%var = OpVariable %ptr " + storage_class;
  }

  spirv += R"(
%caller = OpFunction %void None %caller_ty
%1 = OpLabel
)";

  if (storage_class == "Function") {
    spirv += "%var = OpVariable %ptr Function";
  }

  spirv += R"(
%call = OpFunctionCall %void %callee %var
OpReturn
OpFunctionEnd
%callee = OpFunction %void None %callee_ty
%param = OpFunctionParameter %ptr
%2 = OpLabel
OpReturn
OpFunctionEnd
)";

  return spirv;
}

std::string GenerateShaderParameter(const std::string& storage_class,
                                    const std::string& capabilities,
                                    const std::string& extensions) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability AtomicStorage
)" + capabilities + R"(
OpExtension "SPV_KHR_storage_buffer_storage_class"
)" +
                      extensions + R"(
OpMemoryModel Logical GLSL450
OpName %p "p"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr = OpTypePointer )" + storage_class + R"( %int
%func_ty = OpTypeFunction %void %ptr
%caller = OpFunction %void None %func_ty
%p = OpFunctionParameter %ptr
%1 = OpLabel
%call = OpFunctionCall %void %callee %p
OpReturn
OpFunctionEnd
%callee = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr
%2 = OpLabel
OpReturn
OpFunctionEnd
)";

  return spirv;
}

std::string GenerateShaderAccessChain(const std::string& storage_class,
                                      const std::string& capabilities,
                                      const std::string& extensions) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability AtomicStorage
)" + capabilities + R"(
OpExtension "SPV_KHR_storage_buffer_storage_class"
)" +
                      extensions + R"(
OpMemoryModel Logical GLSL450
OpName %var "var"
OpName %gep "gep"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int2 = OpTypeVector %int 2
%int_0 = OpConstant %int 0
%ptr = OpTypePointer )" + storage_class + R"( %int2
%ptr2 = OpTypePointer )" +
                      storage_class + R"( %int
%caller_ty = OpTypeFunction %void
%callee_ty = OpTypeFunction %void %ptr2
)";

  if (storage_class != "Function") {
    spirv += "%var = OpVariable %ptr " + storage_class;
  }

  spirv += R"(
%caller = OpFunction %void None %caller_ty
%1 = OpLabel
)";

  if (storage_class == "Function") {
    spirv += "%var = OpVariable %ptr Function";
  }

  spirv += R"(
%gep = OpAccessChain %ptr2 %var %int_0
%call = OpFunctionCall %void %callee %gep
OpReturn
OpFunctionEnd
%callee = OpFunction %void None %callee_ty
%param = OpFunctionParameter %ptr2
%2 = OpLabel
OpReturn
OpFunctionEnd
)";

  return spirv;
}

TEST_P(ValidateFunctionCall, VariableNoVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShader(storage_class, "", "");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function", "Private", "Workgroup", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    if (storage_class == "StorageBuffer") {
      EXPECT_THAT(getDiagnosticString(),
                  HasSubstr("StorageBuffer pointer operand 1[%var] requires a "
                            "variable pointers capability"));
    } else {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr("Invalid storage class for pointer operand 1[%var]"));
    }
  }
}

TEST_P(ValidateFunctionCall, VariableVariablePointersStorageClass) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShader(
      storage_class, "OpCapability VariablePointersStorageBuffer",
      "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function",      "Private",
      "Workgroup",       "StorageBuffer", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Invalid storage class for pointer operand 1[%var]"));
  }
}

TEST_P(ValidateFunctionCall, VariableVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv =
      GenerateShader(storage_class, "OpCapability VariablePointers",
                     "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function",      "Private",
      "Workgroup",       "StorageBuffer", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Invalid storage class for pointer operand 1[%var]"));
  }
}

TEST_P(ValidateFunctionCall, ParameterNoVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShaderParameter(storage_class, "", "");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function", "Private", "Workgroup", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    if (storage_class == "StorageBuffer") {
      EXPECT_THAT(getDiagnosticString(),
                  HasSubstr("StorageBuffer pointer operand 1[%p] requires a "
                            "variable pointers capability"));
    } else {
      EXPECT_THAT(getDiagnosticString(),
                  HasSubstr("Invalid storage class for pointer operand 1[%p]"));
    }
  }
}

TEST_P(ValidateFunctionCall, ParameterVariablePointersStorageBuffer) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShaderParameter(
      storage_class, "OpCapability VariablePointersStorageBuffer",
      "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function",      "Private",
      "Workgroup",       "StorageBuffer", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Invalid storage class for pointer operand 1[%p]"));
  }
}

TEST_P(ValidateFunctionCall, ParameterVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv =
      GenerateShaderParameter(storage_class, "OpCapability VariablePointers",
                              "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "UniformConstant", "Function",      "Private",
      "Workgroup",       "StorageBuffer", "AtomicCounter"};
  bool valid =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  if (valid) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Invalid storage class for pointer operand 1[%p]"));
  }
}

TEST_P(ValidateFunctionCall, NonMemoryObjectDeclarationNoVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShaderAccessChain(storage_class, "", "");

  const std::vector<std::string> valid_storage_classes = {
      "Function", "Private", "Workgroup", "AtomicCounter"};
  bool valid_sc =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();

  CompileSuccessfully(spirv);
  spv_result_t expected_result =
      storage_class == "UniformConstant" ? SPV_SUCCESS : SPV_ERROR_INVALID_ID;
  EXPECT_EQ(expected_result, ValidateInstructions());
  if (valid_sc) {
    EXPECT_THAT(
        getDiagnosticString(),
        HasSubstr(
            "Pointer operand 2[%gep] must be a memory object declaration"));
  } else {
    if (storage_class == "StorageBuffer") {
      EXPECT_THAT(getDiagnosticString(),
                  HasSubstr("StorageBuffer pointer operand 2[%gep] requires a "
                            "variable pointers capability"));
    } else if (storage_class != "UniformConstant") {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr("Invalid storage class for pointer operand 2[%gep]"));
    }
  }
}

TEST_P(ValidateFunctionCall,
       NonMemoryObjectDeclarationVariablePointersStorageBuffer) {
  const std::string storage_class = GetParam();

  std::string spirv = GenerateShaderAccessChain(
      storage_class, "OpCapability VariablePointersStorageBuffer",
      "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "Function", "Private", "Workgroup", "StorageBuffer", "AtomicCounter"};
  bool valid_sc =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();
  bool validate =
      storage_class == "StorageBuffer" || storage_class == "UniformConstant";

  CompileSuccessfully(spirv);
  if (validate) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    if (valid_sc) {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr(
              "Pointer operand 2[%gep] must be a memory object declaration"));
    } else {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr("Invalid storage class for pointer operand 2[%gep]"));
    }
  }
}

TEST_P(ValidateFunctionCall, NonMemoryObjectDeclarationVariablePointers) {
  const std::string storage_class = GetParam();

  std::string spirv =
      GenerateShaderAccessChain(storage_class, "OpCapability VariablePointers",
                                "OpExtension \"SPV_KHR_variable_pointers\"");

  const std::vector<std::string> valid_storage_classes = {
      "Function", "Private", "Workgroup", "StorageBuffer", "AtomicCounter"};
  bool valid_sc =
      std::find(valid_storage_classes.begin(), valid_storage_classes.end(),
                storage_class) != valid_storage_classes.end();
  bool validate = storage_class == "StorageBuffer" ||
                  storage_class == "Workgroup" ||
                  storage_class == "UniformConstant";

  CompileSuccessfully(spirv);
  if (validate) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
    if (valid_sc) {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr(
              "Pointer operand 2[%gep] must be a memory object declaration"));
    } else {
      EXPECT_THAT(
          getDiagnosticString(),
          HasSubstr("Invalid storage class for pointer operand 2[%gep]"));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(StorageClass, ValidateFunctionCall,
                         Values("UniformConstant", "Input", "Uniform", "Output",
                                "Workgroup", "Private", "Function",
                                "PushConstant", "Image", "StorageBuffer",
                                "AtomicCounter"));
}  // namespace
}  // namespace val
}  // namespace spvtools
