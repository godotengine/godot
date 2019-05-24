// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <unordered_map>

#include "SPIRV/spirv.hpp"
#include "spirv-tools/libspirv.hpp"

#include "common_shaders_for_test.h"
#include "shaderc/shaderc.hpp"

namespace {

using shaderc::AssemblyCompilationResult;
using shaderc::CompileOptions;
using shaderc::PreprocessedSourceCompilationResult;
using shaderc::SpvCompilationResult;
using testing::Each;
using testing::Eq;
using testing::HasSubstr;
using testing::Not;

// Helper function to check if the compilation result indicates a successful
// compilation.
template <typename T>
bool CompilationResultIsSuccess(const shaderc::CompilationResult<T>& result) {
  return result.GetCompilationStatus() == shaderc_compilation_status_success;
}

// Examines whether a compilation result has valid SPIR-V code, by checking the
// magic number in the fixed postion of the byte array in the result object.
// Returns true if the magic number is found at the correct postion, otherwise
// returns false.
bool IsValidSpv(const SpvCompilationResult& result) {
  if (!CompilationResultIsSuccess(result)) return false;
  size_t length_in_words = result.cend() - result.cbegin();
  if (length_in_words < 5) return false;
  const uint32_t* bytes = result.cbegin();
  return bytes[0] == spv::MagicNumber;
}

// Compiles a shader and returns true if the result is valid SPIR-V. The
// input_file_name is set to "shader".
bool CompilesToValidSpv(const shaderc::Compiler& compiler,
                        const std::string& shader, shaderc_shader_kind kind) {
  return IsValidSpv(compiler.CompileGlslToSpv(shader, kind, "shader"));
}

// Compiles a shader with options and returns true if the result is valid
// SPIR-V. The input_file_name is set to "shader".
bool CompilesToValidSpv(const shaderc::Compiler& compiler,
                        const std::string& shader, shaderc_shader_kind kind,
                        const CompileOptions& options) {
  return IsValidSpv(compiler.CompileGlslToSpv(shader, kind, "shader", options));
}

// Returns the compiler's output from a compilation result as a string.
template <typename T>
std::string CompilerOutputAsString(
    const shaderc::CompilationResult<T>& result) {
  return std::string(reinterpret_cast<const char*>(result.cbegin()),
                     reinterpret_cast<const char*>(result.cend()));
}

class CppInterface : public testing::Test {
 protected:
  // Compiles a shader and returns true on success, false on failure.
  // The input file name is set to "shader" by default.
  bool CompilationSuccess(const std::string& shader,
                          shaderc_shader_kind kind) const {
    return compiler_
               .CompileGlslToSpv(shader.c_str(), shader.length(), kind,
                                 "shader")
               .GetCompilationStatus() == shaderc_compilation_status_success;
  }

  // Compiles a shader with options and returns true on success, false on
  // failure.
  // The input file name is set to "shader" by default.
  bool CompilationSuccess(const std::string& shader, shaderc_shader_kind kind,
                          const CompileOptions& options) const {
    return compiler_
               .CompileGlslToSpv(shader.c_str(), shader.length(), kind,
                                 "shader", options)
               .GetCompilationStatus() == shaderc_compilation_status_success;
  }

  // Compiles a shader, asserts compilation success, and returns the warning
  // messages.
  // The input file name is set to "shader" by default.
  std::string CompilationWarnings(
      const std::string& shader, shaderc_shader_kind kind,
      // This could default to options_, but that can
      // be easily confused with a no-options-provided
      // case:
      const CompileOptions& options) {
    const auto compilation_result =
        compiler_.CompileGlslToSpv(shader, kind, "shader", options);
    EXPECT_TRUE(CompilationResultIsSuccess(compilation_result)) << kind << '\n'
                                                                << shader;
    return compilation_result.GetErrorMessage();
  }

  // Compiles a shader, asserts compilation fail, and returns the error
  // messages.
  std::string CompilationErrors(const std::string& shader,
                                shaderc_shader_kind kind,
                                // This could default to options_, but that can
                                // be easily confused with a no-options-provided
                                // case:
                                const CompileOptions& options) {
    const auto compilation_result =
        compiler_.CompileGlslToSpv(shader, kind, "shader", options);
    EXPECT_FALSE(CompilationResultIsSuccess(compilation_result)) << kind << '\n'
                                                                 << shader;
    return compilation_result.GetErrorMessage();
  }

  // Assembles the given SPIR-V assembly and returns true on success.
  bool AssemblingSuccess(const std::string& shader,
                         const CompileOptions& options) const {
    return compiler_.AssembleToSpv(shader, options).GetCompilationStatus() ==
           shaderc_compilation_status_success;
  }

  // Assembles the given SPIR-V assembly and returns true if the result contains
  // a valid SPIR-V module.
  bool AssemblingValid(const std::string& shader,
                       const CompileOptions& options) const {
    return IsValidSpv(compiler_.AssembleToSpv(shader, options));
  }

  // Compiles a shader, expects compilation success, and returns the output
  // bytes.
  // The input file name is set to "shader" by default.
  std::string CompilationOutput(const std::string& shader,
                                shaderc_shader_kind kind,
                                const CompileOptions& options) const {
    const auto compilation_result =
        compiler_.CompileGlslToSpv(shader, kind, "shader", options);
    EXPECT_TRUE(CompilationResultIsSuccess(compilation_result)) << kind << '\n';
    // Need to make sure you get complete binary data, including embedded nulls.
    return CompilerOutputAsString(compilation_result);
  }

  // Compiles a shader to SPIR-V assembly, expects compilation success, and
  // returns the output bytes.
  // The input file name is set to "shader" by default.
  std::string AssemblyOutput(const std::string& shader,
                             shaderc_shader_kind kind,
                             const CompileOptions& options) const {
    const auto compilation_result =
        compiler_.CompileGlslToSpvAssembly(shader, kind, "shader", options);
    EXPECT_TRUE(CompilationResultIsSuccess(compilation_result)) << kind << '\n';
    // Need to make sure you get complete binary data, including embedded nulls.
    return CompilerOutputAsString(compilation_result);
  }

  // For compiling shaders in subclass tests:
  shaderc::Compiler compiler_;
  CompileOptions options_;
};

TEST_F(CppInterface, CompilerValidUponConstruction) {
  EXPECT_TRUE(compiler_.IsValid());
}

TEST_F(CppInterface, MultipleCalls) {
  shaderc::Compiler compiler1, compiler2, compiler3;
  EXPECT_TRUE(compiler1.IsValid());
  EXPECT_TRUE(compiler2.IsValid());
  EXPECT_TRUE(compiler3.IsValid());
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST_F(CppInterface, MultipleThreadsInitializing) {
  std::unique_ptr<shaderc::Compiler> compiler1;
  std::unique_ptr<shaderc::Compiler> compiler2;
  std::unique_ptr<shaderc::Compiler> compiler3;
  std::thread t1([&compiler1]() {
    compiler1 = std::unique_ptr<shaderc::Compiler>(new shaderc::Compiler());
  });
  std::thread t2([&compiler2]() {
    compiler2 = std::unique_ptr<shaderc::Compiler>(new shaderc::Compiler());
  });
  std::thread t3([&compiler3]() {
    compiler3 = std::unique_ptr<shaderc::Compiler>(new shaderc::Compiler());
  });
  t1.join();
  t2.join();
  t3.join();
  EXPECT_TRUE(compiler1->IsValid());
  EXPECT_TRUE(compiler2->IsValid());
  EXPECT_TRUE(compiler3->IsValid());
}
#endif

TEST_F(CppInterface, CompilerMoves) {
  shaderc::Compiler compiler2(std::move(compiler_));
  ASSERT_FALSE(compiler_.IsValid());
  ASSERT_TRUE(compiler2.IsValid());
}

TEST_F(CppInterface, EmptyString) {
  EXPECT_FALSE(CompilationSuccess("", shaderc_glsl_vertex_shader));
  EXPECT_FALSE(CompilationSuccess("", shaderc_glsl_fragment_shader));
}

TEST_F(CppInterface, AssembleEmptyString) {
  EXPECT_TRUE(AssemblingSuccess("", options_));
}

TEST_F(CppInterface, ResultObjectMoves) {
  SpvCompilationResult result = compiler_.CompileGlslToSpv(
      kMinimalShader, shaderc_glsl_vertex_shader, "shader");
  EXPECT_TRUE(CompilationResultIsSuccess(result));
  const SpvCompilationResult result2(std::move(result));
  EXPECT_FALSE(CompilationResultIsSuccess(result));
  EXPECT_TRUE(CompilationResultIsSuccess(result2));
}

TEST_F(CppInterface, GarbageString) {
  EXPECT_FALSE(CompilationSuccess("jfalkds", shaderc_glsl_vertex_shader));
  EXPECT_FALSE(CompilationSuccess("jfalkds", shaderc_glsl_fragment_shader));
}

TEST_F(CppInterface, AssembleGarbageString) {
  const auto result = compiler_.AssembleToSpv("jfalkds", options_);
  EXPECT_FALSE(CompilationResultIsSuccess(result));
  EXPECT_EQ(0u, result.GetNumWarnings());
  EXPECT_EQ(1u, result.GetNumErrors());
}

// TODO(antiagainst): right now there is no assembling difference for all the
// target environments exposed by shaderc. So the following is just testing the
// target environment is accepted.
TEST_F(CppInterface, AssembleTargetEnv) {
  options_.SetTargetEnvironment(shaderc_target_env_opengl, 0);
  EXPECT_TRUE(AssemblingValid("OpCapability Shader", options_));
}

TEST_F(CppInterface, MinimalShader) {
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_fragment_shader));
}

TEST_F(CppInterface, AssembleMinimalShader) {
  EXPECT_TRUE(AssemblingValid(kMinimalShaderAssembly, options_));
}

TEST_F(CppInterface, BasicOptions) {
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader, options_));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_fragment_shader, options_));
}

TEST_F(CppInterface, CopiedOptions) {
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader, options_));
  CompileOptions copied_options(options_);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_fragment_shader, copied_options));
}

TEST_F(CppInterface, MovedOptions) {
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader, options_));
  CompileOptions copied_options(std::move(options_));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_fragment_shader, copied_options));
}

TEST_F(CppInterface, StdAndCString) {
  const SpvCompilationResult result1 =
      compiler_.CompileGlslToSpv(kMinimalShader, strlen(kMinimalShader),
                                 shaderc_glsl_fragment_shader, "shader");
  const SpvCompilationResult result2 = compiler_.CompileGlslToSpv(
      std::string(kMinimalShader), shaderc_glsl_fragment_shader, "shader");
  EXPECT_TRUE(CompilationResultIsSuccess(result1));
  EXPECT_TRUE(CompilationResultIsSuccess(result2));
  EXPECT_EQ(std::vector<uint32_t>(result1.cbegin(), result1.cend()),
            std::vector<uint32_t>(result2.cbegin(), result2.cend()));
}

TEST_F(CppInterface, ErrorsReported) {
  const SpvCompilationResult result = compiler_.CompileGlslToSpv(
      "int f(){return wrongname;}", shaderc_glsl_vertex_shader, "shader");
  ASSERT_FALSE(CompilationResultIsSuccess(result));
  EXPECT_THAT(result.GetErrorMessage(), HasSubstr("wrongname"));
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST_F(CppInterface, MultipleThreadsCalling) {
  bool results[10];
  std::vector<std::thread> threads;
  for (auto& r : results) {
    threads.emplace_back([this, &r]() {
      r = CompilationSuccess(kMinimalShader, shaderc_glsl_vertex_shader);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_THAT(results, Each(true));
}
#endif

TEST_F(CppInterface, AccessorsOnNullResultObject) {
  const SpvCompilationResult result(nullptr);
  EXPECT_FALSE(CompilationResultIsSuccess(result));
  EXPECT_EQ(std::string(), result.GetErrorMessage());
  EXPECT_EQ(result.cend(), result.cbegin());
  EXPECT_EQ(nullptr, result.cbegin());
  EXPECT_EQ(nullptr, result.cend());
  EXPECT_EQ(nullptr, result.begin());
  EXPECT_EQ(nullptr, result.end());
}

TEST_F(CppInterface, MacroCompileOptions) {
  options_.AddMacroDefinition("E", "main");
  const std::string kMinimalExpandedShader = "#version 150\nvoid E(){}";
  const std::string kMinimalDoubleExpandedShader = "#version 150\nF E(){}";
  EXPECT_TRUE(CompilationSuccess(kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader, options_));

  CompileOptions cloned_options(options_);
  // The simplest should still compile with the cloned options.
  EXPECT_TRUE(CompilationSuccess(kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader, cloned_options));

  EXPECT_FALSE(CompilationSuccess(kMinimalDoubleExpandedShader,
                                  shaderc_glsl_vertex_shader, cloned_options));

  cloned_options.AddMacroDefinition("F", "void");
  // This should still not work with the original options.
  EXPECT_FALSE(CompilationSuccess(kMinimalDoubleExpandedShader,
                                  shaderc_glsl_vertex_shader, options_));
  // This should work with the cloned options that have the additional
  // parameter.
  EXPECT_TRUE(CompilationSuccess(kMinimalDoubleExpandedShader,
                                 shaderc_glsl_vertex_shader, cloned_options));
}

TEST_F(CppInterface, D_DisassemblyOption) {
  const AssemblyCompilationResult result = compiler_.CompileGlslToSpvAssembly(
      kMinimalShader, shaderc_glsl_vertex_shader, "shader", options_);
  EXPECT_TRUE(CompilationResultIsSuccess(result));
  // This should work with both the glslang native disassembly format and the
  // SPIR-V Tools assembly format.
  EXPECT_THAT(CompilerOutputAsString(result), HasSubstr("Capability Shader"));
  EXPECT_THAT(CompilerOutputAsString(result), HasSubstr("MemoryModel"));

  CompileOptions cloned_options(options_);
  auto result_from_cloned_options = compiler_.CompileGlslToSpvAssembly(
      kMinimalShader, shaderc_glsl_vertex_shader, "shader", cloned_options);
  EXPECT_TRUE(CompilationResultIsSuccess(result_from_cloned_options));
  // The mode should be carried into any clone of the original option object.
  EXPECT_THAT(CompilerOutputAsString(result_from_cloned_options),
              HasSubstr("Capability Shader"));
  EXPECT_THAT(CompilerOutputAsString(result_from_cloned_options),
              HasSubstr("MemoryModel"));
}

TEST_F(CppInterface, DisassembleMinimalShader) {
  const AssemblyCompilationResult result = compiler_.CompileGlslToSpvAssembly(
      kMinimalShader, shaderc_glsl_vertex_shader, "shader", options_);
  EXPECT_TRUE(CompilationResultIsSuccess(result));
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(CompilerOutputAsString(result), HasSubstr(substring));
  }
}

TEST_F(CppInterface, ForcedVersionProfileCorrectStd) {
  // Forces the version and profile to 450core, which fixes the missing
  // #version.
  options_.SetForcedVersionProfile(450, shaderc_profile_core);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kCoreVertShaderWithoutVersion,
                                 shaderc_glsl_vertex_shader, options_));
}

TEST_F(CppInterface, ForcedVersionProfileCorrectStdClonedOptions) {
  // Forces the version and profile to 450core, which fixes the missing
  // #version.
  options_.SetForcedVersionProfile(450, shaderc_profile_core);
  CompileOptions cloned_options(options_);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kCoreVertShaderWithoutVersion,
                                 shaderc_glsl_vertex_shader, cloned_options));
}

TEST_F(CppInterface, ForcedVersionProfileInvalidModule) {
  // Forces the version and profile to 310es, while the source module is invalid
  // for this version of GLSL. Compilation should fail.
  options_.SetForcedVersionProfile(310, shaderc_profile_es);
  EXPECT_THAT(CompilationErrors(kCoreVertShaderWithoutVersion,
                                shaderc_glsl_vertex_shader, options_),
              HasSubstr("error: 'gl_ClipDistance' : undeclared identifier\n"));
}

TEST_F(CppInterface, ForcedVersionProfileConflictingStd) {
  // Forces the version and profile to 450core, which is in conflict with the
  // #version in shader.
  const std::string kVertexShader =
      std::string("#version 310 es\n") + kCoreVertShaderWithoutVersion;
  options_.SetForcedVersionProfile(450, shaderc_profile_core);
  EXPECT_THAT(
      CompilationWarnings(kVertexShader, shaderc_glsl_vertex_shader, options_),
      HasSubstr("warning: (version, profile) forced to be (450, core), "
                "while in source code it is (310, es)\n"));
}

TEST_F(CppInterface, ForcedVersionProfileUnknownVersionStd) {
  // Forces the version and profile to 4242core, which is an unknown version.
  options_.SetForcedVersionProfile(4242 /*unknown version*/,
                                   shaderc_profile_core);
  auto const errs =
      CompilationErrors(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  EXPECT_THAT(errs,
              HasSubstr("warning: (version, profile) forced to be (4242, core),"
                        " while in source code it is (140, none)\n"));
  EXPECT_THAT(errs, HasSubstr("error: version not supported\n"));
}

TEST_F(CppInterface, ForcedVersionProfileVersionsBefore150) {
  // Versions before 150 do not allow a profile token, shaderc_profile_none
  // should be passed down as the profile parameter.
  options_.SetForcedVersionProfile(140, shaderc_profile_none);
  EXPECT_TRUE(
      CompilationSuccess(kMinimalShader, shaderc_glsl_vertex_shader, options_));
}

TEST_F(CppInterface, ForcedVersionProfileRedundantProfileStd) {
  // Forces the version and profile to 100core. But versions before 150 do not
  // allow a profile token, compilation should fail.
  options_.SetForcedVersionProfile(100, shaderc_profile_core);
  EXPECT_THAT(
      CompilationErrors(kMinimalShader, shaderc_glsl_vertex_shader, options_),
      HasSubstr("error: #version: versions before 150 do not allow a profile "
                "token\n"));
}

TEST_F(CppInterface, GenerateDebugInfoBinary) {
  options_.SetGenerateDebugInfo();
  const std::string binary_output =
      CompilationOutput(kMinimalDebugInfoShader,
                        shaderc_glsl_vertex_shader, options_);
  // The binary output should contain the name of the vector (debug_info_sample)
  // null-terminated, as well as the whole original source.
  std::string vector_name("debug_info_sample");
  vector_name.resize(vector_name.size() + 1);
  EXPECT_THAT(binary_output, HasSubstr(vector_name));
  EXPECT_THAT(binary_output, HasSubstr(kMinimalDebugInfoShader));
}

TEST_F(CppInterface, GenerateDebugInfoBinaryClonedOptions) {
  options_.SetGenerateDebugInfo();
  CompileOptions cloned_options(options_);
  const std::string binary_output =
      CompilationOutput(kMinimalDebugInfoShader,
                        shaderc_glsl_vertex_shader, cloned_options);
  // The binary output should contain the name of the vector (debug_info_sample)
  // null-terminated, as well as the whole original source.
  std::string vector_name("debug_info_sample");
  vector_name.resize(vector_name.size() + 1);
  EXPECT_THAT(binary_output, HasSubstr(vector_name));
  EXPECT_THAT(binary_output, HasSubstr(kMinimalDebugInfoShader));
}

TEST_F(CppInterface, GenerateDebugInfoDisassembly) {
  options_.SetGenerateDebugInfo();
  // Debug info should also be emitted in disassembly mode.
  // The output disassembly should contain the name of the vector:
  // debug_info_sample.
  EXPECT_THAT(AssemblyOutput(kMinimalDebugInfoShader,
                             shaderc_glsl_vertex_shader, options_),
              HasSubstr("debug_info_sample"));
}

TEST_F(CppInterface, GenerateDebugInfoDisassemblyClonedOptions) {
  options_.SetGenerateDebugInfo();
  // Generate debug info mode should be carried to the cloned options.
  CompileOptions cloned_options(options_);
  EXPECT_THAT(CompilationOutput(kMinimalDebugInfoShader,
                                shaderc_glsl_vertex_shader, cloned_options),
              HasSubstr("debug_info_sample"));
}

TEST_F(CppInterface, CompileAndOptimizeWithLevelZero) {
  options_.SetOptimizationLevel(shaderc_optimization_level_zero);
  const std::string disassembly_text =
      AssemblyOutput(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CppInterface, CompileAndOptimizeWithLevelPerformance) {
  options_.SetOptimizationLevel(shaderc_optimization_level_performance);
  const std::string disassembly_text = AssemblyOutput(
      kGlslMultipleFnShader, shaderc_glsl_fragment_shader, options_);
  // Check that we do not have function calls anymore.
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpFunctionCall")));
}

TEST_F(CppInterface, CompileAndOptimizeWithLevelSize) {
  options_.SetOptimizationLevel(shaderc_optimization_level_size);
  const std::string disassembly_text =
      AssemblyOutput(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we do not have debug instructions.
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpName")));
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpSource")));
}

TEST_F(CppInterface, CompileAndOptimizeForVulkan10Failure) {
  options_.SetSourceLanguage(shaderc_source_language_hlsl);
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_0);
  options_.SetOptimizationLevel(shaderc_optimization_level_performance);

  EXPECT_THAT(CompilationErrors(kHlslWaveActiveSumeComputeShader,
                                shaderc_compute_shader, options_),
              // TODO(antiagainst): the error message can be improved to be more
              // explicit regarding Vulkan 1.1
              HasSubstr("compilation succeeded but failed to optimize: "
                        "Invalid capability operand"));
}

TEST_F(CppInterface, CompileAndOptimizeForVulkan11Success) {
  options_.SetSourceLanguage(shaderc_source_language_hlsl);
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_1);
  options_.SetOptimizationLevel(shaderc_optimization_level_performance);

  const std::string disassembly_text = AssemblyOutput(
      kHlslWaveActiveSumeComputeShader, shaderc_compute_shader, options_);
  EXPECT_THAT(disassembly_text, HasSubstr("OpGroupNonUniformIAdd"));
}

TEST_F(CppInterface, FollowingOptLevelOverridesPreviousOne) {
  options_.SetOptimizationLevel(shaderc_optimization_level_size);
  // Optimization level settings overridden by
  options_.SetOptimizationLevel(shaderc_optimization_level_zero);
  const std::string disassembly_text =
      AssemblyOutput(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CppInterface, GenerateDebugInfoOverridesOptimizationLevel) {
  options_.SetOptimizationLevel(shaderc_optimization_level_size);
  // Optimization level settings overridden by
  options_.SetGenerateDebugInfo();
  const std::string disassembly_text =
      AssemblyOutput(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  for (const auto& substring : kMinimalShaderDebugInfoDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CppInterface, GenerateDebugInfoProhibitsOptimizationLevel) {
  // Setting generate debug info first also works.
  options_.SetGenerateDebugInfo();
  options_.SetOptimizationLevel(shaderc_optimization_level_size);
  const std::string disassembly_text =
      AssemblyOutput(kMinimalShader, shaderc_glsl_vertex_shader, options_);
  for (const auto& substring : kMinimalShaderDebugInfoDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CppInterface, GetNumErrors) {
  std::string shader(kTwoErrorsShader);
  const SpvCompilationResult compilation_result =
      compiler_.CompileGlslToSpv(kTwoErrorsShader, strlen(kTwoErrorsShader),
                                 shaderc_glsl_vertex_shader, "shader");
  EXPECT_FALSE(CompilationResultIsSuccess(compilation_result));
  EXPECT_EQ(2u, compilation_result.GetNumErrors());
  EXPECT_EQ(0u, compilation_result.GetNumWarnings());
}

TEST_F(CppInterface, GetNumWarnings) {
  const SpvCompilationResult compilation_result =
      compiler_.CompileGlslToSpv(kTwoWarningsShader, strlen(kTwoWarningsShader),
                                 shaderc_glsl_vertex_shader, "shader");
  EXPECT_TRUE(CompilationResultIsSuccess(compilation_result));
  EXPECT_EQ(2u, compilation_result.GetNumWarnings());
  EXPECT_EQ(0u, compilation_result.GetNumErrors());
}

TEST_F(CppInterface, ZeroErrorsZeroWarnings) {
  const SpvCompilationResult compilation_result =
      compiler_.CompileGlslToSpv(kMinimalShader, strlen(kMinimalShader),
                                 shaderc_glsl_vertex_shader, "shader");
  EXPECT_TRUE(CompilationResultIsSuccess(compilation_result));
  EXPECT_EQ(0u, compilation_result.GetNumErrors());
  EXPECT_EQ(0u, compilation_result.GetNumWarnings());
}

TEST_F(CppInterface, ErrorTypeUnknownShaderStage) {
  // The shader kind/stage can not be determined, the error type field should
  // indicate the error type is shaderc_shader_kind_error.
  const SpvCompilationResult compilation_result =
      compiler_.CompileGlslToSpv(kMinimalShader, strlen(kMinimalShader),
                                 shaderc_glsl_infer_from_source, "shader");
  EXPECT_EQ(shaderc_compilation_status_invalid_stage,
            compilation_result.GetCompilationStatus());
}

TEST_F(CppInterface, ErrorTypeCompilationError) {
  // The shader kind is valid, the result object's error type field should
  // indicate this compilaion fails due to compilation errors.
  const SpvCompilationResult compilation_result = compiler_.CompileGlslToSpv(
      kTwoErrorsShader, shaderc_glsl_vertex_shader, "shader");
  EXPECT_EQ(shaderc_compilation_status_compilation_error,
            compilation_result.GetCompilationStatus());
}

TEST_F(CppInterface, ErrorTagIsInputFileName) {
  std::string shader(kTwoErrorsShader);
  const SpvCompilationResult compilation_result =
      compiler_.CompileGlslToSpv(kTwoErrorsShader, strlen(kTwoErrorsShader),
                                 shaderc_glsl_vertex_shader, "SampleInputFile");
  // Expects compilation failure errors. The error tag should be
  // 'SampleInputFile'
  EXPECT_FALSE(CompilationResultIsSuccess(compilation_result));
  EXPECT_THAT(compilation_result.GetErrorMessage(),
              HasSubstr("SampleInputFile:3: error:"));
}

TEST_F(CppInterface, PreprocessingOnlyOption) {
  const PreprocessedSourceCompilationResult result = compiler_.PreprocessGlsl(
      kMinimalShaderWithMacro, shaderc_glsl_vertex_shader, "shader", options_);
  EXPECT_TRUE(CompilationResultIsSuccess(result));
  EXPECT_THAT(CompilerOutputAsString(result), HasSubstr("void main(){ }"));

  const std::string kMinimalShaderCloneOption =
      "#version 140\n"
      "#define E_CLONE_OPTION main\n"
      "void E_CLONE_OPTION(){}\n";
  CompileOptions cloned_options(options_);
  const PreprocessedSourceCompilationResult result_from_cloned_options =
      compiler_.PreprocessGlsl(kMinimalShaderCloneOption,
                               shaderc_glsl_vertex_shader, "shader",
                               cloned_options);
  EXPECT_TRUE(CompilationResultIsSuccess(result_from_cloned_options));
  EXPECT_THAT(CompilerOutputAsString(result_from_cloned_options),
              HasSubstr("void main(){ }"));
}

// A shader kind test case needs: 1) A shader text with or without #pragma
// annotation, 2) shader_kind.
struct ShaderKindTestCase {
  const char* shader_;
  shaderc_shader_kind shader_kind_;
};

// Test the shader kind deduction process. If the shader kind is one
// of the non-default ones, the compiler will just try to compile the
// source code in that specified shader kind. If the shader kind is
// shaderc_glsl_deduce_from_pragma, the compiler will determine the shader
// kind from #pragma annotation in the source code and emit error if none
// such annotation is found. When the shader kind is one of the default
// ones, the compiler will fall back to use the specified shader kind if
// and only if #pragma annoation is not found.

// Valid shader kind settings should generate valid SPIR-V code.
using ValidShaderKind = testing::TestWithParam<ShaderKindTestCase>;

TEST_P(ValidShaderKind, ValidSpvCode) {
  const ShaderKindTestCase& test_case = GetParam();
  shaderc::Compiler compiler;
  EXPECT_TRUE(
      CompilesToValidSpv(compiler, test_case.shader_, test_case.shader_kind_));
}

INSTANTIATE_TEST_SUITE_P(
    CompileStringTest, ValidShaderKind,
    testing::ValuesIn(std::vector<ShaderKindTestCase>{
        // Valid default shader kinds.
        {kEmpty310ESShader, shaderc_glsl_default_vertex_shader},
        {kEmpty310ESShader, shaderc_glsl_default_fragment_shader},
        {kEmpty310ESShader, shaderc_glsl_default_compute_shader},
        {kGeometryOnlyShader, shaderc_glsl_default_geometry_shader},
        {kTessControlOnlyShader, shaderc_glsl_default_tess_control_shader},
        {kTessEvaluationOnlyShader,
         shaderc_glsl_default_tess_evaluation_shader},

        // #pragma annotation overrides default shader kinds.
        {kVertexOnlyShaderWithPragma, shaderc_glsl_default_compute_shader},
        {kFragmentOnlyShaderWithPragma, shaderc_glsl_default_vertex_shader},
        {kTessControlOnlyShaderWithPragma,
         shaderc_glsl_default_fragment_shader},
        {kTessEvaluationOnlyShaderWithPragma,
         shaderc_glsl_default_tess_control_shader},
        {kGeometryOnlyShaderWithPragma,
         shaderc_glsl_default_tess_evaluation_shader},
        {kComputeOnlyShaderWithPragma, shaderc_glsl_default_geometry_shader},

        // Specified non-default shader kind overrides #pragma annotation.
        {kVertexOnlyShaderWithInvalidPragma, shaderc_glsl_vertex_shader},
    }));

// Invalid shader kind settings should generate errors.
using InvalidShaderKind = testing::TestWithParam<ShaderKindTestCase>;

TEST_P(InvalidShaderKind, CompilationShouldFail) {
  const ShaderKindTestCase& test_case = GetParam();
  shaderc::Compiler compiler;
  EXPECT_FALSE(
      CompilesToValidSpv(compiler, test_case.shader_, test_case.shader_kind_));
}

INSTANTIATE_TEST_SUITE_P(
    CompileStringTest, InvalidShaderKind,
    testing::ValuesIn(std::vector<ShaderKindTestCase>{
        // Invalid default shader kind.
        {kVertexOnlyShader, shaderc_glsl_default_fragment_shader},
        // Sets to deduce shader kind from #pragma, but #pragma is defined in
        // the source code.
        {kVertexOnlyShader, shaderc_glsl_infer_from_source},
        // Invalid #pragma cause errors, even though default shader kind is set
        // to valid shader kind.
        {kVertexOnlyShaderWithInvalidPragma,
         shaderc_glsl_default_vertex_shader},
    }));

// To test file inclusion, use an unordered_map as a fake file system to store
// fake files to be included. The unordered_map represents a filesystem by
// mapping filename (or path) string to the contents of that file as a string.
using FakeFS = std::unordered_map<std::string, std::string>;

// An includer test case needs: 1) A fake file system which is actually an
// unordered_map, so that we can resolve the content given a string. A valid
// fake file system must have one entry with key:'root' to specify the start
// shader file for compilation. 2) An string that we expect to see in the
// compilation output.
class IncluderTestCase {
 public:
  IncluderTestCase(FakeFS fake_fs, std::string expected_substring)
      : fake_fs_(fake_fs), expected_substring_(expected_substring) {
    assert(fake_fs_.find("root") != fake_fs_.end() &&
           "Valid fake file system needs a 'root' file\n");
  }

  const FakeFS& fake_fs() const { return fake_fs_; }
  const std::string& expected_substring() const { return expected_substring_; }

 private:
  FakeFS fake_fs_;
  std::string expected_substring_;
};

// A mock class that simulates an includer. This class implements
// IncluderInterface to provide GetInclude() and ReleaseInclude() methods.
class TestIncluder : public shaderc::CompileOptions::IncluderInterface {
 public:
  explicit TestIncluder(const FakeFS& fake_fs)
      : fake_fs_(fake_fs), responses_({}) {}

  // Get path and content from the fake file system.
  shaderc_include_result* GetInclude(const char* requested_source,
                                     shaderc_include_type type,
                                     const char* requesting_source,
                                     size_t include_depth) override {
    responses_.emplace_back(shaderc_include_result{
        requested_source, strlen(requested_source),
        fake_fs_.at(std::string(requested_source)).c_str(),
        fake_fs_.at(std::string(requested_source)).size()});
    return &responses_.back();
  }

  // Response data is owned as private property, no need to release explicitly.
  void ReleaseInclude(shaderc_include_result*) override {}

 private:
  const FakeFS& fake_fs_;
  std::vector<shaderc_include_result> responses_;
};

using IncluderTests = testing::TestWithParam<IncluderTestCase>;

// Parameterized tests for includer.
TEST_P(IncluderTests, SetIncluder) {
  const IncluderTestCase& test_case = GetParam();
  const FakeFS& fs = test_case.fake_fs();
  const std::string& shader = fs.at("root");
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetIncluder(std::unique_ptr<TestIncluder>(new TestIncluder(fs)));
  const auto compilation_result = compiler.PreprocessGlsl(
      shader.c_str(), shaderc_glsl_vertex_shader, "shader", options);
  // Checks the existence of the expected string.
  EXPECT_THAT(CompilerOutputAsString(compilation_result),
              HasSubstr(test_case.expected_substring()));
}

TEST_P(IncluderTests, SetIncluderClonedOptions) {
  const IncluderTestCase& test_case = GetParam();
  const FakeFS& fs = test_case.fake_fs();
  const std::string& shader = fs.at("root");
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetIncluder(std::unique_ptr<TestIncluder>(new TestIncluder(fs)));

  // Cloned options should have all the settings.
  CompileOptions cloned_options(options);
  const auto compilation_result = compiler.PreprocessGlsl(
      shader.c_str(), shaderc_glsl_vertex_shader, "shader", cloned_options);
  // Checks the existence of the expected string.
  EXPECT_THAT(CompilerOutputAsString(compilation_result),
              HasSubstr(test_case.expected_substring()));
}

INSTANTIATE_TEST_SUITE_P(CppInterface, IncluderTests,
                        testing::ValuesIn(std::vector<IncluderTestCase>{
                            IncluderTestCase(
                                // Fake file system.
                                {
                                    {"root",
                                     "#version 150\n"
                                     "void foo() {}\n"
                                     "#include \"path/to/file_1\"\n"},
                                    {"path/to/file_1", "content of file_1\n"},
                                },
                                // Expected output.
                                "#line 0 \"path/to/file_1\"\n"
                                " content of file_1\n"
                                "#line 3"),
                            IncluderTestCase(
                                // Fake file system.
                                {{"root",
                                  "#version 150\n"
                                  "void foo() {}\n"
                                  "#include \"path/to/file_1\"\n"},
                                 {"path/to/file_1",
                                  "#include \"path/to/file_2\"\n"
                                  "content of file_1\n"},
                                 {"path/to/file_2", "content of file_2\n"}},
                                // Expected output.
                                "#line 0 \"path/to/file_1\"\n"
                                "#line 0 \"path/to/file_2\"\n"
                                " content of file_2\n"
                                "#line 1 \"path/to/file_1\"\n"
                                " content of file_1\n"
                                "#line 3"),

                        }));

TEST_F(CppInterface, WarningsOnLine) {
  // By default the compiler will emit a warning on line 2 complaining
  // that 'float' is a deprecated attribute in version 130.
  EXPECT_THAT(
      CompilationWarnings(kDeprecatedAttributeShader,
                          shaderc_glsl_vertex_shader, CompileOptions()),
      HasSubstr(":2: warning: attribute deprecated in version 130; may be "
                "removed in future release\n"));
}

TEST_F(CppInterface, SuppressWarningsOnLine) {
  // Sets the compiler to suppress warnings, so that the deprecated attribute
  // warning won't be emitted.
  options_.SetSuppressWarnings();
  EXPECT_EQ("", CompilationWarnings(kDeprecatedAttributeShader,
                                    shaderc_glsl_vertex_shader, options_));
}

TEST_F(CppInterface, SuppressWarningsOnLineClonedOptions) {
  // Sets the compiler to suppress warnings, so that the deprecated attribute
  // warning won't be emitted, and the mode should be carried into any clone of
  // the original option object.
  options_.SetSuppressWarnings();
  CompileOptions cloned_options(options_);
  EXPECT_EQ("",
            CompilationWarnings(kDeprecatedAttributeShader,
                                shaderc_glsl_vertex_shader, cloned_options));
}

TEST_F(CppInterface, WarningsOnLineAsErrors) {
  // Sets the compiler to make warnings into errors. So that the deprecated
  // attribute warning will be emitted as an error and compilation should fail.
  options_.SetWarningsAsErrors();
  EXPECT_THAT(
      CompilationErrors(kDeprecatedAttributeShader, shaderc_glsl_vertex_shader,
                        options_),
      HasSubstr(":2: error: attribute deprecated in version 130; may be "
                "removed in future release\n"));
}

TEST_F(CppInterface, WarningsOnLineAsErrorsClonedOptions) {
  // Sets the compiler to make warnings into errors. So that the deprecated
  // attribute warning will be emitted as an error and compilation should fail.
  options_.SetWarningsAsErrors();
  CompileOptions cloned_options(options_);
  // The error message should show an error instead of a warning.
  EXPECT_THAT(
      CompilationErrors(kDeprecatedAttributeShader, shaderc_glsl_vertex_shader,
                        cloned_options),
      HasSubstr(":2: error: attribute deprecated in version 130; may be "
                "removed in future release\n"));
}

TEST_F(CppInterface, GlobalWarnings) {
  // By default the compiler will emit a warning as version 550 is an unknown
  // version.
  options_.SetForcedVersionProfile(400, shaderc_profile_core);
  EXPECT_THAT(CompilationWarnings(kMinimalUnknownVersionShader,
                                  shaderc_glsl_vertex_shader, options_),
              HasSubstr("(version, profile) forced to be (400, core),"
                        " while in source code it is (550, none)\n"));
}

TEST_F(CppInterface, SuppressGlobalWarnings) {
  // Sets the compiler to suppress warnings, so that the unknown version warning
  // won't be emitted.
  options_.SetSuppressWarnings();
  options_.SetForcedVersionProfile(400, shaderc_profile_core);
  EXPECT_THAT(CompilationWarnings(kMinimalUnknownVersionShader,
                                  shaderc_glsl_vertex_shader, options_),
              Eq(""));
}

TEST_F(CppInterface, SuppressGlobalWarningsClonedOptions) {
  // Sets the compiler to suppress warnings, so that the unknown version warning
  // won't be emitted, and the mode should be carried into any clone of the
  // original option object.
  options_.SetSuppressWarnings();
  options_.SetForcedVersionProfile(400, shaderc_profile_core);
  CompileOptions cloned_options(options_);
  EXPECT_THAT(CompilationWarnings(kMinimalUnknownVersionShader,
                                  shaderc_glsl_vertex_shader, cloned_options),
              Eq(""));
}

TEST_F(CppInterface, GlobalWarningsAsErrors) {
  // Sets the compiler to make warnings into errors. So that the unknown
  // version warning will be emitted as an error and compilation should fail.
  options_.SetWarningsAsErrors();
  options_.SetForcedVersionProfile(400, shaderc_profile_core);
  EXPECT_THAT(CompilationErrors(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, options_),
              HasSubstr("(version, profile) forced to be (400, core),"
                        " while in source code it is (550, none)\n"));
}

TEST_F(CppInterface, GlobalWarningsAsErrorsClonedOptions) {
  // Sets the compiler to make warnings into errors. This mode should be carried
  // into any clone of the original option object.
  options_.SetWarningsAsErrors();
  options_.SetForcedVersionProfile(400, shaderc_profile_core);
  CompileOptions cloned_options(options_);
  EXPECT_THAT(CompilationErrors(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, cloned_options),
              HasSubstr("(version, profile) forced to be (400, core),"
                        " while in source code it is (550, none)\n"));
}

TEST_F(CppInterface, SuppressWarningsModeFirstOverridesWarningsAsErrorsMode) {
  // Sets suppress-warnings mode first, then sets warnings-as-errors mode.
  // suppress-warnings mode should override warnings-as-errors mode, no
  // error message should be output for this case.
  options_.SetSuppressWarnings();
  options_.SetWarningsAsErrors();
  // Warnings on line should be inhibited.
  EXPECT_EQ("", CompilationWarnings(kDeprecatedAttributeShader,
                                    shaderc_glsl_vertex_shader, options_));

  // Global warnings should be inhibited.
  // However, the unknown version will cause an error.
  EXPECT_THAT(CompilationErrors(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, options_),
              Eq("shader: error: version not supported\n"));
}

TEST_F(CppInterface, SuppressWarningsModeSecondOverridesWarningsAsErrorsMode) {
  // Sets warnings-as-errors mode first, then sets suppress-warnings mode.
  // suppress-warnings mode should override warnings-as-errors mode, no
  // error message should be output for this case.
  options_.SetWarningsAsErrors();
  options_.SetSuppressWarnings();
  // Warnings on line should be inhibited.
  EXPECT_EQ("", CompilationWarnings(kDeprecatedAttributeShader,
                                    shaderc_glsl_vertex_shader, options_));

  // Global warnings should be inhibited.
  // However, the unknown version will cause an error.
  EXPECT_THAT(CompilationErrors(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, options_),
              Eq("shader: error: version not supported\n"));
}

TEST_F(CppInterface, TargetEnvCompileOptionsOpenGLCompatibilityShadersFail) {
  // Glslang does not support SPIR-V code generation for OpenGL compatibility
  // profile.
  options_.SetTargetEnvironment(shaderc_target_env_opengl_compat, 0);
  const std::string kGlslShader =
      R"(#version 150 compatibility
       uniform highp sampler2D tex;
       void main() {
         gl_FragColor = texture2D(tex, vec2(0.0,0.0));
       }
  )";

  EXPECT_THAT(
      CompilationErrors(kGlslShader, shaderc_glsl_fragment_shader, options_),
      HasSubstr(
          "compilation for SPIR-V does not support the compatibility profile"));
}

std::string BarrierComputeShader() {
  return R"(#version 450
    void main() { barrier(); })";
};

std::string SubgroupBarrierComputeShader() {
  return R"(#version 450
    #extension GL_KHR_shader_subgroup_basic : enable
    void main() { subgroupBarrier(); })";
};

TEST_F(CppInterface, TargetEnvCompileOptionsVulkanEnvVulkan1_0ShaderSucceeds) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan, 0);
  EXPECT_TRUE(CompilationSuccess(BarrierComputeShader(),
                                 shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface, TargetEnvCompileOptionsVulkanEnvVulkan1_0ShaderFails) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan, 0);
  EXPECT_FALSE(CompilationSuccess(SubgroupBarrierComputeShader(),
                                  shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface,
       TargetEnvCompileOptionsVulkan1_0EnvVulkan1_0ShaderSucceeds) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_0);
  EXPECT_TRUE(CompilationSuccess(BarrierComputeShader(),
                                 shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface, TargetEnvCompileOptionsVulkan1_0EnvVulkan1_1ShaderFails) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_0);
  EXPECT_FALSE(CompilationSuccess(SubgroupBarrierComputeShader(),
                                  shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface,
       TargetEnvCompileOptionsVulkan1_1EnvVulkan1_0ShaderSucceeds) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilationSuccess(BarrierComputeShader(),
                                 shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface,
       TargetEnvCompileOptionsVulkan1_1EnvVulkan1_1ShaderSucceeds) {
  options_.SetTargetEnvironment(shaderc_target_env_vulkan,
                                shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilationSuccess(SubgroupBarrierComputeShader(),
                                 shaderc_glsl_compute_shader, options_));
}

TEST_F(CppInterface, BeginAndEndOnSpvCompilationResult) {
  const SpvCompilationResult compilation_result = compiler_.CompileGlslToSpv(
      kMinimalShader, shaderc_glsl_vertex_shader, "shader");
  EXPECT_TRUE(IsValidSpv(compilation_result));
  // Use range-based for to exercise begin() and end().
  std::vector<uint32_t> binary_words;
  for (const auto& element : compilation_result) {
    binary_words.push_back(element);
  }
  EXPECT_THAT(binary_words,
              Eq(std::vector<uint32_t>(compilation_result.cbegin(),
                                       compilation_result.cend())));
}

TEST_F(CppInterface, BeginAndEndOnAssemblyCompilationResult) {
  const AssemblyCompilationResult compilation_result =
      compiler_.CompileGlslToSpvAssembly(
          kMinimalShader, shaderc_glsl_vertex_shader, "shader", options_);
  const std::string forced_to_be_a_string =
      CompilerOutputAsString(compilation_result);
  EXPECT_THAT(forced_to_be_a_string, HasSubstr("MemoryModel"));
  const std::string string_via_begin_end(compilation_result.begin(),
                                         compilation_result.end());
  EXPECT_THAT(string_via_begin_end, Eq(forced_to_be_a_string));
}

TEST_F(CppInterface, BeginAndEndOnPreprocessedResult) {
  const PreprocessedSourceCompilationResult compilation_result =
      compiler_.PreprocessGlsl(kMinimalShader, shaderc_glsl_vertex_shader,
                               "shader", options_);
  const std::string forced_to_be_a_string =
      CompilerOutputAsString(compilation_result);
  EXPECT_THAT(forced_to_be_a_string, HasSubstr("void main()"));
  const std::string string_via_begin_end(compilation_result.begin(),
                                         compilation_result.end());
  EXPECT_THAT(string_via_begin_end, Eq(forced_to_be_a_string));
}

TEST_F(CppInterface, SourceLangGlslMinimalGlslVertexShaderSucceeds) {
  options_.SetSourceLanguage(shaderc_source_language_glsl);
  EXPECT_TRUE(CompilationSuccess(kVertexOnlyShader, shaderc_glsl_vertex_shader,
                                 options_));
}

TEST_F(CppInterface, SourceLangGlslMinimalHlslVertexShaderFails) {
  options_.SetSourceLanguage(shaderc_source_language_glsl);
  EXPECT_FALSE(CompilationSuccess(kMinimalHlslShader,
                                  shaderc_glsl_vertex_shader, options_));
}

TEST_F(CppInterface, SourceLangHlslMinimalGlslVertexShaderFails) {
  options_.SetSourceLanguage(shaderc_source_language_hlsl);
  EXPECT_FALSE(CompilationSuccess(kVertexOnlyShader, shaderc_glsl_vertex_shader,
                                  options_));
}

TEST_F(CppInterface, SourceLangHlslMinimalHlslVertexShaderSucceeds) {
  options_.SetSourceLanguage(shaderc_source_language_hlsl);
  EXPECT_TRUE(CompilationSuccess(kMinimalHlslShader, shaderc_glsl_vertex_shader,
                                 options_));
}

TEST(
    EntryPointTest,
    SourceLangHlslMinimalHlslVertexShaderAsConstCharPtrSucceedsWithEntryPointName) {
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  auto result = compiler.CompileGlslToSpv(
      kMinimalHlslShader, strlen(kMinimalHlslShader),
      shaderc_glsl_vertex_shader, "shader", "EntryPoint", options);
  std::vector<uint32_t> binary(result.begin(), result.end());
  std::string assembly;
  spvtools::SpirvTools(SPV_ENV_UNIVERSAL_1_0).Disassemble(binary, &assembly);
  EXPECT_THAT(assembly,
              HasSubstr("OpEntryPoint Vertex %EntryPoint \"EntryPoint\""))
      << assembly;
}

TEST(
    EntryPointTest,
    SourceLangHlslMinimalHlslVertexShaderAsStdStringSucceedsWithEntryPointName) {
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  std::string shader(kMinimalHlslShader);
  auto result = compiler.CompileGlslToSpv(shader, shaderc_glsl_vertex_shader,
                                          "shader", "EntryPoint", options);
  std::vector<uint32_t> binary(result.begin(), result.end());
  std::string assembly;
  spvtools::SpirvTools(SPV_ENV_UNIVERSAL_1_0).Disassemble(binary, &assembly);
  EXPECT_THAT(assembly,
              HasSubstr("OpEntryPoint Vertex %EntryPoint \"EntryPoint\""))
      << assembly;
}

TEST(
    EntryPointTest,
    SourceLangHlslMinimalHlslVertexShaderAsConstCharPtrSucceedsToAssemblyWithEntryPointName) {
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  auto assembly = compiler.CompileGlslToSpvAssembly(
      kMinimalHlslShader, strlen(kMinimalHlslShader),
      shaderc_glsl_vertex_shader, "shader", "EntryPoint", options);
  EXPECT_THAT(std::string(assembly.begin(), assembly.end()),
              HasSubstr("OpEntryPoint Vertex %EntryPoint \"EntryPoint\""));
}

TEST(
    EntryPointTest,
    SourceLangHlslMinimalHlslVertexShaderAsStdStringSucceedsToAssemblyWithEntryPointName) {
  shaderc::Compiler compiler;
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  std::string shader(kMinimalHlslShader);
  auto assembly = compiler.CompileGlslToSpvAssembly(
      shader, shaderc_glsl_vertex_shader, "shader", "EntryPoint", options);
  EXPECT_THAT(std::string(assembly.begin(), assembly.end()),
              HasSubstr("OpEntryPoint Vertex %EntryPoint \"EntryPoint\""));
}

// Returns a fragment shader accessing a texture with the given
// offset.
std::string ShaderWithTexOffset(int offset) {
  std::ostringstream oss;
  oss << "#version 450\n"
         "layout (binding=0) uniform sampler1D tex;\n"
         "void main() { vec4 x = textureOffset(tex, 1.0, "
      << offset << "); }\n";
  return oss.str();
}

// Ensure compilation is sensitive to limit setting.  Sample just
// two particular limits.
TEST_F(CppInterface, LimitsTexelOffsetDefault) {
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(-9).c_str(),
                                  shaderc_glsl_fragment_shader, options_));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(-8).c_str(),
                                 shaderc_glsl_fragment_shader, options_));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(7).c_str(),
                                 shaderc_glsl_fragment_shader, options_));
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(8).c_str(),
                                  shaderc_glsl_fragment_shader, options_));
}

TEST_F(CppInterface, LimitsTexelOffsetLowerMinimum) {
  options_.SetLimit(shaderc_limit_min_program_texel_offset, -99);
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(-100).c_str(),
                                  shaderc_glsl_fragment_shader, options_));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(-99).c_str(),
                                 shaderc_glsl_fragment_shader, options_));
}

TEST_F(CppInterface, LimitsTexelOffsetHigherMaximum) {
  options_.SetLimit(shaderc_limit_max_program_texel_offset, 10);
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(10).c_str(),
                                 shaderc_glsl_fragment_shader, options_));
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(11).c_str(),
                                  shaderc_glsl_fragment_shader, options_));
}

TEST_F(CppInterface, UniformsWithoutBindingsFailCompilation) {
  CompileOptions options;
  const std::string errors = CompilationErrors(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(errors,
              HasSubstr("sampler/texture/image requires layout(binding=X)"));
}

TEST_F(CppInterface,
       UniformsWithoutBindingsOptionSetAutoBindingsAssignsBindings) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  const std::string disassembly_text = AssemblyOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 4"));
}

TEST_F(CppInterface, SetBindingBaseForTextureAdjustsTextureBindingsOnly) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  options.SetBindingBase(shaderc_uniform_kind_texture, 44);
  const std::string disassembly_text = AssemblyOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 3"));
}

TEST_F(CppInterface, SetBindingBaseForSamplerAdjustsSamplerBindingsOnly) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  options.SetBindingBase(shaderc_uniform_kind_sampler, 44);
  const std::string disassembly_text = AssemblyOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 3"));
}

TEST_F(CppInterface, SetBindingBaseForImageAdjustsImageBindingsOnly) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  options.SetBindingBase(shaderc_uniform_kind_image, 44);
  const std::string disassembly_text = AssemblyOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 45"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 2"));
}

TEST_F(CppInterface, SetBindingBaseForBufferAdjustsBufferBindingsOnly) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  options.SetBindingBase(shaderc_uniform_kind_buffer, 44);
  const std::string disassembly_text = AssemblyOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 44"));
}

TEST_F(CppInterface, SetBindingBaseSurvivesCloning) {
  CompileOptions options;
  options.SetAutoBindUniforms(true);
  options.SetBindingBase(shaderc_uniform_kind_texture, 40);
  options.SetBindingBase(shaderc_uniform_kind_sampler, 50);
  options.SetBindingBase(shaderc_uniform_kind_image, 60);
  options.SetBindingBase(shaderc_uniform_kind_buffer, 70);
  CompileOptions cloned_options(options);
  const std::string disassembly_text =
      AssemblyOutput(kShaderWithUniformsWithoutBindings,
                     shaderc_glsl_vertex_shader, cloned_options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 40"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 50"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 60"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 61"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 70"));
}

TEST_F(CppInterface, GlslDefaultPackingUsed) {
  CompileOptions options;
  const std::string disassembly_text = AssemblyOutput(
      kGlslShaderWeirdPacking, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 16"));
}

TEST_F(CppInterface, HlslOffsetsOptionDisableRespected) {
  CompileOptions options;
  options.SetHlslOffsets(false);
  const std::string disassembly_text = AssemblyOutput(
      kGlslShaderWeirdPacking, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 16"));
}

TEST_F(CppInterface, HlslOffsetsOptionEnableRespected) {
  CompileOptions options;
  options.SetHlslOffsets(true);
  const std::string disassembly_text = AssemblyOutput(
      kGlslShaderWeirdPacking, shaderc_glsl_vertex_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 4"));
}

TEST_F(CppInterface, HlslRegSetBindingForFragmentRespected) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  options.SetHlslRegisterSetAndBindingForStage(shaderc_fragment_shader, "t4",
                                               "9", "16");
  const std::string disassembly_text = AssemblyOutput(
      kHlslFragShaderWithRegisters, shaderc_glsl_fragment_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 DescriptorSet 9"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 Binding 16"));
}

TEST_F(CppInterface, HlslRegSetBindingForDifferentStageIgnored) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  options.SetHlslRegisterSetAndBindingForStage(shaderc_vertex_shader, "t4", "9",
                                               "16");
  const std::string disassembly_text = AssemblyOutput(
      kHlslFragShaderWithRegisters, shaderc_glsl_fragment_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 DescriptorSet 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 Binding 4"));
}

TEST_F(CppInterface, HlslRegSetBindingForAllStagesRespected) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  options.SetHlslRegisterSetAndBinding("t4", "9", "16");
  const std::string disassembly_text = AssemblyOutput(
      kHlslFragShaderWithRegisters, shaderc_glsl_fragment_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 DescriptorSet 9"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %t4 Binding 16"));
}

TEST_F(CppInterface, HlslFunctionality1OffByDefault) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  // The counter needs a binding, and there is no way to set it in the shader
  // source.
  options.SetAutoBindUniforms(true);
  const std::string disassembly_text = AssemblyOutput(
      kHlslShaderWithCounterBuffer, shaderc_glsl_fragment_shader, options);
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpDecorateStringGOOGLE")));
}

TEST_F(CppInterface, HlslFunctionality1Respected) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  // The counter needs a binding, and there is no way to set it in the shader
  // source.  https://github.com/KhronosGroup/glslang/issues/1616
  options.SetAutoBindUniforms(true);
  options.SetHlslFunctionality1(true);
  const std::string disassembly_text = AssemblyOutput(
      kHlslShaderWithCounterBuffer, shaderc_glsl_fragment_shader, options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorateStringGOOGLE"));
}

TEST_F(CppInterface, HlslFunctionality1SurvivesCloning) {
  CompileOptions options;
  options.SetSourceLanguage(shaderc_source_language_hlsl);
  options.SetHlslFunctionality1(true);
  // The counter needs a binding, and there is no way to set it in the shader
  // source. https://github.com/KhronosGroup/glslang/issues/1616
  options.SetAutoBindUniforms(true);
  CompileOptions cloned_options(options);
  const std::string disassembly_text = AssemblyOutput(
      kHlslShaderWithCounterBuffer, shaderc_glsl_fragment_shader, cloned_options);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorateStringGOOGLE"));
}

}  // anonymous namespace
