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

#include "common_shaders_for_test.h"
#include "shaderc/shaderc.h"

namespace {

using testing::Each;
using testing::HasSubstr;
using testing::Not;

TEST(Init, MultipleCalls) {
  shaderc_compiler_t compiler1, compiler2, compiler3;
  EXPECT_NE(nullptr, compiler1 = shaderc_compiler_initialize());
  EXPECT_NE(nullptr, compiler2 = shaderc_compiler_initialize());
  EXPECT_NE(nullptr, compiler3 = shaderc_compiler_initialize());
  shaderc_compiler_release(compiler1);
  shaderc_compiler_release(compiler2);
  shaderc_compiler_release(compiler3);
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST(Init, MultipleThreadsCalling) {
  shaderc_compiler_t compiler1, compiler2, compiler3;
  std::thread t1([&compiler1]() { compiler1 = shaderc_compiler_initialize(); });
  std::thread t2([&compiler2]() { compiler2 = shaderc_compiler_initialize(); });
  std::thread t3([&compiler3]() { compiler3 = shaderc_compiler_initialize(); });
  t1.join();
  t2.join();
  t3.join();
  EXPECT_NE(nullptr, compiler1);
  EXPECT_NE(nullptr, compiler2);
  EXPECT_NE(nullptr, compiler3);
  shaderc_compiler_release(compiler1);
  shaderc_compiler_release(compiler2);
  shaderc_compiler_release(compiler3);
}
#endif

TEST(Init, SPVVersion) {
  unsigned int version = 0;
  unsigned int revision = 0;
  shaderc_get_spv_version(&version, &revision);
  EXPECT_EQ(spv::Version, version);
  EXPECT_EQ(spv::Revision, revision);
}

// Determines the kind of output required from the compiler.
enum class OutputType {
  SpirvBinary,
  SpirvAssemblyText,
  PreprocessedText,
};

// Generate a compilation result object with the given compile,
// shader source, shader kind, input file name, entry point name, options,
// and for the specified output type.  The entry point name is only significant
// for HLSL compilation.
shaderc_compilation_result_t MakeCompilationResult(
    const shaderc_compiler_t compiler, const std::string& shader,
    shaderc_shader_kind kind, const char* input_file_name,
    const char* entry_point_name, const shaderc_compile_options_t options,
    OutputType output_type) {
  switch (output_type) {
    case OutputType::SpirvBinary:
      return shaderc_compile_into_spv(compiler, shader.c_str(), shader.size(),
                                      kind, input_file_name, entry_point_name,
                                      options);
      break;
    case OutputType::SpirvAssemblyText:
      return shaderc_compile_into_spv_assembly(
          compiler, shader.c_str(), shader.size(), kind, input_file_name,
          entry_point_name, options);
      break;
    case OutputType::PreprocessedText:
      return shaderc_compile_into_preprocessed_text(
          compiler, shader.c_str(), shader.size(), kind, input_file_name,
          entry_point_name, options);
      break;
  }
  // We shouldn't reach here.  But some compilers might not know that.
  // Be a little defensive and produce something.
  return shaderc_compile_into_spv(compiler, shader.c_str(), shader.size(), kind,
                                  input_file_name, entry_point_name, options);
}

// RAII class for shaderc_compilation_result. Used for shader compilation.
class Compilation {
 public:
  // Compiles shader and keeps the result.
  Compilation(const shaderc_compiler_t compiler, const std::string& shader,
              shaderc_shader_kind kind, const char* input_file_name,
              const char* entry_point_name,
              const shaderc_compile_options_t options = nullptr,
              OutputType output_type = OutputType::SpirvBinary)
      : compiled_result_(
            MakeCompilationResult(compiler, shader, kind, input_file_name,
                                  entry_point_name, options, output_type)) {}

  ~Compilation() { shaderc_result_release(compiled_result_); }

  shaderc_compilation_result_t result() const { return compiled_result_; }

 private:
  shaderc_compilation_result_t compiled_result_;
};

// RAII class for shaderc_compilation_result. Used for shader assembling.
class Assembling {
 public:
  // Assembles shader and keeps the result.
  Assembling(const shaderc_compiler_t compiler, const std::string& assembly,
             const shaderc_compile_options_t options = nullptr)
      : compiled_result_(shaderc_assemble_into_spv(compiler, assembly.data(),
                                                   assembly.size(), options)) {}

  ~Assembling() { shaderc_result_release(compiled_result_); }

  shaderc_compilation_result_t result() const { return compiled_result_; }

 private:
  shaderc_compilation_result_t compiled_result_;
};

struct CleanupOptions {
  void operator()(shaderc_compile_options_t options) const {
    shaderc_compile_options_release(options);
  }
};

typedef std::unique_ptr<shaderc_compile_options, CleanupOptions>
    compile_options_ptr;

// RAII class for shaderc_compiler_t
class Compiler {
 public:
  Compiler() { compiler = shaderc_compiler_initialize(); }
  ~Compiler() { shaderc_compiler_release(compiler); }
  shaderc_compiler_t get_compiler_handle() { return compiler; }

 private:
  shaderc_compiler_t compiler;
};

// RAII class for shader_compiler_options_t
class Options {
 public:
  Options() : options_(shaderc_compile_options_initialize()) {}
  ~Options() { shaderc_compile_options_release(options_); }
  shaderc_compile_options_t get() { return options_; }

 private:
  shaderc_compile_options_t options_;
};

// Helper function to check if the compilation result indicates a successful
// compilation.
bool CompilationResultIsSuccess(const shaderc_compilation_result_t result) {
  return shaderc_result_get_compilation_status(result) ==
         shaderc_compilation_status_success;
}

// Returns true if the given result contains a SPIR-V module that contains
// at least the number of bytes of the header and the correct magic number.
bool ResultContainsValidSpv(shaderc_compilation_result_t result) {
  if (!CompilationResultIsSuccess(result)) return false;
  size_t length = shaderc_result_get_length(result);
  if (length < 20) return false;
  const uint32_t* bytes = static_cast<const uint32_t*>(
      static_cast<const void*>(shaderc_result_get_bytes(result)));
  return bytes[0] == spv::MagicNumber;
}

// Compiles a shader and returns true if the result is valid SPIR-V.
bool CompilesToValidSpv(Compiler& compiler, const std::string& shader,
                        shaderc_shader_kind kind,
                        const shaderc_compile_options_t options = nullptr) {
  const Compilation comp(compiler.get_compiler_handle(), shader, kind, "shader",
                         "main", options, OutputType::SpirvBinary);
  return ResultContainsValidSpv(comp.result());
}

// A testing class to test the compilation of a string with or without options.
// This class wraps the initailization of compiler and compiler options and
// groups the result checking methods. Subclass tests can access the compiler
// object and compiler option object to set their properties. Input file names
// are set to "shader".
class CompileStringTest : public testing::Test {
 protected:
  // Compiles a shader and returns true on success, false on failure.
  bool CompilationSuccess(const std::string& shader, shaderc_shader_kind kind,
                          shaderc_compile_options_t options = nullptr,
                          OutputType output_type = OutputType::SpirvBinary) {
    return CompilationResultIsSuccess(
        Compilation(compiler_.get_compiler_handle(), shader, kind, "shader",
                    "main", options, output_type)
            .result());
  }

  // Compiles a shader, expects compilation success, and returns the warning
  // messages.
  const std::string CompilationWarnings(
      const std::string& shader, shaderc_shader_kind kind,
      const shaderc_compile_options_t options = nullptr,
      OutputType output_type = OutputType::SpirvBinary) {
    const Compilation comp(compiler_.get_compiler_handle(), shader, kind,
                           "shader", "main", options, output_type);
    EXPECT_TRUE(CompilationResultIsSuccess(comp.result())) << kind << '\n'
                                                           << shader;
    return shaderc_result_get_error_message(comp.result());
  };

  // Compiles a shader, expects compilation failure, and returns the messages.
  const std::string CompilationErrors(
      const std::string& shader, shaderc_shader_kind kind,
      const shaderc_compile_options_t options = nullptr,
      OutputType output_type = OutputType::SpirvBinary,
      const char* source_name = "shader") {
    const Compilation comp(compiler_.get_compiler_handle(), shader, kind,
                           source_name, "main", options, output_type);
    EXPECT_FALSE(CompilationResultIsSuccess(comp.result())) << kind << '\n'
                                                            << shader;
    EXPECT_EQ(0u, shaderc_result_get_length(comp.result()));
    return shaderc_result_get_error_message(comp.result());
  };

  // Compiles a shader and returns the messages.
  const std::string CompilationMessages(
      const std::string& shader, shaderc_shader_kind kind,
      const shaderc_compile_options_t options = nullptr,
      OutputType output_type = OutputType::SpirvBinary) {
    const Compilation comp(compiler_.get_compiler_handle(), shader, kind,
                           "shader", "main", options, output_type);
    return shaderc_result_get_error_message(comp.result());
  };

  // Compiles a shader, expects compilation success, and returns the output
  // bytes.
  const std::string CompilationOutput(
      const std::string& shader, shaderc_shader_kind kind,
      const shaderc_compile_options_t options = nullptr,
      OutputType output_type = OutputType::SpirvBinary) {
    const Compilation comp(compiler_.get_compiler_handle(), shader, kind,
                           "shader", "main", options, output_type);
    EXPECT_TRUE(CompilationResultIsSuccess(comp.result()))
        << "shader kind: " << kind << "\nerror message: "
        << shaderc_result_get_error_message(comp.result())
        << "\nshader source code: \n"
        << shader;
    // Use string(const char* s, size_t n) constructor instead of
    // string(const char* s) to make sure the string has complete binary data.
    // string(const char* s) assumes a null-terminated C-string, which will cut
    // the binary data when it sees a '\0' byte.
    return std::string(shaderc_result_get_bytes(comp.result()),
                       shaderc_result_get_length(comp.result()));
  };

  Compiler compiler_;
  compile_options_ptr options_;

 public:
  CompileStringTest() : options_(shaderc_compile_options_initialize()) {}
};

// A testing class to test the assembling of a string.
// This class wraps the initailization of compiler and groups the result
// checking methods. Subclass tests can access the compiler object to set their
// properties.
class AssembleStringTest : public testing::Test {
 public:
  AssembleStringTest() : options_(shaderc_compile_options_initialize()) {}
  ~AssembleStringTest() { shaderc_compile_options_release(options_); }

 protected:
  // Assembles the given assembly and returns true on success.
  bool AssemblingSuccess(const std::string& assembly) {
    return CompilationResultIsSuccess(
        Assembling(compiler_.get_compiler_handle(), assembly, options_)
            .result());
  }

  bool AssemblingValid(const std::string& assembly) {
    const auto assembling =
        Assembling(compiler_.get_compiler_handle(), assembly);
    return ResultContainsValidSpv(assembling.result());
  }

  Compiler compiler_;
  shaderc_compile_options_t options_;
};

// Name holders so that we have test cases being grouped with only one real
// compilation class.
using CompileStringWithOptionsTest = CompileStringTest;
using CompileKindsTest = CompileStringTest;

TEST_F(CompileStringTest, EmptyString) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_FALSE(CompilationSuccess("", shaderc_glsl_vertex_shader));
  EXPECT_FALSE(CompilationSuccess("", shaderc_glsl_fragment_shader));
}

TEST_F(AssembleStringTest, EmptyString) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(AssemblingSuccess(""));
}

TEST_F(CompileStringTest, GarbageString) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_FALSE(CompilationSuccess("jfalkds", shaderc_glsl_vertex_shader));
  EXPECT_FALSE(CompilationSuccess("jfalkds", shaderc_glsl_fragment_shader));
}

TEST_F(AssembleStringTest, GarbageString) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  auto assembling = Assembling(compiler_.get_compiler_handle(), "jfalkds");
  EXPECT_FALSE(CompilationResultIsSuccess(assembling.result()));
  EXPECT_EQ(1u, shaderc_result_get_num_errors(assembling.result()));
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(assembling.result()));
}

// TODO(antiagainst): right now there is no assembling difference for all the
// target environments exposed by shaderc. So the following is just testing the
// target environment is accepted.
TEST_F(AssembleStringTest, AcceptTargetEnv) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_set_target_env(options_, shaderc_target_env_opengl,
                                         /* version = */ 0);
  EXPECT_TRUE(AssemblingSuccess("OpCapability Shader"));
}

TEST_F(CompileStringTest, ReallyLongShader) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  std::string minimal_shader = "";
  minimal_shader += "#version 140\n";
  minimal_shader += "void foo(){}";
  minimal_shader.append(1024 * 1024 * 8, ' ');  // 8MB of spaces.
  minimal_shader += "void main(){}";
  EXPECT_TRUE(CompilesToValidSpv(compiler_, minimal_shader,
                                 shaderc_glsl_vertex_shader));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, minimal_shader,
                                 shaderc_glsl_fragment_shader));
}

TEST_F(CompileStringTest, MinimalShader) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_fragment_shader));
}

TEST_F(AssembleStringTest, MinimalShader) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(AssemblingValid(kMinimalShaderAssembly));
}

TEST_F(CompileStringTest, WorksWithCompileOptions) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringTest, GetNumErrors) {
  Compilation comp(compiler_.get_compiler_handle(), kTwoErrorsShader,
                   shaderc_glsl_vertex_shader, "shader", "main");
  // Expects compilation failure and two errors.
  EXPECT_FALSE(CompilationResultIsSuccess(comp.result()));
  EXPECT_EQ(2u, shaderc_result_get_num_errors(comp.result()));
  // Expects the number of warnings to be zero.
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp.result()));
}

TEST_F(CompileStringTest, GetNumWarnings) {
  Compilation comp(compiler_.get_compiler_handle(), kTwoWarningsShader,
                   shaderc_glsl_vertex_shader, "shader", "main");
  // Expects compilation success with two warnings.
  EXPECT_TRUE(CompilationResultIsSuccess(comp.result()));
  EXPECT_EQ(2u, shaderc_result_get_num_warnings(comp.result()));
  // Expects the number of errors to be zero.
  EXPECT_EQ(0u, shaderc_result_get_num_errors(comp.result()));
}

TEST_F(CompileStringTest, ZeroErrorsZeroWarnings) {
  Compilation comp(compiler_.get_compiler_handle(), kMinimalShader,
                   shaderc_glsl_vertex_shader, "shader", "main");
  // Expects compilation success with zero warnings.
  EXPECT_TRUE(CompilationResultIsSuccess(comp.result()));
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp.result()));
  // Expects the number of errors to be zero.
  EXPECT_EQ(0u, shaderc_result_get_num_errors(comp.result()));
}

TEST_F(CompileStringTest, ErrorTypeUnknownShaderStage) {
  // The shader kind/stage can not be determined, the error type field should
  // indicate the error type is shaderc_shader_kind_error.
  Compilation comp(compiler_.get_compiler_handle(), kMinimalShader,
                   shaderc_glsl_infer_from_source, "shader", "main");
  EXPECT_EQ(shaderc_compilation_status_invalid_stage,
            shaderc_result_get_compilation_status(comp.result()));
}
TEST_F(CompileStringTest, ErrorTypeCompilationError) {
  // The shader kind is valid, the result object's error type field should
  // indicate this compilaion fails due to compilation errors.
  Compilation comp(compiler_.get_compiler_handle(), kTwoErrorsShader,
                   shaderc_glsl_vertex_shader, "shader", "main");
  EXPECT_EQ(shaderc_compilation_status_compilation_error,
            shaderc_result_get_compilation_status(comp.result()));
}

TEST_F(CompileStringWithOptionsTest, CloneCompilerOptions) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  compile_options_ptr options_(shaderc_compile_options_initialize());
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalShader,
                                 shaderc_glsl_vertex_shader,
                                 cloned_options.get()));
}

TEST_F(CompileStringWithOptionsTest, MacroCompileOptions) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_add_macro_definition(options_.get(), "E", 1u, "main",
                                               4u);
  const std::string kMinimalExpandedShader = "#version 140\nvoid E(){}";
  const std::string kMinimalDoubleExpandedShader = "#version 140\nF E(){}";
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  // The simplest should still compile with the cloned options.
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader,
                                 cloned_options.get()));
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kMinimalDoubleExpandedShader,
                                  shaderc_glsl_vertex_shader,
                                  cloned_options.get()));

  shaderc_compile_options_add_macro_definition(cloned_options.get(), "F", 1u,
                                               "void", 4u);
  // This should still not work with the original options.
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kMinimalDoubleExpandedShader,
                                  shaderc_glsl_vertex_shader, options_.get()));
  // This should work with the cloned options that have the additional
  // parameter.
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalDoubleExpandedShader,
                                 shaderc_glsl_vertex_shader,
                                 cloned_options.get()));
}

TEST_F(CompileStringWithOptionsTest, MacroCompileOptionsNotNullTerminated) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_add_macro_definition(options_.get(), "EFGH", 1u,
                                               "mainnnnnn", 4u);
  const std::string kMinimalExpandedShader = "#version 140\nvoid E(){}";
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest, ValuelessMacroCompileOptionsZeroLength) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_add_macro_definition(options_.get(), "E", 1u,
                                               "somthing", 0u);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kValuelessPredefinitionShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest, ValuelessMacroCompileOptionsNullPointer) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_add_macro_definition(options_.get(), "E", 1u, nullptr,
                                               100u);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kValuelessPredefinitionShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest, DisassemblyOption) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  // This should work with both the glslang native assembly format and the
  // SPIR-V tools assembly format.
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("Capability Shader"));
  EXPECT_THAT(disassembly_text, HasSubstr("MemoryModel"));
}

TEST_F(CompileStringWithOptionsTest, DisassembleMinimalShader) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileCorrectStd) {
  // Forces the version and profile to 450core, which fixes the missing
  // #version.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 450,
                                                     shaderc_profile_core);
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kCoreVertShaderWithoutVersion,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       ForcedVersionProfileCorrectStdClonedOptions) {
  // Forces the version and profile to 450core, which fixes the missing
  // #version.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 450,
                                                     shaderc_profile_core);
  // This mode should be carried to any clone of the original options object.
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kCoreVertShaderWithoutVersion,
                                 shaderc_glsl_vertex_shader,
                                 cloned_options.get()));
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileInvalidModule) {
  // Forces the version and profile to 310es, while the source module is invalid
  // for this version of GLSL. Compilation should fail.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 310,
                                                     shaderc_profile_es);
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_THAT(CompilationErrors(kCoreVertShaderWithoutVersion,
                                shaderc_glsl_vertex_shader, options_.get()),
              HasSubstr("error: 'gl_ClipDistance' : undeclared identifier\n"));
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileConflictingStd) {
  // Forces the version and profile to 450core, which is in conflict with the
  // #version in shader.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 450,
                                                     shaderc_profile_core);
  const std::string kVertexShader =
      std::string("#version 310 es\n") + kCoreVertShaderWithoutVersion;
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_THAT(CompilationWarnings(kVertexShader, shaderc_glsl_vertex_shader,
                                  options_.get()),
              HasSubstr("warning: (version, profile) forced to be (450, core), "
                        "while in source code it is (310, es)\n"));
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileUnknownVersionStd) {
  // Forces the version and profile to 4242core, which is an unknown version.
  shaderc_compile_options_set_forced_version_profile(
      options_.get(), 4242 /*unknown version*/, shaderc_profile_core);
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  // Warning message should complain about the unknown version.
  //
  // Also, Glslang errors out on unkown versions, due to commit:
  // https://github.com/KhronosGroup/glslang/commit/9353f1afab8d1c2b1811c6acd807675128eaabc5
  const auto errs = CompilationErrors(
      kMinimalShader, shaderc_glsl_vertex_shader, options_.get());
  EXPECT_THAT(
      errs, HasSubstr("warning: (version, profile) forced to be (4242, core), "
                      "while in source code it is (140, none)\n"));
  EXPECT_THAT(errs, HasSubstr("error: version not supported\n"));
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileVersionsBefore150) {
  // Versions before 150 do not allow a profile token, shaderc_profile_none
  // should be passed down as the profile parameter.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 140,
                                                     shaderc_profile_none);
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_TRUE(CompilationSuccess(kMinimalShaderWithoutVersion,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest, ForcedVersionProfileRedundantProfileStd) {
  // Forces the version and profile to 100core. But versions before 150 do not
  // allow a profile token, compilation should fail.
  shaderc_compile_options_set_forced_version_profile(options_.get(), 100,
                                                     shaderc_profile_core);
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_THAT(CompilationErrors(kMinimalShader, shaderc_glsl_vertex_shader,
                                options_.get()),
              HasSubstr("error: #version: versions before 150 do not allow a "
                        "profile token\n"));
}

TEST_F(CompileStringWithOptionsTest, GenerateDebugInfoBinary) {
  shaderc_compile_options_set_generate_debug_info(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string binary_output =
      CompilationOutput(kMinimalDebugInfoShader,
                        shaderc_glsl_vertex_shader, options_.get());
  // The binary output should contain the name of the vector (debug_info_sample)
  // null-terminated, as well as the whole original source.
  std::string vector_name("debug_info_sample");
  vector_name.resize(vector_name.size() + 1);
  EXPECT_THAT(binary_output, HasSubstr(vector_name));
  EXPECT_THAT(binary_output, HasSubstr(kMinimalDebugInfoShader));
}

TEST_F(CompileStringWithOptionsTest, GenerateDebugInfoBinaryClonedOptions) {
  shaderc_compile_options_set_generate_debug_info(options_.get());
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string binary_output =
      CompilationOutput(kMinimalDebugInfoShader,
                        shaderc_glsl_vertex_shader, cloned_options.get());
  // The binary output should contain the name of the vector (debug_info_sample)
  // null-terminated, as well as the whole original source.
  std::string vector_name("debug_info_sample");
  vector_name.resize(vector_name.size() + 1);
  EXPECT_THAT(binary_output, HasSubstr(vector_name));
  EXPECT_THAT(binary_output, HasSubstr(kMinimalDebugInfoShader));
}

TEST_F(CompileStringWithOptionsTest, GenerateDebugInfoDisassembly) {
  shaderc_compile_options_set_generate_debug_info(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  // Generate assembly text we can compare its output as a string.
  // The disassembly output should contain the name of the vector:
  // debug_info_sample.
  EXPECT_THAT(
      CompilationOutput(kMinimalDebugInfoShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText),
      HasSubstr("debug_info_sample"));
}

TEST_F(CompileStringWithOptionsTest, CompileAndOptimizeWithLevelZero) {
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_zero);
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CompileStringWithOptionsTest, CompileAndOptimizeWithLevelPerformance) {
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_performance);
  const std::string disassembly_text =
      CompilationOutput(kGlslMultipleFnShader, shaderc_glsl_fragment_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  // Check that we do not have function calls anymore.
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpFunctionCall")));
}

TEST_F(CompileStringWithOptionsTest, CompileAndOptimizeWithLevelSize) {
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_size);
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we do not have debug instructions.
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpName")));
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpSource")));
}

TEST_F(CompileStringWithOptionsTest, CompileAndOptimizeForVulkan10Failure) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_performance);

  EXPECT_FALSE(CompilesToValidSpv(compiler_, kHlslWaveActiveSumeComputeShader,
                                  shaderc_compute_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest, CompileAndOptimizeForVulkan11Success) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_performance);

  const std::string disassembly_text = CompilationOutput(
      kHlslWaveActiveSumeComputeShader, shaderc_compute_shader, options_.get(),
      OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpGroupNonUniformIAdd"));
}

TEST_F(CompileStringWithOptionsTest, FollowingOptLevelOverridesPreviousOne) {
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_size);
  // Optimization level settings overridden by
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_zero);
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CompileStringWithOptionsTest,
       GenerateDebugInfoOverridesOptimizationLevel) {
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_size);
  // Optimization level settings overridden by
  shaderc_compile_options_set_generate_debug_info(options_.get());
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDebugInfoDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CompileStringWithOptionsTest,
       GenerateDebugInfoProhibitsOptimizationLevel) {
  // Setting generate debug info first also works.
  shaderc_compile_options_set_generate_debug_info(options_.get());
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_size);
  const std::string disassembly_text =
      CompilationOutput(kMinimalShader, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  for (const auto& substring : kMinimalShaderDebugInfoDisassemblySubstrings) {
    EXPECT_THAT(disassembly_text, HasSubstr(substring));
  }
  // Check that we still have debug instructions.
  EXPECT_THAT(disassembly_text, HasSubstr("OpName"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpSource"));
}

TEST_F(CompileStringWithOptionsTest, PreprocessingOnlyOption) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kMinimalShaderWithMacro =
      "#version 150\n"
      "#define E main\n"
      "void E(){}\n";
  const std::string preprocessed_text =
      CompilationOutput(kMinimalShaderWithMacro, shaderc_glsl_vertex_shader,
                        options_.get(), OutputType::PreprocessedText);
  EXPECT_THAT(preprocessed_text, HasSubstr("void main(){ }"));

  const std::string kMinimalShaderWithMacroCloneOption =
      "#version 150\n"
      "#define E_CLONE_OPTION main\n"
      "void E_CLONE_OPTION(){}\n";
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  const std::string preprocessed_text_cloned_options = CompilationOutput(
      kMinimalShaderWithMacroCloneOption, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::PreprocessedText);
  EXPECT_THAT(preprocessed_text_cloned_options, HasSubstr("void main(){ }"));
}

// A shader kind test cases needs: 1) A shader text with or without #pragma
// annotation, 2) shader_kind.
struct ShaderKindTestCase {
  const char* shader_;
  shaderc_shader_kind shader_kind_;
};

// Test the shader kind deduction process. If the shader kind is one of the
// forced ones, the compiler will just try to compile the source code in that
// specified shader kind. If the shader kind is shaderc_glsl_deduce_from_pragma,
// the compiler will determine the shader kind from #pragma annotation in the
// source code and emit error if none such annotation is found. When the shader
// kind is one of the default ones, the compiler will fall back to use the
// specified shader kind if and only if #pragma annoation is not found.

// Valid shader kind settings should generate valid SPIR-V code.
using ValidShaderKind = testing::TestWithParam<ShaderKindTestCase>;

TEST_P(ValidShaderKind, ValidSpvCode) {
  const ShaderKindTestCase& test_case = GetParam();
  Compiler compiler;
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
        {kNVMeshShader, shaderc_glsl_default_mesh_shader},
        {kNVTaskShader, shaderc_glsl_default_task_shader},

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
        {kNVMeshShaderWithPragma, shaderc_glsl_default_geometry_shader},
        {kNVTaskShaderWithPragma, shaderc_glsl_default_geometry_shader},

        // Infer from source
        {kVertexOnlyShaderWithPragma, shaderc_glsl_infer_from_source},
        {kNVMeshShaderWithPragma, shaderc_glsl_infer_from_source},
        {kNVTaskShaderWithPragma, shaderc_glsl_infer_from_source},

        // Specified non-default shader kind overrides #pragma annotation.
        {kVertexOnlyShaderWithInvalidPragma, shaderc_glsl_vertex_shader},
    }));

using InvalidShaderKind = testing::TestWithParam<ShaderKindTestCase>;

// Invalid shader kind settings should generate errors.
TEST_P(InvalidShaderKind, CompilationShouldFail) {
  const ShaderKindTestCase& test_case = GetParam();
  Compiler compiler;
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

// A mock class that simulate an includer. C API needs two function pointers
// each for get including data and release the data. This class defined two
// static functions, which wrap their matching member functions, to be passed to
// libshaderc C API.
class TestIncluder {
 public:
  explicit TestIncluder(const FakeFS& fake_fs)
      : fake_fs_(fake_fs), responses_({}) {}

  // Get path and content from the fake file system.
  shaderc_include_result* GetInclude(const char* filename) {
    responses_.emplace_back(shaderc_include_result{
        filename, strlen(filename), fake_fs_.at(std::string(filename)).c_str(),
        fake_fs_.at(std::string(filename)).size()});
    return &responses_.back();
  }

  // Response data is owned as private property, no need to release explicitly.
  void ReleaseInclude(shaderc_include_result*) {}

  // Wrapper for the corresponding member function.
  static shaderc_include_result* GetIncluderResponseWrapper(
      void* user_data, const char* filename, int, const char* includer,
      size_t include_depth) {
    return static_cast<TestIncluder*>(user_data)->GetInclude(filename);
  }

  // Wrapper for the corresponding member function.
  static void ReleaseIncluderResponseWrapper(void* user_data,
                                             shaderc_include_result* data) {
    return static_cast<TestIncluder*>(user_data)->ReleaseInclude(data);
  }

 private:
  // Includer response data is stored as private property.
  const FakeFS& fake_fs_;
  std::vector<shaderc_include_result> responses_;
};

using IncluderTests = testing::TestWithParam<IncluderTestCase>;

// Parameterized tests for includer.
TEST_P(IncluderTests, SetIncluderCallbacks) {
  const IncluderTestCase& test_case = GetParam();
  const FakeFS& fs = test_case.fake_fs();
  const std::string& shader = fs.at("root");
  TestIncluder includer(fs);
  Compiler compiler;
  compile_options_ptr options(shaderc_compile_options_initialize());
  shaderc_compile_options_set_include_callbacks(
      options.get(), TestIncluder::GetIncluderResponseWrapper,
      TestIncluder::ReleaseIncluderResponseWrapper, &includer);

  const Compilation comp(compiler.get_compiler_handle(), shader,
                         shaderc_glsl_vertex_shader, "shader", "main",
                         options.get(), OutputType::PreprocessedText);
  // Checks the existence of the expected string.
  EXPECT_THAT(shaderc_result_get_bytes(comp.result()),
              HasSubstr(test_case.expected_substring()));
}

TEST_P(IncluderTests, SetIncluderCallbacksClonedOptions) {
  const IncluderTestCase& test_case = GetParam();
  const FakeFS& fs = test_case.fake_fs();
  const std::string& shader = fs.at("root");
  TestIncluder includer(fs);
  Compiler compiler;
  compile_options_ptr options(shaderc_compile_options_initialize());
  shaderc_compile_options_set_include_callbacks(
      options.get(), TestIncluder::GetIncluderResponseWrapper,
      TestIncluder::ReleaseIncluderResponseWrapper, &includer);

  // Cloned options should have all the settings.
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options.get()));

  const Compilation comp(compiler.get_compiler_handle(), shader,
                         shaderc_glsl_vertex_shader, "shader", "main",
                         cloned_options.get(), OutputType::PreprocessedText);
  // Checks the existence of the expected string.
  EXPECT_THAT(shaderc_result_get_bytes(comp.result()),
              HasSubstr(test_case.expected_substring()));
}

INSTANTIATE_TEST_SUITE_P(CompileStringTest, IncluderTests,
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

TEST_F(CompileStringWithOptionsTest, WarningsOnLine) {
  // Some versions of Glslang will return an error, some will return just
  // warnings.
  EXPECT_THAT(
      CompilationMessages(kDeprecatedAttributeShader,
                          shaderc_glsl_vertex_shader, options_.get()),
      HasSubstr(":2: warning: attribute deprecated in version 130; may be "
                "removed in future release\n"));
}

TEST_F(CompileStringWithOptionsTest, WarningsOnLineAsErrors) {
  shaderc_compile_options_set_warnings_as_errors(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_THAT(
      CompilationErrors(kDeprecatedAttributeShader, shaderc_glsl_vertex_shader,
                        options_.get()),
      HasSubstr(":2: error: attribute deprecated in version 130; may be "
                "removed in future release\n"));
}

TEST_F(CompileStringWithOptionsTest, SuppressWarningsOnLine) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_set_suppress_warnings(options_.get());
  EXPECT_THAT(
      CompilationMessages(kDeprecatedAttributeShader,
                          shaderc_glsl_vertex_shader, options_.get()),
      Not(HasSubstr(":2: warning: attribute deprecated in version 130; may be "
                    "removed in future release\n")));
}

TEST_F(CompileStringWithOptionsTest, GlobalWarnings) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_set_forced_version_profile(options_.get(), 400,
                                                     shaderc_profile_core);
  EXPECT_THAT(CompilationWarnings(kMinimalUnknownVersionShader,
                                  shaderc_glsl_vertex_shader, options_.get()),
              HasSubstr("(version, profile) forced to be (400, core),"
                        " while in source code it is (550, none)\n"));
}

TEST_F(CompileStringWithOptionsTest, GlobalWarningsAsErrors) {
  shaderc_compile_options_set_warnings_as_errors(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_set_forced_version_profile(options_.get(), 400,
                                                     shaderc_profile_core);
  EXPECT_THAT(CompilationErrors(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, options_.get()),
              HasSubstr("(version, profile) forced to be (400, core),"
                        " while in source code it is (550, none)\n"));
}

TEST_F(CompileStringWithOptionsTest, SuppressGlobalWarnings) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_set_suppress_warnings(options_.get());
  shaderc_compile_options_set_forced_version_profile(options_.get(), 400,
                                                     shaderc_profile_core);
  EXPECT_EQ("",
            CompilationWarnings(kMinimalUnknownVersionShader,
                                shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       SuppressWarningsModeFirstOverridesWarningsAsErrorsMode) {
  // Sets suppress-warnings mode first, then sets warnings-as-errors mode.
  // suppress-warnings mode should override warnings-as-errors mode.
  shaderc_compile_options_set_suppress_warnings(options_.get());
  shaderc_compile_options_set_warnings_as_errors(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());

  // Warnings on particular lines should be inhibited.
  Compilation comp_line(compiler_.get_compiler_handle(),
                        kDeprecatedAttributeShader, shaderc_glsl_vertex_shader,
                        "shader", "main", options_.get());
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp_line.result()));

  // Global warnings should be inhibited.
  Compilation comp_global(
      compiler_.get_compiler_handle(), kMinimalUnknownVersionShader,
      shaderc_glsl_vertex_shader, "shader", "main", options_.get());
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp_global.result()));
}

TEST_F(CompileStringWithOptionsTest,
       SuppressWarningsModeSecondOverridesWarningsAsErrorsMode) {
  // Sets suppress-warnings mode first, then sets warnings-as-errors mode.
  // suppress-warnings mode should override warnings-as-errors mode.
  shaderc_compile_options_set_warnings_as_errors(options_.get());
  shaderc_compile_options_set_suppress_warnings(options_.get());
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());

  // Warnings on particular lines should be inhibited.
  Compilation comp_line(compiler_.get_compiler_handle(),
                        kDeprecatedAttributeShader, shaderc_glsl_vertex_shader,
                        "shader", "main", options_.get());
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp_line.result()));

  // Global warnings should be inhibited.
  Compilation comp_global(
      compiler_.get_compiler_handle(), kMinimalUnknownVersionShader,
      shaderc_glsl_vertex_shader, "shader", "main", options_.get());
  EXPECT_EQ(0u, shaderc_result_get_num_warnings(comp_global.result()));
}

TEST_F(CompileStringWithOptionsTest, IfDefCompileOption) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  shaderc_compile_options_add_macro_definition(options_.get(), "E", 1u, nullptr,
                                               0u);
  const std::string kMinimalExpandedShader =
      "#version 140\n"
      "#ifdef E\n"
      "void main(){}\n"
      "#else\n"
      "#error\n"
      "#endif";
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kMinimalExpandedShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(
    CompileStringWithOptionsTest,
    TargetEnvRespectedWhenCompilingOpenGLCompatibilityShaderToBinaryAndAlwaysFails) {
  // Glslang does not support generating SPIR-V for compatibility profile
  // shaders.

  EXPECT_FALSE(CompilesToValidSpv(compiler_, kOpenGLCompatibilityFragmentShader,
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl_compat, 0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kOpenGLCompatibilityFragmentShader,
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl, 0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kOpenGLCompatibilityFragmentShader,
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan, 0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kOpenGLCompatibilityFragmentShader,
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingOpenGLCoreShaderToBinary) {
  // Confirm that kOpenGLVertexShader compiles when targeting OpenGL
  // compatibility or core profiles.

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl_compat, 0);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kOpenGLVertexShader,
                                 shaderc_glsl_vertex_shader, options_.get()));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl, 0);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kOpenGLVertexShader,
                                 shaderc_glsl_vertex_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0ShaderToVulkan1_0Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderComputeBarrier,
                                 shaderc_glsl_compute_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0ShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderComputeBarrier,
                                 shaderc_glsl_compute_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1ShaderToVulkan1_0Fails) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kGlslShaderComputeSubgroupBarrier,
                                  shaderc_glsl_compute_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1ShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderComputeSubgroupBarrier,
                                 shaderc_glsl_compute_shader, options_.get()));
}

#ifdef NV_EXTENSIONS
// task shader
TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0TaskShaderToVulkan1_0Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderTaskBarrier,
                                 shaderc_glsl_task_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0TaskShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderTaskBarrier,
                                 shaderc_glsl_task_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1TaskShaderToVulkan1_0Fails) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kGlslShaderTaskSubgroupBarrier,
                                  shaderc_glsl_task_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1TaskShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderTaskSubgroupBarrier,
                                 shaderc_glsl_task_shader, options_.get()));
}

// mesh shader
TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0MeshShaderToVulkan1_0Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderMeshBarrier,
                                 shaderc_glsl_mesh_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_0MeshShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderMeshBarrier,
                                 shaderc_glsl_mesh_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1MeshShaderToVulkan1_0Fails) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  EXPECT_FALSE(CompilesToValidSpv(compiler_, kGlslShaderMeshSubgroupBarrier,
                                  shaderc_glsl_mesh_shader, options_.get()));
}

TEST_F(CompileStringWithOptionsTest,
       TargetEnvRespectedWhenCompilingVulkan1_1MeshShaderToVulkan1_1Succeeds) {
  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_1);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kGlslShaderMeshSubgroupBarrier,
                                 shaderc_glsl_mesh_shader, options_.get()));
}
#endif

TEST_F(CompileStringWithOptionsTest,
       DISABLED_TargetEnvIgnoredWhenPreprocessing) {
  // This test is disabled since some versions of glslang may refuse to compile
  // very old shaders to SPIR-V with OpenGL target. Re-enable and rewrite this
  // test once we have a differential set of environments to test.
  const auto output_type = OutputType::PreprocessedText;

  EXPECT_TRUE(CompilationSuccess(kOpenGLCompatibilityFragmentShader,
                                 shaderc_glsl_fragment_shader, options_.get(),
                                 output_type));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl_compat, 0);
  EXPECT_TRUE(CompilationSuccess(kOpenGLCompatibilityFragmentShader,
                                 shaderc_glsl_fragment_shader, options_.get(),
                                 output_type));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_opengl, 0);
  EXPECT_TRUE(CompilationSuccess(kOpenGLCompatibilityFragmentShader,
                                 shaderc_glsl_fragment_shader, options_.get(),
                                 output_type));

  shaderc_compile_options_set_target_env(options_.get(),
                                         shaderc_target_env_vulkan, 0);
  EXPECT_TRUE(CompilationSuccess(kOpenGLCompatibilityFragmentShader,
                                 shaderc_glsl_fragment_shader, options_.get(),
                                 output_type));
}

TEST_F(CompileStringTest, ShaderKindRespected) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kVertexShader =
      "#version 140\nvoid main(){ gl_Position = vec4(0);}";
  EXPECT_TRUE(CompilationSuccess(kVertexShader, shaderc_glsl_vertex_shader));
  EXPECT_FALSE(CompilationSuccess(kVertexShader, shaderc_glsl_fragment_shader));
}

TEST_F(CompileStringTest, ErrorsReported) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  EXPECT_THAT(CompilationErrors("int f(){return wrongname;}",
                                shaderc_glsl_vertex_shader),
              HasSubstr("wrongname"));
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST_F(CompileStringTest, MultipleThreadsCalling) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  bool results[10];
  std::vector<std::thread> threads;
  for (auto& r : results) {
    threads.emplace_back([&r, this]() {
      r = CompilationSuccess("#version 140\nvoid main(){}",
                             shaderc_glsl_vertex_shader);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_THAT(results, Each(true));
}
#endif

TEST_F(CompileKindsTest, Vertex) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kVertexShader =
      "#version 140\nvoid main(){ gl_Position = vec4(0);}";
  EXPECT_TRUE(CompilationSuccess(kVertexShader, shaderc_glsl_vertex_shader));
}

TEST_F(CompileKindsTest, Fragment) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kFragShader =
      "#version 140\nvoid main(){ gl_FragColor = vec4(0);}";
  EXPECT_TRUE(CompilationSuccess(kFragShader, shaderc_glsl_fragment_shader));
}

TEST_F(CompileKindsTest, Compute) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kCompShader =
      R"(#version 310 es
       void main() {}
  )";
  EXPECT_TRUE(CompilationSuccess(kCompShader, shaderc_glsl_compute_shader));
}

TEST_F(CompileKindsTest, Geometry) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kGeoShader =

      R"(#version 310 es
       #extension GL_OES_geometry_shader : enable
       layout(points) in;
       layout(points, max_vertices=1) out;
       void main() {
         gl_Position = vec4(1.0);
         EmitVertex();
         EndPrimitive();
       }
  )";
  EXPECT_TRUE(CompilationSuccess(kGeoShader, shaderc_glsl_geometry_shader));
}

TEST_F(CompileKindsTest, TessControl) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kTessControlShader =
      R"(#version 310 es
       #extension GL_OES_tessellation_shader : enable
       layout(vertices=1) out;
       void main() {}
  )";
  EXPECT_TRUE(
      CompilationSuccess(kTessControlShader, shaderc_glsl_tess_control_shader));
}

TEST_F(CompileKindsTest, TessEvaluation) {
  ASSERT_NE(nullptr, compiler_.get_compiler_handle());
  const std::string kTessEvaluationShader =
      R"(#version 310 es
       #extension GL_OES_tessellation_shader : enable
       layout(triangles, equal_spacing, ccw) in;
       void main() {
         gl_Position = vec4(gl_TessCoord, 1.0);
       }
  )";
  EXPECT_TRUE(CompilationSuccess(kTessEvaluationShader,
                                 shaderc_glsl_tess_evaluation_shader));
}

// A test case for ParseVersionProfileTest needs: 1) the input string, 2)
// expected parsing results, including 'succeed' flag, version value, and
// profile enum.
struct ParseVersionProfileTestCase {
  ParseVersionProfileTestCase(
      const std::string& input_string, bool expected_succeed,
      int expected_version = 0,
      shaderc_profile expected_profile = shaderc_profile_none)
      : input_string_(input_string),
        expected_succeed_(expected_succeed),
        expected_version_(expected_version),
        expected_profile_(expected_profile) {}
  std::string input_string_;
  bool expected_succeed_;
  int expected_version_;
  shaderc_profile expected_profile_;
};

// Test for a helper function to parse version and profile from string.
using ParseVersionProfileTest =
    testing::TestWithParam<ParseVersionProfileTestCase>;

TEST_P(ParseVersionProfileTest, FromNullTerminatedString) {
  const ParseVersionProfileTestCase& test_case = GetParam();
  int version;
  shaderc_profile profile;
  bool succeed = shaderc_parse_version_profile(test_case.input_string_.c_str(),
                                               &version, &profile);
  EXPECT_EQ(test_case.expected_succeed_, succeed);
  // check the return version and profile only when the parsing succeeds.
  if (succeed) {
    EXPECT_EQ(test_case.expected_version_, version);
    EXPECT_EQ(test_case.expected_profile_, profile);
  }
}

INSTANTIATE_TEST_SUITE_P(
    HelperMethods, ParseVersionProfileTest,
    testing::ValuesIn(std::vector<ParseVersionProfileTestCase>{
        // Valid version profiles
        ParseVersionProfileTestCase("450core", true, 450, shaderc_profile_core),
        ParseVersionProfileTestCase("450compatibility", true, 450,
                                    shaderc_profile_compatibility),
        ParseVersionProfileTestCase("310es", true, 310, shaderc_profile_es),
        ParseVersionProfileTestCase("100", true, 100, shaderc_profile_none),

        // Invalid version profiles, the expected_version and expected_profile
        // doesn't matter as they won't be checked if the tests pass correctly.
        ParseVersionProfileTestCase("totally_wrong", false),
        ParseVersionProfileTestCase("111core", false),
        ParseVersionProfileTestCase("450wrongprofile", false),
        ParseVersionProfileTestCase("", false),
    }));

TEST_F(CompileStringTest, NullSourceNameFailsCompilingToBinary) {
  EXPECT_THAT(CompilationErrors(kEmpty310ESShader, shaderc_glsl_vertex_shader,
                                nullptr, OutputType::SpirvBinary, nullptr),
              HasSubstr("Input file name string was null."));
}

TEST_F(CompileStringTest, NullSourceNameFailsCompilingToAssemblyText) {
  EXPECT_THAT(
      CompilationErrors(kEmpty310ESShader, shaderc_glsl_vertex_shader, nullptr,
                        OutputType::SpirvAssemblyText, nullptr),
      HasSubstr("Input file name string was null."));
}

TEST_F(CompileStringTest, NullSourceNameFailsCompilingToPreprocessedText) {
  EXPECT_THAT(CompilationErrors(kEmpty310ESShader, shaderc_glsl_vertex_shader,
                                nullptr, OutputType::PreprocessedText, nullptr),
              HasSubstr("Input file name string was null."));
}

const char kGlslVertexShader[] =
    "#version 140\nvoid main(){ gl_Position = vec4(0);}";

const char kHlslVertexShader[] =
    "float4 EntryPoint(uint index : SV_VERTEXID) : SV_POSITION\n"
    "{ return float4(1.0, 2.0, 3.0, 4.0); }";

TEST_F(CompileStringTest, LangGlslOnGlslVertexSucceeds) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_glsl);
  EXPECT_TRUE(CompilationSuccess(kGlslVertexShader, shaderc_glsl_vertex_shader,
                                 options_.get()));
}

TEST_F(CompileStringTest, LangGlslOnHlslVertexFails) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_glsl);
  EXPECT_FALSE(CompilationSuccess(kHlslVertexShader, shaderc_glsl_vertex_shader,
                                  options_.get()));
}

TEST_F(CompileStringTest, LangHlslOnGlslVertexFails) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  EXPECT_FALSE(CompilationSuccess(kGlslVertexShader, shaderc_glsl_vertex_shader,
                                  options_.get()));
}

TEST_F(CompileStringTest, LangHlslOnHlslVertexSucceeds) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  EXPECT_TRUE(CompilationSuccess(kHlslVertexShader, shaderc_glsl_vertex_shader,
                                 options_.get()));
}

TEST(EntryPointTest,
     LangGlslOnHlslVertexSucceedsButAssumesEntryPointNameIsMain) {
  Compiler compiler;
  Options options;
  auto compilation =
      Compilation(compiler.get_compiler_handle(), kGlslVertexShader,
                  shaderc_glsl_vertex_shader, "shader", "blah blah blah",
                  options.get(), OutputType::SpirvAssemblyText);

  EXPECT_THAT(shaderc_result_get_bytes(compilation.result()),
              HasSubstr("OpEntryPoint Vertex %main \"main\""))
      << std::string(shaderc_result_get_bytes(compilation.result()));
}

TEST(EntryPointTest, LangHlslOnHlslVertexSucceedsWithGivenEntryPointName) {
  Compiler compiler;
  Options options;
  shaderc_compile_options_set_source_language(options.get(),
                                              shaderc_source_language_hlsl);
  auto compilation =
      Compilation(compiler.get_compiler_handle(), kHlslVertexShader,
                  shaderc_glsl_vertex_shader, "shader", "EntryPoint",
                  options.get(), OutputType::SpirvAssemblyText);

  EXPECT_THAT(shaderc_result_get_bytes(compilation.result()),
              HasSubstr("OpEntryPoint Vertex %EntryPoint \"EntryPoint\""))
      << std::string(shaderc_result_get_bytes(compilation.result()));
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
TEST_F(CompileStringTest, LimitsTexelOffsetDefault) {
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(-9).c_str(),
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(-8).c_str(),
                                 shaderc_glsl_fragment_shader, options_.get()));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(7).c_str(),
                                 shaderc_glsl_fragment_shader, options_.get()));
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(8).c_str(),
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));
}

TEST_F(CompileStringTest, LimitsTexelOffsetLowerMinimum) {
  shaderc_compile_options_set_limit(
      options_.get(), shaderc_limit_min_program_texel_offset, -99);
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(-100).c_str(),
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(-99).c_str(),
                                 shaderc_glsl_fragment_shader, options_.get()));
}

TEST_F(CompileStringTest, LimitsTexelOffsetHigherMaximum) {
  shaderc_compile_options_set_limit(options_.get(),
                                    shaderc_limit_max_program_texel_offset, 10);
  EXPECT_TRUE(CompilationSuccess(ShaderWithTexOffset(10).c_str(),
                                 shaderc_glsl_fragment_shader, options_.get()));
  EXPECT_FALSE(CompilationSuccess(ShaderWithTexOffset(11).c_str(),
                                  shaderc_glsl_fragment_shader,
                                  options_.get()));
}

TEST_F(CompileStringWithOptionsTest, UniformsWithoutBindingsFailCompilation) {
  const std::string errors =
      CompilationErrors(kShaderWithUniformsWithoutBindings,
                        shaderc_glsl_vertex_shader, options_.get());
  EXPECT_THAT(errors,
              HasSubstr("sampler/texture/image requires layout(binding=X)"));
}

TEST_F(CompileStringWithOptionsTest,
       UniformsWithoutBindingsOptionSetAutoBindingsAssignsBindings) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 4"));
}

TEST_F(CompileStringWithOptionsTest, AutoBindUniformsOptionsSurvivesCloning) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      cloned_options.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 4"));
}

TEST_F(CompileStringWithOptionsTest,
       SetBindingBaseForTextureAdjustsTextureBindingsOnly) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_texture, 44);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 3"));
}

TEST_F(CompileStringWithOptionsTest,
       SetBindingBaseForSamplerAdjustsSamplerBindingsOnly) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_sampler, 44);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 3"));
}

TEST_F(CompileStringWithOptionsTest,
       SetBindingBaseForImageAdjustsImageBindingsOnly) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_image, 44);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 44"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 45"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 2"));
}

TEST_F(CompileStringWithOptionsTest,
       SetBindingBaseForBufferAdjustsBufferBindingsOnly) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_buffer, 44);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 44"));
}

TEST_F(CompileStringWithOptionsTest, SetBindingBaseSurvivesCloning) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_texture, 40);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_sampler, 50);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_image, 60);
  shaderc_compile_options_set_binding_base(options_.get(),
                                           shaderc_uniform_kind_buffer, 70);
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_glsl_vertex_shader,
      cloned_options.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 40"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 50"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 60"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 61"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 70"));
}

TEST(Compiler, IncludeWithoutOptionsReturnsValidError) {
  auto compiler = shaderc_compiler_initialize();
  const char source[] = "#version 450\n#include \"no where\"";
  auto result = shaderc_compile_into_spv(compiler, source, strlen(source),
                                         shaderc_glsl_vertex_shader, "file",
                                         "main", nullptr);
  EXPECT_EQ(shaderc_compilation_status_compilation_error,
            shaderc_result_get_compilation_status(result));
  EXPECT_THAT(shaderc_result_get_error_message(result),
              HasSubstr("error: '#include' : #error unexpected include "
                        "directive for header name: no where"));

  shaderc_result_release(result);
  shaderc_compiler_release(compiler);
}

TEST_F(
    CompileStringWithOptionsTest,
    SetBindingBaseForTextureForVertexAdjustsTextureBindingsOnlyCompilingAsVertex) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  shaderc_compile_options_set_binding_base_for_stage(
      options_.get(), shaderc_vertex_shader, shaderc_uniform_kind_texture, 100);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_vertex_shader, options_.get(),
      OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 100"))
      << disassembly_text;
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 3"));
}

TEST_F(CompileStringWithOptionsTest,
       SetBindingBaseForTextureForVertexIgnoredWhenCompilingAsFragment) {
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  // This is ignored since we're compiling as a different stage.
  shaderc_compile_options_set_binding_base_for_stage(
      options_.get(), shaderc_vertex_shader, shaderc_uniform_kind_texture, 100);
  const std::string disassembly_text = CompilationOutput(
      kShaderWithUniformsWithoutBindings, shaderc_fragment_shader,
      options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_tex Binding 0"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_sam Binding 1"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_img Binding 2"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_imbuf Binding 3"));
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorate %my_ubo Binding 4"));
}

TEST_F(CompileStringWithOptionsTest, GlslDefaultPackingUsed) {
  const std::string disassembly_text =
      CompilationOutput(kGlslShaderWeirdPacking, shaderc_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 16"));
}

TEST_F(CompileStringWithOptionsTest, HlslOffsetsOptionDisableRespected) {
  shaderc_compile_options_set_hlsl_offsets(options_.get(), false);
  const std::string disassembly_text =
      CompilationOutput(kGlslShaderWeirdPacking, shaderc_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 16"));
}

TEST_F(CompileStringWithOptionsTest, HlslOffsetsOptionEnableRespected) {
  shaderc_compile_options_set_hlsl_offsets(options_.get(), true);
  const std::string disassembly_text =
      CompilationOutput(kGlslShaderWeirdPacking, shaderc_vertex_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpMemberDecorate %B 1 Offset 4"));
}

TEST_F(CompileStringWithOptionsTest, HlslFunctionality1OffByDefault) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  // The counter should automatically get a binding.
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  const std::string disassembly_text =
      CompilationOutput(kHlslShaderWithCounterBuffer, shaderc_fragment_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, Not(HasSubstr("OpDecorateStringGOOGLE")))
      << disassembly_text;
}

TEST_F(CompileStringWithOptionsTest, HlslFunctionality1Respected) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  shaderc_compile_options_set_hlsl_functionality1(options_.get(), true);
  // The counter should automatically get a binding.
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  const std::string disassembly_text =
      CompilationOutput(kHlslShaderWithCounterBuffer, shaderc_fragment_shader,
                        options_.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorateStringGOOGLE"));
}

TEST_F(CompileStringWithOptionsTest, HlslFunctionality1SurvivesCloning) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  shaderc_compile_options_set_hlsl_functionality1(options_.get(), true);
  // The counter should automatically get a binding.
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  compile_options_ptr cloned_options(
      shaderc_compile_options_clone(options_.get()));
  const std::string disassembly_text =
      CompilationOutput(kHlslShaderWithCounterBuffer, shaderc_fragment_shader,
                        cloned_options.get(), OutputType::SpirvAssemblyText);
  EXPECT_THAT(disassembly_text, HasSubstr("OpDecorateStringGOOGLE"));
}

TEST_F(CompileStringWithOptionsTest, HlslFlexibleMemoryLayoutAllowed) {
  shaderc_compile_options_set_source_language(options_.get(),
                                              shaderc_source_language_hlsl);
  shaderc_compile_options_set_optimization_level(
      options_.get(), shaderc_optimization_level_performance);
  // There is no way to set the counter's binding, so set it automatically.
  // See https://github.com/KhronosGroup/glslang/issues/1616
  shaderc_compile_options_set_auto_bind_uniforms(options_.get(), true);
  EXPECT_TRUE(CompilesToValidSpv(compiler_, kHlslMemLayoutResourceSelect,
                                 shaderc_fragment_shader, options_.get()));
}

}  // anonymous namespace
