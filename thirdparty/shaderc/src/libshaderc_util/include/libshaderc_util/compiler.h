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

#ifndef LIBSHADERC_UTIL_INC_COMPILER_H
#define LIBSHADERC_UTIL_INC_COMPILER_H

#include <array>
#include <cassert>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "glslang/Public/ShaderLang.h"

#include "counting_includer.h"
#include "file_finder.h"
#include "mutex.h"
#include "resources.h"
#include "string_piece.h"

// Fix a typo in glslang/Public/ShaderLang.h
#define EShTargetClientVersion EshTargetClientVersion

namespace shaderc_util {

// To break recursive including. This header is already included in
// spirv_tools_wrapper.h, so cannot include spirv_tools_wrapper.h here.
enum class PassId;

// Initializes glslang on creation, and destroys it on completion.
// This object is expected to be a singleton, so that internal
// glslang state can be correctly handled.
// TODO(awoloszyn): Once glslang no longer has static global mutable state
//                  remove this class.
class GlslangInitializer {
 public:
  GlslangInitializer() { glslang::InitializeProcess(); }

  ~GlslangInitializer() { glslang::FinalizeProcess(); }

  // Calls release on GlslangInitializer used to intialize this object
  // when it is destroyed.
  class InitializationToken {
   public:
    ~InitializationToken() {
      if (initializer_) {
        initializer_->Release();
      }
    }

    InitializationToken(InitializationToken&& other)
        : initializer_(other.initializer_) {
      other.initializer_ = nullptr;
    }

    InitializationToken(const InitializationToken&) = delete;

   private:
    InitializationToken(GlslangInitializer* initializer)
        : initializer_(initializer) {}

    friend class GlslangInitializer;
    GlslangInitializer* initializer_;
  };

  // Obtains exclusive access to the glslang state. The state remains
  // exclusive until the Initialization Token has been destroyed.
  InitializationToken Acquire() {
    state_lock_.lock();
    return InitializationToken(this);
  }

 private:
  void Release() { state_lock_.unlock(); }

  friend class InitializationToken;

  mutex state_lock_;
};

// Maps macro names to their definitions.  Stores string_pieces, so the
// underlying strings must outlive it.
using MacroDictionary = std::unordered_map<std::string, std::string>;

// Holds all of the state required to compile source GLSL into SPIR-V.
class Compiler {
 public:
  // Source language
  enum class SourceLanguage {
    GLSL,  // The default
    HLSL,
  };

  // Target environment.
  enum class TargetEnv {
    Vulkan,        // Default to Vulkan 1.0
    OpenGL,        // Default to OpenGL 4.5
    OpenGLCompat,  // Deprecated.
  };

  // Target environment versions.  These numbers match those used by Glslang.
  enum class TargetEnvVersion : uint32_t {
    Default = 0,  // Default for the corresponding target environment
    // For Vulkan, use numbering scheme from vulkan.h
    Vulkan_1_0 = ((1 << 22)),              // Vulkan 1.0
    Vulkan_1_1 = ((1 << 22) | (1 << 12)),  // Vulkan 1.1
    // For OpenGL, use the numbering from #version in shaders.
    OpenGL_4_5 = 450,
  };

  enum class OutputType {
    SpirvBinary,  // A binary module, as defined by the SPIR-V specification.
    SpirvAssemblyText,  // Assembly syntax defined by the SPIRV-Tools project.
    PreprocessedText,   // Preprocessed source code.
  };

  // Supported optimization levels.
  enum class OptimizationLevel {
    Zero,         // No optimization.
    Size,         // Optimization towards reducing code size.
    Performance,  // Optimization towards better performance.
  };

  // Resource limits.  These map to the "max*" fields in
  // glslang::TBuiltInResource.
  enum class Limit {
#define RESOURCE(NAME, FIELD, CNAME) NAME,
#include "resources.inc"
#undef RESOURCE
  };

  // Types of uniform variables.
  enum class UniformKind {
    // Image, and image buffer.
    Image = 0,
    // Pure sampler.
    Sampler = 1,
    // Sampled texture in GLSL.
    // Shader Resource View, for HLSL.  (Read-only image or storage buffer.)
    Texture = 2,
    // Uniform Buffer Object, or UBO, in GLSL.
    // Also a Cbuffer in HLSL.
    Buffer = 3,
    // Shader Storage Buffer Object, or SSBO
    StorageBuffer = 4,
    // Uniform Access View, in HLSL.  (Writable storage image or storage
    // buffer.)
    UnorderedAccessView = 5,
  };
  enum { kNumUniformKinds = int(UniformKind::UnorderedAccessView) + 1 };

  // Shader pipeline stage.
  // TODO(dneto): Replaces interface uses of EShLanguage with this enum.
  enum class Stage {
    Vertex,
    TessEval,
    TessControl,
    Geometry,
    Fragment,
    Compute,
#ifdef NV_EXTENSIONS
    RayGenNV,
    IntersectNV,
    AnyHitNV,
    ClosestHitNV,
    MissNV,
    CallableNV,
    TaskNV,
    MeshNV,
#endif
    StageEnd,
  };
  enum { kNumStages = int(Stage::StageEnd) };

  // Returns a std::array of all the Stage values.
  const std::array<Stage, kNumStages>& stages() const {
    static std::array<Stage, kNumStages> values{{
        Stage::Vertex,
        Stage::TessEval,
        Stage::TessControl,
        Stage::Geometry,
        Stage::Fragment,
        Stage::Compute,
#ifdef NV_EXTENSIONS
        Stage::RayGenNV,
        Stage::IntersectNV,
        Stage::AnyHitNV,
        Stage::ClosestHitNV,
        Stage::MissNV,
        Stage::CallableNV,
        Stage::TaskNV,
        Stage::MeshNV,
#endif
    }};
    return values;
  }

  // Creates an default compiler instance targeting at Vulkan environment. Uses
  // version 110 and no profile specification as the default for GLSL.
  Compiler()
      // The default version for glsl is 110, or 100 if you are using an es
      // profile. But we want to default to a non-es profile.
      : default_version_(110),
        default_profile_(ENoProfile),
        force_version_profile_(false),
        warnings_as_errors_(false),
        suppress_warnings_(false),
        generate_debug_info_(false),
        enabled_opt_passes_(),
        target_env_(TargetEnv::Vulkan),
        target_env_version_(TargetEnvVersion::Default),
        source_language_(SourceLanguage::GLSL),
        limits_(kDefaultTBuiltInResource),
        auto_bind_uniforms_(false),
        auto_binding_base_(),
        auto_map_locations_(false),
        hlsl_iomap_(false),
        hlsl_offsets_(false),
        hlsl_legalization_enabled_(true),
        hlsl_functionality1_enabled_(false),
        hlsl_explicit_bindings_() {}

  // Requests that the compiler place debug information into the object code,
  // such as identifier names and line numbers.
  void SetGenerateDebugInfo();

  // Sets the optimization level to the given level. Only the last one takes
  // effect if multiple calls of this method exist.
  void SetOptimizationLevel(OptimizationLevel level);

  // Enables or disables HLSL legalization passes.
  void EnableHlslLegalization(bool hlsl_legalization_enabled);

  // Enables or disables extension SPV_GOOGLE_hlsl_functionality1
  void EnableHlslFunctionality1(bool enable);

  // When a warning is encountered it treat it as an error.
  void SetWarningsAsErrors();

  // Any warning message generated is suppressed before it is output.
  void SetSuppressWarnings();

  // Adds an implicit macro definition obeyed by subsequent CompileShader()
  // calls. The macro and definition should be passed in with their char*
  // pointer and their lengths. They can be modified or deleted after this
  // function has returned.
  void AddMacroDefinition(const char* macro, size_t macro_length,
                          const char* definition, size_t definition_length);

  // Sets the target environment, including version.  The version value should
  // be 0 or one of the values from TargetEnvVersion.  The default value maps
  // to Vulkan 1.0 if the target environment is Vulkan, and it maps to OpenGL
  // 4.5 if the target environment is OpenGL.
  void SetTargetEnv(TargetEnv env,
                    TargetEnvVersion version = TargetEnvVersion::Default);

  // Sets the souce language.
  void SetSourceLanguage(SourceLanguage lang);

  // Forces (without any verification) the default version and profile for
  // subsequent CompileShader() calls.
  void SetForcedVersionProfile(int version, EProfile profile);

  // Sets a resource limit.
  void SetLimit(Limit limit, int value);

  // Returns the current limit.
  int GetLimit(Limit limit) const;

  // Set whether the compiler automatically assigns bindings to
  // uniform variables that don't have explicit bindings.
  void SetAutoBindUniforms(bool auto_bind) { auto_bind_uniforms_ = auto_bind; }

  // Sets the lowest binding number used when automatically assigning bindings
  // for uniform resources of the given type, for all shader stages.  The default
  // base is zero.
  void SetAutoBindingBase(UniformKind kind, uint32_t base) {
    for (auto stage : stages()) {
      SetAutoBindingBaseForStage(stage, kind, base);
    }
  }

  // Sets the lowest binding number used when automatically assigning bindings
  // for uniform resources of the given type for a specific shader stage.  The
  // default base is zero.
  void SetAutoBindingBaseForStage(Stage stage, UniformKind kind,
                                  uint32_t base) {
    auto_binding_base_[static_cast<int>(stage)][static_cast<int>(kind)] = base;
  }

  // Sets whether the compiler automatically assigns locations to
  // uniform variables that don't have explicit locations.
  void SetAutoMapLocations(bool auto_map) { auto_map_locations_ = auto_map; }

  // Use HLSL IO mapping rules for bindings.  Default is false.
  void SetHlslIoMapping(bool hlsl_iomap) { hlsl_iomap_ = hlsl_iomap; }

  // Use HLSL rules for offsets in "transparent" memory.  These allow for
  // tighter packing of some combinations of types than standard GLSL packings.
  void SetHlslOffsets(bool hlsl_offsets) { hlsl_offsets_ = hlsl_offsets; }

  // Sets an explicit set and binding for the given HLSL register.
  void SetHlslRegisterSetAndBinding(const std::string& reg,
                                    const std::string& set,
                                    const std::string& binding) {
    for (auto stage : stages()) {
      SetHlslRegisterSetAndBindingForStage(stage, reg, set, binding);
    }
  }

  // Sets an explicit set and binding for the given HLSL register in the given
  // shader stage.  For example,
  //    SetHlslRegisterSetAndBinding(Stage::Fragment, "t1", "4", "5")
  // means register "t1" in a fragment shader should map to binding 5 in set 4.
  // (Glslang wants this data as strings, not ints or enums.)  The string data is
  // copied.
  void SetHlslRegisterSetAndBindingForStage(Stage stage, const std::string& reg,
                                            const std::string& set,
                                            const std::string& binding) {
    hlsl_explicit_bindings_[static_cast<int>(stage)].push_back(reg);
    hlsl_explicit_bindings_[static_cast<int>(stage)].push_back(set);
    hlsl_explicit_bindings_[static_cast<int>(stage)].push_back(binding);
  }

  // Compiles the shader source in the input_source_string parameter.
  //
  // If the forced_shader stage parameter is not EShLangCount then
  // the shader is assumed to be of the given stage.
  //
  // For HLSL compilation, entry_point_name is the null-terminated string for
  // the entry point.  For GLSL compilation, entry_point_name is ignored, and
  // compilation assumes the entry point is named "main".
  //
  // The stage_callback function will be called if a shader_stage has
  // not been forced and the stage can not be determined
  // from the shader text. Any #include directives are parsed with the given
  // includer.
  //
  // The initializer parameter must be a valid GlslangInitializer object.
  // Acquire will be called on the initializer and the result will be
  // destroyed before the function ends.
  //
  // The output_type parameter determines what kind of output should be
  // produced.
  //
  // Any error messages are written as if the file name were error_tag.
  // Any errors are written to the error_stream parameter.
  // total_warnings and total_errors are incremented once for every
  // warning or error encountered respectively.
  //
  // Returns a tuple consisting of three fields. 1) a boolean which is true when
  // the compilation succeeded, and false otherwise; 2) a vector of 32-bit words
  // which contains the compilation output data, either compiled SPIR-V binary
  // code, or the text string generated in preprocessing-only or disassembly
  // mode; 3) the size of the output data in bytes. When the output is SPIR-V
  // binary code, the size is the number of bytes of valid data in the vector.
  // If the output is a text string, the size equals the length of that string.
  std::tuple<bool, std::vector<uint32_t>, size_t> Compile(
      const string_piece& input_source_string, EShLanguage forced_shader_stage,
      const std::string& error_tag, const char* entry_point_name,
      const std::function<EShLanguage(std::ostream* error_stream,
                                      const string_piece& error_tag)>&
          stage_callback,
      CountingIncluder& includer, OutputType output_type,
      std::ostream* error_stream, size_t* total_warnings, size_t* total_errors,
      GlslangInitializer* initializer) const;

  static EShMessages GetDefaultRules() {
    return static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules |
                                    EShMsgCascadingErrors);
  }

 protected:
  // Preprocesses a shader whose filename is filename and content is
  // shader_source. If preprocessing is successful, returns true, the
  // preprocessed shader, and any warning message as a tuple. Otherwise,
  // returns false, an empty string, and error messages as a tuple.
  //
  // The error_tag parameter is the name to use for outputting errors.
  // The shader_source parameter is the input shader's source text.
  // The shader_preamble parameter is a context-specific preamble internally
  // prepended to shader_text without affecting the validity of its #version
  // position.
  //
  // Any #include directives are processed with the given includer.
  //
  // If force_version_profile_ is set, the shader's version/profile is forced
  // to be default_version_/default_profile_ regardless of the #version
  // directive in the source code.
  std::tuple<bool, std::string, std::string> PreprocessShader(
      const std::string& error_tag, const string_piece& shader_source,
      const string_piece& shader_preamble, CountingIncluder& includer) const;

  // Cleans up the preamble in a given preprocessed shader.
  //
  // The error_tag parameter is the name to be given for the main file.
  // The pound_extension parameter is the #extension directive we prepended to
  // the original shader source code via preamble.
  // The num_include_directives parameter is the number of #include directives
  // appearing in the original shader source code.
  // The is_for_next_line means whether the #line sets the line number for the
  // next line.
  //
  // If no #include directive is used in the shader source code, we can safely
  // delete the #extension directive we injected via preamble. Otherwise, we
  // need to adjust it if there exists a #version directive in the original
  // shader source code.
  std::string CleanupPreamble(const string_piece& preprocessed_shader,
                              const string_piece& error_tag,
                              const string_piece& pound_extension,
                              int num_include_directives,
                              bool is_for_next_line) const;

  // Determines version and profile from command line, or the source code.
  // Returns the decoded version and profile pair on success. Otherwise,
  // returns (0, ENoProfile).
  std::pair<int, EProfile> DeduceVersionProfile(
      const std::string& preprocessed_shader) const;

  // Determines the shader stage from pragmas embedded in the source text if
  // possible. In the returned pair, the glslang EShLanguage is the shader
  // stage deduced. If no #pragma directives for shader stage exist, it's
  // EShLangCount.  If errors occur, the second element in the pair is the
  // error message.  Otherwise, it's an empty string.
  std::pair<EShLanguage, std::string> GetShaderStageFromSourceCode(
      string_piece filename, const std::string& preprocessed_shader) const;

  // Determines version and profile from command line, or the source code.
  // Returns the decoded version and profile pair on success. Otherwise,
  // returns (0, ENoProfile).
  std::pair<int, EProfile> DeduceVersionProfile(
      const std::string& preprocessed_shader);

  // Gets version and profile specification from the given preprocessedshader.
  // Returns the decoded version and profile pair on success. Otherwise,
  // returns (0, ENoProfile).
  std::pair<int, EProfile> GetVersionProfileFromSourceCode(
      const std::string& preprocessed_shader) const;

  // Version to use when force_version_profile_ is true.
  int default_version_;
  // Profile to use when force_version_profile_ is true.
  EProfile default_profile_;
  // When true, use the default version and profile from eponymous data members.
  bool force_version_profile_;

  // Macro definitions that must be available to reference in the shader source.
  MacroDictionary predefined_macros_;

  // When true, treat warnings as errors.
  bool warnings_as_errors_;
  // Supress warnings when true.
  bool suppress_warnings_;

  // When true, compilation will generate debug info with the binary SPIR-V
  // output.
  bool generate_debug_info_;

  // Optimization passes to be applied.
  std::vector<PassId> enabled_opt_passes_;

  // The target environment to compile with. This controls the glslang
  // EshMessages bitmask, which determines which dialect of GLSL and which
  // SPIR-V codegen semantics are used. This impacts the warning & error
  // messages as well as the set of available builtins, as per the
  // implementation of glslang.
  TargetEnv target_env_;

  // The version number of the target environment.  The numbering scheme is
  // particular to each target environment.  If this is 0, then use a default
  // for that particular target environment. See libshaders/shaderc/shaderc.h
  // for those defaults.
  TargetEnvVersion target_env_version_;

  // The source language.  Defaults to GLSL.
  SourceLanguage source_language_;

  // The resource limits to be used.
  TBuiltInResource limits_;

  // True if the compiler should automatically bind uniforms that don't
  // have explicit bindings.
  bool auto_bind_uniforms_;

  // The base binding number per uniform type, per stage, used when automatically
  // binding uniforms that don't hzve explicit bindings in the shader source.
  // The default is zero.
  uint32_t auto_binding_base_[kNumStages][kNumUniformKinds];

  // True if the compiler should automatically map uniforms that don't
  // have explicit locations.
  bool auto_map_locations_;

  // True if the compiler should use HLSL IO mapping rules when compiling HLSL.
  bool hlsl_iomap_;

  // True if the compiler should determine block member offsets using HLSL
  // packing rules instead of standard GLSL rules.
  bool hlsl_offsets_;

  // True if the compiler should perform legalization optimization passes if
  // source language is HLSL.
  bool hlsl_legalization_enabled_;

  // True if the compiler should support extension SPV_GOOGLE_hlsl_functionality1.
  bool hlsl_functionality1_enabled_;

  // A sequence of triples, each triple representing a specific HLSL register
  // name, and the set and binding numbers it should be mapped to, but in
  // the form of strings.  This is how Glslang wants to consume the data.
  std::vector<std::string> hlsl_explicit_bindings_[kNumStages];
};

// Converts a string to a vector of uint32_t by copying the content of a given
// string to the vector and returns it. Appends '\0' at the end if extra bytes
// are required to complete the last element.
std::vector<uint32_t> ConvertStringToVector(const std::string& str);

// Converts a valid Glslang shader stage value to a Compiler::Stage value.
inline Compiler::Stage ConvertToStage(EShLanguage stage) {
  switch (stage) {
    case EShLangVertex:
      return Compiler::Stage::Vertex;
    case EShLangTessControl:
      return Compiler::Stage::TessEval;
    case EShLangTessEvaluation:
      return Compiler::Stage::TessControl;
    case EShLangGeometry:
      return Compiler::Stage::Geometry;
    case EShLangFragment:
      return Compiler::Stage::Fragment;
    case EShLangCompute:
      return Compiler::Stage::Compute;
#ifdef NV_EXTENSIONS
    case EShLangRayGenNV:
      return Compiler::Stage::RayGenNV;
    case EShLangIntersectNV:
      return Compiler::Stage::IntersectNV;
    case EShLangAnyHitNV:
      return Compiler::Stage::AnyHitNV;
    case EShLangClosestHitNV:
      return Compiler::Stage::ClosestHitNV;
    case EShLangMissNV:
      return Compiler::Stage::MissNV;
    case EShLangCallableNV:
      return Compiler::Stage::CallableNV;
    case EShLangTaskNV:
      return Compiler::Stage::TaskNV;
    case EShLangMeshNV:
      return Compiler::Stage::MeshNV;
#endif
    default:
      break;
  }
  assert(false && "Invalid case");
  return Compiler::Stage::Compute;
}

}  // namespace shaderc_util
#endif  // LIBSHADERC_UTIL_INC_COMPILER_H
