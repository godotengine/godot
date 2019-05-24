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

#include "shaderc_private.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <vector>

#include "SPIRV/spirv.hpp"

#include "libshaderc_util/compiler.h"
#include "libshaderc_util/counting_includer.h"
#include "libshaderc_util/resources.h"
#include "libshaderc_util/spirv_tools_wrapper.h"
#include "libshaderc_util/version_profile.h"

#if (defined(_MSC_VER) && !defined(_CPPUNWIND)) || !defined(__EXCEPTIONS)
#define TRY_IF_EXCEPTIONS_ENABLED
#define CATCH_IF_EXCEPTIONS_ENABLED(X) if (0)
#else
#define TRY_IF_EXCEPTIONS_ENABLED try
#define CATCH_IF_EXCEPTIONS_ENABLED(X) catch (X)
#endif

namespace {

// Returns shader stage (ie: vertex, fragment, etc.) in response to forced
// shader kinds. If the shader kind is not a forced kind, returns EshLangCount
// to let #pragma annotation or shader stage deducer determine the stage to
// use.
EShLanguage GetForcedStage(shaderc_shader_kind kind) {
  switch (kind) {
    case shaderc_glsl_vertex_shader:
      return EShLangVertex;
    case shaderc_glsl_fragment_shader:
      return EShLangFragment;
    case shaderc_glsl_compute_shader:
      return EShLangCompute;
    case shaderc_glsl_geometry_shader:
      return EShLangGeometry;
    case shaderc_glsl_tess_control_shader:
      return EShLangTessControl;
    case shaderc_glsl_tess_evaluation_shader:
      return EShLangTessEvaluation;

#ifdef NV_EXTENSIONS
    case shaderc_glsl_raygen_shader:
      return EShLangRayGenNV;
    case shaderc_glsl_anyhit_shader:
      return EShLangAnyHitNV;
    case shaderc_glsl_closesthit_shader:
      return EShLangClosestHitNV;
    case shaderc_glsl_miss_shader:
      return EShLangMissNV;
    case shaderc_glsl_intersection_shader:
      return EShLangIntersectNV;
    case shaderc_glsl_callable_shader:
      return EShLangCallableNV;
    case shaderc_glsl_task_shader:
      return EShLangTaskNV;
    case shaderc_glsl_mesh_shader:
      return EShLangMeshNV;
#endif

    case shaderc_glsl_infer_from_source:
    case shaderc_glsl_default_vertex_shader:
    case shaderc_glsl_default_fragment_shader:
    case shaderc_glsl_default_compute_shader:
    case shaderc_glsl_default_geometry_shader:
    case shaderc_glsl_default_tess_control_shader:
    case shaderc_glsl_default_tess_evaluation_shader:
#ifdef NV_EXTENSIONS
    case shaderc_glsl_default_raygen_shader:
    case shaderc_glsl_default_anyhit_shader:
    case shaderc_glsl_default_closesthit_shader:
    case shaderc_glsl_default_miss_shader:
    case shaderc_glsl_default_intersection_shader:
    case shaderc_glsl_default_callable_shader:
    case shaderc_glsl_default_task_shader:
    case shaderc_glsl_default_mesh_shader:
#endif
    case shaderc_spirv_assembly:
      return EShLangCount;
  }
  assert(0 && "Unhandled shaderc_shader_kind");
  return EShLangCount;
}

// A wrapper functor class to be used as stage deducer for libshaderc_util
// Compile() interface. When the given shader kind is one of the default shader
// kinds, this functor will be called if #pragma is not found in the source
// code. And it returns the corresponding shader stage. When the shader kind is
// a forced shader kind, this functor won't be called and it simply returns
// EShLangCount to make the syntax correct. When the shader kind is set to
// shaderc_glsl_deduce_from_pragma, this functor also returns EShLangCount, but
// the compiler should emit error if #pragma annotation is not found in this
// case.
class StageDeducer {
 public:
  explicit StageDeducer(
      shaderc_shader_kind kind = shaderc_glsl_infer_from_source)
      : kind_(kind), error_(false){};
  // The method that underlying glslang will call to determine the shader stage
  // to be used in current compilation. It is called only when there is neither
  // forced shader kind (or say stage, in the view of glslang), nor #pragma
  // annotation in the source code. This method transforms an user defined
  // 'default' shader kind to the corresponding shader stage. As this is the
  // last trial to determine the shader stage, failing to find the corresponding
  // shader stage will record an error.
  // Note that calling this method more than once during one compilation will
  // have the error recorded for the previous call been overwriten by the next
  // call.
  EShLanguage operator()(std::ostream* /*error_stream*/,
                         const shaderc_util::string_piece& /*error_tag*/) {
    EShLanguage stage = GetDefaultStage(kind_);
    if (stage == EShLangCount) {
      error_ = true;
    } else {
      error_ = false;
    }
    return stage;
  };

  // Returns true if there is error during shader stage deduction.
  bool error() const { return error_; }

 private:
  // Gets the corresponding shader stage for a given 'default' shader kind. All
  // other kinds are mapped to EShLangCount which should not be used.
  EShLanguage GetDefaultStage(shaderc_shader_kind kind) const {
    switch (kind) {
      case shaderc_glsl_vertex_shader:
      case shaderc_glsl_fragment_shader:
      case shaderc_glsl_compute_shader:
      case shaderc_glsl_geometry_shader:
      case shaderc_glsl_tess_control_shader:
      case shaderc_glsl_tess_evaluation_shader:
      case shaderc_glsl_infer_from_source:
#ifdef NV_EXTENSIONS
      case shaderc_glsl_raygen_shader:
      case shaderc_glsl_anyhit_shader:
      case shaderc_glsl_closesthit_shader:
      case shaderc_glsl_miss_shader:
      case shaderc_glsl_intersection_shader:
      case shaderc_glsl_callable_shader:
      case shaderc_glsl_task_shader:
      case shaderc_glsl_mesh_shader:
#endif
        return EShLangCount;
      case shaderc_glsl_default_vertex_shader:
        return EShLangVertex;
      case shaderc_glsl_default_fragment_shader:
        return EShLangFragment;
      case shaderc_glsl_default_compute_shader:
        return EShLangCompute;
      case shaderc_glsl_default_geometry_shader:
        return EShLangGeometry;
      case shaderc_glsl_default_tess_control_shader:
        return EShLangTessControl;
      case shaderc_glsl_default_tess_evaluation_shader:
        return EShLangTessEvaluation;
#ifdef NV_EXTENSIONS
      case shaderc_glsl_default_raygen_shader:
        return EShLangRayGenNV;
      case shaderc_glsl_default_anyhit_shader:
        return EShLangAnyHitNV;
      case shaderc_glsl_default_closesthit_shader:
        return EShLangClosestHitNV;
      case shaderc_glsl_default_miss_shader:
        return EShLangMissNV;
      case shaderc_glsl_default_intersection_shader:
        return EShLangIntersectNV;
      case shaderc_glsl_default_callable_shader:
        return EShLangCallableNV;
      case shaderc_glsl_default_task_shader:
        return EShLangTaskNV;
      case shaderc_glsl_default_mesh_shader:
        return EShLangMeshNV;
#endif
      case shaderc_spirv_assembly:
        return EShLangCount;
    }
    assert(0 && "Unhandled shaderc_shader_kind");
    return EShLangCount;
  }

  shaderc_shader_kind kind_;
  bool error_;
};

// A bridge between the libshaderc includer and libshaderc_util includer.
class InternalFileIncluder : public shaderc_util::CountingIncluder {
 public:
  InternalFileIncluder(const shaderc_include_resolve_fn resolver,
                       const shaderc_include_result_release_fn result_releaser,
                       void* user_data)
      : resolver_(resolver),
        result_releaser_(result_releaser),
        user_data_(user_data){};
  InternalFileIncluder()
      : resolver_(nullptr), result_releaser_(nullptr), user_data_(nullptr){};

 private:
  // Check the validity of the callbacks.
  bool AreValidCallbacks() const {
    return resolver_ != nullptr && result_releaser_ != nullptr;
  }

  // Maps CountingIncluder IncludeType value to a shaderc_include_type
  // value.
  shaderc_include_type GetIncludeType(IncludeType type) {
    switch (type) {
      case IncludeType::Local:
        return shaderc_include_type_relative;
      case IncludeType::System:
        return shaderc_include_type_standard;
      default:
        break;
    }
    assert(0 && "Unhandled IncludeType");
    return shaderc_include_type_relative;
  }

  // Resolves an include request for the requested source of the given
  // type in the context of the specified requesting source.  On success,
  // returns a newly allocated IncludeResponse containing the fully resolved
  // name of the requested source and the contents of that source.
  // On failure, returns a newly allocated IncludeResponse where the
  // resolved name member is an empty string, and the contents members
  // contains error details.
  virtual glslang::TShader::Includer::IncludeResult* include_delegate(
      const char* requested_source, const char* requesting_source,
      IncludeType type, size_t include_depth) override {
    if (!AreValidCallbacks()) {
      static const char kUnexpectedIncludeError[] =
          "#error unexpected include directive";
      return new glslang::TShader::Includer::IncludeResult{
          "", kUnexpectedIncludeError, strlen(kUnexpectedIncludeError),
          nullptr};
    }
    shaderc_include_result* include_result =
        resolver_(user_data_, requested_source, GetIncludeType(type),
                  requesting_source, include_depth);
    // Make a glslang IncludeResult from a shaderc_include_result.  The
    // user_data member of the IncludeResult is a pointer to the
    // shaderc_include_result object, so we can later release the latter.
    return new glslang::TShader::Includer::IncludeResult{
        std::string(include_result->source_name,
                    include_result->source_name_length),
        include_result->content, include_result->content_length,
        include_result};
  }

  // Releases the given IncludeResult.
  virtual void release_delegate(
      glslang::TShader::Includer::IncludeResult* result) override {
    if (result && result_releaser_) {
      result_releaser_(user_data_,
                       static_cast<shaderc_include_result*>(result->userData));
    }
    delete result;
  }

  const shaderc_include_resolve_fn resolver_;
  const shaderc_include_result_release_fn result_releaser_;
  void* user_data_;
};

// Converts the target env to the corresponding one in shaderc_util::Compiler.
shaderc_util::Compiler::TargetEnv GetCompilerTargetEnv(shaderc_target_env env) {
  switch (env) {
    case shaderc_target_env_opengl:
      return shaderc_util::Compiler::TargetEnv::OpenGL;
    case shaderc_target_env_opengl_compat:
      return shaderc_util::Compiler::TargetEnv::OpenGLCompat;
    case shaderc_target_env_vulkan:
    default:
      break;
  }

  return shaderc_util::Compiler::TargetEnv::Vulkan;
}

shaderc_util::Compiler::TargetEnvVersion GetCompilerTargetEnvVersion(
    uint32_t version_number) {
  using namespace shaderc_util;

  if (static_cast<uint32_t>(Compiler::TargetEnvVersion::Vulkan_1_0) ==
      version_number) {
    return Compiler::TargetEnvVersion::Vulkan_1_0;
  }
  if (static_cast<uint32_t>(Compiler::TargetEnvVersion::Vulkan_1_1) ==
      version_number) {
    return Compiler::TargetEnvVersion::Vulkan_1_1;
  }
  if (static_cast<uint32_t>(Compiler::TargetEnvVersion::OpenGL_4_5) ==
      version_number) {
    return Compiler::TargetEnvVersion::OpenGL_4_5;
  }

  return Compiler::TargetEnvVersion::Default;
}

// Returns the Compiler::Limit enum for the given shaderc_limit enum.
shaderc_util::Compiler::Limit CompilerLimit(shaderc_limit limit) {
  switch (limit) {
#define RESOURCE(NAME, FIELD, CNAME) \
  case shaderc_limit_##CNAME:        \
    return shaderc_util::Compiler::Limit::NAME;
#include "libshaderc_util/resources.inc"
#undef RESOURCE
    default:
      break;
  }
  assert(0 && "Should not have reached here");
  return static_cast<shaderc_util::Compiler::Limit>(0);
}

// Returns the Compiler::UniformKind for the given shaderc_uniform_kind.
shaderc_util::Compiler::UniformKind GetUniformKind(shaderc_uniform_kind kind) {
  switch (kind) {
    case shaderc_uniform_kind_texture:
      return shaderc_util::Compiler::UniformKind::Texture;
    case shaderc_uniform_kind_sampler:
      return shaderc_util::Compiler::UniformKind::Sampler;
    case shaderc_uniform_kind_image:
      return shaderc_util::Compiler::UniformKind::Image;
    case shaderc_uniform_kind_buffer:
      return shaderc_util::Compiler::UniformKind::Buffer;
    case shaderc_uniform_kind_storage_buffer:
      return shaderc_util::Compiler::UniformKind::StorageBuffer;
    case shaderc_uniform_kind_unordered_access_view:
      return shaderc_util::Compiler::UniformKind::UnorderedAccessView;
  }
  assert(0 && "Should not have reached here");
  return static_cast<shaderc_util::Compiler::UniformKind>(0);
}

// Returns the Compiler::Stage for generic stage values in shaderc_shader_kind.
shaderc_util::Compiler::Stage GetStage(shaderc_shader_kind kind) {
  switch (kind) {
    case shaderc_vertex_shader:
      return shaderc_util::Compiler::Stage::Vertex;
    case shaderc_fragment_shader:
      return shaderc_util::Compiler::Stage::Fragment;
    case shaderc_compute_shader:
      return shaderc_util::Compiler::Stage::Compute;
    case shaderc_tess_control_shader:
      return shaderc_util::Compiler::Stage::TessControl;
    case shaderc_tess_evaluation_shader:
      return shaderc_util::Compiler::Stage::TessEval;
    case shaderc_geometry_shader:
      return shaderc_util::Compiler::Stage::Geometry;
    default:
      break;
  }
  assert(0 && "Should not have reached here");
  return static_cast<shaderc_util::Compiler::Stage>(0);
}

}  // anonymous namespace

struct shaderc_compile_options {
  shaderc_target_env target_env = shaderc_target_env_default;
  uint32_t target_env_version = 0;
  shaderc_util::Compiler compiler;
  shaderc_include_resolve_fn include_resolver = nullptr;
  shaderc_include_result_release_fn include_result_releaser = nullptr;
  void* include_user_data = nullptr;
};

shaderc_compile_options_t shaderc_compile_options_initialize() {
  return new (std::nothrow) shaderc_compile_options;
}

shaderc_compile_options_t shaderc_compile_options_clone(
    const shaderc_compile_options_t options) {
  if (!options) {
    return shaderc_compile_options_initialize();
  }
  return new (std::nothrow) shaderc_compile_options(*options);
}

void shaderc_compile_options_release(shaderc_compile_options_t options) {
  delete options;
}

void shaderc_compile_options_add_macro_definition(
    shaderc_compile_options_t options, const char* name, size_t name_length,
    const char* value, size_t value_length) {
  options->compiler.AddMacroDefinition(name, name_length, value, value_length);
}

void shaderc_compile_options_set_source_language(
    shaderc_compile_options_t options, shaderc_source_language set_lang) {
  auto lang = shaderc_util::Compiler::SourceLanguage::GLSL;
  if (set_lang == shaderc_source_language_hlsl)
    lang = shaderc_util::Compiler::SourceLanguage::HLSL;
  options->compiler.SetSourceLanguage(lang);
}

void shaderc_compile_options_set_generate_debug_info(
    shaderc_compile_options_t options) {
  options->compiler.SetGenerateDebugInfo();
}

void shaderc_compile_options_set_optimization_level(
    shaderc_compile_options_t options, shaderc_optimization_level level) {
  auto opt_level = shaderc_util::Compiler::OptimizationLevel::Zero;
  switch (level) {
    case shaderc_optimization_level_size:
      opt_level = shaderc_util::Compiler::OptimizationLevel::Size;
      break;
    case shaderc_optimization_level_performance:
      opt_level = shaderc_util::Compiler::OptimizationLevel::Performance;
      break;
    default:
      break;
  }

  options->compiler.SetOptimizationLevel(opt_level);
}

void shaderc_compile_options_set_forced_version_profile(
    shaderc_compile_options_t options, int version, shaderc_profile profile) {
  // Transfer the profile parameter from public enum type to glslang internal
  // enum type. No default case here so that compiler will complain if new enum
  // member is added later but not handled here.
  switch (profile) {
    case shaderc_profile_none:
      options->compiler.SetForcedVersionProfile(version, ENoProfile);
      break;
    case shaderc_profile_core:
      options->compiler.SetForcedVersionProfile(version, ECoreProfile);
      break;
    case shaderc_profile_compatibility:
      options->compiler.SetForcedVersionProfile(version, ECompatibilityProfile);
      break;
    case shaderc_profile_es:
      options->compiler.SetForcedVersionProfile(version, EEsProfile);
      break;
  }
}

void shaderc_compile_options_set_include_callbacks(
    shaderc_compile_options_t options, shaderc_include_resolve_fn resolver,
    shaderc_include_result_release_fn result_releaser, void* user_data) {
  options->include_resolver = resolver;
  options->include_result_releaser = result_releaser;
  options->include_user_data = user_data;
}

void shaderc_compile_options_set_suppress_warnings(
    shaderc_compile_options_t options) {
  options->compiler.SetSuppressWarnings();
}

void shaderc_compile_options_set_target_env(shaderc_compile_options_t options,
                                            shaderc_target_env target,
                                            uint32_t version) {
  options->target_env = target;
  options->compiler.SetTargetEnv(GetCompilerTargetEnv(target),
                                 GetCompilerTargetEnvVersion(version));
}

void shaderc_compile_options_set_warnings_as_errors(
    shaderc_compile_options_t options) {
  options->compiler.SetWarningsAsErrors();
}

void shaderc_compile_options_set_limit(shaderc_compile_options_t options,
                                       shaderc_limit limit, int value) {
  options->compiler.SetLimit(CompilerLimit(limit), value);
}

void shaderc_compile_options_set_auto_bind_uniforms(
    shaderc_compile_options_t options, bool auto_bind) {
  options->compiler.SetAutoBindUniforms(auto_bind);
}

void shaderc_compile_options_set_hlsl_io_mapping(
    shaderc_compile_options_t options, bool hlsl_iomap) {
  options->compiler.SetHlslIoMapping(hlsl_iomap);
}

void shaderc_compile_options_set_hlsl_offsets(shaderc_compile_options_t options,
                                              bool hlsl_offsets) {
  options->compiler.SetHlslOffsets(hlsl_offsets);
}

void shaderc_compile_options_set_binding_base(shaderc_compile_options_t options,
                                              shaderc_uniform_kind kind,
                                              uint32_t base) {
  options->compiler.SetAutoBindingBase(GetUniformKind(kind), base);
}

void shaderc_compile_options_set_binding_base_for_stage(
    shaderc_compile_options_t options, shaderc_shader_kind shader_kind,
    shaderc_uniform_kind kind, uint32_t base) {
  options->compiler.SetAutoBindingBaseForStage(GetStage(shader_kind),
                                               GetUniformKind(kind), base);
}

void shaderc_compile_options_set_auto_map_locations(
    shaderc_compile_options_t options, bool auto_map) {
  options->compiler.SetAutoMapLocations(auto_map);
}

void shaderc_compile_options_set_hlsl_register_set_and_binding_for_stage(
    shaderc_compile_options_t options, shaderc_shader_kind shader_kind,
    const char* reg, const char* set, const char* binding) {
  options->compiler.SetHlslRegisterSetAndBindingForStage(GetStage(shader_kind),
                                                         reg, set, binding);
}

void shaderc_compile_options_set_hlsl_register_set_and_binding(
    shaderc_compile_options_t options, const char* reg, const char* set,
    const char* binding) {
  options->compiler.SetHlslRegisterSetAndBinding(reg, set, binding);
}

void shaderc_compile_options_set_hlsl_functionality1(
    shaderc_compile_options_t options, bool enable) {
  options->compiler.EnableHlslFunctionality1(enable);
}

shaderc_compiler_t shaderc_compiler_initialize() {
  static shaderc_util::GlslangInitializer* initializer =
      new shaderc_util::GlslangInitializer;
  shaderc_compiler_t compiler = new (std::nothrow) shaderc_compiler;
  compiler->initializer = initializer;
  return compiler;
}

void shaderc_compiler_release(shaderc_compiler_t compiler) { delete compiler; }

namespace {
shaderc_compilation_result_t CompileToSpecifiedOutputType(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options,
    shaderc_util::Compiler::OutputType output_type) {
  auto* result = new (std::nothrow) shaderc_compilation_result_vector;
  if (!result) return nullptr;

  if (!input_file_name) {
    result->messages = "Input file name string was null.";
    result->num_errors = 1;
    result->compilation_status = shaderc_compilation_status_compilation_error;
    return result;
  }
  result->compilation_status = shaderc_compilation_status_invalid_stage;
  bool compilation_succeeded = false;  // In case we exit early.
  std::vector<uint32_t> compilation_output_data;
  size_t compilation_output_data_size_in_bytes = 0u;
  if (!compiler->initializer) return result;
  TRY_IF_EXCEPTIONS_ENABLED {
    std::stringstream errors;
    size_t total_warnings = 0;
    size_t total_errors = 0;
    std::string input_file_name_str(input_file_name);
    EShLanguage forced_stage = GetForcedStage(shader_kind);
    shaderc_util::string_piece source_string =
        shaderc_util::string_piece(source_text, source_text + source_text_size);
    StageDeducer stage_deducer(shader_kind);
    if (additional_options) {
      InternalFileIncluder includer(additional_options->include_resolver,
                                    additional_options->include_result_releaser,
                                    additional_options->include_user_data);
      // Depends on return value optimization to avoid extra copy.
      std::tie(compilation_succeeded, compilation_output_data,
               compilation_output_data_size_in_bytes) =
          additional_options->compiler.Compile(
              source_string, forced_stage, input_file_name_str, entry_point_name,
              // stage_deducer has a flag: error_, which we need to check later.
              // We need to make this a reference wrapper, so that std::function
              // won't make a copy for this callable object.
              std::ref(stage_deducer), includer, output_type, &errors,
              &total_warnings, &total_errors, compiler->initializer);
    } else {
      // Compile with default options.
      InternalFileIncluder includer;
      std::tie(compilation_succeeded, compilation_output_data,
               compilation_output_data_size_in_bytes) =
          shaderc_util::Compiler().Compile(
              source_string, forced_stage, input_file_name_str, entry_point_name,
              std::ref(stage_deducer), includer, output_type, &errors,
              &total_warnings, &total_errors, compiler->initializer);
    }

    result->messages = errors.str();
    result->SetOutputData(std::move(compilation_output_data));
    result->output_data_size = compilation_output_data_size_in_bytes;
    result->num_warnings = total_warnings;
    result->num_errors = total_errors;
    if (compilation_succeeded) {
      result->compilation_status = shaderc_compilation_status_success;
    } else {
      // Check whether the error is caused by failing to deduce the shader
      // stage. If it is the case, set the error type to shader kind error.
      // Otherwise, set it to compilation error.
      result->compilation_status =
          stage_deducer.error() ? shaderc_compilation_status_invalid_stage
                                : shaderc_compilation_status_compilation_error;
    }
  }
  CATCH_IF_EXCEPTIONS_ENABLED(...) {
    result->compilation_status = shaderc_compilation_status_internal_error;
  }
  return result;
}
}  // anonymous namespace

shaderc_compilation_result_t shaderc_compile_into_spv(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options) {
  return CompileToSpecifiedOutputType(
      compiler, source_text, source_text_size, shader_kind, input_file_name,
      entry_point_name, additional_options,
      shaderc_util::Compiler::OutputType::SpirvBinary);
}

shaderc_compilation_result_t shaderc_compile_into_spv_assembly(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options) {
  return CompileToSpecifiedOutputType(
      compiler, source_text, source_text_size, shader_kind, input_file_name,
      entry_point_name, additional_options,
      shaderc_util::Compiler::OutputType::SpirvAssemblyText);
}

shaderc_compilation_result_t shaderc_compile_into_preprocessed_text(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options) {
  return CompileToSpecifiedOutputType(
      compiler, source_text, source_text_size, shader_kind, input_file_name,
      entry_point_name, additional_options,
      shaderc_util::Compiler::OutputType::PreprocessedText);
}

shaderc_compilation_result_t shaderc_assemble_into_spv(
    const shaderc_compiler_t compiler, const char* source_assembly,
    size_t source_assembly_size,
    const shaderc_compile_options_t additional_options) {
  auto* result = new (std::nothrow) shaderc_compilation_result_spv_binary;
  if (!result) return nullptr;
  result->compilation_status = shaderc_compilation_status_invalid_assembly;
  if (!compiler->initializer) return result;
  if (source_assembly == nullptr) return result;

  TRY_IF_EXCEPTIONS_ENABLED {
    spv_binary assembling_output_data = nullptr;
    std::string errors;
    const auto target_env = additional_options ? additional_options->target_env
                                               : shaderc_target_env_default;
    const uint32_t target_env_version =
        additional_options ? additional_options->target_env_version : 0;
    const bool assembling_succeeded = shaderc_util::SpirvToolsAssemble(
        GetCompilerTargetEnv(target_env),
        GetCompilerTargetEnvVersion(target_env_version),
        {source_assembly, source_assembly + source_assembly_size},
        &assembling_output_data, &errors);
    result->num_errors = !assembling_succeeded;
    if (assembling_succeeded) {
      result->SetOutputData(assembling_output_data);
      result->output_data_size =
          assembling_output_data->wordCount * sizeof(uint32_t);
      result->compilation_status = shaderc_compilation_status_success;
    } else {
      result->messages = std::move(errors);
      result->compilation_status = shaderc_compilation_status_invalid_assembly;
    }
  }
  CATCH_IF_EXCEPTIONS_ENABLED(...) {
    result->compilation_status = shaderc_compilation_status_internal_error;
  }

  return result;
}

size_t shaderc_result_get_length(const shaderc_compilation_result_t result) {
  return result->output_data_size;
}

size_t shaderc_result_get_num_warnings(
    const shaderc_compilation_result_t result) {
  return result->num_warnings;
}

size_t shaderc_result_get_num_errors(
    const shaderc_compilation_result_t result) {
  return result->num_errors;
}

const char* shaderc_result_get_bytes(
    const shaderc_compilation_result_t result) {
  return result->GetBytes();
}

void shaderc_result_release(shaderc_compilation_result_t result) {
  delete result;
}

const char* shaderc_result_get_error_message(
    const shaderc_compilation_result_t result) {
  return result->messages.c_str();
}

shaderc_compilation_status shaderc_result_get_compilation_status(
    const shaderc_compilation_result_t result) {
  return result->compilation_status;
}

void shaderc_get_spv_version(unsigned int* version, unsigned int* revision) {
  *version = spv::Version;
  *revision = spv::Revision;
}

bool shaderc_parse_version_profile(const char* str, int* version,
                                   shaderc_profile* profile) {
  EProfile glslang_profile;
  bool success = shaderc_util::ParseVersionProfile(
      std::string(str, strlen(str)), version, &glslang_profile);
  if (!success) return false;

  switch (glslang_profile) {
    case EEsProfile:
      *profile = shaderc_profile_es;
      return true;
    case ECoreProfile:
      *profile = shaderc_profile_core;
      return true;
    case ECompatibilityProfile:
      *profile = shaderc_profile_compatibility;
      return true;
    case ENoProfile:
      *profile = shaderc_profile_none;
      return true;
    case EBadProfile:
      return false;
  }

  // Shouldn't reach here, all profile enum should be handled above.
  // Be strict to return false.
  return false;
}
