// Copyright 2018 The Shaderc Authors. All rights reserved.
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

#include "shaderc/spvc.h"
#include "libshaderc_util/exceptions.h"

#include "spirv-cross/spirv_glsl.hpp"
#include "spirv-cross/spirv_hlsl.hpp"
#include "spirv-cross/spirv_msl.hpp"
#include "spirv-tools/libspirv.hpp"

// GLSL version produced when none specified nor detected from source.
#define DEFAULT_GLSL_VERSION 450

struct shaderc_spvc_compiler {};

// Described in spvc.h.
struct shaderc_spvc_compilation_result {
  std::string output;
  std::string messages;
  shaderc_compilation_status status =
      shaderc_compilation_status_null_result_object;
};

struct shaderc_spvc_compile_options {
  bool validate = true;
  bool remove_unused_variables = false;
  bool flatten_ubo = false;
  std::string entry_point;
  spv_target_env target_env = SPV_ENV_VULKAN_1_0;
  spirv_cross::CompilerGLSL::Options glsl;
  spirv_cross::CompilerHLSL::Options hlsl;
  spirv_cross::CompilerMSL::Options msl;
};

shaderc_spvc_compile_options_t shaderc_spvc_compile_options_initialize() {
  shaderc_spvc_compile_options_t options =
      new (std::nothrow) shaderc_spvc_compile_options;
  if (options) {
    options->glsl.version = 0;
    options->hlsl.point_size_compat = true;
    options->hlsl.point_coord_compat = true;
  }
  return options;
}

shaderc_spvc_compile_options_t shaderc_spvc_compile_options_clone(
    shaderc_spvc_compile_options_t options) {
  if (options) return new (std::nothrow) shaderc_spvc_compile_options(*options);
  return shaderc_spvc_compile_options_initialize();
}

void shaderc_spvc_compile_options_release(
    shaderc_spvc_compile_options_t options) {
  delete options;
}

void shaderc_spvc_compile_options_set_target_env(
    shaderc_spvc_compile_options_t options, shaderc_target_env target,
    shaderc_env_version version) {
  switch (target) {
    case shaderc_target_env_opengl:
    case shaderc_target_env_opengl_compat:
      switch (version) {
        case shaderc_env_version_opengl_4_5:
          options->target_env = SPV_ENV_OPENGL_4_5;
          break;
        default:
          break;
      }
      break;
    case shaderc_target_env_vulkan:
      switch (version) {
        case shaderc_env_version_vulkan_1_0:
          options->target_env = SPV_ENV_VULKAN_1_0;
          break;
        case shaderc_env_version_vulkan_1_1:
          options->target_env = SPV_ENV_VULKAN_1_1;
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
}

void shaderc_spvc_compile_options_set_entry_point(
    shaderc_spvc_compile_options_t options, const char* entry_point) {
  options->entry_point = entry_point;
}

void shaderc_spvc_compile_options_set_remove_unused_variables(
    shaderc_spvc_compile_options_t options, bool b) {
  options->remove_unused_variables = b;
}

void shaderc_spvc_compile_options_set_vulkan_semantics(
    shaderc_spvc_compile_options_t options, bool b) {
  options->glsl.vulkan_semantics = b;
}

void shaderc_spvc_compile_options_set_separate_shader_objects(
    shaderc_spvc_compile_options_t options, bool b) {
  options->glsl.separate_shader_objects = b;
}

void shaderc_spvc_compile_options_set_flatten_ubo(
    shaderc_spvc_compile_options_t options, bool b) {
  options->flatten_ubo = b;
}

void shaderc_spvc_compile_options_set_glsl_language_version(
    shaderc_spvc_compile_options_t options, uint32_t version) {
  options->glsl.version = version;
}

void shaderc_spvc_compile_options_set_msl_language_version(
    shaderc_spvc_compile_options_t options, uint32_t version) {
  options->msl.msl_version = version;
}

void shaderc_spvc_compile_options_set_shader_model(
    shaderc_spvc_compile_options_t options, uint32_t model) {
  options->hlsl.shader_model = model;
}

void shaderc_spvc_compile_options_set_fixup_clipspace(
    shaderc_spvc_compile_options_t options, bool b) {
  options->glsl.vertex.fixup_clipspace = b;
}

void shaderc_spvc_compile_options_set_flip_vert_y(
    shaderc_spvc_compile_options_t options, bool b) {
  options->glsl.vertex.flip_vert_y = b;
}

size_t shaderc_spvc_compile_options_set_for_fuzzing(
    shaderc_spvc_compile_options_t options, const uint8_t* data, size_t size) {
  if (!data || size < sizeof(*options)) return 0;

  memcpy(options, data, sizeof(*options));
  return sizeof(*options);
}

shaderc_spvc_compiler_t shaderc_spvc_compiler_initialize() {
  return new (std::nothrow) shaderc_spvc_compiler;
}

void shaderc_spvc_compiler_release(shaderc_spvc_compiler_t compiler) {
  delete compiler;
}

namespace {
void consume_validation_message(shaderc_spvc_compilation_result* result,
                                spv_message_level_t level, const char* src,
                                const spv_position_t& pos,
                                const char* message) {
  result->messages.append(message);
  result->messages.append("\n");
}

// Validate the source spir-v if requested, and if valid use the given compiler
// to translate it to a higher level language. CompilerGLSL is the base class
// for all spirv-cross compilers so this function works with a compiler for any
// output language. The given compiler should already have its options set by
// the caller.
shaderc_spvc_compilation_result_t validate_and_compile(
    spirv_cross::CompilerGLSL* compiler, const uint32_t* source,
    size_t source_len, shaderc_spvc_compile_options_t options) {
  auto* result = new (std::nothrow) shaderc_spvc_compilation_result;
  if (!result) return nullptr;

  if (options->validate) {
    spvtools::SpirvTools tools(options->target_env);
    if (!tools.IsValid()) return nullptr;
    tools.SetMessageConsumer(std::bind(
        consume_validation_message, result, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    if (!tools.Validate(source, source_len, spvtools::ValidatorOptions())) {
      result->status = shaderc_compilation_status_validation_error;
      return result;
    }
  }

  TRY_IF_EXCEPTIONS_ENABLED {
    result->output = compiler->compile();
    // An exception during compiling would crash (if exceptions off) or jump to
    // the catch block (if exceptions on) so if we're here we know the compile
    // worked.
    result->status = shaderc_compilation_status_success;
  }
  CATCH_IF_EXCEPTIONS_ENABLED(...) {
    result->status = shaderc_compilation_status_compilation_error;
    result->messages = "Compilation failed.  Partial source:";
    result->messages.append(compiler->get_partial_source());
  }
  return result;
}
}  // namespace

shaderc_spvc_compilation_result_t shaderc_spvc_compile_into_glsl(
    const shaderc_spvc_compiler_t, const uint32_t* source, size_t source_len,
    shaderc_spvc_compile_options_t options) {
  auto* result = new (std::nothrow) shaderc_spvc_compilation_result;
  if (!result) return nullptr;

  spirv_cross::CompilerGLSL compiler(source, source_len);

  if (options->glsl.version == 0) {
    // no version requested, was one detected in source?
    options->glsl.version = compiler.get_common_options().version;
    if (options->glsl.version == 0) {
      // no version detected in source, use default
      options->glsl.version = DEFAULT_GLSL_VERSION;
    } else {
      // version detected implies ES also detected
      options->glsl.es = compiler.get_common_options().es;
    }
  }

  auto entry_points = compiler.get_entry_points_and_stages();
  spv::ExecutionModel model = spv::ExecutionModelMax;
  if (!options->entry_point.empty()) {
    // Make sure there is just one entry point with this name, or the stage is
    // ambiguous.
    uint32_t stage_count = 0;
    for (auto& e : entry_points) {
      if (e.name == options->entry_point) {
        stage_count++;
        model = e.execution_model;
      }
    }
    if (stage_count != 1) {
      result->status = shaderc_compilation_status_compilation_error;
      if (stage_count == 0)
        result->messages =
            "There is no entry point with name: " + options->entry_point;
      else
        result->messages = "There is more than one entry point with name: " +
                           options->entry_point + ". Use --stage.";
      return result;
    }
  }
  if (!options->entry_point.empty()) {
    compiler.set_entry_point(options->entry_point, model);
  }

  if (!options->glsl.vulkan_semantics) {
    uint32_t sampler = compiler.build_dummy_sampler_for_combined_images();
    if (sampler) {
      // Set some defaults to make validation happy.
      compiler.set_decoration(sampler, spv::DecorationDescriptorSet, 0);
      compiler.set_decoration(sampler, spv::DecorationBinding, 0);
    }
  }

  spirv_cross::ShaderResources res;
  if (options->remove_unused_variables) {
    auto active = compiler.get_active_interface_variables();
    res = compiler.get_shader_resources(active);
    compiler.set_enabled_interface_variables(move(active));
  } else {
    res = compiler.get_shader_resources();
  }

  if (options->flatten_ubo) {
    for (auto& ubo : res.uniform_buffers) compiler.flatten_buffer_block(ubo.id);
    for (auto& ubo : res.push_constant_buffers)
      compiler.flatten_buffer_block(ubo.id);
  }

  if (!options->glsl.vulkan_semantics) {
    compiler.build_combined_image_samplers();

    // if (args.combined_samplers_inherit_bindings)
    //  spirv_cross_util::inherit_combined_sampler_bindings(*compiler);

    // Give the remapped combined samplers new names.
    for (auto& remap : compiler.get_combined_image_samplers()) {
      compiler.set_name(remap.combined_id,
                        spirv_cross::join("SPIRV_Cross_Combined",
                                          compiler.get_name(remap.image_id),
                                          compiler.get_name(remap.sampler_id)));
    }
  }

  shaderc_spvc_result_release(result);
  compiler.set_common_options(options->glsl);
  return validate_and_compile(&compiler, source, source_len, options);
}

shaderc_spvc_compilation_result_t shaderc_spvc_compile_into_hlsl(
    const shaderc_spvc_compiler_t, const uint32_t* source, size_t source_len,
    shaderc_spvc_compile_options_t options) {
  spirv_cross::CompilerHLSL compiler(source, source_len);
  compiler.set_common_options(options->glsl);
  compiler.set_hlsl_options(options->hlsl);
  return validate_and_compile(&compiler, source, source_len, options);
}

shaderc_spvc_compilation_result_t shaderc_spvc_compile_into_msl(
    const shaderc_spvc_compiler_t, const uint32_t* source, size_t source_len,
    shaderc_spvc_compile_options_t options) {
  spirv_cross::CompilerMSL compiler(source, source_len);
  compiler.set_common_options(options->glsl);
  return validate_and_compile(&compiler, source, source_len, options);
}

const char* shaderc_spvc_result_get_output(
    const shaderc_spvc_compilation_result_t result) {
  return result->output.c_str();
}

const char* shaderc_spvc_result_get_messages(
    const shaderc_spvc_compilation_result_t result) {
  return result->messages.c_str();
}

shaderc_compilation_status shaderc_spvc_result_get_status(
    const shaderc_spvc_compilation_result_t result) {
  return result->status;
}

void shaderc_spvc_result_release(shaderc_spvc_compilation_result_t result) {
  delete result;
}
