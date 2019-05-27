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

#ifndef SHADERC_SHADERC_H_
#define SHADERC_SHADERC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "shaderc/env.h"
#include "shaderc/status.h"
#include "shaderc/visibility.h"

// Source language kind.
typedef enum {
  shaderc_source_language_glsl,
  shaderc_source_language_hlsl,
} shaderc_source_language;

typedef enum {
  // Forced shader kinds. These shader kinds force the compiler to compile the
  // source code as the specified kind of shader.
  shaderc_vertex_shader,
  shaderc_fragment_shader,
  shaderc_compute_shader,
  shaderc_geometry_shader,
  shaderc_tess_control_shader,
  shaderc_tess_evaluation_shader,

  shaderc_glsl_vertex_shader = shaderc_vertex_shader,
  shaderc_glsl_fragment_shader = shaderc_fragment_shader,
  shaderc_glsl_compute_shader = shaderc_compute_shader,
  shaderc_glsl_geometry_shader = shaderc_geometry_shader,
  shaderc_glsl_tess_control_shader = shaderc_tess_control_shader,
  shaderc_glsl_tess_evaluation_shader = shaderc_tess_evaluation_shader,

  // Deduce the shader kind from #pragma annotation in the source code. Compiler
  // will emit error if #pragma annotation is not found.
  shaderc_glsl_infer_from_source,
  // Default shader kinds. Compiler will fall back to compile the source code as
  // the specified kind of shader when #pragma annotation is not found in the
  // source code.
  shaderc_glsl_default_vertex_shader,
  shaderc_glsl_default_fragment_shader,
  shaderc_glsl_default_compute_shader,
  shaderc_glsl_default_geometry_shader,
  shaderc_glsl_default_tess_control_shader,
  shaderc_glsl_default_tess_evaluation_shader,
  shaderc_spirv_assembly,
#ifdef NV_EXTENSIONS
  shaderc_raygen_shader,
  shaderc_anyhit_shader,
  shaderc_closesthit_shader,
  shaderc_miss_shader,
  shaderc_intersection_shader,
  shaderc_callable_shader,
  shaderc_glsl_raygen_shader = shaderc_raygen_shader,
  shaderc_glsl_anyhit_shader = shaderc_anyhit_shader,
  shaderc_glsl_closesthit_shader = shaderc_closesthit_shader,
  shaderc_glsl_miss_shader = shaderc_miss_shader,
  shaderc_glsl_intersection_shader = shaderc_intersection_shader,
  shaderc_glsl_callable_shader = shaderc_callable_shader,
  shaderc_glsl_default_raygen_shader,
  shaderc_glsl_default_anyhit_shader,
  shaderc_glsl_default_closesthit_shader,
  shaderc_glsl_default_miss_shader,
  shaderc_glsl_default_intersection_shader,
  shaderc_glsl_default_callable_shader,
  shaderc_task_shader,
  shaderc_mesh_shader,
  shaderc_glsl_task_shader = shaderc_task_shader,
  shaderc_glsl_mesh_shader = shaderc_mesh_shader,
  shaderc_glsl_default_task_shader,
  shaderc_glsl_default_mesh_shader,
#endif
} shaderc_shader_kind;

typedef enum {
  shaderc_profile_none,  // Used if and only if GLSL version did not specify
                         // profiles.
  shaderc_profile_core,
  shaderc_profile_compatibility,
  shaderc_profile_es,
} shaderc_profile;

// Optimization level.
typedef enum {
  shaderc_optimization_level_zero,  // no optimization
  shaderc_optimization_level_size,  // optimize towards reducing code size
  shaderc_optimization_level_performance,  // optimize towards performance
} shaderc_optimization_level;

// Resource limits.
typedef enum {
  shaderc_limit_max_lights,
  shaderc_limit_max_clip_planes,
  shaderc_limit_max_texture_units,
  shaderc_limit_max_texture_coords,
  shaderc_limit_max_vertex_attribs,
  shaderc_limit_max_vertex_uniform_components,
  shaderc_limit_max_varying_floats,
  shaderc_limit_max_vertex_texture_image_units,
  shaderc_limit_max_combined_texture_image_units,
  shaderc_limit_max_texture_image_units,
  shaderc_limit_max_fragment_uniform_components,
  shaderc_limit_max_draw_buffers,
  shaderc_limit_max_vertex_uniform_vectors,
  shaderc_limit_max_varying_vectors,
  shaderc_limit_max_fragment_uniform_vectors,
  shaderc_limit_max_vertex_output_vectors,
  shaderc_limit_max_fragment_input_vectors,
  shaderc_limit_min_program_texel_offset,
  shaderc_limit_max_program_texel_offset,
  shaderc_limit_max_clip_distances,
  shaderc_limit_max_compute_work_group_count_x,
  shaderc_limit_max_compute_work_group_count_y,
  shaderc_limit_max_compute_work_group_count_z,
  shaderc_limit_max_compute_work_group_size_x,
  shaderc_limit_max_compute_work_group_size_y,
  shaderc_limit_max_compute_work_group_size_z,
  shaderc_limit_max_compute_uniform_components,
  shaderc_limit_max_compute_texture_image_units,
  shaderc_limit_max_compute_image_uniforms,
  shaderc_limit_max_compute_atomic_counters,
  shaderc_limit_max_compute_atomic_counter_buffers,
  shaderc_limit_max_varying_components,
  shaderc_limit_max_vertex_output_components,
  shaderc_limit_max_geometry_input_components,
  shaderc_limit_max_geometry_output_components,
  shaderc_limit_max_fragment_input_components,
  shaderc_limit_max_image_units,
  shaderc_limit_max_combined_image_units_and_fragment_outputs,
  shaderc_limit_max_combined_shader_output_resources,
  shaderc_limit_max_image_samples,
  shaderc_limit_max_vertex_image_uniforms,
  shaderc_limit_max_tess_control_image_uniforms,
  shaderc_limit_max_tess_evaluation_image_uniforms,
  shaderc_limit_max_geometry_image_uniforms,
  shaderc_limit_max_fragment_image_uniforms,
  shaderc_limit_max_combined_image_uniforms,
  shaderc_limit_max_geometry_texture_image_units,
  shaderc_limit_max_geometry_output_vertices,
  shaderc_limit_max_geometry_total_output_components,
  shaderc_limit_max_geometry_uniform_components,
  shaderc_limit_max_geometry_varying_components,
  shaderc_limit_max_tess_control_input_components,
  shaderc_limit_max_tess_control_output_components,
  shaderc_limit_max_tess_control_texture_image_units,
  shaderc_limit_max_tess_control_uniform_components,
  shaderc_limit_max_tess_control_total_output_components,
  shaderc_limit_max_tess_evaluation_input_components,
  shaderc_limit_max_tess_evaluation_output_components,
  shaderc_limit_max_tess_evaluation_texture_image_units,
  shaderc_limit_max_tess_evaluation_uniform_components,
  shaderc_limit_max_tess_patch_components,
  shaderc_limit_max_patch_vertices,
  shaderc_limit_max_tess_gen_level,
  shaderc_limit_max_viewports,
  shaderc_limit_max_vertex_atomic_counters,
  shaderc_limit_max_tess_control_atomic_counters,
  shaderc_limit_max_tess_evaluation_atomic_counters,
  shaderc_limit_max_geometry_atomic_counters,
  shaderc_limit_max_fragment_atomic_counters,
  shaderc_limit_max_combined_atomic_counters,
  shaderc_limit_max_atomic_counter_bindings,
  shaderc_limit_max_vertex_atomic_counter_buffers,
  shaderc_limit_max_tess_control_atomic_counter_buffers,
  shaderc_limit_max_tess_evaluation_atomic_counter_buffers,
  shaderc_limit_max_geometry_atomic_counter_buffers,
  shaderc_limit_max_fragment_atomic_counter_buffers,
  shaderc_limit_max_combined_atomic_counter_buffers,
  shaderc_limit_max_atomic_counter_buffer_size,
  shaderc_limit_max_transform_feedback_buffers,
  shaderc_limit_max_transform_feedback_interleaved_components,
  shaderc_limit_max_cull_distances,
  shaderc_limit_max_combined_clip_and_cull_distances,
  shaderc_limit_max_samples,
} shaderc_limit;

// Uniform resource kinds.
// In Vulkan, uniform resources are bound to the pipeline via descriptors
// with numbered bindings and sets.
typedef enum {
  // Image and image buffer.
  shaderc_uniform_kind_image,
  // Pure sampler.
  shaderc_uniform_kind_sampler,
  // Sampled texture in GLSL, and Shader Resource View in HLSL.
  shaderc_uniform_kind_texture,
  // Uniform Buffer Object (UBO) in GLSL.  Cbuffer in HLSL.
  shaderc_uniform_kind_buffer,
  // Shader Storage Buffer Object (SSBO) in GLSL.
  shaderc_uniform_kind_storage_buffer,
  // Unordered Access View, in HLSL.  (Writable storage image or storage
  // buffer.)
  shaderc_uniform_kind_unordered_access_view,
} shaderc_uniform_kind;

// Usage examples:
//
// Aggressively release compiler resources, but spend time in initialization
// for each new use.
//      shaderc_compiler_t compiler = shaderc_compiler_initialize();
//      shaderc_compilation_result_t result = shaderc_compile_into_spv(
//          compiler, "#version 450\nvoid main() {}", 27,
//          shaderc_glsl_vertex_shader, "main.vert", "main", nullptr);
//      // Do stuff with compilation results.
//      shaderc_result_release(result);
//      shaderc_compiler_release(compiler);
//
// Keep the compiler object around for a long time, but pay for extra space
// occupied.
//      shaderc_compiler_t compiler = shaderc_compiler_initialize();
//      // On the same, other or multiple simultaneous threads.
//      shaderc_compilation_result_t result = shaderc_compile_into_spv(
//          compiler, "#version 450\nvoid main() {}", 27,
//          shaderc_glsl_vertex_shader, "main.vert", "main", nullptr);
//      // Do stuff with compilation results.
//      shaderc_result_release(result);
//      // Once no more compilations are to happen.
//      shaderc_compiler_release(compiler);

// An opaque handle to an object that manages all compiler state.
typedef struct shaderc_compiler* shaderc_compiler_t;

// Returns a shaderc_compiler_t that can be used to compile modules.
// A return of NULL indicates that there was an error initializing the compiler.
// Any function operating on shaderc_compiler_t must offer the basic
// thread-safety guarantee.
// [http://herbsutter.com/2014/01/13/gotw-95-solution-thread-safety-and-synchronization/]
// That is: concurrent invocation of these functions on DIFFERENT objects needs
// no synchronization; concurrent invocation of these functions on the SAME
// object requires synchronization IF AND ONLY IF some of them take a non-const
// argument.
SHADERC_EXPORT shaderc_compiler_t shaderc_compiler_initialize(void);

// Releases the resources held by the shaderc_compiler_t.
// After this call it is invalid to make any future calls to functions
// involving this shaderc_compiler_t.
SHADERC_EXPORT void shaderc_compiler_release(shaderc_compiler_t);

// An opaque handle to an object that manages options to a single compilation
// result.
typedef struct shaderc_compile_options* shaderc_compile_options_t;

// Returns a default-initialized shaderc_compile_options_t that can be used
// to modify the functionality of a compiled module.
// A return of NULL indicates that there was an error initializing the options.
// Any function operating on shaderc_compile_options_t must offer the
// basic thread-safety guarantee.
SHADERC_EXPORT shaderc_compile_options_t
    shaderc_compile_options_initialize(void);

// Returns a copy of the given shaderc_compile_options_t.
// If NULL is passed as the parameter the call is the same as
// shaderc_compile_options_init.
SHADERC_EXPORT shaderc_compile_options_t shaderc_compile_options_clone(
    const shaderc_compile_options_t options);

// Releases the compilation options. It is invalid to use the given
// shaderc_compile_options_t object in any future calls. It is safe to pass
// NULL to this function, and doing such will have no effect.
SHADERC_EXPORT void shaderc_compile_options_release(
    shaderc_compile_options_t options);

// Adds a predefined macro to the compilation options. This has the same
// effect as passing -Dname=value to the command-line compiler.  If value
// is NULL, it has the same effect as passing -Dname to the command-line
// compiler. If a macro definition with the same name has previously been
// added, the value is replaced with the new value. The macro name and
// value are passed in with char pointers, which point to their data, and
// the lengths of their data. The strings that the name and value pointers
// point to must remain valid for the duration of the call, but can be
// modified or deleted after this function has returned. In case of adding
// a valueless macro, the value argument should be a null pointer or the
// value_length should be 0u.
SHADERC_EXPORT void shaderc_compile_options_add_macro_definition(
    shaderc_compile_options_t options, const char* name, size_t name_length,
    const char* value, size_t value_length);

// Sets the source language.  The default is GLSL.
SHADERC_EXPORT void shaderc_compile_options_set_source_language(
    shaderc_compile_options_t options, shaderc_source_language lang);

// Sets the compiler mode to generate debug information in the output.
SHADERC_EXPORT void shaderc_compile_options_set_generate_debug_info(
    shaderc_compile_options_t options);

// Sets the compiler optimization level to the given level. Only the last one
// takes effect if multiple calls of this function exist.
SHADERC_EXPORT void shaderc_compile_options_set_optimization_level(
    shaderc_compile_options_t options, shaderc_optimization_level level);

// Forces the GLSL language version and profile to a given pair. The version
// number is the same as would appear in the #version annotation in the source.
// Version and profile specified here overrides the #version annotation in the
// source. Use profile: 'shaderc_profile_none' for GLSL versions that do not
// define profiles, e.g. versions below 150.
SHADERC_EXPORT void shaderc_compile_options_set_forced_version_profile(
    shaderc_compile_options_t options, int version, shaderc_profile profile);

// Source text inclusion via #include is supported with a pair of callbacks
// to an "includer" on the client side.  The first callback processes an
// inclusion request, and returns an include result.  The includer owns
// the contents of the result, and those contents must remain valid until the
// second callback is invoked to release the result.  Both callbacks take a
// user_data argument to specify the client context.
// To return an error, set the source_name to an empty string and put your
// error message in content.

// An include result.
typedef struct shaderc_include_result {
  // The name of the source file.  The name should be fully resolved
  // in the sense that it should be a unique name in the context of the
  // includer.  For example, if the includer maps source names to files in
  // a filesystem, then this name should be the absolute path of the file.
  // For a failed inclusion, this string is empty.
  const char* source_name;
  size_t source_name_length;
  // The text contents of the source file in the normal case.
  // For a failed inclusion, this contains the error message.
  const char* content;
  size_t content_length;
  // User data to be passed along with this request.
  void* user_data;
} shaderc_include_result;

// The kinds of include requests.
enum shaderc_include_type {
  shaderc_include_type_relative,  // E.g. #include "source"
  shaderc_include_type_standard   // E.g. #include <source>
};

// An includer callback type for mapping an #include request to an include
// result.  The user_data parameter specifies the client context.  The
// requested_source parameter specifies the name of the source being requested.
// The type parameter specifies the kind of inclusion request being made.
// The requesting_source parameter specifies the name of the source containing
// the #include request.  The includer owns the result object and its contents,
// and both must remain valid until the release callback is called on the result
// object.
typedef shaderc_include_result* (*shaderc_include_resolve_fn)(
    void* user_data, const char* requested_source, int type,
    const char* requesting_source, size_t include_depth);

// An includer callback type for destroying an include result.
typedef void (*shaderc_include_result_release_fn)(
    void* user_data, shaderc_include_result* include_result);

// Sets includer callback functions.
SHADERC_EXPORT void shaderc_compile_options_set_include_callbacks(
    shaderc_compile_options_t options, shaderc_include_resolve_fn resolver,
    shaderc_include_result_release_fn result_releaser, void* user_data);

// Sets the compiler mode to suppress warnings, overriding warnings-as-errors
// mode. When both suppress-warnings and warnings-as-errors modes are
// turned on, warning messages will be inhibited, and will not be emitted
// as error messages.
SHADERC_EXPORT void shaderc_compile_options_set_suppress_warnings(
    shaderc_compile_options_t options);

// Sets the target shader environment, affecting which warnings or errors will
// be issued.  The version will be for distinguishing between different versions
// of the target environment.  The version value should be either 0 or
// a value listed in shaderc_env_version.  The 0 value maps to Vulkan 1.0 if
// |target| is Vulkan, and it maps to OpenGL 4.5 if |target| is OpenGL.
SHADERC_EXPORT void shaderc_compile_options_set_target_env(
    shaderc_compile_options_t options,
    shaderc_target_env target,
    uint32_t version);

// Sets the compiler mode to treat all warnings as errors. Note the
// suppress-warnings mode overrides this option, i.e. if both
// warning-as-errors and suppress-warnings modes are set, warnings will not
// be emitted as error messages.
SHADERC_EXPORT void shaderc_compile_options_set_warnings_as_errors(
    shaderc_compile_options_t options);

// Sets a resource limit.
SHADERC_EXPORT void shaderc_compile_options_set_limit(
    shaderc_compile_options_t options, shaderc_limit limit, int value);

// Sets whether the compiler should automatically assign bindings to uniforms
// that aren't already explicitly bound in the shader source.
SHADERC_EXPORT void shaderc_compile_options_set_auto_bind_uniforms(
    shaderc_compile_options_t options, bool auto_bind);

// Sets whether the compiler should use HLSL IO mapping rules for bindings.
// Defaults to false.
SHADERC_EXPORT void shaderc_compile_options_set_hlsl_io_mapping(
    shaderc_compile_options_t options, bool hlsl_iomap);

// Sets whether the compiler should determine block member offsets using HLSL
// packing rules instead of standard GLSL rules.  Defaults to false.  Only
// affects GLSL compilation.  HLSL rules are always used when compiling HLSL.
SHADERC_EXPORT void shaderc_compile_options_set_hlsl_offsets(
    shaderc_compile_options_t options, bool hlsl_offsets);

// Sets the base binding number used for for a uniform resource type when
// automatically assigning bindings.  For GLSL compilation, sets the lowest
// automatically assigned number.  For HLSL compilation, the regsiter number
// assigned to the resource is added to this specified base.
SHADERC_EXPORT void shaderc_compile_options_set_binding_base(
    shaderc_compile_options_t options,
    shaderc_uniform_kind kind,
    uint32_t base);

// Like shaderc_compile_options_set_binding_base, but only takes effect when
// compiling a given shader stage.  The stage is assumed to be one of vertex,
// fragment, tessellation evaluation, tesselation control, geometry, or compute.
SHADERC_EXPORT void shaderc_compile_options_set_binding_base_for_stage(
    shaderc_compile_options_t options, shaderc_shader_kind shader_kind,
    shaderc_uniform_kind kind, uint32_t base);

// Sets whether the compiler should automatically assign locations to
// uniform variables that don't have explicit locations in the shader source.
SHADERC_EXPORT void shaderc_compile_options_set_auto_map_locations(
    shaderc_compile_options_t options, bool auto_map);

// Sets a descriptor set and binding for an HLSL register in the given stage.
// This method keeps a copy of the string data.
SHADERC_EXPORT void shaderc_compile_options_set_hlsl_register_set_and_binding_for_stage(
    shaderc_compile_options_t options, shaderc_shader_kind shader_kind,
    const char* reg, const char* set, const char* binding);

// Like shaderc_compile_options_set_hlsl_register_set_and_binding_for_stage,
// but affects all shader stages.
SHADERC_EXPORT void shaderc_compile_options_set_hlsl_register_set_and_binding(
    shaderc_compile_options_t options, const char* reg, const char* set,
    const char* binding);

// Sets whether the compiler should enable extension
// SPV_GOOGLE_hlsl_functionality1.
SHADERC_EXPORT void shaderc_compile_options_set_hlsl_functionality1(
    shaderc_compile_options_t options, bool enable);

// An opaque handle to the results of a call to any shaderc_compile_into_*()
// function.
typedef struct shaderc_compilation_result* shaderc_compilation_result_t;

// Takes a GLSL source string and the associated shader kind, input file
// name, compiles it according to the given additional_options. If the shader
// kind is not set to a specified kind, but shaderc_glslc_infer_from_source,
// the compiler will try to deduce the shader kind from the source
// string and a failure in deducing will generate an error. Currently only
// #pragma annotation is supported. If the shader kind is set to one of the
// default shader kinds, the compiler will fall back to the default shader
// kind in case it failed to deduce the shader kind from source string.
// The input_file_name is a null-termintated string. It is used as a tag to
// identify the source string in cases like emitting error messages. It
// doesn't have to be a 'file name'.
// The source string will be compiled into SPIR-V binary and a
// shaderc_compilation_result will be returned to hold the results.
// The entry_point_name null-terminated string defines the name of the entry
// point to associate with this GLSL source. If the additional_options
// parameter is not null, then the compilation is modified by any options
// present.  May be safely called from multiple threads without explicit
// synchronization. If there was failure in allocating the compiler object,
// null will be returned.
SHADERC_EXPORT shaderc_compilation_result_t shaderc_compile_into_spv(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options);

// Like shaderc_compile_into_spv, but the result contains SPIR-V assembly text
// instead of a SPIR-V binary module.  The SPIR-V assembly syntax is as defined
// by the SPIRV-Tools open source project.
SHADERC_EXPORT shaderc_compilation_result_t shaderc_compile_into_spv_assembly(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options);

// Like shaderc_compile_into_spv, but the result contains preprocessed source
// code instead of a SPIR-V binary module
SHADERC_EXPORT shaderc_compilation_result_t shaderc_compile_into_preprocessed_text(
    const shaderc_compiler_t compiler, const char* source_text,
    size_t source_text_size, shaderc_shader_kind shader_kind,
    const char* input_file_name, const char* entry_point_name,
    const shaderc_compile_options_t additional_options);

// Takes an assembly string of the format defined in the SPIRV-Tools project
// (https://github.com/KhronosGroup/SPIRV-Tools/blob/master/syntax.md),
// assembles it into SPIR-V binary and a shaderc_compilation_result will be
// returned to hold the results.
// The assembling will pick options suitable for assembling specified in the
// additional_options parameter.
// May be safely called from multiple threads without explicit synchronization.
// If there was failure in allocating the compiler object, null will be
// returned.
SHADERC_EXPORT shaderc_compilation_result_t shaderc_assemble_into_spv(
    const shaderc_compiler_t compiler, const char* source_assembly,
    size_t source_assembly_size,
    const shaderc_compile_options_t additional_options);

// The following functions, operating on shaderc_compilation_result_t objects,
// offer only the basic thread-safety guarantee.

// Releases the resources held by the result object. It is invalid to use the
// result object for any further operations.
SHADERC_EXPORT void shaderc_result_release(shaderc_compilation_result_t result);

// Returns the number of bytes of the compilation output data in a result
// object.
SHADERC_EXPORT size_t shaderc_result_get_length(const shaderc_compilation_result_t result);

// Returns the number of warnings generated during the compilation.
SHADERC_EXPORT size_t shaderc_result_get_num_warnings(
    const shaderc_compilation_result_t result);

// Returns the number of errors generated during the compilation.
SHADERC_EXPORT size_t shaderc_result_get_num_errors(const shaderc_compilation_result_t result);

// Returns the compilation status, indicating whether the compilation succeeded,
// or failed due to some reasons, like invalid shader stage or compilation
// errors.
SHADERC_EXPORT shaderc_compilation_status shaderc_result_get_compilation_status(
    const shaderc_compilation_result_t);

// Returns a pointer to the start of the compilation output data bytes, either
// SPIR-V binary or char string. When the source string is compiled into SPIR-V
// binary, this is guaranteed to be castable to a uint32_t*. If the result
// contains assembly text or preprocessed source text, the pointer will point to
// the resulting array of characters.
SHADERC_EXPORT const char* shaderc_result_get_bytes(const shaderc_compilation_result_t result);

// Returns a null-terminated string that contains any error messages generated
// during the compilation.
SHADERC_EXPORT const char* shaderc_result_get_error_message(
    const shaderc_compilation_result_t result);

// Provides the version & revision of the SPIR-V which will be produced
SHADERC_EXPORT void shaderc_get_spv_version(unsigned int* version, unsigned int* revision);

// Parses the version and profile from a given null-terminated string
// containing both version and profile, like: '450core'. Returns false if
// the string can not be parsed. Returns true when the parsing succeeds. The
// parsed version and profile are returned through arguments.
SHADERC_EXPORT bool shaderc_parse_version_profile(const char* str, int* version,
                                   shaderc_profile* profile);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // SHADERC_SHADERC_H_
