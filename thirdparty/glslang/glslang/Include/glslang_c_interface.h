/**
    This code is based on the glslang_c_interface implementation by Viktor Latypov
**/

/**
BSD 2-Clause License

Copyright (c) 2019, Viktor Latypov
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#ifndef GLSLANG_C_IFACE_H_INCLUDED
#define GLSLANG_C_IFACE_H_INCLUDED

#include <stdbool.h>
#include <stdlib.h>

#include "glslang_c_shader_types.h"

typedef struct glslang_shader_s glslang_shader_t;
typedef struct glslang_program_s glslang_program_t;

/* TLimits counterpart */
typedef struct glslang_limits_s {
    bool non_inductive_for_loops;
    bool while_loops;
    bool do_while_loops;
    bool general_uniform_indexing;
    bool general_attribute_matrix_vector_indexing;
    bool general_varying_indexing;
    bool general_sampler_indexing;
    bool general_variable_indexing;
    bool general_constant_matrix_vector_indexing;
} glslang_limits_t;

/* TBuiltInResource counterpart */
typedef struct glslang_resource_s {
    int max_lights;
    int max_clip_planes;
    int max_texture_units;
    int max_texture_coords;
    int max_vertex_attribs;
    int max_vertex_uniform_components;
    int max_varying_floats;
    int max_vertex_texture_image_units;
    int max_combined_texture_image_units;
    int max_texture_image_units;
    int max_fragment_uniform_components;
    int max_draw_buffers;
    int max_vertex_uniform_vectors;
    int max_varying_vectors;
    int max_fragment_uniform_vectors;
    int max_vertex_output_vectors;
    int max_fragment_input_vectors;
    int min_program_texel_offset;
    int max_program_texel_offset;
    int max_clip_distances;
    int max_compute_work_group_count_x;
    int max_compute_work_group_count_y;
    int max_compute_work_group_count_z;
    int max_compute_work_group_size_x;
    int max_compute_work_group_size_y;
    int max_compute_work_group_size_z;
    int max_compute_uniform_components;
    int max_compute_texture_image_units;
    int max_compute_image_uniforms;
    int max_compute_atomic_counters;
    int max_compute_atomic_counter_buffers;
    int max_varying_components;
    int max_vertex_output_components;
    int max_geometry_input_components;
    int max_geometry_output_components;
    int max_fragment_input_components;
    int max_image_units;
    int max_combined_image_units_and_fragment_outputs;
    int max_combined_shader_output_resources;
    int max_image_samples;
    int max_vertex_image_uniforms;
    int max_tess_control_image_uniforms;
    int max_tess_evaluation_image_uniforms;
    int max_geometry_image_uniforms;
    int max_fragment_image_uniforms;
    int max_combined_image_uniforms;
    int max_geometry_texture_image_units;
    int max_geometry_output_vertices;
    int max_geometry_total_output_components;
    int max_geometry_uniform_components;
    int max_geometry_varying_components;
    int max_tess_control_input_components;
    int max_tess_control_output_components;
    int max_tess_control_texture_image_units;
    int max_tess_control_uniform_components;
    int max_tess_control_total_output_components;
    int max_tess_evaluation_input_components;
    int max_tess_evaluation_output_components;
    int max_tess_evaluation_texture_image_units;
    int max_tess_evaluation_uniform_components;
    int max_tess_patch_components;
    int max_patch_vertices;
    int max_tess_gen_level;
    int max_viewports;
    int max_vertex_atomic_counters;
    int max_tess_control_atomic_counters;
    int max_tess_evaluation_atomic_counters;
    int max_geometry_atomic_counters;
    int max_fragment_atomic_counters;
    int max_combined_atomic_counters;
    int max_atomic_counter_bindings;
    int max_vertex_atomic_counter_buffers;
    int max_tess_control_atomic_counter_buffers;
    int max_tess_evaluation_atomic_counter_buffers;
    int max_geometry_atomic_counter_buffers;
    int max_fragment_atomic_counter_buffers;
    int max_combined_atomic_counter_buffers;
    int max_atomic_counter_buffer_size;
    int max_transform_feedback_buffers;
    int max_transform_feedback_interleaved_components;
    int max_cull_distances;
    int max_combined_clip_and_cull_distances;
    int max_samples;
    int max_mesh_output_vertices_nv;
    int max_mesh_output_primitives_nv;
    int max_mesh_work_group_size_x_nv;
    int max_mesh_work_group_size_y_nv;
    int max_mesh_work_group_size_z_nv;
    int max_task_work_group_size_x_nv;
    int max_task_work_group_size_y_nv;
    int max_task_work_group_size_z_nv;
    int max_mesh_view_count_nv;
    int max_mesh_output_vertices_ext;
    int max_mesh_output_primitives_ext;
    int max_mesh_work_group_size_x_ext;
    int max_mesh_work_group_size_y_ext;
    int max_mesh_work_group_size_z_ext;
    int max_task_work_group_size_x_ext;
    int max_task_work_group_size_y_ext;
    int max_task_work_group_size_z_ext;
    int max_mesh_view_count_ext;
    int maxDualSourceDrawBuffersEXT;

    glslang_limits_t limits;
} glslang_resource_t;

typedef struct glslang_input_s {
    glslang_source_t language;
    glslang_stage_t stage;
    glslang_client_t client;
    glslang_target_client_version_t client_version;
    glslang_target_language_t target_language;
    glslang_target_language_version_t target_language_version;
    /** Shader source code */
    const char* code;
    int default_version;
    glslang_profile_t default_profile;
    int force_default_version_and_profile;
    int forward_compatible;
    glslang_messages_t messages;
    const glslang_resource_t* resource;
} glslang_input_t;

/* Inclusion result structure allocated by C include_local/include_system callbacks */
typedef struct glsl_include_result_s {
    /* Header file name or NULL if inclusion failed */
    const char* header_name;

    /* Header contents or NULL */
    const char* header_data;
    size_t header_length;

} glsl_include_result_t;

/* Callback for local file inclusion */
typedef glsl_include_result_t* (*glsl_include_local_func)(void* ctx, const char* header_name, const char* includer_name,
                                                          size_t include_depth);

/* Callback for system file inclusion */
typedef glsl_include_result_t* (*glsl_include_system_func)(void* ctx, const char* header_name,
                                                           const char* includer_name, size_t include_depth);

/* Callback for include result destruction */
typedef int (*glsl_free_include_result_func)(void* ctx, glsl_include_result_t* result);

/* Collection of callbacks for GLSL preprocessor */
typedef struct glsl_include_callbacks_s {
    glsl_include_system_func include_system;
    glsl_include_local_func include_local;
    glsl_free_include_result_func free_include_result;
} glsl_include_callbacks_t;

/* SpvOptions counterpart */
typedef struct glslang_spv_options_s {
    bool generate_debug_info;
    bool strip_debug_info;
    bool disable_optimizer;
    bool optimize_size;
    bool disassemble;
    bool validate;
    bool emit_nonsemantic_shader_debug_info;
    bool emit_nonsemantic_shader_debug_source;
} glslang_spv_options_t;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GLSLANG_IS_SHARED_LIBRARY
    #ifdef _WIN32
        #ifdef GLSLANG_EXPORTING
            #define GLSLANG_EXPORT __declspec(dllexport)
        #else
            #define GLSLANG_EXPORT __declspec(dllimport)
        #endif
    #elif __GNUC__ >= 4
        #define GLSLANG_EXPORT __attribute__((visibility("default")))
    #endif
#endif // GLSLANG_IS_SHARED_LIBRARY

#ifndef GLSLANG_EXPORT
#define GLSLANG_EXPORT
#endif

GLSLANG_EXPORT int glslang_initialize_process();
GLSLANG_EXPORT void glslang_finalize_process();

GLSLANG_EXPORT glslang_shader_t* glslang_shader_create(const glslang_input_t* input);
GLSLANG_EXPORT void glslang_shader_delete(glslang_shader_t* shader);
GLSLANG_EXPORT void glslang_shader_set_preamble(glslang_shader_t* shader, const char* s);
GLSLANG_EXPORT void glslang_shader_shift_binding(glslang_shader_t* shader, glslang_resource_type_t res, unsigned int base);
GLSLANG_EXPORT void glslang_shader_shift_binding_for_set(glslang_shader_t* shader, glslang_resource_type_t res, unsigned int base, unsigned int set);
GLSLANG_EXPORT void glslang_shader_set_options(glslang_shader_t* shader, int options); // glslang_shader_options_t
GLSLANG_EXPORT void glslang_shader_set_glsl_version(glslang_shader_t* shader, int version);
GLSLANG_EXPORT int glslang_shader_preprocess(glslang_shader_t* shader, const glslang_input_t* input);
GLSLANG_EXPORT int glslang_shader_parse(glslang_shader_t* shader, const glslang_input_t* input);
GLSLANG_EXPORT const char* glslang_shader_get_preprocessed_code(glslang_shader_t* shader);
GLSLANG_EXPORT const char* glslang_shader_get_info_log(glslang_shader_t* shader);
GLSLANG_EXPORT const char* glslang_shader_get_info_debug_log(glslang_shader_t* shader);

GLSLANG_EXPORT glslang_program_t* glslang_program_create();
GLSLANG_EXPORT void glslang_program_delete(glslang_program_t* program);
GLSLANG_EXPORT void glslang_program_add_shader(glslang_program_t* program, glslang_shader_t* shader);
GLSLANG_EXPORT int glslang_program_link(glslang_program_t* program, int messages); // glslang_messages_t
GLSLANG_EXPORT void glslang_program_add_source_text(glslang_program_t* program, glslang_stage_t stage, const char* text, size_t len);
GLSLANG_EXPORT void glslang_program_set_source_file(glslang_program_t* program, glslang_stage_t stage, const char* file);
GLSLANG_EXPORT int glslang_program_map_io(glslang_program_t* program);
GLSLANG_EXPORT void glslang_program_SPIRV_generate(glslang_program_t* program, glslang_stage_t stage);
GLSLANG_EXPORT void glslang_program_SPIRV_generate_with_options(glslang_program_t* program, glslang_stage_t stage, glslang_spv_options_t* spv_options);
GLSLANG_EXPORT size_t glslang_program_SPIRV_get_size(glslang_program_t* program);
GLSLANG_EXPORT void glslang_program_SPIRV_get(glslang_program_t* program, unsigned int*);
GLSLANG_EXPORT unsigned int* glslang_program_SPIRV_get_ptr(glslang_program_t* program);
GLSLANG_EXPORT const char* glslang_program_SPIRV_get_messages(glslang_program_t* program);
GLSLANG_EXPORT const char* glslang_program_get_info_log(glslang_program_t* program);
GLSLANG_EXPORT const char* glslang_program_get_info_debug_log(glslang_program_t* program);

#ifdef __cplusplus
}
#endif

#endif /* #ifdef GLSLANG_C_IFACE_INCLUDED */
