//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2015-2018 Google, Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _VERSIONS_INCLUDED_
#define _VERSIONS_INCLUDED_

#define LAST_ELEMENT_MARKER(x) x

//
// Help manage multiple profiles, versions, extensions etc.
//

//
// Profiles are set up for masking operations, so queries can be done on multiple
// profiles at the same time.
//
// Don't maintain an ordinal set of enums (0,1,2,3...) to avoid all possible
// defects from mixing the two different forms.
//
typedef enum : unsigned {
    EBadProfile           = 0,
    ENoProfile            = (1 << 0), // only for desktop, before profiles showed up
    ECoreProfile          = (1 << 1),
    ECompatibilityProfile = (1 << 2),
    EEsProfile            = (1 << 3),
    LAST_ELEMENT_MARKER(EProfileCount),
} EProfile;

namespace glslang {

//
// Map from profile enum to externally readable text name.
//
inline const char* ProfileName(EProfile profile)
{
    switch (profile) {
    case ENoProfile:             return "none";
    case ECoreProfile:           return "core";
    case ECompatibilityProfile:  return "compatibility";
    case EEsProfile:             return "es";
    default:                     return "unknown profile";
    }
}

//
// What source rules, validation rules, target language, etc. are needed or
// desired for SPIR-V?
//
// 0 means a target or rule set is not enabled (ignore rules from that entity).
// Non-0 means to apply semantic rules arising from that version of its rule set.
// The union of all requested rule sets will be applied.
//
struct SpvVersion {
    SpvVersion() : spv(0), vulkanGlsl(0), vulkan(0), openGl(0), vulkanRelaxed(false) {}
    unsigned int spv; // the version of SPIR-V to target, as defined by "word 1" of the SPIR-V binary header
    int vulkanGlsl;   // the version of GLSL semantics for Vulkan, from GL_KHR_vulkan_glsl, for "#define VULKAN XXX"
    int vulkan;       // the version of Vulkan, for which SPIR-V execution environment rules to use
    int openGl;       // the version of GLSL semantics for OpenGL, from GL_ARB_gl_spirv, for "#define GL_SPIRV XXX"
    bool vulkanRelaxed; // relax changes to GLSL for Vulkan, allowing some GL-specific to be compiled to Vulkan SPIR-V target
};

//
// The behaviors from the GLSL "#extension extension_name : behavior"
//
typedef enum {
    EBhMissing = 0,
    EBhRequire,
    EBhEnable,
    EBhWarn,
    EBhDisable,
    EBhDisablePartial    // use as initial state of an extension that is only partially implemented
} TExtensionBehavior;

//
// Symbolic names for extensions.  Strings may be directly used when calling the
// functions, but better to have the compiler do spelling checks.
//
const char* const E_GL_OES_texture_3D                   = "GL_OES_texture_3D";
const char* const E_GL_OES_standard_derivatives         = "GL_OES_standard_derivatives";
const char* const E_GL_EXT_frag_depth                   = "GL_EXT_frag_depth";
const char* const E_GL_OES_EGL_image_external           = "GL_OES_EGL_image_external";
const char* const E_GL_OES_EGL_image_external_essl3     = "GL_OES_EGL_image_external_essl3";
const char* const E_GL_EXT_YUV_target                   = "GL_EXT_YUV_target";
const char* const E_GL_EXT_shader_texture_lod           = "GL_EXT_shader_texture_lod";
const char* const E_GL_EXT_shadow_samplers              = "GL_EXT_shadow_samplers";

const char* const E_GL_ARB_texture_rectangle            = "GL_ARB_texture_rectangle";
const char* const E_GL_3DL_array_objects                = "GL_3DL_array_objects";
const char* const E_GL_ARB_shading_language_420pack     = "GL_ARB_shading_language_420pack";
const char* const E_GL_ARB_texture_gather               = "GL_ARB_texture_gather";
const char* const E_GL_ARB_gpu_shader5                  = "GL_ARB_gpu_shader5";
const char* const E_GL_ARB_separate_shader_objects      = "GL_ARB_separate_shader_objects";
const char* const E_GL_ARB_compute_shader               = "GL_ARB_compute_shader";
const char* const E_GL_ARB_tessellation_shader          = "GL_ARB_tessellation_shader";
const char* const E_GL_ARB_enhanced_layouts             = "GL_ARB_enhanced_layouts";
const char* const E_GL_ARB_texture_cube_map_array       = "GL_ARB_texture_cube_map_array";
const char* const E_GL_ARB_texture_multisample          = "GL_ARB_texture_multisample";
const char* const E_GL_ARB_shader_texture_lod           = "GL_ARB_shader_texture_lod";
const char* const E_GL_ARB_explicit_attrib_location     = "GL_ARB_explicit_attrib_location";
const char* const E_GL_ARB_explicit_uniform_location    = "GL_ARB_explicit_uniform_location";
const char* const E_GL_ARB_shader_image_load_store      = "GL_ARB_shader_image_load_store";
const char* const E_GL_ARB_shader_atomic_counters       = "GL_ARB_shader_atomic_counters";
const char* const E_GL_ARB_shader_atomic_counter_ops    = "GL_ARB_shader_atomic_counter_ops";
const char* const E_GL_ARB_shader_draw_parameters       = "GL_ARB_shader_draw_parameters";
const char* const E_GL_ARB_shader_group_vote            = "GL_ARB_shader_group_vote";
const char* const E_GL_ARB_derivative_control           = "GL_ARB_derivative_control";
const char* const E_GL_ARB_shader_texture_image_samples = "GL_ARB_shader_texture_image_samples";
const char* const E_GL_ARB_viewport_array               = "GL_ARB_viewport_array";
const char* const E_GL_ARB_gpu_shader_int64             = "GL_ARB_gpu_shader_int64";
const char* const E_GL_ARB_gpu_shader_fp64              = "GL_ARB_gpu_shader_fp64";
const char* const E_GL_ARB_shader_ballot                = "GL_ARB_shader_ballot";
const char* const E_GL_ARB_sparse_texture2              = "GL_ARB_sparse_texture2";
const char* const E_GL_ARB_sparse_texture_clamp         = "GL_ARB_sparse_texture_clamp";
const char* const E_GL_ARB_shader_stencil_export        = "GL_ARB_shader_stencil_export";
// const char* const E_GL_ARB_cull_distance            = "GL_ARB_cull_distance";  // present for 4.5, but need extension control over block members
const char* const E_GL_ARB_post_depth_coverage          = "GL_ARB_post_depth_coverage";
const char* const E_GL_ARB_shader_viewport_layer_array  = "GL_ARB_shader_viewport_layer_array";
const char* const E_GL_ARB_fragment_shader_interlock    = "GL_ARB_fragment_shader_interlock";
const char* const E_GL_ARB_shader_clock                 = "GL_ARB_shader_clock";
const char* const E_GL_ARB_uniform_buffer_object        = "GL_ARB_uniform_buffer_object";
const char* const E_GL_ARB_sample_shading               = "GL_ARB_sample_shading";
const char* const E_GL_ARB_shader_bit_encoding          = "GL_ARB_shader_bit_encoding";
const char* const E_GL_ARB_shader_image_size            = "GL_ARB_shader_image_size";
const char* const E_GL_ARB_shader_storage_buffer_object = "GL_ARB_shader_storage_buffer_object";
const char* const E_GL_ARB_shading_language_packing     = "GL_ARB_shading_language_packing";
const char* const E_GL_ARB_texture_query_lod            = "GL_ARB_texture_query_lod";
const char* const E_GL_ARB_vertex_attrib_64bit          = "GL_ARB_vertex_attrib_64bit";
const char* const E_GL_ARB_draw_instanced               = "GL_ARB_draw_instanced";
const char* const E_GL_ARB_fragment_coord_conventions   = "GL_ARB_fragment_coord_conventions";
const char* const E_GL_ARB_bindless_texture             = "GL_ARB_bindless_texture";

const char* const E_GL_KHR_shader_subgroup_basic            = "GL_KHR_shader_subgroup_basic";
const char* const E_GL_KHR_shader_subgroup_vote             = "GL_KHR_shader_subgroup_vote";
const char* const E_GL_KHR_shader_subgroup_arithmetic       = "GL_KHR_shader_subgroup_arithmetic";
const char* const E_GL_KHR_shader_subgroup_ballot           = "GL_KHR_shader_subgroup_ballot";
const char* const E_GL_KHR_shader_subgroup_shuffle          = "GL_KHR_shader_subgroup_shuffle";
const char* const E_GL_KHR_shader_subgroup_shuffle_relative = "GL_KHR_shader_subgroup_shuffle_relative";
const char* const E_GL_KHR_shader_subgroup_clustered        = "GL_KHR_shader_subgroup_clustered";
const char* const E_GL_KHR_shader_subgroup_quad             = "GL_KHR_shader_subgroup_quad";
const char* const E_GL_KHR_memory_scope_semantics           = "GL_KHR_memory_scope_semantics";
const char* const E_GL_KHR_cooperative_matrix               = "GL_KHR_cooperative_matrix";

const char* const E_GL_EXT_shader_atomic_int64              = "GL_EXT_shader_atomic_int64";

const char* const E_GL_EXT_shader_non_constant_global_initializers = "GL_EXT_shader_non_constant_global_initializers";
const char* const E_GL_EXT_shader_image_load_formatted = "GL_EXT_shader_image_load_formatted";

const char* const E_GL_EXT_shader_16bit_storage             = "GL_EXT_shader_16bit_storage";
const char* const E_GL_EXT_shader_8bit_storage              = "GL_EXT_shader_8bit_storage";


// EXT extensions
const char* const E_GL_EXT_device_group                     = "GL_EXT_device_group";
const char* const E_GL_EXT_multiview                        = "GL_EXT_multiview";
const char* const E_GL_EXT_post_depth_coverage              = "GL_EXT_post_depth_coverage";
const char* const E_GL_EXT_control_flow_attributes          = "GL_EXT_control_flow_attributes";
const char* const E_GL_EXT_nonuniform_qualifier             = "GL_EXT_nonuniform_qualifier";
const char* const E_GL_EXT_samplerless_texture_functions    = "GL_EXT_samplerless_texture_functions";
const char* const E_GL_EXT_scalar_block_layout              = "GL_EXT_scalar_block_layout";
const char* const E_GL_EXT_fragment_invocation_density      = "GL_EXT_fragment_invocation_density";
const char* const E_GL_EXT_buffer_reference                 = "GL_EXT_buffer_reference";
const char* const E_GL_EXT_buffer_reference2                = "GL_EXT_buffer_reference2";
const char* const E_GL_EXT_buffer_reference_uvec2           = "GL_EXT_buffer_reference_uvec2";
const char* const E_GL_EXT_demote_to_helper_invocation      = "GL_EXT_demote_to_helper_invocation";
const char* const E_GL_EXT_shader_realtime_clock            = "GL_EXT_shader_realtime_clock";
const char* const E_GL_EXT_debug_printf                     = "GL_EXT_debug_printf";
const char* const E_GL_EXT_ray_tracing                      = "GL_EXT_ray_tracing";
const char* const E_GL_EXT_ray_query                        = "GL_EXT_ray_query";
const char* const E_GL_EXT_ray_flags_primitive_culling      = "GL_EXT_ray_flags_primitive_culling";
const char* const E_GL_EXT_ray_cull_mask                    = "GL_EXT_ray_cull_mask";
const char* const E_GL_EXT_blend_func_extended              = "GL_EXT_blend_func_extended";
const char* const E_GL_EXT_shader_implicit_conversions      = "GL_EXT_shader_implicit_conversions";
const char* const E_GL_EXT_fragment_shading_rate            = "GL_EXT_fragment_shading_rate";
const char* const E_GL_EXT_shader_image_int64               = "GL_EXT_shader_image_int64";
const char* const E_GL_EXT_null_initializer                 = "GL_EXT_null_initializer";
const char* const E_GL_EXT_shared_memory_block              = "GL_EXT_shared_memory_block";
const char* const E_GL_EXT_subgroup_uniform_control_flow    = "GL_EXT_subgroup_uniform_control_flow";
const char* const E_GL_EXT_spirv_intrinsics                 = "GL_EXT_spirv_intrinsics";
const char* const E_GL_EXT_fragment_shader_barycentric      = "GL_EXT_fragment_shader_barycentric";
const char* const E_GL_EXT_mesh_shader                      = "GL_EXT_mesh_shader";
const char* const E_GL_EXT_opacity_micromap                 = "GL_EXT_opacity_micromap";

// Arrays of extensions for the above viewportEXTs duplications

const char* const post_depth_coverageEXTs[] = { E_GL_ARB_post_depth_coverage, E_GL_EXT_post_depth_coverage };
const int Num_post_depth_coverageEXTs = sizeof(post_depth_coverageEXTs) / sizeof(post_depth_coverageEXTs[0]);

// Array of extensions to cover both extensions providing ray tracing capabilities.
const char* const ray_tracing_EXTs[] = { E_GL_EXT_ray_query, E_GL_EXT_ray_tracing };
const int Num_ray_tracing_EXTs = sizeof(ray_tracing_EXTs) / sizeof(ray_tracing_EXTs[0]);

// OVR extensions
const char* const E_GL_OVR_multiview                    = "GL_OVR_multiview";
const char* const E_GL_OVR_multiview2                   = "GL_OVR_multiview2";

const char* const OVR_multiview_EXTs[] = { E_GL_OVR_multiview, E_GL_OVR_multiview2 };
const int Num_OVR_multiview_EXTs = sizeof(OVR_multiview_EXTs) / sizeof(OVR_multiview_EXTs[0]);

// #line and #include
const char* const E_GL_GOOGLE_cpp_style_line_directive          = "GL_GOOGLE_cpp_style_line_directive";
const char* const E_GL_GOOGLE_include_directive                 = "GL_GOOGLE_include_directive";

const char* const E_GL_AMD_shader_ballot                        = "GL_AMD_shader_ballot";
const char* const E_GL_AMD_shader_trinary_minmax                = "GL_AMD_shader_trinary_minmax";
const char* const E_GL_AMD_shader_explicit_vertex_parameter     = "GL_AMD_shader_explicit_vertex_parameter";
const char* const E_GL_AMD_gcn_shader                           = "GL_AMD_gcn_shader";
const char* const E_GL_AMD_gpu_shader_half_float                = "GL_AMD_gpu_shader_half_float";
const char* const E_GL_AMD_texture_gather_bias_lod              = "GL_AMD_texture_gather_bias_lod";
const char* const E_GL_AMD_gpu_shader_int16                     = "GL_AMD_gpu_shader_int16";
const char* const E_GL_AMD_shader_image_load_store_lod          = "GL_AMD_shader_image_load_store_lod";
const char* const E_GL_AMD_shader_fragment_mask                 = "GL_AMD_shader_fragment_mask";
const char* const E_GL_AMD_gpu_shader_half_float_fetch          = "GL_AMD_gpu_shader_half_float_fetch";
const char* const E_GL_AMD_shader_early_and_late_fragment_tests = "GL_AMD_shader_early_and_late_fragment_tests";

const char* const E_GL_INTEL_shader_integer_functions2          = "GL_INTEL_shader_integer_functions2";

const char* const E_GL_NV_sample_mask_override_coverage         = "GL_NV_sample_mask_override_coverage";
const char* const E_SPV_NV_geometry_shader_passthrough          = "GL_NV_geometry_shader_passthrough";
const char* const E_GL_NV_viewport_array2                       = "GL_NV_viewport_array2";
const char* const E_GL_NV_stereo_view_rendering                 = "GL_NV_stereo_view_rendering";
const char* const E_GL_NVX_multiview_per_view_attributes        = "GL_NVX_multiview_per_view_attributes";
const char* const E_GL_NV_shader_atomic_int64                   = "GL_NV_shader_atomic_int64";
const char* const E_GL_NV_conservative_raster_underestimation   = "GL_NV_conservative_raster_underestimation";
const char* const E_GL_NV_shader_noperspective_interpolation    = "GL_NV_shader_noperspective_interpolation";
const char* const E_GL_NV_shader_subgroup_partitioned           = "GL_NV_shader_subgroup_partitioned";
const char* const E_GL_NV_shading_rate_image                    = "GL_NV_shading_rate_image";
const char* const E_GL_NV_ray_tracing                           = "GL_NV_ray_tracing";
const char* const E_GL_NV_ray_tracing_motion_blur               = "GL_NV_ray_tracing_motion_blur";
const char* const E_GL_NV_fragment_shader_barycentric           = "GL_NV_fragment_shader_barycentric";
const char* const E_GL_NV_compute_shader_derivatives            = "GL_NV_compute_shader_derivatives";
const char* const E_GL_NV_shader_texture_footprint              = "GL_NV_shader_texture_footprint";
const char* const E_GL_NV_mesh_shader                           = "GL_NV_mesh_shader";
const char* const E_GL_EXT_ray_tracing_position_fetch           = "GL_EXT_ray_tracing_position_fetch";

// ARM
const char* const E_GL_ARM_shader_core_builtins                 = "GL_ARM_shader_core_builtins";

// Arrays of extensions for the above viewportEXTs duplications

const char* const viewportEXTs[] = { E_GL_ARB_shader_viewport_layer_array, E_GL_NV_viewport_array2 };
const int Num_viewportEXTs = sizeof(viewportEXTs) / sizeof(viewportEXTs[0]);

const char* const E_GL_NV_cooperative_matrix                    = "GL_NV_cooperative_matrix";
const char* const E_GL_NV_shader_sm_builtins                    = "GL_NV_shader_sm_builtins";
const char* const E_GL_NV_integer_cooperative_matrix            = "GL_NV_integer_cooperative_matrix";
const char* const E_GL_NV_shader_invocation_reorder             = "GL_NV_shader_invocation_reorder";

// AEP
const char* const E_GL_ANDROID_extension_pack_es31a             = "GL_ANDROID_extension_pack_es31a";
const char* const E_GL_KHR_blend_equation_advanced              = "GL_KHR_blend_equation_advanced";
const char* const E_GL_OES_sample_variables                     = "GL_OES_sample_variables";
const char* const E_GL_OES_shader_image_atomic                  = "GL_OES_shader_image_atomic";
const char* const E_GL_OES_shader_multisample_interpolation     = "GL_OES_shader_multisample_interpolation";
const char* const E_GL_OES_texture_storage_multisample_2d_array = "GL_OES_texture_storage_multisample_2d_array";
const char* const E_GL_EXT_geometry_shader                      = "GL_EXT_geometry_shader";
const char* const E_GL_EXT_geometry_point_size                  = "GL_EXT_geometry_point_size";
const char* const E_GL_EXT_gpu_shader5                          = "GL_EXT_gpu_shader5";
const char* const E_GL_EXT_primitive_bounding_box               = "GL_EXT_primitive_bounding_box";
const char* const E_GL_EXT_shader_io_blocks                     = "GL_EXT_shader_io_blocks";
const char* const E_GL_EXT_tessellation_shader                  = "GL_EXT_tessellation_shader";
const char* const E_GL_EXT_tessellation_point_size              = "GL_EXT_tessellation_point_size";
const char* const E_GL_EXT_texture_buffer                       = "GL_EXT_texture_buffer";
const char* const E_GL_EXT_texture_cube_map_array               = "GL_EXT_texture_cube_map_array";
const char* const E_GL_EXT_shader_integer_mix                   = "GL_EXT_shader_integer_mix";

// OES matching AEP
const char* const E_GL_OES_geometry_shader                      = "GL_OES_geometry_shader";
const char* const E_GL_OES_geometry_point_size                  = "GL_OES_geometry_point_size";
const char* const E_GL_OES_gpu_shader5                          = "GL_OES_gpu_shader5";
const char* const E_GL_OES_primitive_bounding_box               = "GL_OES_primitive_bounding_box";
const char* const E_GL_OES_shader_io_blocks                     = "GL_OES_shader_io_blocks";
const char* const E_GL_OES_tessellation_shader                  = "GL_OES_tessellation_shader";
const char* const E_GL_OES_tessellation_point_size              = "GL_OES_tessellation_point_size";
const char* const E_GL_OES_texture_buffer                       = "GL_OES_texture_buffer";
const char* const E_GL_OES_texture_cube_map_array               = "GL_OES_texture_cube_map_array";

// EXT
const char* const E_GL_EXT_shader_explicit_arithmetic_types          = "GL_EXT_shader_explicit_arithmetic_types";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_int8     = "GL_EXT_shader_explicit_arithmetic_types_int8";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_int16    = "GL_EXT_shader_explicit_arithmetic_types_int16";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_int32    = "GL_EXT_shader_explicit_arithmetic_types_int32";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_int64    = "GL_EXT_shader_explicit_arithmetic_types_int64";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_float16  = "GL_EXT_shader_explicit_arithmetic_types_float16";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_float32  = "GL_EXT_shader_explicit_arithmetic_types_float32";
const char* const E_GL_EXT_shader_explicit_arithmetic_types_float64  = "GL_EXT_shader_explicit_arithmetic_types_float64";

const char* const E_GL_EXT_shader_subgroup_extended_types_int8    = "GL_EXT_shader_subgroup_extended_types_int8";
const char* const E_GL_EXT_shader_subgroup_extended_types_int16   = "GL_EXT_shader_subgroup_extended_types_int16";
const char* const E_GL_EXT_shader_subgroup_extended_types_int64   = "GL_EXT_shader_subgroup_extended_types_int64";
const char* const E_GL_EXT_shader_subgroup_extended_types_float16 = "GL_EXT_shader_subgroup_extended_types_float16";
const char* const E_GL_EXT_terminate_invocation = "GL_EXT_terminate_invocation";

const char* const E_GL_EXT_shader_atomic_float = "GL_EXT_shader_atomic_float";
const char* const E_GL_EXT_shader_atomic_float2 = "GL_EXT_shader_atomic_float2";

const char* const E_GL_EXT_shader_tile_image = "GL_EXT_shader_tile_image";

// Arrays of extensions for the above AEP duplications

const char* const AEP_geometry_shader[] = { E_GL_EXT_geometry_shader, E_GL_OES_geometry_shader };
const int Num_AEP_geometry_shader = sizeof(AEP_geometry_shader)/sizeof(AEP_geometry_shader[0]);

const char* const AEP_geometry_point_size[] = { E_GL_EXT_geometry_point_size, E_GL_OES_geometry_point_size };
const int Num_AEP_geometry_point_size = sizeof(AEP_geometry_point_size)/sizeof(AEP_geometry_point_size[0]);

const char* const AEP_gpu_shader5[] = { E_GL_EXT_gpu_shader5, E_GL_OES_gpu_shader5 };
const int Num_AEP_gpu_shader5 = sizeof(AEP_gpu_shader5)/sizeof(AEP_gpu_shader5[0]);

const char* const AEP_primitive_bounding_box[] = { E_GL_EXT_primitive_bounding_box, E_GL_OES_primitive_bounding_box };
const int Num_AEP_primitive_bounding_box = sizeof(AEP_primitive_bounding_box)/sizeof(AEP_primitive_bounding_box[0]);

const char* const AEP_shader_io_blocks[] = { E_GL_EXT_shader_io_blocks, E_GL_OES_shader_io_blocks };
const int Num_AEP_shader_io_blocks = sizeof(AEP_shader_io_blocks)/sizeof(AEP_shader_io_blocks[0]);

const char* const AEP_tessellation_shader[] = { E_GL_EXT_tessellation_shader, E_GL_OES_tessellation_shader };
const int Num_AEP_tessellation_shader = sizeof(AEP_tessellation_shader)/sizeof(AEP_tessellation_shader[0]);

const char* const AEP_tessellation_point_size[] = { E_GL_EXT_tessellation_point_size, E_GL_OES_tessellation_point_size };
const int Num_AEP_tessellation_point_size = sizeof(AEP_tessellation_point_size)/sizeof(AEP_tessellation_point_size[0]);

const char* const AEP_texture_buffer[] = { E_GL_EXT_texture_buffer, E_GL_OES_texture_buffer };
const int Num_AEP_texture_buffer = sizeof(AEP_texture_buffer)/sizeof(AEP_texture_buffer[0]);

const char* const AEP_texture_cube_map_array[] = { E_GL_EXT_texture_cube_map_array, E_GL_OES_texture_cube_map_array };
const int Num_AEP_texture_cube_map_array = sizeof(AEP_texture_cube_map_array)/sizeof(AEP_texture_cube_map_array[0]);

const char* const AEP_mesh_shader[] = { E_GL_NV_mesh_shader, E_GL_EXT_mesh_shader };
const int Num_AEP_mesh_shader = sizeof(AEP_mesh_shader)/sizeof(AEP_mesh_shader[0]);

} // end namespace glslang

#endif // _VERSIONS_INCLUDED_
