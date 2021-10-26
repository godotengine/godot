//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2015-2020 Google, Inc.
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

//
// Help manage multiple profiles, versions, extensions etc.
//
// These don't return error codes, as the presumption is parsing will
// always continue as if the tested feature were enabled, and thus there
// is no error recovery needed.
//

//
// HOW TO add a feature enabled by an extension.
//
// To add a new hypothetical "Feature F" to the front end, where an extension
// "XXX_extension_X" can be used to enable the feature, do the following.
//
// OVERVIEW: Specific features are what are error-checked for, not
//    extensions:  A specific Feature F might be enabled by an extension, or a
//    particular version in a particular profile, or a stage, or combinations, etc.
//
//    The basic mechanism is to use the following to "declare" all the things that
//    enable/disable Feature F, in a code path that implements Feature F:
//
//        requireProfile()
//        profileRequires()
//        requireStage()
//        checkDeprecated()
//        requireNotRemoved()
//        requireExtensions()
//        extensionRequires()
//
//    Typically, only the first two calls are needed.  They go into a code path that
//    implements Feature F, and will log the proper error/warning messages.  Parsing
//    will then always continue as if the tested feature was enabled.
//
//    There is typically no if-testing or conditional parsing, just insertion of the calls above.
//    However, if symbols specific to the extension are added (step 5), they will
//    only be added under tests that the minimum version and profile are present.
//
// 1) Add a symbol name for the extension string at the bottom of Versions.h:
//
//     const char* const XXX_extension_X = "XXX_extension_X";
//
// 2) Add extension initialization to TParseVersions::initializeExtensionBehavior(),
//    the first function below and optionally a entry to extensionData for additional
//    error checks:
//
//     extensionBehavior[XXX_extension_X] = EBhDisable;
//     (Optional) exts[] = {XXX_extension_X, EShTargetSpv_1_4}
//
// 3) Add any preprocessor directives etc. in the next function, TParseVersions::getPreamble():
//
//           "#define XXX_extension_X 1\n"
//
//    The new-line is important, as that ends preprocess tokens.
//
// 4) Insert a profile check in the feature's path (unless all profiles support the feature,
//    for some version level).  That is, call requireProfile() to constrain the profiles, e.g.:
//
//         // ... in a path specific to Feature F...
//         requireProfile(loc,
//                        ECoreProfile | ECompatibilityProfile,
//                        "Feature F");
//
// 5) For each profile that supports the feature, insert version/extension checks:
//
//    The mostly likely scenario is that Feature F can only be used with a
//    particular profile if XXX_extension_X is present or the version is
//    high enough that the core specification already incorporated it.
//
//        // following the requireProfile() call...
//        profileRequires(loc,
//                        ECoreProfile | ECompatibilityProfile,
//                        420,             // 0 if no version incorporated the feature into the core spec.
//                        XXX_extension_X, // can be a list of extensions that all add the feature
//                        "Feature F Description");
//
//    This allows the feature if either A) one of the extensions is enabled or
//    B) the version is high enough.  If no version yet incorporates the feature
//    into core, pass in 0.
//
//    This can be called multiple times, if different profiles support the
//    feature starting at different version numbers or with different
//    extensions.
//
//    This must be called for each profile allowed by the initial call to requireProfile().
//
//    Profiles are all masks, which can be "or"-ed together.
//
//        ENoProfile
//        ECoreProfile
//        ECompatibilityProfile
//        EEsProfile
//
//    The ENoProfile profile is only for desktop, before profiles showed up in version 150;
//    All other #version with no profile default to either es or core, and so have profiles.
//
//    You can select all but a particular profile using ~.  The following basically means "desktop":
//
//        ~EEsProfile
//
// 6) If built-in symbols are added by the extension, add them in Initialize.cpp:  Their use
//    will be automatically error checked against the extensions enabled at that moment.
//    see the comment at the top of Initialize.cpp for where to put them.  Establish them at
//    the earliest release that supports the extension.  Then, tag them with the
//    set of extensions that both enable them and are necessary, given the version of the symbol
//    table. (There is a different symbol table for each version.)
//
// 7) If the extension has additional requirements like minimum SPIR-V version required, add them
//    to extensionRequires()

#include "parseVersions.h"
#include "localintermediate.h"

namespace glslang {

#ifndef GLSLANG_WEB

//
// Initialize all extensions, almost always to 'disable', as once their features
// are incorporated into a core version, their features are supported through allowing that
// core version, not through a pseudo-enablement of the extension.
//
void TParseVersions::initializeExtensionBehavior()
{
    typedef struct {
        const char *const extensionName;
        EShTargetLanguageVersion minSpvVersion;
    } extensionData;

    const extensionData exts[] = { {E_GL_EXT_ray_tracing, EShTargetSpv_1_4},
                                   {E_GL_NV_ray_tracing_motion_blur, EShTargetSpv_1_4}
                                 };

    for (size_t ii = 0; ii < sizeof(exts) / sizeof(exts[0]); ii++) {
        // Add only extensions which require > spv1.0 to save space in map
        if (exts[ii].minSpvVersion > EShTargetSpv_1_0) {
            extensionMinSpv[exts[ii].extensionName] = exts[ii].minSpvVersion;
        }
    }

    extensionBehavior[E_GL_OES_texture_3D]                   = EBhDisable;
    extensionBehavior[E_GL_OES_standard_derivatives]         = EBhDisable;
    extensionBehavior[E_GL_EXT_frag_depth]                   = EBhDisable;
    extensionBehavior[E_GL_OES_EGL_image_external]           = EBhDisable;
    extensionBehavior[E_GL_OES_EGL_image_external_essl3]     = EBhDisable;
    extensionBehavior[E_GL_EXT_YUV_target]                   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_texture_lod]           = EBhDisable;
    extensionBehavior[E_GL_EXT_shadow_samplers]              = EBhDisable;
    extensionBehavior[E_GL_ARB_texture_rectangle]            = EBhDisable;
    extensionBehavior[E_GL_3DL_array_objects]                = EBhDisable;
    extensionBehavior[E_GL_ARB_shading_language_420pack]     = EBhDisable;
    extensionBehavior[E_GL_ARB_texture_gather]               = EBhDisable;
    extensionBehavior[E_GL_ARB_gpu_shader5]                  = EBhDisablePartial;
    extensionBehavior[E_GL_ARB_separate_shader_objects]      = EBhDisable;
    extensionBehavior[E_GL_ARB_compute_shader]               = EBhDisable;
    extensionBehavior[E_GL_ARB_tessellation_shader]          = EBhDisable;
    extensionBehavior[E_GL_ARB_enhanced_layouts]             = EBhDisable;
    extensionBehavior[E_GL_ARB_texture_cube_map_array]       = EBhDisable;
    extensionBehavior[E_GL_ARB_texture_multisample]          = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_texture_lod]           = EBhDisable;
    extensionBehavior[E_GL_ARB_explicit_attrib_location]     = EBhDisable;
    extensionBehavior[E_GL_ARB_explicit_uniform_location]    = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_image_load_store]      = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_atomic_counters]       = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_atomic_counter_ops]    = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_draw_parameters]       = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_group_vote]            = EBhDisable;
    extensionBehavior[E_GL_ARB_derivative_control]           = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_texture_image_samples] = EBhDisable;
    extensionBehavior[E_GL_ARB_viewport_array]               = EBhDisable;
    extensionBehavior[E_GL_ARB_gpu_shader_int64]             = EBhDisable;
    extensionBehavior[E_GL_ARB_gpu_shader_fp64]              = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_ballot]                = EBhDisable;
    extensionBehavior[E_GL_ARB_sparse_texture2]              = EBhDisable;
    extensionBehavior[E_GL_ARB_sparse_texture_clamp]         = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_stencil_export]        = EBhDisable;
//    extensionBehavior[E_GL_ARB_cull_distance]                = EBhDisable;    // present for 4.5, but need extension control over block members
    extensionBehavior[E_GL_ARB_post_depth_coverage]          = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_viewport_layer_array]  = EBhDisable;
    extensionBehavior[E_GL_ARB_fragment_shader_interlock]    = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_clock]                 = EBhDisable;
    extensionBehavior[E_GL_ARB_uniform_buffer_object]        = EBhDisable;
    extensionBehavior[E_GL_ARB_sample_shading]               = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_bit_encoding]          = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_image_size]            = EBhDisable;
    extensionBehavior[E_GL_ARB_shader_storage_buffer_object] = EBhDisable;
    extensionBehavior[E_GL_ARB_shading_language_packing]     = EBhDisable;
    extensionBehavior[E_GL_ARB_texture_query_lod]            = EBhDisable;
    extensionBehavior[E_GL_ARB_vertex_attrib_64bit]          = EBhDisable;

    extensionBehavior[E_GL_KHR_shader_subgroup_basic]            = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_vote]             = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_arithmetic]       = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_ballot]           = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_shuffle]          = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_shuffle_relative] = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_clustered]        = EBhDisable;
    extensionBehavior[E_GL_KHR_shader_subgroup_quad]             = EBhDisable;
    extensionBehavior[E_GL_KHR_memory_scope_semantics]           = EBhDisable;

    extensionBehavior[E_GL_EXT_shader_atomic_int64]              = EBhDisable;

    extensionBehavior[E_GL_EXT_shader_non_constant_global_initializers] = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_image_load_formatted]             = EBhDisable;
    extensionBehavior[E_GL_EXT_post_depth_coverage]                     = EBhDisable;
    extensionBehavior[E_GL_EXT_control_flow_attributes]                 = EBhDisable;
    extensionBehavior[E_GL_EXT_nonuniform_qualifier]                    = EBhDisable;
    extensionBehavior[E_GL_EXT_samplerless_texture_functions]           = EBhDisable;
    extensionBehavior[E_GL_EXT_scalar_block_layout]                     = EBhDisable;
    extensionBehavior[E_GL_EXT_fragment_invocation_density]             = EBhDisable;
    extensionBehavior[E_GL_EXT_buffer_reference]                        = EBhDisable;
    extensionBehavior[E_GL_EXT_buffer_reference2]                       = EBhDisable;
    extensionBehavior[E_GL_EXT_buffer_reference_uvec2]                  = EBhDisable;
    extensionBehavior[E_GL_EXT_demote_to_helper_invocation]             = EBhDisable;
    extensionBehavior[E_GL_EXT_debug_printf]                            = EBhDisable;

    extensionBehavior[E_GL_EXT_shader_16bit_storage]                    = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_8bit_storage]                     = EBhDisable;
    extensionBehavior[E_GL_EXT_subgroup_uniform_control_flow]           = EBhDisable;

    // #line and #include
    extensionBehavior[E_GL_GOOGLE_cpp_style_line_directive]          = EBhDisable;
    extensionBehavior[E_GL_GOOGLE_include_directive]                 = EBhDisable;

    extensionBehavior[E_GL_AMD_shader_ballot]                        = EBhDisable;
    extensionBehavior[E_GL_AMD_shader_trinary_minmax]                = EBhDisable;
    extensionBehavior[E_GL_AMD_shader_explicit_vertex_parameter]     = EBhDisable;
    extensionBehavior[E_GL_AMD_gcn_shader]                           = EBhDisable;
    extensionBehavior[E_GL_AMD_gpu_shader_half_float]                = EBhDisable;
    extensionBehavior[E_GL_AMD_texture_gather_bias_lod]              = EBhDisable;
    extensionBehavior[E_GL_AMD_gpu_shader_int16]                     = EBhDisable;
    extensionBehavior[E_GL_AMD_shader_image_load_store_lod]          = EBhDisable;
    extensionBehavior[E_GL_AMD_shader_fragment_mask]                 = EBhDisable;
    extensionBehavior[E_GL_AMD_gpu_shader_half_float_fetch]          = EBhDisable;

    extensionBehavior[E_GL_INTEL_shader_integer_functions2]          = EBhDisable;

    extensionBehavior[E_GL_NV_sample_mask_override_coverage]         = EBhDisable;
    extensionBehavior[E_SPV_NV_geometry_shader_passthrough]          = EBhDisable;
    extensionBehavior[E_GL_NV_viewport_array2]                       = EBhDisable;
    extensionBehavior[E_GL_NV_stereo_view_rendering]                 = EBhDisable;
    extensionBehavior[E_GL_NVX_multiview_per_view_attributes]        = EBhDisable;
    extensionBehavior[E_GL_NV_shader_atomic_int64]                   = EBhDisable;
    extensionBehavior[E_GL_NV_conservative_raster_underestimation]   = EBhDisable;
    extensionBehavior[E_GL_NV_shader_noperspective_interpolation]    = EBhDisable;
    extensionBehavior[E_GL_NV_shader_subgroup_partitioned]           = EBhDisable;
    extensionBehavior[E_GL_NV_shading_rate_image]                    = EBhDisable;
    extensionBehavior[E_GL_NV_ray_tracing]                           = EBhDisable;
    extensionBehavior[E_GL_NV_ray_tracing_motion_blur]               = EBhDisable;
    extensionBehavior[E_GL_NV_fragment_shader_barycentric]           = EBhDisable;
    extensionBehavior[E_GL_NV_compute_shader_derivatives]            = EBhDisable;
    extensionBehavior[E_GL_NV_shader_texture_footprint]              = EBhDisable;
    extensionBehavior[E_GL_NV_mesh_shader]                           = EBhDisable;

    extensionBehavior[E_GL_NV_cooperative_matrix]                    = EBhDisable;
    extensionBehavior[E_GL_NV_shader_sm_builtins]                    = EBhDisable;
    extensionBehavior[E_GL_NV_integer_cooperative_matrix]            = EBhDisable;

    // AEP
    extensionBehavior[E_GL_ANDROID_extension_pack_es31a]             = EBhDisable;
    extensionBehavior[E_GL_KHR_blend_equation_advanced]              = EBhDisable;
    extensionBehavior[E_GL_OES_sample_variables]                     = EBhDisable;
    extensionBehavior[E_GL_OES_shader_image_atomic]                  = EBhDisable;
    extensionBehavior[E_GL_OES_shader_multisample_interpolation]     = EBhDisable;
    extensionBehavior[E_GL_OES_texture_storage_multisample_2d_array] = EBhDisable;
    extensionBehavior[E_GL_EXT_geometry_shader]                      = EBhDisable;
    extensionBehavior[E_GL_EXT_geometry_point_size]                  = EBhDisable;
    extensionBehavior[E_GL_EXT_gpu_shader5]                          = EBhDisable;
    extensionBehavior[E_GL_EXT_primitive_bounding_box]               = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_io_blocks]                     = EBhDisable;
    extensionBehavior[E_GL_EXT_tessellation_shader]                  = EBhDisable;
    extensionBehavior[E_GL_EXT_tessellation_point_size]              = EBhDisable;
    extensionBehavior[E_GL_EXT_texture_buffer]                       = EBhDisable;
    extensionBehavior[E_GL_EXT_texture_cube_map_array]               = EBhDisable;
    extensionBehavior[E_GL_EXT_null_initializer]                     = EBhDisable;

    // OES matching AEP
    extensionBehavior[E_GL_OES_geometry_shader]          = EBhDisable;
    extensionBehavior[E_GL_OES_geometry_point_size]      = EBhDisable;
    extensionBehavior[E_GL_OES_gpu_shader5]              = EBhDisable;
    extensionBehavior[E_GL_OES_primitive_bounding_box]   = EBhDisable;
    extensionBehavior[E_GL_OES_shader_io_blocks]         = EBhDisable;
    extensionBehavior[E_GL_OES_tessellation_shader]      = EBhDisable;
    extensionBehavior[E_GL_OES_tessellation_point_size]  = EBhDisable;
    extensionBehavior[E_GL_OES_texture_buffer]           = EBhDisable;
    extensionBehavior[E_GL_OES_texture_cube_map_array]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_integer_mix]       = EBhDisable;

    // EXT extensions
    extensionBehavior[E_GL_EXT_device_group]                = EBhDisable;
    extensionBehavior[E_GL_EXT_multiview]                   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_realtime_clock]       = EBhDisable;
    extensionBehavior[E_GL_EXT_ray_tracing]                 = EBhDisable;
    extensionBehavior[E_GL_EXT_ray_query]                   = EBhDisable;
    extensionBehavior[E_GL_EXT_ray_flags_primitive_culling] = EBhDisable;
    extensionBehavior[E_GL_EXT_blend_func_extended]         = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_implicit_conversions] = EBhDisable;
    extensionBehavior[E_GL_EXT_fragment_shading_rate]       = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_image_int64]   = EBhDisable;
    extensionBehavior[E_GL_EXT_terminate_invocation]        = EBhDisable;
    extensionBehavior[E_GL_EXT_shared_memory_block]         = EBhDisable;
    extensionBehavior[E_GL_EXT_spirv_intrinsics]            = EBhDisable;

    // OVR extensions
    extensionBehavior[E_GL_OVR_multiview]                = EBhDisable;
    extensionBehavior[E_GL_OVR_multiview2]               = EBhDisable;

    // explicit types
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types]         = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_int8]    = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_int16]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_int32]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_int64]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_float16] = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_float32] = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_explicit_arithmetic_types_float64] = EBhDisable;

    // subgroup extended types
    extensionBehavior[E_GL_EXT_shader_subgroup_extended_types_int8]    = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_subgroup_extended_types_int16]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_subgroup_extended_types_int64]   = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_subgroup_extended_types_float16] = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_atomic_float]                    = EBhDisable;
    extensionBehavior[E_GL_EXT_shader_atomic_float2]                   = EBhDisable;
}

#endif // GLSLANG_WEB

// Get code that is not part of a shared symbol table, is specific to this shader,
// or needed by the preprocessor (which does not use a shared symbol table).
void TParseVersions::getPreamble(std::string& preamble)
{
    if (isEsProfile()) {
        preamble =
            "#define GL_ES 1\n"
            "#define GL_FRAGMENT_PRECISION_HIGH 1\n"
#ifdef GLSLANG_WEB
            ;
#else
            "#define GL_OES_texture_3D 1\n"
            "#define GL_OES_standard_derivatives 1\n"
            "#define GL_EXT_frag_depth 1\n"
            "#define GL_OES_EGL_image_external 1\n"
            "#define GL_OES_EGL_image_external_essl3 1\n"
            "#define GL_EXT_YUV_target 1\n"
            "#define GL_EXT_shader_texture_lod 1\n"
            "#define GL_EXT_shadow_samplers 1\n"
            "#define GL_EXT_fragment_shading_rate 1\n"

            // AEP
            "#define GL_ANDROID_extension_pack_es31a 1\n"
            "#define GL_OES_sample_variables 1\n"
            "#define GL_OES_shader_image_atomic 1\n"
            "#define GL_OES_shader_multisample_interpolation 1\n"
            "#define GL_OES_texture_storage_multisample_2d_array 1\n"
            "#define GL_EXT_geometry_shader 1\n"
            "#define GL_EXT_geometry_point_size 1\n"
            "#define GL_EXT_gpu_shader5 1\n"
            "#define GL_EXT_primitive_bounding_box 1\n"
            "#define GL_EXT_shader_io_blocks 1\n"
            "#define GL_EXT_tessellation_shader 1\n"
            "#define GL_EXT_tessellation_point_size 1\n"
            "#define GL_EXT_texture_buffer 1\n"
            "#define GL_EXT_texture_cube_map_array 1\n"
            "#define GL_EXT_shader_implicit_conversions 1\n"
            "#define GL_EXT_shader_integer_mix 1\n"
            "#define GL_EXT_blend_func_extended 1\n"

            // OES matching AEP
            "#define GL_OES_geometry_shader 1\n"
            "#define GL_OES_geometry_point_size 1\n"
            "#define GL_OES_gpu_shader5 1\n"
            "#define GL_OES_primitive_bounding_box 1\n"
            "#define GL_OES_shader_io_blocks 1\n"
            "#define GL_OES_tessellation_shader 1\n"
            "#define GL_OES_tessellation_point_size 1\n"
            "#define GL_OES_texture_buffer 1\n"
            "#define GL_OES_texture_cube_map_array 1\n"
            "#define GL_EXT_shader_non_constant_global_initializers 1\n"
            ;

            if (version >= 300) {
                preamble += "#define GL_NV_shader_noperspective_interpolation 1\n";
            }
            if (version >= 310) {
                preamble += "#define GL_EXT_null_initializer 1\n";
                preamble += "#define GL_EXT_subgroup_uniform_control_flow 1\n";
            }

    } else { // !isEsProfile()
        preamble =
            "#define GL_FRAGMENT_PRECISION_HIGH 1\n"
            "#define GL_ARB_texture_rectangle 1\n"
            "#define GL_ARB_shading_language_420pack 1\n"
            "#define GL_ARB_texture_gather 1\n"
            "#define GL_ARB_gpu_shader5 1\n"
            "#define GL_ARB_separate_shader_objects 1\n"
            "#define GL_ARB_compute_shader 1\n"
            "#define GL_ARB_tessellation_shader 1\n"
            "#define GL_ARB_enhanced_layouts 1\n"
            "#define GL_ARB_texture_cube_map_array 1\n"
            "#define GL_ARB_texture_multisample 1\n"
            "#define GL_ARB_shader_texture_lod 1\n"
            "#define GL_ARB_explicit_attrib_location 1\n"
            "#define GL_ARB_explicit_uniform_location 1\n"
            "#define GL_ARB_shader_image_load_store 1\n"
            "#define GL_ARB_shader_atomic_counters 1\n"
            "#define GL_ARB_shader_draw_parameters 1\n"
            "#define GL_ARB_shader_group_vote 1\n"
            "#define GL_ARB_derivative_control 1\n"
            "#define GL_ARB_shader_texture_image_samples 1\n"
            "#define GL_ARB_viewport_array 1\n"
            "#define GL_ARB_gpu_shader_int64 1\n"
            "#define GL_ARB_gpu_shader_fp64 1\n"
            "#define GL_ARB_shader_ballot 1\n"
            "#define GL_ARB_sparse_texture2 1\n"
            "#define GL_ARB_sparse_texture_clamp 1\n"
            "#define GL_ARB_shader_stencil_export 1\n"
            "#define GL_ARB_sample_shading 1\n"
            "#define GL_ARB_shader_image_size 1\n"
            "#define GL_ARB_shading_language_packing 1\n"
//            "#define GL_ARB_cull_distance 1\n"    // present for 4.5, but need extension control over block members
            "#define GL_ARB_post_depth_coverage 1\n"
            "#define GL_ARB_fragment_shader_interlock 1\n"
            "#define GL_ARB_uniform_buffer_object 1\n"
            "#define GL_ARB_shader_bit_encoding 1\n"
            "#define GL_ARB_shader_storage_buffer_object 1\n"
            "#define GL_ARB_texture_query_lod 1\n"
            "#define GL_ARB_vertex_attrib_64bit 1\n"
            "#define GL_EXT_shader_non_constant_global_initializers 1\n"
            "#define GL_EXT_shader_image_load_formatted 1\n"
            "#define GL_EXT_post_depth_coverage 1\n"
            "#define GL_EXT_control_flow_attributes 1\n"
            "#define GL_EXT_nonuniform_qualifier 1\n"
            "#define GL_EXT_shader_16bit_storage 1\n"
            "#define GL_EXT_shader_8bit_storage 1\n"
            "#define GL_EXT_samplerless_texture_functions 1\n"
            "#define GL_EXT_scalar_block_layout 1\n"
            "#define GL_EXT_fragment_invocation_density 1\n"
            "#define GL_EXT_buffer_reference 1\n"
            "#define GL_EXT_buffer_reference2 1\n"
            "#define GL_EXT_buffer_reference_uvec2 1\n"
            "#define GL_EXT_demote_to_helper_invocation 1\n"
            "#define GL_EXT_debug_printf 1\n"
            "#define GL_EXT_fragment_shading_rate 1\n"
            "#define GL_EXT_shared_memory_block 1\n"

            // GL_KHR_shader_subgroup
            "#define GL_KHR_shader_subgroup_basic 1\n"
            "#define GL_KHR_shader_subgroup_vote 1\n"
            "#define GL_KHR_shader_subgroup_arithmetic 1\n"
            "#define GL_KHR_shader_subgroup_ballot 1\n"
            "#define GL_KHR_shader_subgroup_shuffle 1\n"
            "#define GL_KHR_shader_subgroup_shuffle_relative 1\n"
            "#define GL_KHR_shader_subgroup_clustered 1\n"
            "#define GL_KHR_shader_subgroup_quad 1\n"

            "#define GL_EXT_shader_image_int64 1\n"
            "#define GL_EXT_shader_atomic_int64 1\n"
            "#define GL_EXT_shader_realtime_clock 1\n"
            "#define GL_EXT_ray_tracing 1\n"
            "#define GL_EXT_ray_query 1\n"
            "#define GL_EXT_ray_flags_primitive_culling 1\n"
            "#define GL_EXT_spirv_intrinsics 1\n"

            "#define GL_AMD_shader_ballot 1\n"
            "#define GL_AMD_shader_trinary_minmax 1\n"
            "#define GL_AMD_shader_explicit_vertex_parameter 1\n"
            "#define GL_AMD_gcn_shader 1\n"
            "#define GL_AMD_gpu_shader_half_float 1\n"
            "#define GL_AMD_texture_gather_bias_lod 1\n"
            "#define GL_AMD_gpu_shader_int16 1\n"
            "#define GL_AMD_shader_image_load_store_lod 1\n"
            "#define GL_AMD_shader_fragment_mask 1\n"
            "#define GL_AMD_gpu_shader_half_float_fetch 1\n"

            "#define GL_INTEL_shader_integer_functions2 1\n"

            "#define GL_NV_sample_mask_override_coverage 1\n"
            "#define GL_NV_geometry_shader_passthrough 1\n"
            "#define GL_NV_viewport_array2 1\n"
            "#define GL_NV_shader_atomic_int64 1\n"
            "#define GL_NV_conservative_raster_underestimation 1\n"
            "#define GL_NV_shader_subgroup_partitioned 1\n"
            "#define GL_NV_shading_rate_image 1\n"
            "#define GL_NV_ray_tracing 1\n"
            "#define GL_NV_ray_tracing_motion_blur 1\n"
            "#define GL_NV_fragment_shader_barycentric 1\n"
            "#define GL_NV_compute_shader_derivatives 1\n"
            "#define GL_NV_shader_texture_footprint 1\n"
            "#define GL_NV_mesh_shader 1\n"
            "#define GL_NV_cooperative_matrix 1\n"
            "#define GL_NV_integer_cooperative_matrix 1\n"

            "#define GL_EXT_shader_explicit_arithmetic_types 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_int8 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_int16 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_int32 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_int64 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_float16 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_float32 1\n"
            "#define GL_EXT_shader_explicit_arithmetic_types_float64 1\n"

            "#define GL_EXT_shader_subgroup_extended_types_int8 1\n"
            "#define GL_EXT_shader_subgroup_extended_types_int16 1\n"
            "#define GL_EXT_shader_subgroup_extended_types_int64 1\n"
            "#define GL_EXT_shader_subgroup_extended_types_float16 1\n"

            "#define GL_EXT_shader_atomic_float 1\n"
            "#define GL_EXT_shader_atomic_float2 1\n"
            ;

        if (version >= 150) {
            // define GL_core_profile and GL_compatibility_profile
            preamble += "#define GL_core_profile 1\n";

            if (profile == ECompatibilityProfile)
                preamble += "#define GL_compatibility_profile 1\n";
        }
        if (version >= 140) {
            preamble += "#define GL_EXT_null_initializer 1\n";
            preamble += "#define GL_EXT_subgroup_uniform_control_flow 1\n";
        }
#endif // GLSLANG_WEB
    }

#ifndef GLSLANG_WEB
    if ((!isEsProfile() && version >= 140) ||
        (isEsProfile() && version >= 310)) {
        preamble +=
            "#define GL_EXT_device_group 1\n"
            "#define GL_EXT_multiview 1\n"
            "#define GL_NV_shader_sm_builtins 1\n"
            ;
    }

    if (version >= 300 /* both ES and non-ES */) {
        preamble +=
            "#define GL_OVR_multiview 1\n"
            "#define GL_OVR_multiview2 1\n"
            ;
    }

    // #line and #include
    preamble +=
            "#define GL_GOOGLE_cpp_style_line_directive 1\n"
            "#define GL_GOOGLE_include_directive 1\n"
            "#define GL_KHR_blend_equation_advanced 1\n"
            ;

    // other general extensions
    preamble +=
            "#define GL_EXT_terminate_invocation 1\n"
            ;
#endif

    // #define VULKAN XXXX
    const int numberBufSize = 12;
    char numberBuf[numberBufSize];
    if (spvVersion.vulkanGlsl > 0) {
        preamble += "#define VULKAN ";
        snprintf(numberBuf, numberBufSize, "%d", spvVersion.vulkanGlsl);
        preamble += numberBuf;
        preamble += "\n";
    }

#ifndef GLSLANG_WEB
    // #define GL_SPIRV XXXX
    if (spvVersion.openGl > 0) {
        preamble += "#define GL_SPIRV ";
        snprintf(numberBuf, numberBufSize, "%d", spvVersion.openGl);
        preamble += numberBuf;
        preamble += "\n";
    }
#endif

#ifndef GLSLANG_WEB
    // GL_EXT_spirv_intrinsics
    if (!isEsProfile()) {
        switch (language) {
        case EShLangVertex:         preamble += "#define GL_VERTEX_SHADER 1 \n";                    break;
        case EShLangTessControl:    preamble += "#define GL_TESSELLATION_CONTROL_SHADER 1 \n";      break;
        case EShLangTessEvaluation: preamble += "#define GL_TESSELLATION_EVALUATION_SHADER 1 \n";   break;
        case EShLangGeometry:       preamble += "#define GL_GEOMETRY_SHADER 1 \n";                  break;
        case EShLangFragment:       preamble += "#define GL_FRAGMENT_SHADER 1 \n";                  break;
        case EShLangCompute:        preamble += "#define GL_COMPUTE_SHADER 1 \n";                   break;
        case EShLangRayGen:         preamble += "#define GL_RAY_GENERATION_SHADER_EXT 1 \n";        break;
        case EShLangIntersect:      preamble += "#define GL_INTERSECTION_SHADER_EXT 1 \n";          break;
        case EShLangAnyHit:         preamble += "#define GL_ANY_HIT_SHADER_EXT 1 \n";               break;
        case EShLangClosestHit:     preamble += "#define GL_CLOSEST_HIT_SHADER_EXT 1 \n";           break;
        case EShLangMiss:           preamble += "#define GL_MISS_SHADER_EXT 1 \n";                  break;
        case EShLangCallable:       preamble += "#define GL_CALLABLE_SHADER_EXT 1 \n";              break;
        case EShLangTaskNV:         preamble += "#define GL_TASK_SHADER_NV 1 \n";                   break;
        case EShLangMeshNV:         preamble += "#define GL_MESH_SHADER_NV 1 \n";                   break;
        default:                                                                                    break;
        }
    }
#endif
}

//
// Map from stage enum to externally readable text name.
//
const char* StageName(EShLanguage stage)
{
    switch(stage) {
    case EShLangVertex:         return "vertex";
    case EShLangFragment:       return "fragment";
    case EShLangCompute:        return "compute";
#ifndef GLSLANG_WEB
    case EShLangTessControl:    return "tessellation control";
    case EShLangTessEvaluation: return "tessellation evaluation";
    case EShLangGeometry:       return "geometry";
    case EShLangRayGen:         return "ray-generation";
    case EShLangIntersect:      return "intersection";
    case EShLangAnyHit:         return "any-hit";
    case EShLangClosestHit:     return "closest-hit";
    case EShLangMiss:           return "miss";
    case EShLangCallable:       return "callable";
    case EShLangMeshNV:         return "mesh";
    case EShLangTaskNV:         return "task";
#endif
    default:                    return "unknown stage";
    }
}

//
// When to use requireStage()
//
//     If only some stages support a feature.
//
// Operation: If the current stage is not present, give an error message.
//
void TParseVersions::requireStage(const TSourceLoc& loc, EShLanguageMask languageMask, const char* featureDesc)
{
    if (((1 << language) & languageMask) == 0)
        error(loc, "not supported in this stage:", featureDesc, StageName(language));
}

// If only one stage supports a feature, this can be called.  But, all supporting stages
// must be specified with one call.
void TParseVersions::requireStage(const TSourceLoc& loc, EShLanguage stage, const char* featureDesc)
{
    requireStage(loc, static_cast<EShLanguageMask>(1 << stage), featureDesc);
}

#ifndef GLSLANG_WEB
//
// When to use requireProfile():
//
//     Use if only some profiles support a feature.  However, if within a profile the feature
//     is version or extension specific, follow this call with calls to profileRequires().
//
// Operation:  If the current profile is not one of the profileMask,
// give an error message.
//
void TParseVersions::requireProfile(const TSourceLoc& loc, int profileMask, const char* featureDesc)
{
    if (! (profile & profileMask))
        error(loc, "not supported with this profile:", featureDesc, ProfileName(profile));
}

//
// When to use profileRequires():
//
//     If a set of profiles have the same requirements for what version or extensions
//     are needed to support a feature.
//
//     It must be called for each profile that needs protection.  Use requireProfile() first
//     to reduce that set of profiles.
//
// Operation: Will issue warnings/errors based on the current profile, version, and extension
// behaviors.  It only checks extensions when the current profile is one of the profileMask.
//
// A minVersion of 0 means no version of the profileMask support this in core,
// the extension must be present.
//

// entry point that takes multiple extensions
void TParseVersions::profileRequires(const TSourceLoc& loc, int profileMask, int minVersion, int numExtensions,
    const char* const extensions[], const char* featureDesc)
{
    if (profile & profileMask) {
        bool okay = minVersion > 0 && version >= minVersion;
#ifndef GLSLANG_WEB
        for (int i = 0; i < numExtensions; ++i) {
            switch (getExtensionBehavior(extensions[i])) {
            case EBhWarn:
                infoSink.info.message(EPrefixWarning, ("extension " + TString(extensions[i]) + " is being used for " + featureDesc).c_str(), loc);
                // fall through
            case EBhRequire:
            case EBhEnable:
                okay = true;
                break;
            default: break; // some compilers want this
            }
        }
#endif
        if (! okay)
            error(loc, "not supported for this version or the enabled extensions", featureDesc, "");
    }
}

// entry point for the above that takes a single extension
void TParseVersions::profileRequires(const TSourceLoc& loc, int profileMask, int minVersion, const char* extension,
    const char* featureDesc)
{
    profileRequires(loc, profileMask, minVersion, extension ? 1 : 0, &extension, featureDesc);
}

void TParseVersions::unimplemented(const TSourceLoc& loc, const char* featureDesc)
{
    error(loc, "feature not yet implemented", featureDesc, "");
}

//
// Within a set of profiles, see if a feature is deprecated and give an error or warning based on whether
// a future compatibility context is being use.
//
void TParseVersions::checkDeprecated(const TSourceLoc& loc, int profileMask, int depVersion, const char* featureDesc)
{
    if (profile & profileMask) {
        if (version >= depVersion) {
            if (forwardCompatible)
                error(loc, "deprecated, may be removed in future release", featureDesc, "");
            else if (! suppressWarnings())
                infoSink.info.message(EPrefixWarning, (TString(featureDesc) + " deprecated in version " +
                                                       String(depVersion) + "; may be removed in future release").c_str(), loc);
        }
    }
}

//
// Within a set of profiles, see if a feature has now been removed and if so, give an error.
// The version argument is the first version no longer having the feature.
//
void TParseVersions::requireNotRemoved(const TSourceLoc& loc, int profileMask, int removedVersion, const char* featureDesc)
{
    if (profile & profileMask) {
        if (version >= removedVersion) {
            const int maxSize = 60;
            char buf[maxSize];
            snprintf(buf, maxSize, "%s profile; removed in version %d", ProfileName(profile), removedVersion);
            error(loc, "no longer supported in", featureDesc, buf);
        }
    }
}

// Returns true if at least one of the extensions in the extensions parameter is requested. Otherwise, returns false.
// Warns appropriately if the requested behavior of an extension is "warn".
bool TParseVersions::checkExtensionsRequested(const TSourceLoc& loc, int numExtensions, const char* const extensions[], const char* featureDesc)
{
    // First, see if any of the extensions are enabled
    for (int i = 0; i < numExtensions; ++i) {
        TExtensionBehavior behavior = getExtensionBehavior(extensions[i]);
        if (behavior == EBhEnable || behavior == EBhRequire)
            return true;
    }

    // See if any extensions want to give a warning on use; give warnings for all such extensions
    bool warned = false;
    for (int i = 0; i < numExtensions; ++i) {
        TExtensionBehavior behavior = getExtensionBehavior(extensions[i]);
        if (behavior == EBhDisable && relaxedErrors()) {
            infoSink.info.message(EPrefixWarning, "The following extension must be enabled to use this feature:", loc);
            behavior = EBhWarn;
        }
        if (behavior == EBhWarn) {
            infoSink.info.message(EPrefixWarning, ("extension " + TString(extensions[i]) + " is being used for " + featureDesc).c_str(), loc);
            warned = true;
        }
    }
    if (warned)
        return true;
    return false;
}

//
// Use when there are no profile/version to check, it's just an error if one of the
// extensions is not present.
//
void TParseVersions::requireExtensions(const TSourceLoc& loc, int numExtensions, const char* const extensions[],
    const char* featureDesc)
{
    if (checkExtensionsRequested(loc, numExtensions, extensions, featureDesc))
        return;

    // If we get this far, give errors explaining what extensions are needed
    if (numExtensions == 1)
        error(loc, "required extension not requested:", featureDesc, extensions[0]);
    else {
        error(loc, "required extension not requested:", featureDesc, "Possible extensions include:");
        for (int i = 0; i < numExtensions; ++i)
            infoSink.info.message(EPrefixNone, extensions[i]);
    }
}

//
// Use by preprocessor when there are no profile/version to check, it's just an error if one of the
// extensions is not present.
//
void TParseVersions::ppRequireExtensions(const TSourceLoc& loc, int numExtensions, const char* const extensions[],
    const char* featureDesc)
{
    if (checkExtensionsRequested(loc, numExtensions, extensions, featureDesc))
        return;

    // If we get this far, give errors explaining what extensions are needed
    if (numExtensions == 1)
        ppError(loc, "required extension not requested:", featureDesc, extensions[0]);
    else {
        ppError(loc, "required extension not requested:", featureDesc, "Possible extensions include:");
        for (int i = 0; i < numExtensions; ++i)
            infoSink.info.message(EPrefixNone, extensions[i]);
    }
}

TExtensionBehavior TParseVersions::getExtensionBehavior(const char* extension)
{
    auto iter = extensionBehavior.find(TString(extension));
    if (iter == extensionBehavior.end())
        return EBhMissing;
    else
        return iter->second;
}

// Returns true if the given extension is set to enable, require, or warn.
bool TParseVersions::extensionTurnedOn(const char* const extension)
{
      switch (getExtensionBehavior(extension)) {
      case EBhEnable:
      case EBhRequire:
      case EBhWarn:
          return true;
      default:
          break;
      }
      return false;
}
// See if any of the extensions are set to enable, require, or warn.
bool TParseVersions::extensionsTurnedOn(int numExtensions, const char* const extensions[])
{
    for (int i = 0; i < numExtensions; ++i) {
        if (extensionTurnedOn(extensions[i]))
            return true;
    }
    return false;
}

//
// Change the current state of an extension's behavior.
//
void TParseVersions::updateExtensionBehavior(int line, const char* extension, const char* behaviorString)
{
    // Translate from text string of extension's behavior to an enum.
    TExtensionBehavior behavior = EBhDisable;
    if (! strcmp("require", behaviorString))
        behavior = EBhRequire;
    else if (! strcmp("enable", behaviorString))
        behavior = EBhEnable;
    else if (! strcmp("disable", behaviorString))
        behavior = EBhDisable;
    else if (! strcmp("warn", behaviorString))
        behavior = EBhWarn;
    else {
        error(getCurrentLoc(), "behavior not supported:", "#extension", behaviorString);
        return;
    }
    bool on = behavior != EBhDisable;

    // check if extension is used with correct shader stage
    checkExtensionStage(getCurrentLoc(), extension);

    // check if extension has additional requirements
    extensionRequires(getCurrentLoc(), extension, behaviorString);

    // update the requested extension
    updateExtensionBehavior(extension, behavior);

    // see if need to propagate to implicitly modified things
    if (strcmp(extension, "GL_ANDROID_extension_pack_es31a") == 0) {
        // to everything in AEP
        updateExtensionBehavior(line, "GL_KHR_blend_equation_advanced", behaviorString);
        updateExtensionBehavior(line, "GL_OES_sample_variables", behaviorString);
        updateExtensionBehavior(line, "GL_OES_shader_image_atomic", behaviorString);
        updateExtensionBehavior(line, "GL_OES_shader_multisample_interpolation", behaviorString);
        updateExtensionBehavior(line, "GL_OES_texture_storage_multisample_2d_array", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_geometry_shader", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_gpu_shader5", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_primitive_bounding_box", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_shader_io_blocks", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_tessellation_shader", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_texture_buffer", behaviorString);
        updateExtensionBehavior(line, "GL_EXT_texture_cube_map_array", behaviorString);
    }
    // geometry to io_blocks
    else if (strcmp(extension, "GL_EXT_geometry_shader") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_io_blocks", behaviorString);
    else if (strcmp(extension, "GL_OES_geometry_shader") == 0)
        updateExtensionBehavior(line, "GL_OES_shader_io_blocks", behaviorString);
    // tessellation to io_blocks
    else if (strcmp(extension, "GL_EXT_tessellation_shader") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_io_blocks", behaviorString);
    else if (strcmp(extension, "GL_OES_tessellation_shader") == 0)
        updateExtensionBehavior(line, "GL_OES_shader_io_blocks", behaviorString);
    else if (strcmp(extension, "GL_GOOGLE_include_directive") == 0)
        updateExtensionBehavior(line, "GL_GOOGLE_cpp_style_line_directive", behaviorString);
    // subgroup_* to subgroup_basic
    else if (strcmp(extension, "GL_KHR_shader_subgroup_vote") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_arithmetic") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_ballot") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_shuffle") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_shuffle_relative") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_clustered") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_KHR_shader_subgroup_quad") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_NV_shader_subgroup_partitioned") == 0)
        updateExtensionBehavior(line, "GL_KHR_shader_subgroup_basic", behaviorString);
    else if (strcmp(extension, "GL_EXT_buffer_reference2") == 0 ||
             strcmp(extension, "GL_EXT_buffer_reference_uvec2") == 0)
        updateExtensionBehavior(line, "GL_EXT_buffer_reference", behaviorString);
    else if (strcmp(extension, "GL_NV_integer_cooperative_matrix") == 0)
        updateExtensionBehavior(line, "GL_NV_cooperative_matrix", behaviorString);
    // subgroup extended types to explicit types
    else if (strcmp(extension, "GL_EXT_shader_subgroup_extended_types_int8") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_explicit_arithmetic_types_int8", behaviorString);
    else if (strcmp(extension, "GL_EXT_shader_subgroup_extended_types_int16") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_explicit_arithmetic_types_int16", behaviorString);
    else if (strcmp(extension, "GL_EXT_shader_subgroup_extended_types_int64") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_explicit_arithmetic_types_int64", behaviorString);
    else if (strcmp(extension, "GL_EXT_shader_subgroup_extended_types_float16") == 0)
        updateExtensionBehavior(line, "GL_EXT_shader_explicit_arithmetic_types_float16", behaviorString);

    // see if we need to update the numeric features
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_int8") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_int8, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_int16") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_int16, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_int32") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_int32, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_int64") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_int64, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_float16") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_float16, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_float32") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_float32, on);
    else if (strcmp(extension, "GL_EXT_shader_explicit_arithmetic_types_float64") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_explicit_arithmetic_types_float64, on);
    else if (strcmp(extension, "GL_EXT_shader_implicit_conversions") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::shader_implicit_conversions, on);
    else if (strcmp(extension, "GL_ARB_gpu_shader_fp64") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::gpu_shader_fp64, on);
    else if (strcmp(extension, "GL_AMD_gpu_shader_int16") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::gpu_shader_int16, on);
    else if (strcmp(extension, "GL_AMD_gpu_shader_half_float") == 0)
        intermediate.updateNumericFeature(TNumericFeatures::gpu_shader_half_float, on);
}

void TParseVersions::updateExtensionBehavior(const char* extension, TExtensionBehavior behavior)
{
    // Update the current behavior
    if (strcmp(extension, "all") == 0) {
        // special case for the 'all' extension; apply it to every extension present
        if (behavior == EBhRequire || behavior == EBhEnable) {
            error(getCurrentLoc(), "extension 'all' cannot have 'require' or 'enable' behavior", "#extension", "");
            return;
        } else {
            for (auto iter = extensionBehavior.begin(); iter != extensionBehavior.end(); ++iter)
                iter->second = behavior;
        }
    } else {
        // Do the update for this single extension
        auto iter = extensionBehavior.find(TString(extension));
        if (iter == extensionBehavior.end()) {
            switch (behavior) {
            case EBhRequire:
                error(getCurrentLoc(), "extension not supported:", "#extension", extension);
                break;
            case EBhEnable:
            case EBhWarn:
            case EBhDisable:
                warn(getCurrentLoc(), "extension not supported:", "#extension", extension);
                break;
            default:
                assert(0 && "unexpected behavior");
            }

            return;
        } else {
            if (iter->second == EBhDisablePartial)
                warn(getCurrentLoc(), "extension is only partially supported:", "#extension", extension);
            if (behavior != EBhDisable)
                intermediate.addRequestedExtension(extension);
            iter->second = behavior;
        }
    }
}

// Check if extension is used with correct shader stage.
void TParseVersions::checkExtensionStage(const TSourceLoc& loc, const char * const extension)
{
    // GL_NV_mesh_shader extension is only allowed in task/mesh shaders
    if (strcmp(extension, "GL_NV_mesh_shader") == 0) {
        requireStage(loc, (EShLanguageMask)(EShLangTaskNVMask | EShLangMeshNVMask | EShLangFragmentMask),
                     "#extension GL_NV_mesh_shader");
        profileRequires(loc, ECoreProfile, 450, 0, "#extension GL_NV_mesh_shader");
        profileRequires(loc, EEsProfile, 320, 0, "#extension GL_NV_mesh_shader");
    }
}

// Check if extension has additional requirements
void TParseVersions::extensionRequires(const TSourceLoc &loc, const char * const extension, const char *behaviorString)
{
    bool isEnabled = false;
    if (!strcmp("require", behaviorString))
        isEnabled = true;
    else if (!strcmp("enable", behaviorString))
        isEnabled = true;

    if (isEnabled) {
        unsigned int minSpvVersion = 0;
        auto iter = extensionMinSpv.find(TString(extension));
        if (iter != extensionMinSpv.end())
            minSpvVersion = iter->second;
        requireSpv(loc, extension, minSpvVersion);
    }
}

// Call for any operation needing full GLSL integer data-type support.
void TParseVersions::fullIntegerCheck(const TSourceLoc& loc, const char* op)
{
    profileRequires(loc, ENoProfile, 130, nullptr, op);
    profileRequires(loc, EEsProfile, 300, nullptr, op);
}

// Call for any operation needing GLSL double data-type support.
void TParseVersions::doubleCheck(const TSourceLoc& loc, const char* op)
{

    //requireProfile(loc, ECoreProfile | ECompatibilityProfile, op);
    if (language == EShLangVertex) {
        const char* const f64_Extensions[] = {E_GL_ARB_gpu_shader_fp64, E_GL_ARB_vertex_attrib_64bit};
        profileRequires(loc, ECoreProfile | ECompatibilityProfile, 400, 2, f64_Extensions, op);
    } else
        profileRequires(loc, ECoreProfile | ECompatibilityProfile, 400, E_GL_ARB_gpu_shader_fp64, op);
}

// Call for any operation needing GLSL float16 data-type support.
void TParseVersions::float16Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[] = {
                                           E_GL_AMD_gpu_shader_half_float,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_float16};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

bool TParseVersions::float16Arithmetic()
{
    const char* const extensions[] = {
                                       E_GL_AMD_gpu_shader_half_float,
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_float16};
    return extensionsTurnedOn(sizeof(extensions)/sizeof(extensions[0]), extensions);
}

bool TParseVersions::int16Arithmetic()
{
    const char* const extensions[] = {
                                       E_GL_AMD_gpu_shader_int16,
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_int16};
    return extensionsTurnedOn(sizeof(extensions)/sizeof(extensions[0]), extensions);
}

bool TParseVersions::int8Arithmetic()
{
    const char* const extensions[] = {
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_int8};
    return extensionsTurnedOn(sizeof(extensions)/sizeof(extensions[0]), extensions);
}

void TParseVersions::requireFloat16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc)
{
    TString combined;
    combined = op;
    combined += ": ";
    combined += featureDesc;

    const char* const extensions[] = {
                                       E_GL_AMD_gpu_shader_half_float,
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_float16};
    requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, combined.c_str());
}

void TParseVersions::requireInt16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc)
{
    TString combined;
    combined = op;
    combined += ": ";
    combined += featureDesc;

    const char* const extensions[] = {
                                       E_GL_AMD_gpu_shader_int16,
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_int16};
    requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, combined.c_str());
}

void TParseVersions::requireInt8Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc)
{
    TString combined;
    combined = op;
    combined += ": ";
    combined += featureDesc;

    const char* const extensions[] = {
                                       E_GL_EXT_shader_explicit_arithmetic_types,
                                       E_GL_EXT_shader_explicit_arithmetic_types_int8};
    requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, combined.c_str());
}

void TParseVersions::float16ScalarVectorCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[] = {
                                           E_GL_AMD_gpu_shader_half_float,
                                           E_GL_EXT_shader_16bit_storage,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_float16};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

// Call for any operation needing GLSL float32 data-type support.
void TParseVersions::explicitFloat32Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[2] = {E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_float32};
        requireExtensions(loc, 2, extensions, op);
    }
}

// Call for any operation needing GLSL float64 data-type support.
void TParseVersions::explicitFloat64Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[2] = {E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_float64};
        requireExtensions(loc, 2, extensions, op);
        requireProfile(loc, ECoreProfile | ECompatibilityProfile, op);
        profileRequires(loc, ECoreProfile | ECompatibilityProfile, 400, nullptr, op);
    }
}

// Call for any operation needing GLSL explicit int8 data-type support.
void TParseVersions::explicitInt8Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
        const char* const extensions[2] = {E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int8};
        requireExtensions(loc, 2, extensions, op);
    }
}

// Call for any operation needing GLSL float16 opaque-type support
void TParseVersions::float16OpaqueCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
        requireExtensions(loc, 1, &E_GL_AMD_gpu_shader_half_float_fetch, op);
        requireProfile(loc, ECoreProfile | ECompatibilityProfile, op);
        profileRequires(loc, ECoreProfile | ECompatibilityProfile, 400, nullptr, op);
    }
}

// Call for any operation needing GLSL explicit int16 data-type support.
void TParseVersions::explicitInt16Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
        const char* const extensions[] = {
                                           E_GL_AMD_gpu_shader_int16,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int16};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

void TParseVersions::int16ScalarVectorCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
    	const char* const extensions[] = {
                                           E_GL_AMD_gpu_shader_int16,
                                           E_GL_EXT_shader_16bit_storage,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int16};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

void TParseVersions::int8ScalarVectorCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
    	const char* const extensions[] = {
                                           E_GL_EXT_shader_8bit_storage,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int8};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

// Call for any operation needing GLSL explicit int32 data-type support.
void TParseVersions::explicitInt32Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
        const char* const extensions[2] = {E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int32};
        requireExtensions(loc, 2, extensions, op);
    }
}

// Call for any operation needing GLSL 64-bit integer data-type support.
void TParseVersions::int64Check(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (! builtIn) {
        const char* const extensions[3] = {E_GL_ARB_gpu_shader_int64,
                                           E_GL_EXT_shader_explicit_arithmetic_types,
                                           E_GL_EXT_shader_explicit_arithmetic_types_int64};
        requireExtensions(loc, 3, extensions, op);
        requireProfile(loc, ECoreProfile | ECompatibilityProfile, op);
        profileRequires(loc, ECoreProfile | ECompatibilityProfile, 400, nullptr, op);
    }
}

void TParseVersions::fcoopmatCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[] = {E_GL_NV_cooperative_matrix};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}

void TParseVersions::intcoopmatCheck(const TSourceLoc& loc, const char* op, bool builtIn)
{
    if (!builtIn) {
        const char* const extensions[] = {E_GL_NV_integer_cooperative_matrix};
        requireExtensions(loc, sizeof(extensions)/sizeof(extensions[0]), extensions, op);
    }
}
#endif // GLSLANG_WEB
// Call for any operation removed because SPIR-V is in use.
void TParseVersions::spvRemoved(const TSourceLoc& loc, const char* op)
{
    if (spvVersion.spv != 0)
        error(loc, "not allowed when generating SPIR-V", op, "");
}

// Call for any operation removed because Vulkan SPIR-V is being generated.
void TParseVersions::vulkanRemoved(const TSourceLoc& loc, const char* op)
{
    if (spvVersion.vulkan > 0 && !spvVersion.vulkanRelaxed)
        error(loc, "not allowed when using GLSL for Vulkan", op, "");
}

// Call for any operation that requires Vulkan.
void TParseVersions::requireVulkan(const TSourceLoc& loc, const char* op)
{
#ifndef GLSLANG_WEB
    if (spvVersion.vulkan == 0)
        error(loc, "only allowed when using GLSL for Vulkan", op, "");
#endif
}

// Call for any operation that requires SPIR-V.
void TParseVersions::requireSpv(const TSourceLoc& loc, const char* op)
{
#ifndef GLSLANG_WEB
    if (spvVersion.spv == 0)
        error(loc, "only allowed when generating SPIR-V", op, "");
#endif
}
void TParseVersions::requireSpv(const TSourceLoc& loc, const char *op, unsigned int version)
{
#ifndef GLSLANG_WEB
    if (spvVersion.spv < version)
        error(loc, "not supported for current targeted SPIR-V version", op, "");
#endif
}

} // end namespace glslang
