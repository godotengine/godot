/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file consts_exts.h
 * Mesa Constants and GL Extensions data structures.
 */

#ifndef __CONSTS_EXTS_H__
#define __CONSTS_EXTS_H__

#include "util/glheader.h"
#include "compiler/shader_enums.h"
#include "compiler/shader_info.h"

struct nir_shader_compiler_options;

/**
 * Enable flag for each OpenGL extension.  Different device drivers will
 * enable different extensions at runtime.
 */
struct gl_extensions
{
   GLboolean dummy;  /* don't remove this! */
   GLboolean dummy_true;  /* Set true by _mesa_init_extensions(). */
   GLboolean ANGLE_texture_compression_dxt;
   GLboolean ARB_ES2_compatibility;
   GLboolean ARB_ES3_compatibility;
   GLboolean ARB_ES3_1_compatibility;
   GLboolean ARB_ES3_2_compatibility;
   GLboolean ARB_arrays_of_arrays;
   GLboolean ARB_base_instance;
   GLboolean ARB_bindless_texture;
   GLboolean ARB_blend_func_extended;
   GLboolean ARB_buffer_storage;
   GLboolean ARB_clear_texture;
   GLboolean ARB_clip_control;
   GLboolean ARB_color_buffer_float;
   GLboolean ARB_compatibility;
   GLboolean ARB_compute_shader;
   GLboolean ARB_compute_variable_group_size;
   GLboolean ARB_conditional_render_inverted;
   GLboolean ARB_conservative_depth;
   GLboolean ARB_copy_image;
   GLboolean ARB_cull_distance;
   GLboolean EXT_color_buffer_half_float;
   GLboolean ARB_depth_buffer_float;
   GLboolean ARB_depth_clamp;
   GLboolean ARB_derivative_control;
   GLboolean ARB_draw_buffers_blend;
   GLboolean ARB_draw_elements_base_vertex;
   GLboolean ARB_draw_indirect;
   GLboolean ARB_draw_instanced;
   GLboolean ARB_fragment_coord_conventions;
   GLboolean ARB_fragment_layer_viewport;
   GLboolean ARB_fragment_program;
   GLboolean ARB_fragment_program_shadow;
   GLboolean ARB_fragment_shader;
   GLboolean ARB_framebuffer_no_attachments;
   GLboolean ARB_framebuffer_object;
   GLboolean ARB_fragment_shader_interlock;
   GLboolean ARB_enhanced_layouts;
   GLboolean ARB_explicit_attrib_location;
   GLboolean ARB_explicit_uniform_location;
   GLboolean ARB_gl_spirv;
   GLboolean ARB_gpu_shader5;
   GLboolean ARB_gpu_shader_fp64;
   GLboolean ARB_gpu_shader_int64;
   GLboolean ARB_half_float_vertex;
   GLboolean ARB_indirect_parameters;
   GLboolean ARB_instanced_arrays;
   GLboolean ARB_internalformat_query;
   GLboolean ARB_internalformat_query2;
   GLboolean ARB_map_buffer_range;
   GLboolean ARB_occlusion_query;
   GLboolean ARB_occlusion_query2;
   GLboolean ARB_pipeline_statistics_query;
   GLboolean ARB_polygon_offset_clamp;
   GLboolean ARB_post_depth_coverage;
   GLboolean ARB_query_buffer_object;
   GLboolean ARB_robust_buffer_access_behavior;
   GLboolean ARB_sample_locations;
   GLboolean ARB_sample_shading;
   GLboolean ARB_seamless_cube_map;
   GLboolean ARB_shader_atomic_counter_ops;
   GLboolean ARB_shader_atomic_counters;
   GLboolean ARB_shader_ballot;
   GLboolean ARB_shader_bit_encoding;
   GLboolean ARB_shader_clock;
   GLboolean ARB_shader_draw_parameters;
   GLboolean ARB_shader_group_vote;
   GLboolean ARB_shader_image_load_store;
   GLboolean ARB_shader_image_size;
   GLboolean ARB_shader_precision;
   GLboolean ARB_shader_stencil_export;
   GLboolean ARB_shader_storage_buffer_object;
   GLboolean ARB_shader_texture_image_samples;
   GLboolean ARB_shader_texture_lod;
   GLboolean ARB_shader_viewport_layer_array;
   GLboolean ARB_shading_language_packing;
   GLboolean ARB_shading_language_420pack;
   GLboolean ARB_shadow;
   GLboolean ARB_sparse_buffer;
   GLboolean ARB_sparse_texture;
   GLboolean ARB_sparse_texture2;
   GLboolean ARB_sparse_texture_clamp;
   GLboolean ARB_stencil_texturing;
   GLboolean ARB_spirv_extensions;
   GLboolean ARB_sync;
   GLboolean ARB_tessellation_shader;
   GLboolean ARB_texture_buffer_object;
   GLboolean ARB_texture_buffer_object_rgb32;
   GLboolean ARB_texture_buffer_range;
   GLboolean ARB_texture_compression_bptc;
   GLboolean ARB_texture_compression_rgtc;
   GLboolean ARB_texture_cube_map_array;
   GLboolean ARB_texture_filter_anisotropic;
   GLboolean ARB_texture_filter_minmax;
   GLboolean ARB_texture_float;
   GLboolean ARB_texture_gather;
   GLboolean ARB_texture_mirror_clamp_to_edge;
   GLboolean ARB_texture_multisample;
   GLboolean ARB_texture_non_power_of_two;
   GLboolean ARB_texture_stencil8;
   GLboolean ARB_texture_query_levels;
   GLboolean ARB_texture_query_lod;
   GLboolean ARB_texture_rg;
   GLboolean ARB_texture_rgb10_a2ui;
   GLboolean ARB_texture_view;
   GLboolean ARB_timer_query;
   GLboolean ARB_transform_feedback2;
   GLboolean ARB_transform_feedback3;
   GLboolean ARB_transform_feedback_instanced;
   GLboolean ARB_transform_feedback_overflow_query;
   GLboolean ARB_uniform_buffer_object;
   GLboolean ARB_vertex_attrib_64bit;
   GLboolean ARB_vertex_program;
   GLboolean ARB_vertex_shader;
   GLboolean ARB_vertex_type_10f_11f_11f_rev;
   GLboolean ARB_vertex_type_2_10_10_10_rev;
   GLboolean ARB_viewport_array;
   GLboolean EXT_blend_equation_separate;
   GLboolean EXT_demote_to_helper_invocation;
   GLboolean EXT_depth_bounds_test;
   GLboolean EXT_disjoint_timer_query;
   GLboolean EXT_draw_buffers2;
   GLboolean EXT_EGL_image_storage;
   GLboolean EXT_float_blend;
   GLboolean EXT_framebuffer_multisample;
   GLboolean EXT_framebuffer_multisample_blit_scaled;
   GLboolean EXT_framebuffer_sRGB;
   GLboolean EXT_gpu_program_parameters;
   GLboolean EXT_gpu_shader4;
   GLboolean EXT_memory_object;
   GLboolean EXT_memory_object_fd;
   GLboolean EXT_memory_object_win32;
   GLboolean EXT_multisampled_render_to_texture;
   GLboolean EXT_packed_float;
   GLboolean EXT_provoking_vertex;
   GLboolean EXT_render_snorm;
   GLboolean EXT_semaphore;
   GLboolean EXT_semaphore_fd;
   GLboolean EXT_semaphore_win32;
   GLboolean EXT_shader_image_load_formatted;
   GLboolean EXT_shader_image_load_store;
   GLboolean EXT_shader_integer_mix;
   GLboolean EXT_shader_samples_identical;
   GLboolean EXT_sRGB;
   GLboolean EXT_stencil_two_side;
   GLboolean EXT_texture_array;
   GLboolean EXT_texture_buffer_object;
   GLboolean EXT_texture_compression_latc;
   GLboolean EXT_texture_compression_s3tc;
   GLboolean EXT_texture_compression_s3tc_srgb;
   GLboolean EXT_texture_env_dot3;
   GLboolean EXT_texture_filter_anisotropic;
   GLboolean EXT_texture_filter_minmax;
   GLboolean EXT_texture_integer;
   GLboolean EXT_texture_mirror_clamp;
   GLboolean EXT_texture_norm16;
   GLboolean EXT_texture_shadow_lod;
   GLboolean EXT_texture_shared_exponent;
   GLboolean EXT_texture_snorm;
   GLboolean EXT_texture_sRGB;
   GLboolean EXT_texture_sRGB_R8;
   GLboolean EXT_texture_sRGB_RG8;
   GLboolean EXT_texture_sRGB_decode;
   GLboolean EXT_texture_swizzle;
   GLboolean EXT_texture_type_2_10_10_10_REV;
   GLboolean EXT_transform_feedback;
   GLboolean EXT_timer_query;
   GLboolean EXT_vertex_array_bgra;
   GLboolean EXT_window_rectangles;
   GLboolean OES_copy_image;
   GLboolean OES_primitive_bounding_box;
   GLboolean OES_sample_variables;
   GLboolean OES_standard_derivatives;
   GLboolean OES_texture_buffer;
   GLboolean OES_texture_cube_map_array;
   GLboolean OES_texture_view;
   GLboolean OES_viewport_array;
   /* vendor extensions */
   GLboolean AMD_compressed_ATC_texture;
   GLboolean AMD_framebuffer_multisample_advanced;
   GLboolean AMD_depth_clamp_separate;
   GLboolean AMD_performance_monitor;
   GLboolean AMD_pinned_memory;
   GLboolean AMD_seamless_cubemap_per_texture;
   GLboolean AMD_vertex_shader_layer;
   GLboolean AMD_vertex_shader_viewport_index;
   GLboolean ANDROID_extension_pack_es31a;
   GLboolean ARM_shader_framebuffer_fetch_depth_stencil;
   GLboolean ATI_meminfo;
   GLboolean ATI_texture_compression_3dc;
   GLboolean ATI_texture_mirror_once;
   GLboolean ATI_texture_env_combine3;
   GLboolean ATI_fragment_shader;
   GLboolean GREMEDY_string_marker;
   GLboolean INTEL_blackhole_render;
   GLboolean INTEL_conservative_rasterization;
   GLboolean INTEL_performance_query;
   GLboolean INTEL_shader_atomic_float_minmax;
   GLboolean INTEL_shader_integer_functions2;
   GLboolean KHR_blend_equation_advanced;
   GLboolean KHR_blend_equation_advanced_coherent;
   GLboolean KHR_robustness;
   GLboolean KHR_texture_compression_astc_hdr;
   GLboolean KHR_texture_compression_astc_ldr;
   GLboolean KHR_texture_compression_astc_sliced_3d;
   GLboolean MESA_framebuffer_flip_y;
   GLboolean MESA_pack_invert;
   GLboolean MESA_tile_raster_order;
   GLboolean EXT_shader_framebuffer_fetch;
   GLboolean EXT_shader_framebuffer_fetch_non_coherent;
   GLboolean MESA_shader_integer_functions;
   GLboolean MESA_window_pos;
   GLboolean MESA_ycbcr_texture;
   GLboolean NV_alpha_to_coverage_dither_control;
   GLboolean NV_compute_shader_derivatives;
   GLboolean NV_conditional_render;
   GLboolean NV_copy_depth_to_color;
   GLboolean NV_copy_image;
   GLboolean NV_fill_rectangle;
   GLboolean NV_fog_distance;
   GLboolean NV_primitive_restart;
   GLboolean NV_shader_atomic_float;
   GLboolean NV_shader_atomic_int64;
   GLboolean NV_texture_barrier;
   GLboolean NV_texture_env_combine4;
   GLboolean NV_texture_rectangle;
   GLboolean NV_vdpau_interop;
   GLboolean NV_conservative_raster;
   GLboolean NV_conservative_raster_dilate;
   GLboolean NV_conservative_raster_pre_snap_triangles;
   GLboolean NV_conservative_raster_pre_snap;
   GLboolean NV_viewport_array2;
   GLboolean NV_viewport_swizzle;
   GLboolean NVX_gpu_memory_info;
   GLboolean TDFX_texture_compression_FXT1;
   GLboolean OES_EGL_image;
   GLboolean OES_draw_texture;
   GLboolean OES_depth_texture_cube_map;
   GLboolean OES_EGL_image_external;
   GLboolean OES_texture_3D;
   GLboolean OES_texture_float;
   GLboolean OES_texture_float_linear;
   GLboolean OES_texture_half_float;
   GLboolean OES_texture_half_float_linear;
   GLboolean OES_compressed_ETC1_RGB8_texture;
   GLboolean OES_geometry_shader;
   GLboolean OES_texture_compression_astc;
   GLboolean extension_sentinel;
   /** The extension string */
   const GLubyte *String;
   /** Number of supported extensions */
   GLuint Count;
   /**
    * The context version which extension helper functions compare against.
    * By default, the value is equal to ctx->Version. This changes to ~0
    * while meta is in progress.
    */
   GLubyte Version;
};

/**
 * Compiler options for a single GLSL shaders type
 */
struct gl_shader_compiler_options
{
   /** Driver-selectable options: */
   GLboolean EmitNoCont;                  /**< Emit CONT opcode? */
   GLboolean EmitNoMainReturn;            /**< Emit CONT/RET opcodes? */
   GLboolean LowerCombinedClipCullDistance; /** Lower gl_ClipDistance and
                                              * gl_CullDistance together from
                                              * float[8] to vec4[2]
                                              **/
   GLbitfield LowerBuiltinVariablesXfb;   /**< Which builtin variables should
                                           * be lowered for transform feedback
                                           **/

   /**
    * If we can lower the precision of variables based on precision
    * qualifiers
    */
   GLboolean LowerPrecisionFloat16;
   GLboolean LowerPrecisionInt16;
   GLboolean LowerPrecisionDerivatives;
   GLboolean LowerPrecisionFloat16Uniforms;

   /**
    * This enables lowering of 16b constants.  Some drivers may not
    * to lower constants to 16b (ie. if the hw can do automatic
    * narrowing on constant load)
    */
   GLboolean LowerPrecisionConstants;

   /**
    * \name Forms of indirect addressing the driver cannot do.
    */
   /*@{*/
   GLboolean EmitNoIndirectInput;   /**< No indirect addressing of inputs */
   GLboolean EmitNoIndirectOutput;  /**< No indirect addressing of outputs */
   GLboolean EmitNoIndirectTemp;    /**< No indirect addressing of temps */
   GLboolean EmitNoIndirectUniform; /**< No indirect addressing of constants */
   /*@}*/

   GLuint MaxIfDepth;               /**< Maximum nested IF blocks */

   /**
    * Optimize code for array of structures backends.
    *
    * This is a proxy for:
    *   - preferring DP4 instructions (rather than MUL/MAD) for
    *     matrix * vector operations, such as position transformation.
    */
   GLboolean OptimizeForAOS;

   /** Clamp UBO and SSBO block indices so they don't go out-of-bounds. */
   GLboolean ClampBlockIndicesToArrayBounds;

   /** (driconf) Force gl_Position to be considered invariant */
   GLboolean PositionAlwaysInvariant;

   /** (driconf) Force gl_Position to be considered precise */
   GLboolean PositionAlwaysPrecise;

   const struct nir_shader_compiler_options *NirOptions;
};

/**
 * Precision info for shader datatypes.  See glGetShaderPrecisionFormat().
 */
struct gl_precision
{
   GLushort RangeMin;   /**< min value exponent */
   GLushort RangeMax;   /**< max value exponent */
   GLushort Precision;  /**< number of mantissa bits */
};

/**
 * Limits for vertex, geometry and fragment programs/shaders.
 */
struct gl_program_constants
{
   /* logical limits */
   GLuint MaxInstructions;
   GLuint MaxAluInstructions;
   GLuint MaxTexInstructions;
   GLuint MaxTexIndirections;
   GLuint MaxAttribs;
   GLuint MaxTemps;
   GLuint MaxAddressRegs;
   GLuint MaxAddressOffset;  /**< [-MaxAddressOffset, MaxAddressOffset-1] */
   GLuint MaxParameters;
   GLuint MaxLocalParams;
   GLuint MaxEnvParams;
   /* native/hardware limits */
   GLuint MaxNativeInstructions;
   GLuint MaxNativeAluInstructions;
   GLuint MaxNativeTexInstructions;
   GLuint MaxNativeTexIndirections;
   GLuint MaxNativeAttribs;
   GLuint MaxNativeTemps;
   GLuint MaxNativeAddressRegs;
   GLuint MaxNativeParameters;
   /* For shaders */
   GLuint MaxUniformComponents;  /**< Usually == MaxParameters * 4 */

   /**
    * \name Per-stage input / output limits
    *
    * Previous to OpenGL 3.2, the intrastage data limits were advertised with
    * a single value: GL_MAX_VARYING_COMPONENTS (GL_MAX_VARYING_VECTORS in
    * ES).  This is stored as \c gl_constants::MaxVarying.
    *
    * Starting with OpenGL 3.2, the limits are advertised with per-stage
    * variables.  Each stage as a certain number of outputs that it can feed
    * to the next stage and a certain number inputs that it can consume from
    * the previous stage.
    *
    * Vertex shader inputs do not participate this in this accounting.
    * These are tracked exclusively by \c gl_program_constants::MaxAttribs.
    *
    * Fragment shader outputs do not participate this in this accounting.
    * These are tracked exclusively by \c gl_constants::MaxDrawBuffers.
    */
   /*@{*/
   GLuint MaxInputComponents;
   GLuint MaxOutputComponents;
   /*@}*/

   /* ES 2.0 and GL_ARB_ES2_compatibility */
   struct gl_precision LowFloat, MediumFloat, HighFloat;
   struct gl_precision LowInt, MediumInt, HighInt;
   /* GL_ARB_uniform_buffer_object */
   GLuint MaxUniformBlocks;
   uint64_t MaxCombinedUniformComponents;
   GLuint MaxTextureImageUnits;

   /* GL_ARB_shader_atomic_counters */
   GLuint MaxAtomicBuffers;
   GLuint MaxAtomicCounters;

   /* GL_ARB_shader_image_load_store */
   GLuint MaxImageUniforms;

   /* GL_ARB_shader_storage_buffer_object */
   GLuint MaxShaderStorageBlocks;
};

/**
 * Constants which may be overridden by device driver during context creation
 * but are never changed after that.
 */
struct gl_constants
{
   /**
    * Bitmask of valid primitive types supported by the driver,
    */
   GLbitfield DriverSupportedPrimMask;

   GLuint MaxTextureMbytes;      /**< Max memory per image, in MB */
   GLuint MaxTextureSize;        /**< Max 1D/2D texture size, in pixels*/
   GLuint Max3DTextureLevels;    /**< Max mipmap levels for 3D textures */
   GLuint MaxCubeTextureLevels;  /**< Max mipmap levels for cube textures */
   GLuint MaxArrayTextureLayers; /**< Max layers in array textures */
   GLuint MaxTextureRectSize;    /**< Max rectangle texture size, in pixes */
   GLuint MaxTextureCoordUnits;
   GLuint MaxCombinedTextureImageUnits;
   GLuint MaxTextureUnits; /**< = MIN(CoordUnits, FragmentProgram.ImageUnits) */
   GLfloat MaxTextureMaxAnisotropy;  /**< GL_EXT_texture_filter_anisotropic */
   GLfloat MaxTextureLodBias;        /**< GL_EXT_texture_lod_bias */
   GLuint MaxTextureBufferSize;      /**< GL_ARB_texture_buffer_object */

   GLuint TextureBufferOffsetAlignment; /**< GL_ARB_texture_buffer_range */

   GLuint MaxArrayLockSize;

   GLint SubPixelBits;

   GLfloat MinPointSize, MaxPointSize;	     /**< aliased */
   GLfloat MinPointSizeAA, MaxPointSizeAA;   /**< antialiased */
   GLfloat PointSizeGranularity;
   GLfloat MinLineWidth, MaxLineWidth;       /**< aliased */
   GLfloat MinLineWidthAA, MaxLineWidthAA;   /**< antialiased */
   GLfloat LineWidthGranularity;

   GLuint MaxClipPlanes;
   GLuint MaxLights;
   GLfloat MaxShininess;                     /**< GL_NV_light_max_exponent */
   GLfloat MaxSpotExponent;                  /**< GL_NV_light_max_exponent */

   GLuint MaxViewportWidth, MaxViewportHeight;
   GLuint MaxViewports;                      /**< GL_ARB_viewport_array */
   GLuint ViewportSubpixelBits;              /**< GL_ARB_viewport_array */
   struct {
      GLfloat Min;
      GLfloat Max;
   } ViewportBounds;                         /**< GL_ARB_viewport_array */
   GLuint MaxWindowRectangles;               /**< GL_EXT_window_rectangles */

   struct gl_program_constants Program[MESA_SHADER_STAGES];
   GLuint MaxProgramMatrices;
   GLuint MaxProgramMatrixStackDepth;

   struct {
      GLuint SamplesPassed;
      GLuint TimeElapsed;
      GLuint Timestamp;
      GLuint PrimitivesGenerated;
      GLuint PrimitivesWritten;
      GLuint VerticesSubmitted;
      GLuint PrimitivesSubmitted;
      GLuint VsInvocations;
      GLuint TessPatches;
      GLuint TessInvocations;
      GLuint GsInvocations;
      GLuint GsPrimitives;
      GLuint FsInvocations;
      GLuint ComputeInvocations;
      GLuint ClInPrimitives;
      GLuint ClOutPrimitives;
   } QueryCounterBits;

   GLuint MaxDrawBuffers;    /**< GL_ARB_draw_buffers */

   GLuint MaxColorAttachments;   /**< GL_EXT_framebuffer_object */
   GLuint MaxRenderbufferSize;   /**< GL_EXT_framebuffer_object */
   GLuint MaxSamples;            /**< GL_ARB_framebuffer_object */

   /**
    * GL_ARB_framebuffer_no_attachments
    */
   GLuint MaxFramebufferWidth;
   GLuint MaxFramebufferHeight;
   GLuint MaxFramebufferLayers;
   GLuint MaxFramebufferSamples;

   /** Number of varying vectors between any two shader stages. */
   GLuint MaxVarying;

   /** @{
    * GL_ARB_uniform_buffer_object
    */
   GLuint MaxCombinedUniformBlocks;
   GLuint MaxUniformBufferBindings;
   GLuint MaxUniformBlockSize;
   GLuint UniformBufferOffsetAlignment;
   /** @} */

   /** @{
    * GL_ARB_shader_storage_buffer_object
    */
   GLuint MaxCombinedShaderStorageBlocks;
   GLuint MaxShaderStorageBufferBindings;
   GLuint MaxShaderStorageBlockSize;
   GLuint ShaderStorageBufferOffsetAlignment;
   /** @} */

   /**
    * GL_ARB_explicit_uniform_location
    */
   GLuint MaxUserAssignableUniformLocations;

   /** geometry shader */
   GLuint MaxGeometryOutputVertices;
   GLuint MaxGeometryTotalOutputComponents;
   GLuint MaxGeometryShaderInvocations;

   GLuint GLSLVersion;  /**< Desktop GLSL version supported (ex: 120 = 1.20) */
   GLuint GLSLVersionCompat;  /**< Desktop compat GLSL version supported  */

   /**
    * Changes default GLSL extension behavior from "error" to "warn".  It's out
    * of spec, but it can make some apps work that otherwise wouldn't.
    */
   GLboolean ForceGLSLExtensionsWarn;

   /**
    * Force all shaders to behave as if they were declared with the
    * compatibility token.
    */
   GLboolean ForceCompatShaders;

   /**
    * If non-zero, forces GLSL shaders to behave as if they began
    * with "#version ForceGLSLVersion".
    */
   GLuint ForceGLSLVersion;

   /**
    * Allow GLSL #extension directives in the middle of shaders.
    */
   GLboolean AllowGLSLExtensionDirectiveMidShader;

   /**
    * Allow a subset of GLSL 1.20 in GLSL 1.10 as needed by SPECviewperf13.
    */
   GLboolean AllowGLSL120SubsetIn110;

   /**
    * Allow builtins as part of constant expressions. This was not allowed
    * until GLSL 1.20 this allows it everywhere.
    */
   GLboolean AllowGLSLBuiltinConstantExpression;

   /**
    * Allow some relaxation of GLSL ES shader restrictions. This encompasses
    * a number of relaxations to the ES shader rules.
    */
   GLboolean AllowGLSLRelaxedES;

   /**
    * Allow GLSL built-in variables to be redeclared verbatim
    */
   GLboolean AllowGLSLBuiltinVariableRedeclaration;

   /**
    * Allow GLSL interpolation qualifier mismatch across shader stages.
    */
   GLboolean AllowGLSLCrossStageInterpolationMismatch;

   /**
    * Allow creating a higher compat profile (version 3.1+) for apps that
    * request it. Be careful when adding that driconf option because some
    * features are unimplemented and might not work correctly.
    */
   GLboolean AllowHigherCompatVersion;

   /**
    * Allow GLSL shaders with the compatibility version directive
    * in non-compatibility profiles. (for shader-db)
    */
   GLboolean AllowGLSLCompatShaders;

   /**
    * Allow extra tokens at end of preprocessor directives. The CTS now tests
    * to make sure these are not allowed. However, previously drivers would
    * allow them to exist and just issue a warning so some old applications
    * depend on this.
    */
   GLboolean AllowExtraPPTokens;

   /**
    * The spec forbids a shader to "statically write both gl_ClipVertex
    * and gl_ClipDistance".
    * This option adds a tolerance for shader that statically writes to
    * both but at least one of the write can be removed by a dead code
    * elimination pass.
    */
   GLboolean DoDCEBeforeClipCullAnalysis;

   /**
    * Force computing the absolute value for sqrt() and inversesqrt() to follow
    * D3D9 when apps rely on this behaviour.
    */
   GLboolean ForceGLSLAbsSqrt;

   /**
    * Forces the GLSL compiler to ignore writes to readonly vars rather than
    * throwing an error.
    */
   GLboolean GLSLIgnoreWriteToReadonlyVar;

   /**
    * Types of variable to default initialized to zero. Supported values are:
    *   - 0: no zero initialization
    *   - 1: all shader variables and gl_FragColor are initialiazed to 0
    *   - 2: same as 1, but shader out variables are *not* initialized, while
    *        function out variables are now initialized.
    */
   GLchar GLSLZeroInit;

   /**
    * Force GL names reuse. Needed by SPECviewperf13.
    */
   GLboolean ForceGLNamesReuse;

   /**
    * Treat integer textures using GL_LINEAR filters as GL_NEAREST.
    */
   GLboolean ForceIntegerTexNearest;

   /**
    * Treat 32-bit floating-point textures using GL_LINEAR filters as
    * GL_NEAREST.
    */
   GLboolean ForceFloat32TexNearest;

   /**
    * Does the driver support real 32-bit integers?  (Otherwise, integers are
    * simulated via floats.)
    */
   GLboolean NativeIntegers;

   /**
    * If the driver supports real 32-bit integers, what integer value should be
    * used for boolean true in uniform uploads?  (Usually 1 or ~0.)
    */
   GLuint UniformBooleanTrue;

   /**
    * Maximum amount of time, measured in nanseconds, that the server can wait.
    */
   GLuint64 MaxServerWaitTimeout;

   /** GL_EXT_provoking_vertex */
   GLboolean QuadsFollowProvokingVertexConvention;

   /** GL_ARB_viewport_array */
   GLenum16 LayerAndVPIndexProvokingVertex;

   /** OpenGL version 3.0 */
   GLbitfield ContextFlags;  /**< Ex: GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT */

   /** OpenGL version 3.2 */
   GLbitfield ProfileMask;   /**< Mask of CONTEXT_x_PROFILE_BIT */

   /** OpenGL version 4.4 */
   GLuint MaxVertexAttribStride;

   /** GL_EXT_transform_feedback */
   GLuint MaxTransformFeedbackBuffers;
   GLuint MaxTransformFeedbackSeparateComponents;
   GLuint MaxTransformFeedbackInterleavedComponents;
   GLuint MaxVertexStreams;

   /** GL_EXT_gpu_shader4 */
   GLint MinProgramTexelOffset, MaxProgramTexelOffset;

   /** GL_ARB_texture_gather */
   GLuint MinProgramTextureGatherOffset;
   GLuint MaxProgramTextureGatherOffset;
   GLuint MaxProgramTextureGatherComponents;

   /* GL_ARB_robustness */
   GLenum16 ResetStrategy;

   /* GL_KHR_robustness */
   GLboolean RobustAccess;

   /* GL_ARB_blend_func_extended */
   GLuint MaxDualSourceDrawBuffers;

   /**
    * For drivers which can do a better job at eliminating unused uniforms
    * than the GLSL compiler.
    *
    * XXX Remove these as soon as a better solution is available.
    */
   GLboolean GLSLSkipStrictMaxUniformLimitCheck;

   /**
    * Whether gl_FragCoord, gl_PointCoord and gl_FrontFacing
    * are system values.
    **/
   bool GLSLFragCoordIsSysVal;
   bool GLSLPointCoordIsSysVal;
   bool GLSLFrontFacingIsSysVal;

   /**
    * Whether to call lower_const_arrays_to_uniforms() during linking.
    */
   bool GLSLLowerConstArrays;

   /**
    * True if gl_TessLevelInner/Outer[] in the TES should be inputs
    * (otherwise, they're system values).
    */
   bool GLSLTessLevelsAsInputs;

   /** GL_ARB_map_buffer_alignment */
   GLuint MinMapBufferAlignment;

   /**
    * Disable varying packing.  This is out of spec, but potentially useful
    * for older platforms that supports a limited number of texture
    * indirections--on these platforms, unpacking the varyings in the fragment
    * shader increases the number of texture indirections by 1, which might
    * make some shaders not executable at all.
    *
    * Drivers that support transform feedback must set this value to GL_FALSE.
    */
   GLboolean DisableVaryingPacking;

   /**
    * Disable varying packing if used for transform feedback.  This is needed
    * for some drivers (e.g. Panfrost) where transform feedback requires
    * unpacked varyings.
    *
    * This variable is mutually exlusive with DisableVaryingPacking.
    */
   GLboolean DisableTransformFeedbackPacking;

   /**
    * Disable the glsl optimisation that resizes uniform arrays.
    */
   bool DisableUniformArrayResize;

   /**
    * Align varyings to POT in a slot
    *
    * Drivers that prefer varyings to be aligned to POT must set this value to GL_TRUE
    */
   GLboolean PreferPOTAlignedVaryings;


   /**
    * UBOs and SSBOs can be packed tightly by the OpenGL implementation when
    * layout is set as shared (the default) or packed. However most Mesa drivers
    * just use STD140 for these layouts. This flag allows drivers to use STD430
    * for packed and shared layouts which allows arrays to be packed more
    * tightly.
    */
   bool UseSTD430AsDefaultPacking;

   /**
    * Should meaningful names be generated for compiler temporary variables?
    *
    * Generally, it is not useful to have the compiler generate "meaningful"
    * names for temporary variables that it creates.  This can, however, be a
    * useful debugging aid.  In Mesa debug builds or release builds when
    * MESA_GLSL is set at run-time, meaningful names will be generated.
    * Drivers can also force names to be generated by setting this field.
    * For example, the i965 driver may set it when INTEL_DEBUG=vs (to dump
    * vertex shader assembly) is set at run-time.
    */
   bool GenerateTemporaryNames;

   /*
    * Maximum value supported for an index in DrawElements and friends.
    *
    * This must be at least (1ull<<24)-1.  The default value is
    * (1ull<<32)-1.
    *
    * \since ES 3.0 or GL_ARB_ES3_compatibility
    * \sa _mesa_init_constants
    */
   GLuint64 MaxElementIndex;

   /**
    * Disable interpretation of line continuations (lines ending with a
    * backslash character ('\') in GLSL source.
    */
   GLboolean DisableGLSLLineContinuations;

   /** GL_ARB_texture_multisample */
   GLint MaxColorTextureSamples;
   GLint MaxDepthTextureSamples;
   GLint MaxIntegerSamples;

   /** GL_AMD_framebuffer_multisample_advanced */
   GLint MaxColorFramebufferSamples;
   GLint MaxColorFramebufferStorageSamples;
   GLint MaxDepthStencilFramebufferSamples;

   /* An array of supported MSAA modes allowing different sample
    * counts per attachment type.
    */
   struct {
      GLint NumColorSamples;
      GLint NumColorStorageSamples;
      GLint NumDepthStencilSamples;
   } SupportedMultisampleModes[40];
   GLint NumSupportedMultisampleModes;

   /** GL_ARB_shader_atomic_counters */
   GLuint MaxAtomicBufferBindings;
   GLuint MaxAtomicBufferSize;
   GLuint MaxCombinedAtomicBuffers;
   GLuint MaxCombinedAtomicCounters;

   /** GL_ARB_vertex_attrib_binding */
   GLint MaxVertexAttribRelativeOffset;
   GLint MaxVertexAttribBindings;

   /* GL_ARB_shader_image_load_store */
   GLuint MaxImageUnits;
   GLuint MaxCombinedShaderOutputResources;
   GLuint MaxImageSamples;
   GLuint MaxCombinedImageUniforms;

   /** GL_ARB_compute_shader */
   GLuint MaxComputeWorkGroupCount[3]; /* Array of x, y, z dimensions */
   GLuint MaxComputeWorkGroupSize[3]; /* Array of x, y, z dimensions */
   GLuint MaxComputeWorkGroupInvocations;
   GLuint MaxComputeSharedMemorySize;

   /** GL_ARB_compute_variable_group_size */
   GLuint MaxComputeVariableGroupSize[3]; /* Array of x, y, z dimensions */
   GLuint MaxComputeVariableGroupInvocations;

   /** GL_ARB_gpu_shader5 */
   GLfloat MinFragmentInterpolationOffset;
   GLfloat MaxFragmentInterpolationOffset;

   GLboolean FakeSWMSAA;

   /** GL_KHR_context_flush_control */
   GLenum16 ContextReleaseBehavior;

   struct gl_shader_compiler_options ShaderCompilerOptions[MESA_SHADER_STAGES];

   /** GL_ARB_tessellation_shader */
   GLuint MaxPatchVertices;
   GLuint MaxTessGenLevel;
   GLuint MaxTessPatchComponents;
   GLuint MaxTessControlTotalOutputComponents;
   bool LowerTessLevel; /**< Lower gl_TessLevel* from float[n] to vecn? */
   bool PrimitiveRestartForPatches;

   /** GL_OES_primitive_bounding_box */
   bool NoPrimitiveBoundingBoxOutput;

   /** GL_ARB_sparse_buffer */
   GLuint SparseBufferPageSize;

   /** Used as an input for sha1 generation in the on-disk shader cache */
   unsigned char *dri_config_options_sha1;

   /** When drivers are OK with mapped buffers during draw and other calls. */
   bool AllowMappedBuffersDuringExecution;

   /** Override GL_MAP_UNSYNCHRONIZED_BIT */
   bool ForceMapBufferSynchronized;

   /** GL_ARB_get_program_binary */
   GLuint NumProgramBinaryFormats;

   /** GL_NV_conservative_raster */
   GLuint MaxSubpixelPrecisionBiasBits;

   /** GL_NV_conservative_raster_dilate */
   GLfloat ConservativeRasterDilateRange[2];
   GLfloat ConservativeRasterDilateGranularity;

   /** Is the drivers uniform storage packed or padded to 16 bytes. */
   bool PackedDriverUniformStorage;

   /** Wether or not glBitmap uses red textures rather than alpha */
   bool BitmapUsesRed;

   /** Whether the vertex buffer offset is a signed 32-bit integer. */
   bool VertexBufferOffsetIsInt32;

   /** Whether out-of-order draw (Begin/End) optimizations are allowed. */
   bool AllowDrawOutOfOrder;

   /** Whether to allow the fast path for frequently updated VAOs. */
   bool AllowDynamicVAOFastPath;

   /** Whether the driver can support primitive restart with a fixed index.
    * This is essentially a subset of NV_primitive_restart with enough support
    * to be able to enable GLES 3.1. Some hardware can support this but not the
    * full NV extension with arbitrary restart indices.
    */
   bool PrimitiveRestartFixedIndex;

   /** GL_ARB_gl_spirv */
   struct spirv_supported_capabilities SpirVCapabilities;

   /** GL_ARB_spirv_extensions */
   struct spirv_supported_extensions *SpirVExtensions;

   char *VendorOverride;
   char *RendererOverride;

   /** Buffer size used to upload vertices from glBegin/glEnd. */
   unsigned glBeginEndBufferSize;

   /** Whether the driver doesn't want x/y/width/height clipped based on src size
    *  when doing a copy texture operation (eg: may want out-of-bounds reads that
    *  produce 0 instead of leaving the texture content undefined).
    */
   bool NoClippingOnCopyTex;

   /**
    * Force glthread to always return GL_FRAMEBUFFER_COMPLETE to prevent
    * synchronization. Used for apps that call it every frame or multiple times
    * a frame, but always getting framebuffer completeness.
    */
   bool GLThreadNopCheckFramebufferStatus;

   /** GL_ARB_sparse_texture */
   GLuint MaxSparseTextureSize;
   GLuint MaxSparse3DTextureSize;
   GLuint MaxSparseArrayTextureLayers;
   bool SparseTextureFullArrayCubeMipmaps;

   /** Use hardware accelerated GL_SELECT */
   bool HardwareAcceleratedSelect;

   /** Allow GLThread to convert glBuffer */
   bool AllowGLThreadBufferSubDataOpt;
};
#endif
