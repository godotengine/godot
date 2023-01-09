/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#ifndef SHADER_INFO_H
#define SHADER_INFO_H

#include "util/bitset.h"
#include "util/sha1/sha1.h"
#include "shader_enums.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_XFB_BUFFERS        4
#define MAX_INLINABLE_UNIFORMS 4

struct spirv_supported_capabilities {
   bool address;
   bool atomic_storage;
   bool demote_to_helper_invocation;
   bool derivative_group;
   bool descriptor_array_dynamic_indexing;
   bool descriptor_array_non_uniform_indexing;
   bool descriptor_indexing;
   bool device_group;
   bool draw_parameters;
   bool float16_atomic_add;
   bool float16_atomic_min_max;
   bool float32_atomic_add;
   bool float32_atomic_min_max;
   bool float64;
   bool float64_atomic_add;
   bool float64_atomic_min_max;
   bool fragment_shader_sample_interlock;
   bool fragment_shader_pixel_interlock;
   bool fragment_shading_rate;
   bool generic_pointers;
   bool geometry_streams;
   bool groups;
   bool image_ms_array;
   bool image_read_without_format;
   bool image_write_without_format;
   bool image_atomic_int64;
   bool int8;
   bool int16;
   bool int64;
   bool int64_atomics;
   bool integer_functions2;
   bool kernel;
   bool kernel_image;
   bool kernel_image_read_write;
   bool linkage;
   bool literal_sampler;
   bool mesh_shading;
   bool mesh_shading_nv;
   bool min_lod;
   bool multiview;
   bool per_view_attributes_nv;
   bool physical_storage_buffer_address;
   bool post_depth_coverage;
   bool printf;
   bool ray_cull_mask;
   bool ray_tracing;
   bool ray_query;
   bool ray_traversal_primitive_culling;
   bool runtime_descriptor_array;
   bool float_controls;
   bool shader_clock;
   bool shader_viewport_index_layer;
   bool shader_viewport_mask_nv;
   bool sparse_residency;
   bool stencil_export;
   bool storage_8bit;
   bool storage_16bit;
   bool storage_image_ms;
   bool subgroup_arithmetic;
   bool subgroup_ballot;
   bool subgroup_basic;
   bool subgroup_dispatch;
   bool subgroup_quad;
   bool subgroup_shuffle;
   bool subgroup_uniform_control_flow;
   bool subgroup_vote;
   bool tessellation;
   bool transform_feedback;
   bool variable_pointers;
   bool vk_memory_model;
   bool vk_memory_model_device_scope;
   bool workgroup_memory_explicit_layout;
   bool float16;
   bool amd_fragment_mask;
   bool amd_gcn_shader;
   bool amd_shader_ballot;
   bool amd_trinary_minmax;
   bool amd_image_read_write_lod;
   bool amd_shader_explicit_vertex_parameter;
   bool amd_image_gather_bias_lod;

   bool intel_subgroup_shuffle;
   bool intel_subgroup_buffer_block_io;
};

typedef struct shader_info {
   const char *name;

   /* Descriptive name provided by the client; may be NULL */
   const char *label;

   /* Shader is internal, and should be ignored by things like NIR_DEBUG=print */
   bool internal;

   /* SHA1 of the original source, used by shader detection in drivers. */
   uint8_t source_sha1[SHA1_DIGEST_LENGTH];

   /** The shader stage, such as MESA_SHADER_VERTEX. */
   gl_shader_stage stage:8;

   /** The shader stage in a non SSO linked program that follows this stage,
     * such as MESA_SHADER_FRAGMENT.
     */
   gl_shader_stage next_stage:8;

   /* Number of textures used by this shader */
   uint8_t num_textures;
   /* Number of uniform buffers used by this shader */
   uint8_t num_ubos;
   /* Number of atomic buffers used by this shader */
   uint8_t num_abos;
   /* Number of shader storage buffers (max .driver_location + 1) used by this
    * shader.  In the case of nir_lower_atomics_to_ssbo being used, this will
    * be the number of actual SSBOs in gl_program->info, and the lowered SSBOs
    * and atomic counters in nir_shader->info.
    */
   uint8_t num_ssbos;
   /* Number of images used by this shader */
   uint8_t num_images;

   /* Which inputs are actually read */
   uint64_t inputs_read;
   /* Which outputs are actually written */
   uint64_t outputs_written;
   /* Which outputs are actually read */
   uint64_t outputs_read;
   /* Which system values are actually read */
   BITSET_DECLARE(system_values_read, SYSTEM_VALUE_MAX);

   /* Which I/O is per-primitive, for read/written information combine with
    * the fields above.
    */
   uint64_t per_primitive_inputs;
   uint64_t per_primitive_outputs;

   /* Which I/O is per-view */
   uint64_t per_view_outputs;

   /* Which 16-bit inputs and outputs are used corresponding to
    * VARYING_SLOT_VARn_16BIT.
    */
   uint16_t inputs_read_16bit;
   uint16_t outputs_written_16bit;
   uint16_t outputs_read_16bit;
   uint16_t inputs_read_indirectly_16bit;
   uint16_t outputs_accessed_indirectly_16bit;

   /* Which patch inputs are actually read */
   uint32_t patch_inputs_read;
   /* Which patch outputs are actually written */
   uint32_t patch_outputs_written;
   /* Which patch outputs are read */
   uint32_t patch_outputs_read;

   /* Which inputs are read indirectly (subset of inputs_read) */
   uint64_t inputs_read_indirectly;
   /* Which outputs are read or written indirectly */
   uint64_t outputs_accessed_indirectly;
   /* Which patch inputs are read indirectly (subset of patch_inputs_read) */
   uint64_t patch_inputs_read_indirectly;
   /* Which patch outputs are read or written indirectly */
   uint64_t patch_outputs_accessed_indirectly;

   /** Bitfield of which textures are used */
   BITSET_DECLARE(textures_used, 128);

   /** Bitfield of which textures are used by texelFetch() */
   BITSET_DECLARE(textures_used_by_txf, 128);

   /** Bitfield of which samplers are used */
   BITSET_DECLARE(samplers_used, 32);

   /** Bitfield of which images are used */
   BITSET_DECLARE(images_used, 64);
   /** Bitfield of which images are buffers. */
   BITSET_DECLARE(image_buffers, 64);
   /** Bitfield of which images are MSAA. */
   BITSET_DECLARE(msaa_images, 64);

   /* SPV_KHR_float_controls: execution mode for floating point ops */
   uint16_t float_controls_execution_mode;

   /**
    * Size of shared variables accessed by compute/task/mesh shaders.
    */
   unsigned shared_size;

   /**
    * Size of task payload variables accessed by task/mesh shaders.
    */
   unsigned task_payload_size;

   /**
    * Number of ray tracing queries in the shader (counts all elements of all
    * variables).
    */
   unsigned ray_queries;

   /**
    * Local workgroup size used by compute/task/mesh shaders.
    */
   uint16_t workgroup_size[3];

   enum gl_subgroup_size subgroup_size;

   /**
    * Uses subgroup intrinsics which can communicate across a quad.
    */
   bool uses_wide_subgroup_intrinsics;

   /* Transform feedback buffer strides in dwords, max. 1K - 4. */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   uint16_t inlinable_uniform_dw_offsets[MAX_INLINABLE_UNIFORMS];
   uint8_t num_inlinable_uniforms:4;

   /* The size of the gl_ClipDistance[] array, if declared. */
   uint8_t clip_distance_array_size:4;

   /* The size of the gl_CullDistance[] array, if declared. */
   uint8_t cull_distance_array_size:4;

   /* Whether or not this shader ever uses textureGather() */
   bool uses_texture_gather:1;

   /* Whether texture size, levels, or samples is queried. */
   bool uses_resource_info_query:1;

   /**
    * True if this shader uses the fddx/fddy opcodes.
    *
    * Note that this does not include the "fine" and "coarse" variants.
    */
   bool uses_fddx_fddy:1;

   /** Has divergence analysis ever been run? */
   bool divergence_analysis_run:1;

   /* Bitmask of bit-sizes used with ALU instructions. */
   uint8_t bit_sizes_float;
   uint8_t bit_sizes_int;

   /* Whether the first UBO is the default uniform buffer, i.e. uniforms. */
   bool first_ubo_is_default_ubo:1;

   /* Whether or not separate shader objects were used */
   bool separate_shader:1;

   /** Was this shader linked with any transform feedback varyings? */
   bool has_transform_feedback_varyings:1;

   /* Whether flrp has been lowered. */
   bool flrp_lowered:1;

   /* Whether nir_lower_io has been called to lower derefs.
    * nir_variables for inputs and outputs might not be present in the IR.
    */
   bool io_lowered:1;

   /* Whether the shader writes memory, including transform feedback. */
   bool writes_memory:1;

   /* Whether gl_Layer is viewport-relative */
   bool layer_viewport_relative:1;

   /* Whether explicit barriers are used */
   bool uses_control_barrier : 1;
   bool uses_memory_barrier : 1;

   /* Whether ARB_bindless_texture ops or variables are used */
   bool uses_bindless : 1;

   /**
    * Shared memory types have explicit layout set.  Used for
    * SPV_KHR_workgroup_storage_explicit_layout.
    */
   bool shared_memory_explicit_layout:1;

   /**
    * Used for VK_KHR_zero_initialize_workgroup_memory.
    */
   bool zero_initialize_shared_memory:1;

   /**
    * Used for ARB_compute_variable_group_size.
    */
   bool workgroup_size_variable:1;

   /**
     * Set if this shader uses legacy (DX9 or ARB assembly) math rules.
     *
     * From the ARB_fragment_program specification:
     *
     *    "The following rules apply to multiplication:
     *
     *      1. <x> * <y> == <y> * <x>, for all <x> and <y>.
     *      2. +/-0.0 * <x> = +/-0.0, at least for all <x> that correspond to
     *         *representable numbers (IEEE "not a number" and "infinity"
     *         *encodings may be exceptions).
     *      3. +1.0 * <x> = <x>, for all <x>.""
     *
     * However, in effect this was due to DX9 semantics implying that 0*x=0 even
     * for inf/nan if the hardware generated them instead of float_min/max.  So,
     * you should not have an exception for inf/nan to rule 2 above.
     *
     * One implementation of this behavior would be to flush all generated NaNs
     * to zero, at which point 0*Inf=Nan=0.  Most DX9/ARB-asm hardware did not
     * generate NaNs, and the only way the GPU saw one was to possibly feed it
     * in as a uniform.
     */
   bool use_legacy_math_rules;

   union {
      struct {
         /* Which inputs are doubles */
         uint64_t double_inputs;

         /* For AMD-specific driver-internal shaders. It replaces vertex
          * buffer loads with code generating VS inputs from scalar registers.
          *
          * Valid values: SI_VS_BLIT_SGPRS_POS_*
          */
         uint8_t blit_sgprs_amd:4;

         /* True if the shader writes position in window space coordinates pre-transform */
         bool window_space_position:1;

         /** Is an edge flag input needed? */
         bool needs_edge_flag:1;
      } vs;

      struct {
         /** The output primitive type */
         uint16_t output_primitive;

         /** The input primitive type */
         uint16_t input_primitive;

         /** The maximum number of vertices the geometry shader might write. */
         uint16_t vertices_out;

         /** 1 .. MAX_GEOMETRY_SHADER_INVOCATIONS */
         uint8_t invocations;

         /** The number of vertices received per input primitive (max. 6) */
         uint8_t vertices_in:3;

         /** Whether or not this shader uses EndPrimitive */
         bool uses_end_primitive:1;

         /** The streams used in this shaders (max. 4) */
         uint8_t active_stream_mask:4;
      } gs;

      struct {
         bool uses_discard:1;
         bool uses_demote:1;
         bool uses_fbfetch_output:1;
         bool color_is_dual_source:1;

         /**
          * True if this fragment shader requires helper invocations.  This
          * can be caused by the use of ALU derivative ops, texture
          * instructions which do implicit derivatives, and the use of quad
          * subgroup operations.
          */
         bool needs_quad_helper_invocations:1;

         /**
          * True if this fragment shader requires helper invocations for
          * all subgroup operations, not just quad ops and derivatives.
          */
         bool needs_all_helper_invocations:1;

         /**
          * Whether any inputs are declared with the "sample" qualifier.
          */
         bool uses_sample_qualifier:1;

         /**
          * Whether sample shading is used.
          */
         bool uses_sample_shading:1;

         /**
          * Whether early fragment tests are enabled as defined by
          * ARB_shader_image_load_store.
          */
         bool early_fragment_tests:1;

         /**
          * Defined by INTEL_conservative_rasterization.
          */
         bool inner_coverage:1;

         bool post_depth_coverage:1;

         /**
          * \name ARB_fragment_coord_conventions
          * @{
          */
         bool pixel_center_integer:1;
         bool origin_upper_left:1;
         /*@}*/

         bool pixel_interlock_ordered:1;
         bool pixel_interlock_unordered:1;
         bool sample_interlock_ordered:1;
         bool sample_interlock_unordered:1;

         /**
          * Flags whether NIR's base types on the FS color outputs should be
          * ignored.
          *
          * GLSL requires that fragment shader output base types match the
          * render target's base types for the behavior to be defined.  From
          * the GL 4.6 spec:
          *
          *     "If the values written by the fragment shader do not match the
          *      format(s) of the corresponding color buffer(s), the result is
          *      undefined."
          *
          * However, for NIR shaders translated from TGSI, we don't have the
          * output types any more, so the driver will need to do whatever
          * fixups are necessary to handle effectively untyped data being
          * output from the FS.
          */
         bool untyped_color_outputs:1;

         /** gl_FragDepth layout for ARB_conservative_depth. */
         enum gl_frag_depth_layout depth_layout:3;

         /**
          * Interpolation qualifiers for drivers that lowers color inputs
          * to system values.
          */
         unsigned color0_interp:3; /* glsl_interp_mode */
         bool color0_sample:1;
         bool color0_centroid:1;
         unsigned color1_interp:3; /* glsl_interp_mode */
         bool color1_sample:1;
         bool color1_centroid:1;

         /* Bitmask of gl_advanced_blend_mode values that may be used with this
          * shader.
          */
         unsigned advanced_blend_modes;

         /**
          * Defined by AMD_shader_early_and_late_fragment_tests.
          */
         bool early_and_late_fragment_tests:1;
         enum gl_frag_stencil_layout stencil_front_layout:3;
         enum gl_frag_stencil_layout stencil_back_layout:3;
      } fs;

      struct {
         uint16_t workgroup_size_hint[3];

         uint8_t user_data_components_amd:3;

         /*
          * Arrangement of invocations used to calculate derivatives in a compute
          * shader.  From NV_compute_shader_derivatives.
          */
         enum gl_derivative_group derivative_group:2;

         /**
          * pointer size is:
          *   AddressingModelLogical:    0    (default)
          *   AddressingModelPhysical32: 32
          *   AddressingModelPhysical64: 64
          */
         unsigned ptr_size;
      } cs;

      /* Applies to both TCS and TES. */
      struct {
         enum tess_primitive_mode _primitive_mode;

         /** The number of vertices in the TCS output patch. */
         uint8_t tcs_vertices_out;
         unsigned spacing:2; /*gl_tess_spacing*/

         /** Is the vertex order counterclockwise? */
         bool ccw:1;
         bool point_mode:1;

         /* Bit mask of TCS per-vertex inputs (VS outputs) that are used
          * with a vertex index that is NOT the invocation id
          */
         uint64_t tcs_cross_invocation_inputs_read;

         /* Bit mask of TCS per-vertex outputs that are used
          * with a vertex index that is NOT the invocation id
          */
         uint64_t tcs_cross_invocation_outputs_read;
      } tess;

      /* Applies to MESH. */
      struct {
         /* Bit mask of MS outputs that are used
          * with an index that is NOT the local invocation index.
          */
         uint64_t ms_cross_invocation_output_access;

         uint16_t max_vertices_out;
         uint16_t max_primitives_out;
         uint16_t primitive_type;  /* GL_POINTS, GL_LINES or GL_TRIANGLES. */

         /* TODO: remove this when we stop supporting NV_mesh_shader. */
         bool nv;
      } mesh;
   };
} shader_info;

#ifdef __cplusplus
}
#endif

#endif /* SHADER_INFO_H */
