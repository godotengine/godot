/*
 * Copyright © 2010 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GLSL_PARSER_EXTRAS_H
#define GLSL_PARSER_EXTRAS_H

/*
 * Most of the definitions here only apply to C++
 */
#ifdef __cplusplus


#include <stdlib.h>
#include "glsl_symbol_table.h"

/* THIS is a macro defined somewhere deep in the Windows MSVC header files.
 * Undefine it here to avoid collision with the lexer's THIS token.
 */
#undef THIS

struct gl_context;

struct glsl_switch_state {
   /** Temporary variables needed for switch statement. */
   ir_variable *test_var;
   ir_variable *is_fallthru_var;
   class ast_switch_statement *switch_nesting_ast;

   /** Used to detect if 'continue' was called inside a switch. */
   ir_variable *continue_inside;

   /** Used to set condition if 'default' label should be chosen. */
   ir_variable *run_default;

   /** Table of constant values already used in case labels */
   struct hash_table *labels_ht;
   class ast_case_label *previous_default;

   bool is_switch_innermost; // if switch stmt is closest to break, ...
};

const char *
glsl_compute_version_string(void *mem_ctx, bool is_es, unsigned version);

typedef struct YYLTYPE {
   int first_line;
   int first_column;
   int last_line;
   int last_column;
   unsigned source;
   /* Path for ARB_shading_language_include include source */
   char *path;
} YYLTYPE;
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1

extern void _mesa_glsl_error(YYLTYPE *locp, _mesa_glsl_parse_state *state,
                             const char *fmt, ...);


struct _mesa_glsl_parse_state {
   _mesa_glsl_parse_state(struct gl_context *_ctx, gl_shader_stage stage,
                          void *mem_ctx);

   DECLARE_RZALLOC_CXX_OPERATORS(_mesa_glsl_parse_state);

   /**
    * Generate a string representing the GLSL version currently being compiled
    * (useful for error messages).
    */
   const char *get_version_string()
   {
      return glsl_compute_version_string(this, this->es_shader,
                                         this->language_version);
   }

   /**
    * Determine whether the current GLSL version is sufficiently high to
    * support a certain feature.
    *
    * \param required_glsl_version is the desktop GLSL version that is
    * required to support the feature, or 0 if no version of desktop GLSL
    * supports the feature.
    *
    * \param required_glsl_es_version is the GLSL ES version that is required
    * to support the feature, or 0 if no version of GLSL ES supports the
    * feature.
    */
   bool is_version(unsigned required_glsl_version,
                   unsigned required_glsl_es_version) const
   {
      unsigned required_version = this->es_shader ?
         required_glsl_es_version : required_glsl_version;
      unsigned this_version = this->forced_language_version
         ? this->forced_language_version : this->language_version;
      return required_version != 0
         && this_version >= required_version;
   }

   bool check_version(unsigned required_glsl_version,
                      unsigned required_glsl_es_version,
                      YYLTYPE *locp, const char *fmt, ...) PRINTFLIKE(5, 6);

   bool check_arrays_of_arrays_allowed(YYLTYPE *locp)
   {
      if (!(ARB_arrays_of_arrays_enable || is_version(430, 310))) {
         const char *const requirement = this->es_shader
            ? "GLSL ES 3.10"
            : "GL_ARB_arrays_of_arrays or GLSL 4.30";
         _mesa_glsl_error(locp, this,
                          "%s required for defining arrays of arrays.",
                          requirement);
         return false;
      }
      return true;
   }

   bool check_precision_qualifiers_allowed(YYLTYPE *locp)
   {
      return check_version(130, 100, locp,
                           "precision qualifiers are forbidden");
   }

   bool check_bitwise_operations_allowed(YYLTYPE *locp)
   {
      return EXT_gpu_shader4_enable ||
             check_version(130, 300, locp, "bit-wise operations are forbidden");
   }

   bool check_explicit_attrib_stream_allowed(YYLTYPE *locp)
   {
      if (!this->has_explicit_attrib_stream()) {
         const char *const requirement = "GL_ARB_gpu_shader5 extension or GLSL 4.00";

         _mesa_glsl_error(locp, this, "explicit stream requires %s",
                          requirement);
         return false;
      }

      return true;
   }

   bool check_explicit_attrib_location_allowed(YYLTYPE *locp,
                                               const ir_variable *var)
   {
      if (!this->has_explicit_attrib_location()) {
         const char *const requirement = this->es_shader
            ? "GLSL ES 3.00"
            : "GL_ARB_explicit_attrib_location extension or GLSL 3.30";

         _mesa_glsl_error(locp, this, "%s explicit location requires %s",
                          mode_string(var), requirement);
         return false;
      }

      return true;
   }

   bool check_separate_shader_objects_allowed(YYLTYPE *locp,
                                              const ir_variable *var)
   {
      if (!this->has_separate_shader_objects()) {
         const char *const requirement = this->es_shader
            ? "GL_EXT_separate_shader_objects extension or GLSL ES 3.10"
            : "GL_ARB_separate_shader_objects extension or GLSL 4.20";

         _mesa_glsl_error(locp, this, "%s explicit location requires %s",
                          mode_string(var), requirement);
         return false;
      }

      return true;
   }

   bool check_explicit_uniform_location_allowed(YYLTYPE *locp,
                                                const ir_variable *)
   {
      if (!this->has_explicit_attrib_location() ||
          !this->has_explicit_uniform_location()) {
         const char *const requirement = this->es_shader
            ? "GLSL ES 3.10"
            : "GL_ARB_explicit_uniform_location and either "
              "GL_ARB_explicit_attrib_location or GLSL 3.30.";

         _mesa_glsl_error(locp, this,
                          "uniform explicit location requires %s",
                          requirement);
         return false;
      }

      return true;
   }

   bool has_atomic_counters() const
   {
      return ARB_shader_atomic_counters_enable || is_version(420, 310);
   }

   bool has_enhanced_layouts() const
   {
      return ARB_enhanced_layouts_enable || is_version(440, 0);
   }

   bool has_explicit_attrib_stream() const
   {
      return ARB_gpu_shader5_enable || is_version(400, 0);
   }

   bool has_explicit_attrib_location() const
   {
      return ARB_explicit_attrib_location_enable || is_version(330, 300);
   }

   bool has_explicit_uniform_location() const
   {
      return ARB_explicit_uniform_location_enable || is_version(430, 310);
   }

   bool has_uniform_buffer_objects() const
   {
      return ARB_uniform_buffer_object_enable || is_version(140, 300);
   }

   bool has_shader_storage_buffer_objects() const
   {
      return ARB_shader_storage_buffer_object_enable || is_version(430, 310);
   }

   bool has_separate_shader_objects() const
   {
      return ARB_separate_shader_objects_enable || is_version(410, 310)
         || EXT_separate_shader_objects_enable;
   }

   bool has_double() const
   {
      return ARB_gpu_shader_fp64_enable || is_version(400, 0);
   }

   bool has_int64() const
   {
      return ARB_gpu_shader_int64_enable ||
             AMD_gpu_shader_int64_enable;
   }

   bool has_420pack() const
   {
      return ARB_shading_language_420pack_enable || is_version(420, 0);
   }

   bool has_420pack_or_es31() const
   {
      return ARB_shading_language_420pack_enable || is_version(420, 310);
   }

   bool has_compute_shader() const
   {
      return ARB_compute_shader_enable || is_version(430, 310);
   }

   bool has_shader_io_blocks() const
   {
      /* The OES_geometry_shader_specification says:
       *
       *    "If the OES_geometry_shader extension is enabled, the
       *     OES_shader_io_blocks extension is also implicitly enabled."
       *
       * The OES_tessellation_shader extension has similar wording.
       */
      return OES_shader_io_blocks_enable ||
             EXT_shader_io_blocks_enable ||
             OES_geometry_shader_enable ||
             EXT_geometry_shader_enable ||
             OES_tessellation_shader_enable ||
             EXT_tessellation_shader_enable ||

             is_version(150, 320);
   }

   bool has_geometry_shader() const
   {
      return OES_geometry_shader_enable || EXT_geometry_shader_enable ||
             is_version(150, 320);
   }

   bool has_tessellation_shader() const
   {
      return ARB_tessellation_shader_enable ||
             OES_tessellation_shader_enable ||
             EXT_tessellation_shader_enable ||
             is_version(400, 320);
   }

   bool has_clip_distance() const
   {
      return EXT_clip_cull_distance_enable || is_version(130, 0);
   }

   bool has_cull_distance() const
   {
      return EXT_clip_cull_distance_enable ||
             ARB_cull_distance_enable ||
             is_version(450, 0);
   }

   bool has_framebuffer_fetch() const
   {
      return EXT_shader_framebuffer_fetch_enable ||
             EXT_shader_framebuffer_fetch_non_coherent_enable;
   }

   bool has_framebuffer_fetch_zs() const
   {
      return ARM_shader_framebuffer_fetch_depth_stencil_enable;
   }

   bool has_texture_cube_map_array() const
   {
      return ARB_texture_cube_map_array_enable ||
             EXT_texture_cube_map_array_enable ||
             OES_texture_cube_map_array_enable ||
             is_version(400, 320);
   }

   bool has_shader_image_load_store() const
   {
      return ARB_shader_image_load_store_enable ||
             EXT_shader_image_load_store_enable ||
             is_version(420, 310);
   }

   bool has_bindless() const
   {
      return ARB_bindless_texture_enable;
   }

   bool has_image_load_formatted() const
   {
      return EXT_shader_image_load_formatted_enable;
   }

   bool has_implicit_conversions() const
   {
      return EXT_shader_implicit_conversions_enable ||
             is_version(allow_glsl_120_subset_in_110 ? 110 : 120, 0);
   }

   bool has_implicit_int_to_uint_conversion() const
   {
      return ARB_gpu_shader5_enable ||
             MESA_shader_integer_functions_enable ||
             EXT_shader_implicit_conversions_enable ||
             is_version(400, 0);
   }

   void set_valid_gl_and_glsl_versions(YYLTYPE *locp);

   void process_version_directive(YYLTYPE *locp, int version,
                                  const char *ident);

   struct gl_context *const ctx; /* only to be used for debug callback. */
   const struct gl_extensions *exts;
   const struct gl_constants *consts;
   gl_api api;
   void *scanner;
   exec_list translation_unit;
   glsl_symbol_table *symbols;

   void *linalloc;

   unsigned num_supported_versions;
   struct {
      unsigned ver;
      uint8_t gl_ver;
      bool es;
   } supported_versions[17];

   bool es_shader;
   bool compat_shader;
   unsigned language_version;
   unsigned forced_language_version;
   /* Bitfield of ir_variable_mode to zero init */
   uint32_t zero_init;
   unsigned gl_version;
   gl_shader_stage stage;

   /**
    * Default uniform layout qualifiers tracked during parsing.
    * Currently affects uniform blocks and uniform buffer variables in
    * those blocks.
    */
   struct ast_type_qualifier *default_uniform_qualifier;

   /**
    * Default shader storage layout qualifiers tracked during parsing.
    * Currently affects shader storage blocks and shader storage buffer
    * variables in those blocks.
    */
   struct ast_type_qualifier *default_shader_storage_qualifier;

   /**
    * Variables to track different cases if a fragment shader redeclares
    * built-in variable gl_FragCoord.
    *
    * Note: These values are computed at ast_to_hir time rather than at parse
    * time.
    */
   bool fs_redeclares_gl_fragcoord;
   bool fs_origin_upper_left;
   bool fs_pixel_center_integer;
   bool fs_redeclares_gl_fragcoord_with_no_layout_qualifiers;

   /**
    * True if a geometry shader input primitive type or tessellation control
    * output vertices were specified using a layout directive.
    *
    * Note: these values are computed at ast_to_hir time rather than at parse
    * time.
    */
   bool gs_input_prim_type_specified;
   bool tcs_output_vertices_specified;

   /**
    * Input layout qualifiers from GLSL 1.50 (geometry shader controls),
    * and GLSL 4.00 (tessellation evaluation shader)
    */
   struct ast_type_qualifier *in_qualifier;

   /**
    * True if a compute shader input local size was specified using a layout
    * directive.
    *
    * Note: this value is computed at ast_to_hir time rather than at parse
    * time.
    */
   bool cs_input_local_size_specified;

   /**
    * If cs_input_local_size_specified is true, the local size that was
    * specified.  Otherwise ignored.
    */
   unsigned cs_input_local_size[3];

   /**
    * True if a compute shader input local variable size was specified using
    * a layout directive as specified by ARB_compute_variable_group_size.
    */
   bool cs_input_local_size_variable_specified;

   /**
    * Arrangement of invocations used to calculate derivatives in a compute
    * shader.  From NV_compute_shader_derivatives.
    */
   enum gl_derivative_group cs_derivative_group;

   /**
    * True if a shader declare bindless_sampler/bindless_image, and
    * respectively bound_sampler/bound_image at global scope as specified by
    * ARB_bindless_texture.
    */
   bool bindless_sampler_specified;
   bool bindless_image_specified;
   bool bound_sampler_specified;
   bool bound_image_specified;

   /**
    * Output layout qualifiers from GLSL 1.50 (geometry shader controls),
    * and GLSL 4.00 (tessellation control shader).
    */
   struct ast_type_qualifier *out_qualifier;

   /**
    * Printable list of GLSL versions supported by the current context
    *
    * \note
    * This string should probably be generated per-context instead of per
    * invokation of the compiler.  This should be changed when the method of
    * tracking supported GLSL versions changes.
    */
   const char *supported_version_string;

   /**
    * Implementation defined limits that affect built-in variables, etc.
    *
    * \sa struct gl_constants (in mtypes.h)
    */
   struct {
      /* 1.10 */
      unsigned MaxLights;
      unsigned MaxClipPlanes;
      unsigned MaxTextureUnits;
      unsigned MaxTextureCoords;
      unsigned MaxVertexAttribs;
      unsigned MaxVertexUniformComponents;
      unsigned MaxVertexTextureImageUnits;
      unsigned MaxCombinedTextureImageUnits;
      unsigned MaxTextureImageUnits;
      unsigned MaxFragmentUniformComponents;

      /* ARB_draw_buffers */
      unsigned MaxDrawBuffers;

      /* ARB_enhanced_layouts */
      unsigned MaxTransformFeedbackBuffers;
      unsigned MaxTransformFeedbackInterleavedComponents;

      /* ARB_blend_func_extended */
      unsigned MaxDualSourceDrawBuffers;

      /* 3.00 ES */
      int MinProgramTexelOffset;
      int MaxProgramTexelOffset;

      /* 1.50 */
      unsigned MaxVertexOutputComponents;
      unsigned MaxGeometryInputComponents;
      unsigned MaxGeometryOutputComponents;
      unsigned MaxGeometryShaderInvocations;
      unsigned MaxFragmentInputComponents;
      unsigned MaxGeometryTextureImageUnits;
      unsigned MaxGeometryOutputVertices;
      unsigned MaxGeometryTotalOutputComponents;
      unsigned MaxGeometryUniformComponents;

      /* ARB_shader_atomic_counters */
      unsigned MaxVertexAtomicCounters;
      unsigned MaxTessControlAtomicCounters;
      unsigned MaxTessEvaluationAtomicCounters;
      unsigned MaxGeometryAtomicCounters;
      unsigned MaxFragmentAtomicCounters;
      unsigned MaxCombinedAtomicCounters;
      unsigned MaxAtomicBufferBindings;

      /* These are also atomic counter related, but they weren't added to
       * until atomic counters were added to core in GLSL 4.20 and GLSL ES
       * 3.10.
       */
      unsigned MaxVertexAtomicCounterBuffers;
      unsigned MaxTessControlAtomicCounterBuffers;
      unsigned MaxTessEvaluationAtomicCounterBuffers;
      unsigned MaxGeometryAtomicCounterBuffers;
      unsigned MaxFragmentAtomicCounterBuffers;
      unsigned MaxCombinedAtomicCounterBuffers;
      unsigned MaxAtomicCounterBufferSize;

      /* ARB_compute_shader */
      unsigned MaxComputeAtomicCounterBuffers;
      unsigned MaxComputeAtomicCounters;
      unsigned MaxComputeImageUniforms;
      unsigned MaxComputeTextureImageUnits;
      unsigned MaxComputeUniformComponents;
      unsigned MaxComputeWorkGroupCount[3];
      unsigned MaxComputeWorkGroupSize[3];

      /* ARB_shader_image_load_store */
      unsigned MaxImageUnits;
      unsigned MaxCombinedShaderOutputResources;
      unsigned MaxImageSamples;
      unsigned MaxVertexImageUniforms;
      unsigned MaxTessControlImageUniforms;
      unsigned MaxTessEvaluationImageUniforms;
      unsigned MaxGeometryImageUniforms;
      unsigned MaxFragmentImageUniforms;
      unsigned MaxCombinedImageUniforms;

      /* ARB_viewport_array */
      unsigned MaxViewports;

      /* ARB_tessellation_shader */
      unsigned MaxPatchVertices;
      unsigned MaxTessGenLevel;
      unsigned MaxTessControlInputComponents;
      unsigned MaxTessControlOutputComponents;
      unsigned MaxTessControlTextureImageUnits;
      unsigned MaxTessEvaluationInputComponents;
      unsigned MaxTessEvaluationOutputComponents;
      unsigned MaxTessEvaluationTextureImageUnits;
      unsigned MaxTessPatchComponents;
      unsigned MaxTessControlTotalOutputComponents;
      unsigned MaxTessControlUniformComponents;
      unsigned MaxTessEvaluationUniformComponents;

      /* GL 4.5 / OES_sample_variables */
      unsigned MaxSamples;
   } Const;

   /**
    * During AST to IR conversion, pointer to current IR function
    *
    * Will be \c NULL whenever the AST to IR conversion is not inside a
    * function definition.
    */
   class ir_function_signature *current_function;

   /**
    * During AST to IR conversion, pointer to the toplevel IR
    * instruction list being generated.
    */
   exec_list *toplevel_ir;

   /** Have we found a return statement in this function? */
   bool found_return;

   /** Have we found the interlock builtins in this function? */
   bool found_begin_interlock;
   bool found_end_interlock;

   /** Was there an error during compilation? */
   bool error;

   /**
    * Are all shader inputs / outputs invariant?
    *
    * This is set when the 'STDGL invariant(all)' pragma is used.
    */
   bool all_invariant;

   /** Loop or switch statement containing the current instructions. */
   class ast_iteration_statement *loop_nesting_ast;

   struct glsl_switch_state switch_state;

   /** List of structures defined in user code. */
   const glsl_type **user_structures;
   unsigned num_user_structures;

   char *info_log;

   /**
    * Are warnings enabled?
    *
    * Emission of warngins is controlled by '#pragma warning(...)'.
    */
   bool warnings_enabled;

   /**
    * \name Enable bits for GLSL extensions
    */
   /*@{*/
   /* ARB extensions go here, sorted alphabetically.
    */
   bool ARB_ES3_1_compatibility_enable;
   bool ARB_ES3_1_compatibility_warn;
   bool ARB_ES3_2_compatibility_enable;
   bool ARB_ES3_2_compatibility_warn;
   bool ARB_arrays_of_arrays_enable;
   bool ARB_arrays_of_arrays_warn;
   bool ARB_bindless_texture_enable;
   bool ARB_bindless_texture_warn;
   bool ARB_compatibility_enable;
   bool ARB_compatibility_warn;
   bool ARB_compute_shader_enable;
   bool ARB_compute_shader_warn;
   bool ARB_compute_variable_group_size_enable;
   bool ARB_compute_variable_group_size_warn;
   bool ARB_conservative_depth_enable;
   bool ARB_conservative_depth_warn;
   bool ARB_cull_distance_enable;
   bool ARB_cull_distance_warn;
   bool ARB_derivative_control_enable;
   bool ARB_derivative_control_warn;
   bool ARB_draw_buffers_enable;
   bool ARB_draw_buffers_warn;
   bool ARB_draw_instanced_enable;
   bool ARB_draw_instanced_warn;
   bool ARB_enhanced_layouts_enable;
   bool ARB_enhanced_layouts_warn;
   bool ARB_explicit_attrib_location_enable;
   bool ARB_explicit_attrib_location_warn;
   bool ARB_explicit_uniform_location_enable;
   bool ARB_explicit_uniform_location_warn;
   bool ARB_fragment_coord_conventions_enable;
   bool ARB_fragment_coord_conventions_warn;
   bool ARB_fragment_layer_viewport_enable;
   bool ARB_fragment_layer_viewport_warn;
   bool ARB_fragment_shader_interlock_enable;
   bool ARB_fragment_shader_interlock_warn;
   bool ARB_gpu_shader5_enable;
   bool ARB_gpu_shader5_warn;
   bool ARB_gpu_shader_fp64_enable;
   bool ARB_gpu_shader_fp64_warn;
   bool ARB_gpu_shader_int64_enable;
   bool ARB_gpu_shader_int64_warn;
   bool ARB_post_depth_coverage_enable;
   bool ARB_post_depth_coverage_warn;
   bool ARB_sample_shading_enable;
   bool ARB_sample_shading_warn;
   bool ARB_separate_shader_objects_enable;
   bool ARB_separate_shader_objects_warn;
   bool ARB_shader_atomic_counter_ops_enable;
   bool ARB_shader_atomic_counter_ops_warn;
   bool ARB_shader_atomic_counters_enable;
   bool ARB_shader_atomic_counters_warn;
   bool ARB_shader_ballot_enable;
   bool ARB_shader_ballot_warn;
   bool ARB_shader_bit_encoding_enable;
   bool ARB_shader_bit_encoding_warn;
   bool ARB_shader_clock_enable;
   bool ARB_shader_clock_warn;
   bool ARB_shader_draw_parameters_enable;
   bool ARB_shader_draw_parameters_warn;
   bool ARB_shader_group_vote_enable;
   bool ARB_shader_group_vote_warn;
   bool ARB_shader_image_load_store_enable;
   bool ARB_shader_image_load_store_warn;
   bool ARB_shader_image_size_enable;
   bool ARB_shader_image_size_warn;
   bool ARB_shader_precision_enable;
   bool ARB_shader_precision_warn;
   bool ARB_shader_stencil_export_enable;
   bool ARB_shader_stencil_export_warn;
   bool ARB_shader_storage_buffer_object_enable;
   bool ARB_shader_storage_buffer_object_warn;
   bool ARB_shader_subroutine_enable;
   bool ARB_shader_subroutine_warn;
   bool ARB_shader_texture_image_samples_enable;
   bool ARB_shader_texture_image_samples_warn;
   bool ARB_shader_texture_lod_enable;
   bool ARB_shader_texture_lod_warn;
   bool ARB_shader_viewport_layer_array_enable;
   bool ARB_shader_viewport_layer_array_warn;
   bool ARB_shading_language_420pack_enable;
   bool ARB_shading_language_420pack_warn;
   bool ARB_shading_language_include_enable;
   bool ARB_shading_language_include_warn;
   bool ARB_shading_language_packing_enable;
   bool ARB_shading_language_packing_warn;
   bool ARB_sparse_texture2_enable;
   bool ARB_sparse_texture2_warn;
   bool ARB_sparse_texture_clamp_enable;
   bool ARB_sparse_texture_clamp_warn;
   bool ARB_tessellation_shader_enable;
   bool ARB_tessellation_shader_warn;
   bool ARB_texture_cube_map_array_enable;
   bool ARB_texture_cube_map_array_warn;
   bool ARB_texture_gather_enable;
   bool ARB_texture_gather_warn;
   bool ARB_texture_multisample_enable;
   bool ARB_texture_multisample_warn;
   bool ARB_texture_query_levels_enable;
   bool ARB_texture_query_levels_warn;
   bool ARB_texture_query_lod_enable;
   bool ARB_texture_query_lod_warn;
   bool ARB_texture_rectangle_enable;
   bool ARB_texture_rectangle_warn;
   bool ARB_uniform_buffer_object_enable;
   bool ARB_uniform_buffer_object_warn;
   bool ARB_vertex_attrib_64bit_enable;
   bool ARB_vertex_attrib_64bit_warn;
   bool ARB_viewport_array_enable;
   bool ARB_viewport_array_warn;

   /* KHR extensions go here, sorted alphabetically.
    */
   bool KHR_blend_equation_advanced_enable;
   bool KHR_blend_equation_advanced_warn;

   /* OES extensions go here, sorted alphabetically.
    */
   bool OES_EGL_image_external_enable;
   bool OES_EGL_image_external_warn;
   bool OES_EGL_image_external_essl3_enable;
   bool OES_EGL_image_external_essl3_warn;
   bool OES_geometry_point_size_enable;
   bool OES_geometry_point_size_warn;
   bool OES_geometry_shader_enable;
   bool OES_geometry_shader_warn;
   bool OES_gpu_shader5_enable;
   bool OES_gpu_shader5_warn;
   bool OES_primitive_bounding_box_enable;
   bool OES_primitive_bounding_box_warn;
   bool OES_sample_variables_enable;
   bool OES_sample_variables_warn;
   bool OES_shader_image_atomic_enable;
   bool OES_shader_image_atomic_warn;
   bool OES_shader_io_blocks_enable;
   bool OES_shader_io_blocks_warn;
   bool OES_shader_multisample_interpolation_enable;
   bool OES_shader_multisample_interpolation_warn;
   bool OES_standard_derivatives_enable;
   bool OES_standard_derivatives_warn;
   bool OES_tessellation_point_size_enable;
   bool OES_tessellation_point_size_warn;
   bool OES_tessellation_shader_enable;
   bool OES_tessellation_shader_warn;
   bool OES_texture_3D_enable;
   bool OES_texture_3D_warn;
   bool OES_texture_buffer_enable;
   bool OES_texture_buffer_warn;
   bool OES_texture_cube_map_array_enable;
   bool OES_texture_cube_map_array_warn;
   bool OES_texture_storage_multisample_2d_array_enable;
   bool OES_texture_storage_multisample_2d_array_warn;
   bool OES_viewport_array_enable;
   bool OES_viewport_array_warn;

   /* All other extensions go here, sorted alphabetically.
    */
   bool AMD_conservative_depth_enable;
   bool AMD_conservative_depth_warn;
   bool AMD_gpu_shader_int64_enable;
   bool AMD_gpu_shader_int64_warn;
   bool AMD_shader_stencil_export_enable;
   bool AMD_shader_stencil_export_warn;
   bool AMD_shader_trinary_minmax_enable;
   bool AMD_shader_trinary_minmax_warn;
   bool AMD_texture_texture4_enable;
   bool AMD_texture_texture4_warn;
   bool AMD_vertex_shader_layer_enable;
   bool AMD_vertex_shader_layer_warn;
   bool AMD_vertex_shader_viewport_index_enable;
   bool AMD_vertex_shader_viewport_index_warn;
   bool ANDROID_extension_pack_es31a_enable;
   bool ANDROID_extension_pack_es31a_warn;
   bool ARM_shader_framebuffer_fetch_depth_stencil_enable;
   bool ARM_shader_framebuffer_fetch_depth_stencil_warn;
   bool EXT_blend_func_extended_enable;
   bool EXT_blend_func_extended_warn;
   bool EXT_clip_cull_distance_enable;
   bool EXT_clip_cull_distance_warn;
   bool EXT_demote_to_helper_invocation_enable;
   bool EXT_demote_to_helper_invocation_warn;
   bool EXT_draw_buffers_enable;
   bool EXT_draw_buffers_warn;
   bool EXT_draw_instanced_enable;
   bool EXT_draw_instanced_warn;
   bool EXT_frag_depth_enable;
   bool EXT_frag_depth_warn;
   bool EXT_geometry_point_size_enable;
   bool EXT_geometry_point_size_warn;
   bool EXT_geometry_shader_enable;
   bool EXT_geometry_shader_warn;
   bool EXT_gpu_shader4_enable;
   bool EXT_gpu_shader4_warn;
   bool EXT_gpu_shader5_enable;
   bool EXT_gpu_shader5_warn;
   bool EXT_primitive_bounding_box_enable;
   bool EXT_primitive_bounding_box_warn;
   bool EXT_separate_shader_objects_enable;
   bool EXT_separate_shader_objects_warn;
   bool EXT_shader_framebuffer_fetch_enable;
   bool EXT_shader_framebuffer_fetch_warn;
   bool EXT_shader_framebuffer_fetch_non_coherent_enable;
   bool EXT_shader_framebuffer_fetch_non_coherent_warn;
   bool EXT_shader_group_vote_enable;
   bool EXT_shader_group_vote_warn;
   bool EXT_shader_image_load_formatted_enable;
   bool EXT_shader_image_load_formatted_warn;
   bool EXT_shader_image_load_store_enable;
   bool EXT_shader_image_load_store_warn;
   bool EXT_shader_implicit_conversions_enable;
   bool EXT_shader_implicit_conversions_warn;
   bool EXT_shader_integer_mix_enable;
   bool EXT_shader_integer_mix_warn;
   bool EXT_shader_io_blocks_enable;
   bool EXT_shader_io_blocks_warn;
   bool EXT_shader_samples_identical_enable;
   bool EXT_shader_samples_identical_warn;
   bool EXT_tessellation_point_size_enable;
   bool EXT_tessellation_point_size_warn;
   bool EXT_tessellation_shader_enable;
   bool EXT_tessellation_shader_warn;
   bool EXT_texture_array_enable;
   bool EXT_texture_array_warn;
   bool EXT_texture_buffer_enable;
   bool EXT_texture_buffer_warn;
   bool EXT_texture_cube_map_array_enable;
   bool EXT_texture_cube_map_array_warn;
   bool EXT_texture_query_lod_enable;
   bool EXT_texture_query_lod_warn;
   bool EXT_texture_shadow_lod_enable;
   bool EXT_texture_shadow_lod_warn;
   bool INTEL_conservative_rasterization_enable;
   bool INTEL_conservative_rasterization_warn;
   bool INTEL_shader_atomic_float_minmax_enable;
   bool INTEL_shader_atomic_float_minmax_warn;
   bool INTEL_shader_integer_functions2_enable;
   bool INTEL_shader_integer_functions2_warn;
   bool MESA_shader_integer_functions_enable;
   bool MESA_shader_integer_functions_warn;
   bool NV_compute_shader_derivatives_enable;
   bool NV_compute_shader_derivatives_warn;
   bool NV_fragment_shader_interlock_enable;
   bool NV_fragment_shader_interlock_warn;
   bool NV_image_formats_enable;
   bool NV_image_formats_warn;
   bool NV_shader_atomic_float_enable;
   bool NV_shader_atomic_float_warn;
   bool NV_shader_atomic_int64_enable;
   bool NV_shader_atomic_int64_warn;
   bool NV_shader_noperspective_interpolation_enable;
   bool NV_shader_noperspective_interpolation_warn;
   bool NV_viewport_array2_enable;
   bool NV_viewport_array2_warn;
   /*@}*/

   /** Extensions supported by the OpenGL implementation. */
   const struct gl_extensions *extensions;

   bool uses_builtin_functions;
   bool fs_uses_gl_fragcoord;

   /**
    * For geometry shaders, size of the most recently seen input declaration
    * that was a sized array, or 0 if no sized input array declarations have
    * been seen.
    *
    * Unused for other shader types.
    */
   unsigned gs_input_size;

   bool fs_early_fragment_tests;

   bool fs_inner_coverage;

   bool fs_post_depth_coverage;

   bool fs_pixel_interlock_ordered;
   bool fs_pixel_interlock_unordered;
   bool fs_sample_interlock_ordered;
   bool fs_sample_interlock_unordered;

   unsigned fs_blend_support;

   /**
    * For tessellation control shaders, size of the most recently seen output
    * declaration that was a sized array, or 0 if no sized output array
    * declarations have been seen.
    *
    * Unused for other shader types.
    */
   unsigned tcs_output_size;

   /** Atomic counter offsets by binding */
   unsigned atomic_counter_offsets[MAX_COMBINED_ATOMIC_BUFFERS];

   /** Whether gl_Layer output is viewport-relative. */
   bool redeclares_gl_layer;
   bool layer_viewport_relative;

   bool allow_extension_directive_midshader;
   bool allow_glsl_120_subset_in_110;
   bool allow_builtin_variable_redeclaration;
   bool ignore_write_to_readonly_var;

   /**
    * Known subroutine type declarations.
    */
   int num_subroutine_types;
   ir_function **subroutine_types;

   /**
    * Functions that are associated with
    * subroutine types.
    */
   int num_subroutines;
   ir_function **subroutines;

   /**
    * field selection temporary parser storage -
    * did the parser just parse a dot.
    */
   bool is_field;

   /**
    * seen values for clip/cull distance sizes
    * so we can check totals aren't too large.
    */
   unsigned clip_dist_size, cull_dist_size;
};

# define YYLLOC_DEFAULT(Current, Rhs, N)                        \
do {                                                            \
   if (N)                                                       \
   {                                                            \
      (Current).first_line   = YYRHSLOC(Rhs, 1).first_line;     \
      (Current).first_column = YYRHSLOC(Rhs, 1).first_column;   \
      (Current).last_line    = YYRHSLOC(Rhs, N).last_line;      \
      (Current).last_column  = YYRHSLOC(Rhs, N).last_column;    \
      (Current).path         = YYRHSLOC(Rhs, N).path;           \
   }                                                            \
   else                                                         \
   {                                                            \
      (Current).first_line   = (Current).last_line =            \
         YYRHSLOC(Rhs, 0).last_line;                            \
      (Current).first_column = (Current).last_column =          \
         YYRHSLOC(Rhs, 0).last_column;                          \
      (Current).path = YYRHSLOC(Rhs, 0).path;                   \
   }                                                            \
   (Current).source = 0;                                        \
} while (0)

/**
 * Emit a warning to the shader log
 *
 * \sa _mesa_glsl_error
 */
extern void _mesa_glsl_warning(const YYLTYPE *locp,
                               _mesa_glsl_parse_state *state,
                               const char *fmt, ...);

extern void _mesa_glsl_lexer_ctor(struct _mesa_glsl_parse_state *state,
                                  const char *string);

extern void _mesa_glsl_lexer_dtor(struct _mesa_glsl_parse_state *state);

union YYSTYPE;
extern int _mesa_glsl_lexer_lex(union YYSTYPE *yylval, YYLTYPE *yylloc,
                                void *scanner);

extern int _mesa_glsl_parse(struct _mesa_glsl_parse_state *);

/**
 * Process elements of the #extension directive
 *
 * \return
 * If \c name and \c behavior are valid, \c true is returned.  Otherwise
 * \c false is returned.
 */
extern bool _mesa_glsl_process_extension(const char *name, YYLTYPE *name_locp,
                                         const char *behavior,
                                         YYLTYPE *behavior_locp,
                                         _mesa_glsl_parse_state *state);

#endif /* __cplusplus */


/*
 * These definitions apply to C and C++
 */
#ifdef __cplusplus
extern "C" {
#endif

struct glcpp_parser;
struct _mesa_glsl_parse_state;

typedef void (*glcpp_extension_iterator)(
              struct _mesa_glsl_parse_state *state,
              void (*add_builtin_define)(struct glcpp_parser *, const char *, int),
              struct glcpp_parser *data,
              unsigned version,
              bool es);

extern int glcpp_preprocess(void *ctx, const char **shader, char **info_log,
                            glcpp_extension_iterator extensions,
                            struct _mesa_glsl_parse_state *state,
                            struct gl_context *gl_ctx);

extern void
_mesa_glsl_copy_symbols_from_table(struct exec_list *shader_ir,
                                   struct glsl_symbol_table *src,
                                   struct glsl_symbol_table *dest);

#ifdef __cplusplus
}
#endif


#endif /* GLSL_PARSER_EXTRAS_H */
