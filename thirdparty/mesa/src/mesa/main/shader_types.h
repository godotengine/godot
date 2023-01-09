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
 * \file shader_types.h
 * All the GL shader/program types.
 */

#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include "main/config.h" /* for MAX_FEEDBACK_BUFFERS */
#include "util/glheader.h"
#include "main/menums.h"
#include "util/mesa-sha1.h"
#include "compiler/shader_info.h"
#include "compiler/glsl/list.h"
#include "compiler/glsl/ir_uniform.h"

#include "pipe/p_state.h"

/**
 * Shader information needed by both gl_shader and gl_linked shader.
 */
struct gl_shader_info
{
   /**
    * Tessellation Control shader state from layout qualifiers.
    */
   struct {
      /**
       * 0 - vertices not declared in shader, or
       * 1 .. GL_MAX_PATCH_VERTICES
       */
      GLint VerticesOut;
   } TessCtrl;

   /**
    * Tessellation Evaluation shader state from layout qualifiers.
    */
   struct {
      enum tess_primitive_mode _PrimitiveMode;

      enum gl_tess_spacing Spacing;

      /**
       * GL_CW, GL_CCW, or 0 if it's not set in this shader.
       */
      GLenum16 VertexOrder;
      /**
       * 1, 0, or -1 if it's not set in this shader.
       */
      int PointMode;
   } TessEval;

   /**
    * Geometry shader state from GLSL 1.50 layout qualifiers.
    */
   struct {
      GLint VerticesOut;
      /**
       * 0 - Invocations count not declared in shader, or
       * 1 .. Const.MaxGeometryShaderInvocations
       */
      GLint Invocations;
      /**
       * GL_POINTS, GL_LINES, GL_LINES_ADJACENCY, GL_TRIANGLES, or
       * GL_TRIANGLES_ADJACENCY, or PRIM_UNKNOWN if it's not set in this
       * shader.
       */
      enum shader_prim InputType;
       /**
        * GL_POINTS, GL_LINE_STRIP or GL_TRIANGLE_STRIP, or PRIM_UNKNOWN if
        * it's not set in this shader.
        */
      enum shader_prim OutputType;
   } Geom;

   /**
    * Compute shader state from ARB_compute_shader and
    * ARB_compute_variable_group_size layout qualifiers.
    */
   struct {
      /**
       * Size specified using local_size_{x,y,z}, or all 0's to indicate that
       * it's not set in this shader.
       */
      unsigned LocalSize[3];

      /**
       * Whether a variable work group size has been specified as defined by
       * ARB_compute_variable_group_size.
       */
      bool LocalSizeVariable;

      /*
       * Arrangement of invocations used to calculate derivatives in a compute
       * shader.  From NV_compute_shader_derivatives.
       */
      enum gl_derivative_group DerivativeGroup;
   } Comp;
};

/**
 * Compile status enum. COMPILE_SKIPPED is used to indicate the compile
 * was skipped due to the shader matching one that's been seen before by
 * the on-disk cache.
 */
enum gl_compile_status
{
   COMPILE_FAILURE = 0,
   COMPILE_SUCCESS,
   COMPILE_SKIPPED
};

/**
 * A GLSL shader object.
 */
struct gl_shader
{
   /** GL_FRAGMENT_SHADER || GL_VERTEX_SHADER || GL_GEOMETRY_SHADER_ARB ||
    *  GL_TESS_CONTROL_SHADER || GL_TESS_EVALUATION_SHADER.
    * Must be the first field.
    */
   GLenum16 Type;
   gl_shader_stage Stage;
   GLuint Name;  /**< AKA the handle */
   GLint RefCount;  /**< Reference count */
   GLchar *Label;   /**< GL_KHR_debug */
   GLboolean DeletePending;
   bool IsES;              /**< True if this shader uses GLSL ES */

   enum gl_compile_status CompileStatus;

   /** SHA1 of the pre-processed source used by the disk cache. */
   uint8_t disk_cache_sha1[SHA1_DIGEST_LENGTH];
   /** SHA1 of the original source before replacement, set by glShaderSource. */
   uint8_t source_sha1[SHA1_DIGEST_LENGTH];
   /** SHA1 of FallbackSource (a copy of some original source before replacement). */
   uint8_t fallback_source_sha1[SHA1_DIGEST_LENGTH];
   /** SHA1 of the current compiled source, set by successful glCompileShader. */
   uint8_t compiled_source_sha1[SHA1_DIGEST_LENGTH];

   const GLchar *Source;  /**< Source code string */
   const GLchar *FallbackSource;  /**< Fallback string used by on-disk cache*/

   GLchar *InfoLog;

   unsigned Version;       /**< GLSL version used for linking */

   /**
    * A bitmask of gl_advanced_blend_mode values
    */
   GLbitfield BlendSupport;

   struct exec_list *ir;
   struct glsl_symbol_table *symbols;

   /**
    * Whether early fragment tests are enabled as defined by
    * ARB_shader_image_load_store.
    */
   bool EarlyFragmentTests;

   bool ARB_fragment_coord_conventions_enable;
   bool OES_geometry_point_size_enable;
   bool OES_tessellation_point_size_enable;

   bool redeclares_gl_fragcoord;
   bool uses_gl_fragcoord;

   bool PostDepthCoverage;
   bool PixelInterlockOrdered;
   bool PixelInterlockUnordered;
   bool SampleInterlockOrdered;
   bool SampleInterlockUnordered;
   bool InnerCoverage;

   /**
    * Fragment shader state from GLSL 1.50 layout qualifiers.
    */
   bool origin_upper_left;
   bool pixel_center_integer;

   /**
    * Whether bindless_sampler/bindless_image, and respectively
    * bound_sampler/bound_image are declared at global scope as defined by
    * ARB_bindless_texture.
    */
   bool bindless_sampler;
   bool bindless_image;
   bool bound_sampler;
   bool bound_image;

   /**
    * Whether layer output is viewport-relative.
    */
   bool redeclares_gl_layer;
   bool layer_viewport_relative;

   /** Global xfb_stride out qualifier if any */
   GLuint TransformFeedbackBufferStride[MAX_FEEDBACK_BUFFERS];

   struct gl_shader_info info;

   /* ARB_gl_spirv related data */
   struct gl_shader_spirv_data *spirv_data;
};

/**
 * A linked GLSL shader object.
 */
struct gl_linked_shader
{
   gl_shader_stage Stage;

   /** All gl_shader::compiled_source_sha1 combined. */
   uint8_t linked_source_sha1[SHA1_DIGEST_LENGTH];

   struct gl_program *Program;  /**< Post-compile assembly code */

   /**
    * \name Sampler tracking
    *
    * \note Each of these fields is only set post-linking.
    */
   /*@{*/
   GLbitfield shadow_samplers;	/**< Samplers used for shadow sampling. */
   /*@}*/

   /**
    * Number of default uniform block components used by this shader.
    *
    * This field is only set post-linking.
    */
   unsigned num_uniform_components;

   /**
    * Number of combined uniform components used by this shader.
    *
    * This field is only set post-linking.  It is the sum of the uniform block
    * sizes divided by sizeof(float), and num_uniform_compoennts.
    */
   unsigned num_combined_uniform_components;

   struct exec_list *ir;
   struct glsl_symbol_table *symbols;

   /**
    * ARB_gl_spirv related data.
    *
    * This is actually a reference to the gl_shader::spirv_data, which
    * stores information that is also needed during linking.
    */
   struct gl_shader_spirv_data *spirv_data;
};


/**
 * Link status enum. LINKING_SKIPPED is used to indicate linking
 * was skipped due to the shader being loaded from the on-disk cache.
 */
enum gl_link_status
{
   LINKING_FAILURE = 0,
   LINKING_SUCCESS,
   LINKING_SKIPPED
};

/* All GLSL program resource types are next to each other, so we can use that
 * to make them 0-based like this:
 */
#define GET_PROGRAM_RESOURCE_TYPE_FROM_GLENUM(x) ((x) - GL_UNIFORM)
#define NUM_PROGRAM_RESOURCE_TYPES (GL_TRANSFORM_FEEDBACK_VARYING - GL_UNIFORM + 1)

/**
 * A data structure to be shared by gl_shader_program and gl_program.
 */
struct gl_shader_program_data
{
   GLint RefCount;  /**< Reference count */

   /** SHA1 hash of linked shader program */
   unsigned char sha1[20];

   unsigned NumUniformStorage;
   unsigned NumHiddenUniforms;
   struct gl_uniform_storage *UniformStorage;

   unsigned NumUniformBlocks;
   unsigned NumShaderStorageBlocks;

   struct gl_uniform_block *UniformBlocks;
   struct gl_uniform_block *ShaderStorageBlocks;

   struct gl_active_atomic_buffer *AtomicBuffers;
   unsigned NumAtomicBuffers;

   /* Shader cache variables used during restore */
   unsigned NumUniformDataSlots;
   union gl_constant_value *UniformDataSlots;

   /* Used to hold initial uniform values for program binary restores.
    *
    * From the ARB_get_program_binary spec:
    *
    *    "A successful call to ProgramBinary will reset all uniform
    *    variables to their initial values. The initial value is either
    *    the value of the variable's initializer as specified in the
    *    original shader source, or 0 if no initializer was present.
    */
   union gl_constant_value *UniformDataDefaults;

   /** Hash for quick search by name. */
   struct hash_table *ProgramResourceHash[NUM_PROGRAM_RESOURCE_TYPES];

   GLboolean Validated;

   /** List of all active resources after linking. */
   struct gl_program_resource *ProgramResourceList;
   unsigned NumProgramResourceList;

   enum gl_link_status LinkStatus;   /**< GL_LINK_STATUS */
   GLchar *InfoLog;

   unsigned Version;       /**< GLSL version used for linking */

   /* Mask of stages this program was linked against */
   unsigned linked_stages;

   /* Whether the shaders of this program are loaded from SPIR-V binaries
    * (all have the SPIR_V_BINARY_ARB state). This was introduced by the
    * ARB_gl_spirv extension.
    */
   bool spirv;
};

/**
 * A GLSL program object.
 * Basically a linked collection of vertex and fragment shaders.
 */
struct gl_shader_program
{
   GLenum16 Type;   /**< Always GL_SHADER_PROGRAM (internal token) */
   GLuint Name;  /**< aka handle or ID */
   GLchar *Label;   /**< GL_KHR_debug */
   GLint RefCount;  /**< Reference count */
   GLboolean DeletePending;

   /**
    * Is the application intending to glGetProgramBinary this program?
    *
    * BinaryRetrievableHint is the currently active hint that gets set
    * during initialization and after linking and BinaryRetrievableHintPending
    * is the hint set by the user to be active when program is linked next time.
    */
   GLboolean BinaryRetrievableHint;
   GLboolean BinaryRetrievableHintPending;

   /**
    * Indicates whether program can be bound for individual pipeline stages
    * using UseProgramStages after it is next linked.
    */
   GLboolean SeparateShader;

   GLuint NumShaders;          /**< number of attached shaders */
   struct gl_shader **Shaders; /**< List of attached the shaders */

   /**
    * User-defined attribute bindings
    *
    * These are set via \c glBindAttribLocation and are used to direct the
    * GLSL linker.  These are \b not the values used in the compiled shader,
    * and they are \b not the values returned by \c glGetAttribLocation.
    */
   struct string_to_uint_map *AttributeBindings;

   /**
    * User-defined fragment data bindings
    *
    * These are set via \c glBindFragDataLocation and are used to direct the
    * GLSL linker.  These are \b not the values used in the compiled shader,
    * and they are \b not the values returned by \c glGetFragDataLocation.
    */
   struct string_to_uint_map *FragDataBindings;
   struct string_to_uint_map *FragDataIndexBindings;

   /**
    * Transform feedback varyings last specified by
    * glTransformFeedbackVaryings().
    *
    * For the current set of transform feedback varyings used for transform
    * feedback output, see LinkedTransformFeedback.
    */
   struct {
      GLenum16 BufferMode;
      /** Global xfb_stride out qualifier if any */
      GLuint BufferStride[MAX_FEEDBACK_BUFFERS];
      GLuint NumVarying;
      GLchar **VaryingNames;  /**< Array [NumVarying] of char * */
   } TransformFeedback;

   struct gl_program *last_vert_prog;

   /** Post-link gl_FragDepth layout for ARB_conservative_depth. */
   enum gl_frag_depth_layout FragDepthLayout;

   /**
    * Geometry shader state - copied into gl_program by
    * _mesa_copy_linked_program_data().
    */
   struct {
      GLint VerticesIn;

      bool UsesEndPrimitive;
      unsigned ActiveStreamMask;
   } Geom;

   /** Data shared by gl_program and gl_shader_program */
   struct gl_shader_program_data *data;

   /**
    * Mapping from GL uniform locations returned by \c glUniformLocation to
    * UniformStorage entries. Arrays will have multiple contiguous slots
    * in the UniformRemapTable, all pointing to the same UniformStorage entry.
    */
   unsigned NumUniformRemapTable;
   struct gl_uniform_storage **UniformRemapTable;

   /**
    * Sometimes there are empty slots left over in UniformRemapTable after we
    * allocate slots to explicit locations. This list stores the blocks of
    * continuous empty slots inside UniformRemapTable.
    */
   struct exec_list EmptyUniformLocations;

   /**
    * Total number of explicit uniform location including inactive uniforms.
    */
   unsigned NumExplicitUniformLocations;

   /**
    * Map of active uniform names to locations
    *
    * Maps any active uniform that is not an array element to a location.
    * Each active uniform, including individual structure members will appear
    * in this map.  This roughly corresponds to the set of names that would be
    * enumerated by \c glGetActiveUniform.
    */
   struct string_to_uint_map *UniformHash;

   GLboolean SamplersValidated; /**< Samplers validated against texture units? */

   bool IsES;              /**< True if this program uses GLSL ES */

   /**
    * Per-stage shaders resulting from the first stage of linking.
    *
    * Set of linked shaders for this program.  The array is accessed using the
    * \c MESA_SHADER_* defines.  Entries for non-existent stages will be
    * \c NULL.
    */
   struct gl_linked_shader *_LinkedShaders[MESA_SHADER_STAGES];

   /**
    * True if any of the fragment shaders attached to this program use:
    * #extension ARB_fragment_coord_conventions: enable
    */
   GLboolean ARB_fragment_coord_conventions_enable;
};

/**
 * Base class for any kind of program object
 */
struct gl_program
{
   /** FIXME: This must be first until we split shader_info from nir_shader */
   struct shader_info info;

   GLuint Id;
   GLint RefCount;
   GLubyte *String;  /**< Null-terminated program text */

   /** GL_VERTEX/FRAGMENT_PROGRAM_ARB, GL_GEOMETRY_PROGRAM_NV */
   GLenum16 Target;
   GLenum16 Format;    /**< String encoding format */

   GLboolean _Used;        /**< Ever used for drawing? Used for debugging */

   struct nir_shader *nir;

   /* Saved and restored with metadata. Freed with ralloc. */
   void *driver_cache_blob;
   size_t driver_cache_blob_size;

   /** Is this program written to on disk shader cache */
   bool program_written_to_cache;

   /** whether to skip VARYING_SLOT_PSIZ in st_translate_stream_output_info() */
   bool skip_pointsize_xfb;

   /** A bitfield indicating which vertex shader inputs consume two slots
    *
    * This is used for mapping from single-slot input locations in the GL API
    * to dual-slot double input locations in the shader.  This field is set
    * once as part of linking and never updated again to ensure the mapping
    * remains consistent.
    *
    * Note: There may be dual-slot variables in the original shader source
    * which do not appear in this bitfield due to having been eliminated by
    * the compiler prior to DualSlotInputs being calculated.  There may also
    * be bits set in this bitfield which are set but which the shader never
    * reads due to compiler optimizations eliminating such variables after
    * DualSlotInputs is calculated.
    */
   GLbitfield64 DualSlotInputs;
   /** Subset of OutputsWritten outputs written with non-zero index. */
   GLbitfield64 SecondaryOutputsWritten;
   /** TEXTURE_x_BIT bitmask */
   GLbitfield16 TexturesUsed[MAX_COMBINED_TEXTURE_IMAGE_UNITS];
   /** Bitfield of which samplers are used */
   GLbitfield SamplersUsed;
   /** Texture units used for shadow sampling. */
   GLbitfield ShadowSamplers;
   /** Texture units used for samplerExternalOES */
   GLbitfield ExternalSamplersUsed;

   /** Named parameters, constants, etc. from program text */
   struct gl_program_parameter_list *Parameters;

   /** Map from sampler unit to texture unit (set by glUniform1i()) */
   GLubyte SamplerUnits[MAX_SAMPLERS];

   struct pipe_shader_state state;
   struct ati_fragment_shader *ati_fs;
   uint64_t affected_states; /**< ST_NEW_* flags to mark dirty when binding */

   void *serialized_nir;
   unsigned serialized_nir_size;

   struct gl_shader_program *shader_program;

   struct st_variant *variants;

   union {
      /** Fields used by GLSL programs */
      struct {
         /** Data shared by gl_program and gl_shader_program */
         struct gl_shader_program_data *data;

         struct gl_active_atomic_buffer **AtomicBuffers;

         /** Post-link transform feedback info. */
         struct gl_transform_feedback_info *LinkedTransformFeedback;

         /**
          * Number of types for subroutine uniforms.
          */
         GLuint NumSubroutineUniformTypes;

         /**
          * Subroutine uniform remap table
          * based on the program level uniform remap table.
          */
         GLuint NumSubroutineUniforms; /* non-sparse total */
         GLuint NumSubroutineUniformRemapTable;
         struct gl_uniform_storage **SubroutineUniformRemapTable;

         /**
          * Num of subroutine functions for this stage and storage for them.
          */
         GLuint NumSubroutineFunctions;
         GLuint MaxSubroutineFunctionIndex;
         struct gl_subroutine_function *SubroutineFunctions;

         /**
          * Map from image uniform index to image unit (set by glUniform1i())
          *
          * An image uniform index is associated with each image uniform by
          * the linker.  The image index associated with each uniform is
          * stored in the \c gl_uniform_storage::image field.
          */
         GLubyte ImageUnits[MAX_IMAGE_UNIFORMS];

         /** Access qualifier from linked shader
          */
         enum gl_access_qualifier image_access[MAX_IMAGE_UNIFORMS];

         GLuint NumUniformBlocks;
         struct gl_uniform_block **UniformBlocks;
         struct gl_uniform_block **ShaderStorageBlocks;

         /**
          * Bitmask of shader storage blocks not declared as read-only.
          */
         unsigned ShaderStorageBlocksWriteAccess;

         /** Which texture target is being sampled
          * (TEXTURE_1D/2D/3D/etc_INDEX)
          */
         GLubyte SamplerTargets[MAX_SAMPLERS];

         /**
          * Number of samplers declared with the bindless_sampler layout
          * qualifier as specified by ARB_bindless_texture.
          */
         GLuint NumBindlessSamplers;
         GLboolean HasBoundBindlessSampler;
         struct gl_bindless_sampler *BindlessSamplers;

         /**
          * Number of images declared with the bindless_image layout qualifier
          * as specified by ARB_bindless_texture.
          */
         GLuint NumBindlessImages;
         GLboolean HasBoundBindlessImage;
         struct gl_bindless_image *BindlessImages;
      } sh;

      /** ARB assembly-style program fields */
      struct {
         struct prog_instruction *Instructions;

         /**
          * Local parameters used by the program.
          *
          * It's dynamically allocated because it is rarely used (just
          * assembly-style programs), and MAX_PROGRAM_LOCAL_PARAMS entries
          * once it's allocated.
          */
         GLfloat (*LocalParams)[4];
         unsigned MaxLocalParams;

         /** Bitmask of which register files are read/written with indirect
          * addressing.  Mask of (1 << PROGRAM_x) bits.
          */
         GLbitfield IndirectRegisterFiles;

         /** Logical counts */
         /*@{*/
         GLuint NumInstructions;
         GLuint NumTemporaries;
         GLuint NumParameters;
         GLuint NumAttributes;
         GLuint NumAddressRegs;
         GLuint NumAluInstructions;
         GLuint NumTexInstructions;
         GLuint NumTexIndirections;
         /*@}*/
         /** Native, actual h/w counts */
         /*@{*/
         GLuint NumNativeInstructions;
         GLuint NumNativeTemporaries;
         GLuint NumNativeParameters;
         GLuint NumNativeAttributes;
         GLuint NumNativeAddressRegs;
         GLuint NumNativeAluInstructions;
         GLuint NumNativeTexInstructions;
         GLuint NumNativeTexIndirections;
         /*@}*/

         /** Used by ARB assembly-style programs. Can only be true for vertex
          * programs.
          */
         GLboolean IsPositionInvariant;
      } arb;
   };
};

/*
 * State/IR translators needs to store some extra vp info.
 */
struct gl_vertex_program
{
   struct gl_program Base;

   uint32_t vert_attrib_mask; /**< mask of sourced vertex attribs */
   ubyte num_inputs;

   /** Maps VARYING_SLOT_x to slot */
   ubyte result_to_output[VARYING_SLOT_MAX];
};

/**
 * Structure that represents a reference to an atomic buffer from some
 * shader program.
 */
struct gl_active_atomic_buffer
{
   /** Uniform indices of the atomic counters declared within it. */
   GLuint *Uniforms;
   GLuint NumUniforms;

   /** Binding point index associated with it. */
   GLuint Binding;

   /** Minimum reasonable size it is expected to have. */
   GLuint MinimumSize;

   /** Shader stages making use of it. */
   GLboolean StageReferences[MESA_SHADER_STAGES];
};

struct gl_transform_feedback_varying_info
{
   struct gl_resource_name name;
   GLenum16 Type;
   GLint BufferIndex;
   GLint Size;
   GLint Offset;
};


/**
 * Per-output info vertex shaders for transform feedback.
 */
struct gl_transform_feedback_output
{
   uint32_t OutputRegister;
   uint32_t OutputBuffer;
   uint32_t NumComponents;
   uint32_t StreamId;

   /** offset (in DWORDs) of this output within the interleaved structure */
   uint32_t DstOffset;

   /**
    * Offset into the output register of the data to output.  For example,
    * if NumComponents is 2 and ComponentOffset is 1, then the data to
    * offset is in the y and z components of the output register.
    */
   uint32_t ComponentOffset;
};


struct gl_transform_feedback_buffer
{
   uint32_t Binding;

   uint32_t NumVaryings;

   /**
    * Total number of components stored in each buffer.  This may be used by
    * hardware back-ends to determine the correct stride when interleaving
    * multiple transform feedback outputs in the same buffer.
    */
   uint32_t Stride;

   /**
    * Which transform feedback stream this buffer binding is associated with.
    */
   uint32_t Stream;
};


/** Post-link transform feedback info. */
struct gl_transform_feedback_info
{
   unsigned NumOutputs;

   /* Bitmask of active buffer indices. */
   unsigned ActiveBuffers;

   struct gl_transform_feedback_output *Outputs;

   /** Transform feedback varyings used for the linking of this shader program.
    *
    * Use for glGetTransformFeedbackVarying().
    */
   struct gl_transform_feedback_varying_info *Varyings;
   GLint NumVarying;

   struct gl_transform_feedback_buffer Buffers[MAX_FEEDBACK_BUFFERS];
};

/**
 *  Shader subroutine function definition
 */
struct gl_subroutine_function
{
   struct gl_resource_name name;
   int index;
   int num_compat_types;
   const struct glsl_type **types;
};

/**
 * Active resource in a gl_shader_program
 */
struct gl_program_resource
{
   GLenum16 Type; /** Program interface type. */
   const void *Data; /** Pointer to resource associated data structure. */
   uint8_t StageReferences; /** Bitmask of shader stage references. */
};

struct gl_uniform_buffer_variable
{
   char *Name;

   /**
    * Name of the uniform as seen by glGetUniformIndices.
    *
    * glGetUniformIndices requires that the block instance index \b not be
    * present in the name of queried uniforms.
    *
    * \note
    * \c gl_uniform_buffer_variable::IndexName and
    * \c gl_uniform_buffer_variable::Name may point to identical storage.
    */
   char *IndexName;

   const struct glsl_type *Type;
   unsigned int Offset;
   GLboolean RowMajor;
};


struct gl_uniform_block
{
   /** Declared name of the uniform block */
   struct gl_resource_name name;

   /** Array of supplemental information about UBO ir_variables. */
   struct gl_uniform_buffer_variable *Uniforms;
   GLuint NumUniforms;

   /**
    * Index (GL_UNIFORM_BLOCK_BINDING) into ctx->UniformBufferBindings[] to use
    * with glBindBufferBase to bind a buffer object to this uniform block.
    */
   GLuint Binding;

   /**
    * Minimum size (in bytes) of a buffer object to back this uniform buffer
    * (GL_UNIFORM_BLOCK_DATA_SIZE).
    */
   GLuint UniformBufferSize;

   /** Stages that reference this block */
   uint8_t stageref;

   /**
    * Linearized array index for uniform block instance arrays
    *
    * Given a uniform block instance array declared with size
    * blk[s_0][s_1]..[s_m], the block referenced by blk[i_0][i_1]..[i_m] will
    * have the linearized array index
    *
    *           m-1       m
    *     i_m + ∑   i_j * ∏     s_k
    *           j=0       k=j+1
    *
    * For a uniform block instance that is not an array, this is always 0.
    */
   uint8_t linearized_array_index;

   /**
    * Layout specified in the shader
    *
    * This isn't accessible through the API, but it is used while
    * cross-validating uniform blocks.
    */
   enum glsl_interface_packing _Packing;
   GLboolean _RowMajor;
};

/**
 * A bindless sampler object.
 */
struct gl_bindless_sampler
{
   /** Texture unit (set by glUniform1()). */
   GLubyte unit;

   /** Whether this bindless sampler is bound to a unit. */
   GLboolean bound;

   /** Texture Target (TEXTURE_1D/2D/3D/etc_INDEX). */
   gl_texture_index target;

   /** Pointer to the base of the data. */
   GLvoid *data;
};


/**
 * A bindless image object.
 */
struct gl_bindless_image
{
   /** Image unit (set by glUniform1()). */
   GLubyte unit;

   /** Whether this bindless image is bound to a unit. */
   GLboolean bound;

   /** Access qualifier from linked shader
    */
   enum gl_access_qualifier image_access;

   /** Pointer to the base of the data. */
   GLvoid *data;
};

/**
 * Data container for shader queries. This holds only the minimal
 * amount of required information for resource queries to work.
 */
struct gl_shader_variable
{
   /**
    * Declared type of the variable
    */
   const struct glsl_type *type;

   /**
    * If the variable is in an interface block, this is the type of the block.
    */
   const struct glsl_type *interface_type;

   /**
    * For variables inside structs (possibly recursively), this is the
    * outermost struct type.
    */
   const struct glsl_type *outermost_struct_type;

   /**
    * Declared name of the variable
    */
   struct gl_resource_name name;

   /**
    * Storage location of the base of this variable
    *
    * The precise meaning of this field depends on the nature of the variable.
    *
    *   - Vertex shader input: one of the values from \c gl_vert_attrib.
    *   - Vertex shader output: one of the values from \c gl_varying_slot.
    *   - Geometry shader input: one of the values from \c gl_varying_slot.
    *   - Geometry shader output: one of the values from \c gl_varying_slot.
    *   - Fragment shader input: one of the values from \c gl_varying_slot.
    *   - Fragment shader output: one of the values from \c gl_frag_result.
    *   - Uniforms: Per-stage uniform slot number for default uniform block.
    *   - Uniforms: Index within the uniform block definition for UBO members.
    *   - Non-UBO Uniforms: explicit location until linking then reused to
    *     store uniform slot number.
    *   - Other: This field is not currently used.
    *
    * If the variable is a uniform, shader input, or shader output, and the
    * slot has not been assigned, the value will be -1.
    */
   int location;

   /**
    * Specifies the first component the variable is stored in as per
    * ARB_enhanced_layouts.
    */
   unsigned component:2;

   /**
    * Output index for dual source blending.
    *
    * \note
    * The GLSL spec only allows the values 0 or 1 for the index in \b dual
    * source blending.
    */
   unsigned index:1;

   /**
    * Specifies whether a shader input/output is per-patch in tessellation
    * shader stages.
    */
   unsigned patch:1;

   /**
    * Storage class of the variable.
    *
    * \sa (n)ir_variable_mode
    */
   unsigned mode:4;

   /**
    * Interpolation mode for shader inputs / outputs
    *
    * \sa glsl_interp_mode
    */
   unsigned interpolation:2;

   /**
    * Was the location explicitly set in the shader?
    *
    * If the location is explicitly set in the shader, it \b cannot be changed
    * by the linker or by the API (e.g., calls to \c glBindAttribLocation have
    * no effect).
    */
   unsigned explicit_location:1;

   /**
    * Precision qualifier.
    */
   unsigned precision:2;
};

#endif
