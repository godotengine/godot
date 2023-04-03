/*
 * Copyright © 2012 Intel Corporation
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
 */

/** \file glthread_marshal.h
 *
 * Declarations of functions related to marshalling GL calls from a client
 * thread to a server thread.
 */

#ifndef MARSHAL_H
#define MARSHAL_H

#include "main/glthread.h"
#include "main/context.h"
#include "main/macros.h"
#include "marshal_generated.h"

struct marshal_cmd_base
{
   /**
    * Type of command.  See enum marshal_dispatch_cmd_id.
    */
   uint16_t cmd_id;

   /**
    * Number of uint64_t elements used by the command.
    */
   uint16_t cmd_size;
};

typedef uint32_t (*_mesa_unmarshal_func)(struct gl_context *ctx, const void *cmd);
extern const _mesa_unmarshal_func _mesa_unmarshal_dispatch[NUM_DISPATCH_CMD];

static inline void *
_mesa_glthread_allocate_command(struct gl_context *ctx,
                                uint16_t cmd_id,
                                unsigned size)
{
   struct glthread_state *glthread = &ctx->GLThread;
   const unsigned num_elements = align(size, 8) / 8;

   assert (num_elements <= MARSHAL_MAX_CMD_SIZE / 8);

   if (unlikely(glthread->used + num_elements > MARSHAL_MAX_CMD_SIZE / 8))
      _mesa_glthread_flush_batch(ctx);

   struct glthread_batch *next = glthread->next_batch;
   struct marshal_cmd_base *cmd_base =
      (struct marshal_cmd_base *)&next->buffer[glthread->used];
   glthread->used += num_elements;
   cmd_base->cmd_id = cmd_id;
   cmd_base->cmd_size = num_elements;
   return cmd_base;
}

static inline bool
_mesa_glthread_call_is_last(struct glthread_state *glthread,
                            struct marshal_cmd_base *last)
{
   return last &&
          (uint64_t*)last + last->cmd_size ==
          &glthread->next_batch->buffer[glthread->used];
}

static inline bool
_mesa_glthread_has_no_pack_buffer(const struct gl_context *ctx)
{
   return ctx->GLThread.CurrentPixelPackBufferName == 0;
}

static inline bool
_mesa_glthread_has_no_unpack_buffer(const struct gl_context *ctx)
{
   return ctx->GLThread.CurrentPixelUnpackBufferName == 0;
}

/**
 * Instead of conditionally handling marshaling immediate index data in draw
 * calls (deprecated and removed in GL core), we just disable threading.
 */
static inline bool
_mesa_glthread_has_non_vbo_vertices_or_indices(const struct gl_context *ctx)
{
   const struct glthread_state *glthread = &ctx->GLThread;
   struct glthread_vao *vao = glthread->CurrentVAO;

   return ctx->API != API_OPENGL_CORE &&
          (vao->CurrentElementBufferName == 0 ||
           (vao->UserPointerMask & vao->BufferEnabled));
}

static inline bool
_mesa_glthread_has_non_vbo_vertices(const struct gl_context *ctx)
{
   const struct glthread_state *glthread = &ctx->GLThread;
   const struct glthread_vao *vao = glthread->CurrentVAO;

   return ctx->API != API_OPENGL_CORE &&
          (vao->UserPointerMask & vao->BufferEnabled);
}

static inline bool
_mesa_glthread_has_non_vbo_vertices_or_indirect(const struct gl_context *ctx)
{
   const struct glthread_state *glthread = &ctx->GLThread;
   const struct glthread_vao *vao = glthread->CurrentVAO;

   return ctx->API != API_OPENGL_CORE &&
          (glthread->CurrentDrawIndirectBufferName == 0 ||
           (vao->UserPointerMask & vao->BufferEnabled));
}

static inline bool
_mesa_glthread_has_non_vbo_vertices_or_indices_or_indirect(const struct gl_context *ctx)
{
   const struct glthread_state *glthread = &ctx->GLThread;
   struct glthread_vao *vao = glthread->CurrentVAO;

   return ctx->API != API_OPENGL_CORE &&
          (glthread->CurrentDrawIndirectBufferName == 0 ||
           vao->CurrentElementBufferName == 0 ||
           (vao->UserPointerMask & vao->BufferEnabled));
}

static inline unsigned
_mesa_buffer_enum_to_count(GLenum buffer)
{
   switch (buffer) {
   case GL_COLOR:
      return 4;
   case GL_DEPTH_STENCIL:
      return 2;
   case GL_STENCIL:
   case GL_DEPTH:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_tex_param_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_TEXTURE_MIN_FILTER:
   case GL_TEXTURE_MAG_FILTER:
   case GL_TEXTURE_WRAP_S:
   case GL_TEXTURE_WRAP_T:
   case GL_TEXTURE_WRAP_R:
   case GL_TEXTURE_BASE_LEVEL:
   case GL_TEXTURE_MAX_LEVEL:
   case GL_GENERATE_MIPMAP_SGIS:
   case GL_TEXTURE_COMPARE_MODE_ARB:
   case GL_TEXTURE_COMPARE_FUNC_ARB:
   case GL_DEPTH_TEXTURE_MODE_ARB:
   case GL_DEPTH_STENCIL_TEXTURE_MODE:
   case GL_TEXTURE_SRGB_DECODE_EXT:
   case GL_TEXTURE_REDUCTION_MODE_EXT:
   case GL_TEXTURE_CUBE_MAP_SEAMLESS:
   case GL_TEXTURE_SWIZZLE_R:
   case GL_TEXTURE_SWIZZLE_G:
   case GL_TEXTURE_SWIZZLE_B:
   case GL_TEXTURE_SWIZZLE_A:
   case GL_TEXTURE_MIN_LOD:
   case GL_TEXTURE_MAX_LOD:
   case GL_TEXTURE_PRIORITY:
   case GL_TEXTURE_MAX_ANISOTROPY_EXT:
   case GL_TEXTURE_LOD_BIAS:
   case GL_TEXTURE_TILING_EXT:
   case GL_TEXTURE_SPARSE_ARB:
   case GL_VIRTUAL_PAGE_SIZE_INDEX_ARB:
   case GL_NUM_SPARSE_LEVELS_ARB:
      return 1;
   case GL_TEXTURE_CROP_RECT_OES:
   case GL_TEXTURE_SWIZZLE_RGBA:
   case GL_TEXTURE_BORDER_COLOR:
      return 4;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_fog_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_FOG_MODE:
   case GL_FOG_DENSITY:
   case GL_FOG_START:
   case GL_FOG_END:
   case GL_FOG_INDEX:
   case GL_FOG_COORDINATE_SOURCE_EXT:
   case GL_FOG_DISTANCE_MODE_NV:
      return 1;
   case GL_FOG_COLOR:
      return 4;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_light_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_AMBIENT:
   case GL_DIFFUSE:
   case GL_SPECULAR:
   case GL_POSITION:
      return 4;
   case GL_SPOT_DIRECTION:
      return 3;
   case GL_SPOT_EXPONENT:
   case GL_SPOT_CUTOFF:
   case GL_CONSTANT_ATTENUATION:
   case GL_LINEAR_ATTENUATION:
   case GL_QUADRATIC_ATTENUATION:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_light_model_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_LIGHT_MODEL_AMBIENT:
      return 4;
   case GL_LIGHT_MODEL_LOCAL_VIEWER:
   case GL_LIGHT_MODEL_TWO_SIDE:
   case GL_LIGHT_MODEL_COLOR_CONTROL:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_texenv_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_TEXTURE_ENV_MODE:
   case GL_COMBINE_RGB:
   case GL_COMBINE_ALPHA:
   case GL_SOURCE0_RGB:
   case GL_SOURCE1_RGB:
   case GL_SOURCE2_RGB:
   case GL_SOURCE3_RGB_NV:
   case GL_SOURCE0_ALPHA:
   case GL_SOURCE1_ALPHA:
   case GL_SOURCE2_ALPHA:
   case GL_SOURCE3_ALPHA_NV:
   case GL_OPERAND0_RGB:
   case GL_OPERAND1_RGB:
   case GL_OPERAND2_RGB:
   case GL_OPERAND3_RGB_NV:
   case GL_OPERAND0_ALPHA:
   case GL_OPERAND1_ALPHA:
   case GL_OPERAND2_ALPHA:
   case GL_OPERAND3_ALPHA_NV:
   case GL_RGB_SCALE:
   case GL_ALPHA_SCALE:
   case GL_TEXTURE_LOD_BIAS_EXT:
   case GL_COORD_REPLACE:
      return 1;
   case GL_TEXTURE_ENV_COLOR:
      return 4;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_texgen_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_TEXTURE_GEN_MODE:
      return 1;
   case GL_OBJECT_PLANE:
   case GL_EYE_PLANE:
      return 4;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_material_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_EMISSION:
   case GL_AMBIENT:
   case GL_DIFFUSE:
   case GL_SPECULAR:
   case GL_AMBIENT_AND_DIFFUSE:
      return 4;
   case GL_COLOR_INDEXES:
      return 3;
   case GL_SHININESS:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_point_param_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_DISTANCE_ATTENUATION_EXT:
      return 3;
   case GL_POINT_SIZE_MIN_EXT:
   case GL_POINT_SIZE_MAX_EXT:
   case GL_POINT_FADE_THRESHOLD_SIZE_EXT:
   case GL_POINT_SPRITE_COORD_ORIGIN:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_calllists_enum_to_count(GLenum type)
{
   switch (type) {
   case GL_BYTE:
   case GL_UNSIGNED_BYTE:
      return 1;
   case GL_SHORT:
   case GL_UNSIGNED_SHORT:
   case GL_2_BYTES:
      return 2;
   case GL_3_BYTES:
      return 3;
   case GL_INT:
   case GL_UNSIGNED_INT:
   case GL_FLOAT:
   case GL_4_BYTES:
      return 4;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_patch_param_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_PATCH_DEFAULT_OUTER_LEVEL:
      return 4;
   case GL_PATCH_DEFAULT_INNER_LEVEL:
      return 2;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_memobj_enum_to_count(GLenum pname)
{
   switch (pname) {
   case GL_DEDICATED_MEMORY_OBJECT_EXT:
      return 1;
   default:
      return 0;
   }
}

static inline unsigned
_mesa_semaphore_enum_to_count(GLenum pname)
{
   switch (pname) {
   /* EXT_semaphore and EXT_semaphore_fd define no parameters */
   default:
      return 0;
   }
}

static inline gl_vert_attrib
_mesa_array_to_attrib(struct gl_context *ctx, GLenum array)
{
   switch (array) {
   case GL_VERTEX_ARRAY:
      return VERT_ATTRIB_POS;
   case GL_NORMAL_ARRAY:
      return VERT_ATTRIB_NORMAL;
   case GL_COLOR_ARRAY:
      return VERT_ATTRIB_COLOR0;
   case GL_INDEX_ARRAY:
      return VERT_ATTRIB_COLOR_INDEX;
   case GL_TEXTURE_COORD_ARRAY:
      return VERT_ATTRIB_TEX(ctx->GLThread.ClientActiveTexture);
   case GL_EDGE_FLAG_ARRAY:
      return VERT_ATTRIB_EDGEFLAG;
   case GL_FOG_COORDINATE_ARRAY:
      return VERT_ATTRIB_FOG;
   case GL_SECONDARY_COLOR_ARRAY:
      return VERT_ATTRIB_COLOR1;
   case GL_POINT_SIZE_ARRAY_OES:
      return VERT_ATTRIB_POINT_SIZE;
   case GL_PRIMITIVE_RESTART_NV:
      return VERT_ATTRIB_PRIMITIVE_RESTART_NV;
   default:
      if (array >= GL_TEXTURE0 && array <= GL_TEXTURE7)
         return VERT_ATTRIB_TEX(array - GL_TEXTURE0);
      return VERT_ATTRIB_MAX;
   }
}

static inline gl_matrix_index
_mesa_get_matrix_index(struct gl_context *ctx, GLenum mode)
{
   if (mode == GL_MODELVIEW || mode == GL_PROJECTION)
      return M_MODELVIEW + (mode - GL_MODELVIEW);

   if (mode == GL_TEXTURE)
      return M_TEXTURE0 + ctx->GLThread.ActiveTexture;

   if (mode >= GL_TEXTURE0 && mode <= GL_TEXTURE0 + MAX_TEXTURE_UNITS - 1)
      return M_TEXTURE0 + (mode - GL_TEXTURE0);

   if (mode >= GL_MATRIX0_ARB && mode <= GL_MATRIX0_ARB + MAX_PROGRAM_MATRICES - 1)
      return M_PROGRAM0 + (mode - GL_MATRIX0_ARB);

   return M_DUMMY;
}

static inline bool
_mesa_matrix_is_identity(const float *m)
{
   static float identity[16] = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
   };

   return !memcmp(m, identity, sizeof(identity));
}

static inline void
_mesa_glthread_Enable(struct gl_context *ctx, GLenum cap)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   switch (cap) {
   case GL_PRIMITIVE_RESTART:
   case GL_PRIMITIVE_RESTART_FIXED_INDEX:
      _mesa_glthread_set_prim_restart(ctx, cap, true);
      break;
   case GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB:
      _mesa_glthread_destroy(ctx, "Enable(DEBUG_OUTPUT_SYNCHRONOUS)");
      break;
   case GL_BLEND:
      ctx->GLThread.Blend = true;
      break;
   case GL_DEPTH_TEST:
      ctx->GLThread.DepthTest = true;
      break;
   case GL_CULL_FACE:
      ctx->GLThread.CullFace = true;
      break;
   case GL_LIGHTING:
      ctx->GLThread.Lighting = true;
      break;
   case GL_POLYGON_STIPPLE:
      ctx->GLThread.PolygonStipple = true;
      break;
   case GL_VERTEX_ARRAY:
   case GL_NORMAL_ARRAY:
   case GL_COLOR_ARRAY:
   case GL_TEXTURE_COORD_ARRAY:
   case GL_INDEX_ARRAY:
   case GL_EDGE_FLAG_ARRAY:
   case GL_FOG_COORDINATE_ARRAY:
   case GL_SECONDARY_COLOR_ARRAY:
   case GL_POINT_SIZE_ARRAY_OES:
      _mesa_glthread_ClientState(ctx, NULL, _mesa_array_to_attrib(ctx, cap),
                                 true);
      break;
   }
}

static inline void
_mesa_glthread_Disable(struct gl_context *ctx, GLenum cap)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   switch (cap) {
   case GL_PRIMITIVE_RESTART:
   case GL_PRIMITIVE_RESTART_FIXED_INDEX:
      _mesa_glthread_set_prim_restart(ctx, cap, false);
      break;
   case GL_BLEND:
      ctx->GLThread.Blend = false;
      break;
   case GL_CULL_FACE:
      ctx->GLThread.CullFace = false;
      break;
   case GL_DEPTH_TEST:
      ctx->GLThread.DepthTest = false;
      break;
   case GL_LIGHTING:
      ctx->GLThread.Lighting = false;
      break;
   case GL_POLYGON_STIPPLE:
      ctx->GLThread.PolygonStipple = false;
      break;
   case GL_VERTEX_ARRAY:
   case GL_NORMAL_ARRAY:
   case GL_COLOR_ARRAY:
   case GL_TEXTURE_COORD_ARRAY:
   case GL_INDEX_ARRAY:
   case GL_EDGE_FLAG_ARRAY:
   case GL_FOG_COORDINATE_ARRAY:
   case GL_SECONDARY_COLOR_ARRAY:
   case GL_POINT_SIZE_ARRAY_OES:
      _mesa_glthread_ClientState(ctx, NULL, _mesa_array_to_attrib(ctx, cap),
                                 false);
      break;
   }
}

static inline int
_mesa_glthread_IsEnabled(struct gl_context *ctx, GLenum cap)
{
   /* This will generate GL_INVALID_OPERATION, as it should. */
   if (ctx->GLThread.inside_begin_end)
      return -1;

   switch (cap) {
   case GL_BLEND:
      return ctx->GLThread.Blend;
   case GL_CULL_FACE:
      return ctx->GLThread.CullFace;
   case GL_DEPTH_TEST:
      return ctx->GLThread.DepthTest;
   case GL_LIGHTING:
      return ctx->GLThread.Lighting;
   case GL_POLYGON_STIPPLE:
      return ctx->GLThread.PolygonStipple;
   case GL_VERTEX_ARRAY:
      return !!(ctx->GLThread.CurrentVAO->UserEnabled & VERT_BIT_POS);
   case GL_NORMAL_ARRAY:
      return !!(ctx->GLThread.CurrentVAO->UserEnabled & VERT_BIT_NORMAL);
   case GL_COLOR_ARRAY:
      return !!(ctx->GLThread.CurrentVAO->UserEnabled & VERT_BIT_COLOR0);
   case GL_TEXTURE_COORD_ARRAY:
      return !!(ctx->GLThread.CurrentVAO->UserEnabled &
                (1 << VERT_ATTRIB_TEX(ctx->GLThread.ClientActiveTexture)));
   default:
      return -1; /* sync and call _mesa_IsEnabled. */
   }
}

static inline void
_mesa_glthread_PushAttrib(struct gl_context *ctx, GLbitfield mask)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (ctx->GLThread.AttribStackDepth >= MAX_ATTRIB_STACK_DEPTH)
      return;

   struct glthread_attrib_node *attr =
      &ctx->GLThread.AttribStack[ctx->GLThread.AttribStackDepth++];

   attr->Mask = mask;

   if (mask & GL_ENABLE_BIT)
      attr->Blend = ctx->GLThread.Blend;

   if (mask & (GL_POLYGON_BIT | GL_ENABLE_BIT)) {
      attr->CullFace = ctx->GLThread.CullFace;
      attr->PolygonStipple = ctx->GLThread.PolygonStipple;
   }

   if (mask & (GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT))
      attr->DepthTest = ctx->GLThread.DepthTest;

   if (mask & (GL_LIGHTING_BIT | GL_ENABLE_BIT))
      attr->Lighting = ctx->GLThread.Lighting;

   if (mask & GL_TEXTURE_BIT)
      attr->ActiveTexture = ctx->GLThread.ActiveTexture;

   if (mask & GL_TRANSFORM_BIT)
      attr->MatrixMode = ctx->GLThread.MatrixMode;
}

static inline void
_mesa_glthread_PopAttrib(struct gl_context *ctx)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (ctx->GLThread.AttribStackDepth == 0)
      return;

   struct glthread_attrib_node *attr =
      &ctx->GLThread.AttribStack[--ctx->GLThread.AttribStackDepth];
   unsigned mask = attr->Mask;

   if (mask & GL_ENABLE_BIT)
      ctx->GLThread.Blend = attr->Blend;

   if (mask & (GL_POLYGON_BIT | GL_ENABLE_BIT)) {
      ctx->GLThread.CullFace = attr->CullFace;
      ctx->GLThread.PolygonStipple = attr->PolygonStipple;
   }

   if (mask & (GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT))
      ctx->GLThread.DepthTest = attr->DepthTest;

   if (mask & (GL_LIGHTING_BIT | GL_ENABLE_BIT))
      ctx->GLThread.Lighting = attr->Lighting;

   if (mask & GL_TEXTURE_BIT)
      ctx->GLThread.ActiveTexture = attr->ActiveTexture;

   if (mask & GL_TRANSFORM_BIT) {
      ctx->GLThread.MatrixMode = attr->MatrixMode;
      ctx->GLThread.MatrixIndex = _mesa_get_matrix_index(ctx, attr->MatrixMode);
   }
}

static bool
is_matrix_stack_full(struct gl_context *ctx, gl_matrix_index idx)
{
   int max_stack_depth = 0;
   if (M_MODELVIEW == ctx->GLThread.MatrixIndex) {
      max_stack_depth = MAX_MODELVIEW_STACK_DEPTH;
   } else if (M_PROJECTION == ctx->GLThread.MatrixIndex) {
      max_stack_depth = MAX_PROJECTION_STACK_DEPTH;
   } else if (M_PROGRAM_LAST >= ctx->GLThread.MatrixIndex) {
      max_stack_depth = MAX_PROGRAM_MATRIX_STACK_DEPTH;
   } else if (M_TEXTURE_LAST >= ctx->GLThread.MatrixIndex) {
      max_stack_depth = MAX_TEXTURE_STACK_DEPTH;
   }
   assert(max_stack_depth);

   if (ctx->GLThread.MatrixStackDepth[idx] + 1 >= max_stack_depth)
      return true;

   return false;
}

static inline void
_mesa_glthread_MatrixPushEXT(struct gl_context *ctx, GLenum matrixMode)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (is_matrix_stack_full(ctx, _mesa_get_matrix_index(ctx, matrixMode)))
      return;

   ctx->GLThread.MatrixStackDepth[_mesa_get_matrix_index(ctx, matrixMode)]++;
}

static inline void
_mesa_glthread_MatrixPopEXT(struct gl_context *ctx, GLenum matrixMode)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (ctx->GLThread.MatrixStackDepth[_mesa_get_matrix_index(ctx, matrixMode)] == 0)
      return;

   ctx->GLThread.MatrixStackDepth[_mesa_get_matrix_index(ctx, matrixMode)]--;
}

static inline void
_mesa_glthread_ActiveTexture(struct gl_context *ctx, GLenum texture)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   ctx->GLThread.ActiveTexture = texture - GL_TEXTURE0;
   if (ctx->GLThread.MatrixMode == GL_TEXTURE)
      ctx->GLThread.MatrixIndex = _mesa_get_matrix_index(ctx, texture);
}

static inline void
_mesa_glthread_PushMatrix(struct gl_context *ctx)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (is_matrix_stack_full(ctx, ctx->GLThread.MatrixIndex))
      return;

   ctx->GLThread.MatrixStackDepth[ctx->GLThread.MatrixIndex]++;
}

static inline void
_mesa_glthread_PopMatrix(struct gl_context *ctx)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (ctx->GLThread.MatrixStackDepth[ctx->GLThread.MatrixIndex] == 0)
      return;

   ctx->GLThread.MatrixStackDepth[ctx->GLThread.MatrixIndex]--;
}

static inline void
_mesa_glthread_MatrixMode(struct gl_context *ctx, GLenum mode)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   ctx->GLThread.MatrixIndex = _mesa_get_matrix_index(ctx, mode);
   ctx->GLThread.MatrixMode = MIN2(mode, 0xffff);
}

static inline void
_mesa_glthread_ListBase(struct gl_context *ctx, GLuint base)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   ctx->GLThread.ListBase = base;
}

static inline void
_mesa_glthread_CallList(struct gl_context *ctx, GLuint list)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   /* Wait for all glEndList and glDeleteLists calls to finish to ensure that
    * all display lists are up to date and the driver thread is not
    * modifiying them. We will be executing them in the application thread.
    */
   int batch = p_atomic_read(&ctx->GLThread.LastDListChangeBatchIndex);
   if (batch != -1) {
      util_queue_fence_wait(&ctx->GLThread.batches[batch].fence);
      p_atomic_set(&ctx->GLThread.LastDListChangeBatchIndex, -1);
   }

   if (!ctx->Shared->DisplayListsAffectGLThread)
      return;

   /* Clear GL_COMPILE_AND_EXECUTE if needed. We only execute here. */
   unsigned saved_mode = ctx->GLThread.ListMode;
   ctx->GLThread.ListMode = 0;

   _mesa_glthread_execute_list(ctx, list);

   ctx->GLThread.ListMode = saved_mode;
}

static inline void
_mesa_glthread_CallLists(struct gl_context *ctx, GLsizei n, GLenum type,
                         const GLvoid *lists)
{
   if (ctx->GLThread.ListMode == GL_COMPILE)
      return;

   if (n <= 0 || !lists)
      return;

   /* Wait for all glEndList and glDeleteLists calls to finish to ensure that
    * all display lists are up to date and the driver thread is not
    * modifiying them. We will be executing them in the application thread.
    */
   int batch = p_atomic_read(&ctx->GLThread.LastDListChangeBatchIndex);
   if (batch != -1) {
      util_queue_fence_wait(&ctx->GLThread.batches[batch].fence);
      p_atomic_set(&ctx->GLThread.LastDListChangeBatchIndex, -1);
   }

   /* Clear GL_COMPILE_AND_EXECUTE if needed. We only execute here. */
   unsigned saved_mode = ctx->GLThread.ListMode;
   ctx->GLThread.ListMode = 0;

   unsigned base = ctx->GLThread.ListBase;

   GLbyte *bptr;
   GLubyte *ubptr;
   GLshort *sptr;
   GLushort *usptr;
   GLint *iptr;
   GLuint *uiptr;
   GLfloat *fptr;

   switch (type) {
   case GL_BYTE:
      bptr = (GLbyte *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + bptr[i]);
      break;
   case GL_UNSIGNED_BYTE:
      ubptr = (GLubyte *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + ubptr[i]);
      break;
   case GL_SHORT:
      sptr = (GLshort *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + sptr[i]);
      break;
   case GL_UNSIGNED_SHORT:
      usptr = (GLushort *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + usptr[i]);
      break;
   case GL_INT:
      iptr = (GLint *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + iptr[i]);
      break;
   case GL_UNSIGNED_INT:
      uiptr = (GLuint *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + uiptr[i]);
      break;
   case GL_FLOAT:
      fptr = (GLfloat *) lists;
      for (unsigned i = 0; i < n; i++)
         _mesa_glthread_CallList(ctx, base + fptr[i]);
      break;
   case GL_2_BYTES:
      ubptr = (GLubyte *) lists;
      for (unsigned i = 0; i < n; i++) {
         _mesa_glthread_CallList(ctx, base +
                                 (GLint)ubptr[2 * i] * 256 +
                                 (GLint)ubptr[2 * i + 1]);
      }
      break;
   case GL_3_BYTES:
      ubptr = (GLubyte *) lists;
      for (unsigned i = 0; i < n; i++) {
         _mesa_glthread_CallList(ctx, base +
                                 (GLint)ubptr[3 * i] * 65536 +
                                 (GLint)ubptr[3 * i + 1] * 256 +
                                 (GLint)ubptr[3 * i + 2]);
      }
      break;
   case GL_4_BYTES:
      ubptr = (GLubyte *) lists;
      for (unsigned i = 0; i < n; i++) {
         _mesa_glthread_CallList(ctx, base +
                                 (GLint)ubptr[4 * i] * 16777216 +
                                 (GLint)ubptr[4 * i + 1] * 65536 +
                                 (GLint)ubptr[4 * i + 2] * 256 +
                                 (GLint)ubptr[4 * i + 3]);
      }
      break;
   }

   ctx->GLThread.ListMode = saved_mode;
}

static inline void
_mesa_glthread_NewList(struct gl_context *ctx, GLuint list, GLenum mode)
{
   if (!ctx->GLThread.ListMode)
      ctx->GLThread.ListMode = MIN2(mode, 0xffff);
}

static inline void
_mesa_glthread_EndList(struct gl_context *ctx)
{
   if (!ctx->GLThread.ListMode)
      return;

   ctx->GLThread.ListMode = 0;

   /* Track the last display list change. */
   p_atomic_set(&ctx->GLThread.LastDListChangeBatchIndex, ctx->GLThread.next);
   _mesa_glthread_flush_batch(ctx);
}

static inline void
_mesa_glthread_DeleteLists(struct gl_context *ctx, GLsizei range)
{
   if (range < 0)
      return;

   /* Track the last display list change. */
   p_atomic_set(&ctx->GLThread.LastDListChangeBatchIndex, ctx->GLThread.next);
   _mesa_glthread_flush_batch(ctx);
}

static inline void
_mesa_glthread_BindFramebuffer(struct gl_context *ctx, GLenum target, GLuint id)
{
   switch (target) {
   case GL_FRAMEBUFFER:
      ctx->GLThread.CurrentDrawFramebuffer = id;
      ctx->GLThread.CurrentReadFramebuffer = id;
      break;
   case GL_DRAW_FRAMEBUFFER:
      ctx->GLThread.CurrentDrawFramebuffer = id;
      break;
   case GL_READ_FRAMEBUFFER:
      ctx->GLThread.CurrentReadFramebuffer = id;
      break;
   }
}

static inline void
_mesa_glthread_DeleteFramebuffers(struct gl_context *ctx, GLsizei n,
                                  const GLuint *ids)
{
   if (ctx->GLThread.CurrentDrawFramebuffer) {
      for (int i = 0; i < n; i++) {
         if (ctx->GLThread.CurrentDrawFramebuffer == ids[i])
            ctx->GLThread.CurrentDrawFramebuffer = 0;
         if (ctx->GLThread.CurrentReadFramebuffer == ids[i])
            ctx->GLThread.CurrentReadFramebuffer = 0;
      }
   }
}

#endif /* MARSHAL_H */
