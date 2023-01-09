/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul   All Rights Reserved.
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
 * \file context.h
 * Mesa context and visual-related functions.
 *
 * There are three large Mesa data types/classes which are meant to be
 * used by device drivers:
 * - struct gl_context: this contains the Mesa rendering state
 * - struct gl_config:  this describes the color buffer (RGB vs. ci), whether
 *   or not there's a depth buffer, stencil buffer, etc.
 * - struct gl_framebuffer:  contains pointers to the depth buffer, stencil
 *   buffer, accum buffer and alpha buffers.
 *
 * These types should be encapsulated by corresponding device driver
 * data types.  See xmesa.h and xmesaP.h for an example.
 *
 * In OOP terms, struct gl_context, struct gl_config, and struct gl_framebuffer
 * are base classes which the device driver must derive from.
 *
 * The following functions create and destroy these data types.
 */


#ifndef CONTEXT_H
#define CONTEXT_H


#include "errors.h"

#include "extensions.h"
#include "mtypes.h"
#include "vbo/vbo.h"


#ifdef __cplusplus
extern "C" {
#endif


struct _glapi_table;


/** \name Context-related functions */
/*@{*/

extern void
_mesa_initialize(const char *extensions_override);

extern GLboolean
_mesa_initialize_context( struct gl_context *ctx,
                          gl_api api,
                          bool no_error,
                          const struct gl_config *visual,
                          struct gl_context *share_list,
                          const struct dd_function_table *driverFunctions);

extern struct _glapi_table *
_mesa_alloc_dispatch_table(bool glthread);

extern void
_mesa_init_dispatch(struct gl_context *ctx);

extern void
_mesa_initialize_dispatch_tables(struct gl_context *ctx);

extern struct _glapi_table *
_mesa_new_nop_table(unsigned numEntries, bool glthread);

extern void
_mesa_free_context_data(struct gl_context *ctx, bool destroy_debug_output);

extern void
_mesa_copy_context(const struct gl_context *src, struct gl_context *dst, GLuint mask);

extern GLboolean
_mesa_make_current( struct gl_context *ctx, struct gl_framebuffer *drawBuffer,
                    struct gl_framebuffer *readBuffer );

extern GLboolean
_mesa_share_state(struct gl_context *ctx, struct gl_context *ctxToShare);

extern struct gl_context *
_mesa_get_current_context(void);

/*@}*/

extern void
_mesa_init_constants(struct gl_constants *consts, gl_api api);

extern void
_mesa_set_context_lost_dispatch(struct gl_context *ctx);



/** \name Miscellaneous */
/*@{*/

extern void
_mesa_flush(struct gl_context *ctx);

/*@}*/


/**
 * Are we currently between glBegin and glEnd?
 * During execution, not display list compilation.
 */
static inline GLboolean
_mesa_inside_begin_end(const struct gl_context *ctx)
{
   return ctx->Driver.CurrentExecPrimitive != PRIM_OUTSIDE_BEGIN_END;
}


/**
 * Are we currently between glBegin and glEnd in a display list?
 */
static inline GLboolean
_mesa_inside_dlist_begin_end(const struct gl_context *ctx)
{
   return ctx->Driver.CurrentSavePrimitive <= PRIM_MAX;
}



/**
 * \name Macros for flushing buffered rendering commands before state changes,
 * checking if inside glBegin/glEnd, etc.
 */
/*@{*/

/**
 * Flush vertices.
 *
 * \param ctx GL context.
 * \param newstate new state.
 *
 * Checks if dd_function_table::NeedFlush is marked to flush stored vertices,
 * and calls dd_function_table::FlushVertices if so. Marks
 * __struct gl_contextRec::NewState with \p newstate.
 */
#define FLUSH_VERTICES(ctx, newstate, pop_attrib_mask)          \
do {								\
   if (MESA_VERBOSE & VERBOSE_STATE)				\
      _mesa_debug(ctx, "FLUSH_VERTICES in %s\n", __func__);	\
   if (ctx->Driver.NeedFlush & FLUSH_STORED_VERTICES)		\
      vbo_exec_FlushVertices(ctx, FLUSH_STORED_VERTICES);	\
   ctx->NewState |= newstate;					\
   ctx->PopAttribState |= pop_attrib_mask;                      \
} while (0)

/**
 * Flush current state.
 *
 * \param ctx GL context.
 * \param newstate new state.
 *
 * Checks if dd_function_table::NeedFlush is marked to flush current state,
 * and calls dd_function_table::FlushVertices if so. Marks
 * __struct gl_contextRec::NewState with \p newstate.
 */
#define FLUSH_CURRENT(ctx, newstate)				\
do {								\
   if (MESA_VERBOSE & VERBOSE_STATE)				\
      _mesa_debug(ctx, "FLUSH_CURRENT in %s\n", __func__);	\
   if (ctx->Driver.NeedFlush & FLUSH_UPDATE_CURRENT)		\
      vbo_exec_FlushVertices(ctx, FLUSH_UPDATE_CURRENT);	\
   ctx->NewState |= newstate;					\
} while (0)

/**
 * Flush vertices.
 *
 * \param ctx GL context.
 *
 * Checks if dd_function_table::NeedFlush is marked to flush stored vertices
 * or current state and calls dd_function_table::FlushVertices if so.
 */
#define FLUSH_FOR_DRAW(ctx)                                     \
do {                                                            \
   if (MESA_VERBOSE & VERBOSE_STATE)                            \
      _mesa_debug(ctx, "FLUSH_FOR_DRAW in %s\n", __func__);     \
   if (ctx->Driver.NeedFlush) {                                 \
      if (ctx->_AllowDrawOutOfOrder) {                          \
          if (ctx->Driver.NeedFlush & FLUSH_UPDATE_CURRENT)     \
             vbo_exec_FlushVertices(ctx, FLUSH_UPDATE_CURRENT); \
      } else {                                                  \
         vbo_exec_FlushVertices(ctx, ctx->Driver.NeedFlush);    \
      }                                                         \
   }                                                            \
} while (0)

/**
 * Macro to assert that the API call was made outside the
 * glBegin()/glEnd() pair, with return value.
 *
 * \param ctx GL context.
 * \param retval value to return in case the assertion fails.
 */
#define ASSERT_OUTSIDE_BEGIN_END_WITH_RETVAL(ctx, retval)		\
do {									\
   if (_mesa_inside_begin_end(ctx)) {					\
      _mesa_error(ctx, GL_INVALID_OPERATION, "Inside glBegin/glEnd");	\
      return retval;							\
   }									\
} while (0)

/**
 * Macro to assert that the API call was made outside the
 * glBegin()/glEnd() pair.
 *
 * \param ctx GL context.
 */
#define ASSERT_OUTSIDE_BEGIN_END(ctx)					\
do {									\
   if (_mesa_inside_begin_end(ctx)) {					\
      _mesa_error(ctx, GL_INVALID_OPERATION, "Inside glBegin/glEnd");	\
      return;								\
   }									\
} while (0)

/*@}*/


/**
 * Checks if the context is for Desktop GL (Compatibility or Core)
 */
static inline bool
_mesa_is_desktop_gl(const struct gl_context *ctx)
{
   return ctx->API == API_OPENGL_COMPAT || ctx->API == API_OPENGL_CORE;
}


/**
 * Checks if the context is for any GLES version
 */
static inline bool
_mesa_is_gles(const struct gl_context *ctx)
{
   return ctx->API == API_OPENGLES || ctx->API == API_OPENGLES2;
}


/**
 * Checks if the context is for GLES 3.0 or later
 */
static inline bool
_mesa_is_gles3(const struct gl_context *ctx)
{
   return ctx->API == API_OPENGLES2 && ctx->Version >= 30;
}


/**
 * Checks if the context is for GLES 3.1 or later
 */
static inline bool
_mesa_is_gles31(const struct gl_context *ctx)
{
   return ctx->API == API_OPENGLES2 && ctx->Version >= 31;
}


/**
 * Checks if the context is for GLES 3.2 or later
 */
static inline bool
_mesa_is_gles32(const struct gl_context *ctx)
{
   return ctx->API == API_OPENGLES2 && ctx->Version >= 32;
}


static inline bool
_mesa_is_no_error_enabled(const struct gl_context *ctx)
{
   return ctx->Const.ContextFlags & GL_CONTEXT_FLAG_NO_ERROR_BIT_KHR;
}


static inline bool
_mesa_has_integer_textures(const struct gl_context *ctx)
{
   return _mesa_has_EXT_texture_integer(ctx) || _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_half_float_textures(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_float(ctx) ||
          _mesa_has_OES_texture_half_float(ctx);
}

static inline bool
_mesa_has_float_textures(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_float(ctx) ||
          _mesa_has_OES_texture_float(ctx) || _mesa_is_gles3(ctx);
 }

static inline bool
_mesa_has_texture_rgb10_a2ui(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_rgb10_a2ui(ctx) || _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_float_depth_buffer(const struct gl_context *ctx)
{
   return _mesa_has_ARB_depth_buffer_float(ctx) || _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_packed_float(const struct gl_context *ctx)
{
   return _mesa_has_EXT_packed_float(ctx) || _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_rg_textures(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_rg(ctx) || _mesa_has_EXT_texture_rg(ctx) ||
          _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_texture_shared_exponent(const struct gl_context *ctx)
{
   return _mesa_has_EXT_texture_shared_exponent(ctx) || _mesa_is_gles3(ctx);
}

static inline bool
_mesa_has_texture_type_2_10_10_10_REV(const struct gl_context *ctx)
{
   return _mesa_is_desktop_gl(ctx) ||
          _mesa_has_EXT_texture_type_2_10_10_10_REV(ctx);
}

/**
 * Checks if the context supports geometry shaders.
 */
static inline bool
_mesa_has_geometry_shaders(const struct gl_context *ctx)
{
   return _mesa_has_OES_geometry_shader(ctx) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version >= 32);
}


/**
 * Checks if the context supports compute shaders.
 */
static inline bool
_mesa_has_compute_shaders(const struct gl_context *ctx)
{
   return _mesa_has_ARB_compute_shader(ctx) ||
      (ctx->API == API_OPENGLES2 && ctx->Version >= 31);
}

/**
 * Checks if the context supports tessellation.
 */
static inline bool
_mesa_has_tessellation(const struct gl_context *ctx)
{
   /* _mesa_has_EXT_tessellation_shader(ctx) is redundant with the OES
    * check, so don't bother calling it.
    */
   return _mesa_has_OES_tessellation_shader(ctx) ||
          _mesa_has_ARB_tessellation_shader(ctx);
}

static inline bool
_mesa_has_texture_cube_map_array(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_cube_map_array(ctx) ||
          _mesa_has_OES_texture_cube_map_array(ctx);
}

static inline bool
_mesa_has_texture_view(const struct gl_context *ctx)
{
   return _mesa_has_ARB_texture_view(ctx) ||
          _mesa_has_OES_texture_view(ctx);
}

static inline bool
_mesa_hw_select_enabled(const struct gl_context *ctx)
{
   return ctx->RenderMode == GL_SELECT &&
      ctx->Const.HardwareAcceleratedSelect;
}

static inline bool
_mesa_has_occlusion_query(const struct gl_context *ctx)
{
   return _mesa_has_ARB_occlusion_query(ctx) ||
          _mesa_has_ARB_occlusion_query2(ctx) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version >= 15);
}

static inline bool
_mesa_has_occlusion_query_boolean(const struct gl_context *ctx)
{
   return _mesa_has_ARB_occlusion_query2(ctx) ||
          _mesa_has_EXT_occlusion_query_boolean(ctx) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version >= 33);
}

static inline bool
_mesa_has_pipeline_statistics(const struct gl_context *ctx)
{
   return _mesa_has_ARB_pipeline_statistics_query(ctx) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version >= 46);
}

#ifdef __cplusplus
}
#endif


#endif /* CONTEXT_H */
