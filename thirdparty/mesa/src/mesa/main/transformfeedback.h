/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2010  VMware, Inc.  All Rights Reserved.
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


#ifndef TRANSFORM_FEEDBACK_H
#define TRANSFORM_FEEDBACK_H

#include <stdbool.h>
#include "bufferobj.h"
#include "util/compiler.h"
#include "util/glheader.h"
#include "mtypes.h"

struct _glapi_table;
struct dd_function_table;
struct gl_context;

extern void
_mesa_init_transform_feedback(struct gl_context *ctx);

extern void
_mesa_free_transform_feedback(struct gl_context *ctx);

extern unsigned
_mesa_compute_max_transform_feedback_vertices( struct gl_context *ctx,
      const struct gl_transform_feedback_object *obj,
      const struct gl_transform_feedback_info *info);


/*** GL_EXT_transform_feedback ***/

extern bool
_mesa_validate_buffer_range_xfb(struct gl_context *ctx,
                                struct gl_transform_feedback_object *obj,
                                GLuint index, struct gl_buffer_object *bufObj,
                                GLintptr offset, GLsizeiptr size, bool dsa);

extern void
_mesa_bind_buffer_base_transform_feedback(struct gl_context *ctx,
					  struct gl_transform_feedback_object *obj,
					  GLuint index,
					  struct gl_buffer_object *bufObj,
					  bool dsa);

/*** GL_ARB_transform_feedback2 ***/
extern void
_mesa_init_transform_feedback_object(struct gl_transform_feedback_object *obj,
                                     GLuint name);

extern void
_mesa_delete_transform_feedback_object(struct gl_context *ctx,
                                       struct gl_transform_feedback_object
                                              *obj);

struct gl_transform_feedback_object *
_mesa_lookup_transform_feedback_object(struct gl_context *ctx, GLuint name);

static inline bool
_mesa_is_xfb_active_and_unpaused(const struct gl_context *ctx)
{
   return ctx->TransformFeedback.CurrentObject->Active &&
      !ctx->TransformFeedback.CurrentObject->Paused;
}

extern bool
_mesa_transform_feedback_is_using_program(struct gl_context *ctx,
                                          struct gl_shader_program *shProg);

static inline void
_mesa_set_transform_feedback_binding(struct gl_context *ctx,
                                     struct gl_transform_feedback_object *tfObj, GLuint index,
                                     struct gl_buffer_object *bufObj,
                                     GLintptr offset, GLsizeiptr size)
{
   _mesa_reference_buffer_object(ctx, &tfObj->Buffers[index], bufObj);

   tfObj->BufferNames[index]   = bufObj ? bufObj->Name : 0;
   tfObj->Offset[index]        = offset;
   tfObj->RequestedSize[index] = size;

   if (bufObj)
      bufObj->UsageHistory |= USAGE_TRANSFORM_FEEDBACK_BUFFER;
}

static inline void
_mesa_bind_buffer_range_xfb(struct gl_context *ctx,
                            struct gl_transform_feedback_object *obj,
                            GLuint index, struct gl_buffer_object *bufObj,
                            GLintptr offset, GLsizeiptr size)
{
   /* Note: no need to FLUSH_VERTICES because
    * transform feedback buffers can't be changed while transform feedback is
    * active.
    */

   /* The general binding point */
   _mesa_reference_buffer_object(ctx,
                                 &ctx->TransformFeedback.CurrentBuffer,
                                 bufObj);

   /* The per-attribute binding point */
   _mesa_set_transform_feedback_binding(ctx, obj, index, bufObj, offset, size);
}

#endif /* TRANSFORM_FEEDBACK_H */
