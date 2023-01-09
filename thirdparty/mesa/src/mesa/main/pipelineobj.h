/*
 * Mesa 3-D graphics library
 *
 * Copyright © 2013 Gregory Hainaut <gregory.hainaut@gmail.com>
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

#ifndef PIPELINEOBJ_H
#define PIPELINEOBJ_H

#include "util/glheader.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _glapi_table;
struct gl_context;
struct gl_pipeline_object;

extern void
_mesa_delete_pipeline_object(struct gl_context *ctx, struct gl_pipeline_object *obj);

extern void
_mesa_init_pipeline(struct gl_context *ctx);

extern void
_mesa_free_pipeline_data(struct gl_context *ctx);

extern struct gl_pipeline_object *
_mesa_lookup_pipeline_object(struct gl_context *ctx, GLuint id);

extern void
_mesa_reference_pipeline_object_(struct gl_context *ctx,
                                 struct gl_pipeline_object **ptr,
                                 struct gl_pipeline_object *obj);

static inline void
_mesa_reference_pipeline_object(struct gl_context *ctx,
                                struct gl_pipeline_object **ptr,
                                struct gl_pipeline_object *obj)
{
   if (*ptr != obj)
      _mesa_reference_pipeline_object_(ctx, ptr, obj);
}

extern void
_mesa_bind_pipeline(struct gl_context *ctx,
                    struct gl_pipeline_object *pipe);

extern GLboolean
_mesa_validate_program_pipeline(struct gl_context * ctx,
                                struct gl_pipeline_object *pipe);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINEOBJ_H */
