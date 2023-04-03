
/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2001  Brian Paul   All Rights Reserved.
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


#ifndef API_VALIDATE_H
#define API_VALIDATE_H

#include "mtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gl_buffer_object;
struct gl_context;
struct gl_transform_feedback_object;


extern GLenum
_mesa_valid_prim_mode(struct gl_context *ctx, GLenum mode);

extern void
_mesa_update_valid_to_render_state(struct gl_context *ctx);

/**
 * Is 'mode' a valid value for glBegin(), glDrawArrays(), glDrawElements(),
 * etc?  The set of legal values depends on whether geometry shaders/programs
 * are supported.
 * Note: This may be called during display list compilation.
 */
static inline bool
_mesa_is_valid_prim_mode(const struct gl_context *ctx, GLenum mode)
{
   /* All primitive types are less than 32, which allows us to use a mask. */
   return mode < 32 && (1u << mode) & ctx->SupportedPrimMask;
}

#ifdef __cplusplus
}
#endif

#endif
