/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009-2010  VMware, Inc.  All Rights Reserved.
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


#ifndef PIXELTRANSFER_H
#define PIXELTRANSFER_H


#include "util/glheader.h"

struct gl_context;


extern void
_mesa_scale_and_bias_rgba(GLuint n, GLfloat rgba[][4],
                          GLfloat rScale, GLfloat gScale,
                          GLfloat bScale, GLfloat aScale,
                          GLfloat rBias, GLfloat gBias,
                          GLfloat bBias, GLfloat aBias);

extern void
_mesa_map_rgba(const struct gl_context *ctx, GLuint n, GLfloat rgba[][4]);

extern void
_mesa_map_ci_to_rgba(const struct gl_context *ctx,
                     GLuint n, const GLuint index[], GLfloat rgba[][4]);


extern void
_mesa_scale_and_bias_depth(const struct gl_context *ctx, GLuint n,
                           GLfloat depthValues[]);

extern void
_mesa_scale_and_bias_depth_uint(const struct gl_context *ctx, GLuint n,
                                GLuint depthValues[]);

extern void
_mesa_apply_rgba_transfer_ops(struct gl_context *ctx, GLbitfield transferOps,
                              GLuint n, GLfloat rgba[][4]);

extern void
_mesa_shift_and_offset_ci(const struct gl_context *ctx,
                          GLuint n, GLuint indexes[]);

extern void
_mesa_apply_stencil_transfer_ops(const struct gl_context *ctx, GLuint n,
                                 GLubyte stencil[]);


#endif
