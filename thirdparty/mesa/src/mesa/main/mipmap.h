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


#ifndef MIPMAP_H
#define MIPMAP_H

#include "util/glheader.h"

struct gl_context;
struct gl_texture_object;

unsigned
_mesa_compute_num_levels(struct gl_context *ctx,
                         struct gl_texture_object *texObj,
                         GLenum target);

void
_mesa_prepare_mipmap_levels(struct gl_context *ctx,
                            struct gl_texture_object *texObj,
                            unsigned baseLevel, unsigned maxLevel);

extern void
_mesa_generate_mipmap(struct gl_context *ctx, GLenum target,
                      struct gl_texture_object *texObj);

extern GLboolean
_mesa_next_mipmap_level_size(GLenum target, GLint border,
                       GLint srcWidth, GLint srcHeight, GLint srcDepth,
                       GLint *dstWidth, GLint *dstHeight, GLint *dstDepth);

#endif /* MIPMAP_H */
