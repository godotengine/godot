/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
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


#ifndef READPIXELS_H
#define READPIXELS_H


#include "util/glheader.h"

struct gl_context;
struct gl_pixelstore_attrib;


extern GLboolean
_mesa_readpixels_needs_slow_path(const struct gl_context *ctx, GLenum format,
                                 GLenum type, GLboolean uses_blit);

extern GLboolean
_mesa_need_rgb_to_luminance_conversion(GLenum srcBaseFormat,
                                       GLenum dstBaseFormat);

extern GLboolean
_mesa_need_luminance_to_rgb_conversion(GLenum srcBaseFormat,
                                       GLenum dstBaseFormat);

extern GLbitfield
_mesa_get_readpixels_transfer_ops(const struct gl_context *ctx,
                                  mesa_format texFormat,
                                  GLenum format, GLenum type,
                                  GLboolean uses_blit);

extern void
_mesa_readpixels(struct gl_context *ctx,
                 GLint x, GLint y, GLsizei width, GLsizei height,
                 GLenum format, GLenum type,
                 const struct gl_pixelstore_attrib *packing,
                 GLvoid *pixels);

#endif
