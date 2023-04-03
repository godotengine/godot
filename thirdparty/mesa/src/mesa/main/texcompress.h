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

#ifndef TEXCOMPRESS_H
#define TEXCOMPRESS_H

#include "formats.h"
#include "util/glheader.h"

struct gl_context;

extern GLenum
_mesa_gl_compressed_format_base_format(GLenum format);

extern GLuint
_mesa_get_compressed_formats(struct gl_context *ctx, GLint *formats);

extern mesa_format
_mesa_glenum_to_compressed_format(GLenum format);

extern GLenum
_mesa_compressed_format_to_glenum(struct gl_context *ctx, mesa_format mesaFormat);

extern GLubyte *
_mesa_compressed_image_address(GLint col, GLint row, GLint img,
                               mesa_format mesaFormat,
                               GLsizei width, const GLubyte *image);


/** A function to fetch one texel from a compressed texture */
typedef void (*compressed_fetch_func)(const GLubyte *map,
                                      GLint rowStride,
                                      GLint i, GLint j,
                                      GLfloat *texel);

extern compressed_fetch_func
_mesa_get_compressed_fetch_func(mesa_format format);


extern void
_mesa_decompress_image(mesa_format format, GLuint width, GLuint height,
                       const GLubyte *src, GLint srcRowStride,
                       GLfloat *dest);

#endif /* TEXCOMPRESS_H */
