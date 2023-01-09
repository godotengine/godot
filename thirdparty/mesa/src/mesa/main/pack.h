/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 1999-2010  VMware, Inc.  All Rights Reserved.
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


#ifndef PACK_H
#define PACK_H


#include "util/glheader.h"

struct gl_context;
struct gl_pixelstore_attrib;

extern void
_mesa_unpack_polygon_stipple(const GLubyte *pattern, GLuint dest[32],
                             const struct gl_pixelstore_attrib *unpacking);


extern void
_mesa_pack_polygon_stipple(const GLuint pattern[32], GLubyte *dest,
                           const struct gl_pixelstore_attrib *packing);


extern void
_mesa_pack_bitmap(GLint width, GLint height, const GLubyte *source,
                  GLubyte *dest, const struct gl_pixelstore_attrib *packing);


extern void
_mesa_unpack_stencil_span(struct gl_context *ctx, GLuint n,
                          GLenum dstType, GLvoid *dest,
                          GLenum srcType, const GLvoid *source,
                          const struct gl_pixelstore_attrib *srcPacking,
                          GLbitfield transferOps);

extern void
_mesa_pack_stencil_span(struct gl_context *ctx, GLuint n,
                        GLenum dstType, GLvoid *dest, const GLubyte *source,
                        const struct gl_pixelstore_attrib *dstPacking);


extern void
_mesa_unpack_depth_span(struct gl_context *ctx, GLuint n,
                        GLenum dstType, GLvoid *dest, GLuint depthMax,
                        GLenum srcType, const GLvoid *source,
                        const struct gl_pixelstore_attrib *srcPacking);

extern void
_mesa_pack_depth_span(struct gl_context *ctx, GLuint n, GLvoid *dest,
                      GLenum dstType, const GLfloat *depthSpan,
                      const struct gl_pixelstore_attrib *dstPacking);


extern void
_mesa_pack_depth_stencil_span(struct gl_context *ctx,GLuint n,
                              GLenum dstType, GLuint *dest,
                              const GLfloat *depthVals,
                              const GLubyte *stencilVals,
                              const struct gl_pixelstore_attrib *dstPacking);


extern void *
_mesa_unpack_image(GLuint dimensions,
                   GLsizei width, GLsizei height, GLsizei depth,
                   GLenum format, GLenum type, const GLvoid *pixels,
                   const struct gl_pixelstore_attrib *unpack);

extern void
_mesa_pack_luminance_from_rgba_float(GLuint n, GLfloat rgba[][4],
                                     GLvoid *dstAddr, GLenum dst_format,
                                     GLbitfield transferOps);

extern void
_mesa_pack_luminance_from_rgba_integer(GLuint n, GLuint rgba[][4], bool rgba_is_signed,
                                       GLvoid *dstAddr, GLenum dst_format,
                                       GLenum dst_type);

extern GLfloat *
_mesa_unpack_color_index_to_rgba_float(struct gl_context *ctx, GLuint dims,
                                       const void *src, GLenum srcFormat, GLenum srcType,
                                       int srcWidth, int srcHeight, int srcDepth,
                                       const struct gl_pixelstore_attrib *srcPacking,
                                       GLbitfield transferOps);

extern GLubyte *
_mesa_unpack_color_index_to_rgba_ubyte(struct gl_context *ctx, GLuint dims,
                                       const void *src, GLenum srcFormat, GLenum srcType,
                                       int srcWidth, int srcHeight, int srcDepth,
                                       const struct gl_pixelstore_attrib *srcPacking,
                                       GLbitfield transferOps);

#endif
