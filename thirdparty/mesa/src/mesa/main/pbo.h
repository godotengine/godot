/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009-2011  VMware, Inc.  All Rights Reserved.
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


#ifndef PBO_H
#define PBO_H


#include "util/glheader.h"

struct gl_context;
struct gl_pixelstore_attrib;

extern GLboolean
_mesa_validate_pbo_access(GLuint dimensions,
                          const struct gl_pixelstore_attrib *pack,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLenum format, GLenum type, GLsizei clientMemSize,
                          const GLvoid *ptr);

extern const GLvoid *
_mesa_map_pbo_source(struct gl_context *ctx,
                     const struct gl_pixelstore_attrib *unpack,
                     const GLvoid *src);

extern const GLvoid *
_mesa_map_validate_pbo_source(struct gl_context *ctx,
                              GLuint dimensions,
                              const struct gl_pixelstore_attrib *unpack,
                              GLsizei width, GLsizei height, GLsizei depth,
                              GLenum format, GLenum type, GLsizei clientMemSize,
                              const GLvoid *ptr, const char *where);

extern void
_mesa_unmap_pbo_source(struct gl_context *ctx,
                       const struct gl_pixelstore_attrib *unpack);

extern void *
_mesa_map_pbo_dest(struct gl_context *ctx,
                   const struct gl_pixelstore_attrib *pack,
                   GLvoid *dest);

extern GLvoid *
_mesa_map_validate_pbo_dest(struct gl_context *ctx,
                            GLuint dimensions,
                            const struct gl_pixelstore_attrib *unpack,
                            GLsizei width, GLsizei height, GLsizei depth,
                            GLenum format, GLenum type, GLsizei clientMemSize,
                            GLvoid *ptr, const char *where);

extern void
_mesa_unmap_pbo_dest(struct gl_context *ctx,
                     const struct gl_pixelstore_attrib *pack);


extern const GLvoid *
_mesa_validate_pbo_teximage(struct gl_context *ctx, GLuint dimensions,
			    GLsizei width, GLsizei height, GLsizei depth,
			    GLenum format, GLenum type, const GLvoid *pixels,
			    const struct gl_pixelstore_attrib *unpack,
			    const char *funcName);

extern const GLvoid *
_mesa_validate_pbo_compressed_teximage(struct gl_context *ctx,
                                    GLuint dimensions, GLsizei imageSize,
                                    const GLvoid *pixels,
                                    const struct gl_pixelstore_attrib *packing,
                                    const char *funcName);

extern void
_mesa_unmap_teximage_pbo(struct gl_context *ctx,
                         const struct gl_pixelstore_attrib *unpack);


extern bool
_mesa_validate_pbo_source(struct gl_context *ctx, GLuint dimensions,
                          const struct gl_pixelstore_attrib *unpack,
                          GLsizei width, GLsizei height, GLsizei depth,
                          GLenum format, GLenum type,
                          GLsizei clientMemSize,
                          const GLvoid *ptr, const char *where);

extern bool
_mesa_validate_pbo_source_compressed(struct gl_context *ctx, GLuint dimensions,
                                     const struct gl_pixelstore_attrib *unpack,
                                     GLsizei imageSize, const GLvoid *ptr,
                                     const char *where);

#endif
