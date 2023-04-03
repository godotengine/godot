/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul   All Rights Reserved.
 * Copyright (c) 2008 VMware, Inc.
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
 * \file texstore.h
 * Texture image storage routines.
 *
 * \author Brian Paul
 */


#ifndef TEXSTORE_H
#define TEXSTORE_H


#include "util/glheader.h"
#include "formats.h"
#include "util/macros.h"

struct gl_context;
struct gl_pixelstore_attrib;
struct gl_texture_image;

/**
 * This macro defines the (many) parameters to the texstore functions.
 * \param dims  either 1 or 2 or 3
 * \param baseInternalFormat  user-specified base internal format
 * \param dstFormat  destination Mesa texture format
 * \param dstX/Y/Zoffset  destination x/y/z offset (ala TexSubImage), in texels
 * \param dstRowStride  destination image row stride, in bytes
 * \param dstSlices  array of addresses of image slices (for 3D, array texture)
 * \param srcWidth/Height/Depth  source image size, in pixels
 * \param srcFormat  incoming image format
 * \param srcType  incoming image data type
 * \param srcAddr  source image address
 * \param srcPacking  source image packing parameters
 */
#define TEXSTORE_PARAMS \
	struct gl_context *ctx, GLuint dims, \
        UNUSED GLenum baseInternalFormat, \
        UNUSED mesa_format dstFormat, \
        GLint dstRowStride, \
        GLubyte **dstSlices, \
	GLint srcWidth, GLint srcHeight, GLint srcDepth, \
	GLenum srcFormat, GLenum srcType, \
	const GLvoid *srcAddr, \
	const struct gl_pixelstore_attrib *srcPacking

/* This macro must be kept in sync with TEXSTORE_PARAMS.  It is used in the
 * few places where none of the parameters are used (i.e., the ETC texstore
 * functions).
 */
#define UNUSED_TEXSTORE_PARAMS                                          \
        UNUSED struct gl_context *ctx, UNUSED GLuint dims,              \
        UNUSED GLenum baseInternalFormat,                               \
        UNUSED mesa_format dstFormat,                                   \
        UNUSED GLint dstRowStride,                                      \
        UNUSED GLubyte **dstSlices,                                     \
        UNUSED GLint srcWidth, UNUSED GLint srcHeight, UNUSED GLint srcDepth, \
        UNUSED GLenum srcFormat, UNUSED GLenum srcType,                 \
        UNUSED const GLvoid *srcAddr,                                   \
        UNUSED const struct gl_pixelstore_attrib *srcPacking

extern GLboolean
_mesa_texstore(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_needs_transfer_ops(struct gl_context *ctx,
                                  GLenum baseInternalFormat,
                                  mesa_format dstFormat);

extern void
_mesa_memcpy_texture(struct gl_context *ctx,
                     GLuint dimensions,
                     mesa_format dstFormat,
                     GLint dstRowStride,
                     GLubyte **dstSlices,
                     GLint srcWidth, GLint srcHeight, GLint srcDepth,
                     GLenum srcFormat, GLenum srcType,
                     const GLvoid *srcAddr,
                     const struct gl_pixelstore_attrib *srcPacking);

extern GLboolean
_mesa_texstore_can_use_memcpy(struct gl_context *ctx,
                              GLenum baseInternalFormat, mesa_format dstFormat,
                              GLenum srcFormat, GLenum srcType,
                              const struct gl_pixelstore_attrib *srcPacking);


extern void
_mesa_store_teximage(struct gl_context *ctx,
                     GLuint dims,
                     struct gl_texture_image *texImage,
                     GLenum format, GLenum type, const GLvoid *pixels,
                     const struct gl_pixelstore_attrib *packing);


extern void
_mesa_store_texsubimage(struct gl_context *ctx, GLuint dims,
                        struct gl_texture_image *texImage,
                        GLint xoffset, GLint yoffset, GLint zoffset,
                        GLint width, GLint height, GLint depth,
                        GLenum format, GLenum type, const GLvoid *pixels,
                        const struct gl_pixelstore_attrib *packing);


extern void
_mesa_store_cleartexsubimage(struct gl_context *ctx,
                             struct gl_texture_image *texImage,
                             GLint xoffset, GLint yoffset, GLint zoffset,
                             GLsizei width, GLsizei height, GLsizei depth,
                             const GLvoid *clearValue);

extern void
_mesa_store_compressed_teximage(struct gl_context *ctx, GLuint dims,
                                struct gl_texture_image *texImage,
                                GLsizei imageSize, const GLvoid *data);


extern void
_mesa_store_compressed_texsubimage(struct gl_context *ctx, GLuint dims,
                                   struct gl_texture_image *texImage,
                                   GLint xoffset, GLint yoffset, GLint zoffset,
                                   GLsizei width, GLsizei height, GLsizei depth,
                                   GLenum format,
                                   GLsizei imageSize, const GLvoid *data);


struct compressed_pixelstore {
   int SkipBytes;
   int CopyBytesPerRow;
   int CopyRowsPerSlice;
   int TotalBytesPerRow;
   int TotalRowsPerSlice;
   int CopySlices;
};


extern void
_mesa_compute_compressed_pixelstore(GLuint dims, mesa_format texFormat,
                                    GLsizei width, GLsizei height,
                                    GLsizei depth,
                                    const struct gl_pixelstore_attrib *packing,
                                    struct compressed_pixelstore *store);


#endif
