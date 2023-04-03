/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2011  VMware, Inc.  All Rights Reserved.
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


#ifndef TEXSTORAGE_H
#define TEXSTORAGE_H

/**
 * \name Internal functions
 */
/*@{*/

/**
 * Texture width, height and depth check shared with the
 * multisample variants of TexStorage functions.
 *
 * From OpenGL 4.5 Core spec, page 260 (section 8.19)
 *
 *     "An INVALID_VALUE error is generated if width, height, depth
 *     or levels are less than 1, for commands with the corresponding
 *     parameters."
 *
 * (referring to TextureStorage* commands, these also match values
 * specified for OpenGL ES 3.1.)
 */
static inline bool
_mesa_valid_tex_storage_dim(GLsizei width, GLsizei height, GLsizei depth)
{
   if (width < 1 || height < 1 || depth < 1)
      return false;
   return true;
}

/*@}*/

extern GLboolean
_mesa_is_legal_tex_storage_format(const struct gl_context *ctx,
                                  GLenum internalformat);

extern bool
_mesa_is_legal_tex_storage_target(const struct gl_context *ctx,
                                  GLuint dims, GLenum target);

extern void
_mesa_texture_storage_memory(struct gl_context *ctx, GLuint dims,
                             struct gl_texture_object *texObj,
                             struct gl_memory_object *memObj,
                             GLenum target, GLsizei levels,
                             GLenum internalformat, GLsizei width,
                             GLsizei height, GLsizei depth,
                             GLuint64 offset, bool dsa);

#endif /* TEXSTORAGE_H */
