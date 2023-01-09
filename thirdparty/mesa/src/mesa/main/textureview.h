/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2012-2013 LunarG, Inc.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Courtney Goeltzenleuchter <courtney@lunarg.com>
 */


#ifndef TEXTUREVIEW_H
#define TEXTUREVIEW_H

bool
_mesa_texture_view_compatible_format(const struct gl_context *ctx,
                                     GLenum origInternalFormat,
                                     GLenum newInternalFormat);

/**
 * Lookup format view class based on internalformat
 * \return VIEW_CLASS if internalformat found in table, false otherwise.
 */
GLenum
_mesa_texture_view_lookup_view_class(const struct gl_context *ctx,
                                     GLenum internalformat);

extern void
_mesa_set_texture_view_state(struct gl_context *ctx,
                             struct gl_texture_object *texObj,
                             GLenum target, GLuint levels);

#endif /* TEXTUREVIEW_H */
