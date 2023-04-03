/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2004-2007  Brian Paul   All Rights Reserved.
 * Copyright (C) 2010  VMware, Inc.  All Rights Reserved.
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


#ifndef PROGRAM_BINARY_H
#define PROGRAM_BINARY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void
_mesa_get_program_binary_length(struct gl_context *ctx,
                                struct gl_shader_program *sh_prog,
                                GLint *length);

void
_mesa_get_program_binary(struct gl_context *ctx,
                       struct gl_shader_program *sh_prog,
                       GLsizei buf_size, GLsizei *length,
                       GLenum *binary_format, GLvoid *binary);

void
_mesa_program_binary(struct gl_context *ctx, struct gl_shader_program *sh_prog,
                     GLenum binary_format, const GLvoid *binary,
                     GLsizei length);

#ifdef __cplusplus
}
#endif

#endif /* PROGRAM_BINARY_H */
