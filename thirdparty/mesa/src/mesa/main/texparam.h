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


#ifndef TEXPARAM_H
#define TEXPARAM_H


#include "util/glheader.h"

/**
 * \name Internal functions
 */
/*@{*/

extern void
_mesa_texture_parameterf(struct gl_context *ctx,
                         struct gl_texture_object *texObj,
                         GLenum pname, GLfloat param, bool dsa);

extern void
_mesa_texture_parameterfv(struct gl_context *ctx,
                          struct gl_texture_object *texObj,
                          GLenum pname, const GLfloat *params, bool dsa);


extern void
_mesa_texture_parameteri(struct gl_context *ctx,
                         struct gl_texture_object *texObj,
                         GLenum pname, GLint param, bool dsa);

extern void
_mesa_texture_parameteriv(struct gl_context *ctx,
                          struct gl_texture_object *texObj,
                          GLenum pname, const GLint *params, bool dsa);

extern void
_mesa_texture_parameterIiv(struct gl_context *ctx,
                           struct gl_texture_object *texObj,
                           GLenum pname, const GLint *params, bool dsa);

extern void
_mesa_texture_parameterIuiv(struct gl_context *ctx,
                            struct gl_texture_object *texObj,
                            GLenum pname, const GLuint *params, bool dsa);

GLboolean
_mesa_legal_get_tex_level_parameter_target(struct gl_context *ctx, GLenum target,
                                           bool dsa);

GLboolean
_mesa_target_allows_setting_sampler_parameters(GLenum target);

/*@}*/

#endif /* TEXPARAM_H */
