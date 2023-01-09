/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
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


#ifndef LIGHT_H
#define LIGHT_H


#include <stdbool.h>
#include "util/glheader.h"

struct gl_context;
struct gl_light;
struct gl_material;

extern GLuint _mesa_material_bitmask( struct gl_context *ctx,
                                      GLenum face, GLenum pname,
                                      GLuint legal,
                                      const char * );

extern GLbitfield _mesa_update_lighting( struct gl_context *ctx );
extern void _mesa_update_light_materials(struct gl_context *ctx);
extern bool _mesa_update_tnl_spaces( struct gl_context *ctx, GLuint new_state );

extern void _mesa_update_material( struct gl_context *ctx,
                                   GLuint bitmask );

extern void _mesa_update_color_material( struct gl_context *ctx,
                                         const GLfloat rgba[4] );

extern void _mesa_init_lighting( struct gl_context *ctx );

#endif
