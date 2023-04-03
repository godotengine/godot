/*
 * mesa 3-D graphics library
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

/**
 * \brief Array type draw functions, the main workhorse of any OpenGL API
 * \author Keith Whitwell
 */


#ifndef DRAW_H
#define DRAW_H

#include <stdbool.h>
#include "util/glheader.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gl_context;
struct gl_vertex_array_object;
struct _mesa_prim
{
   GLubyte mode;    /**< GL_POINTS, GL_LINES, GL_QUAD_STRIP, etc */

   /**
    * tnl: If true, line stipple emulation will reset the pattern walker.
    * vbo: If false and the primitive is a line loop, the first vertex is
    *      the beginning of the line loop and it won't be drawn.
    *      Instead, it will be moved to the end.
    */
   bool begin;

   /**
    * tnl: If true and the primitive is a line loop, it will be closed.
    * vbo: Same as tnl.
    */
   bool end;

   GLuint start;
   GLuint count;
   GLint basevertex;
   GLuint draw_id;
};

/* Would like to call this a "vbo_index_buffer", but this would be
 * confusing as the indices are not neccessarily yet in a non-null
 * buffer object.
 */
struct _mesa_index_buffer
{
   GLuint count;
   uint8_t index_size_shift; /* logbase2(index_size) */
   struct gl_buffer_object *obj;
   const void *ptr;
};


void
_mesa_set_varying_vp_inputs(struct gl_context *ctx, GLbitfield varying_inputs);

void
_mesa_set_draw_vao(struct gl_context *ctx, struct gl_vertex_array_object *vao);

void
_mesa_save_and_set_draw_vao(struct gl_context *ctx,
                            struct gl_vertex_array_object *vao,
                            GLbitfield vp_input_filter,
                            struct gl_vertex_array_object **old_vao,
                            GLbitfield *old_vp_input_filter);

void
_mesa_restore_draw_vao(struct gl_context *ctx,
                       struct gl_vertex_array_object *saved,
                       GLbitfield saved_vp_input_filter);

void
_mesa_bitmap(struct gl_context *ctx, GLsizei width, GLsizei height,
             GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove,
             const GLubyte *bitmap, struct pipe_resource *tex);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
