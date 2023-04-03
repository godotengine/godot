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


#ifndef VARRAY_H
#define VARRAY_H

#include "bufferobj.h"

struct gl_interleaved_layout {
   bool tflag, cflag, nflag;      /* enable/disable flags */
   int tcomps, ccomps, vcomps;    /* components per texcoord, color, vertex */
   GLenum ctype;                  /* color type */
   int coffset, noffset, voffset; /* color, normal, vertex offsets */
   int toffset;                   /* always zero */
   int defstride;                 /* default stride */
};

bool
_mesa_get_interleaved_layout(GLenum format,
                             struct gl_interleaved_layout *layout);

void
_mesa_set_vertex_format(struct gl_vertex_format *vertex_format,
                        GLubyte size, GLenum16 type, GLenum16 format,
                        GLboolean normalized, GLboolean integer,
                        GLboolean doubles);


/**
 * Returns a pointer to the vertex attribute data in a client array,
 * or the offset into the vertex buffer for an array that resides in
 * a vertex buffer.
 */
static inline const GLubyte *
_mesa_vertex_attrib_address(const struct gl_array_attributes *array,
                            const struct gl_vertex_buffer_binding *binding)
{
   if (binding->BufferObj)
      return (const GLubyte *) (binding->Offset + array->RelativeOffset);
   else
      return array->Ptr;
}


static inline bool
_mesa_attr_zero_aliases_vertex(const struct gl_context *ctx)
{
   return ctx->_AttribZeroAliasesVertex;
}


extern void
_mesa_update_array_format(struct gl_context *ctx,
                          struct gl_vertex_array_object *vao,
                          gl_vert_attrib attrib, GLint size, GLenum type,
                          GLenum format, GLboolean normalized,
                          GLboolean integer, GLboolean doubles,
                          GLuint relativeOffset);

extern void
_mesa_enable_vertex_array_attribs(struct gl_context *ctx,
                                 struct gl_vertex_array_object *vao,
                                 GLbitfield attrib_bits);

static inline void
_mesa_enable_vertex_array_attrib(struct gl_context *ctx,
                                 struct gl_vertex_array_object *vao,
                                 gl_vert_attrib attrib)
{
   assert(attrib < VERT_ATTRIB_MAX);
   _mesa_enable_vertex_array_attribs(ctx, vao, VERT_BIT(attrib));
}


extern void
_mesa_disable_vertex_array_attribs(struct gl_context *ctx,
                                   struct gl_vertex_array_object *vao,
                                   GLbitfield attrib_bits);

static inline void
_mesa_disable_vertex_array_attrib(struct gl_context *ctx,
                                  struct gl_vertex_array_object *vao,
                                  gl_vert_attrib attrib)
{
   assert(attrib < VERT_ATTRIB_MAX);
   _mesa_disable_vertex_array_attribs(ctx, vao, VERT_BIT(attrib));
}


extern void
_mesa_vertex_attrib_binding(struct gl_context *ctx,
                            struct gl_vertex_array_object *vao,
                            gl_vert_attrib attribIndex,
                            GLuint bindingIndex);


extern void
_mesa_bind_vertex_buffer(struct gl_context *ctx,
                         struct gl_vertex_array_object *vao,
                         GLuint index,
                         struct gl_buffer_object *vbo,
                         GLintptr offset, GLsizei stride,
                         bool offset_is_int32, bool take_vbo_ownership);

static inline unsigned
_mesa_get_prim_restart_index(bool fixed_index, unsigned restart_index,
                             unsigned index_size)
{
   /* The index_size parameter is meant to be in bytes. */
   assert(index_size == 1 || index_size == 2 || index_size == 4);

   /* From the OpenGL 4.3 core specification, page 302:
    * "If both PRIMITIVE_RESTART and PRIMITIVE_RESTART_FIXED_INDEX are
    *  enabled, the index value determined by PRIMITIVE_RESTART_FIXED_INDEX
    *  is used."
    */
   if (fixed_index) {
      /* 1 -> 0xff, 2 -> 0xffff, 4 -> 0xffffffff */
      return 0xffffffffu >> 8 * (4 - index_size);
   }

   return restart_index;
}

static inline unsigned
_mesa_primitive_restart_index(const struct gl_context *ctx,
                              unsigned index_size)
{
   return _mesa_get_prim_restart_index(ctx->Array.PrimitiveRestartFixedIndex,
                                       ctx->Array.RestartIndex, index_size);
}

void
_mesa_InternalBindVertexBuffers(struct gl_context *ctx,
                                const struct glthread_attrib_binding *buffers,
                                GLbitfield buffer_mask,
                                GLboolean restore_pointers);

extern void
_mesa_print_arrays(struct gl_context *ctx);

extern void
_mesa_init_varray(struct gl_context *ctx);

extern void
_mesa_free_varray_data(struct gl_context *ctx);

void
_mesa_update_edgeflag_state_explicit(struct gl_context *ctx,
                                     bool per_vertex_enable);

void
_mesa_update_edgeflag_state_vao(struct gl_context *ctx);


/**
 * Get the number of bytes for a vertex attrib with the given number of
 * components and type.
 *
 * Note that this function will return some number between 0 and
 * "8 * comps" if the type is invalid. It's assumed that error checking
 * was done before this, or was skipped intentionally by mesa_no_error.
 *
 * \param comps number of components.
 * \param type data type.
 */
static inline int
_mesa_bytes_per_vertex_attrib(int comps, GLenum type)
{
   /* This has comps = 3, but should return 4, so it's difficult to
    * incorporate it into the "bytes * comps" formula below.
    */
   if (type == GL_UNSIGNED_INT_10F_11F_11F_REV)
      return 4;

   /* This is a perfect hash for the specific set of GLenums that is valid
    * here. It injectively maps a small set of GLenums into smaller numbers
    * that can be used for indexing into small translation tables. It has
    * hash collisions with enums that are invalid here.
    */
   #define PERF_HASH_GL_VERTEX_TYPE(x) ((((x) * 17175) >> 14) & 0xf)

   extern const uint8_t _mesa_vertex_type_bytes[16];

   assert(PERF_HASH_GL_VERTEX_TYPE(type) < ARRAY_SIZE(_mesa_vertex_type_bytes));
   return _mesa_vertex_type_bytes[PERF_HASH_GL_VERTEX_TYPE(type)] * comps;
}

#endif
