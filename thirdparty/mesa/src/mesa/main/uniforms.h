/*
 * Mesa 3-D graphics library
 *
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


#ifndef UNIFORMS_H
#define UNIFORMS_H

#include "util/glheader.h"
#include "compiler/glsl_types.h"
#include "compiler/glsl/ir_uniform.h"
#include "program/prog_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif


struct gl_program;
struct _glapi_table;

void
_mesa_uniform(GLint location, GLsizei count, const GLvoid *values,
              struct gl_context *, struct gl_shader_program *,
              enum glsl_base_type basicType, unsigned src_components);

void
_mesa_uniform_matrix(GLint location, GLsizei count,
                     GLboolean transpose, const void *values,
                     struct gl_context *, struct gl_shader_program *,
                     GLuint cols, GLuint rows, enum glsl_base_type basicType);

void
_mesa_uniform_handle(GLint location, GLsizei count, const GLvoid *values,
                     struct gl_context *, struct gl_shader_program *);

void
_mesa_get_uniform(struct gl_context *ctx, GLuint program, GLint location,
		  GLsizei bufSize, enum glsl_base_type returnType,
		  GLvoid *paramsOut);

extern void
_mesa_uniform_attach_driver_storage(struct gl_uniform_storage *,
				    unsigned element_stride,
				    unsigned vector_stride,
				    enum gl_uniform_driver_format format,
				    void *data);

extern void
_mesa_uniform_detach_all_driver_storage(struct gl_uniform_storage *uni);

extern void
_mesa_propagate_uniforms_to_driver_storage(struct gl_uniform_storage *uni,
					   unsigned array_index,
					   unsigned count);

void
_mesa_ensure_and_associate_uniform_storage(struct gl_context *ctx,
                             struct gl_shader_program *shader_program,
                             struct gl_program *prog, unsigned required_space);

extern void
_mesa_update_shader_textures_used(struct gl_shader_program *shProg,
				  struct gl_program *prog);

extern bool
_mesa_sampler_uniforms_are_valid(const struct gl_shader_program *shProg,
				 char *errMsg, size_t errMsgLength);
extern bool
_mesa_sampler_uniforms_pipeline_are_valid(struct gl_pipeline_object *);

extern void
_mesa_flush_vertices_for_uniforms(struct gl_context *ctx,
                                  const struct gl_uniform_storage *uni);

extern GLint
_mesa_GetUniformLocation_impl(GLuint programObj, const GLcharARB *name,
                              bool glthread);

extern void
_mesa_GetActiveUniform_impl(GLuint program, GLuint index,
                            GLsizei maxLength, GLsizei *length, GLint *size,
                            GLenum *type, GLcharARB *nameOut, bool glthread);

struct gl_builtin_uniform_element {
   const char *field;
   gl_state_index16 tokens[STATE_LENGTH];
   int swizzle;
};

struct gl_builtin_uniform_desc {
   const char *name;
   const struct gl_builtin_uniform_element *elements;
   unsigned int num_elements;
};

#ifdef __cplusplus
}
#endif


#endif /* UNIFORMS_H */
