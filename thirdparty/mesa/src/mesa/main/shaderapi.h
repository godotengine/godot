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


#ifndef SHADERAPI_H
#define SHADERAPI_H


#include "util/glheader.h"
#include "compiler/shader_enums.h"
#include "util/mesa-sha1.h"

#ifdef __cplusplus
extern "C" {
#endif

struct hash_entry;
struct _glapi_table;
struct gl_context;
struct gl_linked_shader;
struct gl_pipeline_object;
struct gl_program;
struct gl_program_resource;
struct gl_shader;
struct gl_shader_program;
struct gl_resource_name;
struct gl_shared_state;
struct gl_uniform_block;

extern GLbitfield
_mesa_get_shader_flags(void);

extern const char *
_mesa_get_shader_capture_path(void);

extern void
_mesa_copy_string(GLchar *dst, GLsizei maxLength,
                  GLsizei *length, const GLchar *src);

extern void
_mesa_use_shader_program(struct gl_context *ctx,
                         struct gl_shader_program *shProg);

extern void
_mesa_active_program(struct gl_context *ctx, struct gl_shader_program *shProg,
		     const char *caller);

extern void
_mesa_compile_shader(struct gl_context *ctx, struct gl_shader *sh);

extern void
_mesa_link_program(struct gl_context *ctx, struct gl_shader_program *sh_prog);

extern unsigned
_mesa_count_active_attribs(struct gl_shader_program *shProg);

extern size_t
_mesa_longest_attribute_name_length(struct gl_shader_program *shProg);

extern void
_mesa_shader_write_subroutine_indices(struct gl_context *ctx,
                                      gl_shader_stage stage);

void
_mesa_use_program(struct gl_context *ctx, gl_shader_stage stage,
                  struct gl_shader_program *shProg, struct gl_program *prog,
                  struct gl_pipeline_object *shTarget);

extern void
_mesa_copy_linked_program_data(const struct gl_shader_program *src,
                               struct gl_linked_shader *dst_sh);

extern bool
_mesa_validate_shader_target(const struct gl_context *ctx, GLenum type);


/* GL_ARB_program_resource_query */
extern const char*
_mesa_program_resource_name(struct gl_program_resource *res);

int
_mesa_program_resource_name_length(struct gl_program_resource *res);

bool
_mesa_program_get_resource_name(struct gl_program_resource *res,
                                struct gl_resource_name *out);

extern unsigned
_mesa_program_resource_array_size(struct gl_program_resource *res);

extern GLuint
_mesa_program_resource_index(struct gl_shader_program *shProg,
                             struct gl_program_resource *res);

extern struct gl_program_resource *
_mesa_program_resource_find_name(struct gl_shader_program *shProg,
                                 GLenum programInterface, const char *name,
                                 unsigned *array_index);

extern struct gl_program_resource *
_mesa_program_resource_find_index(struct gl_shader_program *shProg,
                                  GLenum programInterface, GLuint index);

extern struct gl_program_resource *
_mesa_program_resource_find_active_variable(struct gl_shader_program *shProg,
                                            GLenum programInterface,
                                            const struct gl_uniform_block *block,
                                            unsigned index);

extern bool
_mesa_get_program_resource_name(struct gl_shader_program *shProg,
                                GLenum programInterface, GLuint index,
                                GLsizei bufSize, GLsizei *length,
                                GLchar *name, bool glthread,
                                const char *caller);

extern unsigned
_mesa_program_resource_name_length_array(struct gl_program_resource *res);

extern GLint
_mesa_program_resource_location(struct gl_shader_program *shProg,
                                GLenum programInterface, const char *name);

extern GLint
_mesa_program_resource_location_index(struct gl_shader_program *shProg,
                                      GLenum programInterface, const char *name);

extern unsigned
_mesa_program_resource_prop(struct gl_shader_program *shProg,
                            struct gl_program_resource *res, GLuint index,
                            const GLenum prop, GLint *val, bool glthread,
                            const char *caller);

extern void
_mesa_get_program_resourceiv(struct gl_shader_program *shProg,
                             GLenum programInterface, GLuint index,
                             GLsizei propCount, const GLenum *props,
                             GLsizei bufSize, GLsizei *length,
                             GLint *params);

extern void
_mesa_get_program_interfaceiv(struct gl_shader_program *shProg,
                              GLenum programInterface, GLenum pname,
                              GLint *params);

extern void
_mesa_program_resource_hash_destroy(struct gl_shader_program *shProg);

extern void
_mesa_create_program_resource_hash(struct gl_shader_program *shProg);

/* GL_ARB_shader_subroutine */
void
_mesa_program_init_subroutine_defaults(struct gl_context *ctx,
                                       struct gl_program *prog);

GLcharARB *
_mesa_read_shader_source(const gl_shader_stage stage, const char *source,
                         const uint8_t sha1[SHA1_DIGEST_LENGTH]);

void
_mesa_dump_shader_source(const gl_shader_stage stage, const char *source,
                         const uint8_t sha1[SHA1_DIGEST_LENGTH]);

void
_mesa_init_shader_includes(struct gl_shared_state *shared);

size_t
_mesa_get_shader_include_cursor(struct gl_shared_state *shared);

void
_mesa_set_shader_include_cursor(struct gl_shared_state *shared, size_t cusor);

void
_mesa_destroy_shader_includes(struct gl_shared_state *shared);

const char *
_mesa_lookup_shader_include(struct gl_context *ctx, char *path,
                            bool error_check);

#ifdef __cplusplus
}
#endif

#endif /* SHADERAPI_H */
