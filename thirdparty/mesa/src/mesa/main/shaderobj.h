/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2004-2007  Brian Paul   All Rights Reserved.
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


#ifndef SHADEROBJ_H
#define SHADEROBJ_H


#include "util/glheader.h"
#include "compiler/shader_enums.h"
#include "program/link_program.h"
#include "util/macros.h"


#ifdef __cplusplus
extern "C" {
#endif

struct gl_shader_program_data;
struct gl_linked_shader;
struct dd_function_table;
struct gl_pipeline_object;

/**
 * Internal functions
 */

extern void
_mesa_init_shader_state(struct gl_context * ctx);

extern void
_mesa_free_shader_state(struct gl_context *ctx);


extern void
_mesa_reference_shader(struct gl_context *ctx, struct gl_shader **ptr,
                       struct gl_shader *sh);

extern struct gl_shader *
_mesa_lookup_shader(struct gl_context *ctx, GLuint name);

extern struct gl_shader *
_mesa_lookup_shader_err(struct gl_context *ctx, GLuint name, const char *caller);



extern void
_mesa_reference_shader_program_(struct gl_context *ctx,
                               struct gl_shader_program **ptr,
                               struct gl_shader_program *shProg);

void
_mesa_reference_shader_program_data(struct gl_shader_program_data **ptr,
                                    struct gl_shader_program_data *data);

static inline void
_mesa_reference_shader_program(struct gl_context *ctx,
                               struct gl_shader_program **ptr,
                               struct gl_shader_program *shProg)
{
   if (*ptr != shProg)
      _mesa_reference_shader_program_(ctx, ptr, shProg);
}

extern struct gl_shader *
_mesa_new_shader(GLuint name, gl_shader_stage type);

extern void
_mesa_delete_shader(struct gl_context *ctx, struct gl_shader *sh);

extern void
_mesa_delete_linked_shader(struct gl_context *ctx,
                           struct gl_linked_shader *sh);

extern struct gl_shader_program *
_mesa_lookup_shader_program(struct gl_context *ctx, GLuint name);

extern struct gl_shader_program *
_mesa_lookup_shader_program_err_glthread(struct gl_context *ctx, GLuint name,
                                         bool glthread, const char *caller);

extern struct gl_shader_program *
_mesa_lookup_shader_program_err(struct gl_context *ctx, GLuint name,
                                const char *caller);

extern struct gl_shader_program *
_mesa_new_shader_program(GLuint name);

extern struct gl_shader_program_data *
_mesa_create_shader_program_data(void);

extern void
_mesa_clear_shader_program_data(struct gl_context *ctx,
                                struct gl_shader_program *shProg);

extern void
_mesa_free_shader_program_data(struct gl_context *ctx,
                               struct gl_shader_program *shProg);

extern void
_mesa_delete_shader_program(struct gl_context *ctx,
                            struct gl_shader_program *shProg);

static inline gl_shader_stage
_mesa_shader_enum_to_shader_stage(GLenum v)
{
   switch (v) {
   case GL_VERTEX_SHADER:
      return MESA_SHADER_VERTEX;
   case GL_FRAGMENT_SHADER:
      return MESA_SHADER_FRAGMENT;
   case GL_GEOMETRY_SHADER:
      return MESA_SHADER_GEOMETRY;
   case GL_TESS_CONTROL_SHADER:
      return MESA_SHADER_TESS_CTRL;
   case GL_TESS_EVALUATION_SHADER:
      return MESA_SHADER_TESS_EVAL;
   case GL_COMPUTE_SHADER:
      return MESA_SHADER_COMPUTE;
   default:
      unreachable("bad value in _mesa_shader_enum_to_shader_stage()");
   }
}

/* 8 bytes + another underscore */
#define MESA_SUBROUTINE_PREFIX_LEN 9
static inline const char *
_mesa_shader_stage_to_subroutine_prefix(gl_shader_stage stage)
{
  switch (stage) {
  case MESA_SHADER_VERTEX:
    return "__subu_v";
  case MESA_SHADER_GEOMETRY:
    return "__subu_g";
  case MESA_SHADER_FRAGMENT:
    return "__subu_f";
  case MESA_SHADER_COMPUTE:
    return "__subu_c";
  case MESA_SHADER_TESS_CTRL:
    return "__subu_t";
  case MESA_SHADER_TESS_EVAL:
    return "__subu_e";
  default:
    return NULL;
  }
}

static inline gl_shader_stage
_mesa_shader_stage_from_subroutine_uniform(GLenum subuniform)
{
   switch (subuniform) {
   case GL_VERTEX_SUBROUTINE_UNIFORM:
      return MESA_SHADER_VERTEX;
   case GL_GEOMETRY_SUBROUTINE_UNIFORM:
      return MESA_SHADER_GEOMETRY;
   case GL_FRAGMENT_SUBROUTINE_UNIFORM:
      return MESA_SHADER_FRAGMENT;
   case GL_COMPUTE_SUBROUTINE_UNIFORM:
      return MESA_SHADER_COMPUTE;
   case GL_TESS_CONTROL_SUBROUTINE_UNIFORM:
      return MESA_SHADER_TESS_CTRL;
   case GL_TESS_EVALUATION_SUBROUTINE_UNIFORM:
      return MESA_SHADER_TESS_EVAL;
   }
   unreachable("not reached");
}

static inline gl_shader_stage
_mesa_shader_stage_from_subroutine(GLenum subroutine)
{
   switch (subroutine) {
   case GL_VERTEX_SUBROUTINE:
      return MESA_SHADER_VERTEX;
   case GL_GEOMETRY_SUBROUTINE:
      return MESA_SHADER_GEOMETRY;
   case GL_FRAGMENT_SUBROUTINE:
      return MESA_SHADER_FRAGMENT;
   case GL_COMPUTE_SUBROUTINE:
      return MESA_SHADER_COMPUTE;
   case GL_TESS_CONTROL_SUBROUTINE:
      return MESA_SHADER_TESS_CTRL;
   case GL_TESS_EVALUATION_SUBROUTINE:
      return MESA_SHADER_TESS_EVAL;
   default:
      unreachable("not reached");
   }
}

static inline GLenum
_mesa_shader_stage_to_subroutine(gl_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return GL_VERTEX_SUBROUTINE;
   case MESA_SHADER_GEOMETRY:
      return GL_GEOMETRY_SUBROUTINE;
   case MESA_SHADER_FRAGMENT:
      return GL_FRAGMENT_SUBROUTINE;
   case MESA_SHADER_COMPUTE:
      return GL_COMPUTE_SUBROUTINE;
   case MESA_SHADER_TESS_CTRL:
      return GL_TESS_CONTROL_SUBROUTINE;
   case MESA_SHADER_TESS_EVAL:
      return GL_TESS_EVALUATION_SUBROUTINE;
   default:
      unreachable("not reached");
   }
}

static inline GLenum
_mesa_shader_stage_to_subroutine_uniform(gl_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return GL_VERTEX_SUBROUTINE_UNIFORM;
   case MESA_SHADER_GEOMETRY:
      return GL_GEOMETRY_SUBROUTINE_UNIFORM;
   case MESA_SHADER_FRAGMENT:
      return GL_FRAGMENT_SUBROUTINE_UNIFORM;
   case MESA_SHADER_COMPUTE:
      return GL_COMPUTE_SUBROUTINE_UNIFORM;
   case MESA_SHADER_TESS_CTRL:
      return GL_TESS_CONTROL_SUBROUTINE_UNIFORM;
   case MESA_SHADER_TESS_EVAL:
      return GL_TESS_EVALUATION_SUBROUTINE_UNIFORM;
   default:
      unreachable("not reached");
   }
}

extern bool
_mesa_validate_pipeline_io(struct gl_pipeline_object *);

#ifdef __cplusplus
}
#endif

#endif /* SHADEROBJ_H */
