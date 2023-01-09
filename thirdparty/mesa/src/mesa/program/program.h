/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2007  Brian Paul   All Rights Reserved.
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
 * \file program.c
 * Vertex and fragment program support functions.
 * \author Brian Paul
 */


/**
 * \mainpage Mesa vertex and fragment program module
 *
 * This module or directory contains most of the code for vertex and
 * fragment programs and shaders, including state management, parsers,
 * and (some) software routines for executing programs
 */

#ifndef PROGRAM_H
#define PROGRAM_H

#include "prog_parameter.h"


#ifdef __cplusplus
extern "C" {
#endif

extern struct gl_program _mesa_DummyProgram;


extern void
_mesa_init_program(struct gl_context *ctx);

extern void
_mesa_free_program_data(struct gl_context *ctx);

extern void
_mesa_update_default_objects_program(struct gl_context *ctx);

extern void
_mesa_set_program_error(struct gl_context *ctx, GLint pos, const char *string);

extern struct gl_program *
_mesa_init_gl_program(struct gl_program *prog, gl_shader_stage stage,
                      GLuint id, bool is_arb_asm);

extern struct gl_program *
_mesa_new_program(struct gl_context *ctx, gl_shader_stage stage, GLuint id,
                  bool is_arb_asm);

extern void
_mesa_delete_program(struct gl_context *ctx, struct gl_program *prog);

extern struct gl_program *
_mesa_lookup_program(struct gl_context *ctx, GLuint id);

extern void
_mesa_reference_program_(struct gl_context *ctx,
                         struct gl_program **ptr,
                         struct gl_program *prog);

static inline void
_mesa_reference_program(struct gl_context *ctx,
                        struct gl_program **ptr,
                        struct gl_program *prog)
{
   if (*ptr != prog)
      _mesa_reference_program_(ctx, ptr, prog);
}

extern GLint
_mesa_get_min_invocations_per_fragment(struct gl_context *ctx,
                                       const struct gl_program *prog);

static inline GLuint
_mesa_program_enum_to_shader_stage(GLenum v)
{
   switch (v) {
   case GL_VERTEX_PROGRAM_ARB:
      return MESA_SHADER_VERTEX;
   case GL_FRAGMENT_PROGRAM_ARB:
      return MESA_SHADER_FRAGMENT;
   case GL_FRAGMENT_SHADER_ATI:
      return MESA_SHADER_FRAGMENT;
   case GL_GEOMETRY_PROGRAM_NV:
      return MESA_SHADER_GEOMETRY;
   case GL_TESS_CONTROL_PROGRAM_NV:
      return MESA_SHADER_TESS_CTRL;
   case GL_TESS_EVALUATION_PROGRAM_NV:
      return MESA_SHADER_TESS_EVAL;
   case GL_COMPUTE_PROGRAM_NV:
      return MESA_SHADER_COMPUTE;
   default:
      assert(0);
      return ~0;
   }
}


static inline GLenum
_mesa_shader_stage_to_program(unsigned stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return GL_VERTEX_PROGRAM_ARB;
   case MESA_SHADER_FRAGMENT:
      return GL_FRAGMENT_PROGRAM_ARB;
   case MESA_SHADER_GEOMETRY:
      return GL_GEOMETRY_PROGRAM_NV;
   case MESA_SHADER_TESS_CTRL:
      return GL_TESS_CONTROL_PROGRAM_NV;
   case MESA_SHADER_TESS_EVAL:
      return GL_TESS_EVALUATION_PROGRAM_NV;
   case MESA_SHADER_COMPUTE:
      return GL_COMPUTE_PROGRAM_NV;
   }

   assert(!"Unexpected shader stage in _mesa_shader_stage_to_program");
   return GL_VERTEX_PROGRAM_ARB;
}


GLbitfield
gl_external_samplers(const struct gl_program *prog);

void
_mesa_add_separate_state_parameters(struct gl_program *prog,
                                    struct gl_program_parameter_list *state_params);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROGRAM_H */
