/*
 * Copyright © 2011 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* This file declares stripped-down versions of functions that
 * normally exist outside of the glsl folder, so that they can be used
 * when running the GLSL compiler standalone (for unit testing or
 * compiling builtins).
 */

#ifndef STANDALONE_SCAFFOLDING_H
#define STANDALONE_SCAFFOLDING_H

#include <assert.h>
#include "main/menums.h"
#include "program/prog_statevars.h"

extern "C" void
_mesa_warning(struct gl_context *ctx, const char *fmtString, ... );

extern "C" void
_mesa_problem(struct gl_context *ctx, const char *fmtString, ... );

extern "C" void
_mesa_reference_shader_program_data(struct gl_context *ctx,
                                    struct gl_shader_program_data **ptr,
                                    struct gl_shader_program_data *data);

extern "C" void
_mesa_reference_shader(struct gl_context *ctx, struct gl_shader **ptr,
                       struct gl_shader *sh);

extern "C" void
_mesa_reference_program_(struct gl_context *ctx, struct gl_program **ptr,
                         struct gl_program *prog);

extern "C" struct gl_shader *
_mesa_new_shader(GLuint name, gl_shader_stage stage);

extern "C" void
_mesa_delete_shader(struct gl_context *ctx, struct gl_shader *sh);

extern "C" void
_mesa_delete_linked_shader(struct gl_context *ctx,
                           struct gl_linked_shader *sh);

extern "C" void
_mesa_clear_shader_program_data(struct gl_context *ctx,
                                struct gl_shader_program *);

extern "C" void
_mesa_shader_debug(struct gl_context *ctx, GLenum type, GLuint *id,
                   const char *msg);

extern "C" GLbitfield
_mesa_program_state_flags(const gl_state_index16 state[STATE_LENGTH]);


extern "C" char *
_mesa_program_state_string(const gl_state_index16 state[STATE_LENGTH]);

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
      assert(!"bad value in _mesa_shader_enum_to_shader_stage()");
      return MESA_SHADER_VERTEX;
   }
}

/**
 * Initialize the given gl_context structure to a reasonable set of
 * defaults representing the minimum capabilities required by the
 * OpenGL spec.
 *
 * This is used when compiling builtin functions and in testing, when
 * we don't have a connection to an actual driver.
 */
void initialize_context_to_defaults(struct gl_context *ctx, gl_api api);


#endif /* STANDALONE_SCAFFOLDING_H */
