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


#ifndef STATE_H
#define STATE_H

#include "mtypes.h"

extern void
_mesa_update_allow_draw_out_of_order(struct gl_context *ctx);

extern uint64_t
_mesa_get_active_states(struct gl_context *ctx);

extern void
_mesa_update_state(struct gl_context *ctx);

/* As above but can only be called between _mesa_lock_context_textures() and 
 * _mesa_unlock_context_textures().
 */
extern void
_mesa_update_state_locked(struct gl_context *ctx);

/*
 * Update state for glClear calls
*/
extern void
_mesa_update_clear_state(struct gl_context *ctx);


extern void
_mesa_set_vp_override(struct gl_context *ctx, GLboolean flag);


/**
 * Update ctx->VertexProgram._VPMode.
 */
extern void
_mesa_update_vertex_processing_mode(struct gl_context *ctx);

extern void
_mesa_reset_vertex_processing_mode(struct gl_context *ctx);


static inline bool
_mesa_ati_fragment_shader_enabled(const struct gl_context *ctx)
{
   return ctx->ATIFragmentShader.Enabled &&
          ctx->ATIFragmentShader.Current->Instructions[0];
}

/**
 * Is the secondary color needed?
 */
static inline GLboolean
_mesa_need_secondary_color(const struct gl_context *ctx)
{
   if (ctx->Light.Enabled &&
       ctx->Light.Model.ColorControl == GL_SEPARATE_SPECULAR_COLOR)
       return GL_TRUE;

   if (ctx->Fog.ColorSumEnabled)
      return GL_TRUE;

   if (ctx->VertexProgram._Current &&
       (ctx->VertexProgram._Current != ctx->VertexProgram._TnlProgram) &&
       (ctx->VertexProgram._Current->info.inputs_read & VERT_BIT_COLOR1))
      return GL_TRUE;

   if (ctx->FragmentProgram._Current &&
       (ctx->FragmentProgram._Current != ctx->FragmentProgram._TexEnvProgram) &&
       (ctx->FragmentProgram._Current->info.inputs_read & VARYING_BIT_COL1))
      return GL_TRUE;

   if (_mesa_ati_fragment_shader_enabled(ctx))
      return GL_TRUE;

   return GL_FALSE;
}

static inline bool
_mesa_arb_vertex_program_enabled(const struct gl_context *ctx)
{
   return ctx->VertexProgram.Enabled &&
          ctx->VertexProgram.Current->arb.Instructions;
}

/** Compute two sided lighting state for fixed function or programs. */
static inline bool
_mesa_vertex_program_two_side_enabled(const struct gl_context *ctx)
{
   if (ctx->_Shader->CurrentProgram[MESA_SHADER_VERTEX] ||
       _mesa_arb_vertex_program_enabled(ctx))
      return ctx->VertexProgram.TwoSideEnabled;

   return ctx->Light.Enabled && ctx->Light.Model.TwoSide;
}

/** Return 0=GL_CCW or 1=GL_CW */
static inline bool
_mesa_polygon_get_front_bit(const struct gl_context *ctx)
{
   if (ctx->Transform.ClipOrigin == GL_LOWER_LEFT)
      return ctx->Polygon.FrontFace == GL_CW;

   return ctx->Polygon.FrontFace == GL_CCW;
}

static inline bool
_mesa_arb_fragment_program_enabled(const struct gl_context *ctx)
{
   return ctx->FragmentProgram.Enabled &&
          ctx->FragmentProgram.Current->arb.Instructions;
}

#endif
