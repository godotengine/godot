/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2011  VMware, Inc.  All Rights Reserved.
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



#ifndef SAMPLEROBJ_H
#define SAMPLEROBJ_H

#include "mtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dd_function_table;

static inline struct gl_sampler_object *
_mesa_get_samplerobj(struct gl_context *ctx, GLuint unit)
{
   if (ctx->Texture.Unit[unit].Sampler)
      return ctx->Texture.Unit[unit].Sampler;
   else if (ctx->Texture.Unit[unit]._Current)
      return &ctx->Texture.Unit[unit]._Current->Sampler;
   else
      return NULL;
}


/** Does the given filter state do mipmap filtering? */
static inline GLboolean
_mesa_is_mipmap_filter(const struct gl_sampler_object *samp)
{
   return samp->Attrib.MinFilter != GL_NEAREST && samp->Attrib.MinFilter != GL_LINEAR;
}


extern void
_mesa_reference_sampler_object_(struct gl_context *ctx,
                                struct gl_sampler_object **ptr,
                                struct gl_sampler_object *samp);

static inline void
_mesa_reference_sampler_object(struct gl_context *ctx,
                               struct gl_sampler_object **ptr,
                               struct gl_sampler_object *samp)
{
   if (*ptr != samp)
      _mesa_reference_sampler_object_(ctx, ptr, samp);
}

extern struct gl_sampler_object *
_mesa_lookup_samplerobj(struct gl_context *ctx, GLuint name);

extern void
_mesa_bind_sampler(struct gl_context *ctx, GLuint unit,
                   struct gl_sampler_object *sampObj);

extern const enum pipe_tex_wrap wrap_to_gallium_table[32];

/**
 * Convert GLenum texcoord wrap tokens to pipe tokens.
 */
static inline enum pipe_tex_wrap
wrap_to_gallium(GLenum wrap)
{
   return wrap_to_gallium_table[wrap & 0x1f];
}


static inline enum pipe_tex_mipfilter
mipfilter_to_gallium(GLenum filter)
{
   /* Take advantage of how the enums are defined. */
   if (filter <= GL_LINEAR)
      return PIPE_TEX_MIPFILTER_NONE;
   if (filter <= GL_LINEAR_MIPMAP_NEAREST)
      return PIPE_TEX_MIPFILTER_NEAREST;

   return PIPE_TEX_MIPFILTER_LINEAR;
}


static inline enum pipe_tex_filter
filter_to_gallium(GLenum filter)
{
   /* Take advantage of how the enums are defined. */
   if (filter & 1)
      return PIPE_TEX_FILTER_LINEAR;

   return PIPE_TEX_FILTER_NEAREST;
}

static inline enum pipe_tex_reduction_mode
reduction_to_gallium(GLenum reduction_mode)
{
   switch (reduction_mode) {
   case GL_MIN:
      return PIPE_TEX_REDUCTION_MIN;
   case GL_MAX:
      return PIPE_TEX_REDUCTION_MAX;
   case GL_WEIGHTED_AVERAGE_EXT:
   default:
      return PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE;
   }
}

/**
 * Convert an OpenGL compare mode to a pipe tokens.
 */
static inline enum pipe_compare_func
func_to_gallium(GLenum func)
{
   /* Same values, just biased */
   STATIC_ASSERT(PIPE_FUNC_NEVER == GL_NEVER - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_LESS == GL_LESS - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_EQUAL == GL_EQUAL - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_LEQUAL == GL_LEQUAL - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_GREATER == GL_GREATER - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_NOTEQUAL == GL_NOTEQUAL - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_GEQUAL == GL_GEQUAL - GL_NEVER);
   STATIC_ASSERT(PIPE_FUNC_ALWAYS == GL_ALWAYS - GL_NEVER);
   assert(func >= GL_NEVER);
   assert(func <= GL_ALWAYS);
   return (enum pipe_compare_func)(func - GL_NEVER);
}

static inline void
_mesa_update_is_border_color_nonzero(struct gl_sampler_object *samp)
{
   samp->Attrib.IsBorderColorNonZero = samp->Attrib.state.border_color.ui[0] ||
                                       samp->Attrib.state.border_color.ui[1] ||
                                       samp->Attrib.state.border_color.ui[2] ||
                                       samp->Attrib.state.border_color.ui[3];
}

static inline enum pipe_tex_wrap
lower_gl_clamp(enum pipe_tex_wrap old_wrap, GLenum wrap, bool clamp_to_border)
{
   if (wrap == GL_CLAMP)
      return clamp_to_border ? PIPE_TEX_WRAP_CLAMP_TO_BORDER :
                               PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   else if (wrap == GL_MIRROR_CLAMP_EXT)
      return clamp_to_border ? PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER :
                               PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE;
   return old_wrap;
}

static inline void
_mesa_lower_gl_clamp(struct gl_context *ctx, struct gl_sampler_object *samp)
{
   if (ctx->DriverFlags.NewSamplersWithClamp) {
      struct pipe_sampler_state *s = &samp->Attrib.state;
      bool clamp_to_border = s->min_img_filter != PIPE_TEX_FILTER_NEAREST &&
                             s->mag_img_filter != PIPE_TEX_FILTER_NEAREST;

      s->wrap_s = lower_gl_clamp((enum pipe_tex_wrap)s->wrap_s,
                                 samp->Attrib.WrapS, clamp_to_border);
      s->wrap_t = lower_gl_clamp((enum pipe_tex_wrap)s->wrap_t,
                                 samp->Attrib.WrapT, clamp_to_border);
      s->wrap_r = lower_gl_clamp((enum pipe_tex_wrap)s->wrap_r,
                                 samp->Attrib.WrapR, clamp_to_border);
   }
}

static inline GLboolean
is_wrap_gl_clamp(GLint param)
{
   return param == GL_CLAMP || param == GL_MIRROR_CLAMP_EXT;
}

static inline void
update_sampler_gl_clamp(struct gl_context *ctx, struct gl_sampler_object *samp, bool cur_state, bool new_state, gl_sampler_wrap wrap)
{
   if (cur_state == new_state)
      return;
   ctx->NewDriverState |= ctx->DriverFlags.NewSamplersWithClamp;
   if (new_state)
      samp->glclamp_mask |= wrap;
   else
      samp->glclamp_mask &= ~wrap;
}
#ifdef __cplusplus
}
#endif

#endif /* SAMPLEROBJ_H */
