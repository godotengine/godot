/**
 * \file blend.h
 * Blending functions operations.
 */

/*
 * Mesa 3-D graphics library
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



#ifndef BLEND_H
#define BLEND_H


#include "util/glheader.h"
#include "context.h"
#include "formats.h"
#include "extensions.h"

#include "state_tracker/st_context.h"

struct gl_context;
struct gl_framebuffer;

extern GLboolean
_mesa_get_clamp_fragment_color(const struct gl_context *ctx,
                               const struct gl_framebuffer *drawFb);

extern GLboolean
_mesa_get_clamp_vertex_color(const struct gl_context *ctx,
                             const struct gl_framebuffer *drawFb);

extern GLboolean
_mesa_get_clamp_read_color(const struct gl_context *ctx,
                           const struct gl_framebuffer *readFb);

extern void
_mesa_update_clamp_fragment_color(struct gl_context *ctx,
                                  const struct gl_framebuffer *drawFb);

extern void
_mesa_update_clamp_vertex_color(struct gl_context *ctx,
                                const struct gl_framebuffer *drawFb);

extern void  
_mesa_init_color( struct gl_context * ctx );


static inline enum gl_advanced_blend_mode
_mesa_get_advanced_blend_sh_constant(GLbitfield blend_enabled,
                                     enum gl_advanced_blend_mode mode)
{
   return blend_enabled ? mode : BLEND_NONE;
}

static inline bool
_mesa_advanded_blend_sh_constant_changed(struct gl_context *ctx,
                                         GLbitfield new_blend_enabled,
                                         enum gl_advanced_blend_mode new_mode)
{
   return _mesa_get_advanced_blend_sh_constant(new_blend_enabled, new_mode) !=
          _mesa_get_advanced_blend_sh_constant(ctx->Color.BlendEnabled,
                                               ctx->Color._AdvancedBlendMode);
}

static inline void
_mesa_flush_vertices_for_blend_state(struct gl_context *ctx)
{
   FLUSH_VERTICES(ctx, 0, GL_COLOR_BUFFER_BIT);
   ctx->NewDriverState |= ST_NEW_BLEND;
}

static inline void
_mesa_flush_vertices_for_blend_adv(struct gl_context *ctx,
                                   GLbitfield new_blend_enabled,
                                   enum gl_advanced_blend_mode new_mode)
{
   /* The advanced blend mode needs _NEW_COLOR to update the state constant. */
   if (_mesa_has_KHR_blend_equation_advanced(ctx) &&
       _mesa_advanded_blend_sh_constant_changed(ctx, new_blend_enabled,
                                                new_mode)) {
      FLUSH_VERTICES(ctx, _NEW_COLOR, GL_COLOR_BUFFER_BIT);
      ctx->NewDriverState |= ST_NEW_BLEND;
      return;
   }
   _mesa_flush_vertices_for_blend_state(ctx);
}

static inline GLbitfield
_mesa_replicate_colormask(GLbitfield mask0, unsigned num_buffers)
{
   GLbitfield mask = mask0;

   for (unsigned i = 1; i < num_buffers; i++)
      mask |= mask0 << (i * 4);
   return mask;
}

#endif
