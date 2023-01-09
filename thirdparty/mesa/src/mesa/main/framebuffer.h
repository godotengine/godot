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


#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "mtypes.h"

struct gl_config;
struct gl_context;
struct gl_renderbuffer;

extern struct gl_framebuffer *
_mesa_new_framebuffer(struct gl_context *ctx, GLuint name);

extern void
_mesa_initialize_window_framebuffer(struct gl_framebuffer *fb,
				     const struct gl_config *visual);

extern void
_mesa_initialize_user_framebuffer(struct gl_framebuffer *fb, GLuint name);

extern void
_mesa_destroy_framebuffer(struct gl_framebuffer *buffer);

extern void
_mesa_free_framebuffer_data(struct gl_framebuffer *buffer);

extern void
_mesa_reference_framebuffer_(struct gl_framebuffer **ptr,
                             struct gl_framebuffer *fb);

static inline void
_mesa_reference_framebuffer(struct gl_framebuffer **ptr,
                            struct gl_framebuffer *fb)
{
   if (*ptr != fb)
      _mesa_reference_framebuffer_(ptr, fb);
}

extern void
_mesa_resize_framebuffer(struct gl_context *ctx, struct gl_framebuffer *fb,
                         GLuint width, GLuint height);


extern void
_mesa_resizebuffers( struct gl_context *ctx );

extern void
_mesa_intersect_scissor_bounding_box(const struct gl_context *ctx,
                                     unsigned idx, int *bbox);

static inline GLuint
_mesa_geometric_width(const struct gl_framebuffer *buffer)
{
   return buffer->_HasAttachments ?
      buffer->Width : buffer->DefaultGeometry.Width;
}

static inline GLuint
_mesa_geometric_height(const struct gl_framebuffer *buffer)
{
   return buffer->_HasAttachments ?
      buffer->Height : buffer->DefaultGeometry.Height;
}

static inline GLuint
_mesa_geometric_samples(const struct gl_framebuffer *buffer)
{
   return buffer->_HasAttachments ?
      buffer->Visual.samples :
      buffer->DefaultGeometry._NumSamples;
}

static inline GLuint
_mesa_geometric_layers(const struct gl_framebuffer *buffer)
{
   return buffer->_HasAttachments ?
      buffer->MaxNumLayers : buffer->DefaultGeometry.Layers;
}

#define Y_0_TOP 1
#define Y_0_BOTTOM 2

static inline GLuint
_mesa_fb_orientation(const struct gl_framebuffer *fb)
{
   if (fb && fb->FlipY) {
      /* Drawing into a window (on-screen buffer).
       *
       * Negate Y scale to flip image vertically.
       * The NDC Y coords prior to viewport transformation are in the range
       * [y=-1=bottom, y=1=top]
       * Hardware window coords are in the range [y=0=top, y=H-1=bottom] where
       * H is the window height.
       * Use the viewport transformation to invert Y.
       */
      return Y_0_TOP;
   }
   else {
      /* Drawing into user-created FBO (very likely a texture).
       *
       * For textures, T=0=Bottom, so by extension Y=0=Bottom for rendering.
       */
      return Y_0_BOTTOM;
   }
}

extern void 
_mesa_update_draw_buffer_bounds(struct gl_context *ctx,
                                struct gl_framebuffer *drawFb);

extern void
_mesa_update_framebuffer_visual(struct gl_context *ctx,
				struct gl_framebuffer *fb);

extern void
_mesa_update_framebuffer(struct gl_context *ctx,
                         struct gl_framebuffer *readFb,
                         struct gl_framebuffer *drawFb);

extern GLboolean
_mesa_source_buffer_exists(struct gl_context *ctx, GLenum format);

extern GLboolean
_mesa_dest_buffer_exists(struct gl_context *ctx, GLenum format);

extern GLenum
_mesa_get_color_read_type(struct gl_context *ctx,
                          struct gl_framebuffer *fb,
                          const char *caller);

extern GLenum
_mesa_get_color_read_format(struct gl_context *ctx,
                            struct gl_framebuffer *fb,
                            const char *caller);

extern struct gl_renderbuffer *
_mesa_get_read_renderbuffer_for_format(const struct gl_context *ctx,
                                       GLenum format);

extern void
_mesa_print_framebuffer(const struct gl_framebuffer *fb);

extern bool
_mesa_is_multisample_enabled(const struct gl_context *ctx);

extern bool
_mesa_is_alpha_test_enabled(const struct gl_context *ctx);

void
_mesa_draw_buffer_allocate(struct gl_context *ctx);

#endif /* FRAMEBUFFER_H */
