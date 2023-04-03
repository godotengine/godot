/**
 * \file stencil.h
 * Stencil operations.
 */

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


#ifndef STENCIL_H
#define STENCIL_H


#include "util/glheader.h"
#include "macros.h"

struct gl_context;

extern void 
_mesa_init_stencil( struct gl_context * ctx );

/* From the GL 4.3 spec, 17.3.5:
 *    "Stencil comparison operations and queries of <ref> clamp its value
 *    to the range [0, 2^s-1], where <s> is the number of bits in the
 *    stencil buffer attached to the draw framebuffer."
 */

static inline GLint
_mesa_get_stencil_ref(struct gl_context const *ctx, int face)
{
   GLint stencilMax = (1 << ctx->DrawBuffer->Visual.stencilBits) - 1;
   GLint ref = ctx->Stencil.Ref[face];
   return CLAMP(ref, 0, stencilMax);
}

static inline bool
_mesa_stencil_is_enabled(const struct gl_context *ctx)
{
   return ctx->Stencil.Enabled &&
          ctx->DrawBuffer->Visual.stencilBits > 0;
}

static inline bool
_mesa_stencil_is_two_sided(const struct gl_context *ctx)
{
   const int face = ctx->Stencil._BackFace;

   return _mesa_stencil_is_enabled(ctx) &&
          (ctx->Stencil.Function[0] != ctx->Stencil.Function[face] ||
           ctx->Stencil.FailFunc[0] != ctx->Stencil.FailFunc[face] ||
           ctx->Stencil.ZPassFunc[0] != ctx->Stencil.ZPassFunc[face] ||
           ctx->Stencil.ZFailFunc[0] != ctx->Stencil.ZFailFunc[face] ||
           ctx->Stencil.Ref[0] != ctx->Stencil.Ref[face] ||
           ctx->Stencil.ValueMask[0] != ctx->Stencil.ValueMask[face] ||
           ctx->Stencil.WriteMask[0] != ctx->Stencil.WriteMask[face]);
}

static inline bool
_mesa_stencil_is_write_enabled(const struct gl_context *ctx, bool is_two_sided)
{
   return _mesa_stencil_is_enabled(ctx) &&
          (ctx->Stencil.WriteMask[0] != 0 ||
           (is_two_sided &&
            ctx->Stencil.WriteMask[ctx->Stencil._BackFace] != 0));
}

#endif
