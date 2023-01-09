/**
 * \file texstate.h
 * Texture state management.
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


#ifndef TEXSTATE_H
#define TEXSTATE_H


#include "enums.h"
#include "macros.h"
#include "mtypes.h"


static inline struct gl_texture_unit *
_mesa_get_tex_unit(struct gl_context *ctx, GLuint unit)
{
   assert(unit < ARRAY_SIZE(ctx->Texture.Unit));
   return &(ctx->Texture.Unit[unit]);
}

/**
 * Return pointer to current texture unit.
 * This the texture unit set by glActiveTexture(), not glClientActiveTexture().
 */
static inline struct gl_texture_unit *
_mesa_get_current_tex_unit(struct gl_context *ctx)
{
   return _mesa_get_tex_unit(ctx, ctx->Texture.CurrentUnit);
}


/**
 * Return pointer to current fixed-func texture unit.
 * This the texture unit set by glActiveTexture(), not glClientActiveTexture().
 * \return NULL if the current unit is not a fixed-func texture unit
 */
static inline struct gl_fixedfunc_texture_unit *
_mesa_get_fixedfunc_tex_unit(struct gl_context *ctx, GLuint unit)
{
   if (unit >= ARRAY_SIZE(ctx->Texture.FixedFuncUnit))
      return NULL;

   return &ctx->Texture.FixedFuncUnit[unit];
}


static inline GLuint
_mesa_max_tex_unit(struct gl_context *ctx)
{
   /* See OpenGL spec for glActiveTexture: */
   return MAX2(ctx->Const.MaxCombinedTextureImageUnits,
               ctx->Const.MaxTextureCoordUnits);
}


extern void
_mesa_copy_texture_state( const struct gl_context *src, struct gl_context *dst );

extern void
_mesa_print_texunit_state( struct gl_context *ctx, GLuint unit );


/**
 * \name Initialization, state maintenance
 */
/*@{*/

extern GLbitfield
_mesa_update_texture_matrices(struct gl_context *ctx);

extern GLbitfield
_mesa_update_texture_state(struct gl_context *ctx);

extern GLboolean
_mesa_init_texture( struct gl_context *ctx );

extern void 
_mesa_free_texture_data( struct gl_context *ctx );

extern void
_mesa_update_default_objects_texture(struct gl_context *ctx);

/*@}*/

#endif
