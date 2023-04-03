/**
 * \file texobj.h
 * Texture object management.
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


#ifndef TEXTOBJ_H
#define TEXTOBJ_H


#include "util/glheader.h"
#include "samplerobj.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * \name Internal functions
 */
/*@{*/

extern struct gl_texture_object *
_mesa_lookup_texture(struct gl_context *ctx, GLuint id);

extern struct gl_texture_object *
_mesa_lookup_texture_err(struct gl_context *ctx, GLuint id, const char* func);

extern struct gl_texture_object *
_mesa_lookup_texture_locked(struct gl_context *ctx, GLuint id);

extern struct gl_texture_object *
_mesa_get_current_tex_object(struct gl_context *ctx, GLenum target);

extern struct gl_texture_object *
_mesa_get_texobj_by_target_and_texunit(struct gl_context *ctx, GLenum target,
                                       GLuint texunit,
                                       bool allowProxyTargets,
                                       const char* caller);

extern struct gl_texture_object *
_mesa_new_texture_object( struct gl_context *ctx, GLuint name, GLenum target );

extern int
_mesa_tex_target_to_index(const struct gl_context *ctx, GLenum target);

extern void
_mesa_delete_texture_object( struct gl_context *ctx,
                             struct gl_texture_object *obj );

extern void
_mesa_clear_texture_object(struct gl_context *ctx,
                           struct gl_texture_object *obj,
                           struct gl_texture_image *retainTexImage);

extern void
_mesa_reference_texobj_(struct gl_texture_object **ptr,
                        struct gl_texture_object *tex);

static inline void
_mesa_reference_texobj(struct gl_texture_object **ptr,
                       struct gl_texture_object *tex)
{
   if (*ptr != tex)
      _mesa_reference_texobj_(ptr, tex);
}

/**
 * Lock a texture for updating.  See also _mesa_lock_context_textures().
 */
static inline void
_mesa_lock_texture(struct gl_context *ctx, struct gl_texture_object *texObj)
{
   if (!ctx->TexturesLocked)
      simple_mtx_lock(&ctx->Shared->TexMutex);
   ctx->Shared->TextureStateStamp++;
   (void) texObj;
}

static inline void
_mesa_unlock_texture(struct gl_context *ctx, struct gl_texture_object *texObj)
{
   (void) texObj;
   if (!ctx->TexturesLocked)
      simple_mtx_unlock(&ctx->Shared->TexMutex);
}


/** Is the texture "complete" with respect to the given sampler state? */
static inline GLboolean
_mesa_is_texture_complete(const struct gl_texture_object *texObj,
                          const struct gl_sampler_object *sampler,
                          bool linear_as_nearest_for_int_tex)
{
   struct gl_texture_image *img = texObj->Image[0][texObj->Attrib.BaseLevel];
   bool isMultisample = img && img->NumSamples >= 2;

   /*
    * According to ARB_stencil_texturing, NEAREST_MIPMAP_NEAREST would
    * be forbidden, however it is allowed per GL 4.5 rules, allow it
    * even without GL 4.5 since it was a spec mistake.
    */
   /* Section 8.17 (texture completeness) of the OpenGL 4.6 core profile spec:
    *
    *  "The texture is not multisample; either the magnification filter is not
    *  NEAREST, or the minification filter is neither NEAREST nor NEAREST_-
    *  MIPMAP_NEAREST; and any of
    *  – The internal format of the texture is integer.
    *  – The internal format is STENCIL_INDEX.
    *  – The internal format is DEPTH_STENCIL, and the value of DEPTH_-
    *    STENCIL_TEXTURE_MODE for the texture is STENCIL_INDEX.""
    */
   /* GL_EXT_texture_filter_minmax further modifies this to explain it does
    * not apply to MIN/MAX reduction, only WEIGHTED_AVERAGE (i.e. default)
    */
   if (!isMultisample &&
       (texObj->_IsIntegerFormat ||
        (texObj->StencilSampling &&
         img->_BaseFormat == GL_DEPTH_STENCIL)) &&
       sampler->Attrib.ReductionMode == GL_WEIGHTED_AVERAGE_EXT &&
       (sampler->Attrib.MagFilter != GL_NEAREST ||
        (sampler->Attrib.MinFilter != GL_NEAREST &&
         sampler->Attrib.MinFilter != GL_NEAREST_MIPMAP_NEAREST))) {
      /* If the format is integer, only nearest filtering is allowed,
       * but some applications (eg: Grid Autosport) uses the default
       * filtering values.
       */
      if (texObj->_IsIntegerFormat &&
          linear_as_nearest_for_int_tex) {
         /* Skip return */
      } else {
         return GL_FALSE;
      }
   }

   /* Section 8.17 (texture completeness) of the OpenGL 4.6 core profile spec:
    *
    *  "The minification filter requires a mipmap (is neither NEAREST nor LINEAR),
    *  the texture is not multisample, and the texture is not mipmap complete.""
    */
   if (!isMultisample &&_mesa_is_mipmap_filter(sampler))
      return texObj->_MipmapComplete;
   else
      return texObj->_BaseComplete;
}


extern void
_mesa_test_texobj_completeness( const struct gl_context *ctx,
                                struct gl_texture_object *obj );

extern GLboolean
_mesa_cube_level_complete(const struct gl_texture_object *texObj,
                          const GLint level);

extern GLboolean
_mesa_cube_complete(const struct gl_texture_object *texObj);

extern void
_mesa_dirty_texobj(struct gl_context *ctx, struct gl_texture_object *texObj);

extern struct gl_texture_object *
_mesa_get_fallback_texture(struct gl_context *ctx, gl_texture_index tex);

extern GLuint
_mesa_total_texture_memory(struct gl_context *ctx);

extern GLenum
_mesa_texture_base_format(const struct gl_texture_object *texObj);

extern void
_mesa_unlock_context_textures( struct gl_context *ctx );

extern void
_mesa_lock_context_textures( struct gl_context *ctx );

extern struct gl_texture_object *
_mesa_lookup_or_create_texture(struct gl_context *ctx, GLenum target,
                               GLuint texName, bool no_error, bool is_ext_dsa,
                               const char *name);
void
_mesa_update_texture_object_swizzle(struct gl_context *ctx,
                                    struct gl_texture_object *texObj);
/*@}*/

#ifdef __cplusplus
}
#endif


#endif
