/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_RENDER_SW

#include "../SDL_sysrender.h"
#include "SDL_render_sw_c.h"

#include "SDL_draw.h"
#include "SDL_blendfillrect.h"
#include "SDL_blendline.h"
#include "SDL_blendpoint.h"
#include "SDL_drawline.h"
#include "SDL_drawpoint.h"
#include "SDL_rotate.h"
#include "SDL_triangle.h"
#include "../../video/SDL_pixels_c.h"

// SDL surface based renderer implementation

typedef struct
{
    const SDL_Rect *viewport;
    const SDL_Rect *cliprect;
    bool surface_cliprect_dirty;
    SDL_Color color;
} SW_DrawStateCache;

typedef struct
{
    SDL_Surface *surface;
    SDL_Surface *window;
} SW_RenderData;

static SDL_Surface *SW_ActivateRenderer(SDL_Renderer *renderer)
{
    SW_RenderData *data = (SW_RenderData *)renderer->internal;

    if (!data->surface) {
        data->surface = data->window;
    }
    if (!data->surface) {
        SDL_Surface *surface = SDL_GetWindowSurface(renderer->window);
        if (surface) {
            data->surface = data->window = surface;
        }
    }
    return data->surface;
}

static void SW_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    SW_RenderData *data = (SW_RenderData *)renderer->internal;

    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        data->surface = NULL;
        data->window = NULL;
    }
}

static bool SW_GetOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    SW_RenderData *data = (SW_RenderData *)renderer->internal;

    if (data->surface) {
        if (w) {
            *w = data->surface->w;
        }
        if (h) {
            *h = data->surface->h;
        }
        return true;
    }

    if (renderer->window) {
        SDL_GetWindowSizeInPixels(renderer->window, w, h);
        return true;
    }

    return SDL_SetError("Software renderer doesn't have an output surface");
}

static bool SW_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    SDL_Surface *surface = SDL_CreateSurface(texture->w, texture->h, texture->format);
    Uint8 r, g, b, a;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_SetError("Cannot create surface");
    }
    texture->internal = surface;
    r = (Uint8)SDL_roundf(SDL_clamp(texture->color.r, 0.0f, 1.0f) * 255.0f);
    g = (Uint8)SDL_roundf(SDL_clamp(texture->color.g, 0.0f, 1.0f) * 255.0f);
    b = (Uint8)SDL_roundf(SDL_clamp(texture->color.b, 0.0f, 1.0f) * 255.0f);
    a = (Uint8)SDL_roundf(SDL_clamp(texture->color.a, 0.0f, 1.0f) * 255.0f);
    SDL_SetSurfaceColorMod(surface, r, g, b);
    SDL_SetSurfaceAlphaMod(surface, a);
    SDL_SetSurfaceBlendMode(surface, texture->blendMode);

    /* Only RLE encode textures without an alpha channel since the RLE coder
     * discards the color values of pixels with an alpha value of zero.
     */
    if (texture->access == SDL_TEXTUREACCESS_STATIC && !SDL_ISPIXELFORMAT_ALPHA(surface->format)) {
        SDL_SetSurfaceRLE(surface, 1);
    }

    return true;
}

static bool SW_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect, const void *pixels, int pitch)
{
    SDL_Surface *surface = (SDL_Surface *)texture->internal;
    Uint8 *src, *dst;
    int row;
    size_t length;

    if (SDL_MUSTLOCK(surface)) {
        if (!SDL_LockSurface(surface)) {
            return false;
        }
    }
    src = (Uint8 *)pixels;
    dst = (Uint8 *)surface->pixels +
          rect->y * surface->pitch +
          rect->x * surface->fmt->bytes_per_pixel;
    length = (size_t)rect->w * surface->fmt->bytes_per_pixel;
    for (row = 0; row < rect->h; ++row) {
        SDL_memcpy(dst, src, length);
        src += pitch;
        dst += surface->pitch;
    }
    if (SDL_MUSTLOCK(surface)) {
        SDL_UnlockSurface(surface);
    }
    return true;
}

static bool SW_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                          const SDL_Rect *rect, void **pixels, int *pitch)
{
    SDL_Surface *surface = (SDL_Surface *)texture->internal;

    *pixels =
        (void *)((Uint8 *)surface->pixels + rect->y * surface->pitch +
                 rect->x * surface->fmt->bytes_per_pixel);
    *pitch = surface->pitch;
    return true;
}

static void SW_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
}

static bool SW_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    SW_RenderData *data = (SW_RenderData *)renderer->internal;

    if (texture) {
        data->surface = (SDL_Surface *)texture->internal;
    } else {
        data->surface = data->window;
    }
    return true;
}

static bool SW_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool SW_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    SDL_Point *verts = (SDL_Point *)SDL_AllocateRenderVertices(renderer, count * sizeof(SDL_Point), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, points++) {
        verts->x = (int)points->x;
        verts->y = (int)points->y;
    }

    return true;
}

static bool SW_QueueFillRects(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FRect *rects, int count)
{
    SDL_Rect *verts = (SDL_Rect *)SDL_AllocateRenderVertices(renderer, count * sizeof(SDL_Rect), 0, &cmd->data.draw.first);
    int i;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;

    for (i = 0; i < count; i++, verts++, rects++) {
        verts->x = (int)rects->x;
        verts->y = (int)rects->y;
        verts->w = SDL_max((int)rects->w, 1);
        verts->h = SDL_max((int)rects->h, 1);
    }

    return true;
}

static bool SW_QueueCopy(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                        const SDL_FRect *srcrect, const SDL_FRect *dstrect)
{
    SDL_Rect *verts = (SDL_Rect *)SDL_AllocateRenderVertices(renderer, 2 * sizeof(SDL_Rect), 0, &cmd->data.draw.first);

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = 1;

    verts->x = (int)srcrect->x;
    verts->y = (int)srcrect->y;
    verts->w = (int)srcrect->w;
    verts->h = (int)srcrect->h;
    verts++;

    verts->x = (int)dstrect->x;
    verts->y = (int)dstrect->y;
    verts->w = (int)dstrect->w;
    verts->h = (int)dstrect->h;

    return true;
}

typedef struct CopyExData
{
    SDL_Rect srcrect;
    SDL_Rect dstrect;
    double angle;
    SDL_FPoint center;
    SDL_FlipMode flip;
    float scale_x;
    float scale_y;
} CopyExData;

static bool SW_QueueCopyEx(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                          const SDL_FRect *srcrect, const SDL_FRect *dstrect,
                          const double angle, const SDL_FPoint *center, const SDL_FlipMode flip, float scale_x, float scale_y)
{
    CopyExData *verts = (CopyExData *)SDL_AllocateRenderVertices(renderer, sizeof(CopyExData), 0, &cmd->data.draw.first);

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = 1;

    verts->srcrect.x = (int)srcrect->x;
    verts->srcrect.y = (int)srcrect->y;
    verts->srcrect.w = (int)srcrect->w;
    verts->srcrect.h = (int)srcrect->h;
    verts->dstrect.x = (int)dstrect->x;
    verts->dstrect.y = (int)dstrect->y;
    verts->dstrect.w = (int)dstrect->w;
    verts->dstrect.h = (int)dstrect->h;
    verts->angle = angle;
    SDL_copyp(&verts->center, center);
    verts->flip = flip;
    verts->scale_x = scale_x;
    verts->scale_y = scale_y;

    return true;
}

static bool Blit_to_Screen(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *surface, SDL_Rect *dstrect,
                          float scale_x, float scale_y, SDL_ScaleMode scaleMode)
{
    bool result;
    // Renderer scaling, if needed
    if (scale_x != 1.0f || scale_y != 1.0f) {
        SDL_Rect r;
        r.x = (int)((float)dstrect->x * scale_x);
        r.y = (int)((float)dstrect->y * scale_y);
        r.w = (int)((float)dstrect->w * scale_x);
        r.h = (int)((float)dstrect->h * scale_y);
        result = SDL_BlitSurfaceScaled(src, srcrect, surface, &r, scaleMode);
    } else {
        result = SDL_BlitSurface(src, srcrect, surface, dstrect);
    }
    return result;
}

static bool SW_RenderCopyEx(SDL_Renderer *renderer, SDL_Surface *surface, SDL_Texture *texture,
                            const SDL_Rect *srcrect, const SDL_Rect *final_rect,
                            const double angle, const SDL_FPoint *center, const SDL_FlipMode flip, float scale_x, float scale_y, const SDL_ScaleMode scaleMode)
{
    SDL_Surface *src = (SDL_Surface *)texture->internal;
    SDL_Rect tmp_rect;
    SDL_Surface *src_clone, *src_rotated, *src_scaled;
    SDL_Surface *mask = NULL, *mask_rotated = NULL;
    bool result = true;
    SDL_BlendMode blendmode;
    Uint8 alphaMod, rMod, gMod, bMod;
    int applyModulation = false;
    int blitRequired = false;
    int isOpaque = false;

    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    tmp_rect.x = 0;
    tmp_rect.y = 0;
    tmp_rect.w = final_rect->w;
    tmp_rect.h = final_rect->h;

    /* It is possible to encounter an RLE encoded surface here and locking it is
     * necessary because this code is going to access the pixel buffer directly.
     */
    if (SDL_MUSTLOCK(src)) {
        if (!SDL_LockSurface(src)) {
            return false;
        }
    }

    /* Clone the source surface but use its pixel buffer directly.
     * The original source surface must be treated as read-only.
     */
    src_clone = SDL_CreateSurfaceFrom(src->w, src->h, src->format, src->pixels, src->pitch);
    if (!src_clone) {
        if (SDL_MUSTLOCK(src)) {
            SDL_UnlockSurface(src);
        }
        return false;
    }

    SDL_GetSurfaceBlendMode(src, &blendmode);
    SDL_GetSurfaceAlphaMod(src, &alphaMod);
    SDL_GetSurfaceColorMod(src, &rMod, &gMod, &bMod);

    // SDLgfx_rotateSurface only accepts 32-bit surfaces with a 8888 layout. Everything else has to be converted.
    if (src->fmt->bits_per_pixel != 32 || SDL_PIXELLAYOUT(src->format) != SDL_PACKEDLAYOUT_8888 || !SDL_ISPIXELFORMAT_ALPHA(src->format)) {
        blitRequired = true;
    }

    // If scaling and cropping is necessary, it has to be taken care of before the rotation.
    if (!(srcrect->w == final_rect->w && srcrect->h == final_rect->h && srcrect->x == 0 && srcrect->y == 0)) {
        blitRequired = true;
    }

    // srcrect is not selecting the whole src surface, so cropping is needed
    if (!(srcrect->w == src->w && srcrect->h == src->h && srcrect->x == 0 && srcrect->y == 0)) {
        blitRequired = true;
    }

    // The color and alpha modulation has to be applied before the rotation when using the NONE, MOD or MUL blend modes.
    if ((blendmode == SDL_BLENDMODE_NONE || blendmode == SDL_BLENDMODE_MOD || blendmode == SDL_BLENDMODE_MUL) && (alphaMod & rMod & gMod & bMod) != 255) {
        applyModulation = true;
        SDL_SetSurfaceAlphaMod(src_clone, alphaMod);
        SDL_SetSurfaceColorMod(src_clone, rMod, gMod, bMod);
    }

    // Opaque surfaces are much easier to handle with the NONE blend mode.
    if (blendmode == SDL_BLENDMODE_NONE && !SDL_ISPIXELFORMAT_ALPHA(src->format) && alphaMod == 255) {
        isOpaque = true;
    }

    /* The NONE blend mode requires a mask for non-opaque surfaces. This mask will be used
     * to clear the pixels in the destination surface. The other steps are explained below.
     */
    if (blendmode == SDL_BLENDMODE_NONE && !isOpaque) {
        mask = SDL_CreateSurface(final_rect->w, final_rect->h, SDL_PIXELFORMAT_ARGB8888);
        if (!mask) {
            result = false;
        } else {
            SDL_SetSurfaceBlendMode(mask, SDL_BLENDMODE_MOD);
        }
    }

    /* Create a new surface should there be a format mismatch or if scaling, cropping,
     * or modulation is required. It's possible to use the source surface directly otherwise.
     */
    if (result && (blitRequired || applyModulation)) {
        SDL_Rect scale_rect = tmp_rect;
        src_scaled = SDL_CreateSurface(final_rect->w, final_rect->h, SDL_PIXELFORMAT_ARGB8888);
        if (!src_scaled) {
            result = false;
        } else {
            SDL_SetSurfaceBlendMode(src_clone, SDL_BLENDMODE_NONE);
            result = SDL_BlitSurfaceScaled(src_clone, srcrect, src_scaled, &scale_rect, scaleMode);
            SDL_DestroySurface(src_clone);
            src_clone = src_scaled;
            src_scaled = NULL;
        }
    }

    // SDLgfx_rotateSurface is going to make decisions depending on the blend mode.
    SDL_SetSurfaceBlendMode(src_clone, blendmode);

    if (result) {
        SDL_Rect rect_dest;
        double cangle, sangle;

        SDLgfx_rotozoomSurfaceSizeTrig(tmp_rect.w, tmp_rect.h, angle, center,
                                       &rect_dest, &cangle, &sangle);
        src_rotated = SDLgfx_rotateSurface(src_clone, angle,
                                           (scaleMode == SDL_SCALEMODE_NEAREST || scaleMode == SDL_SCALEMODE_PIXELART) ? 0 : 1, flip & SDL_FLIP_HORIZONTAL, flip & SDL_FLIP_VERTICAL,
                                           &rect_dest, cangle, sangle, center);
        if (!src_rotated) {
            result = false;
        }
        if (result && mask) {
            // The mask needed for the NONE blend mode gets rotated with the same parameters.
            mask_rotated = SDLgfx_rotateSurface(mask, angle,
                                                false, 0, 0,
                                                &rect_dest, cangle, sangle, center);
            if (!mask_rotated) {
                result = false;
            }
        }
        if (result) {

            tmp_rect.x = final_rect->x + rect_dest.x;
            tmp_rect.y = final_rect->y + rect_dest.y;
            tmp_rect.w = rect_dest.w;
            tmp_rect.h = rect_dest.h;

            /* The NONE blend mode needs some special care with non-opaque surfaces.
             * Other blend modes or opaque surfaces can be blitted directly.
             */
            if (blendmode != SDL_BLENDMODE_NONE || isOpaque) {
                if (applyModulation == false) {
                    // If the modulation wasn't already applied, make it happen now.
                    SDL_SetSurfaceAlphaMod(src_rotated, alphaMod);
                    SDL_SetSurfaceColorMod(src_rotated, rMod, gMod, bMod);
                }
                // Renderer scaling, if needed
                result = Blit_to_Screen(src_rotated, NULL, surface, &tmp_rect, scale_x, scale_y, scaleMode);
            } else {
                /* The NONE blend mode requires three steps to get the pixels onto the destination surface.
                 * First, the area where the rotated pixels will be blitted to get set to zero.
                 * This is accomplished by simply blitting a mask with the NONE blend mode.
                 * The colorkey set by the rotate function will discard the correct pixels.
                 */
                SDL_Rect mask_rect = tmp_rect;
                SDL_SetSurfaceBlendMode(mask_rotated, SDL_BLENDMODE_NONE);
                // Renderer scaling, if needed
                result = Blit_to_Screen(mask_rotated, NULL, surface, &mask_rect, scale_x, scale_y, scaleMode);
                if (result) {
                    /* The next step copies the alpha value. This is done with the BLEND blend mode and
                     * by modulating the source colors with 0. Since the destination is all zeros, this
                     * will effectively set the destination alpha to the source alpha.
                     */
                    SDL_SetSurfaceColorMod(src_rotated, 0, 0, 0);
                    mask_rect = tmp_rect;
                    // Renderer scaling, if needed
                    result = Blit_to_Screen(src_rotated, NULL, surface, &mask_rect, scale_x, scale_y, scaleMode);
                    if (result) {
                        /* The last step gets the color values in place. The ADD blend mode simply adds them to
                         * the destination (where the color values are all zero). However, because the ADD blend
                         * mode modulates the colors with the alpha channel, a surface without an alpha mask needs
                         * to be created. This makes all source pixels opaque and the colors get copied correctly.
                         */
                        SDL_Surface *src_rotated_rgb = SDL_CreateSurfaceFrom(src_rotated->w, src_rotated->h, src_rotated->format, src_rotated->pixels, src_rotated->pitch);
                        if (!src_rotated_rgb) {
                            result = false;
                        } else {
                            SDL_SetSurfaceBlendMode(src_rotated_rgb, SDL_BLENDMODE_ADD);
                            // Renderer scaling, if needed
                            result = Blit_to_Screen(src_rotated_rgb, NULL, surface, &tmp_rect, scale_x, scale_y, scaleMode);
                            SDL_DestroySurface(src_rotated_rgb);
                        }
                    }
                }
                SDL_DestroySurface(mask_rotated);
            }
            if (src_rotated) {
                SDL_DestroySurface(src_rotated);
            }
        }
    }

    if (SDL_MUSTLOCK(src)) {
        SDL_UnlockSurface(src);
    }
    if (mask) {
        SDL_DestroySurface(mask);
    }
    if (src_clone) {
        SDL_DestroySurface(src_clone);
    }
    return result;
}

typedef struct GeometryFillData
{
    SDL_Point dst;
    SDL_Color color;
} GeometryFillData;

typedef struct GeometryCopyData
{
    SDL_Point src;
    SDL_Point dst;
    SDL_Color color;
} GeometryCopyData;

static bool SW_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                            const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                            int num_vertices, const void *indices, int num_indices, int size_indices,
                            float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    void *verts;
    size_t sz = texture ? sizeof(GeometryCopyData) : sizeof(GeometryFillData);
    const float color_scale = cmd->data.draw.color_scale;

    verts = SDL_AllocateRenderVertices(renderer, count * sz, 0, &cmd->data.draw.first);
    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    if (texture) {
        GeometryCopyData *ptr = (GeometryCopyData *)verts;
        for (i = 0; i < count; i++) {
            int j;
            float *xy_;
            SDL_FColor col_;
            float *uv_;
            if (size_indices == 4) {
                j = ((const Uint32 *)indices)[i];
            } else if (size_indices == 2) {
                j = ((const Uint16 *)indices)[i];
            } else if (size_indices == 1) {
                j = ((const Uint8 *)indices)[i];
            } else {
                j = i;
            }

            xy_ = (float *)((char *)xy + j * xy_stride);
            col_ = *(SDL_FColor *)((char *)color + j * color_stride);

            uv_ = (float *)((char *)uv + j * uv_stride);

            ptr->src.x = (int)(uv_[0] * texture->w);
            ptr->src.y = (int)(uv_[1] * texture->h);

            ptr->dst.x = (int)(xy_[0] * scale_x);
            ptr->dst.y = (int)(xy_[1] * scale_y);
            trianglepoint_2_fixedpoint(&ptr->dst);

            ptr->color.r = (Uint8)SDL_roundf(SDL_clamp(col_.r * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.g = (Uint8)SDL_roundf(SDL_clamp(col_.g * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.b = (Uint8)SDL_roundf(SDL_clamp(col_.b * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.a = (Uint8)SDL_roundf(SDL_clamp(col_.a, 0.0f, 1.0f) * 255.0f);

            ptr++;
        }
    } else {
        GeometryFillData *ptr = (GeometryFillData *)verts;

        for (i = 0; i < count; i++) {
            int j;
            float *xy_;
            SDL_FColor col_;
            if (size_indices == 4) {
                j = ((const Uint32 *)indices)[i];
            } else if (size_indices == 2) {
                j = ((const Uint16 *)indices)[i];
            } else if (size_indices == 1) {
                j = ((const Uint8 *)indices)[i];
            } else {
                j = i;
            }

            xy_ = (float *)((char *)xy + j * xy_stride);
            col_ = *(SDL_FColor *)((char *)color + j * color_stride);

            ptr->dst.x = (int)(xy_[0] * scale_x);
            ptr->dst.y = (int)(xy_[1] * scale_y);
            trianglepoint_2_fixedpoint(&ptr->dst);

            ptr->color.r = (Uint8)SDL_roundf(SDL_clamp(col_.r * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.g = (Uint8)SDL_roundf(SDL_clamp(col_.g * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.b = (Uint8)SDL_roundf(SDL_clamp(col_.b * color_scale, 0.0f, 1.0f) * 255.0f);
            ptr->color.a = (Uint8)SDL_roundf(SDL_clamp(col_.a, 0.0f, 1.0f) * 255.0f);

            ptr++;
        }
    }
    return true;
}

static void PrepTextureForCopy(const SDL_RenderCommand *cmd, SW_DrawStateCache *drawstate)
{
    const Uint8 r = drawstate->color.r;
    const Uint8 g = drawstate->color.g;
    const Uint8 b = drawstate->color.b;
    const Uint8 a = drawstate->color.a;
    const SDL_BlendMode blend = cmd->data.draw.blend;
    SDL_Texture *texture = cmd->data.draw.texture;
    SDL_Surface *surface = (SDL_Surface *)texture->internal;
    const bool colormod = ((r & g & b) != 0xFF);
    const bool alphamod = (a != 0xFF);
    const bool blending = ((blend == SDL_BLENDMODE_ADD) || (blend == SDL_BLENDMODE_MOD) || (blend == SDL_BLENDMODE_MUL));

    if (colormod || alphamod || blending) {
        SDL_SetSurfaceRLE(surface, 0);
    }

    // !!! FIXME: we can probably avoid some of these calls.
    SDL_SetSurfaceColorMod(surface, r, g, b);
    SDL_SetSurfaceAlphaMod(surface, a);
    SDL_SetSurfaceBlendMode(surface, blend);
}

static void SetDrawState(SDL_Surface *surface, SW_DrawStateCache *drawstate)
{
    if (drawstate->surface_cliprect_dirty) {
        const SDL_Rect *viewport = drawstate->viewport;
        const SDL_Rect *cliprect = drawstate->cliprect;
        SDL_assert_release(viewport != NULL); // the higher level should have forced a SDL_RENDERCMD_SETVIEWPORT

        if (cliprect && viewport) {
            SDL_Rect clip_rect;
            clip_rect.x = cliprect->x + viewport->x;
            clip_rect.y = cliprect->y + viewport->y;
            clip_rect.w = cliprect->w;
            clip_rect.h = cliprect->h;
            SDL_GetRectIntersection(viewport, &clip_rect, &clip_rect);
            SDL_SetSurfaceClipRect(surface, &clip_rect);
        } else {
            SDL_SetSurfaceClipRect(surface, drawstate->viewport);
        }
        drawstate->surface_cliprect_dirty = false;
    }
}

static void SW_InvalidateCachedState(SDL_Renderer *renderer)
{
    // SW_DrawStateCache only lives during SW_RunCommandQueue, so nothing to do here!
}


static bool SW_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    SDL_Surface *surface = SW_ActivateRenderer(renderer);
    SW_DrawStateCache drawstate;

    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    drawstate.viewport = NULL;
    drawstate.cliprect = NULL;
    drawstate.surface_cliprect_dirty = true;
    drawstate.color.r = 0;
    drawstate.color.g = 0;
    drawstate.color.b = 0;
    drawstate.color.a = 0;

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            drawstate.color.r = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.r * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            drawstate.color.g = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.g * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            drawstate.color.b = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.b * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            drawstate.color.a = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.a, 0.0f, 1.0f) * 255.0f);
            break;
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            drawstate.viewport = &cmd->data.viewport.rect;
            drawstate.surface_cliprect_dirty = true;
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            drawstate.cliprect = cmd->data.cliprect.enabled ? &cmd->data.cliprect.rect : NULL;
            drawstate.surface_cliprect_dirty = true;
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            const Uint8 r = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.r * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            const Uint8 g = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.g * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            const Uint8 b = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.b * cmd->data.color.color_scale, 0.0f, 1.0f) * 255.0f);
            const Uint8 a = (Uint8)SDL_roundf(SDL_clamp(cmd->data.color.color.a, 0.0f, 1.0f) * 255.0f);
            // By definition the clear ignores the clip rect
            SDL_SetSurfaceClipRect(surface, NULL);
            SDL_FillSurfaceRect(surface, NULL, SDL_MapSurfaceRGBA(surface, r, g, b, a));
            drawstate.surface_cliprect_dirty = true;
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const Uint8 r = drawstate.color.r;
            const Uint8 g = drawstate.color.g;
            const Uint8 b = drawstate.color.b;
            const Uint8 a = drawstate.color.a;
            const int count = (int)cmd->data.draw.count;
            SDL_Point *verts = (SDL_Point *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const SDL_BlendMode blend = cmd->data.draw.blend;
            SetDrawState(surface, &drawstate);

            // Apply viewport
            if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                int i;
                for (i = 0; i < count; i++) {
                    verts[i].x += drawstate.viewport->x;
                    verts[i].y += drawstate.viewport->y;
                }
            }

            if (blend == SDL_BLENDMODE_NONE) {
                SDL_DrawPoints(surface, verts, count, SDL_MapSurfaceRGBA(surface, r, g, b, a));
            } else {
                SDL_BlendPoints(surface, verts, count, blend, r, g, b, a);
            }
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const Uint8 r = drawstate.color.r;
            const Uint8 g = drawstate.color.g;
            const Uint8 b = drawstate.color.b;
            const Uint8 a = drawstate.color.a;
            const int count = (int)cmd->data.draw.count;
            SDL_Point *verts = (SDL_Point *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const SDL_BlendMode blend = cmd->data.draw.blend;
            SetDrawState(surface, &drawstate);

            // Apply viewport
            if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                int i;
                for (i = 0; i < count; i++) {
                    verts[i].x += drawstate.viewport->x;
                    verts[i].y += drawstate.viewport->y;
                }
            }

            if (blend == SDL_BLENDMODE_NONE) {
                SDL_DrawLines(surface, verts, count, SDL_MapSurfaceRGBA(surface, r, g, b, a));
            } else {
                SDL_BlendLines(surface, verts, count, blend, r, g, b, a);
            }
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS:
        {
            const Uint8 r = drawstate.color.r;
            const Uint8 g = drawstate.color.g;
            const Uint8 b = drawstate.color.b;
            const Uint8 a = drawstate.color.a;
            const int count = (int)cmd->data.draw.count;
            SDL_Rect *verts = (SDL_Rect *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const SDL_BlendMode blend = cmd->data.draw.blend;
            SetDrawState(surface, &drawstate);

            // Apply viewport
            if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                int i;
                for (i = 0; i < count; i++) {
                    verts[i].x += drawstate.viewport->x;
                    verts[i].y += drawstate.viewport->y;
                }
            }

            if (blend == SDL_BLENDMODE_NONE) {
                SDL_FillSurfaceRects(surface, verts, count, SDL_MapSurfaceRGBA(surface, r, g, b, a));
            } else {
                SDL_BlendFillRects(surface, verts, count, blend, r, g, b, a);
            }
            break;
        }

        case SDL_RENDERCMD_COPY:
        {
            SDL_Rect *verts = (SDL_Rect *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const SDL_Rect *srcrect = verts;
            SDL_Rect *dstrect = verts + 1;
            SDL_Texture *texture = cmd->data.draw.texture;
            SDL_Surface *src = (SDL_Surface *)texture->internal;

            SetDrawState(surface, &drawstate);

            PrepTextureForCopy(cmd, &drawstate);

            // Apply viewport
            if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                dstrect->x += drawstate.viewport->x;
                dstrect->y += drawstate.viewport->y;
            }

            if (srcrect->w == dstrect->w && srcrect->h == dstrect->h) {
                SDL_BlitSurface(src, srcrect, surface, dstrect);
            } else {
                /* If scaling is ever done, permanently disable RLE (which doesn't support scaling)
                 * to avoid potentially frequent RLE encoding/decoding.
                 */
                SDL_SetSurfaceRLE(surface, 0);

                // Prevent to do scaling + clipping on viewport boundaries as it may lose proportion
                if (dstrect->x < 0 || dstrect->y < 0 || dstrect->x + dstrect->w > surface->w || dstrect->y + dstrect->h > surface->h) {
                    SDL_Surface *tmp = SDL_CreateSurface(dstrect->w, dstrect->h, src->format);
                    // Scale to an intermediate surface, then blit
                    if (tmp) {
                        SDL_Rect r;
                        SDL_BlendMode blendmode;
                        Uint8 alphaMod, rMod, gMod, bMod;

                        SDL_GetSurfaceBlendMode(src, &blendmode);
                        SDL_GetSurfaceAlphaMod(src, &alphaMod);
                        SDL_GetSurfaceColorMod(src, &rMod, &gMod, &bMod);

                        r.x = 0;
                        r.y = 0;
                        r.w = dstrect->w;
                        r.h = dstrect->h;

                        SDL_SetSurfaceBlendMode(src, SDL_BLENDMODE_NONE);
                        SDL_SetSurfaceColorMod(src, 255, 255, 255);
                        SDL_SetSurfaceAlphaMod(src, 255);

                        SDL_BlitSurfaceScaled(src, srcrect, tmp, &r, cmd->data.draw.texture_scale_mode);

                        SDL_SetSurfaceColorMod(tmp, rMod, gMod, bMod);
                        SDL_SetSurfaceAlphaMod(tmp, alphaMod);
                        SDL_SetSurfaceBlendMode(tmp, blendmode);

                        SDL_BlitSurface(tmp, NULL, surface, dstrect);
                        SDL_DestroySurface(tmp);
                        // No need to set back r/g/b/a/blendmode to 'src' since it's done in PrepTextureForCopy()
                    }
                } else {
                    SDL_BlitSurfaceScaled(src, srcrect, surface, dstrect, cmd->data.draw.texture_scale_mode);
                }
            }
            break;
        }

        case SDL_RENDERCMD_COPY_EX:
        {
            CopyExData *copydata = (CopyExData *)(((Uint8 *)vertices) + cmd->data.draw.first);
            SetDrawState(surface, &drawstate);
            PrepTextureForCopy(cmd, &drawstate);

            // Apply viewport
            if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                copydata->dstrect.x += drawstate.viewport->x;
                copydata->dstrect.y += drawstate.viewport->y;
            }

            SW_RenderCopyEx(renderer, surface, cmd->data.draw.texture, &copydata->srcrect,
                            &copydata->dstrect, copydata->angle, &copydata->center, copydata->flip,
                            copydata->scale_x, copydata->scale_y, cmd->data.draw.texture_scale_mode);
            break;
        }

        case SDL_RENDERCMD_GEOMETRY:
        {
            int i;
            SDL_Rect *verts = (SDL_Rect *)(((Uint8 *)vertices) + cmd->data.draw.first);
            const int count = (int)cmd->data.draw.count;
            SDL_Texture *texture = cmd->data.draw.texture;
            const SDL_BlendMode blend = cmd->data.draw.blend;

            SetDrawState(surface, &drawstate);

            if (texture) {
                SDL_Surface *src = (SDL_Surface *)texture->internal;

                GeometryCopyData *ptr = (GeometryCopyData *)verts;

                PrepTextureForCopy(cmd, &drawstate);

                // Apply viewport
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    SDL_Point vp;
                    vp.x = drawstate.viewport->x;
                    vp.y = drawstate.viewport->y;
                    trianglepoint_2_fixedpoint(&vp);
                    for (i = 0; i < count; i++) {
                        ptr[i].dst.x += vp.x;
                        ptr[i].dst.y += vp.y;
                    }
                }

                for (i = 0; i < count; i += 3, ptr += 3) {
                    SDL_SW_BlitTriangle(
                        src,
                        &(ptr[0].src), &(ptr[1].src), &(ptr[2].src),
                        surface,
                        &(ptr[0].dst), &(ptr[1].dst), &(ptr[2].dst),
                        ptr[0].color, ptr[1].color, ptr[2].color,
                        cmd->data.draw.texture_address_mode_u,
                        cmd->data.draw.texture_address_mode_v);
                }
            } else {
                GeometryFillData *ptr = (GeometryFillData *)verts;

                // Apply viewport
                if (drawstate.viewport && (drawstate.viewport->x || drawstate.viewport->y)) {
                    SDL_Point vp;
                    vp.x = drawstate.viewport->x;
                    vp.y = drawstate.viewport->y;
                    trianglepoint_2_fixedpoint(&vp);
                    for (i = 0; i < count; i++) {
                        ptr[i].dst.x += vp.x;
                        ptr[i].dst.y += vp.y;
                    }
                }

                for (i = 0; i < count; i += 3, ptr += 3) {
                    SDL_SW_FillTriangle(surface, &(ptr[0].dst), &(ptr[1].dst), &(ptr[2].dst), blend, ptr[0].color, ptr[1].color, ptr[2].color);
                }
            }
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    return true;
}

static SDL_Surface *SW_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    SDL_Surface *surface = SW_ActivateRenderer(renderer);
    void *pixels;

    if (!SDL_SurfaceValid(surface)) {
        return NULL;
    }

    /* NOTE: The rect is already adjusted according to the viewport by
     * SDL_RenderReadPixels.
     */

    if (rect->x < 0 || rect->x + rect->w > surface->w ||
        rect->y < 0 || rect->y + rect->h > surface->h) {
        SDL_SetError("Tried to read outside of surface bounds");
        return NULL;
    }

    pixels = (void *)((Uint8 *)surface->pixels +
                      rect->y * surface->pitch +
                      rect->x * surface->fmt->bytes_per_pixel);

    return SDL_DuplicatePixels(rect->w, rect->h, surface->format, SDL_COLORSPACE_SRGB, pixels, surface->pitch);
}

static bool SW_RenderPresent(SDL_Renderer *renderer)
{
    SDL_Window *window = renderer->window;

    if (!window) {
        return false;
    }
    return SDL_UpdateWindowSurface(window);
}

static void SW_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    SDL_Surface *surface = (SDL_Surface *)texture->internal;

    SDL_DestroySurface(surface);
}

static void SW_DestroyRenderer(SDL_Renderer *renderer)
{
    SDL_Window *window = renderer->window;
    SW_RenderData *data = (SW_RenderData *)renderer->internal;

    if (window) {
        SDL_DestroyWindowSurface(window);
    }
    SDL_free(data);
}

static void SW_SelectBestFormats(SDL_Renderer *renderer, SDL_PixelFormat format)
{
    // Prefer the format used by the framebuffer by default.
    SDL_AddSupportedTextureFormat(renderer, format);

    switch (format) {
    case SDL_PIXELFORMAT_XRGB4444:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB4444);
        break;
    case SDL_PIXELFORMAT_XBGR4444:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR4444);
        break;
    case SDL_PIXELFORMAT_ARGB4444:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB4444);
        break;
    case SDL_PIXELFORMAT_ABGR4444:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XBGR4444);
        break;

    case SDL_PIXELFORMAT_XRGB1555:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB1555);
        break;
    case SDL_PIXELFORMAT_XBGR1555:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR1555);
        break;
    case SDL_PIXELFORMAT_ARGB1555:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB1555);
        break;
    case SDL_PIXELFORMAT_ABGR1555:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XBGR1555);
        break;

    case SDL_PIXELFORMAT_XRGB8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
        break;
    case SDL_PIXELFORMAT_RGBX8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA8888);
        break;
    case SDL_PIXELFORMAT_XBGR8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
        break;
    case SDL_PIXELFORMAT_BGRX8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRA8888);
        break;
    case SDL_PIXELFORMAT_ARGB8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB8888);
        break;
    case SDL_PIXELFORMAT_RGBA8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBX8888);
        break;
    case SDL_PIXELFORMAT_ABGR8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XBGR8888);
        break;
    case SDL_PIXELFORMAT_BGRA8888:
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRX8888);
        break;
    default:
        break;
    }

    /* Ensure that we always have a SDL_PACKEDLAYOUT_8888 format. Having a matching component order increases the
     * chances of getting a fast path for blitting.
     */
    if (SDL_ISPIXELFORMAT_PACKED(format)) {
        if (SDL_PIXELLAYOUT(format) != SDL_PACKEDLAYOUT_8888) {
            switch (SDL_PIXELORDER(format)) {
            case SDL_PACKEDORDER_BGRX:
            case SDL_PACKEDORDER_BGRA:
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRX8888);
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRA8888);
                break;
            case SDL_PACKEDORDER_RGBX:
            case SDL_PACKEDORDER_RGBA:
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBX8888);
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA8888);
                break;
            case SDL_PACKEDORDER_XBGR:
            case SDL_PACKEDORDER_ABGR:
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XBGR8888);
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
                break;
            case SDL_PACKEDORDER_XRGB:
            case SDL_PACKEDORDER_ARGB:
            default:
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB8888);
                SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
                break;
            }
        }
    } else {
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_XRGB8888);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
    }
}

bool SW_CreateRendererForSurface(SDL_Renderer *renderer, SDL_Surface *surface, SDL_PropertiesID create_props)
{
    SW_RenderData *data;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    renderer->software = true;

    data = (SW_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }
    data->surface = surface;
    data->window = surface;

    renderer->WindowEvent = SW_WindowEvent;
    renderer->GetOutputSize = SW_GetOutputSize;
    renderer->CreateTexture = SW_CreateTexture;
    renderer->UpdateTexture = SW_UpdateTexture;
    renderer->LockTexture = SW_LockTexture;
    renderer->UnlockTexture = SW_UnlockTexture;
    renderer->SetRenderTarget = SW_SetRenderTarget;
    renderer->QueueSetViewport = SW_QueueNoOp;
    renderer->QueueSetDrawColor = SW_QueueNoOp;
    renderer->QueueDrawPoints = SW_QueueDrawPoints;
    renderer->QueueDrawLines = SW_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueFillRects = SW_QueueFillRects;
    renderer->QueueCopy = SW_QueueCopy;
    renderer->QueueCopyEx = SW_QueueCopyEx;
    renderer->QueueGeometry = SW_QueueGeometry;
    renderer->InvalidateCachedState = SW_InvalidateCachedState;
    renderer->RunCommandQueue = SW_RunCommandQueue;
    renderer->RenderReadPixels = SW_RenderReadPixels;
    renderer->RenderPresent = SW_RenderPresent;
    renderer->DestroyTexture = SW_DestroyTexture;
    renderer->DestroyRenderer = SW_DestroyRenderer;
    renderer->internal = data;
    SW_InvalidateCachedState(renderer);

    renderer->name = SW_RenderDriver.name;

    SW_SelectBestFormats(renderer, surface->format);

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB) {
        return SDL_SetError("Unsupported output colorspace");
    }

    return true;
}

static bool SW_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    // Set the vsync hint based on our flags, if it's not already set
    const char *hint = SDL_GetHint(SDL_HINT_RENDER_VSYNC);
    const bool no_hint_set = (!hint || !*hint);

    if (no_hint_set) {
        if (SDL_GetBooleanProperty(create_props, SDL_PROP_RENDERER_CREATE_PRESENT_VSYNC_NUMBER, 0)) {
            SDL_SetHint(SDL_HINT_RENDER_VSYNC, "1");
        } else {
            SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
        }
    }

    SDL_Surface *surface = SDL_GetWindowSurface(window);

    // Reset the vsync hint if we set it above
    if (no_hint_set) {
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "");
    }

    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    return SW_CreateRendererForSurface(renderer, surface, create_props);
}

SDL_RenderDriver SW_RenderDriver = {
    SW_CreateRenderer, SDL_SOFTWARE_RENDERER
};

#endif // SDL_VIDEO_RENDER_SW
