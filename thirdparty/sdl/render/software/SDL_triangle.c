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

#include <limits.h>

#include "SDL_triangle.h"

#include "../../video/SDL_surface_c.h"

/* fixed points bits precision
 * Set to 1, so that it can start rendering with middle of a pixel precision.
 * It doesn't need to be increased.
 * But, if increased too much, it overflows (srcx, srcy) coordinates used for filling with texture.
 * (which could be turned to int64).
 */
#define FP_BITS 1

#define COLOR_EQ(c1, c2) ((c1).r == (c2).r && (c1).g == (c2).g && (c1).b == (c2).b && (c1).a == (c2).a)

static void SDL_BlitTriangle_Slow(SDL_BlitInfo *info,
                                  SDL_Point s2_x_area, SDL_Rect dstrect, int area, int bias_w0, int bias_w1, int bias_w2,
                                  int d2d1_y, int d1d2_x, int d0d2_y, int d2d0_x, int d1d0_y, int d0d1_x,
                                  int s2s0_x, int s2s1_x, int s2s0_y, int s2s1_y, int w0_row, int w1_row, int w2_row,
                                  SDL_Color c0, SDL_Color c1, SDL_Color c2, bool is_uniform,
                                  SDL_TextureAddressMode texture_address_mode_u,
                                  SDL_TextureAddressMode texture_address_mode_v);

#if 0
bool SDL_BlitTriangle(SDL_Surface *src, const SDL_Point srcpoints[3], SDL_Surface *dst, const SDL_Point dstpoints[3])
{
    int i;
    SDL_Point points[6];

    if (src == NULL || dst == NULL) {
        return false;
    }

    for (i = 0; i < 3; i++) {
        if (srcpoints[i].x < 0 || srcpoints[i].y < 0 || srcpoints[i].x >= src->w || srcpoints[i].y >= src->h) {
            return SDL_SetError("Values of 'srcpoints' out of bounds");
        }
    }

    points[0] = srcpoints[0];
    points[1] = dstpoints[0];
    points[2] = srcpoints[1];
    points[3] = dstpoints[1];
    points[4] = srcpoints[2];
    points[5] = dstpoints[2];
    for (i = 0; i < 3; i++) {
        trianglepoint_2_fixedpoint(&points[2 * i + 1]);
    }
    return SDL_SW_BlitTriangle(src, dst, points);
}

bool SDL_FillTriangle(SDL_Surface *dst, const SDL_Point points[3], Uint32 color)
{
    int i;
    SDL_Point points_tmp[3];
    if (dst == NULL) {
        return false;
    }
    for (i = 0; i < 3; i++) {
        points_tmp[i] = points[i];
        trianglepoint_2_fixedpoint(&points_tmp[i]);
    }
    return SDL_SW_FillTriangle(dst, points_tmp, SDL_BLENDMODE_NONE, color);
}
#endif

// cross product AB x AC
static Sint64 cross_product(const SDL_Point *a, const SDL_Point *b, int c_x, int c_y)
{
    return ((Sint64)(b->x - a->x)) * ((Sint64)(c_y - a->y)) - ((Sint64)(b->y - a->y)) * ((Sint64)(c_x - a->x));
}

// check for top left rules
static bool is_top_left(const SDL_Point *a, const SDL_Point *b, int is_clockwise)
{
    if (is_clockwise) {
        if (a->y == b->y && a->x < b->x) {
            return true;
        }
        if (b->y < a->y) {
            return true;
        }
    } else {
        if (a->y == b->y && b->x < a->x) {
            return true;
        }
        if (a->y < b->y) {
            return true;
        }
    }
    return false;
}

// x = (y << FP_BITS)
// prevent runtime error: left shift of negative value
#define PRECOMP(x, y)               \
        val = y;                    \
        if (val >= 0) {             \
            x = val << FP_BITS;     \
        } else {                    \
            val *= -1;              \
            x = val << FP_BITS;     \
            x *= -1;                \
        }

void trianglepoint_2_fixedpoint(SDL_Point *a)
{
    int val;
    PRECOMP(a->x, a->x);
    PRECOMP(a->y, a->y);
}

// bounding rect of three points (in fixed point)
static void bounding_rect_fixedpoint(const SDL_Point *a, const SDL_Point *b, const SDL_Point *c, SDL_Rect *r)
{
    int min_x = SDL_min(a->x, SDL_min(b->x, c->x));
    int max_x = SDL_max(a->x, SDL_max(b->x, c->x));
    int min_y = SDL_min(a->y, SDL_min(b->y, c->y));
    int max_y = SDL_max(a->y, SDL_max(b->y, c->y));
    // points are in fixed point, shift back
    r->x = min_x >> FP_BITS;
    r->y = min_y >> FP_BITS;
    r->w = (max_x - min_x) >> FP_BITS;
    r->h = (max_y - min_y) >> FP_BITS;
}

// bounding rect of three points
static void bounding_rect(const SDL_Point *a, const SDL_Point *b, const SDL_Point *c, SDL_Rect *r)
{
    int min_x = SDL_min(a->x, SDL_min(b->x, c->x));
    int max_x = SDL_max(a->x, SDL_max(b->x, c->x));
    int min_y = SDL_min(a->y, SDL_min(b->y, c->y));
    int max_y = SDL_max(a->y, SDL_max(b->y, c->y));
    r->x = min_x;
    r->y = min_y;
    r->w = (max_x - min_x);
    r->h = (max_y - min_y);
}

/* Triangle rendering, using Barycentric coordinates (w0, w1, w2)
 *
 * The cross product isn't computed from scratch at each iteration,
 * but optimized using constant step increments
 *
 */

#define TRIANGLE_BEGIN_LOOP                                                        \
    {                                                                              \
        int x, y;                                                                  \
        for (y = 0; y < dstrect.h; y++) {                                          \
            /* y start */                                                          \
            Sint64 w0 = w0_row;                                                    \
            Sint64 w1 = w1_row;                                                    \
            Sint64 w2 = w2_row;                                                    \
            for (x = 0; x < dstrect.w; x++) {                                      \
                /* In triangle */                                                  \
                if (w0 + bias_w0 >= 0 && w1 + bias_w1 >= 0 && w2 + bias_w2 >= 0) { \
                    Uint8 *dptr = (Uint8 *)dst_ptr + x * dstbpp;

// Use 64 bits precision to prevent overflow when interpolating color / texture with wide triangles
#define TRIANGLE_GET_TEXTCOORD                                                          \
    int srcx = (int)(((Sint64)w0 * s2s0_x + (Sint64)w1 * s2s1_x + s2_x_area.x) / area); \
    int srcy = (int)(((Sint64)w0 * s2s0_y + (Sint64)w1 * s2s1_y + s2_x_area.y) / area); \
    if (texture_address_mode_u == SDL_TEXTURE_ADDRESS_WRAP) {                           \
        srcx %= src_surface->w;                                                         \
        if (srcx < 0) {                                                                 \
            srcx += (src_surface->w - 1);                                               \
        }                                                                               \
    }                                                                                   \
    if (texture_address_mode_v == SDL_TEXTURE_ADDRESS_WRAP) {                           \
        srcy %= src_surface->h;                                                         \
        if (srcy < 0) {                                                                 \
            srcy += (src_surface->h - 1);                                               \
        }                                                                               \
    }

#define TRIANGLE_GET_MAPPED_COLOR                                                      \
    Uint8 r = (Uint8)(((Sint64)w0 * c0.r + (Sint64)w1 * c1.r + (Sint64)w2 * c2.r) / area); \
    Uint8 g = (Uint8)(((Sint64)w0 * c0.g + (Sint64)w1 * c1.g + (Sint64)w2 * c2.g) / area); \
    Uint8 b = (Uint8)(((Sint64)w0 * c0.b + (Sint64)w1 * c1.b + (Sint64)w2 * c2.b) / area); \
    Uint8 a = (Uint8)(((Sint64)w0 * c0.a + (Sint64)w1 * c1.a + (Sint64)w2 * c2.a) / area); \
    Uint32 color = SDL_MapRGBA(format, palette, r, g, b, a);

#define TRIANGLE_GET_COLOR                                                             \
    int r = (int)(((Sint64)w0 * c0.r + (Sint64)w1 * c1.r + (Sint64)w2 * c2.r) / area); \
    int g = (int)(((Sint64)w0 * c0.g + (Sint64)w1 * c1.g + (Sint64)w2 * c2.g) / area); \
    int b = (int)(((Sint64)w0 * c0.b + (Sint64)w1 * c1.b + (Sint64)w2 * c2.b) / area); \
    int a = (int)(((Sint64)w0 * c0.a + (Sint64)w1 * c1.a + (Sint64)w2 * c2.a) / area);

#define TRIANGLE_END_LOOP \
    }                     \
    /* x += 1 */          \
    w0 += d2d1_y;         \
    w1 += d0d2_y;         \
    w2 += d1d0_y;         \
    }                     \
    /* y += 1 */          \
    w0_row += d1d2_x;     \
    w1_row += d2d0_x;     \
    w2_row += d0d1_x;     \
    dst_ptr += dst_pitch; \
    }                     \
    }

bool SDL_SW_FillTriangle(SDL_Surface *dst, SDL_Point *d0, SDL_Point *d1, SDL_Point *d2, SDL_BlendMode blend, SDL_Color c0, SDL_Color c1, SDL_Color c2)
{
    bool result = true;
    int dst_locked = 0;

    SDL_Rect dstrect;

    int dstbpp;
    Uint8 *dst_ptr;
    int dst_pitch;

    Sint64 area;
    int is_clockwise;

    int d2d1_y, d1d2_x, d0d2_y, d2d0_x, d1d0_y, d0d1_x;
    Sint64 w0_row, w1_row, w2_row;
    int bias_w0, bias_w1, bias_w2;

    bool is_uniform;

    SDL_Surface *tmp = NULL;

    if (!SDL_SurfaceValid(dst)) {
        return false;
    }

    area = cross_product(d0, d1, d2->x, d2->y);

    is_uniform = COLOR_EQ(c0, c1) && COLOR_EQ(c1, c2);

    // Flat triangle
    if (area == 0) {
        return true;
    }

    // Lock the destination, if needed
    if (SDL_MUSTLOCK(dst)) {
        if (!SDL_LockSurface(dst)) {
            result = false;
            goto end;
        } else {
            dst_locked = 1;
        }
    }

    bounding_rect_fixedpoint(d0, d1, d2, &dstrect);

    {
        // Clip triangle rect with surface rect
        SDL_Rect rect;
        rect.x = 0;
        rect.y = 0;
        rect.w = dst->w;
        rect.h = dst->h;
        SDL_GetRectIntersection(&dstrect, &rect, &dstrect);
    }

    {
        // Clip triangle with surface clip rect
        SDL_Rect rect;
        SDL_GetSurfaceClipRect(dst, &rect);
        SDL_GetRectIntersection(&dstrect, &rect, &dstrect);
    }

    if (blend != SDL_BLENDMODE_NONE) {
        SDL_PixelFormat format = dst->format;

        // need an alpha format
        if (!SDL_ISPIXELFORMAT_ALPHA(format)) {
            format = SDL_PIXELFORMAT_ARGB8888;
        }

        // Use an intermediate surface
        tmp = SDL_CreateSurface(dstrect.w, dstrect.h, format);
        if (!tmp) {
            result = false;
            goto end;
        }

        if (blend == SDL_BLENDMODE_MOD) {
            Uint32 c = SDL_MapSurfaceRGBA(tmp, 255, 255, 255, 255);
            SDL_FillSurfaceRect(tmp, NULL, c);
        }

        SDL_SetSurfaceBlendMode(tmp, blend);

        dstbpp = tmp->fmt->bytes_per_pixel;
        dst_ptr = (Uint8 *)tmp->pixels;
        dst_pitch = tmp->pitch;

    } else {
        // Write directly to destination surface
        dstbpp = dst->fmt->bytes_per_pixel;
        dst_ptr = (Uint8 *)dst->pixels + dstrect.x * dstbpp + dstrect.y * dst->pitch;
        dst_pitch = dst->pitch;
    }

    is_clockwise = area > 0;
    if (area < 0) {
        area = -area;
    }

    {
        int val;
        PRECOMP(d2d1_y, d1->y - d2->y)
        PRECOMP(d0d2_y, d2->y - d0->y)
        PRECOMP(d1d0_y, d0->y - d1->y)
        PRECOMP(d1d2_x, d2->x - d1->x)
        PRECOMP(d2d0_x, d0->x - d2->x)
        PRECOMP(d0d1_x, d1->x - d0->x)
    }

    // Starting point for rendering, at the middle of a pixel
    {
        SDL_Point p;
        p.x = dstrect.x;
        p.y = dstrect.y;
        trianglepoint_2_fixedpoint(&p);
        p.x += (1 << FP_BITS) / 2;
        p.y += (1 << FP_BITS) / 2;
        w0_row = cross_product(d1, d2, p.x, p.y);
        w1_row = cross_product(d2, d0, p.x, p.y);
        w2_row = cross_product(d0, d1, p.x, p.y);
    }

    // Handle anti-clockwise triangles
    if (!is_clockwise) {
        d2d1_y *= -1;
        d0d2_y *= -1;
        d1d0_y *= -1;
        d1d2_x *= -1;
        d2d0_x *= -1;
        d0d1_x *= -1;
        w0_row *= -1;
        w1_row *= -1;
        w2_row *= -1;
    }

    // Add a bias to respect top-left rasterization rule
    bias_w0 = (is_top_left(d1, d2, is_clockwise) ? 0 : -1);
    bias_w1 = (is_top_left(d2, d0, is_clockwise) ? 0 : -1);
    bias_w2 = (is_top_left(d0, d1, is_clockwise) ? 0 : -1);

    if (is_uniform) {
        Uint32 color;
        if (tmp) {
            color = SDL_MapSurfaceRGBA(tmp, c0.r, c0.g, c0.b, c0.a);
        } else {
            color = SDL_MapSurfaceRGBA(dst, c0.r, c0.g, c0.b, c0.a);
        }

        if (dstbpp == 4) {
            TRIANGLE_BEGIN_LOOP
            {
                *(Uint32 *)dptr = color;
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 3) {
            TRIANGLE_BEGIN_LOOP
            {
                Uint8 *s = (Uint8 *)&color;
                dptr[0] = s[0];
                dptr[1] = s[1];
                dptr[2] = s[2];
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 2) {
            TRIANGLE_BEGIN_LOOP
            {
                *(Uint16 *)dptr = (Uint16)color;
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 1) {
            TRIANGLE_BEGIN_LOOP
            {
                *dptr = (Uint8)color;
            }
            TRIANGLE_END_LOOP
        }
    } else {
        const SDL_PixelFormatDetails *format;
        SDL_Palette *palette;
        if (tmp) {
            format = tmp->fmt;
            palette = tmp->palette;
        } else {
            format = dst->fmt;
            palette = dst->palette;
        }
        if (dstbpp == 4) {
            TRIANGLE_BEGIN_LOOP
            {
                TRIANGLE_GET_MAPPED_COLOR
                *(Uint32 *)dptr = color;
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 3) {
            TRIANGLE_BEGIN_LOOP
            {
                TRIANGLE_GET_MAPPED_COLOR
                Uint8 *s = (Uint8 *)&color;
                dptr[0] = s[0];
                dptr[1] = s[1];
                dptr[2] = s[2];
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 2) {
            TRIANGLE_BEGIN_LOOP
            {
                TRIANGLE_GET_MAPPED_COLOR
                *(Uint16 *)dptr = (Uint16)color;
            }
            TRIANGLE_END_LOOP
        } else if (dstbpp == 1) {
            TRIANGLE_BEGIN_LOOP
            {
                TRIANGLE_GET_MAPPED_COLOR
                *dptr = (Uint8)color;
            }
            TRIANGLE_END_LOOP
        }
    }

    if (tmp) {
        SDL_BlitSurface(tmp, NULL, dst, &dstrect);
        SDL_DestroySurface(tmp);
    }

end:
    if (dst_locked) {
        SDL_UnlockSurface(dst);
    }

    return result;
}

bool SDL_SW_BlitTriangle(
    SDL_Surface *src,
    SDL_Point *s0, SDL_Point *s1, SDL_Point *s2,
    SDL_Surface *dst,
    SDL_Point *d0, SDL_Point *d1, SDL_Point *d2,
    SDL_Color c0, SDL_Color c1, SDL_Color c2,
    SDL_TextureAddressMode texture_address_mode_u,
    SDL_TextureAddressMode texture_address_mode_v)
{
    bool result = true;
    SDL_Surface *src_surface = src;
    int src_locked = 0;
    int dst_locked = 0;

    SDL_BlendMode blend;

    SDL_Rect dstrect;

    SDL_Point s2_x_area;

    int dstbpp;
    Uint8 *dst_ptr;
    int dst_pitch;

    const int *src_ptr;
    int src_pitch;

    Sint64 area, tmp64;
    int is_clockwise;

    int d2d1_y, d1d2_x, d0d2_y, d2d0_x, d1d0_y, d0d1_x;
    int s2s0_x, s2s1_x, s2s0_y, s2s1_y;

    Sint64 w0_row, w1_row, w2_row;
    int bias_w0, bias_w1, bias_w2;

    bool is_uniform;

    bool has_modulation;

    if (!SDL_SurfaceValid(src)) {
        return SDL_InvalidParamError("src");
    }
    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("dst");
    }

    area = cross_product(d0, d1, d2->x, d2->y);

    // Flat triangle
    if (area == 0) {
        return true;
    }

    // Lock the destination, if needed
    if (SDL_MUSTLOCK(dst)) {
        if (!SDL_LockSurface(dst)) {
            result = false;
            goto end;
        } else {
            dst_locked = 1;
        }
    }

    // Lock the source, if needed
    if (SDL_MUSTLOCK(src)) {
        if (!SDL_LockSurface(src)) {
            result = false;
            goto end;
        } else {
            src_locked = 1;
        }
    }

    is_uniform = COLOR_EQ(c0, c1) && COLOR_EQ(c1, c2);

    bounding_rect_fixedpoint(d0, d1, d2, &dstrect);

    SDL_GetSurfaceBlendMode(src, &blend);

    // TRIANGLE_GET_TEXTCOORD interpolates up to the max values included, so reduce by 1
    if (texture_address_mode_u == SDL_TEXTURE_ADDRESS_CLAMP ||
        texture_address_mode_v == SDL_TEXTURE_ADDRESS_CLAMP) {
        SDL_Rect srcrect;
        bounding_rect(s0, s1, s2, &srcrect);
        if (texture_address_mode_u == SDL_TEXTURE_ADDRESS_CLAMP) {
            int maxx = srcrect.x + srcrect.w;
            if (srcrect.w > 0) {
                if (s0->x == maxx) {
                    s0->x--;
                }
                if (s1->x == maxx) {
                    s1->x--;
                }
                if (s2->x == maxx) {
                    s2->x--;
                }
            }
        }
        if (texture_address_mode_v == SDL_TEXTURE_ADDRESS_CLAMP) {
            int maxy = srcrect.y + srcrect.h;
            if (srcrect.h > 0) {
                if (s0->y == maxy) {
                    s0->y--;
                }
                if (s1->y == maxy) {
                    s1->y--;
                }
                if (s2->y == maxy) {
                    s2->y--;
                }
            }
        }
    }

    if (is_uniform) {
        // SDL_GetSurfaceColorMod(src, &r, &g, &b);
        has_modulation = c0.r != 255 || c0.g != 255 || c0.b != 255 || c0.a != 255;
    } else {
        has_modulation = true;
    }

    {
        // Clip triangle with surface clip rect
        SDL_Rect rect;
        SDL_GetSurfaceClipRect(dst, &rect);
        SDL_GetRectIntersection(&dstrect, &rect, &dstrect);
    }

    // Set destination pointer
    dstbpp = dst->fmt->bytes_per_pixel;
    dst_ptr = (Uint8 *)dst->pixels + dstrect.x * dstbpp + dstrect.y * dst->pitch;
    dst_pitch = dst->pitch;

    // Set source pointer
    src_ptr = (const int *)src->pixels;
    src_pitch = src->pitch;

    is_clockwise = area > 0;
    if (area < 0) {
        area = -area;
    }

    {
        int val;
        PRECOMP(d2d1_y, d1->y - d2->y)
        PRECOMP(d0d2_y, d2->y - d0->y)
        PRECOMP(d1d0_y, d0->y - d1->y)
        PRECOMP(d1d2_x, d2->x - d1->x)
        PRECOMP(d2d0_x, d0->x - d2->x)
        PRECOMP(d0d1_x, d1->x - d0->x)
    }

    s2s0_x = s0->x - s2->x;
    s2s1_x = s1->x - s2->x;
    s2s0_y = s0->y - s2->y;
    s2s1_y = s1->y - s2->y;

    // Starting point for rendering, at the middle of a pixel
    {
        SDL_Point p;
        p.x = dstrect.x;
        p.y = dstrect.y;
        trianglepoint_2_fixedpoint(&p);
        p.x += (1 << FP_BITS) / 2;
        p.y += (1 << FP_BITS) / 2;
        w0_row = cross_product(d1, d2, p.x, p.y);
        w1_row = cross_product(d2, d0, p.x, p.y);
        w2_row = cross_product(d0, d1, p.x, p.y);
    }

    // Handle anti-clockwise triangles
    if (!is_clockwise) {
        d2d1_y *= -1;
        d0d2_y *= -1;
        d1d0_y *= -1;
        d1d2_x *= -1;
        d2d0_x *= -1;
        d0d1_x *= -1;
        w0_row *= -1;
        w1_row *= -1;
        w2_row *= -1;
    }

    // Add a bias to respect top-left rasterization rule
    bias_w0 = (is_top_left(d1, d2, is_clockwise) ? 0 : -1);
    bias_w1 = (is_top_left(d2, d0, is_clockwise) ? 0 : -1);
    bias_w2 = (is_top_left(d0, d1, is_clockwise) ? 0 : -1);

    /* precompute constant 's2->x * area' used in TRIANGLE_GET_TEXTCOORD */
    tmp64 = s2->x * area;
    if (tmp64 >= INT_MIN && tmp64 <= INT_MAX) {
        s2_x_area.x = (int)tmp64;
    } else {
        result = SDL_SetError("triangle area overflow");
        goto end;
    }
    tmp64 = s2->y * area;
    if (tmp64 >= INT_MIN && tmp64 <= INT_MAX) {
        s2_x_area.y = (int)tmp64;
    } else {
        result = SDL_SetError("triangle area overflow");
        goto end;
    }

    if (blend != SDL_BLENDMODE_NONE || src->format != dst->format || has_modulation || !is_uniform) {
        // Use SDL_BlitTriangle_Slow

        SDL_BlitInfo *info = &src->map.info;
        SDL_BlitInfo tmp_info;

        SDL_zero(tmp_info);

        tmp_info.src_fmt = src->fmt;
        tmp_info.dst_fmt = dst->fmt;
        tmp_info.flags = info->flags;
        /*
        tmp_info.r = info->r;
        tmp_info.g = info->g;
        tmp_info.b = info->b;
        tmp_info.a = info->a;
        */
        tmp_info.r = c0.r;
        tmp_info.g = c0.g;
        tmp_info.b = c0.b;
        tmp_info.a = c0.a;

        tmp_info.flags &= ~(SDL_COPY_MODULATE_COLOR | SDL_COPY_MODULATE_ALPHA);

        if (c0.r != 255 || c1.r != 255 || c2.r != 255 ||
            c0.g != 255 || c1.g != 255 || c2.g != 255 ||
            c0.b != 255 || c1.b != 255 || c2.b != 255) {
            tmp_info.flags |= SDL_COPY_MODULATE_COLOR;
        }

        if (c0.a != 255 || c1.a != 255 || c2.a != 255) {
            tmp_info.flags |= SDL_COPY_MODULATE_ALPHA;
        }

        tmp_info.colorkey = info->colorkey;

        // src
        tmp_info.src_surface = src_surface;
        tmp_info.src = (Uint8 *)src_ptr;
        tmp_info.src_pitch = src_pitch;

        // dst
        tmp_info.dst = dst_ptr;
        tmp_info.dst_pitch = dst_pitch;

#define CHECK_INT_RANGE(X) \
    if ((X) < INT_MIN || (X) > INT_MAX) { \
        result = SDL_SetError("integer overflow (%s = %" SDL_PRIs64 ")", #X, X); \
        goto end; \
    }
        CHECK_INT_RANGE(area);
        CHECK_INT_RANGE(w0_row);
        CHECK_INT_RANGE(w1_row);
        CHECK_INT_RANGE(w2_row);
        SDL_BlitTriangle_Slow(&tmp_info, s2_x_area, dstrect, (int)area, bias_w0, bias_w1, bias_w2,
                              d2d1_y, d1d2_x, d0d2_y, d2d0_x, d1d0_y, d0d1_x,
                              s2s0_x, s2s1_x, s2s0_y, s2s1_y, (int)w0_row, (int)w1_row, (int)w2_row,
                              c0, c1, c2, is_uniform, texture_address_mode_u, texture_address_mode_v);

        goto end;
    }

    if (dstbpp == 4) {
        TRIANGLE_BEGIN_LOOP
        {
            TRIANGLE_GET_TEXTCOORD
            Uint32 *sptr = (Uint32 *)((Uint8 *)src_ptr + srcy * src_pitch);
            *(Uint32 *)dptr = sptr[srcx];
        }
        TRIANGLE_END_LOOP
    } else if (dstbpp == 3) {
        TRIANGLE_BEGIN_LOOP
        {
            TRIANGLE_GET_TEXTCOORD
            Uint8 *sptr = (Uint8 *)src_ptr + srcy * src_pitch;
            dptr[0] = sptr[3 * srcx];
            dptr[1] = sptr[3 * srcx + 1];
            dptr[2] = sptr[3 * srcx + 2];
        }
        TRIANGLE_END_LOOP
    } else if (dstbpp == 2) {
        TRIANGLE_BEGIN_LOOP
        {
            TRIANGLE_GET_TEXTCOORD
            Uint16 *sptr = (Uint16 *)((Uint8 *)src_ptr + srcy * src_pitch);
            *(Uint16 *)dptr = sptr[srcx];
        }
        TRIANGLE_END_LOOP
    } else if (dstbpp == 1) {
        TRIANGLE_BEGIN_LOOP
        {
            TRIANGLE_GET_TEXTCOORD
            Uint8 *sptr = (Uint8 *)src_ptr + srcy * src_pitch;
            *dptr = sptr[srcx];
        }
        TRIANGLE_END_LOOP
    }

end:
    if (dst_locked) {
        SDL_UnlockSurface(dst);
    }
    if (src_locked) {
        SDL_UnlockSurface(src);
    }

    return result;
}

#define FORMAT_ALPHA                0
#define FORMAT_NO_ALPHA             -1
#define FORMAT_2101010              1
#define FORMAT_HAS_ALPHA(format)    format == 0
#define FORMAT_HAS_NO_ALPHA(format) format < 0
static int detect_format(const SDL_PixelFormatDetails *pf)
{
    if (pf->format == SDL_PIXELFORMAT_ARGB2101010) {
        return FORMAT_2101010;
    } else if (pf->Amask) {
        return FORMAT_ALPHA;
    } else {
        return FORMAT_NO_ALPHA;
    }
}

static void SDL_BlitTriangle_Slow(SDL_BlitInfo *info,
                                  SDL_Point s2_x_area, SDL_Rect dstrect, int area, int bias_w0, int bias_w1, int bias_w2,
                                  int d2d1_y, int d1d2_x, int d0d2_y, int d2d0_x, int d1d0_y, int d0d1_x,
                                  int s2s0_x, int s2s1_x, int s2s0_y, int s2s1_y, int w0_row, int w1_row, int w2_row,
                                  SDL_Color c0, SDL_Color c1, SDL_Color c2, bool is_uniform,
                                  SDL_TextureAddressMode texture_address_mode_u,
                                  SDL_TextureAddressMode texture_address_mode_v)
{
    SDL_Surface *src_surface = info->src_surface;
    const int flags = info->flags;
    Uint32 modulateR = info->r;
    Uint32 modulateG = info->g;
    Uint32 modulateB = info->b;
    Uint32 modulateA = info->a;
    Uint32 srcpixel;
    Uint32 srcR, srcG, srcB, srcA;
    Uint32 dstpixel;
    Uint32 dstR, dstG, dstB, dstA;
    const SDL_PixelFormatDetails *src_fmt = info->src_fmt;
    const SDL_PixelFormatDetails *dst_fmt = info->dst_fmt;
    int srcbpp = src_fmt->bytes_per_pixel;
    int dstbpp = dst_fmt->bytes_per_pixel;
    int srcfmt_val;
    int dstfmt_val;
    Uint32 rgbmask = ~src_fmt->Amask;
    Uint32 ckey = info->colorkey & rgbmask;

    Uint8 *dst_ptr = info->dst;
    int dst_pitch = info->dst_pitch;

    srcfmt_val = detect_format(src_fmt);
    dstfmt_val = detect_format(dst_fmt);

    TRIANGLE_BEGIN_LOOP
    {
        Uint8 *src;
        Uint8 *dst = dptr;
        TRIANGLE_GET_TEXTCOORD
        src = (info->src + (srcy * info->src_pitch) + (srcx * srcbpp));
        if (FORMAT_HAS_ALPHA(srcfmt_val)) {
            DISEMBLE_RGBA(src, srcbpp, src_fmt, srcpixel, srcR, srcG, srcB, srcA);
        } else if (FORMAT_HAS_NO_ALPHA(srcfmt_val)) {
            DISEMBLE_RGB(src, srcbpp, src_fmt, srcpixel, srcR, srcG, srcB);
            srcA = 0xFF;
        } else {
            // SDL_PIXELFORMAT_ARGB2101010
            srcpixel = *((Uint32 *)(src));
            RGBA_FROM_ARGB2101010(srcpixel, srcR, srcG, srcB, srcA);
        }
        if (flags & SDL_COPY_COLORKEY) {
            // srcpixel isn't set for 24 bpp
            if (srcbpp == 3) {
                srcpixel = (srcR << src_fmt->Rshift) |
                           (srcG << src_fmt->Gshift) | (srcB << src_fmt->Bshift);
            }
            if ((srcpixel & rgbmask) == ckey) {
                continue;
            }
        }
        if ((flags & (SDL_COPY_BLEND | SDL_COPY_ADD | SDL_COPY_MOD | SDL_COPY_MUL))) {
            if (FORMAT_HAS_ALPHA(dstfmt_val)) {
                DISEMBLE_RGBA(dst, dstbpp, dst_fmt, dstpixel, dstR, dstG, dstB, dstA);
            } else if (FORMAT_HAS_NO_ALPHA(dstfmt_val)) {
                DISEMBLE_RGB(dst, dstbpp, dst_fmt, dstpixel, dstR, dstG, dstB);
                dstA = 0xFF;
            } else {
                // SDL_PIXELFORMAT_ARGB2101010
                dstpixel = *((Uint32 *) (dst));
                RGBA_FROM_ARGB2101010(dstpixel, dstR, dstG, dstB, dstA);
            }
        } else {
            // don't care
            dstR = dstG = dstB = dstA = 0;
        }

        if (!is_uniform) {
            TRIANGLE_GET_COLOR
            modulateR = r;
            modulateG = g;
            modulateB = b;
            modulateA = a;
        }

        if (flags & SDL_COPY_MODULATE_COLOR) {
            srcR = (srcR * modulateR) / 255;
            srcG = (srcG * modulateG) / 255;
            srcB = (srcB * modulateB) / 255;
        }
        if (flags & SDL_COPY_MODULATE_ALPHA) {
            srcA = (srcA * modulateA) / 255;
        }
        if (flags & (SDL_COPY_BLEND | SDL_COPY_ADD)) {
            // This goes away if we ever use premultiplied alpha
            if (srcA < 255) {
                srcR = (srcR * srcA) / 255;
                srcG = (srcG * srcA) / 255;
                srcB = (srcB * srcA) / 255;
            }
        }
        switch (flags & (SDL_COPY_BLEND | SDL_COPY_ADD | SDL_COPY_MOD | SDL_COPY_MUL)) {
        case 0:
            dstR = srcR;
            dstG = srcG;
            dstB = srcB;
            dstA = srcA;
            break;
        case SDL_COPY_BLEND:
            dstR = srcR + ((255 - srcA) * dstR) / 255;
            dstG = srcG + ((255 - srcA) * dstG) / 255;
            dstB = srcB + ((255 - srcA) * dstB) / 255;
            dstA = srcA + ((255 - srcA) * dstA) / 255;
            break;
        case SDL_COPY_ADD:
            dstR = srcR + dstR;
            if (dstR > 255) {
                dstR = 255;
            }
            dstG = srcG + dstG;
            if (dstG > 255) {
                dstG = 255;
            }
            dstB = srcB + dstB;
            if (dstB > 255) {
                dstB = 255;
            }
            break;
        case SDL_COPY_MOD:
            dstR = (srcR * dstR) / 255;
            dstG = (srcG * dstG) / 255;
            dstB = (srcB * dstB) / 255;
            break;
        case SDL_COPY_MUL:
            dstR = ((srcR * dstR) + (dstR * (255 - srcA))) / 255;
            if (dstR > 255) {
                dstR = 255;
            }
            dstG = ((srcG * dstG) + (dstG * (255 - srcA))) / 255;
            if (dstG > 255) {
                dstG = 255;
            }
            dstB = ((srcB * dstB) + (dstB * (255 - srcA))) / 255;
            if (dstB > 255) {
                dstB = 255;
            }
            break;
        }
        if (FORMAT_HAS_ALPHA(dstfmt_val)) {
            ASSEMBLE_RGBA(dst, dstbpp, dst_fmt, dstR, dstG, dstB, dstA);
        } else if (FORMAT_HAS_NO_ALPHA(dstfmt_val)) {
            ASSEMBLE_RGB(dst, dstbpp, dst_fmt, dstR, dstG, dstB);
        } else {
            // SDL_PIXELFORMAT_ARGB2101010
            Uint32 pixelvalue;
            ARGB2101010_FROM_RGBA(pixelvalue, dstR, dstG, dstB, dstA);
            *(Uint32 *)dst = pixelvalue;
        }
    }
    TRIANGLE_END_LOOP
}

#endif // SDL_VIDEO_RENDER_SW
