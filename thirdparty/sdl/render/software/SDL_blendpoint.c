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

#include "SDL_draw.h"
#include "SDL_blendpoint.h"

static bool SDL_BlendPoint_RGB555(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r,
                                 Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        DRAW_SETPIXELXY_BLEND_RGB555(x, y);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        DRAW_SETPIXELXY_BLEND_CLAMPED_RGB555(x, y);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        DRAW_SETPIXELXY_ADD_RGB555(x, y);
        break;
    case SDL_BLENDMODE_MOD:
        DRAW_SETPIXELXY_MOD_RGB555(x, y);
        break;
    case SDL_BLENDMODE_MUL:
        DRAW_SETPIXELXY_MUL_RGB555(x, y);
        break;
    default:
        DRAW_SETPIXELXY_RGB555(x, y);
        break;
    }
    return true;
}

static bool SDL_BlendPoint_RGB565(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r,
                                 Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        DRAW_SETPIXELXY_BLEND_RGB565(x, y);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        DRAW_SETPIXELXY_BLEND_CLAMPED_RGB565(x, y);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        DRAW_SETPIXELXY_ADD_RGB565(x, y);
        break;
    case SDL_BLENDMODE_MOD:
        DRAW_SETPIXELXY_MOD_RGB565(x, y);
        break;
    case SDL_BLENDMODE_MUL:
        DRAW_SETPIXELXY_MUL_RGB565(x, y);
        break;
    default:
        DRAW_SETPIXELXY_RGB565(x, y);
        break;
    }
    return true;
}

static bool SDL_BlendPoint_XRGB8888(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r,
                                 Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        DRAW_SETPIXELXY_BLEND_XRGB8888(x, y);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        DRAW_SETPIXELXY_BLEND_CLAMPED_XRGB8888(x, y);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        DRAW_SETPIXELXY_ADD_XRGB8888(x, y);
        break;
    case SDL_BLENDMODE_MOD:
        DRAW_SETPIXELXY_MOD_XRGB8888(x, y);
        break;
    case SDL_BLENDMODE_MUL:
        DRAW_SETPIXELXY_MUL_XRGB8888(x, y);
        break;
    default:
        DRAW_SETPIXELXY_XRGB8888(x, y);
        break;
    }
    return true;
}

static bool SDL_BlendPoint_ARGB8888(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode,
                                   Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        DRAW_SETPIXELXY_BLEND_ARGB8888(x, y);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        DRAW_SETPIXELXY_BLEND_CLAMPED_ARGB8888(x, y);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        DRAW_SETPIXELXY_ADD_ARGB8888(x, y);
        break;
    case SDL_BLENDMODE_MOD:
        DRAW_SETPIXELXY_MOD_ARGB8888(x, y);
        break;
    case SDL_BLENDMODE_MUL:
        DRAW_SETPIXELXY_MUL_ARGB8888(x, y);
        break;
    default:
        DRAW_SETPIXELXY_ARGB8888(x, y);
        break;
    }
    return true;
}

static bool SDL_BlendPoint_RGB(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r,
                              Uint8 g, Uint8 b, Uint8 a)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned inva = 0xff - a;

    switch (fmt->bytes_per_pixel) {
    case 2:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DRAW_SETPIXELXY2_BLEND_RGB(x, y);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DRAW_SETPIXELXY2_BLEND_CLAMPED_RGB(x, y);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DRAW_SETPIXELXY2_ADD_RGB(x, y);
            break;
        case SDL_BLENDMODE_MOD:
            DRAW_SETPIXELXY2_MOD_RGB(x, y);
            break;
        case SDL_BLENDMODE_MUL:
            DRAW_SETPIXELXY2_MUL_RGB(x, y);
            break;
        default:
            DRAW_SETPIXELXY2_RGB(x, y);
            break;
        }
        return true;
    case 4:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DRAW_SETPIXELXY4_BLEND_RGB(x, y);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DRAW_SETPIXELXY4_BLEND_CLAMPED_RGB(x, y);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DRAW_SETPIXELXY4_ADD_RGB(x, y);
            break;
        case SDL_BLENDMODE_MOD:
            DRAW_SETPIXELXY4_MOD_RGB(x, y);
            break;
        case SDL_BLENDMODE_MUL:
            DRAW_SETPIXELXY4_MUL_RGB(x, y);
            break;
        default:
            DRAW_SETPIXELXY4_RGB(x, y);
            break;
        }
        return true;
    default:
        return SDL_Unsupported();
    }
}

static bool SDL_BlendPoint_RGBA(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r,
                               Uint8 g, Uint8 b, Uint8 a)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned inva = 0xff - a;

    switch (fmt->bytes_per_pixel) {
    case 4:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DRAW_SETPIXELXY4_BLEND_RGBA(x, y);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DRAW_SETPIXELXY4_BLEND_CLAMPED_RGBA(x, y);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DRAW_SETPIXELXY4_ADD_RGBA(x, y);
            break;
        case SDL_BLENDMODE_MOD:
            DRAW_SETPIXELXY4_MOD_RGBA(x, y);
            break;
        case SDL_BLENDMODE_MUL:
            DRAW_SETPIXELXY4_MUL_RGBA(x, y);
            break;
        default:
            DRAW_SETPIXELXY4_RGBA(x, y);
            break;
        }
        return true;
    default:
        return SDL_Unsupported();
    }
}

bool SDL_BlendPoint(SDL_Surface *dst, int x, int y, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_BlendPoint(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (SDL_BITSPERPIXEL(dst->format) < 8) {
        return SDL_SetError("SDL_BlendPoint(): Unsupported surface format");
    }

    // Perform clipping
    if (x < dst->clip_rect.x || y < dst->clip_rect.y ||
        x >= (dst->clip_rect.x + dst->clip_rect.w) ||
        y >= (dst->clip_rect.y + dst->clip_rect.h)) {
        return true;
    }

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(r, a);
        g = DRAW_MUL(g, a);
        b = DRAW_MUL(b, a);
    }

    switch (dst->fmt->bits_per_pixel) {
    case 15:
        switch (dst->fmt->Rmask) {
        case 0x7C00:
            return SDL_BlendPoint_RGB555(dst, x, y, blendMode, r, g, b, a);
        }
        break;
    case 16:
        switch (dst->fmt->Rmask) {
        case 0xF800:
            return SDL_BlendPoint_RGB565(dst, x, y, blendMode, r, g, b, a);
        }
        break;
    case 32:
        switch (dst->fmt->Rmask) {
        case 0x00FF0000:
            if (!dst->fmt->Amask) {
                return SDL_BlendPoint_XRGB8888(dst, x, y, blendMode, r, g, b, a);
            } else {
                return SDL_BlendPoint_ARGB8888(dst, x, y, blendMode, r, g, b, a);
            }
            // break; -Wunreachable-code-break
        }
        break;
    default:
        break;
    }

    if (!dst->fmt->Amask) {
        return SDL_BlendPoint_RGB(dst, x, y, blendMode, r, g, b, a);
    } else {
        return SDL_BlendPoint_RGBA(dst, x, y, blendMode, r, g, b, a);
    }
}

bool SDL_BlendPoints(SDL_Surface *dst, const SDL_Point *points, int count, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    int minx, miny;
    int maxx, maxy;
    int i;
    int x, y;
    bool (*func)(SDL_Surface * dst, int x, int y, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a) = NULL;
    bool result = true;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_BlendPoints(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (dst->fmt->bits_per_pixel < 8) {
        return SDL_SetError("SDL_BlendPoints(): Unsupported surface format");
    }

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(r, a);
        g = DRAW_MUL(g, a);
        b = DRAW_MUL(b, a);
    }

    // FIXME: Does this function pointer slow things down significantly?
    switch (dst->fmt->bits_per_pixel) {
    case 15:
        switch (dst->fmt->Rmask) {
        case 0x7C00:
            func = SDL_BlendPoint_RGB555;
            break;
        }
        break;
    case 16:
        switch (dst->fmt->Rmask) {
        case 0xF800:
            func = SDL_BlendPoint_RGB565;
            break;
        }
        break;
    case 32:
        switch (dst->fmt->Rmask) {
        case 0x00FF0000:
            if (!dst->fmt->Amask) {
                func = SDL_BlendPoint_XRGB8888;
            } else {
                func = SDL_BlendPoint_ARGB8888;
            }
            break;
        }
        break;
    default:
        break;
    }

    if (!func) {
        if (!dst->fmt->Amask) {
            func = SDL_BlendPoint_RGB;
        } else {
            func = SDL_BlendPoint_RGBA;
        }
    }

    minx = dst->clip_rect.x;
    maxx = dst->clip_rect.x + dst->clip_rect.w - 1;
    miny = dst->clip_rect.y;
    maxy = dst->clip_rect.y + dst->clip_rect.h - 1;

    for (i = 0; i < count; ++i) {
        x = points[i].x;
        y = points[i].y;

        if (x < minx || x > maxx || y < miny || y > maxy) {
            continue;
        }
        result = func(dst, x, y, blendMode, r, g, b, a);
    }
    return result;
}

#endif // SDL_VIDEO_RENDER_SW
