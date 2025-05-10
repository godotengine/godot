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
#include "SDL_blendfillrect.h"

static bool SDL_BlendFillRect_RGB555(SDL_Surface *dst, const SDL_Rect *rect,
                                    SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_RGB555);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB555);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        FILLRECT(Uint16, DRAW_SETPIXEL_ADD_RGB555);
        break;
    case SDL_BLENDMODE_MOD:
        FILLRECT(Uint16, DRAW_SETPIXEL_MOD_RGB555);
        break;
    case SDL_BLENDMODE_MUL:
        FILLRECT(Uint16, DRAW_SETPIXEL_MUL_RGB555);
        break;
    default:
        FILLRECT(Uint16, DRAW_SETPIXEL_RGB555);
        break;
    }
    return true;
}

static bool SDL_BlendFillRect_RGB565(SDL_Surface *dst, const SDL_Rect *rect,
                                    SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_RGB565);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB565);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        FILLRECT(Uint16, DRAW_SETPIXEL_ADD_RGB565);
        break;
    case SDL_BLENDMODE_MOD:
        FILLRECT(Uint16, DRAW_SETPIXEL_MOD_RGB565);
        break;
    case SDL_BLENDMODE_MUL:
        FILLRECT(Uint16, DRAW_SETPIXEL_MUL_RGB565);
        break;
    default:
        FILLRECT(Uint16, DRAW_SETPIXEL_RGB565);
        break;
    }
    return true;
}

static bool SDL_BlendFillRect_XRGB8888(SDL_Surface *dst, const SDL_Rect *rect,
                                    SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_XRGB8888);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        FILLRECT(Uint32, DRAW_SETPIXEL_ADD_XRGB8888);
        break;
    case SDL_BLENDMODE_MOD:
        FILLRECT(Uint32, DRAW_SETPIXEL_MOD_XRGB8888);
        break;
    case SDL_BLENDMODE_MUL:
        FILLRECT(Uint32, DRAW_SETPIXEL_MUL_XRGB8888);
        break;
    default:
        FILLRECT(Uint32, DRAW_SETPIXEL_XRGB8888);
        break;
    }
    return true;
}

static bool SDL_BlendFillRect_ARGB8888(SDL_Surface *dst, const SDL_Rect *rect,
                                      SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    unsigned inva = 0xff - a;

    switch (blendMode) {
    case SDL_BLENDMODE_BLEND:
        FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_ARGB8888);
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888);
        break;
    case SDL_BLENDMODE_ADD:
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        FILLRECT(Uint32, DRAW_SETPIXEL_ADD_ARGB8888);
        break;
    case SDL_BLENDMODE_MOD:
        FILLRECT(Uint32, DRAW_SETPIXEL_MOD_ARGB8888);
        break;
    case SDL_BLENDMODE_MUL:
        FILLRECT(Uint32, DRAW_SETPIXEL_MUL_ARGB8888);
        break;
    default:
        FILLRECT(Uint32, DRAW_SETPIXEL_ARGB8888);
        break;
    }
    return true;
}

static bool SDL_BlendFillRect_RGB(SDL_Surface *dst, const SDL_Rect *rect,
                                 SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned inva = 0xff - a;

    switch (fmt->bytes_per_pixel) {
    case 2:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_RGB);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            FILLRECT(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            FILLRECT(Uint16, DRAW_SETPIXEL_ADD_RGB);
            break;
        case SDL_BLENDMODE_MOD:
            FILLRECT(Uint16, DRAW_SETPIXEL_MOD_RGB);
            break;
        case SDL_BLENDMODE_MUL:
            FILLRECT(Uint16, DRAW_SETPIXEL_MUL_RGB);
            break;
        default:
            FILLRECT(Uint16, DRAW_SETPIXEL_RGB);
            break;
        }
        return true;
    case 4:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_RGB);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGB);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            FILLRECT(Uint32, DRAW_SETPIXEL_ADD_RGB);
            break;
        case SDL_BLENDMODE_MOD:
            FILLRECT(Uint32, DRAW_SETPIXEL_MOD_RGB);
            break;
        case SDL_BLENDMODE_MUL:
            FILLRECT(Uint32, DRAW_SETPIXEL_MUL_RGB);
            break;
        default:
            FILLRECT(Uint32, DRAW_SETPIXEL_RGB);
            break;
        }
        return true;
    default:
        return SDL_Unsupported();
    }
}

static bool SDL_BlendFillRect_RGBA(SDL_Surface *dst, const SDL_Rect *rect,
                                  SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned inva = 0xff - a;

    switch (fmt->bytes_per_pixel) {
    case 4:
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_RGBA);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            FILLRECT(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGBA);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            FILLRECT(Uint32, DRAW_SETPIXEL_ADD_RGBA);
            break;
        case SDL_BLENDMODE_MOD:
            FILLRECT(Uint32, DRAW_SETPIXEL_MOD_RGBA);
            break;
        case SDL_BLENDMODE_MUL:
            FILLRECT(Uint32, DRAW_SETPIXEL_MUL_RGBA);
            break;
        default:
            FILLRECT(Uint32, DRAW_SETPIXEL_RGBA);
            break;
        }
        return true;
    default:
        return SDL_Unsupported();
    }
}

bool SDL_BlendFillRect(SDL_Surface *dst, const SDL_Rect *rect, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    SDL_Rect clipped;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_BlendFillRect(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (SDL_BITSPERPIXEL(dst->format) < 8) {
        return SDL_SetError("SDL_BlendFillRect(): Unsupported surface format");
    }

    // If 'rect' == NULL, then fill the whole surface
    if (rect) {
        // Perform clipping
        if (!SDL_GetRectIntersection(rect, &dst->clip_rect, &clipped)) {
            return true;
        }
        rect = &clipped;
    } else {
        rect = &dst->clip_rect;
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
            return SDL_BlendFillRect_RGB555(dst, rect, blendMode, r, g, b, a);
        }
        break;
    case 16:
        switch (dst->fmt->Rmask) {
        case 0xF800:
            return SDL_BlendFillRect_RGB565(dst, rect, blendMode, r, g, b, a);
        }
        break;
    case 32:
        switch (dst->fmt->Rmask) {
        case 0x00FF0000:
            if (!dst->fmt->Amask) {
                return SDL_BlendFillRect_XRGB8888(dst, rect, blendMode, r, g, b, a);
            } else {
                return SDL_BlendFillRect_ARGB8888(dst, rect, blendMode, r, g, b, a);
            }
            // break; -Wunreachable-code-break
        }
        break;
    default:
        break;
    }

    if (!dst->fmt->Amask) {
        return SDL_BlendFillRect_RGB(dst, rect, blendMode, r, g, b, a);
    } else {
        return SDL_BlendFillRect_RGBA(dst, rect, blendMode, r, g, b, a);
    }
}

bool SDL_BlendFillRects(SDL_Surface *dst, const SDL_Rect *rects, int count, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    SDL_Rect rect;
    int i;
    bool (*func)(SDL_Surface * dst, const SDL_Rect *rect, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a) = NULL;
    bool result = true;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_BlendFillRects(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (dst->fmt->bits_per_pixel < 8) {
        return SDL_SetError("SDL_BlendFillRects(): Unsupported surface format");
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
            func = SDL_BlendFillRect_RGB555;
        }
        break;
    case 16:
        switch (dst->fmt->Rmask) {
        case 0xF800:
            func = SDL_BlendFillRect_RGB565;
        }
        break;
    case 32:
        switch (dst->fmt->Rmask) {
        case 0x00FF0000:
            if (!dst->fmt->Amask) {
                func = SDL_BlendFillRect_XRGB8888;
            } else {
                func = SDL_BlendFillRect_ARGB8888;
            }
            break;
        }
        break;
    default:
        break;
    }

    if (!func) {
        if (!dst->fmt->Amask) {
            func = SDL_BlendFillRect_RGB;
        } else {
            func = SDL_BlendFillRect_RGBA;
        }
    }

    for (i = 0; i < count; ++i) {
        // Perform clipping
        if (!SDL_GetRectIntersection(&rects[i], &dst->clip_rect, &rect)) {
            continue;
        }
        result = func(dst, &rect, blendMode, r, g, b, a);
    }
    return result;
}

#endif // SDL_VIDEO_RENDER_SW
