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
#include "SDL_drawline.h"
#include "SDL_drawpoint.h"

static void SDL_DrawLine1(SDL_Surface *dst, int x1, int y1, int x2, int y2, Uint32 color,
                          bool draw_end)
{
    if (y1 == y2) {
        int length;
        int pitch = (dst->pitch / dst->fmt->bytes_per_pixel);
        Uint8 *pixels;
        if (x1 <= x2) {
            pixels = (Uint8 *)dst->pixels + y1 * pitch + x1;
            length = draw_end ? (x2 - x1 + 1) : (x2 - x1);
        } else {
            pixels = (Uint8 *)dst->pixels + y1 * pitch + x2;
            if (!draw_end) {
                ++pixels;
            }
            length = draw_end ? (x1 - x2 + 1) : (x1 - x2);
        }
        SDL_memset(pixels, color, length);
    } else if (x1 == x2) {
        VLINE(Uint8, DRAW_FASTSETPIXEL1, draw_end);
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        DLINE(Uint8, DRAW_FASTSETPIXEL1, draw_end);
    } else {
        BLINE(x1, y1, x2, y2, DRAW_FASTSETPIXELXY1, draw_end);
    }
}

static void SDL_DrawLine2(SDL_Surface *dst, int x1, int y1, int x2, int y2, Uint32 color,
                          bool draw_end)
{
    if (y1 == y2) {
        HLINE(Uint16, DRAW_FASTSETPIXEL2, draw_end);
    } else if (x1 == x2) {
        VLINE(Uint16, DRAW_FASTSETPIXEL2, draw_end);
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        DLINE(Uint16, DRAW_FASTSETPIXEL2, draw_end);
    } else {
        Uint8 _r, _g, _b, _a;
        const SDL_PixelFormatDetails *fmt = dst->fmt;
        SDL_GetRGBA(color, fmt, dst->palette, &_r, &_g, &_b, &_a);
        if (fmt->Rmask == 0x7C00) {
            AALINE(x1, y1, x2, y2,
                   DRAW_FASTSETPIXELXY2, DRAW_SETPIXELXY_BLEND_RGB555,
                   draw_end);
        } else if (fmt->Rmask == 0xF800) {
            AALINE(x1, y1, x2, y2,
                   DRAW_FASTSETPIXELXY2, DRAW_SETPIXELXY_BLEND_RGB565,
                   draw_end);
        } else {
            AALINE(x1, y1, x2, y2,
                   DRAW_FASTSETPIXELXY2, DRAW_SETPIXELXY2_BLEND_RGB,
                   draw_end);
        }
    }
}

static void SDL_DrawLine4(SDL_Surface *dst, int x1, int y1, int x2, int y2, Uint32 color,
                          bool draw_end)
{
    if (y1 == y2) {
        HLINE(Uint32, DRAW_FASTSETPIXEL4, draw_end);
    } else if (x1 == x2) {
        VLINE(Uint32, DRAW_FASTSETPIXEL4, draw_end);
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        DLINE(Uint32, DRAW_FASTSETPIXEL4, draw_end);
    } else {
        Uint8 _r, _g, _b, _a;
        const SDL_PixelFormatDetails *fmt = dst->fmt;
        SDL_GetRGBA(color, fmt, dst->palette, &_r, &_g, &_b, &_a);
        if (fmt->Rmask == 0x00FF0000) {
            if (!fmt->Amask) {
                AALINE(x1, y1, x2, y2,
                       DRAW_FASTSETPIXELXY4, DRAW_SETPIXELXY_BLEND_XRGB8888,
                       draw_end);
            } else {
                AALINE(x1, y1, x2, y2,
                       DRAW_FASTSETPIXELXY4, DRAW_SETPIXELXY_BLEND_ARGB8888,
                       draw_end);
            }
        } else {
            AALINE(x1, y1, x2, y2,
                   DRAW_FASTSETPIXELXY4, DRAW_SETPIXELXY4_BLEND_RGB,
                   draw_end);
        }
    }
}

typedef void (*DrawLineFunc)(SDL_Surface *dst,
                             int x1, int y1, int x2, int y2,
                             Uint32 color, bool draw_end);

static DrawLineFunc SDL_CalculateDrawLineFunc(const SDL_PixelFormatDetails *fmt)
{
    switch (fmt->bytes_per_pixel) {
    case 1:
        if (fmt->bits_per_pixel < 8) {
            break;
        }
        return SDL_DrawLine1;
    case 2:
        return SDL_DrawLine2;
    case 4:
        return SDL_DrawLine4;
    }
    return NULL;
}

bool SDL_DrawLine(SDL_Surface *dst, int x1, int y1, int x2, int y2, Uint32 color)
{
    DrawLineFunc func;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_DrawLine(): dst");
    }

    func = SDL_CalculateDrawLineFunc(dst->fmt);
    if (!func) {
        return SDL_SetError("SDL_DrawLine(): Unsupported surface format");
    }

    // Perform clipping
    // FIXME: We don't actually want to clip, as it may change line slope
    if (!SDL_GetRectAndLineIntersection(&dst->clip_rect, &x1, &y1, &x2, &y2)) {
        return true;
    }

    func(dst, x1, y1, x2, y2, color, true);
    return true;
}

bool SDL_DrawLines(SDL_Surface *dst, const SDL_Point *points, int count, Uint32 color)
{
    int i;
    int x1, y1;
    int x2, y2;
    bool draw_end;
    DrawLineFunc func;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_DrawLines(): dst");
    }

    func = SDL_CalculateDrawLineFunc(dst->fmt);
    if (!func) {
        return SDL_SetError("SDL_DrawLines(): Unsupported surface format");
    }

    for (i = 1; i < count; ++i) {
        x1 = points[i - 1].x;
        y1 = points[i - 1].y;
        x2 = points[i].x;
        y2 = points[i].y;

        // Perform clipping
        // FIXME: We don't actually want to clip, as it may change line slope
        if (!SDL_GetRectAndLineIntersection(&dst->clip_rect, &x1, &y1, &x2, &y2)) {
            continue;
        }

        // Draw the end if the whole line is a single point or it was clipped
        draw_end = ((x1 == x2) && (y1 == y2)) || (x2 != points[i].x || y2 != points[i].y);

        func(dst, x1, y1, x2, y2, color, draw_end);
    }
    if (points[0].x != points[count - 1].x || points[0].y != points[count - 1].y) {
        SDL_DrawPoint(dst, points[count - 1].x, points[count - 1].y, color);
    }
    return true;
}

#endif // SDL_VIDEO_RENDER_SW
