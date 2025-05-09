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
#include "SDL_drawpoint.h"

bool SDL_DrawPoint(SDL_Surface *dst, int x, int y, Uint32 color)
{
    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_DrawPoint(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (dst->fmt->bits_per_pixel < 8) {
        return SDL_SetError("SDL_DrawPoint(): Unsupported surface format");
    }

    // Perform clipping
    if (x < dst->clip_rect.x || y < dst->clip_rect.y ||
        x >= (dst->clip_rect.x + dst->clip_rect.w) ||
        y >= (dst->clip_rect.y + dst->clip_rect.h)) {
        return true;
    }

    switch (dst->fmt->bytes_per_pixel) {
    case 1:
        DRAW_FASTSETPIXELXY1(x, y);
        break;
    case 2:
        DRAW_FASTSETPIXELXY2(x, y);
        break;
    case 3:
        return SDL_Unsupported();
    case 4:
        DRAW_FASTSETPIXELXY4(x, y);
        break;
    }
    return true;
}

bool SDL_DrawPoints(SDL_Surface *dst, const SDL_Point *points, int count, Uint32 color)
{
    int minx, miny;
    int maxx, maxy;
    int i;
    int x, y;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_DrawPoints(): dst");
    }

    // This function doesn't work on surfaces < 8 bpp
    if (dst->fmt->bits_per_pixel < 8) {
        return SDL_SetError("SDL_DrawPoints(): Unsupported surface format");
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

        switch (dst->fmt->bytes_per_pixel) {
        case 1:
            DRAW_FASTSETPIXELXY1(x, y);
            break;
        case 2:
            DRAW_FASTSETPIXELXY2(x, y);
            break;
        case 3:
            return SDL_Unsupported();
        case 4:
            DRAW_FASTSETPIXELXY4(x, y);
            break;
        }
    }
    return true;
}

#endif // SDL_VIDEO_RENDER_SW
