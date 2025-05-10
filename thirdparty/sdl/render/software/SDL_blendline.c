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
#include "SDL_blendline.h"
#include "SDL_blendpoint.h"

static void SDL_BlendLine_RGB2(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                               SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                               bool draw_end)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint16, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint16, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            HLINE(Uint16, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint16, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint16, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            VLINE(Uint16, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint16, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint16, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            DLINE(Uint16, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_BLEND_RGB, DRAW_SETPIXELXY2_BLEND_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_BLEND_CLAMPED_RGB, DRAW_SETPIXELXY2_BLEND_CLAMPED_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_ADD_RGB, DRAW_SETPIXELXY2_ADD_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_MOD_RGB, DRAW_SETPIXELXY2_MOD_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_MUL_RGB, DRAW_SETPIXELXY2_MUL_RGB,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY2_RGB, DRAW_SETPIXELXY2_BLEND_RGB,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_RGB555(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                                 SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                                 bool draw_end)
{
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_ADD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint16, DRAW_SETPIXEL_MOD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint16, DRAW_SETPIXEL_MUL_RGB555, draw_end);
            break;
        default:
            HLINE(Uint16, DRAW_SETPIXEL_RGB555, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_ADD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint16, DRAW_SETPIXEL_MOD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint16, DRAW_SETPIXEL_MUL_RGB555, draw_end);
            break;
        default:
            VLINE(Uint16, DRAW_SETPIXEL_RGB555, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_ADD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint16, DRAW_SETPIXEL_MOD_RGB555, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint16, DRAW_SETPIXEL_MUL_RGB555, draw_end);
            break;
        default:
            DLINE(Uint16, DRAW_SETPIXEL_RGB555, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_RGB555, DRAW_SETPIXELXY_BLEND_RGB555,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_CLAMPED_RGB555, DRAW_SETPIXELXY_BLEND_CLAMPED_RGB555,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_ADD_RGB555, DRAW_SETPIXELXY_ADD_RGB555,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MOD_RGB555, DRAW_SETPIXELXY_MOD_RGB555,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MUL_RGB555, DRAW_SETPIXELXY_MUL_RGB555,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_RGB555, DRAW_SETPIXELXY_BLEND_RGB555,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_RGB565(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                                 SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                                 bool draw_end)
{
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint16, DRAW_SETPIXEL_ADD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint16, DRAW_SETPIXEL_MOD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint16, DRAW_SETPIXEL_MUL_RGB565, draw_end);
            break;
        default:
            HLINE(Uint16, DRAW_SETPIXEL_RGB565, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint16, DRAW_SETPIXEL_ADD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint16, DRAW_SETPIXEL_MOD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint16, DRAW_SETPIXEL_MUL_RGB565, draw_end);
            break;
        default:
            VLINE(Uint16, DRAW_SETPIXEL_RGB565, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_BLEND_CLAMPED_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint16, DRAW_SETPIXEL_ADD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint16, DRAW_SETPIXEL_MOD_RGB565, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint16, DRAW_SETPIXEL_MUL_RGB565, draw_end);
            break;
        default:
            DLINE(Uint16, DRAW_SETPIXEL_RGB565, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_RGB565, DRAW_SETPIXELXY_BLEND_RGB565,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_CLAMPED_RGB565, DRAW_SETPIXELXY_BLEND_CLAMPED_RGB565,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_ADD_RGB565, DRAW_SETPIXELXY_ADD_RGB565,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MOD_RGB565, DRAW_SETPIXELXY_MOD_RGB565,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MUL_RGB565, DRAW_SETPIXELXY_MUL_RGB565,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_RGB565, DRAW_SETPIXELXY_BLEND_RGB565,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_RGB4(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                               SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                               bool draw_end)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint32, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint32, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            HLINE(Uint32, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint32, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint32, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            VLINE(Uint32, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_RGB, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGB, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_ADD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint32, DRAW_SETPIXEL_MOD_RGB, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint32, DRAW_SETPIXEL_MUL_RGB, draw_end);
            break;
        default:
            DLINE(Uint32, DRAW_SETPIXEL_RGB, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_BLEND_RGB, DRAW_SETPIXELXY4_BLEND_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_BLEND_CLAMPED_RGB, DRAW_SETPIXELXY4_BLEND_CLAMPED_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_ADD_RGB, DRAW_SETPIXELXY4_ADD_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_MOD_RGB, DRAW_SETPIXELXY4_MOD_RGB,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_MUL_RGB, DRAW_SETPIXELXY4_MUL_RGB,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_RGB, DRAW_SETPIXELXY4_BLEND_RGB,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_RGBA4(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                                SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                                bool draw_end)
{
    const SDL_PixelFormatDetails *fmt = dst->fmt;
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_ADD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint32, DRAW_SETPIXEL_MOD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint32, DRAW_SETPIXEL_MUL_RGBA, draw_end);
            break;
        default:
            HLINE(Uint32, DRAW_SETPIXEL_RGBA, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_ADD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint32, DRAW_SETPIXEL_MOD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint32, DRAW_SETPIXEL_MUL_RGBA, draw_end);
            break;
        default:
            VLINE(Uint32, DRAW_SETPIXEL_RGBA, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_ADD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint32, DRAW_SETPIXEL_MOD_RGBA, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint32, DRAW_SETPIXEL_MUL_RGBA, draw_end);
            break;
        default:
            DLINE(Uint32, DRAW_SETPIXEL_RGBA, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_BLEND_RGBA, DRAW_SETPIXELXY4_BLEND_RGBA,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_BLEND_CLAMPED_RGBA, DRAW_SETPIXELXY4_BLEND_CLAMPED_RGBA,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_ADD_RGBA, DRAW_SETPIXELXY4_ADD_RGBA,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_MOD_RGBA, DRAW_SETPIXELXY4_MOD_RGBA,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_MUL_RGBA, DRAW_SETPIXELXY4_MUL_RGBA,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY4_RGBA, DRAW_SETPIXELXY4_BLEND_RGBA,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_XRGB8888(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                                 SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                                 bool draw_end)
{
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_ADD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint32, DRAW_SETPIXEL_MOD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint32, DRAW_SETPIXEL_MUL_XRGB8888, draw_end);
            break;
        default:
            HLINE(Uint32, DRAW_SETPIXEL_XRGB8888, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_ADD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint32, DRAW_SETPIXEL_MOD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint32, DRAW_SETPIXEL_MUL_XRGB8888, draw_end);
            break;
        default:
            VLINE(Uint32, DRAW_SETPIXEL_XRGB8888, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_ADD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint32, DRAW_SETPIXEL_MOD_XRGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint32, DRAW_SETPIXEL_MUL_XRGB8888, draw_end);
            break;
        default:
            DLINE(Uint32, DRAW_SETPIXEL_XRGB8888, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_XRGB8888, DRAW_SETPIXELXY_BLEND_XRGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_CLAMPED_XRGB8888, DRAW_SETPIXELXY_BLEND_CLAMPED_XRGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_ADD_XRGB8888, DRAW_SETPIXELXY_ADD_XRGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MOD_XRGB8888, DRAW_SETPIXELXY_MOD_XRGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MUL_XRGB8888, DRAW_SETPIXELXY_MUL_XRGB8888,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_XRGB8888, DRAW_SETPIXELXY_BLEND_XRGB8888,
                   draw_end);
            break;
        }
    }
}

static void SDL_BlendLine_ARGB8888(SDL_Surface *dst, int x1, int y1, int x2, int y2,
                                   SDL_BlendMode blendMode, Uint8 _r, Uint8 _g, Uint8 _b, Uint8 _a,
                                   bool draw_end)
{
    unsigned r, g, b, a, inva;

    if (blendMode == SDL_BLENDMODE_BLEND || blendMode == SDL_BLENDMODE_ADD) {
        r = DRAW_MUL(_r, _a);
        g = DRAW_MUL(_g, _a);
        b = DRAW_MUL(_b, _a);
        a = _a;
    } else {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }
    inva = (a ^ 0xff);

    if (y1 == y2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            HLINE(Uint32, DRAW_SETPIXEL_ADD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            HLINE(Uint32, DRAW_SETPIXEL_MOD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            HLINE(Uint32, DRAW_SETPIXEL_MUL_ARGB8888, draw_end);
            break;
        default:
            HLINE(Uint32, DRAW_SETPIXEL_ARGB8888, draw_end);
            break;
        }
    } else if (x1 == x2) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            VLINE(Uint32, DRAW_SETPIXEL_ADD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            VLINE(Uint32, DRAW_SETPIXEL_MOD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            VLINE(Uint32, DRAW_SETPIXEL_MUL_ARGB8888, draw_end);
            break;
        default:
            VLINE(Uint32, DRAW_SETPIXEL_ARGB8888, draw_end);
            break;
        }
    } else if (ABS(x1 - x2) == ABS(y1 - y2)) {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            DLINE(Uint32, DRAW_SETPIXEL_ADD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            DLINE(Uint32, DRAW_SETPIXEL_MOD_ARGB8888, draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            DLINE(Uint32, DRAW_SETPIXEL_MUL_ARGB8888, draw_end);
            break;
        default:
            DLINE(Uint32, DRAW_SETPIXEL_ARGB8888, draw_end);
            break;
        }
    } else {
        switch (blendMode) {
        case SDL_BLENDMODE_BLEND:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_ARGB8888, DRAW_SETPIXELXY_BLEND_ARGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_BLEND_CLAMPED_ARGB8888, DRAW_SETPIXELXY_BLEND_CLAMPED_ARGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_ADD:
        case SDL_BLENDMODE_ADD_PREMULTIPLIED:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_ADD_ARGB8888, DRAW_SETPIXELXY_ADD_ARGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_MOD:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MOD_ARGB8888, DRAW_SETPIXELXY_MOD_ARGB8888,
                   draw_end);
            break;
        case SDL_BLENDMODE_MUL:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_MUL_ARGB8888, DRAW_SETPIXELXY_MUL_ARGB8888,
                   draw_end);
            break;
        default:
            AALINE(x1, y1, x2, y2,
                   DRAW_SETPIXELXY_ARGB8888, DRAW_SETPIXELXY_BLEND_ARGB8888,
                   draw_end);
            break;
        }
    }
}

typedef void (*BlendLineFunc)(SDL_Surface *dst,
                              int x1, int y1, int x2, int y2,
                              SDL_BlendMode blendMode,
                              Uint8 r, Uint8 g, Uint8 b, Uint8 a,
                              bool draw_end);

static BlendLineFunc SDL_CalculateBlendLineFunc(const SDL_PixelFormatDetails *fmt)
{
    switch (fmt->bytes_per_pixel) {
    case 2:
        if (fmt->Rmask == 0x7C00) {
            return SDL_BlendLine_RGB555;
        } else if (fmt->Rmask == 0xF800) {
            return SDL_BlendLine_RGB565;
        } else {
            return SDL_BlendLine_RGB2;
        }
        // break; -Wunreachable-code-break
    case 4:
        if (fmt->Rmask == 0x00FF0000) {
            if (fmt->Amask) {
                return SDL_BlendLine_ARGB8888;
            } else {
                return SDL_BlendLine_XRGB8888;
            }
        } else {
            if (fmt->Amask) {
                return SDL_BlendLine_RGBA4;
            } else {
                return SDL_BlendLine_RGB4;
            }
        }
    }
    return NULL;
}

bool SDL_BlendLine(SDL_Surface *dst, int x1, int y1, int x2, int y2, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    BlendLineFunc func;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("SDL_BlendLine(): dst");
    }

    func = SDL_CalculateBlendLineFunc(dst->fmt);
    if (!func) {
        return SDL_SetError("SDL_BlendLine(): Unsupported surface format");
    }

    // Perform clipping
    // FIXME: We don't actually want to clip, as it may change line slope
    if (!SDL_GetRectAndLineIntersection(&dst->clip_rect, &x1, &y1, &x2, &y2)) {
        return true;
    }

    func(dst, x1, y1, x2, y2, blendMode, r, g, b, a, true);
    return true;
}

bool SDL_BlendLines(SDL_Surface *dst, const SDL_Point *points, int count, SDL_BlendMode blendMode, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    int i;
    int x1, y1;
    int x2, y2;
    bool draw_end;
    BlendLineFunc func;

    if (!SDL_SurfaceValid(dst)) {
        return SDL_SetError("SDL_BlendLines(): Passed NULL destination surface");
    }

    func = SDL_CalculateBlendLineFunc(dst->fmt);
    if (!func) {
        return SDL_SetError("SDL_BlendLines(): Unsupported surface format");
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

        // Draw the end if it was clipped
        draw_end = (x2 != points[i].x || y2 != points[i].y);

        func(dst, x1, y1, x2, y2, blendMode, r, g, b, a, draw_end);
    }
    if (points[0].x != points[count - 1].x || points[0].y != points[count - 1].y) {
        SDL_BlendPoint(dst, points[count - 1].x, points[count - 1].y,
                       blendMode, r, g, b, a);
    }
    return true;
}

#endif // SDL_VIDEO_RENDER_SW
