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

#include "../../video/SDL_surface_c.h"

/* This code assumes that r, g, b, a are the source color,
 * and in the blend and add case, the RGB values are premultiplied by a.
 */

#define DRAW_MUL(_a, _b) (((unsigned)(_a) * (_b)) / 255)

#define DRAW_FASTSETPIXEL(type) \
    *pixels = (type)color

#define DRAW_FASTSETPIXEL1 DRAW_FASTSETPIXEL(Uint8)
#define DRAW_FASTSETPIXEL2 DRAW_FASTSETPIXEL(Uint16)
#define DRAW_FASTSETPIXEL4 DRAW_FASTSETPIXEL(Uint32)

#define DRAW_FASTSETPIXELXY(x, y, type, bpp, color) \
    *(type *)((Uint8 *)dst->pixels + (y)*dst->pitch + (x)*bpp) = (type)color

#define DRAW_FASTSETPIXELXY1(x, y) DRAW_FASTSETPIXELXY(x, y, Uint8, 1, color)
#define DRAW_FASTSETPIXELXY2(x, y) DRAW_FASTSETPIXELXY(x, y, Uint16, 2, color)
#define DRAW_FASTSETPIXELXY4(x, y) DRAW_FASTSETPIXELXY(x, y, Uint32, 4, color)

#define DRAW_SETPIXEL(setpixel)                  \
    do {                                         \
        unsigned sr = r, sg = g, sb = b, sa = a; \
        (void)sa;                                \
        setpixel;                                \
    } while (0)

#define DRAW_SETPIXEL_BLEND(getpixel, setpixel) \
    do {                                        \
        unsigned sr, sg, sb, sa = 0xFF;         \
        getpixel;                               \
        sr = DRAW_MUL(inva, sr) + r;            \
        sg = DRAW_MUL(inva, sg) + g;            \
        sb = DRAW_MUL(inva, sb) + b;            \
        sa = DRAW_MUL(inva, sa) + a;            \
        setpixel;                               \
    } while (0)

#define DRAW_SETPIXEL_BLEND_CLAMPED(getpixel, setpixel) \
    do {                                                \
        unsigned sr, sg, sb, sa = 0xFF;                 \
        getpixel;                                       \
        sr = DRAW_MUL(inva, sr) + r;                    \
        if (sr > 0xff)                                  \
            sr = 0xff;                                  \
        sg = DRAW_MUL(inva, sg) + g;                    \
        if (sg > 0xff)                                  \
            sg = 0xff;                                  \
        sb = DRAW_MUL(inva, sb) + b;                    \
        if (sb > 0xff)                                  \
            sb = 0xff;                                  \
        sa = DRAW_MUL(inva, sa) + a;                    \
        if (sa > 0xff)                                  \
            sa = 0xff;                                  \
        setpixel;                                       \
    } while (0)

#define DRAW_SETPIXEL_ADD(getpixel, setpixel) \
    do {                                      \
        unsigned sr, sg, sb, sa;              \
        (void)sa;                             \
        getpixel;                             \
        sr += r;                              \
        if (sr > 0xff)                        \
            sr = 0xff;                        \
        sg += g;                              \
        if (sg > 0xff)                        \
            sg = 0xff;                        \
        sb += b;                              \
        if (sb > 0xff)                        \
            sb = 0xff;                        \
        setpixel;                             \
    } while (0)

#define DRAW_SETPIXEL_MOD(getpixel, setpixel) \
    do {                                      \
        unsigned sr, sg, sb, sa;              \
        (void)sa;                             \
        getpixel;                             \
        sr = DRAW_MUL(sr, r);                 \
        sg = DRAW_MUL(sg, g);                 \
        sb = DRAW_MUL(sb, b);                 \
        setpixel;                             \
    } while (0)

#define DRAW_SETPIXEL_MUL(getpixel, setpixel)      \
    do {                                           \
        unsigned sr, sg, sb, sa;                   \
        (void)sa;                                  \
        getpixel;                                  \
        sr = DRAW_MUL(sr, r) + DRAW_MUL(inva, sr); \
        if (sr > 0xff)                             \
            sr = 0xff;                             \
        sg = DRAW_MUL(sg, g) + DRAW_MUL(inva, sg); \
        if (sg > 0xff)                             \
            sg = 0xff;                             \
        sb = DRAW_MUL(sb, b) + DRAW_MUL(inva, sb); \
        if (sb > 0xff)                             \
            sb = 0xff;                             \
        setpixel;                                  \
    } while (0)

#define DRAW_SETPIXELXY(x, y, type, bpp, op)                                     \
    do {                                                                         \
        type *pixels = (type *)((Uint8 *)dst->pixels + (y)*dst->pitch + (x)*bpp);\
        op;                                                                      \
    } while (0)

/*
 * Define draw operators for RGB555
 */

#define DRAW_SETPIXEL_RGB555 \
    DRAW_SETPIXEL(RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_RGB555                           \
    DRAW_SETPIXEL_BLEND(RGB_FROM_RGB555(*pixels, sr, sg, sb),\
                        RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_CLAMPED_RGB555                           \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGB_FROM_RGB555(*pixels, sr, sg, sb),\
                                RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_ADD_RGB555                           \
    DRAW_SETPIXEL_ADD(RGB_FROM_RGB555(*pixels, sr, sg, sb),\
                      RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MOD_RGB555                           \
    DRAW_SETPIXEL_MOD(RGB_FROM_RGB555(*pixels, sr, sg, sb),\
                      RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MUL_RGB555                           \
    DRAW_SETPIXEL_MUL(RGB_FROM_RGB555(*pixels, sr, sg, sb),\
                      RGB555_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXELXY_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_RGB555)

#define DRAW_SETPIXELXY_BLEND_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_RGB555)

#define DRAW_SETPIXELXY_BLEND_CLAMPED_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_CLAMPED_RGB555)

#define DRAW_SETPIXELXY_ADD_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_ADD_RGB555)

#define DRAW_SETPIXELXY_MOD_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MOD_RGB555)

#define DRAW_SETPIXELXY_MUL_RGB555(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MUL_RGB555)

/*
 * Define draw operators for RGB565
 */

#define DRAW_SETPIXEL_RGB565 \
    DRAW_SETPIXEL(RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_RGB565                           \
    DRAW_SETPIXEL_BLEND(RGB_FROM_RGB565(*pixels, sr, sg, sb),\
                        RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_CLAMPED_RGB565                           \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGB_FROM_RGB565(*pixels, sr, sg, sb),\
                                RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_ADD_RGB565                           \
    DRAW_SETPIXEL_ADD(RGB_FROM_RGB565(*pixels, sr, sg, sb),\
                      RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MOD_RGB565                           \
    DRAW_SETPIXEL_MOD(RGB_FROM_RGB565(*pixels, sr, sg, sb),\
                      RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MUL_RGB565                           \
    DRAW_SETPIXEL_MUL(RGB_FROM_RGB565(*pixels, sr, sg, sb),\
                      RGB565_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXELXY_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_RGB565)

#define DRAW_SETPIXELXY_BLEND_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_RGB565)

#define DRAW_SETPIXELXY_BLEND_CLAMPED_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_CLAMPED_RGB565)

#define DRAW_SETPIXELXY_ADD_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_ADD_RGB565)

#define DRAW_SETPIXELXY_MOD_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MOD_RGB565)

#define DRAW_SETPIXELXY_MUL_RGB565(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MUL_RGB565)

/*
 * Define draw operators for RGB888
 */

#define DRAW_SETPIXEL_XRGB8888 \
    DRAW_SETPIXEL(XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_XRGB8888                           \
    DRAW_SETPIXEL_BLEND(RGB_FROM_XRGB8888(*pixels, sr, sg, sb),\
                        XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888                           \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGB_FROM_XRGB8888(*pixels, sr, sg, sb),\
                                XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_ADD_XRGB8888                           \
    DRAW_SETPIXEL_ADD(RGB_FROM_XRGB8888(*pixels, sr, sg, sb),\
                      XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MOD_XRGB8888                           \
    DRAW_SETPIXEL_MOD(RGB_FROM_XRGB8888(*pixels, sr, sg, sb),\
                      XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXEL_MUL_XRGB8888                           \
    DRAW_SETPIXEL_MUL(RGB_FROM_XRGB8888(*pixels, sr, sg, sb),\
                      XRGB8888_FROM_RGB(*pixels, sr, sg, sb))

#define DRAW_SETPIXELXY_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_XRGB8888)

#define DRAW_SETPIXELXY_BLEND_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_XRGB8888)

#define DRAW_SETPIXELXY_BLEND_CLAMPED_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_CLAMPED_XRGB8888)

#define DRAW_SETPIXELXY_ADD_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_ADD_XRGB8888)

#define DRAW_SETPIXELXY_MOD_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MOD_XRGB8888)

#define DRAW_SETPIXELXY_MUL_XRGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MUL_XRGB8888)

/*
 * Define draw operators for ARGB8888
 */

#define DRAW_SETPIXEL_ARGB8888 \
    DRAW_SETPIXEL(ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXEL_BLEND_ARGB8888                                \
    DRAW_SETPIXEL_BLEND(RGBA_FROM_ARGB8888(*pixels, sr, sg, sb, sa),\
                        ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888                                \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGBA_FROM_ARGB8888(*pixels, sr, sg, sb, sa),\
                                ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXEL_ADD_ARGB8888                                \
    DRAW_SETPIXEL_ADD(RGBA_FROM_ARGB8888(*pixels, sr, sg, sb, sa),\
                      ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXEL_MOD_ARGB8888                                \
    DRAW_SETPIXEL_MOD(RGBA_FROM_ARGB8888(*pixels, sr, sg, sb, sa),\
                      ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXEL_MUL_ARGB8888                                \
    DRAW_SETPIXEL_MUL(RGBA_FROM_ARGB8888(*pixels, sr, sg, sb, sa),\
                      ARGB8888_FROM_RGBA(*pixels, sr, sg, sb, sa))

#define DRAW_SETPIXELXY_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_ARGB8888)

#define DRAW_SETPIXELXY_BLEND_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_ARGB8888)

#define DRAW_SETPIXELXY_BLEND_CLAMPED_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_CLAMPED_ARGB8888)

#define DRAW_SETPIXELXY_ADD_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_ADD_ARGB8888)

#define DRAW_SETPIXELXY_MOD_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MOD_ARGB8888)

#define DRAW_SETPIXELXY_MUL_ARGB8888(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MUL_ARGB8888)

/*
 * Define draw operators for general RGB
 */

#define DRAW_SETPIXEL_RGB \
    DRAW_SETPIXEL(PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_RGB                                  \
    DRAW_SETPIXEL_BLEND(RGB_FROM_PIXEL(*pixels, fmt, sr, sg, sb),\
                        PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXEL_BLEND_CLAMPED_RGB                                  \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGB_FROM_PIXEL(*pixels, fmt, sr, sg, sb),\
                                PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXEL_ADD_RGB                                  \
    DRAW_SETPIXEL_ADD(RGB_FROM_PIXEL(*pixels, fmt, sr, sg, sb),\
                      PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXEL_MOD_RGB                                  \
    DRAW_SETPIXEL_MOD(RGB_FROM_PIXEL(*pixels, fmt, sr, sg, sb),\
                      PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXEL_MUL_RGB                                  \
    DRAW_SETPIXEL_MUL(RGB_FROM_PIXEL(*pixels, fmt, sr, sg, sb),\
                      PIXEL_FROM_RGB(*pixels, fmt, sr, sg, sb))

#define DRAW_SETPIXELXY2_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_RGB)

#define DRAW_SETPIXELXY4_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_RGB)

#define DRAW_SETPIXELXY2_BLEND_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_RGB)

#define DRAW_SETPIXELXY2_BLEND_CLAMPED_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_BLEND_CLAMPED_RGB)

#define DRAW_SETPIXELXY4_BLEND_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_RGB)

#define DRAW_SETPIXELXY4_BLEND_CLAMPED_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_CLAMPED_RGB)

#define DRAW_SETPIXELXY2_ADD_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_ADD_RGB)

#define DRAW_SETPIXELXY4_ADD_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_ADD_RGB)

#define DRAW_SETPIXELXY2_MOD_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MOD_RGB)

#define DRAW_SETPIXELXY4_MOD_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MOD_RGB)

#define DRAW_SETPIXELXY2_MUL_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint16, 2, DRAW_SETPIXEL_MUL_RGB)

#define DRAW_SETPIXELXY4_MUL_RGB(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MUL_RGB)

/*
 * Define draw operators for general RGBA
 */

#define DRAW_SETPIXEL_RGBA \
    DRAW_SETPIXEL(PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXEL_BLEND_RGBA                                      \
    DRAW_SETPIXEL_BLEND(RGBA_FROM_PIXEL(*pixels, fmt, sr, sg, sb, sa),\
                        PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXEL_BLEND_CLAMPED_RGBA                                      \
    DRAW_SETPIXEL_BLEND_CLAMPED(RGBA_FROM_PIXEL(*pixels, fmt, sr, sg, sb, sa),\
                                PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXEL_ADD_RGBA                                      \
    DRAW_SETPIXEL_ADD(RGBA_FROM_PIXEL(*pixels, fmt, sr, sg, sb, sa),\
                      PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXEL_MOD_RGBA                                      \
    DRAW_SETPIXEL_MOD(RGBA_FROM_PIXEL(*pixels, fmt, sr, sg, sb, sa),\
                      PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXEL_MUL_RGBA                                      \
    DRAW_SETPIXEL_MUL(RGBA_FROM_PIXEL(*pixels, fmt, sr, sg, sb, sa),\
                      PIXEL_FROM_RGBA(*pixels, fmt, sr, sg, sb, sa))

#define DRAW_SETPIXELXY4_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_RGBA)

#define DRAW_SETPIXELXY4_BLEND_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_RGBA)

#define DRAW_SETPIXELXY4_BLEND_CLAMPED_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_BLEND_CLAMPED_RGBA)

#define DRAW_SETPIXELXY4_ADD_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_ADD_RGBA)

#define DRAW_SETPIXELXY4_MOD_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MOD_RGBA)

#define DRAW_SETPIXELXY4_MUL_RGBA(x, y) \
    DRAW_SETPIXELXY(x, y, Uint32, 4, DRAW_SETPIXEL_MUL_RGBA)

/*
 * Define line drawing macro
 */

#define ABS(_x) ((_x) < 0 ? -(_x) : (_x))

// Horizontal line
#define HLINE(type, op, draw_end)                              \
    {                                                          \
        int length;                                            \
        int pitch = (dst->pitch / dst->fmt->bytes_per_pixel);  \
        type *pixels;                                          \
        if (x1 <= x2) {                                        \
            pixels = (type *)dst->pixels + y1 * pitch + x1;    \
            length = draw_end ? (x2 - x1 + 1) : (x2 - x1);     \
        } else {                                               \
            pixels = (type *)dst->pixels + y1 * pitch + x2;    \
            if (!draw_end) {                                   \
                ++pixels;                                      \
            }                                                  \
            length = draw_end ? (x1 - x2 + 1) : (x1 - x2);     \
        }                                                      \
        while (length--) {                                     \
            op;                                                \
            ++pixels;                                          \
        }                                                      \
    }

// Vertical line
#define VLINE(type, op, draw_end)                              \
    {                                                          \
        int length;                                            \
        int pitch = (dst->pitch / dst->fmt->bytes_per_pixel);  \
        type *pixels;                                          \
        if (y1 <= y2) {                                        \
            pixels = (type *)dst->pixels + y1 * pitch + x1;    \
            length = draw_end ? (y2 - y1 + 1) : (y2 - y1);     \
        } else {                                               \
            pixels = (type *)dst->pixels + y2 * pitch + x1;    \
            if (!draw_end) {                                   \
                pixels += pitch;                               \
            }                                                  \
            length = draw_end ? (y1 - y2 + 1) : (y1 - y2);     \
        }                                                      \
        while (length--) {                                     \
            op;                                                \
            pixels += pitch;                                   \
        }                                                      \
    }

// Diagonal line
#define DLINE(type, op, draw_end)                              \
    {                                                          \
        int length;                                            \
        int pitch = (dst->pitch / dst->fmt->bytes_per_pixel);  \
        type *pixels;                                          \
        if (y1 <= y2) {                                        \
            pixels = (type *)dst->pixels + y1 * pitch + x1;    \
            if (x1 <= x2) {                                    \
                ++pitch;                                       \
            } else {                                           \
                --pitch;                                       \
            }                                                  \
            length = (y2 - y1);                                \
        } else {                                               \
            pixels = (type *)dst->pixels + y2 * pitch + x2;    \
            if (x2 <= x1) {                                    \
                ++pitch;                                       \
            } else {                                           \
                --pitch;                                       \
            }                                                  \
            if (!draw_end) {                                   \
                pixels += pitch;                               \
            }                                                  \
            length = (y1 - y2);                                \
        }                                                      \
        if (draw_end) {                                        \
            ++length;                                          \
        }                                                      \
        while (length--) {                                     \
            op;                                                \
            pixels += pitch;                                   \
        }                                                      \
    }

// Bresenham's line algorithm
#define BLINE(x1, y1, x2, y2, op, draw_end) \
    {                                       \
        int i, deltax, deltay, numpixels;   \
        int d, dinc1, dinc2;                \
        int x, xinc1, xinc2;                \
        int y, yinc1, yinc2;                \
                                            \
        deltax = ABS(x2 - x1);              \
        deltay = ABS(y2 - y1);              \
                                            \
        if (deltax >= deltay) {             \
            numpixels = deltax + 1;         \
            d = (2 * deltay) - deltax;      \
            dinc1 = deltay * 2;             \
            dinc2 = (deltay - deltax) * 2;  \
            xinc1 = 1;                      \
            xinc2 = 1;                      \
            yinc1 = 0;                      \
            yinc2 = 1;                      \
        } else {                            \
            numpixels = deltay + 1;         \
            d = (2 * deltax) - deltay;      \
            dinc1 = deltax * 2;             \
            dinc2 = (deltax - deltay) * 2;  \
            xinc1 = 0;                      \
            xinc2 = 1;                      \
            yinc1 = 1;                      \
            yinc2 = 1;                      \
        }                                   \
                                            \
        if (x1 > x2) {                      \
            xinc1 = -xinc1;                 \
            xinc2 = -xinc2;                 \
        }                                   \
        if (y1 > y2) {                      \
            yinc1 = -yinc1;                 \
            yinc2 = -yinc2;                 \
        }                                   \
                                            \
        x = x1;                             \
        y = y1;                             \
                                            \
        if (!draw_end) {                    \
            --numpixels;                    \
        }                                   \
        for (i = 0; i < numpixels; ++i) {   \
            op(x, y);                       \
            if (d < 0) {                    \
                d += dinc1;                 \
                x += xinc1;                 \
                y += yinc1;                 \
            } else {                        \
                d += dinc2;                 \
                x += xinc2;                 \
                y += yinc2;                 \
            }                               \
        }                                   \
    }

// Xiaolin Wu's line algorithm, based on Michael Abrash's implementation
#define WULINE(x1, y1, x2, y2, opaque_op, blend_op, draw_end)                       \
    {                                                                               \
        Uint16 ErrorAdj, ErrorAcc;                                                  \
        Uint16 ErrorAccTemp, Weighting;                                             \
        int DeltaX, DeltaY, Temp, XDir;                                             \
        unsigned r, g, b, a, inva;                                                  \
                                                                                    \
        /* Draw the initial pixel, which is always exactly intersected by           \
           the line and so needs no weighting */                                    \
        opaque_op(x1, y1);                                                          \
                                                                                    \
        /* Draw the final pixel, which is always exactly intersected by the line    \
           and so needs no weighting */                                             \
        if (draw_end) {                                                             \
            opaque_op(x2, y2);                                                      \
        }                                                                           \
                                                                                    \
        /* Make sure the line runs top to bottom */                                 \
        if (y1 > y2) {                                                              \
            Temp = y1;                                                              \
            y1 = y2;                                                                \
            y2 = Temp;                                                              \
            Temp = x1;                                                              \
            x1 = x2;                                                                \
            x2 = Temp;                                                              \
        }                                                                           \
        DeltaY = y2 - y1;                                                           \
                                                                                    \
        if ((DeltaX = x2 - x1) >= 0) {                                              \
            XDir = 1;                                                               \
        } else {                                                                    \
            XDir = -1;                                                              \
            DeltaX = -DeltaX; /* make DeltaX positive */                            \
        }                                                                           \
                                                                                    \
        /* line is not horizontal, diagonal, or vertical */                         \
        ErrorAcc = 0; /* initialize the line error accumulator to 0 */              \
                                                                                    \
        /* Is this an X-major or Y-major line? */                                   \
        if (DeltaY > DeltaX) {                                                      \
            /* Y-major line; calculate 16-bit fixed-point fractional part of a      \
              pixel that X advances each time Y advances 1 pixel, truncating the    \
              result so that we won't overrun the endpoint along the X axis */      \
            ErrorAdj = ((unsigned long)DeltaX << 16) / (unsigned long)DeltaY;       \
            /* Draw all pixels other than the first and last */                     \
            while (--DeltaY) {                                                      \
                ErrorAccTemp = ErrorAcc; /* remember current accumulated error */   \
                ErrorAcc += ErrorAdj;    /* calculate error for next pixel */       \
                if (ErrorAcc <= ErrorAccTemp) {                                     \
                    /* The error accumulator turned over, so advance the X coord */ \
                    x1 += XDir;                                                     \
                }                                                                   \
                y1++; /* Y-major, so always advance Y */                            \
                /* The IntensityBits most significant bits of ErrorAcc give us the  \
                 intensity weighting for this pixel, and the complement of the      \
                 weighting for the paired pixel */                                  \
                Weighting = ErrorAcc >> 8;                                          \
                {                                                                   \
                    a = DRAW_MUL(_a, (Weighting ^ 255));                            \
                    r = DRAW_MUL(_r, a);                                            \
                    g = DRAW_MUL(_g, a);                                            \
                    b = DRAW_MUL(_b, a);                                            \
                    inva = (a ^ 0xFF);                                              \
                    blend_op(x1, y1);                                               \
                }                                                                   \
                {                                                                   \
                    a = DRAW_MUL(_a, Weighting);                                    \
                    r = DRAW_MUL(_r, a);                                            \
                    g = DRAW_MUL(_g, a);                                            \
                    b = DRAW_MUL(_b, a);                                            \
                    inva = (a ^ 0xFF);                                              \
                    blend_op(x1 + XDir, y1);                                        \
                }                                                                   \
            }                                                                       \
        } else {                                                                    \
            /* X-major line; calculate 16-bit fixed-point fractional part of a      \
               pixel that Y advances each time X advances 1 pixel, truncating the   \
               result to avoid overrunning the endpoint along the X axis */         \
            ErrorAdj = ((unsigned long)DeltaY << 16) / (unsigned long)DeltaX;       \
            /* Draw all pixels other than the first and last */                     \
            while (--DeltaX) {                                                      \
                ErrorAccTemp = ErrorAcc; /* remember current accumulated error */   \
                ErrorAcc += ErrorAdj;    /* calculate error for next pixel */       \
                if (ErrorAcc <= ErrorAccTemp) {                                     \
                    /* The error accumulator turned over, so advance the Y coord */ \
                    y1++;                                                           \
                }                                                                   \
                x1 += XDir; /* X-major, so always advance X */                      \
                /* The IntensityBits most significant bits of ErrorAcc give us the  \
                  intensity weighting for this pixel, and the complement of the     \
                  weighting for the paired pixel */                                 \
                Weighting = ErrorAcc >> 8;                                          \
                {                                                                   \
                    a = DRAW_MUL(_a, (Weighting ^ 255));                            \
                    r = DRAW_MUL(_r, a);                                            \
                    g = DRAW_MUL(_g, a);                                            \
                    b = DRAW_MUL(_b, a);                                            \
                    inva = (a ^ 0xFF);                                              \
                    blend_op(x1, y1);                                               \
                }                                                                   \
                {                                                                   \
                    a = DRAW_MUL(_a, Weighting);                                    \
                    r = DRAW_MUL(_r, a);                                            \
                    g = DRAW_MUL(_g, a);                                            \
                    b = DRAW_MUL(_b, a);                                            \
                    inva = (a ^ 0xFF);                                              \
                    blend_op(x1, y1 + 1);                                           \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    }

#ifdef AA_LINES
#define AALINE(x1, y1, x2, y2, opaque_op, blend_op, draw_end) \
    WULINE(x1, y1, x2, y2, opaque_op, blend_op, draw_end)
#else
#define AALINE(x1, y1, x2, y2, opaque_op, blend_op, draw_end) \
    BLINE(x1, y1, x2, y2, opaque_op, draw_end)
#endif

/*
 * Define fill rect macro
 */

#define FILLRECT(type, op)                                             \
    do {                                                               \
        int width = rect->w;                                           \
        int height = rect->h;                                          \
        int pitch = (dst->pitch / dst->fmt->bytes_per_pixel);          \
        int skip = pitch - width;                                      \
        type *pixels = (type *)dst->pixels + rect->y * pitch + rect->x;\
        while (height--) {                                             \
            {                                                          \
                int n = (width + 3) / 4;                               \
                switch (width & 3) {                                   \
                case 0:                                                \
                    do {                                               \
                        op;                                            \
                        pixels++;                                      \
                        SDL_FALLTHROUGH;                               \
                    case 3:                                            \
                        op;                                            \
                        pixels++;                                      \
                        SDL_FALLTHROUGH;                               \
                    case 2:                                            \
                        op;                                            \
                        pixels++;                                      \
                        SDL_FALLTHROUGH;                               \
                    case 1:                                            \
                        op;                                            \
                        pixels++;                                      \
                    } while (--n > 0);                                 \
                }                                                      \
            }                                                          \
            pixels += skip;                                            \
        }                                                              \
    } while (0)
