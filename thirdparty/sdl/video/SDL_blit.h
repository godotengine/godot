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

#ifndef SDL_blit_h_
#define SDL_blit_h_

// Table to do pixel byte expansion
extern const Uint8 *SDL_expand_byte[9];
extern const Uint16 SDL_expand_byte_10[];

// SDL blit copy flags
#define SDL_COPY_MODULATE_COLOR         0x00000001
#define SDL_COPY_MODULATE_ALPHA         0x00000002
#define SDL_COPY_MODULATE_MASK		(SDL_COPY_MODULATE_COLOR | SDL_COPY_MODULATE_ALPHA)
#define SDL_COPY_BLEND                  0x00000010
#define SDL_COPY_BLEND_PREMULTIPLIED    0x00000020
#define SDL_COPY_ADD                    0x00000040
#define SDL_COPY_ADD_PREMULTIPLIED      0x00000080
#define SDL_COPY_MOD                    0x00000100
#define SDL_COPY_MUL                    0x00000200
#define SDL_COPY_BLEND_MASK		(SDL_COPY_BLEND | SDL_COPY_BLEND_PREMULTIPLIED | SDL_COPY_ADD | SDL_COPY_ADD_PREMULTIPLIED | SDL_COPY_MOD | SDL_COPY_MUL)
#define SDL_COPY_COLORKEY               0x00000400
#define SDL_COPY_NEAREST                0x00000800
#define SDL_COPY_RLE_DESIRED            0x00001000
#define SDL_COPY_RLE_COLORKEY           0x00002000
#define SDL_COPY_RLE_ALPHAKEY           0x00004000
#define SDL_COPY_RLE_MASK               (SDL_COPY_RLE_DESIRED | SDL_COPY_RLE_COLORKEY | SDL_COPY_RLE_ALPHAKEY)

// SDL blit CPU flags
#define SDL_CPU_ANY                0x00000000
#define SDL_CPU_MMX                0x00000001
#define SDL_CPU_SSE                0x00000002
#define SDL_CPU_SSE2               0x00000004
#define SDL_CPU_ALTIVEC_PREFETCH   0x00000008
#define SDL_CPU_ALTIVEC_NOPREFETCH 0x00000010

typedef struct
{
    SDL_Surface *src_surface;
    Uint8 *src;
    int src_w, src_h;
    int src_pitch;
    int src_skip;
    SDL_Surface *dst_surface;
    Uint8 *dst;
    int dst_w, dst_h;
    int dst_pitch;
    int dst_skip;
    const SDL_PixelFormatDetails *src_fmt;
    const SDL_Palette *src_pal;
    const SDL_PixelFormatDetails *dst_fmt;
    const SDL_Palette *dst_pal;
    Uint8 *table;
    SDL_HashTable *palette_map;
    int flags;
    Uint32 colorkey;
    Uint8 r, g, b, a;
} SDL_BlitInfo;

typedef void (*SDL_BlitFunc)(SDL_BlitInfo *info);

typedef struct
{
    SDL_PixelFormat src_format;
    SDL_PixelFormat dst_format;
    int flags;
    unsigned int cpu;
    SDL_BlitFunc func;
} SDL_BlitFuncEntry;

typedef bool (SDLCALL *SDL_Blit) (struct SDL_Surface *src, const SDL_Rect *srcrect, struct SDL_Surface *dst, const SDL_Rect *dstrect);

// Blit mapping definition
typedef struct SDL_BlitMap
{
    int identity;
    SDL_Blit blit;
    void *data;
    SDL_BlitInfo info;

    /* the version count matches the destination; mismatch indicates
       an invalid mapping */
    Uint32 dst_palette_version;
    Uint32 src_palette_version;
} SDL_BlitMap;

// Functions found in SDL_blit.c
extern bool SDL_CalculateBlit(SDL_Surface *surface, SDL_Surface *dst);

/* Functions found in SDL_blit_*.c */
extern SDL_BlitFunc SDL_CalculateBlit0(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlit1(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlitN(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlitA(SDL_Surface *surface);

/*
 * Useful macros for blitting routines
 */

#ifdef __GNUC__
#define DECLARE_ALIGNED(t, v, a) t __attribute__((aligned(a))) v
#elif defined(_MSC_VER)
#define DECLARE_ALIGNED(t, v, a) __declspec(align(a)) t v
#else
#define DECLARE_ALIGNED(t, v, a) t v
#endif

// Load pixel of the specified format from a buffer and get its R-G-B values
#define RGB_FROM_PIXEL(Pixel, fmt, r, g, b)                                     \
    {                                                                           \
        r = SDL_expand_byte[fmt->Rbits][((Pixel & fmt->Rmask) >> fmt->Rshift)]; \
        g = SDL_expand_byte[fmt->Gbits][((Pixel & fmt->Gmask) >> fmt->Gshift)]; \
        b = SDL_expand_byte[fmt->Bbits][((Pixel & fmt->Bmask) >> fmt->Bshift)]; \
    }
#define RGB_FROM_RGB565(Pixel, r, g, b)                   \
    {                                                     \
        r = SDL_expand_byte[5][((Pixel & 0xF800) >> 11)]; \
        g = SDL_expand_byte[6][((Pixel & 0x07E0) >> 5)];  \
        b = SDL_expand_byte[5][(Pixel & 0x001F)];         \
    }
#define RGB_FROM_RGB555(Pixel, r, g, b)                   \
    {                                                     \
        r = SDL_expand_byte[5][((Pixel & 0x7C00) >> 10)]; \
        g = SDL_expand_byte[5][((Pixel & 0x03E0) >> 5)];  \
        b = SDL_expand_byte[5][(Pixel & 0x001F)];         \
    }
#define RGB_FROM_XRGB8888(Pixel, r, g, b) \
    {                                   \
        r = ((Pixel & 0xFF0000) >> 16); \
        g = ((Pixel & 0xFF00) >> 8);    \
        b = (Pixel & 0xFF);             \
    }
#define RETRIEVE_RGB_PIXEL(buf, bpp, Pixel)                \
    do {                                                   \
        switch (bpp) {                                     \
        case 1:                                            \
            Pixel = *((Uint8 *)(buf));                     \
            break;                                         \
                                                           \
        case 2:                                            \
            Pixel = *((Uint16 *)(buf));                    \
            break;                                         \
                                                           \
        case 3:                                            \
        {                                                  \
            Uint8 *B = (Uint8 *)(buf);                     \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {         \
                Pixel = B[0] + (B[1] << 8) + (B[2] << 16); \
            } else {                                       \
                Pixel = (B[0] << 16) + (B[1] << 8) + B[2]; \
            }                                              \
        } break;                                           \
                                                           \
        case 4:                                            \
            Pixel = *((Uint32 *)(buf));                    \
            break;                                         \
                                                           \
        default:                                           \
            Pixel = 0; /* stop gcc complaints */           \
            break;                                         \
        }                                                  \
    } while (0)

#define DISEMBLE_RGB(buf, bpp, fmt, Pixel, r, g, b) \
    do {                                            \
        switch (bpp) {                              \
        case 1:                                     \
            Pixel = *((Uint8 *)(buf));              \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        case 2:                                     \
            Pixel = *((Uint16 *)(buf));             \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        case 3:                                     \
        {                                           \
            Pixel = 0;                              \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {  \
                r = *((buf) + fmt->Rshift / 8);     \
                g = *((buf) + fmt->Gshift / 8);     \
                b = *((buf) + fmt->Bshift / 8);     \
            } else {                                \
                r = *((buf) + 2 - fmt->Rshift / 8); \
                g = *((buf) + 2 - fmt->Gshift / 8); \
                b = *((buf) + 2 - fmt->Bshift / 8); \
            }                                       \
        } break;                                    \
                                                    \
        case 4:                                     \
            Pixel = *((Uint32 *)(buf));             \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        default:                                    \
            /* stop gcc complaints */               \
            Pixel = 0;                              \
            r = g = b = 0;                          \
            break;                                  \
        }                                           \
    } while (0)

// Assemble R-G-B values into a specified pixel format and store them
#define PIXEL_FROM_RGB(Pixel, fmt, r, g, b)                 \
    {                                                       \
        Pixel = ((r >> (8 - fmt->Rbits)) << fmt->Rshift) |  \
                ((g >> (8 - fmt->Gbits)) << fmt->Gshift) |  \
                ((b >> (8 - fmt->Bbits)) << fmt->Bshift) |  \
                fmt->Amask;                                 \
    }
#define RGB332_FROM_RGB(Pixel, r, g, b)                        \
    {                                                          \
        Pixel = (Uint8)(((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6)); \
    }
#define RGB565_FROM_RGB(Pixel, r, g, b)                        \
    {                                                          \
        Pixel = (Uint16)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)); \
    }
#define RGB555_FROM_RGB(Pixel, r, g, b)                        \
    {                                                          \
        Pixel = (Uint16)(((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)); \
    }
#define XRGB8888_FROM_RGB(Pixel, r, g, b)   \
    {                                     \
        Pixel = (r << 16) | (g << 8) | b; \
    }
#define ARGB8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (a << 24) | (r << 16) | (g << 8) | b; \
    }
#define RGBA8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (r << 24) | (g << 16) | (b << 8) | a; \
    }
#define ABGR8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (a << 24) | (b << 16) | (g << 8) | r; \
    }
#define BGRA8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (b << 24) | (g << 16) | (r << 8) | a; \
    }
#define ARGB2101010_FROM_RGBA(Pixel, r, g, b, a)       \
    {                                                  \
        r = r ? ((r << 2) | 0x3) : 0;                  \
        g = g ? ((g << 2) | 0x3) : 0;                  \
        b = b ? ((b << 2) | 0x3) : 0;                  \
        a = (a * 3) / 255;                             \
        Pixel = (a << 30) | (r << 20) | (g << 10) | b; \
    }
#define ARGB2101010_FROM_RGBAFLOAT(Pixel, r, g, b, a) \
    {                                                 \
        r = SDL_clamp(r, 0.0f, 1.0f) * 1023.0f;       \
        g = SDL_clamp(g, 0.0f, 1.0f) * 1023.0f;       \
        b = SDL_clamp(b, 0.0f, 1.0f) * 1023.0f;       \
        a = SDL_clamp(a, 0.0f, 1.0f) * 3.0f;          \
        Pixel = (((Uint32)SDL_roundf(a)) << 30) |     \
                (((Uint32)SDL_roundf(r)) << 20) |     \
                (((Uint32)SDL_roundf(g)) << 10) |     \
                (Uint32)SDL_roundf(b);                \
    }
#define ABGR2101010_FROM_RGBA(Pixel, r, g, b, a)       \
    {                                                  \
        r = r ? ((r << 2) | 0x3) : 0;                  \
        g = g ? ((g << 2) | 0x3) : 0;                  \
        b = b ? ((b << 2) | 0x3) : 0;                  \
        a = (a * 3) / 255;                             \
        Pixel = (a << 30) | (b << 20) | (g << 10) | r; \
    }
#define ABGR2101010_FROM_RGBAFLOAT(Pixel, r, g, b, a) \
    {                                                 \
        r = SDL_clamp(r, 0.0f, 1.0f) * 1023.0f;       \
        g = SDL_clamp(g, 0.0f, 1.0f) * 1023.0f;       \
        b = SDL_clamp(b, 0.0f, 1.0f) * 1023.0f;       \
        a = SDL_clamp(a, 0.0f, 1.0f) * 3.0f;          \
        Pixel = (((Uint32)SDL_roundf(a)) << 30) |     \
                (((Uint32)SDL_roundf(b)) << 20) |     \
                (((Uint32)SDL_roundf(g)) << 10) |     \
                (Uint32)SDL_roundf(r);                \
    }
#define ASSEMBLE_RGB(buf, bpp, fmt, r, g, b)        \
    {                                               \
        switch (bpp) {                              \
        case 1:                                     \
        {                                           \
            Uint8 _pixel;                           \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint8 *)(buf)) = _pixel;             \
        } break;                                    \
                                                    \
        case 2:                                     \
        {                                           \
            Uint16 _pixel;                          \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint16 *)(buf)) = _pixel;            \
        } break;                                    \
                                                    \
        case 3:                                     \
        {                                           \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {  \
                *((buf) + fmt->Rshift / 8) = r;     \
                *((buf) + fmt->Gshift / 8) = g;     \
                *((buf) + fmt->Bshift / 8) = b;     \
            } else {                                \
                *((buf) + 2 - fmt->Rshift / 8) = r; \
                *((buf) + 2 - fmt->Gshift / 8) = g; \
                *((buf) + 2 - fmt->Bshift / 8) = b; \
            }                                       \
        } break;                                    \
                                                    \
        case 4:                                     \
        {                                           \
            Uint32 _pixel;                          \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint32 *)(buf)) = _pixel;            \
        } break;                                    \
        }                                           \
    }

// FIXME: Should we rescale alpha into 0..255 here?
#define RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a)                                 \
    {                                                                           \
        r = SDL_expand_byte[fmt->Rbits][((Pixel & fmt->Rmask) >> fmt->Rshift)]; \
        g = SDL_expand_byte[fmt->Gbits][((Pixel & fmt->Gmask) >> fmt->Gshift)]; \
        b = SDL_expand_byte[fmt->Bbits][((Pixel & fmt->Bmask) >> fmt->Bshift)]; \
        a = SDL_expand_byte[fmt->Abits][((Pixel & fmt->Amask) >> fmt->Ashift)]; \
    }
#define RGBA_FROM_8888(Pixel, fmt, r, g, b, a)   \
    {                                            \
        r = (Pixel & fmt->Rmask) >> fmt->Rshift; \
        g = (Pixel & fmt->Gmask) >> fmt->Gshift; \
        b = (Pixel & fmt->Bmask) >> fmt->Bshift; \
        a = (Pixel & fmt->Amask) >> fmt->Ashift; \
    }
#define RGBA_FROM_RGBA8888(Pixel, r, g, b, a) \
    {                                         \
        r = (Pixel >> 24);                    \
        g = ((Pixel >> 16) & 0xFF);           \
        b = ((Pixel >> 8) & 0xFF);            \
        a = (Pixel & 0xFF);                   \
    }
#define RGBA_FROM_ARGB8888(Pixel, r, g, b, a) \
    {                                         \
        r = ((Pixel >> 16) & 0xFF);           \
        g = ((Pixel >> 8) & 0xFF);            \
        b = (Pixel & 0xFF);                   \
        a = (Pixel >> 24);                    \
    }
#define RGBA_FROM_ABGR8888(Pixel, r, g, b, a) \
    {                                         \
        r = (Pixel & 0xFF);                   \
        g = ((Pixel >> 8) & 0xFF);            \
        b = ((Pixel >> 16) & 0xFF);           \
        a = (Pixel >> 24);                    \
    }
#define RGBA_FROM_BGRA8888(Pixel, r, g, b, a) \
    {                                         \
        r = ((Pixel >> 8) & 0xFF);            \
        g = ((Pixel >> 16) & 0xFF);           \
        b = (Pixel >> 24);                    \
        a = (Pixel & 0xFF);                   \
    }
#define RGBA_FROM_ARGB2101010(Pixel, r, g, b, a) \
    {                                            \
        r = ((Pixel >> 22) & 0xFF);              \
        g = ((Pixel >> 12) & 0xFF);              \
        b = ((Pixel >> 2) & 0xFF);               \
        a = SDL_expand_byte[2][(Pixel >> 30)];   \
    }
#define RGBAFLOAT_FROM_ARGB2101010(Pixel, r, g, b, a)   \
    {                                                   \
        r = (float)((Pixel >> 20) & 0x3FF) / 1023.0f;   \
        g = (float)((Pixel >> 10) & 0x3FF) / 1023.0f;   \
        b = (float)((Pixel >> 0) & 0x3FF) / 1023.0f;    \
        a = (float)(Pixel >> 30) / 3.0f;                \
    }
#define RGBA_FROM_ABGR2101010(Pixel, r, g, b, a) \
    {                                            \
        r = ((Pixel >> 2) & 0xFF);               \
        g = ((Pixel >> 12) & 0xFF);              \
        b = ((Pixel >> 22) & 0xFF);              \
        a = SDL_expand_byte[2][(Pixel >> 30)];   \
    }
#define RGBAFLOAT_FROM_ABGR2101010(Pixel, r, g, b, a)   \
    {                                                   \
        r = (float)((Pixel >> 0) & 0x3FF) / 1023.0f;    \
        g = (float)((Pixel >> 10) & 0x3FF) / 1023.0f;   \
        b = (float)((Pixel >> 20) & 0x3FF) / 1023.0f;   \
        a = (float)(Pixel >> 30) / 3.0f;                \
    }
#define DISEMBLE_RGBA(buf, bpp, fmt, Pixel, r, g, b, a) \
    do {                                                \
        switch (bpp) {                                  \
        case 1:                                         \
            Pixel = *((Uint8 *)(buf));                  \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        case 2:                                         \
            Pixel = *((Uint16 *)(buf));                 \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        case 3:                                         \
        {                                               \
            Pixel = 0;                                  \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {      \
                r = *((buf) + fmt->Rshift / 8);         \
                g = *((buf) + fmt->Gshift / 8);         \
                b = *((buf) + fmt->Bshift / 8);         \
            } else {                                    \
                r = *((buf) + 2 - fmt->Rshift / 8);     \
                g = *((buf) + 2 - fmt->Gshift / 8);     \
                b = *((buf) + 2 - fmt->Bshift / 8);     \
            }                                           \
            a = 0xFF;                                   \
        } break;                                        \
                                                        \
        case 4:                                         \
            Pixel = *((Uint32 *)(buf));                 \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        default:                                        \
            /* stop gcc complaints */                   \
            Pixel = 0;                                  \
            r = g = b = a = 0;                          \
            break;                                      \
        }                                               \
    } while (0)

// FIXME: this isn't correct, especially for Alpha (maximum != 255)
#define PIXEL_FROM_RGBA(Pixel, fmt, r, g, b, a)             \
    {                                                       \
        Pixel = ((r >> (8 - fmt->Rbits)) << fmt->Rshift) |  \
                ((g >> (8 - fmt->Gbits)) << fmt->Gshift) |  \
                ((b >> (8 - fmt->Bbits)) << fmt->Bshift) |  \
                ((a >> (8 - fmt->Abits)) << fmt->Ashift);   \
    }
#define ASSEMBLE_RGBA(buf, bpp, fmt, r, g, b, a)      \
    {                                                 \
        switch (bpp) {                                \
        case 1:                                       \
        {                                             \
            Uint8 _pixel;                             \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint8 *)(buf)) = _pixel;               \
        } break;                                      \
                                                      \
        case 2:                                       \
        {                                             \
            Uint16 _pixel;                            \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint16 *)(buf)) = _pixel;              \
        } break;                                      \
                                                      \
        case 3:                                       \
        {                                             \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {    \
                *((buf) + fmt->Rshift / 8) = r;       \
                *((buf) + fmt->Gshift / 8) = g;       \
                *((buf) + fmt->Bshift / 8) = b;       \
            } else {                                  \
                *((buf) + 2 - fmt->Rshift / 8) = r;   \
                *((buf) + 2 - fmt->Gshift / 8) = g;   \
                *((buf) + 2 - fmt->Bshift / 8) = b;   \
            }                                         \
        } break;                                      \
                                                      \
        case 4:                                       \
        {                                             \
            Uint32 _pixel;                            \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint32 *)(buf)) = _pixel;              \
        } break;                                      \
        }                                             \
    }

// Convert any 32-bit 4-bpp pixel to ARGB format
#define PIXEL_TO_ARGB_PIXEL(src, srcfmt, dst)         \
    do {                                              \
        Uint8 a, r, g, b;                         \
        RGBA_FROM_PIXEL(src, srcfmt, r, g, b, a); \
        dst = a << 24 | r << 16 | g << 8 | b;     \
    } while (0)
// Blend a single color channel or alpha value
/* dC = ((sC * sA) + (dC * (255 - sA))) / 255 */
#define ALPHA_BLEND_CHANNEL(sC, dC, sA)                  \
    do {                                                 \
        Uint16 x;                                        \
        x = ((sC - dC) * sA) + ((dC << 8) - dC);         \
        x += 0x1U;                                       \
        x += x >> 8;                                     \
        dC = x >> 8;                                     \
    } while (0)
// Perform a division by 255 after a multiplication of two 8-bit color channels
/* out = (sC * dC) / 255 */
#define MULT_DIV_255(sC, dC, out) \
    do {                          \
        Uint16 x = sC * dC;       \
        x += 0x1U;                \
        x += x >> 8;              \
        out = x >> 8;             \
    } while (0)
// Blend the RGB values of two pixels with an alpha value
#define ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB)            \
    do {                                                      \
        ALPHA_BLEND_CHANNEL(sR, dR, A);                       \
        ALPHA_BLEND_CHANNEL(sG, dG, A);                       \
        ALPHA_BLEND_CHANNEL(sB, dB, A);                       \
    } while (0)

// Blend two 8888 pixels with the same format
/* Calculates dst = ((src * factor) + (dst * (255 - factor))) / 255 */
// FIXME: SDL_SIZE_MAX might not be an integer literal
#if defined(SIZE_MAX) && (SIZE_MAX == 0xffffffffffffffff)
#define FACTOR_BLEND_8888(src, dst, factor)                        \
    do {                                                           \
        Uint64 src64 = src;                                        \
        src64 = (src64 | (src64 << 24)) & 0x00FF00FF00FF00FF;      \
                                                                   \
        Uint64 dst64 = dst;                                        \
        dst64 = (dst64 | (dst64 << 24)) & 0x00FF00FF00FF00FF;      \
                                                                   \
        dst64 = ((src64 - dst64) * factor) + (dst64 << 8) - dst64; \
        dst64 += 0x0001000100010001;                               \
        dst64 += (dst64 >> 8) & 0x00FF00FF00FF00FF;                \
        dst64 &= 0xFF00FF00FF00FF00;                               \
                                                                   \
        dst = (Uint32)((dst64 >> 8) | (dst64 >> 32));              \
    } while (0)
#else
#define FACTOR_BLEND_8888(src, dst, factor)                               \
    do {                                                                  \
        Uint32 src02 = src & 0x00FF00FF;                                  \
        Uint32 dst02 = dst & 0x00FF00FF;                                  \
                                                                          \
        Uint32 src13 = (src >> 8) & 0x00FF00FF;                           \
        Uint32 dst13 = (dst >> 8) & 0x00FF00FF;                           \
                                                                          \
        Uint32 res02 = ((src02 - dst02) * factor) + (dst02 << 8) - dst02; \
        res02 += 0x00010001;                                              \
        res02 += (res02 >> 8) & 0x00FF00FF;                               \
        res02 = (res02 >> 8) & 0x00FF00FF;                                \
                                                                          \
        Uint32 res13 = ((src13 - dst13) * factor) + (dst13 << 8) - dst13; \
        res13 += 0x00010001;                                              \
        res13 += (res13 >> 8) & 0x00FF00FF;                               \
        res13 &= 0xFF00FF00;                                              \
        dst = res02 | res13;                                              \
    } while (0)
#endif

// Alpha blend two 8888 pixels with the same formats.
#define ALPHA_BLEND_8888(src, dst, fmt)            \
    do {                                           \
        Uint32 srcA = (src >> fmt->Ashift) & 0xFF; \
        Uint32 tmp = src | fmt->Amask;             \
        FACTOR_BLEND_8888(tmp, dst, srcA);         \
    } while (0)

// Alpha blend two 8888 pixels with differing formats.
#define ALPHA_BLEND_SWIZZLE_8888(src, dst, srcfmt, dstfmt)                  \
    do {                                                                    \
        Uint32 srcA = (src >> srcfmt->Ashift) & 0xFF;                       \
        Uint32 tmp = (((src >> srcfmt->Rshift) & 0xFF) << dstfmt->Rshift) | \
                     (((src >> srcfmt->Gshift) & 0xFF) << dstfmt->Gshift) | \
                     (((src >> srcfmt->Bshift) & 0xFF) << dstfmt->Bshift) | \
                     dstfmt->Amask;                                         \
        FACTOR_BLEND_8888(tmp, dst, srcA);                                  \
    } while (0)
// Blend the RGBA values of two pixels
#define ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA) \
    do {                                                 \
        ALPHA_BLEND_CHANNEL(sR, dR, sA);                 \
        ALPHA_BLEND_CHANNEL(sG, dG, sA);                 \
        ALPHA_BLEND_CHANNEL(sB, dB, sA);                 \
        ALPHA_BLEND_CHANNEL(255, dA, sA);                \
    } while (0)

// This is a very useful loop for optimizing blitters
#if defined(_MSC_VER) && (_MSC_VER == 1300)
// There's a bug in the Visual C++ 7 optimizer when compiling this code
#else
#define USE_DUFFS_LOOP
#endif

#define DUFFS_LOOP1(pixel_copy_increment, width) \
    {                                            \
        int n;                                   \
        for (n = width; n > 0; --n) {            \
            pixel_copy_increment;                \
        }                                        \
    }

#ifdef USE_DUFFS_LOOP

// 8-times unrolled loop
#define DUFFS_LOOP8(pixel_copy_increment, width) \
    {                                            \
        int n = (width + 7) / 8;                 \
        switch (width & 7) {                     \
        case 0:                                  \
            do {                                 \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 7:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 6:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 5:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 4:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 3:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 2:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 1:                              \
                pixel_copy_increment;            \
            } while (--n > 0);                   \
        }                                        \
    }

// 4-times unrolled loop
#define DUFFS_LOOP4(pixel_copy_increment, width) \
    {                                            \
        int n = (width + 3) / 4;                 \
        switch (width & 3) {                     \
        case 0:                                  \
            do {                                 \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 3:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 2:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 1:                              \
                pixel_copy_increment;            \
            } while (--n > 0);                   \
        }                                        \
    }

// 2-times unrolled loop
#define DUFFS_LOOP2(pixel_copy_increment, width) \
    {                                            \
        int n = (width + 1) / 2;                 \
        switch (width & 1) {                     \
        case 0:                                  \
            do {                                 \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 1:                              \
                pixel_copy_increment;            \
            } while (--n > 0);                   \
        }                                        \
    }

// Use the 4-times version of the loop by default
#define DUFFS_LOOP(pixel_copy_increment, width) \
    DUFFS_LOOP4(pixel_copy_increment, width)
// Use the 8-times version of the loop for simple routines
#define DUFFS_LOOP_TRIVIAL(pixel_copy_increment, width) \
    DUFFS_LOOP8(pixel_copy_increment, width)

// Special version of Duff's device for even more optimization
#define DUFFS_LOOP_124(pixel_copy_increment1,        \
                       pixel_copy_increment2,        \
                       pixel_copy_increment4, width) \
    {                                                \
        int n = width;                               \
        if (n & 1) {                                 \
            pixel_copy_increment1;                   \
            n -= 1;                                  \
        }                                            \
        if (n & 2) {                                 \
            pixel_copy_increment2;                   \
            n -= 2;                                  \
        }                                            \
        if (n & 4) {                                 \
            pixel_copy_increment4;                   \
            n -= 4;                                  \
        }                                            \
        if (n) {                                     \
            n /= 8;                                  \
            do {                                     \
                pixel_copy_increment4;               \
                pixel_copy_increment4;               \
            } while (--n > 0);                       \
        }                                            \
    }

#else

// Don't use Duff's device to unroll loops
#define DUFFS_LOOP(pixel_copy_increment, width) \
    DUFFS_LOOP1(pixel_copy_increment, width)
#define DUFFS_LOOP_TRIVIAL(pixel_copy_increment, width) \
    DUFFS_LOOP1(pixel_copy_increment, width)
#define DUFFS_LOOP8(pixel_copy_increment, width) \
    DUFFS_LOOP1(pixel_copy_increment, width)
#define DUFFS_LOOP4(pixel_copy_increment, width) \
    DUFFS_LOOP1(pixel_copy_increment, width)
#define DUFFS_LOOP2(pixel_copy_increment, width) \
    DUFFS_LOOP1(pixel_copy_increment, width)
#define DUFFS_LOOP_124(pixel_copy_increment1,        \
                       pixel_copy_increment2,        \
                       pixel_copy_increment4, width) \
    DUFFS_LOOP1(pixel_copy_increment1, width)

#endif // USE_DUFFS_LOOP

#if defined(_MSC_VER) && (_MSC_VER >= 600)
#pragma warning(disable : 4244) // '=': conversion from 'X' to 'Y', possible loss of data
#endif

#endif // SDL_blit_h_
