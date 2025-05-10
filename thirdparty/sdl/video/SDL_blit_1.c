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

#ifdef SDL_HAVE_BLIT_1

#include "SDL_surface_c.h"
#include "SDL_sysvideo.h"

// Functions to blit from 8-bit surfaces to other surfaces

static void Blit1to1(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint8 *src, *map, *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = info->src;
    srcskip = info->src_skip;
    dst = info->dst;
    dstskip = info->dst_skip;
    map = info->table;

    while (height--) {
#ifdef USE_DUFFS_LOOP
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
            {
              *dst = map[*src];
            }
            dst++;
            src++;
        , width);
        /* *INDENT-ON* */ // clang-format on
#else
        for (c = width; c; --c) {
            *dst = map[*src];
            dst++;
            src++;
        }
#endif
        src += srcskip;
        dst += dstskip;
    }
}

// This is now endian dependent
#ifndef USE_DUFFS_LOOP
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
#define HI 1
#define LO 0
#else // ( SDL_BYTEORDER == SDL_BIG_ENDIAN )
#define HI 0
#define LO 1
#endif
#endif
static void Blit1to2(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint8 *src, *dst;
    Uint16 *map;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = info->src;
    srcskip = info->src_skip;
    dst = info->dst;
    dstskip = info->dst_skip;
    map = (Uint16 *)info->table;

#ifdef USE_DUFFS_LOOP
    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
        {
            *(Uint16 *)dst = map[*src++];
            dst += 2;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
#else
    // Memory align at 4-byte boundary, if necessary
    if ((long)dst & 0x03) {
        // Don't do anything if width is 0
        if (width == 0) {
            return;
        }
        --width;

        while (height--) {
            // Perform copy alignment
            *(Uint16 *)dst = map[*src++];
            dst += 2;

            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                *(Uint16 *)dst = map[*src++];
                dst += 2;
                SDL_FALLTHROUGH;
            case 2:
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
                break;
            case 1:
                *(Uint16 *)dst = map[*src++];
                dst += 2;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    } else {
        while (height--) {
            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                *(Uint16 *)dst = map[*src++];
                dst += 2;
                SDL_FALLTHROUGH;
            case 2:
                *(Uint32 *)dst = (map[src[HI]] << 16) | (map[src[LO]]);
                src += 2;
                dst += 4;
                break;
            case 1:
                *(Uint16 *)dst = map[*src++];
                dst += 2;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    }
#endif // USE_DUFFS_LOOP
}

static void Blit1to3(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int o;
    int width, height;
    Uint8 *src, *map, *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = info->src;
    srcskip = info->src_skip;
    dst = info->dst;
    dstskip = info->dst_skip;
    map = info->table;

    while (height--) {
#ifdef USE_DUFFS_LOOP
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
            {
                o = *src * 4;
                dst[0] = map[o++];
                dst[1] = map[o++];
                dst[2] = map[o++];
            }
            src++;
            dst += 3;
        , width);
        /* *INDENT-ON* */ // clang-format on
#else
        for (c = width; c; --c) {
            o = *src * 4;
            dst[0] = map[o++];
            dst[1] = map[o++];
            dst[2] = map[o++];
            src++;
            dst += 3;
        }
#endif // USE_DUFFS_LOOP
        src += srcskip;
        dst += dstskip;
    }
}

static void Blit1to4(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint8 *src;
    Uint32 *map, *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = info->src;
    srcskip = info->src_skip;
    dst = (Uint32 *)info->dst;
    dstskip = info->dst_skip / 4;
    map = (Uint32 *)info->table;

    while (height--) {
#ifdef USE_DUFFS_LOOP
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
            *dst++ = map[*src++];
        , width);
        /* *INDENT-ON* */ // clang-format on
#else
        for (c = width / 4; c; --c) {
            *dst++ = map[*src++];
            *dst++ = map[*src++];
            *dst++ = map[*src++];
            *dst++ = map[*src++];
        }
        switch (width & 3) {
        case 3:
            *dst++ = map[*src++];
            SDL_FALLTHROUGH;
        case 2:
            *dst++ = map[*src++];
            SDL_FALLTHROUGH;
        case 1:
            *dst++ = map[*src++];
        }
#endif // USE_DUFFS_LOOP
        src += srcskip;
        dst += dstskip;
    }
}

static void Blit1to1Key(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    Uint32 ckey = info->colorkey;

    if (palmap) {
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                if ( *src != ckey ) {
                  *dst = palmap[*src];
                }
                dst++;
                src++;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    } else {
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                if ( *src != ckey ) {
                  *dst = *src;
                }
                dst++;
                src++;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    }
}

static void Blit1to2Key(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip;
    Uint16 *palmap = (Uint16 *)info->table;
    Uint32 ckey = info->colorkey;

    // Set up some basic variables
    dstskip /= 2;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
        {
            if ( *src != ckey ) {
                *dstp=palmap[*src];
            }
            src++;
            dstp++;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dstp += dstskip;
    }
}

static void Blit1to3Key(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    Uint32 ckey = info->colorkey;
    int o;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            if ( *src != ckey ) {
                o = *src * 4;
                dst[0] = palmap[o++];
                dst[1] = palmap[o++];
                dst[2] = palmap[o++];
            }
            src++;
            dst += 3;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

static void Blit1to4Key(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip;
    Uint32 *palmap = (Uint32 *)info->table;
    Uint32 ckey = info->colorkey;

    // Set up some basic variables
    dstskip /= 4;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
        {
            if ( *src != ckey ) {
                *dstp = palmap[*src];
            }
            src++;
            dstp++;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dstp += dstskip;
    }
}

static void Blit1toNAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    const SDL_Color *srcpal = info->src_pal->colors;
    int dstbpp;
    Uint32 pixelvalue;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB, dA;
    const unsigned A = info->a;

    // Set up some basic variables
    dstbpp = dstfmt->bytes_per_pixel;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            sR = srcpal[*src].r;
            sG = srcpal[*src].g;
            sB = srcpal[*src].b;
            sA = (srcpal[*src].a * A) / 255;
            DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixelvalue, dR, dG, dB, dA);
            ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
            src++;
            dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

static void Blit1toNAlphaKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    const SDL_Color *srcpal = info->src_pal->colors;
    Uint32 ckey = info->colorkey;
    int dstbpp;
    Uint32 pixelvalue;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB, dA;
    const unsigned A = info->a;

    // Set up some basic variables
    dstbpp = dstfmt->bytes_per_pixel;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            if ( *src != ckey ) {
                sR = srcpal[*src].r;
                sG = srcpal[*src].g;
                sB = srcpal[*src].b;
                sA = (srcpal[*src].a * A) / 255;
                DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixelvalue, dR, dG, dB, dA);
                ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
            }
            src++;
            dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

static const SDL_BlitFunc one_blit[] = {
    (SDL_BlitFunc)NULL, Blit1to1, Blit1to2, Blit1to3, Blit1to4
};

static const SDL_BlitFunc one_blitkey[] = {
    (SDL_BlitFunc)NULL, Blit1to1Key, Blit1to2Key, Blit1to3Key, Blit1to4Key
};

SDL_BlitFunc SDL_CalculateBlit1(SDL_Surface *surface)
{
    int which;

    if (SDL_BITSPERPIXEL(surface->map.info.dst_fmt->format) < 8) {
        which = 0;
    } else {
        which = SDL_BYTESPERPIXEL(surface->map.info.dst_fmt->format);
    }

    switch (surface->map.info.flags & ~SDL_COPY_RLE_MASK) {
    case 0:
        if (which < SDL_arraysize(one_blit)) {
            return one_blit[which];
        }
        break;

    case SDL_COPY_COLORKEY:
        if (which < SDL_arraysize(one_blitkey)) {
            return one_blitkey[which];
        }
        break;

    case SDL_COPY_COLORKEY | SDL_COPY_BLEND:  // this is not super-robust but handles a specific case we found sdl12-compat.
        if (surface->map.info.a == 255) {
            if (which < SDL_arraysize(one_blitkey)) {
                return one_blitkey[which];
            }
        } else {
            return which >= 2 ? Blit1toNAlphaKey : (SDL_BlitFunc)NULL;
        }
        break;

    case SDL_COPY_BLEND:
    case SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        /* Supporting 8bpp->8bpp alpha is doable but requires lots of
           tables which consume space and takes time to precompute,
           so is better left to the user */
        return which >= 2 ? Blit1toNAlpha : (SDL_BlitFunc)NULL;

    case SDL_COPY_COLORKEY | SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        return which >= 2 ? Blit1toNAlphaKey : (SDL_BlitFunc)NULL;
    }
    return (SDL_BlitFunc)NULL;
}

#endif // SDL_HAVE_BLIT_1
