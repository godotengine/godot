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

#ifdef SDL_HAVE_BLIT_A

#include "SDL_pixels_c.h"
#include "SDL_surface_c.h"

// Functions to perform alpha blended blitting

// N->1 blending with per-surface alpha
static void BlitNto1SurfaceAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_Color *dstpal = info->dst_pal->colors;
    int srcbpp = srcfmt->bytes_per_pixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB;
    const unsigned A = info->a;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        dR = dstpal[*dst].r;
        dG = dstpal[*dst].g;
        dB = dstpal[*dst].b;
        ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB);
        dR &= 0xff;
        dG &= 0xff;
        dB &= 0xff;
        // Pack RGB into 8bit pixel
        if ( palmap == NULL ) {
            *dst = (Uint8)(((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0)));
        } else {
            *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

// N->1 blending with pixel alpha
static void BlitNto1PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_Color *dstpal = info->dst_pal->colors;
    int srcbpp = srcfmt->bytes_per_pixel;
    Uint32 Pixel;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
        DISEMBLE_RGBA(src,srcbpp,srcfmt,Pixel,sR,sG,sB,sA);
        dR = dstpal[*dst].r;
        dG = dstpal[*dst].g;
        dB = dstpal[*dst].b;
        ALPHA_BLEND_RGB(sR, sG, sB, sA, dR, dG, dB);
        dR &= 0xff;
        dG &= 0xff;
        dB &= 0xff;
        // Pack RGB into 8bit pixel
        if ( palmap == NULL ) {
            *dst = (Uint8)(((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0)));
        } else {
            *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

// colorkeyed N->1 blending with per-surface alpha
static void BlitNto1SurfaceAlphaKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_Color *dstpal = info->dst_pal->colors;
    int srcbpp = srcfmt->bytes_per_pixel;
    Uint32 ckey = info->colorkey;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB;
    const unsigned A = info->a;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        if ( Pixel != ckey ) {
            dR = dstpal[*dst].r;
            dG = dstpal[*dst].g;
            dB = dstpal[*dst].b;
            ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB);
            dR &= 0xff;
            dG &= 0xff;
            dB &= 0xff;
            // Pack RGB into 8bit pixel
            if ( palmap == NULL ) {
                *dst = (Uint8)(((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0)));
            } else {
                *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
            }
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

#ifdef SDL_SSE2_INTRINSICS

static void SDL_TARGETING("sse2") Blit888to888SurfaceAlphaSSE2(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 alpha = info->a;

    const __m128i alpha_fill_mask = _mm_set1_epi32((int)0xff000000);
    const __m128i srcA = _mm_set1_epi16(alpha);

    while (height--) {
        int i = 0;

        for (; i + 4 <= width; i += 4) {
            // Load 4 src pixels
            __m128i src128 = _mm_loadu_si128((__m128i *)src);

            // Load 4 dst pixels
            __m128i dst128 = _mm_loadu_si128((__m128i *)dst);

            __m128i src_lo = _mm_unpacklo_epi8(src128, _mm_setzero_si128());
            __m128i src_hi = _mm_unpackhi_epi8(src128, _mm_setzero_si128());

            __m128i dst_lo = _mm_unpacklo_epi8(dst128, _mm_setzero_si128());
            __m128i dst_hi = _mm_unpackhi_epi8(dst128, _mm_setzero_si128());

            // dst = ((src - dst) * srcA) + ((dst << 8) - dst)
            dst_lo = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(src_lo, dst_lo), srcA),
                                      _mm_sub_epi16(_mm_slli_epi16(dst_lo, 8), dst_lo));
            dst_hi = _mm_add_epi16(_mm_mullo_epi16(_mm_sub_epi16(src_hi, dst_hi), srcA),
                                      _mm_sub_epi16(_mm_slli_epi16(dst_hi, 8), dst_hi));

            // dst += 0x1U (use 0x80 to round instead of floor)
            dst_lo = _mm_add_epi16(dst_lo, _mm_set1_epi16(1));
            dst_hi = _mm_add_epi16(dst_hi, _mm_set1_epi16(1));

            // dst = (dst + (dst >> 8)) >> 8
            dst_lo = _mm_srli_epi16(_mm_add_epi16(dst_lo, _mm_srli_epi16(dst_lo, 8)), 8);
            dst_hi = _mm_srli_epi16(_mm_add_epi16(dst_hi, _mm_srli_epi16(dst_hi, 8)), 8);

            dst128 = _mm_packus_epi16(dst_lo, dst_hi);

            // Set the alpha channels of dst to 255
            dst128 = _mm_or_si128(dst128, alpha_fill_mask);

            _mm_storeu_si128((__m128i *)dst, dst128);

            src += 16;
            dst += 16;
        }

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32 = *(Uint32 *)dst;

            FACTOR_BLEND_8888(src32, dst32, alpha);

            *dst = dst32 | 0xff000000;

            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#endif

// fast RGB888->(A)RGB888 blending with surface alpha=128 special case
static void BlitRGBtoRGBSurfaceAlpha128(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP({
            Uint32 s = *srcp++;
            Uint32 d = *dstp;
            *dstp++ = ((((s & 0x00fefefe) + (d & 0x00fefefe)) >> 1)
                   + (s & d & 0x00010101)) | 0xff000000;
        }, width);
        /* *INDENT-ON* */ // clang-format on
        srcp += srcskip;
        dstp += dstskip;
    }
}

// fast RGB888->(A)RGB888 blending with surface alpha
static void BlitRGBtoRGBSurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        BlitRGBtoRGBSurfaceAlpha128(info);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint32 *srcp = (Uint32 *)info->src;
        int srcskip = info->src_skip >> 2;
        Uint32 *dstp = (Uint32 *)info->dst;
        int dstskip = info->dst_skip >> 2;
        Uint32 s;
        Uint32 d;

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP({
                s = *srcp;
                d = *dstp;

                FACTOR_BLEND_8888(s, d, alpha);

                *dstp = d | 0xff000000;
                ++srcp;
                ++dstp;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

// 16bpp special case for per-surface alpha=50%: blend 2 pixels in parallel

// blend a single 16 bit pixel at 50%
#define BLEND16_50(d, s, mask) \
    ((((s & mask) + (d & mask)) >> 1) + (s & d & (~mask & 0xffff)))

// blend two 16 bit pixels at 50%
#define BLEND2x16_50(d, s, mask) \
    (((s & (mask | mask << 16)) >> 1) + ((d & (mask | mask << 16)) >> 1) + (s & d & (~(mask | mask << 16))))

static void Blit16to16SurfaceAlpha128(SDL_BlitInfo *info, Uint16 mask)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint16 *srcp = (Uint16 *)info->src;
    int srcskip = info->src_skip >> 1;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        if (((uintptr_t)srcp ^ (uintptr_t)dstp) & 2) {
            /*
             * Source and destination not aligned, pipeline it.
             * This is mostly a win for big blits but no loss for
             * small ones
             */
            Uint32 prev_sw;
            int w = width;

            // handle odd destination
            if ((uintptr_t)dstp & 2) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                dstp++;
                srcp++;
                w--;
            }
            srcp++; // srcp is now 32-bit aligned

            // bootstrap pipeline with first halfword
            prev_sw = ((Uint32 *)srcp)[-1];

            while (w > 1) {
                Uint32 sw, dw, s;
                sw = *(Uint32 *)srcp;
                dw = *(Uint32 *)dstp;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                s = (prev_sw << 16) + (sw >> 16);
#else
                s = (prev_sw >> 16) + (sw << 16);
#endif
                prev_sw = sw;
                *(Uint32 *)dstp = BLEND2x16_50(dw, s, mask);
                dstp += 2;
                srcp += 2;
                w -= 2;
            }

            // final pixel if any
            if (w) {
                Uint16 d = *dstp, s;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                s = (Uint16)prev_sw;
#else
                s = (Uint16)(prev_sw >> 16);
#endif
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
            }
            srcp += srcskip - 1;
            dstp += dstskip;
        } else {
            // source and destination are aligned
            int w = width;

            // first odd pixel?
            if ((uintptr_t)srcp & 2) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
                w--;
            }
            // srcp and dstp are now 32-bit aligned

            while (w > 1) {
                Uint32 sw = *(Uint32 *)srcp;
                Uint32 dw = *(Uint32 *)dstp;
                *(Uint32 *)dstp = BLEND2x16_50(dw, sw, mask);
                srcp += 2;
                dstp += 2;
                w -= 2;
            }

            // last odd pixel?
            if (w) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
            }
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

#ifdef SDL_MMX_INTRINSICS

// fast RGB565->RGB565 blending with surface alpha
static void SDL_TARGETING("mmx") Blit565to565SurfaceAlphaMMX(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xf7de);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        Uint32 s, d;

#ifdef USE_DUFFS_LOOP
        __m64 src1, dst1, src2, dst2, gmask, bmask, mm_res, mm_alpha;

        alpha &= ~(1 + 2 + 4);             // cut alpha to get the exact same behaviour
        mm_alpha = _mm_set_pi32(0, alpha); // 0000000A -> mm_alpha
        alpha >>= 3;                       // downscale alpha to 5 bits

        mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); // 00000A0A -> mm_alpha
        mm_alpha = _mm_unpacklo_pi32(mm_alpha, mm_alpha); // 0A0A0A0A -> mm_alpha
        /* position alpha to allow for mullo and mulhi on diff channels
           to reduce the number of operations */
        mm_alpha = _mm_slli_si64(mm_alpha, 3);

        // Setup the 565 color channel masks
        gmask = _mm_set_pi32(0x07E007E0, 0x07E007E0); // MASKGREEN -> gmask
        bmask = _mm_set_pi32(0x001F001F, 0x001F001F); // MASKBLUE -> bmask
#endif

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_124(
            {
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                src1 = *(__m64*)srcp; // 4 src pixels -> src1
                dst1 = *(__m64*)dstp; // 4 dst pixels -> dst1

                // red
                src2 = src1;
                src2 = _mm_srli_pi16(src2, 11); // src2 >> 11 -> src2 [000r 000r 000r 000r]

                dst2 = dst1;
                dst2 = _mm_srli_pi16(dst2, 11); // dst2 >> 11 -> dst2 [000r 000r 000r 000r]

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); // src2 >> 11 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2
                dst2 = _mm_slli_pi16(dst2, 11); // dst2 << 11 -> dst2

                mm_res = dst2; // RED -> mm_res

                // green -- process the bits in place
                src2 = src1;
                src2 = _mm_and_si64(src2, gmask); // src & MASKGREEN -> src2

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, gmask); // dst & MASKGREEN -> dst2

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); // src2 << 5 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2

                mm_res = _mm_or_si64(mm_res, dst2); // RED | GREEN -> mm_res

                // blue
                src2 = src1;
                src2 = _mm_and_si64(src2, bmask); // src & MASKBLUE -> src2[000b 000b 000b 000b]

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, bmask); // dst & MASKBLUE -> dst2[000b 000b 000b 000b]

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); // src2 >> 11 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2
                dst2 = _mm_and_si64(dst2, bmask); // dst2 & MASKBLUE -> dst2

                mm_res = _mm_or_si64(mm_res, dst2); // RED | GREEN | BLUE -> mm_res

                *(__m64*)dstp = mm_res; // mm_res -> 4 dst pixels

                srcp += 4;
                dstp += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            srcp += srcskip;
            dstp += dstskip;
        }
        _mm_empty();
    }
}

// fast RGB555->RGB555 blending with surface alpha
static void SDL_TARGETING("mmx") Blit555to555SurfaceAlphaMMX(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xfbde);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        Uint32 s, d;

#ifdef USE_DUFFS_LOOP
        __m64 src1, dst1, src2, dst2, rmask, gmask, bmask, mm_res, mm_alpha;

        alpha &= ~(1 + 2 + 4);             // cut alpha to get the exact same behaviour
        mm_alpha = _mm_set_pi32(0, alpha); // 0000000A -> mm_alpha
        alpha >>= 3;                       // downscale alpha to 5 bits

        mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); // 00000A0A -> mm_alpha
        mm_alpha = _mm_unpacklo_pi32(mm_alpha, mm_alpha); // 0A0A0A0A -> mm_alpha
        /* position alpha to allow for mullo and mulhi on diff channels
           to reduce the number of operations */
        mm_alpha = _mm_slli_si64(mm_alpha, 3);

        // Setup the 555 color channel masks
        rmask = _mm_set_pi32(0x7C007C00, 0x7C007C00); // MASKRED -> rmask
        gmask = _mm_set_pi32(0x03E003E0, 0x03E003E0); // MASKGREEN -> gmask
        bmask = _mm_set_pi32(0x001F001F, 0x001F001F); // MASKBLUE -> bmask
#endif
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_124(
            {
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
                    s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                src1 = *(__m64*)srcp; // 4 src pixels -> src1
                dst1 = *(__m64*)dstp; // 4 dst pixels -> dst1

                // red -- process the bits in place
                src2 = src1;
                src2 = _mm_and_si64(src2, rmask); // src & MASKRED -> src2

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, rmask); // dst & MASKRED -> dst2

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); // src2 << 5 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2
                dst2 = _mm_and_si64(dst2, rmask); // dst2 & MASKRED -> dst2

                mm_res = dst2; // RED -> mm_res

                // green -- process the bits in place
                src2 = src1;
                src2 = _mm_and_si64(src2, gmask); // src & MASKGREEN -> src2

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, gmask); // dst & MASKGREEN -> dst2

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); // src2 << 5 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2

                mm_res = _mm_or_si64(mm_res, dst2); // RED | GREEN -> mm_res

                // blue
                src2 = src1; // src -> src2
                src2 = _mm_and_si64(src2, bmask); // src & MASKBLUE -> src2[000b 000b 000b 000b]

                dst2 = dst1; // dst -> dst2
                dst2 = _mm_and_si64(dst2, bmask); // dst & MASKBLUE -> dst2[000b 000b 000b 000b]

                // blend
                src2 = _mm_sub_pi16(src2, dst2);// src - dst -> src2
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); // src2 >> 11 -> src2
                dst2 = _mm_add_pi16(src2, dst2); // src2 + dst2 -> dst2
                dst2 = _mm_and_si64(dst2, bmask); // dst2 & MASKBLUE -> dst2

                mm_res = _mm_or_si64(mm_res, dst2); // RED | GREEN | BLUE -> mm_res

                *(__m64*)dstp = mm_res; // mm_res -> 4 dst pixels

                srcp += 4;
                dstp += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            srcp += srcskip;
            dstp += dstskip;
        }
        _mm_empty();
    }
}

#endif // SDL_MMX_INTRINSICS

// fast RGB565->RGB565 blending with surface alpha
static void Blit565to565SurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xf7de);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        alpha >>= 3; // downscale alpha to 5 bits

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP({
                Uint32 s = *srcp++;
                Uint32 d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            }, width);
            /* *INDENT-ON* */ // clang-format on
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

// fast RGB555->RGB555 blending with surface alpha
static void Blit555to555SurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a; // downscale alpha to 5 bits
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xfbde);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        alpha >>= 3; // downscale alpha to 5 bits

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP({
                Uint32 s = *srcp++;
                Uint32 d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            }, width);
            /* *INDENT-ON* */ // clang-format on
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

// fast ARGB8888->RGB565 blending with pixel alpha
static void BlitARGBto565PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP({
        Uint32 s = *srcp;
        unsigned alpha = s >> 27; // downscale alpha to 5 bits
        /* Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. */
        if (alpha) {
          if (alpha == (SDL_ALPHA_OPAQUE >> 3)) {
            *dstp = (Uint16)((s >> 8 & 0xf800) + (s >> 5 & 0x7e0) + (s >> 3  & 0x1f));
          } else {
            Uint32 d = *dstp;
            /*
             * convert source and destination to G0RAB65565
             * and blend all components at the same time
             */
            s = ((s & 0xfc00) << 11) + (s >> 8 & 0xf800) + (s >> 3 & 0x1f);
            d = (d | d << 16) & 0x07e0f81f;
            d += (s - d) * alpha >> 5;
            d &= 0x07e0f81f;
            *dstp = (Uint16)(d | d >> 16);
          }
        }
        srcp++;
        dstp++;
        }, width);
        /* *INDENT-ON* */ // clang-format on
        srcp += srcskip;
        dstp += dstskip;
    }
}

// fast ARGB8888->RGB555 blending with pixel alpha
static void BlitARGBto555PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP({
        unsigned alpha;
        Uint32 s = *srcp;
        alpha = s >> 27; // downscale alpha to 5 bits
        /* Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. */
        if (alpha) {
          if (alpha == (SDL_ALPHA_OPAQUE >> 3)) {
            *dstp = (Uint16)((s >> 9 & 0x7c00) + (s >> 6 & 0x3e0) + (s >> 3  & 0x1f));
          } else {
            Uint32 d = *dstp;
            /*
             * convert source and destination to G0RAB55555
             * and blend all components at the same time
             */
            s = ((s & 0xf800) << 10) + (s >> 9 & 0x7c00) + (s >> 3 & 0x1f);
            d = (d | d << 16) & 0x03e07c1f;
            d += (s - d) * alpha >> 5;
            d &= 0x03e07c1f;
            *dstp = (Uint16)(d | d >> 16);
          }
        }
        srcp++;
        dstp++;
        }, width);
        /* *INDENT-ON* */ // clang-format on
        srcp += srcskip;
        dstp += dstskip;
    }
}

// General (slow) N->N blending with per-surface alpha
static void BlitNtoNSurfaceAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    int dstbpp = dstfmt->bytes_per_pixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB, dA;
    const unsigned sA = info->a;

    if (sA) {
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
        ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
        ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    }
}

// General (slow) colorkeyed N->N blending with per-surface alpha
static void BlitNtoNSurfaceAlphaKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    Uint32 ckey = info->colorkey;
    int srcbpp = srcfmt->bytes_per_pixel;
    int dstbpp = dstfmt->bytes_per_pixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB, dA;
    const unsigned sA = info->a;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
        RETRIEVE_RGB_PIXEL(src, srcbpp, Pixel);
        if (sA && Pixel != ckey) {
            RGB_FROM_PIXEL(Pixel, srcfmt, sR, sG, sB);
            DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
            ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        }
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

// Fast 32-bit RGBA->RGBA blending with pixel alpha
static void Blit8888to8888PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;

    while (height--) {
        int i = 0;

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32 = *(Uint32 *)dst;
            ALPHA_BLEND_8888(src32, dst32, srcfmt);
            *(Uint32 *)dst = dst32;
            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

// Fast 32-bit RGBA->RGB(A) blending with pixel alpha and src swizzling
static void Blit8888to8888PixelAlphaSwizzle(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = !dstfmt->Amask;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    while (height--) {
        int i = 0;

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32 = *(Uint32 *)dst;
            ALPHA_BLEND_SWIZZLE_8888(src32, dst32, srcfmt, dstfmt);
            if (fill_alpha) {
                dst32 |= dstAmask;
            }
            *(Uint32 *)dst = dst32;
            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#ifdef SDL_SSE4_1_INTRINSICS

static void SDL_TARGETING("sse4.1") Blit8888to8888PixelAlphaSwizzleSSE41(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = !dstfmt->Amask;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const __m128i mask_offsets = _mm_set_epi8(
        12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    const __m128i convert_mask = _mm_add_epi32(
        _mm_set1_epi32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift)),
        mask_offsets);

    const __m128i alpha_splat_mask = _mm_add_epi8(_mm_set1_epi8(srcfmt->Ashift >> 3), mask_offsets);
    const __m128i alpha_fill_mask = _mm_set1_epi32((int)dstAmask);

    while (height--) {
        int i = 0;

        for (; i + 4 <= width; i += 4) {
            // Load 4 src pixels
            __m128i src128 = _mm_loadu_si128((__m128i *)src);

            // Load 4 dst pixels
            __m128i dst128 = _mm_loadu_si128((__m128i *)dst);

            // Extract the alpha from each pixel and splat it into all the channels
            __m128i srcA = _mm_shuffle_epi8(src128, alpha_splat_mask);

            // Convert to dst format
            src128 = _mm_shuffle_epi8(src128, convert_mask);

            // Set the alpha channels of src to 255
            src128 = _mm_or_si128(src128, alpha_fill_mask);

            // Duplicate each 8-bit alpha value into both bytes of 16-bit lanes
            __m128i srca_lo = _mm_unpacklo_epi8(srcA, srcA);
            __m128i srca_hi = _mm_unpackhi_epi8(srcA, srcA);

            // Calculate 255-srcA in every second 8-bit lane (255-srcA = srcA^0xff)
            srca_lo = _mm_xor_si128(srca_lo, _mm_set1_epi16(0xff00));
            srca_hi = _mm_xor_si128(srca_hi, _mm_set1_epi16(0xff00));

            // maddubs expects second argument to be signed, so subtract 128
            src128 = _mm_sub_epi8(src128, _mm_set1_epi8((Uint8)128));
            dst128 = _mm_sub_epi8(dst128, _mm_set1_epi8((Uint8)128));

            // dst = srcA*(src-128) + (255-srcA)*(dst-128) = srcA*src + (255-srcA)*dst - 128*255
            __m128i dst_lo = _mm_maddubs_epi16(srca_lo, _mm_unpacklo_epi8(src128, dst128));
            __m128i dst_hi = _mm_maddubs_epi16(srca_hi, _mm_unpackhi_epi8(src128, dst128));

            // dst += 0x1U (use 0x80 to round instead of floor) + 128*255 (to fix maddubs result)
            dst_lo = _mm_add_epi16(dst_lo, _mm_set1_epi16(1 + 128*255));
            dst_hi = _mm_add_epi16(dst_hi, _mm_set1_epi16(1 + 128*255));

            // dst = (dst + (dst >> 8)) >> 8 = (dst * 257) >> 16
            dst_lo = _mm_mulhi_epu16(dst_lo, _mm_set1_epi16(257));
            dst_hi = _mm_mulhi_epu16(dst_hi, _mm_set1_epi16(257));

            // Blend the pixels together and save the result
            dst128 = _mm_packus_epi16(dst_lo, dst_hi);
            if (fill_alpha) {
                dst128 = _mm_or_si128(dst128, alpha_fill_mask);
            }
            _mm_storeu_si128((__m128i *)dst, dst128);

            src += 16;
            dst += 16;
        }

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32 = *(Uint32 *)dst;
            ALPHA_BLEND_SWIZZLE_8888(src32, dst32, srcfmt, dstfmt);
            if (fill_alpha) {
                dst32 |= dstAmask;
            }
            *(Uint32 *)dst = dst32;
            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#endif

#ifdef SDL_AVX2_INTRINSICS

static void SDL_TARGETING("avx2") Blit8888to8888PixelAlphaSwizzleAVX2(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = !dstfmt->Amask;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const __m256i mask_offsets = _mm256_set_epi8(
        28, 28, 28, 28, 24, 24, 24, 24, 20, 20, 20, 20, 16, 16, 16, 16, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    const __m256i convert_mask = _mm256_add_epi32(
        _mm256_set1_epi32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift)),
        mask_offsets);

    const __m256i alpha_splat_mask = _mm256_add_epi8(_mm256_set1_epi8(srcfmt->Ashift >> 3), mask_offsets);
    const __m256i alpha_fill_mask = _mm256_set1_epi32((int)dstAmask);

    while (height--) {
        int i = 0;

        for (; i + 8 <= width; i += 8) {
            // Load 8 src pixels
            __m256i src256 = _mm256_loadu_si256((__m256i *)src);

            // Load 8 dst pixels
            __m256i dst256 = _mm256_loadu_si256((__m256i *)dst);

            // Extract the alpha from each pixel and splat it into all the channels
            __m256i srcA = _mm256_shuffle_epi8(src256, alpha_splat_mask);

            // Convert to dst format
            src256 = _mm256_shuffle_epi8(src256, convert_mask);

            // Set the alpha channels of src to 255
            src256 = _mm256_or_si256(src256, alpha_fill_mask);

            // Duplicate each 8-bit alpha value into both bytes of 16-bit lanes
            __m256i alpha_lo = _mm256_unpacklo_epi8(srcA, srcA);
            __m256i alpha_hi = _mm256_unpackhi_epi8(srcA, srcA);

            // Calculate 255-srcA in every second 8-bit lane (255-srcA = srcA^0xff)
            alpha_lo = _mm256_xor_si256(alpha_lo, _mm256_set1_epi16(0xff00));
            alpha_hi = _mm256_xor_si256(alpha_hi, _mm256_set1_epi16(0xff00));

            // maddubs expects second argument to be signed, so subtract 128
            src256 = _mm256_sub_epi8(src256, _mm256_set1_epi8((Uint8)128));
            dst256 = _mm256_sub_epi8(dst256, _mm256_set1_epi8((Uint8)128));

            // dst = srcA*(src-128) + (255-srcA)*(dst-128) = srcA*src + (255-srcA)*dst - 128*255
            __m256i dst_lo = _mm256_maddubs_epi16(alpha_lo, _mm256_unpacklo_epi8(src256, dst256));
            __m256i dst_hi = _mm256_maddubs_epi16(alpha_hi, _mm256_unpackhi_epi8(src256, dst256));

            // dst += 0x1U (use 0x80 to round instead of floor) + 128*255 (to fix maddubs result)
            dst_lo = _mm256_add_epi16(dst_lo, _mm256_set1_epi16(1 + 128*255));
            dst_hi = _mm256_add_epi16(dst_hi, _mm256_set1_epi16(1 + 128*255));

            // dst = (dst + (dst >> 8)) >> 8 = (dst * 257) >> 16
            dst_lo = _mm256_mulhi_epu16(dst_lo, _mm256_set1_epi16(257));
            dst_hi = _mm256_mulhi_epu16(dst_hi, _mm256_set1_epi16(257));

            // Blend the pixels together and save the result
            dst256 = _mm256_packus_epi16(dst_lo, dst_hi);
            if (fill_alpha) {
                dst256 = _mm256_or_si256(dst256, alpha_fill_mask);
            }
            _mm256_storeu_si256((__m256i *)dst, dst256);

            src += 32;
            dst += 32;
        }

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32 = *(Uint32 *)dst;
            ALPHA_BLEND_SWIZZLE_8888(src32, dst32, srcfmt, dstfmt);
            if (fill_alpha) {
                dst32 |= dstAmask;
            }
            *(Uint32 *)dst = dst32;
            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#endif

#if defined(SDL_NEON_INTRINSICS) && (__ARM_ARCH >= 8)

static void Blit8888to8888PixelAlphaSwizzleNEON(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = !dstfmt->Amask;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const uint8x16_t mask_offsets = vreinterpretq_u8_u64(vcombine_u64(
        vcreate_u64(0x0404040400000000), vcreate_u64(0x0c0c0c0c08080808)));

    const uint8x16_t convert_mask = vreinterpretq_u8_u32(vaddq_u32(
        vreinterpretq_u32_u8(mask_offsets),
        vdupq_n_u32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift))));

    const uint8x16_t alpha_splat_mask = vaddq_u8(vdupq_n_u8(srcfmt->Ashift >> 3), mask_offsets);
    const uint8x16_t alpha_fill_mask = vreinterpretq_u8_u32(vdupq_n_u32(dstAmask));

    while (height--) {
        int i = 0;

        for (; i + 4 <= width; i += 4) {
            // Load 4 src pixels
            uint8x16_t src128 = vld1q_u8(src);

            // Load 4 dst pixels
            uint8x16_t dst128 = vld1q_u8(dst);

            // Extract the alpha from each pixel and splat it into all the channels
            uint8x16_t srcA = vqtbl1q_u8(src128, alpha_splat_mask);

            // Convert to dst format
            src128 = vqtbl1q_u8(src128, convert_mask);

            // Set the alpha channels of src to 255
            src128 = vorrq_u8(src128, alpha_fill_mask);

            // 255 - srcA = ~srcA
            uint8x16_t srcInvA = vmvnq_u8(srcA);

            // Result initialized with 1, this is for truncated divide later
            uint16x8_t res_lo = vdupq_n_u16(1);
            uint16x8_t res_hi = vdupq_n_u16(1);

            // res = alpha * src + (255 - alpha) * dst
            res_lo = vmlal_u8(res_lo, vget_low_u8(srcA),    vget_low_u8(src128));
            res_lo = vmlal_u8(res_lo, vget_low_u8(srcInvA), vget_low_u8(dst128));
            res_hi = vmlal_high_u8(res_hi, srcA,    src128);
            res_hi = vmlal_high_u8(res_hi, srcInvA, dst128);

            // Now result has +1 already added for truncated division
            // dst = (res + (res >> 8)) >> 8
            uint8x8_t temp;
            temp   = vaddhn_u16(res_lo, vshrq_n_u16(res_lo, 8));
            dst128 = vaddhn_high_u16(temp, res_hi, vshrq_n_u16(res_hi, 8));

            // For rounded division remove the constant 1 and change first two vmlal_u8 to vmull_u8
            // Then replace two previous lines with following code:
            // temp   = vraddhn_u16(res_lo, vrshrq_n_u16(res_lo, 8));
            // dst128 = vraddhn_high_u16(temp, res_hi, vrshrq_n_u16(res_hi, 8));

            if (fill_alpha) {
                dst128 = vorrq_u8(dst128, alpha_fill_mask);
            }

            // Save the result
            vst1q_u8(dst, dst128);

            src += 16;
            dst += 16;
        }

        // Process 1 pixel per iteration, max 3 iterations, same calculations as above
        for (; i < width; ++i) {
            // Top 32-bits will be not used in src32 & dst32
            uint8x8_t src32 = vreinterpret_u8_u32(vld1_dup_u32((Uint32*)src));
            uint8x8_t dst32 = vreinterpret_u8_u32(vld1_dup_u32((Uint32*)dst));

            uint8x8_t srcA = vtbl1_u8(src32, vget_low_u8(alpha_splat_mask));
            src32 = vtbl1_u8(src32, vget_low_u8(convert_mask));
            src32 = vorr_u8(src32, vget_low_u8(alpha_fill_mask));
            uint8x8_t srcInvA = vmvn_u8(srcA);

            uint16x8_t res = vdupq_n_u16(1);
            res = vmlal_u8(res, srcA,    src32);
            res = vmlal_u8(res, srcInvA, dst32);

            dst32 = vaddhn_u16(res, vshrq_n_u16(res, 8));

            if (fill_alpha) {
                dst32 = vorr_u8(dst32, vget_low_u8(alpha_fill_mask));
            }

            // Save the result, only low 32-bits
            vst1_lane_u32((Uint32*)dst, vreinterpret_u32_u8(dst32), 0);

            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#endif

// General (slow) N->N blending with pixel alpha
static void BlitNtoNPixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int srcbpp;
    int dstbpp;
    Uint32 Pixel;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB, dA;

    // Set up some basic variables
    srcbpp = srcfmt->bytes_per_pixel;
    dstbpp = dstfmt->bytes_per_pixel;

    while (height--) {
        DUFFS_LOOP(
        {
        DISEMBLE_RGBA(src, srcbpp, srcfmt, Pixel, sR, sG, sB, sA);
        if (sA) {
            DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
            ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        }
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

SDL_BlitFunc SDL_CalculateBlitA(SDL_Surface *surface)
{
    const SDL_PixelFormatDetails *sf = surface->fmt;
    const SDL_PixelFormatDetails *df = surface->map.info.dst_fmt;

    switch (surface->map.info.flags & ~SDL_COPY_RLE_MASK) {
    case SDL_COPY_BLEND:
        // Per-pixel alpha blits
        switch (df->bytes_per_pixel) {
        case 1:
            if (surface->map.info.dst_pal) {
                return BlitNto1PixelAlpha;
            } else {
                // RGB332 has no palette !
                return BlitNtoNPixelAlpha;
            }

        case 2:
            if (sf->bytes_per_pixel == 4 && sf->Amask == 0xff000000 && sf->Gmask == 0xff00 && ((sf->Rmask == 0xff && df->Rmask == 0x1f) || (sf->Bmask == 0xff && df->Bmask == 0x1f))) {
                if (df->Gmask == 0x7e0) {
                    return BlitARGBto565PixelAlpha;
                } else if (df->Gmask == 0x3e0 && !df->Amask) {
                    return BlitARGBto555PixelAlpha;
                }
            }
            return BlitNtoNPixelAlpha;

        case 4:
            if (SDL_PIXELLAYOUT(sf->format) == SDL_PACKEDLAYOUT_8888 && sf->Amask &&
                SDL_PIXELLAYOUT(df->format) == SDL_PACKEDLAYOUT_8888) {
#ifdef SDL_AVX2_INTRINSICS
                if (SDL_HasAVX2()) {
                    return Blit8888to8888PixelAlphaSwizzleAVX2;
                }
#endif
#ifdef SDL_SSE4_1_INTRINSICS
                if (SDL_HasSSE41()) {
                    return Blit8888to8888PixelAlphaSwizzleSSE41;
                }
#endif
#if defined(SDL_NEON_INTRINSICS) && (__ARM_ARCH >= 8)
                // To prevent "unused function" compiler warnings/errors
                (void)Blit8888to8888PixelAlpha;
                (void)Blit8888to8888PixelAlphaSwizzle;
                return Blit8888to8888PixelAlphaSwizzleNEON;
#else
                if (sf->format == df->format) {
                    return Blit8888to8888PixelAlpha;
                } else {
                    return Blit8888to8888PixelAlphaSwizzle;
                }
#endif
            }
            return BlitNtoNPixelAlpha;

        case 3:
        default:
            break;
        }
        return BlitNtoNPixelAlpha;

    case SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        if (sf->Amask == 0) {
            // Per-surface alpha blits
            switch (df->bytes_per_pixel) {
            case 1:
                if (surface->map.info.dst_pal) {
                    return BlitNto1SurfaceAlpha;
                } else {
                    // RGB332 has no palette !
                    return BlitNtoNSurfaceAlpha;
                }

            case 2:
                if (surface->map.identity) {
                    if (df->Gmask == 0x7e0) {
#ifdef SDL_MMX_INTRINSICS
                        if (SDL_HasMMX()) {
                            return Blit565to565SurfaceAlphaMMX;
                        } else
#endif
                        {
                            return Blit565to565SurfaceAlpha;
                        }
                    } else if (df->Gmask == 0x3e0) {
#ifdef SDL_MMX_INTRINSICS
                        if (SDL_HasMMX()) {
                            return Blit555to555SurfaceAlphaMMX;
                        } else
#endif
                        {
                            return Blit555to555SurfaceAlpha;
                        }
                    }
                }
                return BlitNtoNSurfaceAlpha;

            case 4:
                if (sf->Rmask == df->Rmask && sf->Gmask == df->Gmask && sf->Bmask == df->Bmask && sf->bytes_per_pixel == 4) {
#ifdef SDL_SSE2_INTRINSICS
                    if (sf->Rshift % 8 == 0 && sf->Gshift % 8 == 0 && sf->Bshift % 8 == 0 && SDL_HasSSE2()) {
                        return Blit888to888SurfaceAlphaSSE2;
                    }
#endif
                    if ((sf->Rmask | sf->Gmask | sf->Bmask) == 0xffffff) {
                        return BlitRGBtoRGBSurfaceAlpha;
                    }
                }
                return BlitNtoNSurfaceAlpha;

            case 3:
            default:
                return BlitNtoNSurfaceAlpha;
            }
        }
        break;

    case SDL_COPY_COLORKEY | SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        if (sf->Amask == 0) {
            if (df->bytes_per_pixel == 1) {

                if (surface->map.info.dst_pal) {
                    return BlitNto1SurfaceAlphaKey;
                } else {
                    // RGB332 has no palette !
                    return BlitNtoNSurfaceAlphaKey;
                }
            } else {
                return BlitNtoNSurfaceAlphaKey;
            }
        }
        break;
    }

    return NULL;
}

#endif // SDL_HAVE_BLIT_A

