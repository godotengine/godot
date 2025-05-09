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

#ifdef SDL_HAVE_BLIT_N

#include "SDL_pixels_c.h"
#include "SDL_surface_c.h"
#include "SDL_blit_copy.h"

// General optimized routines that write char by char
#define HAVE_FAST_WRITE_INT8 1

// On some CPU, it's slower than combining and write a word
#ifdef __MIPS__
#undef HAVE_FAST_WRITE_INT8
#define HAVE_FAST_WRITE_INT8 0
#endif

// Functions to blit from N-bit surfaces to other surfaces

#define BLIT_FEATURE_NONE                       0x00
#define BLIT_FEATURE_HAS_MMX                    0x01
#define BLIT_FEATURE_HAS_ALTIVEC                0x02
#define BLIT_FEATURE_ALTIVEC_DONT_USE_PREFETCH  0x04

#ifdef SDL_ALTIVEC_BLITTERS
#ifdef SDL_PLATFORM_MACOS
#include <sys/sysctl.h>
static size_t GetL3CacheSize(void)
{
    const char key[] = "hw.l3cachesize";
    u_int64_t result = 0;
    size_t typeSize = sizeof(result);

    int err = sysctlbyname(key, &result, &typeSize, NULL, 0);
    if (0 != err) {
        return 0;
    }

    return result;
}
#else
static size_t GetL3CacheSize(void)
{
    // XXX: Just guess G4
    return 2097152;
}
#endif // SDL_PLATFORM_MACOS

#if (defined(SDL_PLATFORM_MACOS) && (__GNUC__ < 4))
#define VECUINT8_LITERAL(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) \
    (vector unsigned char)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#define VECUINT16_LITERAL(a, b, c, d, e, f, g, h) \
    (vector unsigned short)(a, b, c, d, e, f, g, h)
#else
#define VECUINT8_LITERAL(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) \
    (vector unsigned char)                                               \
    {                                                                    \
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p                   \
    }
#define VECUINT16_LITERAL(a, b, c, d, e, f, g, h) \
    (vector unsigned short)                       \
    {                                             \
        a, b, c, d, e, f, g, h                    \
    }
#endif

#define UNALIGNED_PTR(x)       (((size_t)x) & 0x0000000F)
#define VSWIZZLE32(a, b, c, d) (vector unsigned char)(0x00 + a, 0x00 + b, 0x00 + c, 0x00 + d, \
                                                      0x04 + a, 0x04 + b, 0x04 + c, 0x04 + d, \
                                                      0x08 + a, 0x08 + b, 0x08 + c, 0x08 + d, \
                                                      0x0C + a, 0x0C + b, 0x0C + c, 0x0C + d)

#define MAKE8888(dstfmt, r, g, b, a)           \
    (((r << dstfmt->Rshift) & dstfmt->Rmask) | \
     ((g << dstfmt->Gshift) & dstfmt->Gmask) | \
     ((b << dstfmt->Bshift) & dstfmt->Bmask) | \
     ((a << dstfmt->Ashift) & dstfmt->Amask))

/*
 * Data Stream Touch...Altivec cache prefetching.
 *
 *  Don't use this on a G5...however, the speed boost is very significant
 *   on a G4.
 */
#define DST_CHAN_SRC  1
#define DST_CHAN_DEST 2

// macro to set DST control word value...
#define DST_CTRL(size, count, stride) \
    (((size) << 24) | ((count) << 16) | (stride))

#define VEC_ALIGNER(src) ((UNALIGNED_PTR(src))   \
                              ? vec_lvsl(0, src) \
                              : vec_add(vec_lvsl(8, src), vec_splat_u8(8)))

// Calculate the permute vector used for 32->32 swizzling
static vector unsigned char calc_swizzle32(const SDL_PixelFormatDetails *srcfmt, const SDL_PixelFormatDetails *dstfmt)
{
    /*
     * We have to assume that the bits that aren't used by other
     *  colors is alpha, and it's one complete byte, since some formats
     *  leave alpha with a zero mask, but we should still swizzle the bits.
     */
    // ARGB
    static const SDL_PixelFormatDetails default_pixel_format = {
        SDL_PIXELFORMAT_ARGB8888, 0, 0, { 0, 0 }, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, 8, 8, 8, 8, 16, 8, 0, 24
    };
    const vector unsigned char plus = VECUINT8_LITERAL(0x00, 0x00, 0x00, 0x00,
                                                       0x04, 0x04, 0x04, 0x04,
                                                       0x08, 0x08, 0x08, 0x08,
                                                       0x0C, 0x0C, 0x0C,
                                                       0x0C);
    vector unsigned char vswiz;
    vector unsigned int srcvec;
    Uint32 rmask, gmask, bmask, amask;

    if (!srcfmt) {
        srcfmt = &default_pixel_format;
    }
    if (!dstfmt) {
        dstfmt = &default_pixel_format;
    }

#define RESHIFT(X) (3 - ((X) >> 3))
    rmask = RESHIFT(srcfmt->Rshift) << (dstfmt->Rshift);
    gmask = RESHIFT(srcfmt->Gshift) << (dstfmt->Gshift);
    bmask = RESHIFT(srcfmt->Bshift) << (dstfmt->Bshift);

    // Use zero for alpha if either surface doesn't have alpha
    if (dstfmt->Amask) {
        amask =
            ((srcfmt->Amask) ? RESHIFT(srcfmt->Ashift) : 0x10) << (dstfmt->Ashift);
    } else {
        amask =
            0x10101010 & ((dstfmt->Rmask | dstfmt->Gmask | dstfmt->Bmask) ^
                          0xFFFFFFFF);
    }
#undef RESHIFT

    ((unsigned int *)(char *)&srcvec)[0] = (rmask | gmask | bmask | amask);
    vswiz = vec_add(plus, (vector unsigned char)vec_splat(srcvec, 0));
    return (vswiz);
}

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
// reorder bytes for PowerPC little endian
static vector unsigned char reorder_ppc64le_vec(vector unsigned char vpermute)
{
    /* The result vector of calc_swizzle32 reorder bytes using vec_perm.
       The LE transformation for vec_perm has an implicit assumption
       that the permutation is being used to reorder vector elements,
       not to reorder bytes within those elements.
       Unfortunately the result order is not the expected one for powerpc
       little endian when the two first vector parameters of vec_perm are
       not of type 'vector char'. This is because the numbering from the
       left for BE, and numbering from the right for LE, produces a
       different interpretation of what the odd and even lanes are.
       Refer to fedora bug 1392465
     */

    const vector unsigned char ppc64le_reorder = VECUINT8_LITERAL(
        0x01, 0x00, 0x03, 0x02,
        0x05, 0x04, 0x07, 0x06,
        0x09, 0x08, 0x0B, 0x0A,
        0x0D, 0x0C, 0x0F, 0x0E);

    vector unsigned char vswiz_ppc64le;
    vswiz_ppc64le = vec_perm(vpermute, vpermute, ppc64le_reorder);
    return (vswiz_ppc64le);
}
#endif

static void Blit_XRGB8888_RGB565(SDL_BlitInfo *info);
static void Blit_XRGB8888_RGB565Altivec(SDL_BlitInfo *info)
{
    int height = info->dst_h;
    Uint8 *src = (Uint8 *)info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = (Uint8 *)info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    vector unsigned char valpha = vec_splat_u8(0);
    vector unsigned char vpermute = calc_swizzle32(srcfmt, NULL);
    vector unsigned char vgmerge = VECUINT8_LITERAL(0x00, 0x02, 0x00, 0x06,
                                                    0x00, 0x0a, 0x00, 0x0e,
                                                    0x00, 0x12, 0x00, 0x16,
                                                    0x00, 0x1a, 0x00, 0x1e);
    vector unsigned short v1 = vec_splat_u16(1);
    vector unsigned short v3 = vec_splat_u16(3);
    vector unsigned short v3f =
        VECUINT16_LITERAL(0x003f, 0x003f, 0x003f, 0x003f,
                          0x003f, 0x003f, 0x003f, 0x003f);
    vector unsigned short vfc =
        VECUINT16_LITERAL(0x00fc, 0x00fc, 0x00fc, 0x00fc,
                          0x00fc, 0x00fc, 0x00fc, 0x00fc);
    vector unsigned short vf800 = (vector unsigned short)vec_splat_u8(-7);
    vf800 = vec_sl(vf800, vec_splat_u16(8));

    while (height--) {
        vector unsigned char valigner;
        vector unsigned char voverflow;
        vector unsigned char vsrc;

        int width = info->dst_w;
        int extrawidth;

        // do scalar until we can align...
#define ONE_PIXEL_BLEND(condition, widthvar)           \
    while (condition) {                                \
        Uint32 Pixel;                                  \
        unsigned sR, sG, sB, sA;                       \
        DISEMBLE_RGBA((Uint8 *)src, 4, srcfmt, Pixel,  \
                      sR, sG, sB, sA);                 \
        *(Uint16 *)(dst) = (((sR << 8) & 0x0000F800) | \
                            ((sG << 3) & 0x000007E0) | \
                            ((sB >> 3) & 0x0000001F)); \
        dst += 2;                                      \
        src += 4;                                      \
        widthvar--;                                    \
    }

        ONE_PIXEL_BLEND(((UNALIGNED_PTR(dst)) && (width)), width);

        // After all that work, here's the vector part!
        extrawidth = (width % 8); // trailing unaligned stores
        width -= extrawidth;
        vsrc = vec_ld(0, src);
        valigner = VEC_ALIGNER(src);

        while (width) {
            vector unsigned short vpixel, vrpixel, vgpixel, vbpixel;
            vector unsigned int vsrc1, vsrc2;
            vector unsigned char vdst;

            voverflow = vec_ld(15, src);
            vsrc = vec_perm(vsrc, voverflow, valigner);
            vsrc1 = (vector unsigned int)vec_perm(vsrc, valpha, vpermute);
            src += 16;
            vsrc = voverflow;
            voverflow = vec_ld(15, src);
            vsrc = vec_perm(vsrc, voverflow, valigner);
            vsrc2 = (vector unsigned int)vec_perm(vsrc, valpha, vpermute);
            // 1555
            vpixel = (vector unsigned short)vec_packpx(vsrc1, vsrc2);
            vgpixel = (vector unsigned short)vec_perm(vsrc1, vsrc2, vgmerge);
            vgpixel = vec_and(vgpixel, vfc);
            vgpixel = vec_sl(vgpixel, v3);
            vrpixel = vec_sl(vpixel, v1);
            vrpixel = vec_and(vrpixel, vf800);
            vbpixel = vec_and(vpixel, v3f);
            vdst =
                vec_or((vector unsigned char)vrpixel,
                       (vector unsigned char)vgpixel);
            // 565
            vdst = vec_or(vdst, (vector unsigned char)vbpixel);
            vec_st(vdst, 0, dst);

            width -= 8;
            src += 16;
            dst += 16;
            vsrc = voverflow;
        }

        SDL_assert(width == 0);

        // do scalar until we can align...
        ONE_PIXEL_BLEND((extrawidth), extrawidth);
#undef ONE_PIXEL_BLEND

        src += srcskip; // move to next row, accounting for pitch.
        dst += dstskip;
    }
}

static void Blit_RGB565_32Altivec(SDL_BlitInfo *info)
{
    int height = info->dst_h;
    Uint8 *src = (Uint8 *)info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = (Uint8 *)info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    unsigned alpha;
    vector unsigned char valpha;
    vector unsigned char vpermute;
    vector unsigned short vf800;
    vector unsigned int v8 = vec_splat_u32(8);
    vector unsigned int v16 = vec_add(v8, v8);
    vector unsigned short v2 = vec_splat_u16(2);
    vector unsigned short v3 = vec_splat_u16(3);
    /*
       0x10 - 0x1f is the alpha
       0x00 - 0x0e evens are the red
       0x01 - 0x0f odds are zero
     */
    vector unsigned char vredalpha1 = VECUINT8_LITERAL(0x10, 0x00, 0x01, 0x01,
                                                       0x10, 0x02, 0x01, 0x01,
                                                       0x10, 0x04, 0x01, 0x01,
                                                       0x10, 0x06, 0x01,
                                                       0x01);
    vector unsigned char vredalpha2 =
        (vector unsigned char)(vec_add((vector unsigned int)vredalpha1, vec_sl(v8, v16)));
    /*
       0x00 - 0x0f is ARxx ARxx ARxx ARxx
       0x11 - 0x0f odds are blue
     */
    vector unsigned char vblue1 = VECUINT8_LITERAL(0x00, 0x01, 0x02, 0x11,
                                                   0x04, 0x05, 0x06, 0x13,
                                                   0x08, 0x09, 0x0a, 0x15,
                                                   0x0c, 0x0d, 0x0e, 0x17);
    vector unsigned char vblue2 =
        (vector unsigned char)(vec_add((vector unsigned int)vblue1, v8));
    /*
       0x00 - 0x0f is ARxB ARxB ARxB ARxB
       0x10 - 0x0e evens are green
     */
    vector unsigned char vgreen1 = VECUINT8_LITERAL(0x00, 0x01, 0x10, 0x03,
                                                    0x04, 0x05, 0x12, 0x07,
                                                    0x08, 0x09, 0x14, 0x0b,
                                                    0x0c, 0x0d, 0x16, 0x0f);
    vector unsigned char vgreen2 =
        (vector unsigned char)(vec_add((vector unsigned int)vgreen1, vec_sl(v8, v8)));

    SDL_assert(srcfmt->bytes_per_pixel == 2);
    SDL_assert(dstfmt->bytes_per_pixel == 4);

    vf800 = (vector unsigned short)vec_splat_u8(-7);
    vf800 = vec_sl(vf800, vec_splat_u16(8));

    if (dstfmt->Amask && info->a) {
        ((unsigned char *)&valpha)[0] = alpha = info->a;
        valpha = vec_splat(valpha, 0);
    } else {
        alpha = 0;
        valpha = vec_splat_u8(0);
    }

    vpermute = calc_swizzle32(NULL, dstfmt);
    while (height--) {
        vector unsigned char valigner;
        vector unsigned char voverflow;
        vector unsigned char vsrc;

        int width = info->dst_w;
        int extrawidth;

        // do scalar until we can align...
#define ONE_PIXEL_BLEND(condition, widthvar)              \
    while (condition) {                                   \
        unsigned sR, sG, sB;                              \
        unsigned short Pixel = *((unsigned short *)src);  \
        sR = (Pixel >> 8) & 0xf8;                         \
        sG = (Pixel >> 3) & 0xfc;                         \
        sB = (Pixel << 3) & 0xf8;                         \
        ASSEMBLE_RGBA(dst, 4, dstfmt, sR, sG, sB, alpha); \
        src += 2;                                         \
        dst += 4;                                         \
        widthvar--;                                       \
    }
        ONE_PIXEL_BLEND(((UNALIGNED_PTR(dst)) && (width)), width);

        // After all that work, here's the vector part!
        extrawidth = (width % 8); // trailing unaligned stores
        width -= extrawidth;
        vsrc = vec_ld(0, src);
        valigner = VEC_ALIGNER(src);

        while (width) {
            vector unsigned short vR, vG, vB;
            vector unsigned char vdst1, vdst2;

            voverflow = vec_ld(15, src);
            vsrc = vec_perm(vsrc, voverflow, valigner);

            vR = vec_and((vector unsigned short)vsrc, vf800);
            vB = vec_sl((vector unsigned short)vsrc, v3);
            vG = vec_sl(vB, v2);

            vdst1 =
                (vector unsigned char)vec_perm((vector unsigned char)vR,
                                               valpha, vredalpha1);
            vdst1 = vec_perm(vdst1, (vector unsigned char)vB, vblue1);
            vdst1 = vec_perm(vdst1, (vector unsigned char)vG, vgreen1);
            vdst1 = vec_perm(vdst1, valpha, vpermute);
            vec_st(vdst1, 0, dst);

            vdst2 =
                (vector unsigned char)vec_perm((vector unsigned char)vR,
                                               valpha, vredalpha2);
            vdst2 = vec_perm(vdst2, (vector unsigned char)vB, vblue2);
            vdst2 = vec_perm(vdst2, (vector unsigned char)vG, vgreen2);
            vdst2 = vec_perm(vdst2, valpha, vpermute);
            vec_st(vdst2, 16, dst);

            width -= 8;
            dst += 32;
            src += 16;
            vsrc = voverflow;
        }

        SDL_assert(width == 0);

        // do scalar until we can align...
        ONE_PIXEL_BLEND((extrawidth), extrawidth);
#undef ONE_PIXEL_BLEND

        src += srcskip; // move to next row, accounting for pitch.
        dst += dstskip;
    }
}

static void Blit_RGB555_32Altivec(SDL_BlitInfo *info)
{
    int height = info->dst_h;
    Uint8 *src = (Uint8 *)info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = (Uint8 *)info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    unsigned alpha;
    vector unsigned char valpha;
    vector unsigned char vpermute;
    vector unsigned short vf800;
    vector unsigned int v8 = vec_splat_u32(8);
    vector unsigned int v16 = vec_add(v8, v8);
    vector unsigned short v1 = vec_splat_u16(1);
    vector unsigned short v3 = vec_splat_u16(3);
    /*
       0x10 - 0x1f is the alpha
       0x00 - 0x0e evens are the red
       0x01 - 0x0f odds are zero
     */
    vector unsigned char vredalpha1 = VECUINT8_LITERAL(0x10, 0x00, 0x01, 0x01,
                                                       0x10, 0x02, 0x01, 0x01,
                                                       0x10, 0x04, 0x01, 0x01,
                                                       0x10, 0x06, 0x01,
                                                       0x01);
    vector unsigned char vredalpha2 =
        (vector unsigned char)(vec_add((vector unsigned int)vredalpha1, vec_sl(v8, v16)));
    /*
       0x00 - 0x0f is ARxx ARxx ARxx ARxx
       0x11 - 0x0f odds are blue
     */
    vector unsigned char vblue1 = VECUINT8_LITERAL(0x00, 0x01, 0x02, 0x11,
                                                   0x04, 0x05, 0x06, 0x13,
                                                   0x08, 0x09, 0x0a, 0x15,
                                                   0x0c, 0x0d, 0x0e, 0x17);
    vector unsigned char vblue2 =
        (vector unsigned char)(vec_add((vector unsigned int)vblue1, v8));
    /*
       0x00 - 0x0f is ARxB ARxB ARxB ARxB
       0x10 - 0x0e evens are green
     */
    vector unsigned char vgreen1 = VECUINT8_LITERAL(0x00, 0x01, 0x10, 0x03,
                                                    0x04, 0x05, 0x12, 0x07,
                                                    0x08, 0x09, 0x14, 0x0b,
                                                    0x0c, 0x0d, 0x16, 0x0f);
    vector unsigned char vgreen2 =
        (vector unsigned char)(vec_add((vector unsigned int)vgreen1, vec_sl(v8, v8)));

    SDL_assert(srcfmt->bytes_per_pixel == 2);
    SDL_assert(dstfmt->bytes_per_pixel == 4);

    vf800 = (vector unsigned short)vec_splat_u8(-7);
    vf800 = vec_sl(vf800, vec_splat_u16(8));

    if (dstfmt->Amask && info->a) {
        ((unsigned char *)&valpha)[0] = alpha = info->a;
        valpha = vec_splat(valpha, 0);
    } else {
        alpha = 0;
        valpha = vec_splat_u8(0);
    }

    vpermute = calc_swizzle32(NULL, dstfmt);
    while (height--) {
        vector unsigned char valigner;
        vector unsigned char voverflow;
        vector unsigned char vsrc;

        int width = info->dst_w;
        int extrawidth;

        // do scalar until we can align...
#define ONE_PIXEL_BLEND(condition, widthvar)              \
    while (condition) {                                   \
        unsigned sR, sG, sB;                              \
        unsigned short Pixel = *((unsigned short *)src);  \
        sR = (Pixel >> 7) & 0xf8;                         \
        sG = (Pixel >> 2) & 0xf8;                         \
        sB = (Pixel << 3) & 0xf8;                         \
        ASSEMBLE_RGBA(dst, 4, dstfmt, sR, sG, sB, alpha); \
        src += 2;                                         \
        dst += 4;                                         \
        widthvar--;                                       \
    }
        ONE_PIXEL_BLEND(((UNALIGNED_PTR(dst)) && (width)), width);

        // After all that work, here's the vector part!
        extrawidth = (width % 8); // trailing unaligned stores
        width -= extrawidth;
        vsrc = vec_ld(0, src);
        valigner = VEC_ALIGNER(src);

        while (width) {
            vector unsigned short vR, vG, vB;
            vector unsigned char vdst1, vdst2;

            voverflow = vec_ld(15, src);
            vsrc = vec_perm(vsrc, voverflow, valigner);

            vR = vec_and(vec_sl((vector unsigned short)vsrc, v1), vf800);
            vB = vec_sl((vector unsigned short)vsrc, v3);
            vG = vec_sl(vB, v3);

            vdst1 =
                (vector unsigned char)vec_perm((vector unsigned char)vR,
                                               valpha, vredalpha1);
            vdst1 = vec_perm(vdst1, (vector unsigned char)vB, vblue1);
            vdst1 = vec_perm(vdst1, (vector unsigned char)vG, vgreen1);
            vdst1 = vec_perm(vdst1, valpha, vpermute);
            vec_st(vdst1, 0, dst);

            vdst2 =
                (vector unsigned char)vec_perm((vector unsigned char)vR,
                                               valpha, vredalpha2);
            vdst2 = vec_perm(vdst2, (vector unsigned char)vB, vblue2);
            vdst2 = vec_perm(vdst2, (vector unsigned char)vG, vgreen2);
            vdst2 = vec_perm(vdst2, valpha, vpermute);
            vec_st(vdst2, 16, dst);

            width -= 8;
            dst += 32;
            src += 16;
            vsrc = voverflow;
        }

        SDL_assert(width == 0);

        // do scalar until we can align...
        ONE_PIXEL_BLEND((extrawidth), extrawidth);
#undef ONE_PIXEL_BLEND

        src += srcskip; // move to next row, accounting for pitch.
        dst += dstskip;
    }
}

static void BlitNtoNKey(SDL_BlitInfo *info);
static void BlitNtoNKeyCopyAlpha(SDL_BlitInfo *info);
static void Blit32to32KeyAltivec(SDL_BlitInfo *info)
{
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip / 4;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip / 4;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int dstbpp = dstfmt->bytes_per_pixel;
    int copy_alpha = (srcfmt->Amask && dstfmt->Amask);
    unsigned alpha = dstfmt->Amask ? info->a : 0;
    Uint32 rgbmask = srcfmt->Rmask | srcfmt->Gmask | srcfmt->Bmask;
    Uint32 ckey = info->colorkey;
    vector unsigned int valpha;
    vector unsigned char vpermute;
    vector unsigned char vzero;
    vector unsigned int vckey;
    vector unsigned int vrgbmask;
    vpermute = calc_swizzle32(srcfmt, dstfmt);
    if (info->dst_w < 16) {
        if (copy_alpha) {
            BlitNtoNKeyCopyAlpha(info);
        } else {
            BlitNtoNKey(info);
        }
        return;
    }
    vzero = vec_splat_u8(0);
    if (alpha) {
        ((unsigned char *)&valpha)[0] = (unsigned char)alpha;
        valpha =
            (vector unsigned int)vec_splat((vector unsigned char)valpha, 0);
    } else {
        valpha = (vector unsigned int)vzero;
    }
    ckey &= rgbmask;
    ((unsigned int *)(char *)&vckey)[0] = ckey;
    vckey = vec_splat(vckey, 0);
    ((unsigned int *)(char *)&vrgbmask)[0] = rgbmask;
    vrgbmask = vec_splat(vrgbmask, 0);

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    // reorder bytes for PowerPC little endian
    vpermute = reorder_ppc64le_vec(vpermute);
#endif

    while (height--) {
#define ONE_PIXEL_BLEND(condition, widthvar)                    \
    if (copy_alpha) {                                           \
        while (condition) {                                     \
            Uint32 Pixel;                                       \
            unsigned sR, sG, sB, sA;                            \
            DISEMBLE_RGBA((Uint8 *)srcp, srcbpp, srcfmt, Pixel, \
                          sR, sG, sB, sA);                      \
            if ((Pixel & rgbmask) != ckey) {                    \
                ASSEMBLE_RGBA((Uint8 *)dstp, dstbpp, dstfmt,    \
                              sR, sG, sB, sA);                  \
            }                                                   \
            dstp = (Uint32 *)(((Uint8 *)dstp) + dstbpp);        \
            srcp = (Uint32 *)(((Uint8 *)srcp) + srcbpp);        \
            widthvar--;                                         \
        }                                                       \
    } else {                                                    \
        while (condition) {                                     \
            Uint32 Pixel;                                       \
            unsigned sR, sG, sB;                                \
            RETRIEVE_RGB_PIXEL((Uint8 *)srcp, srcbpp, Pixel);   \
            if (Pixel != ckey) {                                \
                RGB_FROM_PIXEL(Pixel, srcfmt, sR, sG, sB);      \
                ASSEMBLE_RGBA((Uint8 *)dstp, dstbpp, dstfmt,    \
                              sR, sG, sB, alpha);               \
            }                                                   \
            dstp = (Uint32 *)(((Uint8 *)dstp) + dstbpp);        \
            srcp = (Uint32 *)(((Uint8 *)srcp) + srcbpp);        \
            widthvar--;                                         \
        }                                                       \
    }
        int width = info->dst_w;
        ONE_PIXEL_BLEND((UNALIGNED_PTR(dstp)) && (width), width);
        SDL_assert(width > 0);
        if (width > 0) {
            int extrawidth = (width % 4);
            vector unsigned char valigner = VEC_ALIGNER(srcp);
            vector unsigned int vs = vec_ld(0, srcp);
            width -= extrawidth;
            SDL_assert(width >= 4);
            while (width) {
                vector unsigned char vsel;
                vector unsigned int vd;
                vector unsigned int voverflow = vec_ld(15, srcp);
                // load the source vec
                vs = vec_perm(vs, voverflow, valigner);
                // vsel is set for items that match the key
                vsel = (vector unsigned char)vec_and(vs, vrgbmask);
                vsel = (vector unsigned char)vec_cmpeq(vs, vckey);
                // permute the src vec to the dest format
                vs = vec_perm(vs, valpha, vpermute);
                // load the destination vec
                vd = vec_ld(0, dstp);
                // select the source and dest into vs
                vd = (vector unsigned int)vec_sel((vector unsigned char)vs,
                                                  (vector unsigned char)vd,
                                                  vsel);

                vec_st(vd, 0, dstp);
                srcp += 4;
                width -= 4;
                dstp += 4;
                vs = voverflow;
            }
            ONE_PIXEL_BLEND((extrawidth), extrawidth);
#undef ONE_PIXEL_BLEND
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

// Altivec code to swizzle one 32-bit surface to a different 32-bit format.
// Use this on a G5
static void ConvertAltivec32to32_noprefetch(SDL_BlitInfo *info)
{
    int height = info->dst_h;
    Uint32 *src = (Uint32 *)info->src;
    int srcskip = info->src_skip / 4;
    Uint32 *dst = (Uint32 *)info->dst;
    int dstskip = info->dst_skip / 4;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    vector unsigned int vzero = vec_splat_u32(0);
    vector unsigned char vpermute = calc_swizzle32(srcfmt, dstfmt);
    if (dstfmt->Amask && !srcfmt->Amask) {
        if (info->a) {
            vector unsigned char valpha;
            ((unsigned char *)&valpha)[0] = info->a;
            vzero = (vector unsigned int)vec_splat(valpha, 0);
        }
    }

    SDL_assert(srcfmt->bytes_per_pixel == 4);
    SDL_assert(dstfmt->bytes_per_pixel == 4);

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    // reorder bytes for PowerPC little endian
    vpermute = reorder_ppc64le_vec(vpermute);
#endif

    while (height--) {
        vector unsigned char valigner;
        vector unsigned int vbits;
        vector unsigned int voverflow;
        Uint32 bits;
        Uint8 r, g, b, a;

        int width = info->dst_w;
        int extrawidth;

        // do scalar until we can align...
        while ((UNALIGNED_PTR(dst)) && (width)) {
            bits = *(src++);
            RGBA_FROM_8888(bits, srcfmt, r, g, b, a);
            if (!srcfmt->Amask)
                a = info->a;
            *(dst++) = MAKE8888(dstfmt, r, g, b, a);
            width--;
        }

        // After all that work, here's the vector part!
        extrawidth = (width % 4);
        width -= extrawidth;
        valigner = VEC_ALIGNER(src);
        vbits = vec_ld(0, src);

        while (width) {
            voverflow = vec_ld(15, src);
            src += 4;
            width -= 4;
            vbits = vec_perm(vbits, voverflow, valigner); // src is ready.
            vbits = vec_perm(vbits, vzero, vpermute); // swizzle it.
            vec_st(vbits, 0, dst);                    // store it back out.
            dst += 4;
            vbits = voverflow;
        }

        SDL_assert(width == 0);

        // cover pixels at the end of the row that didn't fit in 16 bytes.
        while (extrawidth) {
            bits = *(src++); // max 7 pixels, don't bother with prefetch.
            RGBA_FROM_8888(bits, srcfmt, r, g, b, a);
            if (!srcfmt->Amask)
                a = info->a;
            *(dst++) = MAKE8888(dstfmt, r, g, b, a);
            extrawidth--;
        }

        src += srcskip;
        dst += dstskip;
    }
}

// Altivec code to swizzle one 32-bit surface to a different 32-bit format.
// Use this on a G4
static void ConvertAltivec32to32_prefetch(SDL_BlitInfo *info)
{
    const int scalar_dst_lead = sizeof(Uint32) * 4;
    const int vector_dst_lead = sizeof(Uint32) * 16;

    int height = info->dst_h;
    Uint32 *src = (Uint32 *)info->src;
    int srcskip = info->src_skip / 4;
    Uint32 *dst = (Uint32 *)info->dst;
    int dstskip = info->dst_skip / 4;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    vector unsigned int vzero = vec_splat_u32(0);
    vector unsigned char vpermute = calc_swizzle32(srcfmt, dstfmt);
    if (dstfmt->Amask && !srcfmt->Amask) {
        if (info->a) {
            vector unsigned char valpha;
            ((unsigned char *)&valpha)[0] = info->a;
            vzero = (vector unsigned int)vec_splat(valpha, 0);
        }
    }

    SDL_assert(srcfmt->bytes_per_pixel == 4);
    SDL_assert(dstfmt->bytes_per_pixel == 4);

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    // reorder bytes for PowerPC little endian
    vpermute = reorder_ppc64le_vec(vpermute);
#endif

    while (height--) {
        vector unsigned char valigner;
        vector unsigned int vbits;
        vector unsigned int voverflow;
        Uint32 bits;
        Uint8 r, g, b, a;

        int width = info->dst_w;
        int extrawidth;

        // do scalar until we can align...
        while ((UNALIGNED_PTR(dst)) && (width)) {
            vec_dstt(src + scalar_dst_lead, DST_CTRL(2, 32, 1024),
                     DST_CHAN_SRC);
            vec_dstst(dst + scalar_dst_lead, DST_CTRL(2, 32, 1024),
                      DST_CHAN_DEST);
            bits = *(src++);
            RGBA_FROM_8888(bits, srcfmt, r, g, b, a);
            if (!srcfmt->Amask)
                a = info->a;
            *(dst++) = MAKE8888(dstfmt, r, g, b, a);
            width--;
        }

        // After all that work, here's the vector part!
        extrawidth = (width % 4);
        width -= extrawidth;
        valigner = VEC_ALIGNER(src);
        vbits = vec_ld(0, src);

        while (width) {
            vec_dstt(src + vector_dst_lead, DST_CTRL(2, 32, 1024),
                     DST_CHAN_SRC);
            vec_dstst(dst + vector_dst_lead, DST_CTRL(2, 32, 1024),
                      DST_CHAN_DEST);
            voverflow = vec_ld(15, src);
            src += 4;
            width -= 4;
            vbits = vec_perm(vbits, voverflow, valigner); // src is ready.
            vbits = vec_perm(vbits, vzero, vpermute); // swizzle it.
            vec_st(vbits, 0, dst);                    // store it back out.
            dst += 4;
            vbits = voverflow;
        }

        SDL_assert(width == 0);

        // cover pixels at the end of the row that didn't fit in 16 bytes.
        while (extrawidth) {
            bits = *(src++); // max 7 pixels, don't bother with prefetch.
            RGBA_FROM_8888(bits, srcfmt, r, g, b, a);
            if (!srcfmt->Amask)
                a = info->a;
            *(dst++) = MAKE8888(dstfmt, r, g, b, a);
            extrawidth--;
        }

        src += srcskip;
        dst += dstskip;
    }

    vec_dss(DST_CHAN_SRC);
    vec_dss(DST_CHAN_DEST);
}

static Uint32 GetBlitFeatures(void)
{
    static Uint32 features = ~0u;
    if (features == ~0u) {
        features = (0
                    // Feature 1 is has-MMX
                    | ((SDL_HasMMX()) ? BLIT_FEATURE_HAS_MMX : 0)
                    // Feature 2 is has-AltiVec
                    | ((SDL_HasAltiVec()) ? BLIT_FEATURE_HAS_ALTIVEC : 0)
                    // Feature 4 is dont-use-prefetch
                    // !!!! FIXME: Check for G5 or later, not the cache size! Always prefetch on a G4.
                    | ((GetL3CacheSize() == 0) ? BLIT_FEATURE_ALTIVEC_DONT_USE_PREFETCH : 0));
    }
    return features;
}

#ifdef __MWERKS__
#pragma altivec_model off
#endif
#else
// Feature 1 is has-MMX
#define GetBlitFeatures() ((SDL_HasMMX() ? BLIT_FEATURE_HAS_MMX : 0))
#endif

// This is now endian dependent
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define HI 1
#define LO 0
#else // SDL_BYTEORDER == SDL_BIG_ENDIAN
#define HI 0
#define LO 1
#endif

// Special optimized blit for RGB 8-8-8 --> RGB 5-5-5
#define RGB888_RGB555(dst, src)                                    \
    {                                                              \
        *(Uint16 *)(dst) = (Uint16)((((*src) & 0x00F80000) >> 9) | \
                                    (((*src) & 0x0000F800) >> 6) | \
                                    (((*src) & 0x000000F8) >> 3)); \
    }
#ifndef USE_DUFFS_LOOP
#define RGB888_RGB555_TWO(dst, src)                            \
    {                                                          \
        *(Uint32 *)(dst) = (((((src[HI]) & 0x00F80000) >> 9) | \
                             (((src[HI]) & 0x0000F800) >> 6) | \
                             (((src[HI]) & 0x000000F8) >> 3))  \
                            << 16) |                           \
                           (((src[LO]) & 0x00F80000) >> 9) |   \
                           (((src[LO]) & 0x0000F800) >> 6) |   \
                           (((src[LO]) & 0x000000F8) >> 3);    \
    }
#endif
static void Blit_XRGB8888_RGB555(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint32 *src;
    Uint16 *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = (Uint32 *)info->src;
    srcskip = info->src_skip / 4;
    dst = (Uint16 *)info->dst;
    dstskip = info->dst_skip / 2;

#ifdef USE_DUFFS_LOOP
    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
            RGB888_RGB555(dst, src);
            ++src;
            ++dst;
        , width);
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
            RGB888_RGB555(dst, src);
            ++src;
            ++dst;

            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                RGB888_RGB555(dst, src);
                ++src;
                ++dst;
                SDL_FALLTHROUGH;
            case 2:
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
                break;
            case 1:
                RGB888_RGB555(dst, src);
                ++src;
                ++dst;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    } else {
        while (height--) {
            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                RGB888_RGB555(dst, src);
                ++src;
                ++dst;
                SDL_FALLTHROUGH;
            case 2:
                RGB888_RGB555_TWO(dst, src);
                src += 2;
                dst += 2;
                break;
            case 1:
                RGB888_RGB555(dst, src);
                ++src;
                ++dst;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    }
#endif // USE_DUFFS_LOOP
}

// Special optimized blit for RGB 8-8-8 --> RGB 5-6-5
#define RGB888_RGB565(dst, src)                                    \
    {                                                              \
        *(Uint16 *)(dst) = (Uint16)((((*src) & 0x00F80000) >> 8) | \
                                    (((*src) & 0x0000FC00) >> 5) | \
                                    (((*src) & 0x000000F8) >> 3)); \
    }
#ifndef USE_DUFFS_LOOP
#define RGB888_RGB565_TWO(dst, src)                            \
    {                                                          \
        *(Uint32 *)(dst) = (((((src[HI]) & 0x00F80000) >> 8) | \
                             (((src[HI]) & 0x0000FC00) >> 5) | \
                             (((src[HI]) & 0x000000F8) >> 3))  \
                            << 16) |                           \
                           (((src[LO]) & 0x00F80000) >> 8) |   \
                           (((src[LO]) & 0x0000FC00) >> 5) |   \
                           (((src[LO]) & 0x000000F8) >> 3);    \
    }
#endif
static void Blit_XRGB8888_RGB565(SDL_BlitInfo *info)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint32 *src;
    Uint16 *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = (Uint32 *)info->src;
    srcskip = info->src_skip / 4;
    dst = (Uint16 *)info->dst;
    dstskip = info->dst_skip / 2;

#ifdef USE_DUFFS_LOOP
    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
            RGB888_RGB565(dst, src);
            ++src;
            ++dst;
        , width);
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
            RGB888_RGB565(dst, src);
            ++src;
            ++dst;

            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                RGB888_RGB565(dst, src);
                ++src;
                ++dst;
                SDL_FALLTHROUGH;
            case 2:
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
                break;
            case 1:
                RGB888_RGB565(dst, src);
                ++src;
                ++dst;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    } else {
        while (height--) {
            // Copy in 4 pixel chunks
            for (c = width / 4; c; --c) {
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
            }
            // Get any leftovers
            switch (width & 3) {
            case 3:
                RGB888_RGB565(dst, src);
                ++src;
                ++dst;
                SDL_FALLTHROUGH;
            case 2:
                RGB888_RGB565_TWO(dst, src);
                src += 2;
                dst += 2;
                break;
            case 1:
                RGB888_RGB565(dst, src);
                ++src;
                ++dst;
                break;
            }
            src += srcskip;
            dst += dstskip;
        }
    }
#endif // USE_DUFFS_LOOP
}

#ifdef SDL_HAVE_BLIT_N_RGB565

// Special optimized blit for RGB 5-6-5 --> 32-bit RGB surfaces
#define RGB565_32(dst, src, map) (map[src[LO] * 2] + map[src[HI] * 2 + 1])
static void Blit_RGB565_32(SDL_BlitInfo *info, const Uint32 *map)
{
#ifndef USE_DUFFS_LOOP
    int c;
#endif
    int width, height;
    Uint8 *src;
    Uint32 *dst;
    int srcskip, dstskip;

    // Set up some basic variables
    width = info->dst_w;
    height = info->dst_h;
    src = info->src;
    srcskip = info->src_skip;
    dst = (Uint32 *)info->dst;
    dstskip = info->dst_skip / 4;

#ifdef USE_DUFFS_LOOP
    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
#else
    while (height--) {
        // Copy in 4 pixel chunks
        for (c = width / 4; c; --c) {
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
        }
        // Get any leftovers
        switch (width & 3) {
        case 3:
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            SDL_FALLTHROUGH;
        case 2:
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            SDL_FALLTHROUGH;
        case 1:
            *dst++ = RGB565_32(dst, src, map);
            src += 2;
            break;
        }
        src += srcskip;
        dst += dstskip;
    }
#endif // USE_DUFFS_LOOP
}

/* *INDENT-OFF* */ // clang-format off

// Special optimized blit for RGB 5-6-5 --> ARGB 8-8-8-8
static const Uint32 RGB565_ARGB8888_LUT[512] = {
    0x00000000, 0xff000000, 0x00000008, 0xff002000,
    0x00000010, 0xff004000, 0x00000018, 0xff006100,
    0x00000020, 0xff008100, 0x00000029, 0xff00a100,
    0x00000031, 0xff00c200, 0x00000039, 0xff00e200,
    0x00000041, 0xff080000, 0x0000004a, 0xff082000,
    0x00000052, 0xff084000, 0x0000005a, 0xff086100,
    0x00000062, 0xff088100, 0x0000006a, 0xff08a100,
    0x00000073, 0xff08c200, 0x0000007b, 0xff08e200,
    0x00000083, 0xff100000, 0x0000008b, 0xff102000,
    0x00000094, 0xff104000, 0x0000009c, 0xff106100,
    0x000000a4, 0xff108100, 0x000000ac, 0xff10a100,
    0x000000b4, 0xff10c200, 0x000000bd, 0xff10e200,
    0x000000c5, 0xff180000, 0x000000cd, 0xff182000,
    0x000000d5, 0xff184000, 0x000000de, 0xff186100,
    0x000000e6, 0xff188100, 0x000000ee, 0xff18a100,
    0x000000f6, 0xff18c200, 0x000000ff, 0xff18e200,
    0x00000400, 0xff200000, 0x00000408, 0xff202000,
    0x00000410, 0xff204000, 0x00000418, 0xff206100,
    0x00000420, 0xff208100, 0x00000429, 0xff20a100,
    0x00000431, 0xff20c200, 0x00000439, 0xff20e200,
    0x00000441, 0xff290000, 0x0000044a, 0xff292000,
    0x00000452, 0xff294000, 0x0000045a, 0xff296100,
    0x00000462, 0xff298100, 0x0000046a, 0xff29a100,
    0x00000473, 0xff29c200, 0x0000047b, 0xff29e200,
    0x00000483, 0xff310000, 0x0000048b, 0xff312000,
    0x00000494, 0xff314000, 0x0000049c, 0xff316100,
    0x000004a4, 0xff318100, 0x000004ac, 0xff31a100,
    0x000004b4, 0xff31c200, 0x000004bd, 0xff31e200,
    0x000004c5, 0xff390000, 0x000004cd, 0xff392000,
    0x000004d5, 0xff394000, 0x000004de, 0xff396100,
    0x000004e6, 0xff398100, 0x000004ee, 0xff39a100,
    0x000004f6, 0xff39c200, 0x000004ff, 0xff39e200,
    0x00000800, 0xff410000, 0x00000808, 0xff412000,
    0x00000810, 0xff414000, 0x00000818, 0xff416100,
    0x00000820, 0xff418100, 0x00000829, 0xff41a100,
    0x00000831, 0xff41c200, 0x00000839, 0xff41e200,
    0x00000841, 0xff4a0000, 0x0000084a, 0xff4a2000,
    0x00000852, 0xff4a4000, 0x0000085a, 0xff4a6100,
    0x00000862, 0xff4a8100, 0x0000086a, 0xff4aa100,
    0x00000873, 0xff4ac200, 0x0000087b, 0xff4ae200,
    0x00000883, 0xff520000, 0x0000088b, 0xff522000,
    0x00000894, 0xff524000, 0x0000089c, 0xff526100,
    0x000008a4, 0xff528100, 0x000008ac, 0xff52a100,
    0x000008b4, 0xff52c200, 0x000008bd, 0xff52e200,
    0x000008c5, 0xff5a0000, 0x000008cd, 0xff5a2000,
    0x000008d5, 0xff5a4000, 0x000008de, 0xff5a6100,
    0x000008e6, 0xff5a8100, 0x000008ee, 0xff5aa100,
    0x000008f6, 0xff5ac200, 0x000008ff, 0xff5ae200,
    0x00000c00, 0xff620000, 0x00000c08, 0xff622000,
    0x00000c10, 0xff624000, 0x00000c18, 0xff626100,
    0x00000c20, 0xff628100, 0x00000c29, 0xff62a100,
    0x00000c31, 0xff62c200, 0x00000c39, 0xff62e200,
    0x00000c41, 0xff6a0000, 0x00000c4a, 0xff6a2000,
    0x00000c52, 0xff6a4000, 0x00000c5a, 0xff6a6100,
    0x00000c62, 0xff6a8100, 0x00000c6a, 0xff6aa100,
    0x00000c73, 0xff6ac200, 0x00000c7b, 0xff6ae200,
    0x00000c83, 0xff730000, 0x00000c8b, 0xff732000,
    0x00000c94, 0xff734000, 0x00000c9c, 0xff736100,
    0x00000ca4, 0xff738100, 0x00000cac, 0xff73a100,
    0x00000cb4, 0xff73c200, 0x00000cbd, 0xff73e200,
    0x00000cc5, 0xff7b0000, 0x00000ccd, 0xff7b2000,
    0x00000cd5, 0xff7b4000, 0x00000cde, 0xff7b6100,
    0x00000ce6, 0xff7b8100, 0x00000cee, 0xff7ba100,
    0x00000cf6, 0xff7bc200, 0x00000cff, 0xff7be200,
    0x00001000, 0xff830000, 0x00001008, 0xff832000,
    0x00001010, 0xff834000, 0x00001018, 0xff836100,
    0x00001020, 0xff838100, 0x00001029, 0xff83a100,
    0x00001031, 0xff83c200, 0x00001039, 0xff83e200,
    0x00001041, 0xff8b0000, 0x0000104a, 0xff8b2000,
    0x00001052, 0xff8b4000, 0x0000105a, 0xff8b6100,
    0x00001062, 0xff8b8100, 0x0000106a, 0xff8ba100,
    0x00001073, 0xff8bc200, 0x0000107b, 0xff8be200,
    0x00001083, 0xff940000, 0x0000108b, 0xff942000,
    0x00001094, 0xff944000, 0x0000109c, 0xff946100,
    0x000010a4, 0xff948100, 0x000010ac, 0xff94a100,
    0x000010b4, 0xff94c200, 0x000010bd, 0xff94e200,
    0x000010c5, 0xff9c0000, 0x000010cd, 0xff9c2000,
    0x000010d5, 0xff9c4000, 0x000010de, 0xff9c6100,
    0x000010e6, 0xff9c8100, 0x000010ee, 0xff9ca100,
    0x000010f6, 0xff9cc200, 0x000010ff, 0xff9ce200,
    0x00001400, 0xffa40000, 0x00001408, 0xffa42000,
    0x00001410, 0xffa44000, 0x00001418, 0xffa46100,
    0x00001420, 0xffa48100, 0x00001429, 0xffa4a100,
    0x00001431, 0xffa4c200, 0x00001439, 0xffa4e200,
    0x00001441, 0xffac0000, 0x0000144a, 0xffac2000,
    0x00001452, 0xffac4000, 0x0000145a, 0xffac6100,
    0x00001462, 0xffac8100, 0x0000146a, 0xffaca100,
    0x00001473, 0xffacc200, 0x0000147b, 0xfface200,
    0x00001483, 0xffb40000, 0x0000148b, 0xffb42000,
    0x00001494, 0xffb44000, 0x0000149c, 0xffb46100,
    0x000014a4, 0xffb48100, 0x000014ac, 0xffb4a100,
    0x000014b4, 0xffb4c200, 0x000014bd, 0xffb4e200,
    0x000014c5, 0xffbd0000, 0x000014cd, 0xffbd2000,
    0x000014d5, 0xffbd4000, 0x000014de, 0xffbd6100,
    0x000014e6, 0xffbd8100, 0x000014ee, 0xffbda100,
    0x000014f6, 0xffbdc200, 0x000014ff, 0xffbde200,
    0x00001800, 0xffc50000, 0x00001808, 0xffc52000,
    0x00001810, 0xffc54000, 0x00001818, 0xffc56100,
    0x00001820, 0xffc58100, 0x00001829, 0xffc5a100,
    0x00001831, 0xffc5c200, 0x00001839, 0xffc5e200,
    0x00001841, 0xffcd0000, 0x0000184a, 0xffcd2000,
    0x00001852, 0xffcd4000, 0x0000185a, 0xffcd6100,
    0x00001862, 0xffcd8100, 0x0000186a, 0xffcda100,
    0x00001873, 0xffcdc200, 0x0000187b, 0xffcde200,
    0x00001883, 0xffd50000, 0x0000188b, 0xffd52000,
    0x00001894, 0xffd54000, 0x0000189c, 0xffd56100,
    0x000018a4, 0xffd58100, 0x000018ac, 0xffd5a100,
    0x000018b4, 0xffd5c200, 0x000018bd, 0xffd5e200,
    0x000018c5, 0xffde0000, 0x000018cd, 0xffde2000,
    0x000018d5, 0xffde4000, 0x000018de, 0xffde6100,
    0x000018e6, 0xffde8100, 0x000018ee, 0xffdea100,
    0x000018f6, 0xffdec200, 0x000018ff, 0xffdee200,
    0x00001c00, 0xffe60000, 0x00001c08, 0xffe62000,
    0x00001c10, 0xffe64000, 0x00001c18, 0xffe66100,
    0x00001c20, 0xffe68100, 0x00001c29, 0xffe6a100,
    0x00001c31, 0xffe6c200, 0x00001c39, 0xffe6e200,
    0x00001c41, 0xffee0000, 0x00001c4a, 0xffee2000,
    0x00001c52, 0xffee4000, 0x00001c5a, 0xffee6100,
    0x00001c62, 0xffee8100, 0x00001c6a, 0xffeea100,
    0x00001c73, 0xffeec200, 0x00001c7b, 0xffeee200,
    0x00001c83, 0xfff60000, 0x00001c8b, 0xfff62000,
    0x00001c94, 0xfff64000, 0x00001c9c, 0xfff66100,
    0x00001ca4, 0xfff68100, 0x00001cac, 0xfff6a100,
    0x00001cb4, 0xfff6c200, 0x00001cbd, 0xfff6e200,
    0x00001cc5, 0xffff0000, 0x00001ccd, 0xffff2000,
    0x00001cd5, 0xffff4000, 0x00001cde, 0xffff6100,
    0x00001ce6, 0xffff8100, 0x00001cee, 0xffffa100,
    0x00001cf6, 0xffffc200, 0x00001cff, 0xffffe200
};

static void Blit_RGB565_ARGB8888(SDL_BlitInfo * info)
{
    Blit_RGB565_32(info, RGB565_ARGB8888_LUT);
}

// Special optimized blit for RGB 5-6-5 --> ABGR 8-8-8-8
static const Uint32 RGB565_ABGR8888_LUT[512] = {
    0xff000000, 0x00000000, 0xff080000, 0x00002000,
    0xff100000, 0x00004000, 0xff180000, 0x00006100,
    0xff200000, 0x00008100, 0xff290000, 0x0000a100,
    0xff310000, 0x0000c200, 0xff390000, 0x0000e200,
    0xff410000, 0x00000008, 0xff4a0000, 0x00002008,
    0xff520000, 0x00004008, 0xff5a0000, 0x00006108,
    0xff620000, 0x00008108, 0xff6a0000, 0x0000a108,
    0xff730000, 0x0000c208, 0xff7b0000, 0x0000e208,
    0xff830000, 0x00000010, 0xff8b0000, 0x00002010,
    0xff940000, 0x00004010, 0xff9c0000, 0x00006110,
    0xffa40000, 0x00008110, 0xffac0000, 0x0000a110,
    0xffb40000, 0x0000c210, 0xffbd0000, 0x0000e210,
    0xffc50000, 0x00000018, 0xffcd0000, 0x00002018,
    0xffd50000, 0x00004018, 0xffde0000, 0x00006118,
    0xffe60000, 0x00008118, 0xffee0000, 0x0000a118,
    0xfff60000, 0x0000c218, 0xffff0000, 0x0000e218,
    0xff000400, 0x00000020, 0xff080400, 0x00002020,
    0xff100400, 0x00004020, 0xff180400, 0x00006120,
    0xff200400, 0x00008120, 0xff290400, 0x0000a120,
    0xff310400, 0x0000c220, 0xff390400, 0x0000e220,
    0xff410400, 0x00000029, 0xff4a0400, 0x00002029,
    0xff520400, 0x00004029, 0xff5a0400, 0x00006129,
    0xff620400, 0x00008129, 0xff6a0400, 0x0000a129,
    0xff730400, 0x0000c229, 0xff7b0400, 0x0000e229,
    0xff830400, 0x00000031, 0xff8b0400, 0x00002031,
    0xff940400, 0x00004031, 0xff9c0400, 0x00006131,
    0xffa40400, 0x00008131, 0xffac0400, 0x0000a131,
    0xffb40400, 0x0000c231, 0xffbd0400, 0x0000e231,
    0xffc50400, 0x00000039, 0xffcd0400, 0x00002039,
    0xffd50400, 0x00004039, 0xffde0400, 0x00006139,
    0xffe60400, 0x00008139, 0xffee0400, 0x0000a139,
    0xfff60400, 0x0000c239, 0xffff0400, 0x0000e239,
    0xff000800, 0x00000041, 0xff080800, 0x00002041,
    0xff100800, 0x00004041, 0xff180800, 0x00006141,
    0xff200800, 0x00008141, 0xff290800, 0x0000a141,
    0xff310800, 0x0000c241, 0xff390800, 0x0000e241,
    0xff410800, 0x0000004a, 0xff4a0800, 0x0000204a,
    0xff520800, 0x0000404a, 0xff5a0800, 0x0000614a,
    0xff620800, 0x0000814a, 0xff6a0800, 0x0000a14a,
    0xff730800, 0x0000c24a, 0xff7b0800, 0x0000e24a,
    0xff830800, 0x00000052, 0xff8b0800, 0x00002052,
    0xff940800, 0x00004052, 0xff9c0800, 0x00006152,
    0xffa40800, 0x00008152, 0xffac0800, 0x0000a152,
    0xffb40800, 0x0000c252, 0xffbd0800, 0x0000e252,
    0xffc50800, 0x0000005a, 0xffcd0800, 0x0000205a,
    0xffd50800, 0x0000405a, 0xffde0800, 0x0000615a,
    0xffe60800, 0x0000815a, 0xffee0800, 0x0000a15a,
    0xfff60800, 0x0000c25a, 0xffff0800, 0x0000e25a,
    0xff000c00, 0x00000062, 0xff080c00, 0x00002062,
    0xff100c00, 0x00004062, 0xff180c00, 0x00006162,
    0xff200c00, 0x00008162, 0xff290c00, 0x0000a162,
    0xff310c00, 0x0000c262, 0xff390c00, 0x0000e262,
    0xff410c00, 0x0000006a, 0xff4a0c00, 0x0000206a,
    0xff520c00, 0x0000406a, 0xff5a0c00, 0x0000616a,
    0xff620c00, 0x0000816a, 0xff6a0c00, 0x0000a16a,
    0xff730c00, 0x0000c26a, 0xff7b0c00, 0x0000e26a,
    0xff830c00, 0x00000073, 0xff8b0c00, 0x00002073,
    0xff940c00, 0x00004073, 0xff9c0c00, 0x00006173,
    0xffa40c00, 0x00008173, 0xffac0c00, 0x0000a173,
    0xffb40c00, 0x0000c273, 0xffbd0c00, 0x0000e273,
    0xffc50c00, 0x0000007b, 0xffcd0c00, 0x0000207b,
    0xffd50c00, 0x0000407b, 0xffde0c00, 0x0000617b,
    0xffe60c00, 0x0000817b, 0xffee0c00, 0x0000a17b,
    0xfff60c00, 0x0000c27b, 0xffff0c00, 0x0000e27b,
    0xff001000, 0x00000083, 0xff081000, 0x00002083,
    0xff101000, 0x00004083, 0xff181000, 0x00006183,
    0xff201000, 0x00008183, 0xff291000, 0x0000a183,
    0xff311000, 0x0000c283, 0xff391000, 0x0000e283,
    0xff411000, 0x0000008b, 0xff4a1000, 0x0000208b,
    0xff521000, 0x0000408b, 0xff5a1000, 0x0000618b,
    0xff621000, 0x0000818b, 0xff6a1000, 0x0000a18b,
    0xff731000, 0x0000c28b, 0xff7b1000, 0x0000e28b,
    0xff831000, 0x00000094, 0xff8b1000, 0x00002094,
    0xff941000, 0x00004094, 0xff9c1000, 0x00006194,
    0xffa41000, 0x00008194, 0xffac1000, 0x0000a194,
    0xffb41000, 0x0000c294, 0xffbd1000, 0x0000e294,
    0xffc51000, 0x0000009c, 0xffcd1000, 0x0000209c,
    0xffd51000, 0x0000409c, 0xffde1000, 0x0000619c,
    0xffe61000, 0x0000819c, 0xffee1000, 0x0000a19c,
    0xfff61000, 0x0000c29c, 0xffff1000, 0x0000e29c,
    0xff001400, 0x000000a4, 0xff081400, 0x000020a4,
    0xff101400, 0x000040a4, 0xff181400, 0x000061a4,
    0xff201400, 0x000081a4, 0xff291400, 0x0000a1a4,
    0xff311400, 0x0000c2a4, 0xff391400, 0x0000e2a4,
    0xff411400, 0x000000ac, 0xff4a1400, 0x000020ac,
    0xff521400, 0x000040ac, 0xff5a1400, 0x000061ac,
    0xff621400, 0x000081ac, 0xff6a1400, 0x0000a1ac,
    0xff731400, 0x0000c2ac, 0xff7b1400, 0x0000e2ac,
    0xff831400, 0x000000b4, 0xff8b1400, 0x000020b4,
    0xff941400, 0x000040b4, 0xff9c1400, 0x000061b4,
    0xffa41400, 0x000081b4, 0xffac1400, 0x0000a1b4,
    0xffb41400, 0x0000c2b4, 0xffbd1400, 0x0000e2b4,
    0xffc51400, 0x000000bd, 0xffcd1400, 0x000020bd,
    0xffd51400, 0x000040bd, 0xffde1400, 0x000061bd,
    0xffe61400, 0x000081bd, 0xffee1400, 0x0000a1bd,
    0xfff61400, 0x0000c2bd, 0xffff1400, 0x0000e2bd,
    0xff001800, 0x000000c5, 0xff081800, 0x000020c5,
    0xff101800, 0x000040c5, 0xff181800, 0x000061c5,
    0xff201800, 0x000081c5, 0xff291800, 0x0000a1c5,
    0xff311800, 0x0000c2c5, 0xff391800, 0x0000e2c5,
    0xff411800, 0x000000cd, 0xff4a1800, 0x000020cd,
    0xff521800, 0x000040cd, 0xff5a1800, 0x000061cd,
    0xff621800, 0x000081cd, 0xff6a1800, 0x0000a1cd,
    0xff731800, 0x0000c2cd, 0xff7b1800, 0x0000e2cd,
    0xff831800, 0x000000d5, 0xff8b1800, 0x000020d5,
    0xff941800, 0x000040d5, 0xff9c1800, 0x000061d5,
    0xffa41800, 0x000081d5, 0xffac1800, 0x0000a1d5,
    0xffb41800, 0x0000c2d5, 0xffbd1800, 0x0000e2d5,
    0xffc51800, 0x000000de, 0xffcd1800, 0x000020de,
    0xffd51800, 0x000040de, 0xffde1800, 0x000061de,
    0xffe61800, 0x000081de, 0xffee1800, 0x0000a1de,
    0xfff61800, 0x0000c2de, 0xffff1800, 0x0000e2de,
    0xff001c00, 0x000000e6, 0xff081c00, 0x000020e6,
    0xff101c00, 0x000040e6, 0xff181c00, 0x000061e6,
    0xff201c00, 0x000081e6, 0xff291c00, 0x0000a1e6,
    0xff311c00, 0x0000c2e6, 0xff391c00, 0x0000e2e6,
    0xff411c00, 0x000000ee, 0xff4a1c00, 0x000020ee,
    0xff521c00, 0x000040ee, 0xff5a1c00, 0x000061ee,
    0xff621c00, 0x000081ee, 0xff6a1c00, 0x0000a1ee,
    0xff731c00, 0x0000c2ee, 0xff7b1c00, 0x0000e2ee,
    0xff831c00, 0x000000f6, 0xff8b1c00, 0x000020f6,
    0xff941c00, 0x000040f6, 0xff9c1c00, 0x000061f6,
    0xffa41c00, 0x000081f6, 0xffac1c00, 0x0000a1f6,
    0xffb41c00, 0x0000c2f6, 0xffbd1c00, 0x0000e2f6,
    0xffc51c00, 0x000000ff, 0xffcd1c00, 0x000020ff,
    0xffd51c00, 0x000040ff, 0xffde1c00, 0x000061ff,
    0xffe61c00, 0x000081ff, 0xffee1c00, 0x0000a1ff,
    0xfff61c00, 0x0000c2ff, 0xffff1c00, 0x0000e2ff
};

static void Blit_RGB565_ABGR8888(SDL_BlitInfo * info)
{
    Blit_RGB565_32(info, RGB565_ABGR8888_LUT);
}

// Special optimized blit for RGB 5-6-5 --> RGBA 8-8-8-8
static const Uint32 RGB565_RGBA8888_LUT[512] = {
    0x000000ff, 0x00000000, 0x000008ff, 0x00200000,
    0x000010ff, 0x00400000, 0x000018ff, 0x00610000,
    0x000020ff, 0x00810000, 0x000029ff, 0x00a10000,
    0x000031ff, 0x00c20000, 0x000039ff, 0x00e20000,
    0x000041ff, 0x08000000, 0x00004aff, 0x08200000,
    0x000052ff, 0x08400000, 0x00005aff, 0x08610000,
    0x000062ff, 0x08810000, 0x00006aff, 0x08a10000,
    0x000073ff, 0x08c20000, 0x00007bff, 0x08e20000,
    0x000083ff, 0x10000000, 0x00008bff, 0x10200000,
    0x000094ff, 0x10400000, 0x00009cff, 0x10610000,
    0x0000a4ff, 0x10810000, 0x0000acff, 0x10a10000,
    0x0000b4ff, 0x10c20000, 0x0000bdff, 0x10e20000,
    0x0000c5ff, 0x18000000, 0x0000cdff, 0x18200000,
    0x0000d5ff, 0x18400000, 0x0000deff, 0x18610000,
    0x0000e6ff, 0x18810000, 0x0000eeff, 0x18a10000,
    0x0000f6ff, 0x18c20000, 0x0000ffff, 0x18e20000,
    0x000400ff, 0x20000000, 0x000408ff, 0x20200000,
    0x000410ff, 0x20400000, 0x000418ff, 0x20610000,
    0x000420ff, 0x20810000, 0x000429ff, 0x20a10000,
    0x000431ff, 0x20c20000, 0x000439ff, 0x20e20000,
    0x000441ff, 0x29000000, 0x00044aff, 0x29200000,
    0x000452ff, 0x29400000, 0x00045aff, 0x29610000,
    0x000462ff, 0x29810000, 0x00046aff, 0x29a10000,
    0x000473ff, 0x29c20000, 0x00047bff, 0x29e20000,
    0x000483ff, 0x31000000, 0x00048bff, 0x31200000,
    0x000494ff, 0x31400000, 0x00049cff, 0x31610000,
    0x0004a4ff, 0x31810000, 0x0004acff, 0x31a10000,
    0x0004b4ff, 0x31c20000, 0x0004bdff, 0x31e20000,
    0x0004c5ff, 0x39000000, 0x0004cdff, 0x39200000,
    0x0004d5ff, 0x39400000, 0x0004deff, 0x39610000,
    0x0004e6ff, 0x39810000, 0x0004eeff, 0x39a10000,
    0x0004f6ff, 0x39c20000, 0x0004ffff, 0x39e20000,
    0x000800ff, 0x41000000, 0x000808ff, 0x41200000,
    0x000810ff, 0x41400000, 0x000818ff, 0x41610000,
    0x000820ff, 0x41810000, 0x000829ff, 0x41a10000,
    0x000831ff, 0x41c20000, 0x000839ff, 0x41e20000,
    0x000841ff, 0x4a000000, 0x00084aff, 0x4a200000,
    0x000852ff, 0x4a400000, 0x00085aff, 0x4a610000,
    0x000862ff, 0x4a810000, 0x00086aff, 0x4aa10000,
    0x000873ff, 0x4ac20000, 0x00087bff, 0x4ae20000,
    0x000883ff, 0x52000000, 0x00088bff, 0x52200000,
    0x000894ff, 0x52400000, 0x00089cff, 0x52610000,
    0x0008a4ff, 0x52810000, 0x0008acff, 0x52a10000,
    0x0008b4ff, 0x52c20000, 0x0008bdff, 0x52e20000,
    0x0008c5ff, 0x5a000000, 0x0008cdff, 0x5a200000,
    0x0008d5ff, 0x5a400000, 0x0008deff, 0x5a610000,
    0x0008e6ff, 0x5a810000, 0x0008eeff, 0x5aa10000,
    0x0008f6ff, 0x5ac20000, 0x0008ffff, 0x5ae20000,
    0x000c00ff, 0x62000000, 0x000c08ff, 0x62200000,
    0x000c10ff, 0x62400000, 0x000c18ff, 0x62610000,
    0x000c20ff, 0x62810000, 0x000c29ff, 0x62a10000,
    0x000c31ff, 0x62c20000, 0x000c39ff, 0x62e20000,
    0x000c41ff, 0x6a000000, 0x000c4aff, 0x6a200000,
    0x000c52ff, 0x6a400000, 0x000c5aff, 0x6a610000,
    0x000c62ff, 0x6a810000, 0x000c6aff, 0x6aa10000,
    0x000c73ff, 0x6ac20000, 0x000c7bff, 0x6ae20000,
    0x000c83ff, 0x73000000, 0x000c8bff, 0x73200000,
    0x000c94ff, 0x73400000, 0x000c9cff, 0x73610000,
    0x000ca4ff, 0x73810000, 0x000cacff, 0x73a10000,
    0x000cb4ff, 0x73c20000, 0x000cbdff, 0x73e20000,
    0x000cc5ff, 0x7b000000, 0x000ccdff, 0x7b200000,
    0x000cd5ff, 0x7b400000, 0x000cdeff, 0x7b610000,
    0x000ce6ff, 0x7b810000, 0x000ceeff, 0x7ba10000,
    0x000cf6ff, 0x7bc20000, 0x000cffff, 0x7be20000,
    0x001000ff, 0x83000000, 0x001008ff, 0x83200000,
    0x001010ff, 0x83400000, 0x001018ff, 0x83610000,
    0x001020ff, 0x83810000, 0x001029ff, 0x83a10000,
    0x001031ff, 0x83c20000, 0x001039ff, 0x83e20000,
    0x001041ff, 0x8b000000, 0x00104aff, 0x8b200000,
    0x001052ff, 0x8b400000, 0x00105aff, 0x8b610000,
    0x001062ff, 0x8b810000, 0x00106aff, 0x8ba10000,
    0x001073ff, 0x8bc20000, 0x00107bff, 0x8be20000,
    0x001083ff, 0x94000000, 0x00108bff, 0x94200000,
    0x001094ff, 0x94400000, 0x00109cff, 0x94610000,
    0x0010a4ff, 0x94810000, 0x0010acff, 0x94a10000,
    0x0010b4ff, 0x94c20000, 0x0010bdff, 0x94e20000,
    0x0010c5ff, 0x9c000000, 0x0010cdff, 0x9c200000,
    0x0010d5ff, 0x9c400000, 0x0010deff, 0x9c610000,
    0x0010e6ff, 0x9c810000, 0x0010eeff, 0x9ca10000,
    0x0010f6ff, 0x9cc20000, 0x0010ffff, 0x9ce20000,
    0x001400ff, 0xa4000000, 0x001408ff, 0xa4200000,
    0x001410ff, 0xa4400000, 0x001418ff, 0xa4610000,
    0x001420ff, 0xa4810000, 0x001429ff, 0xa4a10000,
    0x001431ff, 0xa4c20000, 0x001439ff, 0xa4e20000,
    0x001441ff, 0xac000000, 0x00144aff, 0xac200000,
    0x001452ff, 0xac400000, 0x00145aff, 0xac610000,
    0x001462ff, 0xac810000, 0x00146aff, 0xaca10000,
    0x001473ff, 0xacc20000, 0x00147bff, 0xace20000,
    0x001483ff, 0xb4000000, 0x00148bff, 0xb4200000,
    0x001494ff, 0xb4400000, 0x00149cff, 0xb4610000,
    0x0014a4ff, 0xb4810000, 0x0014acff, 0xb4a10000,
    0x0014b4ff, 0xb4c20000, 0x0014bdff, 0xb4e20000,
    0x0014c5ff, 0xbd000000, 0x0014cdff, 0xbd200000,
    0x0014d5ff, 0xbd400000, 0x0014deff, 0xbd610000,
    0x0014e6ff, 0xbd810000, 0x0014eeff, 0xbda10000,
    0x0014f6ff, 0xbdc20000, 0x0014ffff, 0xbde20000,
    0x001800ff, 0xc5000000, 0x001808ff, 0xc5200000,
    0x001810ff, 0xc5400000, 0x001818ff, 0xc5610000,
    0x001820ff, 0xc5810000, 0x001829ff, 0xc5a10000,
    0x001831ff, 0xc5c20000, 0x001839ff, 0xc5e20000,
    0x001841ff, 0xcd000000, 0x00184aff, 0xcd200000,
    0x001852ff, 0xcd400000, 0x00185aff, 0xcd610000,
    0x001862ff, 0xcd810000, 0x00186aff, 0xcda10000,
    0x001873ff, 0xcdc20000, 0x00187bff, 0xcde20000,
    0x001883ff, 0xd5000000, 0x00188bff, 0xd5200000,
    0x001894ff, 0xd5400000, 0x00189cff, 0xd5610000,
    0x0018a4ff, 0xd5810000, 0x0018acff, 0xd5a10000,
    0x0018b4ff, 0xd5c20000, 0x0018bdff, 0xd5e20000,
    0x0018c5ff, 0xde000000, 0x0018cdff, 0xde200000,
    0x0018d5ff, 0xde400000, 0x0018deff, 0xde610000,
    0x0018e6ff, 0xde810000, 0x0018eeff, 0xdea10000,
    0x0018f6ff, 0xdec20000, 0x0018ffff, 0xdee20000,
    0x001c00ff, 0xe6000000, 0x001c08ff, 0xe6200000,
    0x001c10ff, 0xe6400000, 0x001c18ff, 0xe6610000,
    0x001c20ff, 0xe6810000, 0x001c29ff, 0xe6a10000,
    0x001c31ff, 0xe6c20000, 0x001c39ff, 0xe6e20000,
    0x001c41ff, 0xee000000, 0x001c4aff, 0xee200000,
    0x001c52ff, 0xee400000, 0x001c5aff, 0xee610000,
    0x001c62ff, 0xee810000, 0x001c6aff, 0xeea10000,
    0x001c73ff, 0xeec20000, 0x001c7bff, 0xeee20000,
    0x001c83ff, 0xf6000000, 0x001c8bff, 0xf6200000,
    0x001c94ff, 0xf6400000, 0x001c9cff, 0xf6610000,
    0x001ca4ff, 0xf6810000, 0x001cacff, 0xf6a10000,
    0x001cb4ff, 0xf6c20000, 0x001cbdff, 0xf6e20000,
    0x001cc5ff, 0xff000000, 0x001ccdff, 0xff200000,
    0x001cd5ff, 0xff400000, 0x001cdeff, 0xff610000,
    0x001ce6ff, 0xff810000, 0x001ceeff, 0xffa10000,
    0x001cf6ff, 0xffc20000, 0x001cffff, 0xffe20000,
};

static void Blit_RGB565_RGBA8888(SDL_BlitInfo * info)
{
    Blit_RGB565_32(info, RGB565_RGBA8888_LUT);
}

// Special optimized blit for RGB 5-6-5 --> BGRA 8-8-8-8
static const Uint32 RGB565_BGRA8888_LUT[512] = {
    0x00000000, 0x000000ff, 0x08000000, 0x002000ff,
    0x10000000, 0x004000ff, 0x18000000, 0x006100ff,
    0x20000000, 0x008100ff, 0x29000000, 0x00a100ff,
    0x31000000, 0x00c200ff, 0x39000000, 0x00e200ff,
    0x41000000, 0x000008ff, 0x4a000000, 0x002008ff,
    0x52000000, 0x004008ff, 0x5a000000, 0x006108ff,
    0x62000000, 0x008108ff, 0x6a000000, 0x00a108ff,
    0x73000000, 0x00c208ff, 0x7b000000, 0x00e208ff,
    0x83000000, 0x000010ff, 0x8b000000, 0x002010ff,
    0x94000000, 0x004010ff, 0x9c000000, 0x006110ff,
    0xa4000000, 0x008110ff, 0xac000000, 0x00a110ff,
    0xb4000000, 0x00c210ff, 0xbd000000, 0x00e210ff,
    0xc5000000, 0x000018ff, 0xcd000000, 0x002018ff,
    0xd5000000, 0x004018ff, 0xde000000, 0x006118ff,
    0xe6000000, 0x008118ff, 0xee000000, 0x00a118ff,
    0xf6000000, 0x00c218ff, 0xff000000, 0x00e218ff,
    0x00040000, 0x000020ff, 0x08040000, 0x002020ff,
    0x10040000, 0x004020ff, 0x18040000, 0x006120ff,
    0x20040000, 0x008120ff, 0x29040000, 0x00a120ff,
    0x31040000, 0x00c220ff, 0x39040000, 0x00e220ff,
    0x41040000, 0x000029ff, 0x4a040000, 0x002029ff,
    0x52040000, 0x004029ff, 0x5a040000, 0x006129ff,
    0x62040000, 0x008129ff, 0x6a040000, 0x00a129ff,
    0x73040000, 0x00c229ff, 0x7b040000, 0x00e229ff,
    0x83040000, 0x000031ff, 0x8b040000, 0x002031ff,
    0x94040000, 0x004031ff, 0x9c040000, 0x006131ff,
    0xa4040000, 0x008131ff, 0xac040000, 0x00a131ff,
    0xb4040000, 0x00c231ff, 0xbd040000, 0x00e231ff,
    0xc5040000, 0x000039ff, 0xcd040000, 0x002039ff,
    0xd5040000, 0x004039ff, 0xde040000, 0x006139ff,
    0xe6040000, 0x008139ff, 0xee040000, 0x00a139ff,
    0xf6040000, 0x00c239ff, 0xff040000, 0x00e239ff,
    0x00080000, 0x000041ff, 0x08080000, 0x002041ff,
    0x10080000, 0x004041ff, 0x18080000, 0x006141ff,
    0x20080000, 0x008141ff, 0x29080000, 0x00a141ff,
    0x31080000, 0x00c241ff, 0x39080000, 0x00e241ff,
    0x41080000, 0x00004aff, 0x4a080000, 0x00204aff,
    0x52080000, 0x00404aff, 0x5a080000, 0x00614aff,
    0x62080000, 0x00814aff, 0x6a080000, 0x00a14aff,
    0x73080000, 0x00c24aff, 0x7b080000, 0x00e24aff,
    0x83080000, 0x000052ff, 0x8b080000, 0x002052ff,
    0x94080000, 0x004052ff, 0x9c080000, 0x006152ff,
    0xa4080000, 0x008152ff, 0xac080000, 0x00a152ff,
    0xb4080000, 0x00c252ff, 0xbd080000, 0x00e252ff,
    0xc5080000, 0x00005aff, 0xcd080000, 0x00205aff,
    0xd5080000, 0x00405aff, 0xde080000, 0x00615aff,
    0xe6080000, 0x00815aff, 0xee080000, 0x00a15aff,
    0xf6080000, 0x00c25aff, 0xff080000, 0x00e25aff,
    0x000c0000, 0x000062ff, 0x080c0000, 0x002062ff,
    0x100c0000, 0x004062ff, 0x180c0000, 0x006162ff,
    0x200c0000, 0x008162ff, 0x290c0000, 0x00a162ff,
    0x310c0000, 0x00c262ff, 0x390c0000, 0x00e262ff,
    0x410c0000, 0x00006aff, 0x4a0c0000, 0x00206aff,
    0x520c0000, 0x00406aff, 0x5a0c0000, 0x00616aff,
    0x620c0000, 0x00816aff, 0x6a0c0000, 0x00a16aff,
    0x730c0000, 0x00c26aff, 0x7b0c0000, 0x00e26aff,
    0x830c0000, 0x000073ff, 0x8b0c0000, 0x002073ff,
    0x940c0000, 0x004073ff, 0x9c0c0000, 0x006173ff,
    0xa40c0000, 0x008173ff, 0xac0c0000, 0x00a173ff,
    0xb40c0000, 0x00c273ff, 0xbd0c0000, 0x00e273ff,
    0xc50c0000, 0x00007bff, 0xcd0c0000, 0x00207bff,
    0xd50c0000, 0x00407bff, 0xde0c0000, 0x00617bff,
    0xe60c0000, 0x00817bff, 0xee0c0000, 0x00a17bff,
    0xf60c0000, 0x00c27bff, 0xff0c0000, 0x00e27bff,
    0x00100000, 0x000083ff, 0x08100000, 0x002083ff,
    0x10100000, 0x004083ff, 0x18100000, 0x006183ff,
    0x20100000, 0x008183ff, 0x29100000, 0x00a183ff,
    0x31100000, 0x00c283ff, 0x39100000, 0x00e283ff,
    0x41100000, 0x00008bff, 0x4a100000, 0x00208bff,
    0x52100000, 0x00408bff, 0x5a100000, 0x00618bff,
    0x62100000, 0x00818bff, 0x6a100000, 0x00a18bff,
    0x73100000, 0x00c28bff, 0x7b100000, 0x00e28bff,
    0x83100000, 0x000094ff, 0x8b100000, 0x002094ff,
    0x94100000, 0x004094ff, 0x9c100000, 0x006194ff,
    0xa4100000, 0x008194ff, 0xac100000, 0x00a194ff,
    0xb4100000, 0x00c294ff, 0xbd100000, 0x00e294ff,
    0xc5100000, 0x00009cff, 0xcd100000, 0x00209cff,
    0xd5100000, 0x00409cff, 0xde100000, 0x00619cff,
    0xe6100000, 0x00819cff, 0xee100000, 0x00a19cff,
    0xf6100000, 0x00c29cff, 0xff100000, 0x00e29cff,
    0x00140000, 0x0000a4ff, 0x08140000, 0x0020a4ff,
    0x10140000, 0x0040a4ff, 0x18140000, 0x0061a4ff,
    0x20140000, 0x0081a4ff, 0x29140000, 0x00a1a4ff,
    0x31140000, 0x00c2a4ff, 0x39140000, 0x00e2a4ff,
    0x41140000, 0x0000acff, 0x4a140000, 0x0020acff,
    0x52140000, 0x0040acff, 0x5a140000, 0x0061acff,
    0x62140000, 0x0081acff, 0x6a140000, 0x00a1acff,
    0x73140000, 0x00c2acff, 0x7b140000, 0x00e2acff,
    0x83140000, 0x0000b4ff, 0x8b140000, 0x0020b4ff,
    0x94140000, 0x0040b4ff, 0x9c140000, 0x0061b4ff,
    0xa4140000, 0x0081b4ff, 0xac140000, 0x00a1b4ff,
    0xb4140000, 0x00c2b4ff, 0xbd140000, 0x00e2b4ff,
    0xc5140000, 0x0000bdff, 0xcd140000, 0x0020bdff,
    0xd5140000, 0x0040bdff, 0xde140000, 0x0061bdff,
    0xe6140000, 0x0081bdff, 0xee140000, 0x00a1bdff,
    0xf6140000, 0x00c2bdff, 0xff140000, 0x00e2bdff,
    0x00180000, 0x0000c5ff, 0x08180000, 0x0020c5ff,
    0x10180000, 0x0040c5ff, 0x18180000, 0x0061c5ff,
    0x20180000, 0x0081c5ff, 0x29180000, 0x00a1c5ff,
    0x31180000, 0x00c2c5ff, 0x39180000, 0x00e2c5ff,
    0x41180000, 0x0000cdff, 0x4a180000, 0x0020cdff,
    0x52180000, 0x0040cdff, 0x5a180000, 0x0061cdff,
    0x62180000, 0x0081cdff, 0x6a180000, 0x00a1cdff,
    0x73180000, 0x00c2cdff, 0x7b180000, 0x00e2cdff,
    0x83180000, 0x0000d5ff, 0x8b180000, 0x0020d5ff,
    0x94180000, 0x0040d5ff, 0x9c180000, 0x0061d5ff,
    0xa4180000, 0x0081d5ff, 0xac180000, 0x00a1d5ff,
    0xb4180000, 0x00c2d5ff, 0xbd180000, 0x00e2d5ff,
    0xc5180000, 0x0000deff, 0xcd180000, 0x0020deff,
    0xd5180000, 0x0040deff, 0xde180000, 0x0061deff,
    0xe6180000, 0x0081deff, 0xee180000, 0x00a1deff,
    0xf6180000, 0x00c2deff, 0xff180000, 0x00e2deff,
    0x001c0000, 0x0000e6ff, 0x081c0000, 0x0020e6ff,
    0x101c0000, 0x0040e6ff, 0x181c0000, 0x0061e6ff,
    0x201c0000, 0x0081e6ff, 0x291c0000, 0x00a1e6ff,
    0x311c0000, 0x00c2e6ff, 0x391c0000, 0x00e2e6ff,
    0x411c0000, 0x0000eeff, 0x4a1c0000, 0x0020eeff,
    0x521c0000, 0x0040eeff, 0x5a1c0000, 0x0061eeff,
    0x621c0000, 0x0081eeff, 0x6a1c0000, 0x00a1eeff,
    0x731c0000, 0x00c2eeff, 0x7b1c0000, 0x00e2eeff,
    0x831c0000, 0x0000f6ff, 0x8b1c0000, 0x0020f6ff,
    0x941c0000, 0x0040f6ff, 0x9c1c0000, 0x0061f6ff,
    0xa41c0000, 0x0081f6ff, 0xac1c0000, 0x00a1f6ff,
    0xb41c0000, 0x00c2f6ff, 0xbd1c0000, 0x00e2f6ff,
    0xc51c0000, 0x0000ffff, 0xcd1c0000, 0x0020ffff,
    0xd51c0000, 0x0040ffff, 0xde1c0000, 0x0061ffff,
    0xe61c0000, 0x0081ffff, 0xee1c0000, 0x00a1ffff,
    0xf61c0000, 0x00c2ffff, 0xff1c0000, 0x00e2ffff
};

static void Blit_RGB565_BGRA8888(SDL_BlitInfo * info)
{
    Blit_RGB565_32(info, RGB565_BGRA8888_LUT);
}

/* *INDENT-ON* */ // clang-format on

#endif // SDL_HAVE_BLIT_N_RGB565

// blits 16 bit RGB<->RGBA with both surfaces having the same R,G,B fields
static void Blit2to2MaskAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint16 *src = (Uint16 *)info->src;
    int srcskip = info->src_skip;
    Uint16 *dst = (Uint16 *)info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;

    if (dstfmt->Amask) {
        // RGB->RGBA, SET_ALPHA
        Uint16 mask = ((Uint32)info->a >> (8 - dstfmt->Abits)) << dstfmt->Ashift;

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                *dst = *src | mask;
                ++dst;
                ++src;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src = (Uint16 *)((Uint8 *)src + srcskip);
            dst = (Uint16 *)((Uint8 *)dst + dstskip);
        }
    } else {
        // RGBA->RGB, NO_ALPHA
        Uint16 mask = srcfmt->Rmask | srcfmt->Gmask | srcfmt->Bmask;

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                *dst = *src & mask;
                ++dst;
                ++src;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src = (Uint16 *)((Uint8 *)src + srcskip);
            dst = (Uint16 *)((Uint8 *)dst + dstskip);
        }
    }
}

// blits 32 bit RGB<->RGBA with both surfaces having the same R,G,B fields
static void Blit4to4MaskAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *src = (Uint32 *)info->src;
    int srcskip = info->src_skip;
    Uint32 *dst = (Uint32 *)info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;

    if (dstfmt->Amask) {
        // RGB->RGBA, SET_ALPHA
        Uint32 mask = ((Uint32)info->a >> (8 - dstfmt->Abits)) << dstfmt->Ashift;

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                *dst = *src | mask;
                ++dst;
                ++src;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src = (Uint32 *)((Uint8 *)src + srcskip);
            dst = (Uint32 *)((Uint8 *)dst + dstskip);
        }
    } else {
        // RGBA->RGB, NO_ALPHA
        Uint32 mask = srcfmt->Rmask | srcfmt->Gmask | srcfmt->Bmask;

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP_TRIVIAL(
            {
                *dst = *src & mask;
                ++dst;
                ++src;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src = (Uint32 *)((Uint8 *)src + srcskip);
            dst = (Uint32 *)((Uint8 *)dst + dstskip);
        }
    }
}

// permutation for mapping srcfmt to dstfmt, overloading or not the alpha channel
static void get_permutation(const SDL_PixelFormatDetails *srcfmt, const SDL_PixelFormatDetails *dstfmt,
                            int *_p0, int *_p1, int *_p2, int *_p3, int *_alpha_channel)
{
    int alpha_channel = 0, p0, p1, p2, p3;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    int Pixel = 0x04030201; // identity permutation
#else
    int Pixel = 0x01020304; // identity permutation
    int srcbpp = srcfmt->bytes_per_pixel;
    int dstbpp = dstfmt->bytes_per_pixel;
#endif

    if (srcfmt->Amask) {
        RGBA_FROM_PIXEL(Pixel, srcfmt, p0, p1, p2, p3);
    } else {
        RGB_FROM_PIXEL(Pixel, srcfmt, p0, p1, p2);
        p3 = 0;
    }

    if (dstfmt->Amask) {
        if (srcfmt->Amask) {
            PIXEL_FROM_RGBA(Pixel, dstfmt, p0, p1, p2, p3);
        } else {
            PIXEL_FROM_RGBA(Pixel, dstfmt, p0, p1, p2, 0);
        }
    } else {
        PIXEL_FROM_RGB(Pixel, dstfmt, p0, p1, p2);
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    p0 = Pixel & 0xFF;
    p1 = (Pixel >> 8) & 0xFF;
    p2 = (Pixel >> 16) & 0xFF;
    p3 = (Pixel >> 24) & 0xFF;
#else
    p3 = Pixel & 0xFF;
    p2 = (Pixel >> 8) & 0xFF;
    p1 = (Pixel >> 16) & 0xFF;
    p0 = (Pixel >> 24) & 0xFF;
#endif

    if (p0 == 0) {
        p0 = 1;
        alpha_channel = 0;
    } else if (p1 == 0) {
        p1 = 1;
        alpha_channel = 1;
    } else if (p2 == 0) {
        p2 = 1;
        alpha_channel = 2;
    } else if (p3 == 0) {
        p3 = 1;
        alpha_channel = 3;
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#else
    if (srcbpp == 3 && dstbpp == 4) {
        if (p0 != 1) {
            p0--;
        }
        if (p1 != 1) {
            p1--;
        }
        if (p2 != 1) {
            p2--;
        }
        if (p3 != 1) {
            p3--;
        }
    } else if (srcbpp == 4 && dstbpp == 3) {
        p0 = p1;
        p1 = p2;
        p2 = p3;
    }
#endif
    *_p0 = p0 - 1;
    *_p1 = p1 - 1;
    *_p2 = p2 - 1;
    *_p3 = p3 - 1;

    if (_alpha_channel) {
        *_alpha_channel = alpha_channel;
    }
}

static void BlitNtoN(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int dstbpp = dstfmt->bytes_per_pixel;
    unsigned alpha = dstfmt->Amask ? info->a : 0;

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 4->4
    if (srcbpp == 4 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format) &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

        // Find the appropriate permutation
        int alpha_channel, p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, &alpha_channel);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                dst[0] = src[p0];
                dst[1] = src[p1];
                dst[2] = src[p2];
                dst[3] = src[p3];
                dst[alpha_channel] = (Uint8)alpha;
                src += 4;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    // Blit with permutation: 4->3
    if (srcbpp == 4 && dstbpp == 3 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format)) {

        // Find the appropriate permutation
        int p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, NULL);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                dst[0] = src[p0];
                dst[1] = src[p1];
                dst[2] = src[p2];
                src += 4;
                dst += 3;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 3->4
    if (srcbpp == 3 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

        // Find the appropriate permutation
        int alpha_channel, p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, &alpha_channel);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                dst[0] = src[p0];
                dst[1] = src[p1];
                dst[2] = src[p2];
                dst[3] = src[p3];
                dst[alpha_channel] = (Uint8)alpha;
                src += 3;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            Uint32 Pixel;
            unsigned sR;
            unsigned sG;
            unsigned sB;
            DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, sR, sG, sB, alpha);
            dst += dstbpp;
            src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

static void BlitNtoNCopyAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int dstbpp = dstfmt->bytes_per_pixel;
    int c;

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 4->4
    if (srcbpp == 4 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format) &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

        // Find the appropriate permutation
        int p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, NULL);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                dst[0] = src[p0];
                dst[1] = src[p1];
                dst[2] = src[p2];
                dst[3] = src[p3];
                src += 4;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    while (height--) {
        for (c = width; c; --c) {
            Uint32 Pixel;
            unsigned sR, sG, sB, sA;
            DISEMBLE_RGBA(src, srcbpp, srcfmt, Pixel, sR, sG, sB, sA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, sR, sG, sB, sA);
            dst += dstbpp;
            src += srcbpp;
        }
        src += srcskip;
        dst += dstskip;
    }
}

static void Blit2to2Key(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint16 *srcp = (Uint16 *)info->src;
    int srcskip = info->src_skip;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip;
    Uint32 ckey = info->colorkey;
    Uint32 rgbmask = ~info->src_fmt->Amask;

    // Set up some basic variables
    srcskip /= 2;
    dstskip /= 2;
    ckey &= rgbmask;

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP_TRIVIAL(
        {
            if ( (*srcp & rgbmask) != ckey ) {
                *dstp = *srcp;
            }
            dstp++;
            srcp++;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        srcp += srcskip;
        dstp += dstskip;
    }
}

static void BlitNtoNKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint32 ckey = info->colorkey;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    int dstbpp = dstfmt->bytes_per_pixel;
    unsigned alpha = dstfmt->Amask ? info->a : 0;
    Uint32 rgbmask = ~srcfmt->Amask;
    int sfmt = srcfmt->format;
    int dfmt = dstfmt->format;

    // Set up some basic variables
    ckey &= rgbmask;

    // BPP 4, same rgb
    if (srcbpp == 4 && dstbpp == 4 && srcfmt->Rmask == dstfmt->Rmask && srcfmt->Gmask == dstfmt->Gmask && srcfmt->Bmask == dstfmt->Bmask) {
        Uint32 *src32 = (Uint32 *)src;
        Uint32 *dst32 = (Uint32 *)dst;

        if (dstfmt->Amask) {
            // RGB->RGBA, SET_ALPHA
            Uint32 mask = ((Uint32)info->a) << dstfmt->Ashift;
            while (height--) {
                /* *INDENT-OFF* */ // clang-format off
                DUFFS_LOOP_TRIVIAL(
                {
                    if ((*src32 & rgbmask) != ckey) {
                        *dst32 = *src32 | mask;
                    }
                    ++dst32;
                    ++src32;
                }, width);
                /* *INDENT-ON* */ // clang-format on
                src32 = (Uint32 *)((Uint8 *)src32 + srcskip);
                dst32 = (Uint32 *)((Uint8 *)dst32 + dstskip);
            }
            return;
        } else {
            // RGBA->RGB, NO_ALPHA
            Uint32 mask = srcfmt->Rmask | srcfmt->Gmask | srcfmt->Bmask;
            while (height--) {
                /* *INDENT-OFF* */ // clang-format off
                DUFFS_LOOP_TRIVIAL(
                {
                    if ((*src32 & rgbmask) != ckey) {
                        *dst32 = *src32 & mask;
                    }
                    ++dst32;
                    ++src32;
                }, width);
                /* *INDENT-ON* */ // clang-format on
                src32 = (Uint32 *)((Uint8 *)src32 + srcskip);
                dst32 = (Uint32 *)((Uint8 *)dst32 + dstskip);
            }
            return;
        }
    }

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 4->4
    if (srcbpp == 4 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format) &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

        // Find the appropriate permutation
        int alpha_channel, p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, &alpha_channel);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint32 *src32 = (Uint32*)src;

                if ((*src32 & rgbmask) != ckey) {
                    dst[0] = src[p0];
                    dst[1] = src[p1];
                    dst[2] = src[p2];
                    dst[3] = src[p3];
                    dst[alpha_channel] = (Uint8)alpha;
                }
                src += 4;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    // BPP 3, same rgb triplet
    if ((sfmt == SDL_PIXELFORMAT_RGB24 && dfmt == SDL_PIXELFORMAT_RGB24) ||
        (sfmt == SDL_PIXELFORMAT_BGR24 && dfmt == SDL_PIXELFORMAT_BGR24)) {

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        Uint8 k0 = ckey & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = (ckey >> 16) & 0xFF;
#else
        Uint8 k0 = (ckey >> 16) & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = ckey & 0xFF;
#endif

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint8 s0 = src[0];
                Uint8 s1 = src[1];
                Uint8 s2 = src[2];

                if (k0 != s0 || k1 != s1 || k2 != s2) {
                    dst[0] = s0;
                    dst[1] = s1;
                    dst[2] = s2;
                }
                src += 3;
                dst += 3;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    // BPP 3, inversed rgb triplet
    if ((sfmt == SDL_PIXELFORMAT_RGB24 && dfmt == SDL_PIXELFORMAT_BGR24) ||
        (sfmt == SDL_PIXELFORMAT_BGR24 && dfmt == SDL_PIXELFORMAT_RGB24)) {

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        Uint8 k0 = ckey & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = (ckey >> 16) & 0xFF;
#else
        Uint8 k0 = (ckey >> 16) & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = ckey & 0xFF;
#endif

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint8 s0 = src[0];
                Uint8 s1 = src[1];
                Uint8 s2 = src[2];
                if (k0 != s0 || k1 != s1 || k2 != s2) {
                    // Inversed RGB
                    dst[0] = s2;
                    dst[1] = s1;
                    dst[2] = s0;
                }
                src += 3;
                dst += 3;
            },
            width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

    // Blit with permutation: 4->3
    if (srcbpp == 4 && dstbpp == 3 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format)) {

        // Find the appropriate permutation
        int p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, NULL);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint32 *src32 = (Uint32*)src;
                if ((*src32 & rgbmask) != ckey) {
                    dst[0] = src[p0];
                    dst[1] = src[p1];
                    dst[2] = src[p2];
                }
                src += 4;
                dst += 3;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 3->4
    if (srcbpp == 3 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        Uint8 k0 = ckey & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = (ckey >> 16) & 0xFF;
#else
        Uint8 k0 = (ckey >> 16) & 0xFF;
        Uint8 k1 = (ckey >> 8) & 0xFF;
        Uint8 k2 = ckey & 0xFF;
#endif

        // Find the appropriate permutation
        int alpha_channel, p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, &alpha_channel);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint8 s0 = src[0];
                Uint8 s1 = src[1];
                Uint8 s2 = src[2];

                if (k0 != s0 || k1 != s1 || k2 != s2) {
                    dst[0] = src[p0];
                    dst[1] = src[p1];
                    dst[2] = src[p2];
                    dst[3] = src[p3];
                    dst[alpha_channel] = (Uint8)alpha;
                }
                src += 3;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            Uint32 Pixel;
            unsigned sR;
            unsigned sG;
            unsigned sB;
            RETRIEVE_RGB_PIXEL(src, srcbpp, Pixel);
            if ( (Pixel & rgbmask) != ckey ) {
                RGB_FROM_PIXEL(Pixel, srcfmt, sR, sG, sB);
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, sR, sG, sB, alpha);
            }
            dst += dstbpp;
            src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

static void BlitNtoNKeyCopyAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint32 ckey = info->colorkey;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    Uint32 rgbmask = ~srcfmt->Amask;

    Uint8 srcbpp;
    Uint8 dstbpp;
    Uint32 Pixel;
    unsigned sR, sG, sB, sA;

    // Set up some basic variables
    srcbpp = srcfmt->bytes_per_pixel;
    dstbpp = dstfmt->bytes_per_pixel;
    ckey &= rgbmask;

    // Fastpath: same source/destination format, with Amask, bpp 32, loop is vectorized. ~10x faster
    if (srcfmt->format == dstfmt->format) {

        if (srcfmt->format == SDL_PIXELFORMAT_ARGB8888 ||
            srcfmt->format == SDL_PIXELFORMAT_ABGR8888 ||
            srcfmt->format == SDL_PIXELFORMAT_BGRA8888 ||
            srcfmt->format == SDL_PIXELFORMAT_RGBA8888) {

            Uint32 *src32 = (Uint32 *)src;
            Uint32 *dst32 = (Uint32 *)dst;
            while (height--) {
                /* *INDENT-OFF* */ // clang-format off
                DUFFS_LOOP_TRIVIAL(
                {
                    if ((*src32 & rgbmask) != ckey) {
                        *dst32 = *src32;
                    }
                    ++src32;
                    ++dst32;
                },
                width);
                /* *INDENT-ON* */ // clang-format on
                src32 = (Uint32 *)((Uint8 *)src32 + srcskip);
                dst32 = (Uint32 *)((Uint8 *)dst32 + dstskip);
            }
        }
        return;
    }

#if HAVE_FAST_WRITE_INT8
    // Blit with permutation: 4->4
    if (srcbpp == 4 && dstbpp == 4 &&
        !SDL_ISPIXELFORMAT_10BIT(srcfmt->format) &&
        !SDL_ISPIXELFORMAT_10BIT(dstfmt->format)) {

        // Find the appropriate permutation
        int p0, p1, p2, p3;
        get_permutation(srcfmt, dstfmt, &p0, &p1, &p2, &p3, NULL);

        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint32 *src32 = (Uint32*)src;
                if ((*src32 & rgbmask) != ckey) {
                    dst[0] = src[p0];
                    dst[1] = src[p1];
                    dst[2] = src[p2];
                    dst[3] = src[p3];
                }
                src += 4;
                dst += 4;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

    while (height--) {
        /* *INDENT-OFF* */ // clang-format off
        DUFFS_LOOP(
        {
            DISEMBLE_RGBA(src, srcbpp, srcfmt, Pixel, sR, sG, sB, sA);
            if ( (Pixel & rgbmask) != ckey ) {
                  ASSEMBLE_RGBA(dst, dstbpp, dstfmt, sR, sG, sB, sA);
            }
            dst += dstbpp;
            src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ // clang-format on
        src += srcskip;
        dst += dstskip;
    }
}

// Convert between two 8888 pixels with differing formats.
#define SWIZZLE_8888_SRC_ALPHA(src, dst, srcfmt, dstfmt)                \
    do {                                                                \
        dst = (((src >> srcfmt->Rshift) & 0xFF) << dstfmt->Rshift) |    \
              (((src >> srcfmt->Gshift) & 0xFF) << dstfmt->Gshift) |    \
              (((src >> srcfmt->Bshift) & 0xFF) << dstfmt->Bshift) |    \
              (((src >> srcfmt->Ashift) & 0xFF) << dstfmt->Ashift);     \
    } while (0)

#define SWIZZLE_8888_DST_ALPHA(src, dst, srcfmt, dstfmt, dstAmask)      \
    do {                                                                \
        dst = (((src >> srcfmt->Rshift) & 0xFF) << dstfmt->Rshift) |    \
              (((src >> srcfmt->Gshift) & 0xFF) << dstfmt->Gshift) |    \
              (((src >> srcfmt->Bshift) & 0xFF) << dstfmt->Bshift) |    \
              dstAmask;                                                 \
    } while (0)

#ifdef SDL_SSE4_1_INTRINSICS

static void SDL_TARGETING("sse4.1") Blit8888to8888PixelSwizzleSSE41(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = (!srcfmt->Amask || !dstfmt->Amask);
    Uint32 srcAmask, srcAshift;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(srcfmt, &srcAmask, &srcAshift);
    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const __m128i mask_offsets = _mm_set_epi8(
        12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    const __m128i convert_mask = _mm_add_epi32(
        _mm_set1_epi32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift) |
            ((srcAshift >> 3) << dstAshift)),
        mask_offsets);

    const __m128i alpha_fill_mask = _mm_set1_epi32((int)dstAmask);

    while (height--) {
        int i = 0;

        for (; i + 4 <= width; i += 4) {
            // Load 4 src pixels
            __m128i src128 = _mm_loadu_si128((__m128i *)src);

            // Convert to dst format
            src128 = _mm_shuffle_epi8(src128, convert_mask);

            if (fill_alpha) {
                // Set the alpha channels of src to 255
                src128 = _mm_or_si128(src128, alpha_fill_mask);
            }

            // Save the result
            _mm_storeu_si128((__m128i *)dst, src128);

            src += 16;
            dst += 16;
        }

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32;
            if (fill_alpha) {
                SWIZZLE_8888_DST_ALPHA(src32, dst32, srcfmt, dstfmt, dstAmask);
            } else {
                SWIZZLE_8888_SRC_ALPHA(src32, dst32, srcfmt, dstfmt);
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

static void SDL_TARGETING("avx2") Blit8888to8888PixelSwizzleAVX2(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = (!srcfmt->Amask || !dstfmt->Amask);
    Uint32 srcAmask, srcAshift;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(srcfmt, &srcAmask, &srcAshift);
    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const __m256i mask_offsets = _mm256_set_epi8(
        28, 28, 28, 28, 24, 24, 24, 24, 20, 20, 20, 20, 16, 16, 16, 16, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    const __m256i convert_mask = _mm256_add_epi32(
        _mm256_set1_epi32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift) |
            ((srcAshift >> 3) << dstAshift)),
        mask_offsets);

    const __m256i alpha_fill_mask = _mm256_set1_epi32((int)dstAmask);

    while (height--) {
        int i = 0;

        for (; i + 8 <= width; i += 8) {
            // Load 8 src pixels
            __m256i src256 = _mm256_loadu_si256((__m256i *)src);

            // Convert to dst format
            src256 = _mm256_shuffle_epi8(src256, convert_mask);

            if (fill_alpha) {
                // Set the alpha channels of src to 255
                src256 = _mm256_or_si256(src256, alpha_fill_mask);
            }

            // Save the result
            _mm256_storeu_si256((__m256i *)dst, src256);

            src += 32;
            dst += 32;
        }

        for (; i < width; ++i) {
            Uint32 src32 = *(Uint32 *)src;
            Uint32 dst32;
            if (fill_alpha) {
                SWIZZLE_8888_DST_ALPHA(src32, dst32, srcfmt, dstfmt, dstAmask);
            } else {
                SWIZZLE_8888_SRC_ALPHA(src32, dst32, srcfmt, dstfmt);
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

static void Blit8888to8888PixelSwizzleNEON(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    bool fill_alpha = (!srcfmt->Amask || !dstfmt->Amask);
    Uint32 srcAmask, srcAshift;
    Uint32 dstAmask, dstAshift;

    SDL_Get8888AlphaMaskAndShift(srcfmt, &srcAmask, &srcAshift);
    SDL_Get8888AlphaMaskAndShift(dstfmt, &dstAmask, &dstAshift);

    // The byte offsets for the start of each pixel
    const uint8x16_t mask_offsets = vreinterpretq_u8_u64(vcombine_u64(
        vcreate_u64(0x0404040400000000), vcreate_u64(0x0c0c0c0c08080808)));

    const uint8x16_t convert_mask = vreinterpretq_u8_u32(vaddq_u32(
        vreinterpretq_u32_u8(mask_offsets),
        vdupq_n_u32(
            ((srcfmt->Rshift >> 3) << dstfmt->Rshift) |
            ((srcfmt->Gshift >> 3) << dstfmt->Gshift) |
            ((srcfmt->Bshift >> 3) << dstfmt->Bshift) |
            ((srcAshift >> 3) << dstAshift))));

    const uint8x16_t alpha_fill_mask = vreinterpretq_u8_u32(vdupq_n_u32(dstAmask));

    while (height--) {
        int i = 0;

        for (; i + 4 <= width; i += 4) {
            // Load 4 src pixels
            uint8x16_t src128 = vld1q_u8(src);

            // Convert to dst format
            src128 = vqtbl1q_u8(src128, convert_mask);

            if (fill_alpha) {
                // Set the alpha channels of src to 255
                src128 = vorrq_u8(src128, alpha_fill_mask);
            }

            // Save the result
            vst1q_u8(dst, src128);

            src += 16;
            dst += 16;
        }

        // Process 1 pixel per iteration, max 3 iterations, same calculations as above
        for (; i < width; ++i) {
            // Top 32-bits will be not used in src32
            uint8x8_t src32 = vreinterpret_u8_u32(vld1_dup_u32((Uint32*)src));

            // Convert to dst format
            src32 = vtbl1_u8(src32, vget_low_u8(convert_mask));

            if (fill_alpha) {
                // Set the alpha channels of src to 255
                src32 = vorr_u8(src32, vget_low_u8(alpha_fill_mask));
            }

            // Save the result, only low 32-bits
            vst1_lane_u32((Uint32*)dst, vreinterpret_u32_u8(src32), 0);

            src += 4;
            dst += 4;
        }

        src += srcskip;
        dst += dstskip;
    }
}

#endif

// Blit_3or4_to_3or4__same_rgb: 3 or 4 bpp, same RGB triplet
static void Blit_3or4_to_3or4__same_rgb(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int dstbpp = dstfmt->bytes_per_pixel;

    if (dstfmt->Amask) {
        // SET_ALPHA
        Uint32 mask = ((Uint32)info->a) << dstfmt->Ashift;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        int i0 = 0, i1 = 1, i2 = 2;
#else
        int i0 = srcbpp - 1 - 0;
        int i1 = srcbpp - 1 - 1;
        int i2 = srcbpp - 1 - 2;
#endif
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint32 *dst32 = (Uint32*)dst;
                Uint8 s0 = src[i0];
                Uint8 s1 = src[i1];
                Uint8 s2 = src[i2];
                *dst32 = (s0) | (s1 << 8) | (s2 << 16) | mask;
                dst += 4;
                src += srcbpp;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    } else {
        // NO_ALPHA
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        int i0 = 0, i1 = 1, i2 = 2;
        int j0 = 0, j1 = 1, j2 = 2;
#else
        int i0 = srcbpp - 1 - 0;
        int i1 = srcbpp - 1 - 1;
        int i2 = srcbpp - 1 - 2;
        int j0 = dstbpp - 1 - 0;
        int j1 = dstbpp - 1 - 1;
        int j2 = dstbpp - 1 - 2;
#endif
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint8 s0 = src[i0];
                Uint8 s1 = src[i1];
                Uint8 s2 = src[i2];
                dst[j0] = s0;
                dst[j1] = s1;
                dst[j2] = s2;
                dst += dstbpp;
                src += srcbpp;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    }
}

// Blit_3or4_to_3or4__inversed_rgb: 3 or 4 bpp, inversed RGB triplet
static void Blit_3or4_to_3or4__inversed_rgb(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    const SDL_PixelFormatDetails *srcfmt = info->src_fmt;
    int srcbpp = srcfmt->bytes_per_pixel;
    const SDL_PixelFormatDetails *dstfmt = info->dst_fmt;
    int dstbpp = dstfmt->bytes_per_pixel;

    if (dstfmt->Amask) {
        if (srcfmt->Amask) {
            // COPY_ALPHA
            // Only to switch ABGR8888 <-> ARGB8888
            while (height--) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
#else
                int i0 = 3, i1 = 2, i2 = 1, i3 = 0;
#endif
                /* *INDENT-OFF* */ // clang-format off
                DUFFS_LOOP(
                {
                    Uint32 *dst32 = (Uint32*)dst;
                    Uint8 s0 = src[i0];
                    Uint8 s1 = src[i1];
                    Uint8 s2 = src[i2];
                    Uint32 alphashift = ((Uint32)src[i3]) << dstfmt->Ashift;
                    // inversed, compared to Blit_3or4_to_3or4__same_rgb
                    *dst32 = (s0 << 16) | (s1 << 8) | (s2) | alphashift;
                    dst += 4;
                    src += 4;
                }, width);
                /* *INDENT-ON* */ // clang-format on
                src += srcskip;
                dst += dstskip;
            }
        } else {
            // SET_ALPHA
            Uint32 mask = ((Uint32)info->a) << dstfmt->Ashift;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            int i0 = 0, i1 = 1, i2 = 2;
#else
            int i0 = srcbpp - 1 - 0;
            int i1 = srcbpp - 1 - 1;
            int i2 = srcbpp - 1 - 2;
#endif
            while (height--) {
                /* *INDENT-OFF* */ // clang-format off
                DUFFS_LOOP(
                {
                    Uint32 *dst32 = (Uint32*)dst;
                    Uint8 s0 = src[i0];
                    Uint8 s1 = src[i1];
                    Uint8 s2 = src[i2];
                    // inversed, compared to Blit_3or4_to_3or4__same_rgb
                    *dst32 = (s0 << 16) | (s1 << 8) | (s2) | mask;
                    dst += 4;
                    src += srcbpp;
                }, width);
                /* *INDENT-ON* */ // clang-format on
                src += srcskip;
                dst += dstskip;
            }
        }
    } else {
        // NO_ALPHA
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        int i0 = 0, i1 = 1, i2 = 2;
        int j0 = 2, j1 = 1, j2 = 0;
#else
        int i0 = srcbpp - 1 - 0;
        int i1 = srcbpp - 1 - 1;
        int i2 = srcbpp - 1 - 2;
        int j0 = dstbpp - 1 - 2;
        int j1 = dstbpp - 1 - 1;
        int j2 = dstbpp - 1 - 0;
#endif
        while (height--) {
            /* *INDENT-OFF* */ // clang-format off
            DUFFS_LOOP(
            {
                Uint8 s0 = src[i0];
                Uint8 s1 = src[i1];
                Uint8 s2 = src[i2];
                // inversed, compared to Blit_3or4_to_3or4__same_rgb
                dst[j0] = s0;
                dst[j1] = s1;
                dst[j2] = s2;
                dst += dstbpp;
                src += srcbpp;
            }, width);
            /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
        }
    }
}

// Normal N to N optimized blitters
#define NO_ALPHA   1
#define SET_ALPHA  2
#define COPY_ALPHA 4
struct blit_table
{
    Uint32 srcR, srcG, srcB;
    int dstbpp;
    Uint32 dstR, dstG, dstB;
    Uint32 blit_features;
    SDL_BlitFunc blitfunc;
    Uint32 alpha; // bitwise NO_ALPHA, SET_ALPHA, COPY_ALPHA
};
static const struct blit_table normal_blit_1[] = {
    // Default for 8-bit RGB source, never optimized
    { 0, 0, 0, 0, 0, 0, 0, 0, BlitNtoN, 0 }
};

static const struct blit_table normal_blit_2[] = {
#ifdef SDL_ALTIVEC_BLITTERS
    // has-altivec
    { 0x0000F800, 0x000007E0, 0x0000001F, 4, 0x00000000, 0x00000000, 0x00000000,
      BLIT_FEATURE_HAS_ALTIVEC, Blit_RGB565_32Altivec, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    { 0x00007C00, 0x000003E0, 0x0000001F, 4, 0x00000000, 0x00000000, 0x00000000,
      BLIT_FEATURE_HAS_ALTIVEC, Blit_RGB555_32Altivec, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
#endif
#ifdef SDL_HAVE_BLIT_N_RGB565
    { 0x0000F800, 0x000007E0, 0x0000001F, 4, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_RGB565_ARGB8888, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    { 0x0000F800, 0x000007E0, 0x0000001F, 4, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_RGB565_ABGR8888, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    { 0x0000F800, 0x000007E0, 0x0000001F, 4, 0xFF000000, 0x00FF0000, 0x0000FF00,
      0, Blit_RGB565_RGBA8888, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    { 0x0000F800, 0x000007E0, 0x0000001F, 4, 0x0000FF00, 0x00FF0000, 0xFF000000,
      0, Blit_RGB565_BGRA8888, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
#endif
    // Default for 16-bit RGB source, used if no other blitter matches
    { 0, 0, 0, 0, 0, 0, 0, 0, BlitNtoN, 0 }
};

static const struct blit_table normal_blit_3[] = {
    // 3->4 with same rgb triplet
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 4, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__same_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 4, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__same_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA },
    // 3->4 with inversed rgb triplet
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 4, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__inversed_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 4, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__inversed_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA },
    // 3->3 to switch RGB 24 <-> BGR 24
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 3, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__inversed_rgb, NO_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 3, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__inversed_rgb, NO_ALPHA },
    // Default for 24-bit RGB source, never optimized
    { 0, 0, 0, 0, 0, 0, 0, 0, BlitNtoN, 0 }
};

static const struct blit_table normal_blit_4[] = {
#ifdef SDL_ALTIVEC_BLITTERS
    // has-altivec | dont-use-prefetch
    { 0x00000000, 0x00000000, 0x00000000, 4, 0x00000000, 0x00000000, 0x00000000,
      BLIT_FEATURE_HAS_ALTIVEC | BLIT_FEATURE_ALTIVEC_DONT_USE_PREFETCH, ConvertAltivec32to32_noprefetch, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    // has-altivec
    { 0x00000000, 0x00000000, 0x00000000, 4, 0x00000000, 0x00000000, 0x00000000,
      BLIT_FEATURE_HAS_ALTIVEC, ConvertAltivec32to32_prefetch, NO_ALPHA | COPY_ALPHA | SET_ALPHA },
    // has-altivec
    { 0x00000000, 0x00000000, 0x00000000, 2, 0x0000F800, 0x000007E0, 0x0000001F,
      BLIT_FEATURE_HAS_ALTIVEC, Blit_XRGB8888_RGB565Altivec, NO_ALPHA },
#endif
    // 4->3 with same rgb triplet
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 3, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__same_rgb, NO_ALPHA | SET_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 3, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__same_rgb, NO_ALPHA | SET_ALPHA },
    // 4->3 with inversed rgb triplet
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 3, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__inversed_rgb, NO_ALPHA | SET_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 3, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__inversed_rgb, NO_ALPHA | SET_ALPHA },
    // 4->4 with inversed rgb triplet, and COPY_ALPHA to switch ABGR8888 <-> ARGB8888
    { 0x000000FF, 0x0000FF00, 0x00FF0000, 4, 0x00FF0000, 0x0000FF00, 0x000000FF,
      0, Blit_3or4_to_3or4__inversed_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA | COPY_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 4, 0x000000FF, 0x0000FF00, 0x00FF0000,
      0, Blit_3or4_to_3or4__inversed_rgb,
#if HAVE_FAST_WRITE_INT8
      NO_ALPHA |
#endif
          SET_ALPHA | COPY_ALPHA },
    // RGB 888 and RGB 565
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 2, 0x0000F800, 0x000007E0, 0x0000001F,
      0, Blit_XRGB8888_RGB565, NO_ALPHA },
    { 0x00FF0000, 0x0000FF00, 0x000000FF, 2, 0x00007C00, 0x000003E0, 0x0000001F,
      0, Blit_XRGB8888_RGB555, NO_ALPHA },
    // Default for 32-bit RGB source, used if no other blitter matches
    { 0, 0, 0, 0, 0, 0, 0, 0, BlitNtoN, 0 }
};

static const struct blit_table *const normal_blit[] = {
    normal_blit_1, normal_blit_2, normal_blit_3, normal_blit_4
};

// Mask matches table, or table entry is zero
#define MASKOK(x, y) (((x) == (y)) || ((y) == 0x00000000))

SDL_BlitFunc SDL_CalculateBlitN(SDL_Surface *surface)
{
    const SDL_PixelFormatDetails *srcfmt;
    const SDL_PixelFormatDetails *dstfmt;
    const struct blit_table *table;
    int which;
    SDL_BlitFunc blitfun;

    // Set up data for choosing the blit
    srcfmt = surface->fmt;
    dstfmt = surface->map.info.dst_fmt;

    // We don't support destinations less than 8-bits
    if (dstfmt->bits_per_pixel < 8) {
        return NULL;
    }

    switch (surface->map.info.flags & ~SDL_COPY_RLE_MASK) {
    case 0:
        if (SDL_PIXELLAYOUT(srcfmt->format) == SDL_PACKEDLAYOUT_8888 &&
            SDL_PIXELLAYOUT(dstfmt->format) == SDL_PACKEDLAYOUT_8888) {
#ifdef SDL_AVX2_INTRINSICS
            if (SDL_HasAVX2()) {
                return Blit8888to8888PixelSwizzleAVX2;
            }
#endif
#ifdef SDL_SSE4_1_INTRINSICS
            if (SDL_HasSSE41()) {
                return Blit8888to8888PixelSwizzleSSE41;
            }
#endif
#if defined(SDL_NEON_INTRINSICS) && (__ARM_ARCH >= 8)
            return Blit8888to8888PixelSwizzleNEON;
#endif
        }

        blitfun = NULL;
        if (dstfmt->bits_per_pixel > 8) {
            Uint32 a_need = NO_ALPHA;
            if (dstfmt->Amask) {
                a_need = srcfmt->Amask ? COPY_ALPHA : SET_ALPHA;
            }
            if (srcfmt->bytes_per_pixel > 0 &&
                srcfmt->bytes_per_pixel <= SDL_arraysize(normal_blit)) {
                table = normal_blit[srcfmt->bytes_per_pixel - 1];
                for (which = 0; table[which].dstbpp; ++which) {
                    if (MASKOK(srcfmt->Rmask, table[which].srcR) &&
                        MASKOK(srcfmt->Gmask, table[which].srcG) &&
                        MASKOK(srcfmt->Bmask, table[which].srcB) &&
                        MASKOK(dstfmt->Rmask, table[which].dstR) &&
                        MASKOK(dstfmt->Gmask, table[which].dstG) &&
                        MASKOK(dstfmt->Bmask, table[which].dstB) &&
                        dstfmt->bytes_per_pixel == table[which].dstbpp &&
                        (a_need & table[which].alpha) == a_need &&
                        ((table[which].blit_features & GetBlitFeatures()) ==
                         table[which].blit_features)) {
                        break;
                    }
                }
                blitfun = table[which].blitfunc;
            }

            if (blitfun == BlitNtoN) { // default C fallback catch-all. Slow!
                if (srcfmt->bytes_per_pixel == dstfmt->bytes_per_pixel &&
                    srcfmt->Rmask == dstfmt->Rmask &&
                    srcfmt->Gmask == dstfmt->Gmask &&
                    srcfmt->Bmask == dstfmt->Bmask) {
                    if (a_need == COPY_ALPHA) {
                        if (srcfmt->Amask == dstfmt->Amask) {
                            // Fastpath C fallback: RGBA<->RGBA blit with matching RGBA
                            blitfun = SDL_BlitCopy;
                        } else {
                            blitfun = BlitNtoNCopyAlpha;
                        }
                    } else {
                        if (srcfmt->bytes_per_pixel == 4) {
                            // Fastpath C fallback: 32bit RGB<->RGBA blit with matching RGB
                            blitfun = Blit4to4MaskAlpha;
                        } else if (srcfmt->bytes_per_pixel == 2) {
                            // Fastpath C fallback: 16bit RGB<->RGBA blit with matching RGB
                            blitfun = Blit2to2MaskAlpha;
                        }
                    }
                } else if (a_need == COPY_ALPHA) {
                    blitfun = BlitNtoNCopyAlpha;
                }
            }
        }
        return blitfun;

    case SDL_COPY_COLORKEY:
        /* colorkey blit: Here we don't have too many options, mostly
           because RLE is the preferred fast way to deal with this.
           If a particular case turns out to be useful we'll add it. */

        if (srcfmt->bytes_per_pixel == 2 && surface->map.identity != 0) {
            return Blit2to2Key;
        } else {
#ifdef SDL_ALTIVEC_BLITTERS
            if ((srcfmt->bytes_per_pixel == 4) && (dstfmt->bytes_per_pixel == 4) && SDL_HasAltiVec()) {
                return Blit32to32KeyAltivec;
            } else
#endif
            if (srcfmt->Amask && dstfmt->Amask) {
                return BlitNtoNKeyCopyAlpha;
            } else {
                return BlitNtoNKey;
            }
        }
    }

    return NULL;
}

#endif // SDL_HAVE_BLIT_N
