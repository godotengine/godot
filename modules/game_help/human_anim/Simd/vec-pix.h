#pragma once

#include "vec-pix-types.h"
#include "vec-math.h"
#include "Runtime/Math/Color.h"

namespace math
{
    // Truncates integer values within int4 vector to a single pixel which is
    // broadcasted across lanes
    static MATH_FORCEINLINE pix4 convert_pix1(const int4& a)
    {
#   if defined(MATH_HAS_SIMD_INT) && defined(__ARM_NEON)
        uint16x4_t a16 = vmovn_u32(vreinterpretq_u32_s32((int32x4_t)a));
        uint8x8_t a8 = vmovn_u16(vcombine_u16(a16, a16));
#       if defined(MATH_HAS_NATIVE_SIMD)
        return pix4(vreinterpretq_s32_u8(vcombine_u8(a8, a8)));
#       else
        return pix4(int4(vreinterpretq_s32_u8(vcombine_u8(a8, a8))));
#       endif
#   elif defined(MATH_HAS_SIMD_INT) && defined(__SSE2__)
        __m128i a0 = _mm_slli_epi32((__m128i)a, 24);
        a0 = _mm_srai_epi32(a0, 24);
        a0 = _mm_packs_epi32(a0, a0);
        a0 = _mm_packs_epi16(a0, a0);
        return pix4(a0);
#   else
        int4 res;
        res.x = (a.x & 0x000000ff);
        res.x |= ((a.y <<  8) & 0x0000ff00);
        res.x |= ((a.z << 16) & 0x00ff0000);
        res.x |= ((a.w << 24) & 0xff000000);
        res.y = res.z = res.w = res.x;
        return pix4(res);
#   endif
    }

    static MATH_FORCEINLINE pix4 convert_pix1(const ColorRGBAf& color)
    {
#   if MATH_HAS_SIMD_INT
        float4 value = vload4f(color.GetPtr());
        return convert_pix1(normalizeToByte(value));
#   else
        ColorRGBA32 result = color;
        return int4(math::vload1i((int*)&result));
#   endif
    }

    static MATH_FORCEINLINE pix4 pix_permute4(const pix4& pa, const pix4& mask)
    {
#   if MATH_HAS_SIMD_INT && defined(__SSSE3__)
        return pix4(_mm_shuffle_epi8((__m128i)pa.i, (__m128i)mask.i));
#   elif defined(MATH_HAS_SIMD_INT) && defined(__ARM_NEON) && (defined(__aarch64__) || defined(_M_ARM64))
#       if defined(MATH_HAS_NATIVE_SIMD)
        return vqtbl1q_u8(pa.i, mask.i);
#       else
        return int4(vqtbl1q_u8(pa.i.p, mask.i.p));
#       endif
#   elif defined(MATH_HAS_SIMD_INT) && defined(__ARM_NEON)
        uint8x8x2_t valueTable = { { vget_low_u8(pa.i), vget_high_u8(pa.i) } };
        uint8x8_t lo = vtbl2_u8(valueTable, vget_low_u8(mask.i));
        uint8x8_t hi = vtbl2_u8(valueTable, vget_high_u8(mask.i));
        return pix4(vcombine_u8(lo, hi));
#   else
        int4 res;
        UInt8 data[16] =
        {
            (UInt8)pa.i.x, (UInt8)(pa.i.x >> 8), (UInt8)(pa.i.x >> 16), (UInt8)(pa.i.x >> 24),
            (UInt8)pa.i.y, (UInt8)(pa.i.y >> 8), (UInt8)(pa.i.y >> 16), (UInt8)(pa.i.y >> 24),
            (UInt8)pa.i.z, (UInt8)(pa.i.z >> 8), (UInt8)(pa.i.z >> 16), (UInt8)(pa.i.z >> 24),
            (UInt8)pa.i.w, (UInt8)(pa.i.w >> 8), (UInt8)(pa.i.w >> 16), (UInt8)(pa.i.w >> 24)
        };

        res.x = data[(mask.i.x >>  0) & 0x0f] |
            data[(mask.i.x >>  8) & 0x0f] << 8 |
            data[(mask.i.x >> 16) & 0x0f] << 16 |
            data[(mask.i.x >> 24) & 0x0f] << 24;
        res.y = data[(mask.i.y >>  0) & 0x0f] |
            data[(mask.i.y >>  8) & 0x0f] << 8 |
            data[(mask.i.y >> 16) & 0x0f] << 16 |
            data[(mask.i.y >> 24) & 0x0f] << 24;
        res.z = data[(mask.i.z >>  0) & 0x0f] |
            data[(mask.i.z >>  8) & 0x0f] << 8 |
            data[(mask.i.z >> 16) & 0x0f] << 16 |
            data[(mask.i.z >> 24) & 0x0f] << 24;
        res.w = data[(mask.i.w >>  0) & 0x0f] |
            data[(mask.i.w >>  8) & 0x0f] << 8 |
            data[(mask.i.w >> 16) & 0x0f] << 16 |
            data[(mask.i.w >> 24) & 0x0f] << 24;

        return pix4(res);
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)
    static MATH_FORCEINLINE pix4 operator+(const pix4& pa, const pix4& pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;

#   if defined(__ARM_NEON)
#       if defined(MATH_HAS_NATIVE_SIMD)
        return pix4(vreinterpretq_s32_u8(vqaddq_u8(vreinterpretq_u8_s32((int32x4_t)a), vreinterpretq_u8_s32((int32x4_t)b))));
#       else
        return pix4(int4(vreinterpretq_s32_u8(vqaddq_u8(vreinterpretq_u8_s32((int32x4_t)a), vreinterpretq_u8_s32((int32x4_t)b)))));
#       endif
#   elif defined(__SSE2__)
        return pix4(_mm_adds_epu8((__m128i)a, (__m128i)b));
#   else
        int4 res;
        unsigned d;
        {
            d = comp_adds((unsigned(a.x)) & 0xff, ((unsigned(b.x)) & 0xff));
            d |= comp_adds((unsigned(a.x) >> 8) & 0xff, ((unsigned(b.x) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.x) >> 16) & 0xff, ((unsigned(b.x) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.x) >> 24) & 0xff, ((unsigned(b.x) >> 24) & 0xff)) << 24;
            res.x = int(d);
        }
        {
            d = comp_adds((unsigned(a.y)) & 0xff, ((unsigned(b.y)) & 0xff));
            d |= comp_adds((unsigned(a.y) >> 8) & 0xff, ((unsigned(b.y) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.y) >> 16) & 0xff, ((unsigned(b.y) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.y) >> 24) & 0xff, ((unsigned(b.y) >> 24) & 0xff)) << 24;
            res.y = int(d);
        }
        {
            d = comp_adds((unsigned(a.z)) & 0xff, ((unsigned(b.z)) & 0xff));
            d |= comp_adds((unsigned(a.z) >> 8) & 0xff, ((unsigned(b.z) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.z) >> 16) & 0xff, ((unsigned(b.z) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.z) >> 24) & 0xff, ((unsigned(b.z) >> 24) & 0xff)) << 24;
            res.z = int(d);
        }
        {
            d = comp_adds((unsigned(a.w)) & 0xff, ((unsigned(b.w)) & 0xff));
            d |= comp_adds((unsigned(a.w) >> 8) & 0xff, ((unsigned(b.w) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.w) >> 16) & 0xff, ((unsigned(b.w) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.w) >> 24) & 0xff, ((unsigned(b.w) >> 24) & 0xff)) << 24;
            res.w = int(d);
        }
        return pix4(res);
#   endif
    }

#else
    static MATH_FORCEINLINE unsigned comp_adds(unsigned a, unsigned b)
    {
        return min(a + b, 255);
    }

    static MATH_FORCEINLINE pix4 operator+(const pix4& pa, const pix4& pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;
        int4 res;
        unsigned d;
        {
            d = comp_adds((unsigned(a.x)) & 0xff, ((unsigned(b.x)) & 0xff));
            d |= comp_adds((unsigned(a.x) >> 8) & 0xff, ((unsigned(b.x) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.x) >> 16) & 0xff, ((unsigned(b.x) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.x) >> 24) & 0xff, ((unsigned(b.x) >> 24) & 0xff)) << 24;
            res.x = int(d);
        }
        {
            d = comp_adds((unsigned(a.y)) & 0xff, ((unsigned(b.y)) & 0xff));
            d |= comp_adds((unsigned(a.y) >> 8) & 0xff, ((unsigned(b.y) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.y) >> 16) & 0xff, ((unsigned(b.y) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.y) >> 24) & 0xff, ((unsigned(b.y) >> 24) & 0xff)) << 24;
            res.y = int(d);
        }
        {
            d = comp_adds((unsigned(a.z)) & 0xff, ((unsigned(b.z)) & 0xff));
            d |= comp_adds((unsigned(a.z) >> 8) & 0xff, ((unsigned(b.z) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.z) >> 16) & 0xff, ((unsigned(b.z) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.z) >> 24) & 0xff, ((unsigned(b.z) >> 24) & 0xff)) << 24;
            res.z = int(d);
        }
        {
            d = comp_adds((unsigned(a.w)) & 0xff, ((unsigned(b.w)) & 0xff));
            d |= comp_adds((unsigned(a.w) >> 8) & 0xff, ((unsigned(b.w) >> 8) & 0xff)) << 8;
            d |= comp_adds((unsigned(a.w) >> 16) & 0xff, ((unsigned(b.w) >> 16) & 0xff)) << 16;
            d |= comp_adds((unsigned(a.w) >> 24) & 0xff, ((unsigned(b.w) >> 24) & 0xff)) << 24;
            res.w = int(d);
        }
        return pix4(res);
    }

#endif

    static MATH_FORCEINLINE int1 pix_weight(const float1& w)
    {
        float1 f = float1(255.f) * w;
        int i = (int)convert_int(f);
#   if defined(MATH_HAS_SIMD_INT) && (defined(__ARM_NEON) || defined(__SSE2__))
        i |= (i << 16);
        return int1(i | (i << 8));
#   else
        return int1(i);
#   endif
    }

    static MATH_FORCEINLINE int4 pix_weight(const float4& w)
    {
        float4 f = float4(255.f) * w;
        int4 i = convert_int4(f);
#   if defined(MATH_HAS_SIMD_INT) && (defined(__ARM_NEON) || defined(__SSE2__))
        i |= (i << 16);
        return (i | (i << 8));
#   else
        return i;
#   endif
    }

    /*
     *  Based on Jim Blinn’s Three Wrongs Make a Right technique.
     */
    static MATH_FORCEINLINE unsigned comp_mul(unsigned a, unsigned b)
    {
        unsigned i = a * b + 128U;
        return (i + (i >> 8)) >> 8;
    }

#   if defined(MATH_HAS_SIMD_INT)
    static MATH_FORCEINLINE pix4 operator*(const pix4& pa, const pix4& pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;

#   if defined(__ARM_NEON)
        static const uint16x8_t c1 = vdupq_n_u16(128);
        uint16x8_t l = vaddq_u16(vmull_u8(vget_low_u8(vreinterpretq_u8_s32((int32x4_t)a)), vget_low_u8(vreinterpretq_u8_s32((int32x4_t)b))), c1);
        uint16x8_t h = vaddq_u16(vmull_u8(vget_high_u8(vreinterpretq_u8_s32((int32x4_t)a)), vget_high_u8(vreinterpretq_u8_s32((int32x4_t)b))), c1);
        l = vaddq_u16(l, vshrq_n_u16(l, 8));
        h = vaddq_u16(h, vshrq_n_u16(h, 8));
        l = vshrq_n_u16(l, 8);
        h = vshrq_n_u16(h, 8);
#       if defined(MATH_HAS_NATIVE_SIMD)
        return pix4(vcombine_s32(vreinterpret_s32_u8(vmovn_u16(l)), vreinterpret_s32_u8(vmovn_u16(h))));
#       else
        return pix4(int4(vcombine_s32(vreinterpret_s32_u8(vmovn_u16(l)), vreinterpret_s32_u8(vmovn_u16(h)))));
#       endif
#   elif defined(__SSE2__)
        static const __m128i c1 = _mm_set1_epi16(128);
        const __m128i c0 = _mm_setzero_si128();
        __m128i l = _mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8((__m128i)a, c0), _mm_unpacklo_epi8((__m128i)b, c0)), c1);
        __m128i h = _mm_add_epi16(_mm_mullo_epi16(_mm_unpackhi_epi8((__m128i)a, c0), _mm_unpackhi_epi8((__m128i)b, c0)), c1);
        l = _mm_add_epi16(l, _mm_srli_epi16(l, 8));
        h = _mm_add_epi16(h, _mm_srli_epi16(h, 8));
        l = _mm_srli_epi16(l, 8);
        h = _mm_srli_epi16(h, 8);
        return pix4(_mm_packus_epi16(l, h));
#   else
        int4 res;
        unsigned d;
        {
            d = comp_mul((unsigned(a.x)) & 0xff, (unsigned(b.x)) & 0xff);
            d |= comp_mul((unsigned(a.x) >> 8) & 0xff, (unsigned(b.x) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.x) >> 16) & 0xff, (unsigned(b.x) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.x) >> 24) & 0xff, (unsigned(b.x) >> 24) & 0xff) << 24;
            res.x = int(d);
        }
        {
            d = comp_mul((unsigned(a.y)) & 0xff, (unsigned(b.y)) & 0xff);
            d |= comp_mul((unsigned(a.y) >> 8) & 0xff, (unsigned(b.y) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.y) >> 16) & 0xff, (unsigned(b.y) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.y) >> 24) & 0xff, (unsigned(b.y) >> 24) & 0xff) << 24;
            res.y = int(d);
        }
        {
            d = comp_mul((unsigned(a.z)) & 0xff, (unsigned(b.z)) & 0xff);
            d |= comp_mul((unsigned(a.z) >> 8) & 0xff, (unsigned(b.z) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.z) >> 16) & 0xff, (unsigned(b.z) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.z) >> 24) & 0xff, (unsigned(b.z) >> 24) & 0xff) << 24;
            res.z = int(d);
        }
        {
            d = comp_mul((unsigned(a.w)) & 0xff, (unsigned(b.w)) & 0xff);
            d |= comp_mul((unsigned(a.w) >> 8) & 0xff, (unsigned(b.w) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.w) >> 16) & 0xff, (unsigned(b.w) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.w) >> 24) & 0xff, (unsigned(b.w) >> 24) & 0xff) << 24;
            res.w = int(d);
        }
        return pix4(res);
#   endif
    }

#   else
    static MATH_FORCEINLINE pix4 operator*(const pix4& pa, const pix4& pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;
        int4 res;
        unsigned d;
        {
            d = comp_mul((unsigned(a.x)) & 0xff, (unsigned(b.x)) & 0xff);
            d |= comp_mul((unsigned(a.x) >> 8) & 0xff, (unsigned(b.x) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.x) >> 16) & 0xff, (unsigned(b.x) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.x) >> 24) & 0xff, (unsigned(b.x) >> 24) & 0xff) << 24;
            res.x = int(d);
        }
        {
            d = comp_mul((unsigned(a.y)) & 0xff, (unsigned(b.y)) & 0xff);
            d |= comp_mul((unsigned(a.y) >> 8) & 0xff, (unsigned(b.y) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.y) >> 16) & 0xff, (unsigned(b.y) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.y) >> 24) & 0xff, (unsigned(b.y) >> 24) & 0xff) << 24;
            res.y = int(d);
        }
        {
            d = comp_mul((unsigned(a.z)) & 0xff, (unsigned(b.z)) & 0xff);
            d |= comp_mul((unsigned(a.z) >> 8) & 0xff, (unsigned(b.z) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.z) >> 16) & 0xff, (unsigned(b.z) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.z) >> 24) & 0xff, (unsigned(b.z) >> 24) & 0xff) << 24;
            res.z = int(d);
        }
        {
            d = comp_mul((unsigned(a.w)) & 0xff, (unsigned(b.w)) & 0xff);
            d |= comp_mul((unsigned(a.w) >> 8) & 0xff, (unsigned(b.w) >> 8) & 0xff) << 8;
            d |= comp_mul((unsigned(a.w) >> 16) & 0xff, (unsigned(b.w) >> 16) & 0xff) << 16;
            d |= comp_mul((unsigned(a.w) >> 24) & 0xff, (unsigned(b.w) >> 24) & 0xff) << 24;
            res.w = int(d);
        }
        return pix4(res);
    }

#   endif

    static MATH_FORCEINLINE int comp_lerp(unsigned a, unsigned b, int c)
    {
        int i = int(b - a) * c + 128;
        return unsigned(a + (i >> 8));
    }

    static MATH_FORCEINLINE unsigned comp_lerp(unsigned a, unsigned b, unsigned c)
    {
        int i = int(b - a) * int(c) + 128;
        return unsigned(a + (i >> 8));
    }

#   if defined(MATH_HAS_SIMD_INT)
    static MATH_FORCEINLINE pix4 lerp(const pix4& pa, const pix4& pb, const int4& w)
    {
        int4 a = pa.i;
        int4 b = pb.i;

#   if defined(__ARM_NEON)
        const int16x8_t c1 = vdupq_n_s16(128);
        const int8x8_t zero = vdup_n_s8(0);
        int8x16_t aa = vreinterpretq_s8_s32((int32x4_t)a);
        int8x16_t bb = vreinterpretq_s8_s32((int32x4_t)b);
        int8x16_t ww = vreinterpretq_s8_s32((int32x4_t)w);

        int8x8x2_t aal8x8x2 = vzip_s8(vget_low_s8(aa), zero);
        int8x8x2_t bbl8x8x2 = vzip_s8(vget_low_s8(bb), zero);
        int16x8_t aal16 = vcombine_s16(vreinterpret_s16_s8(aal8x8x2.val[0]), vreinterpret_s16_s8(aal8x8x2.val[1]));
        int16x8_t bbl16 = vcombine_s16(vreinterpret_s16_s8(bbl8x8x2.val[0]), vreinterpret_s16_s8(bbl8x8x2.val[1]));
        int16x8_t bal = vsubq_s16(bbl16, aal16);

        int8x8x2_t aah8x8x2 = vzip_s8(vget_high_s8(aa), zero);
        int8x8x2_t bbh8x8x2 = vzip_s8(vget_high_s8(bb), zero);
        int16x8_t aah16 = vcombine_s16(vreinterpret_s16_s8(aah8x8x2.val[0]), vreinterpret_s16_s8(aah8x8x2.val[1]));
        int16x8_t bbh16 = vcombine_s16(vreinterpret_s16_s8(bbh8x8x2.val[0]), vreinterpret_s16_s8(bbh8x8x2.val[1]));
        int16x8_t bah = vsubq_s16(bbh16, aah16);

        int8x8x2_t wwl8x8x2 = vzip_s8(vget_low_s8(ww), zero);
        int8x8x2_t wwh8x8x2 = vzip_s8(vget_high_s8(ww), zero);
        int16x8_t wwl16 = vcombine_s16(vreinterpret_s16_s8(wwl8x8x2.val[0]), vreinterpret_s16_s8(wwl8x8x2.val[1]));
        int16x8_t wwh16 = vcombine_s16(vreinterpret_s16_s8(wwh8x8x2.val[0]), vreinterpret_s16_s8(wwh8x8x2.val[1]));

        int16x8_t low = vaddq_s16(vmulq_s16(bal, wwl16), c1);
        int16x8_t high = vaddq_s16(vmulq_s16(bah, wwh16), c1);

        int16x8_t baw128shiftl = vaddq_s16(aal16, vshrq_n_s16(low, 8));
        int16x8_t baw128shifth = vaddq_s16(aah16, vshrq_n_s16(high, 8));

        int8x16_t r = vcombine_s8(vmovn_s16(baw128shiftl), vmovn_s16(baw128shifth));
        return pix4(vreinterpretq_s32_s8(r));
#   elif defined(__SSE2__)
        static const __m128i c1 = _mm_set1_epi16(128);
        static const __m128i c2 = _mm_set1_epi16(0xff);
        const __m128i c0 = _mm_setzero_si128();
        __m128i al = _mm_unpacklo_epi8((__m128i)a, c0);
        __m128i ah = _mm_unpackhi_epi8((__m128i)a, c0);
        __m128i l = _mm_sub_epi16(_mm_unpacklo_epi8((__m128i)b, c0), al);
        __m128i h = _mm_sub_epi16(_mm_unpackhi_epi8((__m128i)b, c0), ah);
        l = _mm_add_epi16(_mm_mullo_epi16(l, _mm_unpacklo_epi8((__m128i)w, c0)), c1);
        h = _mm_add_epi16(_mm_mullo_epi16(h, _mm_unpackhi_epi8((__m128i)w, c0)), c1);
        l = _mm_and_si128(_mm_add_epi16(al, _mm_srai_epi16(l, 8)), c2);
        h = _mm_and_si128(_mm_add_epi16(ah, _mm_srai_epi16(h, 8)), c2);
        return pix4(_mm_packus_epi16(l, h));
#   else
        int4 res;
        int d;
        {
            d = comp_lerp((unsigned(a.x)) & 0xff, (unsigned(b.x)) & 0xff, unsigned(w.x) & 0xff);
            d |= comp_lerp((unsigned(a.x) >> 8) & 0xff, (unsigned(b.x) >> 8) & 0xff, unsigned(w.x) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.x) >> 16) & 0xff, (unsigned(b.x) >> 16) & 0xff, unsigned(w.x) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.x) >> 24) & 0xff, (unsigned(b.x) >> 24) & 0xff, unsigned(w.x) & 0xff) << 24;
            res.x = int(d);
        }
        {
            d = comp_lerp((unsigned(a.y)) & 0xff, (unsigned(b.y)) & 0xff, unsigned(w.y) & 0xff);
            d |= comp_lerp((unsigned(a.y) >> 8) & 0xff, (unsigned(b.y) >> 8) & 0xff, unsigned(w.y) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.y) >> 16) & 0xff, (unsigned(b.y) >> 16) & 0xff, unsigned(w.y) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.y) >> 24) & 0xff, (unsigned(b.y) >> 24) & 0xff, unsigned(w.y) & 0xff) << 24;
            res.y = int(d);
        }
        {
            d = comp_lerp((unsigned(a.z)) & 0xff, (unsigned(b.z)) & 0xff, unsigned(w.z) & 0xff);
            d |= comp_lerp((unsigned(a.z) >> 8) & 0xff, (unsigned(b.z) >> 8) & 0xff, unsigned(w.z) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.z) >> 16) & 0xff, (unsigned(b.z) >> 16) & 0xff, unsigned(w.z) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.z) >> 24) & 0xff, (unsigned(b.z) >> 24) & 0xff, unsigned(w.z) & 0xff) << 24;
            res.z = int(d);
        }
        {
            d = comp_lerp((unsigned(a.w)) & 0xff, (unsigned(b.w)) & 0xff, unsigned(w.w) & 0xff);
            d |= comp_lerp((unsigned(a.w) >> 8) & 0xff, (unsigned(b.w) >> 8) & 0xff, unsigned(w.w) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.w) >> 16) & 0xff, (unsigned(b.w) >> 16) & 0xff, unsigned(w.w) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.w) >> 24) & 0xff, (unsigned(b.w) >> 24) & 0xff, unsigned(w.w) & 0xff) << 24;
            res.w = int(d);
        }
        return pix4(res);
#   endif
    }

#   else
    static MATH_FORCEINLINE pix4 lerp(const pix4& pa, const pix4& pb, const int4& w)
    {
        int4 a = pa.i;
        int4 b = pb.i;
        int4 res;
        int d;
        {
            d = comp_lerp((unsigned(a.x)) & 0xff, (unsigned(b.x)) & 0xff, unsigned(w.x) & 0xff);
            d |= comp_lerp((unsigned(a.x) >> 8) & 0xff, (unsigned(b.x) >> 8) & 0xff, unsigned(w.x) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.x) >> 16) & 0xff, (unsigned(b.x) >> 16) & 0xff, unsigned(w.x) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.x) >> 24) & 0xff, (unsigned(b.x) >> 24) & 0xff, unsigned(w.x) & 0xff) << 24;
            res.x = int(d);
        }
        {
            d = comp_lerp((unsigned(a.y)) & 0xff, (unsigned(b.y)) & 0xff, unsigned(w.y) & 0xff);
            d |= comp_lerp((unsigned(a.y) >> 8) & 0xff, (unsigned(b.y) >> 8) & 0xff, unsigned(w.y) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.y) >> 16) & 0xff, (unsigned(b.y) >> 16) & 0xff, unsigned(w.y) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.y) >> 24) & 0xff, (unsigned(b.y) >> 24) & 0xff, unsigned(w.y) & 0xff) << 24;
            res.y = int(d);
        }
        {
            d = comp_lerp((unsigned(a.z)) & 0xff, (unsigned(b.z)) & 0xff, unsigned(w.z) & 0xff);
            d |= comp_lerp((unsigned(a.z) >> 8) & 0xff, (unsigned(b.z) >> 8) & 0xff, unsigned(w.z) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.z) >> 16) & 0xff, (unsigned(b.z) >> 16) & 0xff, unsigned(w.z) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.z) >> 24) & 0xff, (unsigned(b.z) >> 24) & 0xff, unsigned(w.z) & 0xff) << 24;
            res.z = int(d);
        }
        {
            d = comp_lerp((unsigned(a.w)) & 0xff, (unsigned(b.w)) & 0xff, unsigned(w.w) & 0xff);
            d |= comp_lerp((unsigned(a.w) >> 8) & 0xff, (unsigned(b.w) >> 8) & 0xff, unsigned(w.w) & 0xff) << 8;
            d |= comp_lerp((unsigned(a.w) >> 16) & 0xff, (unsigned(b.w) >> 16) & 0xff, unsigned(w.w) & 0xff) << 16;
            d |= comp_lerp((unsigned(a.w) >> 24) & 0xff, (unsigned(b.w) >> 24) & 0xff, unsigned(w.w) & 0xff) << 24;
            res.w = int(d);
        }
        return pix4(res);
    }

#   endif

#   if defined(MATH_HAS_SIMD_INT)
    static MATH_FORCEINLINE pix4 copy_alpha(const pix4 &pa, const pix4 &pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;

#   if defined(__ARM_NEON)
        const uint32x4_t c = vdupq_n_u32(0xff000000);
#       if defined(MATH_HAS_NATIVE_SIMD)
        return pix4(vbslq_s32(c, (int32x4_t)b, (int32x4_t)a));
#       else
        return pix4(int4(vbslq_s32(c, (int32x4_t)b, (int32x4_t)a)));
#       endif
#   elif defined(__SSE2__)
        const __m128i c = _mm_set1_epi32(0xff000000);
        return pix4(_mm_or_si128(_mm_and_si128((__m128i)b, c), _mm_andnot_si128(c, (__m128i)a)));
#   else
    #   if PLATFORM_ARCH_BIG_ENDIAN
        int4 r;
        r.x = int((UInt32(a.x) & ~0x000000ff) | (UInt32(b.x) & 0x000000ff));
        r.y = int((UInt32(a.y) & ~0x000000ff) | (UInt32(b.y) & 0x000000ff));
        r.z = int((UInt32(a.z) & ~0x000000ff) | (UInt32(b.z) & 0x000000ff));
        r.w = int((UInt32(a.w) & ~0x000000ff) | (UInt32(b.w) & 0x000000ff));
        return pix4(r);
    #   else
        int4 r;
        r.x = int((UInt32(a.x) & ~0xff000000) | (UInt32(b.x) & 0xff000000));
        r.y = int((UInt32(a.y) & ~0xff000000) | (UInt32(b.y) & 0xff000000));
        r.z = int((UInt32(a.z) & ~0xff000000) | (UInt32(b.z) & 0xff000000));
        r.w = int((UInt32(a.w) & ~0xff000000) | (UInt32(b.w) & 0xff000000));
        return pix4(r);
    #   endif
#   endif
    }

#   else
    static MATH_FORCEINLINE pix4 copy_alpha(const pix4 &pa, const pix4 &pb)
    {
        int4 a = pa.i;
        int4 b = pb.i;
#   if PLATFORM_ARCH_BIG_ENDIAN
        int4 r;
        r.x = int((UInt32(a.x) & ~0x000000ff) | (UInt32(b.x) & 0x000000ff));
        r.y = int((UInt32(a.y) & ~0x000000ff) | (UInt32(b.y) & 0x000000ff));
        r.z = int((UInt32(a.z) & ~0x000000ff) | (UInt32(b.z) & 0x000000ff));
        r.w = int((UInt32(a.w) & ~0x000000ff) | (UInt32(b.w) & 0x000000ff));
        return pix4(r);
#   else
        int4 r;
        r.x = int((UInt32(a.x) & ~0xff000000) | (UInt32(b.x) & 0xff000000));
        r.y = int((UInt32(a.y) & ~0xff000000) | (UInt32(b.y) & 0xff000000));
        r.z = int((UInt32(a.z) & ~0xff000000) | (UInt32(b.z) & 0xff000000));
        r.w = int((UInt32(a.w) & ~0xff000000) | (UInt32(b.w) & 0xff000000));
        return pix4(r);
#   endif
    }

#   endif
}
