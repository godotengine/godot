#ifdef __SSE4_1__

#include <array>
#include <string.h>

#include "Math.hpp"
#include "ProcessCommon.hpp"
#include "ProcessRGB_AVX2.hpp"
#include "Tables.hpp"
#include "Vector.hpp"
#ifdef _MSC_VER
#  include <intrin.h>
#  include <Windows.h>
#  define _bswap(x) _byteswap_ulong(x)
#  define VS_VECTORCALL _vectorcall
#else
#  include <x86intrin.h>
#  pragma GCC push_options
#  pragma GCC target ("avx2,fma,bmi2")
#  define VS_VECTORCALL
#endif

#ifndef _bswap
#  define _bswap(x) __builtin_bswap32(x)
#endif

namespace
{

#ifdef _MSC_VER
    inline unsigned long _bit_scan_forward( unsigned long mask )
    {
        unsigned long ret;
        _BitScanForward( &ret, mask );
        return ret;
    }
#endif

typedef std::array<uint16_t, 4> v4i;

__m256i VS_VECTORCALL Sum4_AVX2( const uint8_t* data) noexcept
{
    __m128i d0 = _mm_loadu_si128(((__m128i*)data) + 0);
    __m128i d1 = _mm_loadu_si128(((__m128i*)data) + 1);
    __m128i d2 = _mm_loadu_si128(((__m128i*)data) + 2);
    __m128i d3 = _mm_loadu_si128(((__m128i*)data) + 3);

    __m128i dm0 = _mm_and_si128(d0, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm1 = _mm_and_si128(d1, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm2 = _mm_and_si128(d2, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm3 = _mm_and_si128(d3, _mm_set1_epi32(0x00FFFFFF));

    __m256i t0 = _mm256_cvtepu8_epi16(dm0);
    __m256i t1 = _mm256_cvtepu8_epi16(dm1);
    __m256i t2 = _mm256_cvtepu8_epi16(dm2);
    __m256i t3 = _mm256_cvtepu8_epi16(dm3);

    __m256i sum0 = _mm256_add_epi16(t0, t1);
    __m256i sum1 = _mm256_add_epi16(t2, t3);

    __m256i s0 = _mm256_permute2x128_si256(sum0, sum1, (0) | (3 << 4)); // 0, 0, 3, 3
    __m256i s1 = _mm256_permute2x128_si256(sum0, sum1, (1) | (2 << 4)); // 1, 1, 2, 2

    __m256i s2 = _mm256_permute4x64_epi64(s0, _MM_SHUFFLE(1, 3, 0, 2));
    __m256i s3 = _mm256_permute4x64_epi64(s0, _MM_SHUFFLE(0, 2, 1, 3));
    __m256i s4 = _mm256_permute4x64_epi64(s1, _MM_SHUFFLE(3, 1, 0, 2));
    __m256i s5 = _mm256_permute4x64_epi64(s1, _MM_SHUFFLE(2, 0, 1, 3));

    __m256i sum5 = _mm256_add_epi16(s2, s3); //   3,   0,   3,   0
    __m256i sum6 = _mm256_add_epi16(s4, s5); //   2,   1,   1,   2
    return _mm256_add_epi16(sum5, sum6);     // 3+2, 0+1, 3+1, 3+2
}

__m256i VS_VECTORCALL Average_AVX2( const __m256i data) noexcept
{
    __m256i a = _mm256_add_epi16(data, _mm256_set1_epi16(4));

    return _mm256_srli_epi16(a, 3);
}

__m128i VS_VECTORCALL CalcErrorBlock_AVX2( const __m256i data, const v4i a[8]) noexcept
{
    //
    __m256i a0 = _mm256_load_si256((__m256i*)a[0].data());
    __m256i a1 = _mm256_load_si256((__m256i*)a[4].data());

    // err = 8 * ( sq( average[0] ) + sq( average[1] ) + sq( average[2] ) );
    __m256i a4 = _mm256_madd_epi16(a0, a0);
    __m256i a5 = _mm256_madd_epi16(a1, a1);

    __m256i a6 = _mm256_hadd_epi32(a4, a5);
    __m256i a7 = _mm256_slli_epi32(a6, 3);

    __m256i a8 = _mm256_add_epi32(a7, _mm256_set1_epi32(0x3FFFFFFF)); // Big value to prevent negative values, but small enough to prevent overflow

    // average is not swapped
    // err -= block[0] * 2 * average[0];
    // err -= block[1] * 2 * average[1];
    // err -= block[2] * 2 * average[2];
    __m256i a2 = _mm256_slli_epi16(a0, 1);
    __m256i a3 = _mm256_slli_epi16(a1, 1);
    __m256i b0 = _mm256_madd_epi16(a2, data);
    __m256i b1 = _mm256_madd_epi16(a3, data);

    __m256i b2 = _mm256_hadd_epi32(b0, b1);
    __m256i b3 = _mm256_sub_epi32(a8, b2);
    __m256i b4 = _mm256_hadd_epi32(b3, b3);

    __m256i b5 = _mm256_permutevar8x32_epi32(b4, _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0));

    return _mm256_castsi256_si128(b5);
}

void VS_VECTORCALL ProcessAverages_AVX2(const __m256i d, v4i a[8] ) noexcept
{
    __m256i t = _mm256_add_epi16(_mm256_mullo_epi16(d, _mm256_set1_epi16(31)), _mm256_set1_epi16(128));

    __m256i c = _mm256_srli_epi16(_mm256_add_epi16(t, _mm256_srli_epi16(t, 8)), 8);

    __m256i c1 = _mm256_shuffle_epi32(c, _MM_SHUFFLE(3, 2, 3, 2));
    __m256i diff = _mm256_sub_epi16(c, c1);
    diff = _mm256_max_epi16(diff, _mm256_set1_epi16(-4));
    diff = _mm256_min_epi16(diff, _mm256_set1_epi16(3));

    __m256i co = _mm256_add_epi16(c1, diff);

    c = _mm256_blend_epi16(co, c, 0xF0);

    __m256i a0 = _mm256_or_si256(_mm256_slli_epi16(c, 3), _mm256_srli_epi16(c, 2));

    _mm256_store_si256((__m256i*)a[4].data(), a0);

    __m256i t0 = _mm256_add_epi16(_mm256_mullo_epi16(d, _mm256_set1_epi16(15)), _mm256_set1_epi16(128));
    __m256i t1 = _mm256_srli_epi16(_mm256_add_epi16(t0, _mm256_srli_epi16(t0, 8)), 8);

    __m256i t2 = _mm256_or_si256(t1, _mm256_slli_epi16(t1, 4));

    _mm256_store_si256((__m256i*)a[0].data(), t2);
}

uint64_t VS_VECTORCALL EncodeAverages_AVX2( const v4i a[8], size_t idx ) noexcept
{
    uint64_t d = ( idx << 24 );
    size_t base = idx << 1;

    __m128i a0 = _mm_load_si128((const __m128i*)a[base].data());

    __m128i r0, r1;

    if( ( idx & 0x2 ) == 0 )
    {
        r0 = _mm_srli_epi16(a0, 4);

        __m128i a1 = _mm_unpackhi_epi64(r0, r0);
        r1 = _mm_slli_epi16(a1, 4);
    }
    else
    {
        __m128i a1 = _mm_and_si128(a0, _mm_set1_epi16(-8));

        r0 = _mm_unpackhi_epi64(a1, a1);
        __m128i a2 = _mm_sub_epi16(a1, r0);
        __m128i a3 = _mm_srai_epi16(a2, 3);
        r1 = _mm_and_si128(a3, _mm_set1_epi16(0x07));
    }

    __m128i r2 = _mm_or_si128(r0, r1);
    // do missing swap for average values
    __m128i r3 = _mm_shufflelo_epi16(r2, _MM_SHUFFLE(3, 0, 1, 2));
    __m128i r4 = _mm_packus_epi16(r3, _mm_setzero_si128());
    d |= _mm_cvtsi128_si32(r4);

    return d;
}

uint64_t VS_VECTORCALL CheckSolid_AVX2( const uint8_t* src ) noexcept
{
    __m256i d0 = _mm256_loadu_si256(((__m256i*)src) + 0);
    __m256i d1 = _mm256_loadu_si256(((__m256i*)src) + 1);

    __m256i c = _mm256_broadcastd_epi32(_mm256_castsi256_si128(d0));

    __m256i c0 = _mm256_cmpeq_epi8(d0, c);
    __m256i c1 = _mm256_cmpeq_epi8(d1, c);

    __m256i m = _mm256_and_si256(c0, c1);

    if (!_mm256_testc_si256(m, _mm256_set1_epi32(-1)))
    {
        return 0;
    }

    return 0x02000000 |
        ( (unsigned int)( src[0] & 0xF8 ) << 16 ) |
        ( (unsigned int)( src[1] & 0xF8 ) << 8 ) |
        ( (unsigned int)( src[2] & 0xF8 ) );
}

__m128i VS_VECTORCALL PrepareAverages_AVX2( v4i a[8], const uint8_t* src) noexcept
{
    __m256i sum4 = Sum4_AVX2( src );

    ProcessAverages_AVX2(Average_AVX2( sum4 ), a );

    return CalcErrorBlock_AVX2( sum4, a);
}

__m128i VS_VECTORCALL PrepareAverages_AVX2( v4i a[8], const __m256i sum4) noexcept
{
    ProcessAverages_AVX2(Average_AVX2( sum4 ), a );

    return CalcErrorBlock_AVX2( sum4, a);
}

void VS_VECTORCALL FindBestFit_4x2_AVX2( uint32_t terr[2][8], uint32_t tsel[8], v4i a[8], const uint32_t offset, const uint8_t* data) noexcept
{
    __m256i sel0 = _mm256_setzero_si256();
    __m256i sel1 = _mm256_setzero_si256();

    for (unsigned int j = 0; j < 2; ++j)
    {
        unsigned int bid = offset + 1 - j;

        __m256i squareErrorSum = _mm256_setzero_si256();

        __m128i a0 = _mm_loadl_epi64((const __m128i*)a[bid].data());
        __m256i a1 = _mm256_broadcastq_epi64(a0);

        // Processing one full row each iteration
        for (size_t i = 0; i < 8; i += 4)
        {
            __m128i rgb = _mm_loadu_si128((const __m128i*)(data + i * 4));

            __m256i rgb16 = _mm256_cvtepu8_epi16(rgb);
            __m256i d = _mm256_sub_epi16(a1, rgb16);

            // The scaling values are divided by two and rounded, to allow the differences to be in the range of signed int16
            // This produces slightly different results, but is significant faster
            __m256i pixel0 = _mm256_madd_epi16(d, _mm256_set_epi16(0, 38, 76, 14, 0, 38, 76, 14, 0, 38, 76, 14, 0, 38, 76, 14));
            __m256i pixel1 = _mm256_packs_epi32(pixel0, pixel0);
            __m256i pixel2 = _mm256_hadd_epi16(pixel1, pixel1);
            __m128i pixel3 = _mm256_castsi256_si128(pixel2);

            __m128i pix0 = _mm_broadcastw_epi16(pixel3);
            __m128i pix1 = _mm_broadcastw_epi16(_mm_srli_epi32(pixel3, 16));
            __m256i pixel = _mm256_insertf128_si256(_mm256_castsi128_si256(pix0), pix1, 1);

            // Processing first two pixels of the row
            {
                __m256i pix = _mm256_abs_epi16(pixel);

                // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
                // Since the selector table is symmetrical, we need to calculate the difference only for half of the entries.
                __m256i error0 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[0])));
                __m256i error1 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[1])));

                __m256i minIndex0 = _mm256_and_si256(_mm256_cmpgt_epi16(error0, error1), _mm256_set1_epi16(1));
                __m256i minError = _mm256_min_epi16(error0, error1);

                // Exploiting symmetry of the selector table and use the sign bit
                // This produces slightly different results, but is significant faster
                __m256i minIndex1 = _mm256_srli_epi16(pixel, 15);

                // Interleaving values so madd instruction can be used
                __m256i minErrorLo = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(1, 1, 0, 0));
                __m256i minErrorHi = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(3, 3, 2, 2));

                __m256i minError2 = _mm256_unpacklo_epi16(minErrorLo, minErrorHi);
                // Squaring the minimum error to produce correct values when adding
                __m256i squareError = _mm256_madd_epi16(minError2, minError2);

                squareErrorSum = _mm256_add_epi32(squareErrorSum, squareError);

                // Packing selector bits
                __m256i minIndexLo2 = _mm256_sll_epi16(minIndex0, _mm_cvtsi64_si128(i + j * 8));
                __m256i minIndexHi2 = _mm256_sll_epi16(minIndex1, _mm_cvtsi64_si128(i + j * 8));

                sel0 = _mm256_or_si256(sel0, minIndexLo2);
                sel1 = _mm256_or_si256(sel1, minIndexHi2);
            }

            pixel3 = _mm256_extracti128_si256(pixel2, 1);
            pix0 = _mm_broadcastw_epi16(pixel3);
            pix1 = _mm_broadcastw_epi16(_mm_srli_epi32(pixel3, 16));
            pixel = _mm256_insertf128_si256(_mm256_castsi128_si256(pix0), pix1, 1);

            // Processing second two pixels of the row
            {
                __m256i pix = _mm256_abs_epi16(pixel);

                // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
                // Since the selector table is symmetrical, we need to calculate the difference only for half of the entries.
                __m256i error0 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[0])));
                __m256i error1 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[1])));

                __m256i minIndex0 = _mm256_and_si256(_mm256_cmpgt_epi16(error0, error1), _mm256_set1_epi16(1));
                __m256i minError = _mm256_min_epi16(error0, error1);

                // Exploiting symmetry of the selector table and use the sign bit
                __m256i minIndex1 = _mm256_srli_epi16(pixel, 15);

                // Interleaving values so madd instruction can be used
                __m256i minErrorLo = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(1, 1, 0, 0));
                __m256i minErrorHi = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(3, 3, 2, 2));

                __m256i minError2 = _mm256_unpacklo_epi16(minErrorLo, minErrorHi);
                // Squaring the minimum error to produce correct values when adding
                __m256i squareError = _mm256_madd_epi16(minError2, minError2);

                squareErrorSum = _mm256_add_epi32(squareErrorSum, squareError);

                // Packing selector bits
                __m256i minIndexLo2 = _mm256_sll_epi16(minIndex0, _mm_cvtsi64_si128(i + j * 8));
                __m256i minIndexHi2 = _mm256_sll_epi16(minIndex1, _mm_cvtsi64_si128(i + j * 8));
                __m256i minIndexLo3 = _mm256_slli_epi16(minIndexLo2, 2);
                __m256i minIndexHi3 = _mm256_slli_epi16(minIndexHi2, 2);

                sel0 = _mm256_or_si256(sel0, minIndexLo3);
                sel1 = _mm256_or_si256(sel1, minIndexHi3);
            }
        }

        data += 8 * 4;

        _mm256_store_si256((__m256i*)terr[1 - j], squareErrorSum);
    }

    // Interleave selector bits
    __m256i minIndexLo0 = _mm256_unpacklo_epi16(sel0, sel1);
    __m256i minIndexHi0 = _mm256_unpackhi_epi16(sel0, sel1);

    __m256i minIndexLo1 = _mm256_permute2x128_si256(minIndexLo0, minIndexHi0, (0) | (2 << 4));
    __m256i minIndexHi1 = _mm256_permute2x128_si256(minIndexLo0, minIndexHi0, (1) | (3 << 4));

    __m256i minIndexHi2 = _mm256_slli_epi32(minIndexHi1, 1);

    __m256i sel = _mm256_or_si256(minIndexLo1, minIndexHi2);

    _mm256_store_si256((__m256i*)tsel, sel);
}

void VS_VECTORCALL FindBestFit_2x4_AVX2( uint32_t terr[2][8], uint32_t tsel[8], v4i a[8], const uint32_t offset, const uint8_t* data) noexcept
{
    __m256i sel0 = _mm256_setzero_si256();
    __m256i sel1 = _mm256_setzero_si256();

    __m256i squareErrorSum0 = _mm256_setzero_si256();
    __m256i squareErrorSum1 = _mm256_setzero_si256();

    __m128i a0 = _mm_loadl_epi64((const __m128i*)a[offset + 1].data());
    __m128i a1 = _mm_loadl_epi64((const __m128i*)a[offset + 0].data());

    __m128i a2 = _mm_broadcastq_epi64(a0);
    __m128i a3 = _mm_broadcastq_epi64(a1);
    __m256i a4 = _mm256_insertf128_si256(_mm256_castsi128_si256(a2), a3, 1);

    // Processing one full row each iteration
    for (size_t i = 0; i < 16; i += 4)
    {
        __m128i rgb = _mm_loadu_si128((const __m128i*)(data + i * 4));

        __m256i rgb16 = _mm256_cvtepu8_epi16(rgb);
        __m256i d = _mm256_sub_epi16(a4, rgb16);

        // The scaling values are divided by two and rounded, to allow the differences to be in the range of signed int16
        // This produces slightly different results, but is significant faster
        __m256i pixel0 = _mm256_madd_epi16(d, _mm256_set_epi16(0, 38, 76, 14, 0, 38, 76, 14, 0, 38, 76, 14, 0, 38, 76, 14));
        __m256i pixel1 = _mm256_packs_epi32(pixel0, pixel0);
        __m256i pixel2 = _mm256_hadd_epi16(pixel1, pixel1);
        __m128i pixel3 = _mm256_castsi256_si128(pixel2);

        __m128i pix0 = _mm_broadcastw_epi16(pixel3);
        __m128i pix1 = _mm_broadcastw_epi16(_mm_srli_epi32(pixel3, 16));
        __m256i pixel = _mm256_insertf128_si256(_mm256_castsi128_si256(pix0), pix1, 1);

        // Processing first two pixels of the row
        {
            __m256i pix = _mm256_abs_epi16(pixel);

            // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
            // Since the selector table is symmetrical, we need to calculate the difference only for half of the entries.
            __m256i error0 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[0])));
            __m256i error1 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[1])));

            __m256i minIndex0 = _mm256_and_si256(_mm256_cmpgt_epi16(error0, error1), _mm256_set1_epi16(1));
            __m256i minError = _mm256_min_epi16(error0, error1);

            // Exploiting symmetry of the selector table and use the sign bit
            __m256i minIndex1 = _mm256_srli_epi16(pixel, 15);

            // Interleaving values so madd instruction can be used
            __m256i minErrorLo = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(1, 1, 0, 0));
            __m256i minErrorHi = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(3, 3, 2, 2));

            __m256i minError2 = _mm256_unpacklo_epi16(minErrorLo, minErrorHi);
            // Squaring the minimum error to produce correct values when adding
            __m256i squareError = _mm256_madd_epi16(minError2, minError2);

            squareErrorSum0 = _mm256_add_epi32(squareErrorSum0, squareError);

            // Packing selector bits
            __m256i minIndexLo2 = _mm256_sll_epi16(minIndex0, _mm_cvtsi64_si128(i));
            __m256i minIndexHi2 = _mm256_sll_epi16(minIndex1, _mm_cvtsi64_si128(i));

            sel0 = _mm256_or_si256(sel0, minIndexLo2);
            sel1 = _mm256_or_si256(sel1, minIndexHi2);
        }

        pixel3 = _mm256_extracti128_si256(pixel2, 1);
        pix0 = _mm_broadcastw_epi16(pixel3);
        pix1 = _mm_broadcastw_epi16(_mm_srli_epi32(pixel3, 16));
        pixel = _mm256_insertf128_si256(_mm256_castsi128_si256(pix0), pix1, 1);

        // Processing second two pixels of the row
        {
            __m256i pix = _mm256_abs_epi16(pixel);

            // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
            // Since the selector table is symmetrical, we need to calculate the difference only for half of the entries.
            __m256i error0 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[0])));
            __m256i error1 = _mm256_abs_epi16(_mm256_sub_epi16(pix, _mm256_broadcastsi128_si256(g_table128_SIMD[1])));

            __m256i minIndex0 = _mm256_and_si256(_mm256_cmpgt_epi16(error0, error1), _mm256_set1_epi16(1));
            __m256i minError = _mm256_min_epi16(error0, error1);

            // Exploiting symmetry of the selector table and use the sign bit
            __m256i minIndex1 = _mm256_srli_epi16(pixel, 15);

            // Interleaving values so madd instruction can be used
            __m256i minErrorLo = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(1, 1, 0, 0));
            __m256i minErrorHi = _mm256_permute4x64_epi64(minError, _MM_SHUFFLE(3, 3, 2, 2));

            __m256i minError2 = _mm256_unpacklo_epi16(minErrorLo, minErrorHi);
            // Squaring the minimum error to produce correct values when adding
            __m256i squareError = _mm256_madd_epi16(minError2, minError2);

            squareErrorSum1 = _mm256_add_epi32(squareErrorSum1, squareError);

            // Packing selector bits
            __m256i minIndexLo2 = _mm256_sll_epi16(minIndex0, _mm_cvtsi64_si128(i));
            __m256i minIndexHi2 = _mm256_sll_epi16(minIndex1, _mm_cvtsi64_si128(i));
            __m256i minIndexLo3 = _mm256_slli_epi16(minIndexLo2, 2);
            __m256i minIndexHi3 = _mm256_slli_epi16(minIndexHi2, 2);

            sel0 = _mm256_or_si256(sel0, minIndexLo3);
            sel1 = _mm256_or_si256(sel1, minIndexHi3);
        }
    }

    _mm256_store_si256((__m256i*)terr[1], squareErrorSum0);
    _mm256_store_si256((__m256i*)terr[0], squareErrorSum1);

    // Interleave selector bits
    __m256i minIndexLo0 = _mm256_unpacklo_epi16(sel0, sel1);
    __m256i minIndexHi0 = _mm256_unpackhi_epi16(sel0, sel1);

    __m256i minIndexLo1 = _mm256_permute2x128_si256(minIndexLo0, minIndexHi0, (0) | (2 << 4));
    __m256i minIndexHi1 = _mm256_permute2x128_si256(minIndexLo0, minIndexHi0, (1) | (3 << 4));

    __m256i minIndexHi2 = _mm256_slli_epi32(minIndexHi1, 1);

    __m256i sel = _mm256_or_si256(minIndexLo1, minIndexHi2);

    _mm256_store_si256((__m256i*)tsel, sel);
}

uint64_t VS_VECTORCALL EncodeSelectors_AVX2( uint64_t d, const uint32_t terr[2][8], const uint32_t tsel[8], const bool rotate) noexcept
{
    size_t tidx[2];

    // Get index of minimum error (terr[0] and terr[1])
    __m256i err0 = _mm256_load_si256((const __m256i*)terr[0]);
    __m256i err1 = _mm256_load_si256((const __m256i*)terr[1]);

    __m256i errLo = _mm256_permute2x128_si256(err0, err1, (0) | (2 << 4));
    __m256i errHi = _mm256_permute2x128_si256(err0, err1, (1) | (3 << 4));

    __m256i errMin0 = _mm256_min_epu32(errLo, errHi);

    __m256i errMin1 = _mm256_shuffle_epi32(errMin0, _MM_SHUFFLE(2, 3, 0, 1));
    __m256i errMin2 = _mm256_min_epu32(errMin0, errMin1);

    __m256i errMin3 = _mm256_shuffle_epi32(errMin2, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i errMin4 = _mm256_min_epu32(errMin3, errMin2);

    __m256i errMin5 = _mm256_permute2x128_si256(errMin4, errMin4, (0) | (0 << 4));
    __m256i errMin6 = _mm256_permute2x128_si256(errMin4, errMin4, (1) | (1 << 4));

    __m256i errMask0 = _mm256_cmpeq_epi32(errMin5, err0);
    __m256i errMask1 = _mm256_cmpeq_epi32(errMin6, err1);

    uint32_t mask0 = _mm256_movemask_epi8(errMask0);
    uint32_t mask1 = _mm256_movemask_epi8(errMask1);

    tidx[0] = _bit_scan_forward(mask0) >> 2;
    tidx[1] = _bit_scan_forward(mask1) >> 2;

    d |= tidx[0] << 26;
    d |= tidx[1] << 29;

    unsigned int t0 = tsel[tidx[0]];
    unsigned int t1 = tsel[tidx[1]];

    if (!rotate)
    {
        t0 &= 0xFF00FF00;
        t1 &= 0x00FF00FF;
    }
    else
    {
        t0 &= 0xCCCCCCCC;
        t1 &= 0x33333333;
    }

    // Flip selectors from sign bit
    unsigned int t2 = (t0 | t1) ^ 0xFFFF0000;

    return d | static_cast<uint64_t>(_bswap(t2)) << 32;
}

__m128i VS_VECTORCALL r6g7b6_AVX2(__m128 cof, __m128 chf, __m128 cvf) noexcept
{
    __m128i co = _mm_cvttps_epi32(cof);
    __m128i ch = _mm_cvttps_epi32(chf);
    __m128i cv = _mm_cvttps_epi32(cvf);

    __m128i coh = _mm_packus_epi32(co, ch);
    __m128i cv0 = _mm_packus_epi32(cv, _mm_setzero_si128());

	__m256i cohv0 = _mm256_inserti128_si256(_mm256_castsi128_si256(coh), cv0, 1);
    __m256i cohv1 = _mm256_min_epu16(cohv0, _mm256_set1_epi16(1023));

    __m256i cohv2 = _mm256_sub_epi16(cohv1, _mm256_set1_epi16(15));
    __m256i cohv3 = _mm256_srai_epi16(cohv2, 1);

    __m256i cohvrb0 = _mm256_add_epi16(cohv3, _mm256_set1_epi16(11));
    __m256i cohvrb1 = _mm256_add_epi16(cohv3, _mm256_set1_epi16(4));
    __m256i cohvg0 = _mm256_add_epi16(cohv3, _mm256_set1_epi16(9));
    __m256i cohvg1 = _mm256_add_epi16(cohv3, _mm256_set1_epi16(6));

    __m256i cohvrb2 = _mm256_srai_epi16(cohvrb0, 7);
    __m256i cohvrb3 = _mm256_srai_epi16(cohvrb1, 7);
    __m256i cohvg2 = _mm256_srai_epi16(cohvg0, 8);
    __m256i cohvg3 = _mm256_srai_epi16(cohvg1, 8);

    __m256i cohvrb4 = _mm256_sub_epi16(cohvrb0, cohvrb2);
    __m256i cohvrb5 = _mm256_sub_epi16(cohvrb4, cohvrb3);
    __m256i cohvg4 = _mm256_sub_epi16(cohvg0, cohvg2);
    __m256i cohvg5 = _mm256_sub_epi16(cohvg4, cohvg3);

    __m256i cohvrb6 = _mm256_srai_epi16(cohvrb5, 3);
    __m256i cohvg6 = _mm256_srai_epi16(cohvg5, 2);

	__m256i cohv4 = _mm256_blend_epi16(cohvg6, cohvrb6, 0x55);

    __m128i cohv5 = _mm_packus_epi16(_mm256_castsi256_si128(cohv4), _mm256_extracti128_si256(cohv4, 1));
    return _mm_shuffle_epi8(cohv5, _mm_setr_epi8(6, 5, 4, -1, 2, 1, 0, -1, 10, 9, 8, -1, -1, -1, -1, -1));
}

struct Plane
{
	uint64_t plane;
	uint64_t error;
	__m256i sum4;
};

Plane Planar_AVX2(const uint8_t* src)
{
    __m128i d0 = _mm_loadu_si128(((__m128i*)src) + 0);
    __m128i d1 = _mm_loadu_si128(((__m128i*)src) + 1);
    __m128i d2 = _mm_loadu_si128(((__m128i*)src) + 2);
    __m128i d3 = _mm_loadu_si128(((__m128i*)src) + 3);

    __m128i rgb0 = _mm_shuffle_epi8(d0, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, -1, -1, -1, -1));
    __m128i rgb1 = _mm_shuffle_epi8(d1, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, -1, -1, -1, -1));
    __m128i rgb2 = _mm_shuffle_epi8(d2, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, -1, -1, -1, -1));
    __m128i rgb3 = _mm_shuffle_epi8(d3, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, -1, -1, -1, -1));

    __m128i rg0 = _mm_unpacklo_epi32(rgb0, rgb1);
    __m128i rg1 = _mm_unpacklo_epi32(rgb2, rgb3);
    __m128i b0 = _mm_unpackhi_epi32(rgb0, rgb1);
    __m128i b1 = _mm_unpackhi_epi32(rgb2, rgb3);

    // swap channels
    __m128i b8 = _mm_unpacklo_epi64(rg0, rg1);
    __m128i g8 = _mm_unpackhi_epi64(rg0, rg1);
    __m128i r8 = _mm_unpacklo_epi64(b0, b1);

    __m128i t0 = _mm_sad_epu8(r8, _mm_setzero_si128());
    __m128i t1 = _mm_sad_epu8(g8, _mm_setzero_si128());
    __m128i t2 = _mm_sad_epu8(b8, _mm_setzero_si128());

    __m128i r8s = _mm_shuffle_epi8(r8, _mm_set_epi8(0xF, 0xE, 0xB, 0xA, 0x7, 0x6, 0x3, 0x2, 0xD, 0xC, 0x9, 0x8, 0x5, 0x4, 0x1, 0x0));
    __m128i g8s = _mm_shuffle_epi8(g8, _mm_set_epi8(0xF, 0xE, 0xB, 0xA, 0x7, 0x6, 0x3, 0x2, 0xD, 0xC, 0x9, 0x8, 0x5, 0x4, 0x1, 0x0));
    __m128i b8s = _mm_shuffle_epi8(b8, _mm_set_epi8(0xF, 0xE, 0xB, 0xA, 0x7, 0x6, 0x3, 0x2, 0xD, 0xC, 0x9, 0x8, 0x5, 0x4, 0x1, 0x0));

    __m128i s0 = _mm_sad_epu8(r8s, _mm_setzero_si128());
    __m128i s1 = _mm_sad_epu8(g8s, _mm_setzero_si128());
    __m128i s2 = _mm_sad_epu8(b8s, _mm_setzero_si128());

    __m256i sr0 = _mm256_insertf128_si256(_mm256_castsi128_si256(t0), s0, 1);
    __m256i sg0 = _mm256_insertf128_si256(_mm256_castsi128_si256(t1), s1, 1);
    __m256i sb0 = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), s2, 1);

    __m256i sr1 = _mm256_slli_epi64(sr0, 32);
    __m256i sg1 = _mm256_slli_epi64(sg0, 16);

    __m256i srb = _mm256_or_si256(sr1, sb0);
    __m256i srgb = _mm256_or_si256(srb, sg1);

    __m128i t3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i t4 = _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i t5 = _mm_hadd_epi32(t3, t4);
    __m128i t6 = _mm_shuffle_epi32(t5, _MM_SHUFFLE(1, 1, 1, 1));
    __m128i t7 = _mm_shuffle_epi32(t5, _MM_SHUFFLE(2, 2, 2, 2));

    __m256i sr = _mm256_broadcastw_epi16(t5);
    __m256i sg = _mm256_broadcastw_epi16(t6);
    __m256i sb = _mm256_broadcastw_epi16(t7);

    __m256i r08 = _mm256_cvtepu8_epi16(r8);
    __m256i g08 = _mm256_cvtepu8_epi16(g8);
    __m256i b08 = _mm256_cvtepu8_epi16(b8);

    __m256i r16 = _mm256_slli_epi16(r08, 4);
    __m256i g16 = _mm256_slli_epi16(g08, 4);
    __m256i b16 = _mm256_slli_epi16(b08, 4);

    __m256i difR0 = _mm256_sub_epi16(r16, sr);
    __m256i difG0 = _mm256_sub_epi16(g16, sg);
    __m256i difB0 = _mm256_sub_epi16(b16, sb);

    __m256i difRyz = _mm256_madd_epi16(difR0, _mm256_set_epi16(255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255));
    __m256i difGyz = _mm256_madd_epi16(difG0, _mm256_set_epi16(255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255));
    __m256i difByz = _mm256_madd_epi16(difB0, _mm256_set_epi16(255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255, 255, 85, -85, -255));

    __m256i difRxz = _mm256_madd_epi16(difR0, _mm256_set_epi16(255, 255, 255, 255, 85, 85, 85, 85, -85, -85, -85, -85, -255, -255, -255, -255));
    __m256i difGxz = _mm256_madd_epi16(difG0, _mm256_set_epi16(255, 255, 255, 255, 85, 85, 85, 85, -85, -85, -85, -85, -255, -255, -255, -255));
    __m256i difBxz = _mm256_madd_epi16(difB0, _mm256_set_epi16(255, 255, 255, 255, 85, 85, 85, 85, -85, -85, -85, -85, -255, -255, -255, -255));

    __m256i difRGyz = _mm256_hadd_epi32(difRyz, difGyz);
    __m256i difByzxz = _mm256_hadd_epi32(difByz, difBxz);

    __m256i difRGxz = _mm256_hadd_epi32(difRxz, difGxz);

    __m128i sumRGyz = _mm_add_epi32(_mm256_castsi256_si128(difRGyz), _mm256_extracti128_si256(difRGyz, 1));
    __m128i sumByzxz = _mm_add_epi32(_mm256_castsi256_si128(difByzxz), _mm256_extracti128_si256(difByzxz, 1));
    __m128i sumRGxz = _mm_add_epi32(_mm256_castsi256_si128(difRGxz), _mm256_extracti128_si256(difRGxz, 1));

    __m128i sumRGByz = _mm_hadd_epi32(sumRGyz, sumByzxz);
    __m128i sumRGByzxz = _mm_hadd_epi32(sumRGxz, sumByzxz);

    __m128i sumRGBxz = _mm_shuffle_epi32(sumRGByzxz, _MM_SHUFFLE(2, 3, 1, 0));

    __m128 sumRGByzf = _mm_cvtepi32_ps(sumRGByz);
    __m128 sumRGBxzf = _mm_cvtepi32_ps(sumRGBxz);

    const float value = (255 * 255 * 8.0f + 85 * 85 * 8.0f) * 16.0f;

    __m128 scale = _mm_set1_ps(-4.0f / value);

    __m128 af = _mm_mul_ps(sumRGBxzf, scale);
    __m128 bf = _mm_mul_ps(sumRGByzf, scale);

    __m128 df = _mm_mul_ps(_mm_cvtepi32_ps(t5), _mm_set1_ps(4.0f / 16.0f));

    // calculating the three colors RGBO, RGBH, and RGBV.  RGB = df - af * x - bf * y;
    __m128 cof0 = _mm_fnmadd_ps(af, _mm_set1_ps(-255.0f), _mm_fnmadd_ps(bf, _mm_set1_ps(-255.0f), df));
    __m128 chf0 = _mm_fnmadd_ps(af, _mm_set1_ps( 425.0f), _mm_fnmadd_ps(bf, _mm_set1_ps(-255.0f), df));
    __m128 cvf0 = _mm_fnmadd_ps(af, _mm_set1_ps(-255.0f), _mm_fnmadd_ps(bf, _mm_set1_ps( 425.0f), df));

    // convert to r6g7b6
    __m128i cohv = r6g7b6_AVX2(cof0, chf0, cvf0);

    uint64_t rgbho = _mm_extract_epi64(cohv, 0);
    uint32_t rgbv0 = _mm_extract_epi32(cohv, 2);

	// Error calculation
	auto ro0 = (rgbho >> 48) & 0x3F;
	auto go0 = (rgbho >> 40) & 0x7F;
	auto bo0 = (rgbho >> 32) & 0x3F;
	auto ro1 = (ro0 >> 4) | (ro0 << 2);
	auto go1 = (go0 >> 6) | (go0 << 1);
	auto bo1 = (bo0 >> 4) | (bo0 << 2);
	auto ro2 = (ro1 << 2) + 2;
	auto go2 = (go1 << 2) + 2;
	auto bo2 = (bo1 << 2) + 2;

    __m256i ro3 = _mm256_set1_epi16(ro2);
    __m256i go3 = _mm256_set1_epi16(go2);
    __m256i bo3 = _mm256_set1_epi16(bo2);

	auto rh0 = (rgbho >> 16) & 0x3F;
	auto gh0 = (rgbho >>  8) & 0x7F;
	auto bh0 = (rgbho >>  0) & 0x3F;
	auto rh1 = (rh0 >> 4) | (rh0 << 2);
	auto gh1 = (gh0 >> 6) | (gh0 << 1);
	auto bh1 = (bh0 >> 4) | (bh0 << 2);

	auto rh2 = rh1 - ro1;
	auto gh2 = gh1 - go1;
	auto bh2 = bh1 - bo1;

    __m256i rh3 = _mm256_set1_epi16(rh2);
    __m256i gh3 = _mm256_set1_epi16(gh2);
    __m256i bh3 = _mm256_set1_epi16(bh2);

	auto rv0 = (rgbv0 >> 16) & 0x3F;
	auto gv0 = (rgbv0 >>  8) & 0x7F;
	auto bv0 = (rgbv0 >>  0) & 0x3F;
	auto rv1 = (rv0 >> 4) | (rv0 << 2);
	auto gv1 = (gv0 >> 6) | (gv0 << 1);
	auto bv1 = (bv0 >> 4) | (bv0 << 2);

	auto rv2 = rv1 - ro1;
	auto gv2 = gv1 - go1;
	auto bv2 = bv1 - bo1;

    __m256i rv3 = _mm256_set1_epi16(rv2);
    __m256i gv3 = _mm256_set1_epi16(gv2);
    __m256i bv3 = _mm256_set1_epi16(bv2);

    __m256i x = _mm256_set_epi16(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

    __m256i rh4 = _mm256_mullo_epi16(rh3, x);
    __m256i gh4 = _mm256_mullo_epi16(gh3, x);
    __m256i bh4 = _mm256_mullo_epi16(bh3, x);

    __m256i y = _mm256_set_epi16(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

    __m256i rv4 = _mm256_mullo_epi16(rv3, y);
    __m256i gv4 = _mm256_mullo_epi16(gv3, y);
    __m256i bv4 = _mm256_mullo_epi16(bv3, y);

    __m256i rxy = _mm256_add_epi16(rh4, rv4);
    __m256i gxy = _mm256_add_epi16(gh4, gv4);
    __m256i bxy = _mm256_add_epi16(bh4, bv4);

    __m256i rp0 = _mm256_add_epi16(rxy, ro3);
    __m256i gp0 = _mm256_add_epi16(gxy, go3);
    __m256i bp0 = _mm256_add_epi16(bxy, bo3);

    __m256i rp1 = _mm256_srai_epi16(rp0, 2);
    __m256i gp1 = _mm256_srai_epi16(gp0, 2);
    __m256i bp1 = _mm256_srai_epi16(bp0, 2);

    __m256i rp2 = _mm256_max_epi16(_mm256_min_epi16(rp1, _mm256_set1_epi16(255)), _mm256_setzero_si256());
    __m256i gp2 = _mm256_max_epi16(_mm256_min_epi16(gp1, _mm256_set1_epi16(255)), _mm256_setzero_si256());
    __m256i bp2 = _mm256_max_epi16(_mm256_min_epi16(bp1, _mm256_set1_epi16(255)), _mm256_setzero_si256());

    __m256i rdif = _mm256_sub_epi16(r08, rp2);
    __m256i gdif = _mm256_sub_epi16(g08, gp2);
    __m256i bdif = _mm256_sub_epi16(b08, bp2);

    __m256i rerr = _mm256_mullo_epi16(rdif, _mm256_set1_epi16(38));
    __m256i gerr = _mm256_mullo_epi16(gdif, _mm256_set1_epi16(76));
    __m256i berr = _mm256_mullo_epi16(bdif, _mm256_set1_epi16(14));

    __m256i sum0 = _mm256_add_epi16(rerr, gerr);
    __m256i sum1 = _mm256_add_epi16(sum0, berr);

    __m256i sum2 = _mm256_madd_epi16(sum1, sum1);

    __m128i sum3 = _mm_add_epi32(_mm256_castsi256_si128(sum2), _mm256_extracti128_si256(sum2, 1));

	uint32_t err0 = _mm_extract_epi32(sum3, 0);
	uint32_t err1 = _mm_extract_epi32(sum3, 1);
	uint32_t err2 = _mm_extract_epi32(sum3, 2);
	uint32_t err3 = _mm_extract_epi32(sum3, 3);

	uint64_t error = err0 + err1 + err2 + err3;
	/**/

    uint32_t rgbv = _pext_u32(rgbv0, 0x3F7F3F);
    uint64_t rgbho0 = _pext_u64(rgbho, 0x3F7F3F003F7F3F);

    uint32_t hi = rgbv | ((rgbho0 & 0x1FFF) << 19);
    uint32_t lo = _pdep_u32(rgbho0 >> 13, 0x7F7F1BFD);

    uint32_t idx = _pext_u64(rgbho, 0x20201E00000000);
    lo |= _pdep_u32(g_flags_AVX2[idx], 0x8080E402);
    uint64_t result = static_cast<uint32_t>(_bswap(lo));
    result |= static_cast<uint64_t>(static_cast<uint32_t>(_bswap(hi))) << 32;

	Plane plane;

	plane.plane = result;
	plane.error = error;
	plane.sum4 = _mm256_permute4x64_epi64(srgb, _MM_SHUFFLE(2, 3, 0, 1));

    return plane;
}

uint64_t VS_VECTORCALL EncodeSelectors_AVX2( uint64_t d, const uint32_t terr[2][8], const uint32_t tsel[8], const bool rotate, const uint64_t value, const uint32_t error) noexcept
{
    size_t tidx[2];

    // Get index of minimum error (terr[0] and terr[1])
    __m256i err0 = _mm256_load_si256((const __m256i*)terr[0]);
    __m256i err1 = _mm256_load_si256((const __m256i*)terr[1]);

    __m256i errLo = _mm256_permute2x128_si256(err0, err1, (0) | (2 << 4));
    __m256i errHi = _mm256_permute2x128_si256(err0, err1, (1) | (3 << 4));

    __m256i errMin0 = _mm256_min_epu32(errLo, errHi);

    __m256i errMin1 = _mm256_shuffle_epi32(errMin0, _MM_SHUFFLE(2, 3, 0, 1));
    __m256i errMin2 = _mm256_min_epu32(errMin0, errMin1);

    __m256i errMin3 = _mm256_shuffle_epi32(errMin2, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i errMin4 = _mm256_min_epu32(errMin3, errMin2);

    __m256i errMin5 = _mm256_permute2x128_si256(errMin4, errMin4, (0) | (0 << 4));
    __m256i errMin6 = _mm256_permute2x128_si256(errMin4, errMin4, (1) | (1 << 4));

    __m256i errMask0 = _mm256_cmpeq_epi32(errMin5, err0);
    __m256i errMask1 = _mm256_cmpeq_epi32(errMin6, err1);

    uint32_t mask0 = _mm256_movemask_epi8(errMask0);
    uint32_t mask1 = _mm256_movemask_epi8(errMask1);

    tidx[0] = _bit_scan_forward(mask0) >> 2;
    tidx[1] = _bit_scan_forward(mask1) >> 2;

	if ((terr[0][tidx[0]] + terr[1][tidx[1]]) >= error)
	{
		return value;
	}

    d |= tidx[0] << 26;
    d |= tidx[1] << 29;

    unsigned int t0 = tsel[tidx[0]];
    unsigned int t1 = tsel[tidx[1]];

    if (!rotate)
    {
        t0 &= 0xFF00FF00;
        t1 &= 0x00FF00FF;
    }
    else
    {
        t0 &= 0xCCCCCCCC;
        t1 &= 0x33333333;
    }

    // Flip selectors from sign bit
    unsigned int t2 = (t0 | t1) ^ 0xFFFF0000;

    return d | static_cast<uint64_t>(_bswap(t2)) << 32;
}

}

uint64_t ProcessRGB_AVX2( const uint8_t* src )
{
    uint64_t d = CheckSolid_AVX2( src );
    if( d != 0 ) return d;

    alignas(32) v4i a[8];

    __m128i err0 = PrepareAverages_AVX2( a, src );

    // Get index of minimum error (err0)
    __m128i err1 = _mm_shuffle_epi32(err0, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i errMin0 = _mm_min_epu32(err0, err1);

    __m128i errMin1 = _mm_shuffle_epi32(errMin0, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i errMin2 = _mm_min_epu32(errMin1, errMin0);

    __m128i errMask = _mm_cmpeq_epi32(errMin2, err0);

    uint32_t mask = _mm_movemask_epi8(errMask);

    uint32_t idx = _bit_scan_forward(mask) >> 2;

    d |= EncodeAverages_AVX2( a, idx );

    alignas(32) uint32_t terr[2][8] = {};
    alignas(32) uint32_t tsel[8];

    if ((idx == 0) || (idx == 2))
    {
        FindBestFit_4x2_AVX2( terr, tsel, a, idx * 2, src );
    }
    else
    {
        FindBestFit_2x4_AVX2( terr, tsel, a, idx * 2, src );
    }

    return EncodeSelectors_AVX2( d, terr, tsel, (idx % 2) == 1 );
}

uint64_t ProcessRGB_4x2_AVX2( const uint8_t* src )
{
    uint64_t d = CheckSolid_AVX2( src );
    if( d != 0 ) return d;

    alignas(32) v4i a[8];

    __m128i err0 = PrepareAverages_AVX2( a, src );

    uint32_t idx = _mm_extract_epi32(err0, 0) < _mm_extract_epi32(err0, 2) ? 0 : 2;

    d |= EncodeAverages_AVX2( a, idx );

    alignas(32) uint32_t terr[2][8] = {};
    alignas(32) uint32_t tsel[8];

    FindBestFit_4x2_AVX2( terr, tsel, a, idx * 2, src );

    return EncodeSelectors_AVX2( d, terr, tsel, false);
}

uint64_t ProcessRGB_2x4_AVX2( const uint8_t* src )
{
    uint64_t d = CheckSolid_AVX2( src );
    if( d != 0 ) return d;

    alignas(32) v4i a[8];

    __m128i err0 = PrepareAverages_AVX2( a, src );

    uint32_t idx = _mm_extract_epi32(err0, 1) < _mm_extract_epi32(err0, 3) ? 1 : 3;

    d |= EncodeAverages_AVX2( a, idx );

    alignas(32) uint32_t terr[2][8] = {};
    alignas(32) uint32_t tsel[8];

    FindBestFit_2x4_AVX2( terr, tsel, a, idx * 2, src );

    return EncodeSelectors_AVX2( d, terr, tsel, true);
}

uint64_t ProcessRGB_ETC2_AVX2( const uint8_t* src )
{
    auto plane = Planar_AVX2( src );

    alignas(32) v4i a[8];

    __m128i err0 = PrepareAverages_AVX2( a, plane.sum4 );

    // Get index of minimum error (err0)
    __m128i err1 = _mm_shuffle_epi32(err0, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i errMin0 = _mm_min_epu32(err0, err1);

    __m128i errMin1 = _mm_shuffle_epi32(errMin0, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i errMin2 = _mm_min_epu32(errMin1, errMin0);

    __m128i errMask = _mm_cmpeq_epi32(errMin2, err0);

    uint32_t mask = _mm_movemask_epi8(errMask);

    size_t idx = _bit_scan_forward(mask) >> 2;

    uint64_t d = EncodeAverages_AVX2( a, idx );

    alignas(32) uint32_t terr[2][8] = {};
    alignas(32) uint32_t tsel[8];

    if ((idx == 0) || (idx == 2))
    {
        FindBestFit_4x2_AVX2( terr, tsel, a, idx * 2, src );
    }
    else
    {
        FindBestFit_2x4_AVX2( terr, tsel, a, idx * 2, src );
    }

    return EncodeSelectors_AVX2( d, terr, tsel, (idx % 2) == 1, plane.plane, plane.error );
}

#ifndef _MSC_VER
#  pragma GCC pop_options
#endif

#endif
