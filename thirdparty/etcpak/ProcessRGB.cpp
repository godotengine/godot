#include <array>
#include <string.h>
#include <limits>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#include "Dither.hpp"
#include "ForceInline.hpp"
#include "Math.hpp"
#include "ProcessCommon.hpp"
#include "ProcessRGB.hpp"
#include "Tables.hpp"
#include "Vector.hpp"
#if defined __SSE4_1__ || defined __AVX2__ || defined _MSC_VER
#  ifdef _MSC_VER
#    include <intrin.h>
#    include <Windows.h>
#    define _bswap(x) _byteswap_ulong(x)
#    define _bswap64(x) _byteswap_uint64(x)
#  else
#    include <x86intrin.h>
#  endif
#endif

#ifndef _bswap
#  define _bswap(x) __builtin_bswap32(x)
#  define _bswap64(x) __builtin_bswap64(x)
#endif

namespace
{

#if defined _MSC_VER && !defined __clang__
static etcpak_force_inline unsigned long _bit_scan_forward( unsigned long mask )
{
    unsigned long ret;
    _BitScanForward( &ret, mask );
    return ret;
}
#endif

typedef std::array<uint16_t, 4> v4i;

#ifdef __AVX2__
static etcpak_force_inline __m256i Sum4_AVX2( const uint8_t* data) noexcept
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

static etcpak_force_inline __m256i Average_AVX2( const __m256i data) noexcept
{
    __m256i a = _mm256_add_epi16(data, _mm256_set1_epi16(4));

    return _mm256_srli_epi16(a, 3);
}

static etcpak_force_inline __m128i CalcErrorBlock_AVX2( const __m256i data, const v4i a[8]) noexcept
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

static etcpak_force_inline void ProcessAverages_AVX2(const __m256i d, v4i a[8] ) noexcept
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

static etcpak_force_inline uint64_t EncodeAverages_AVX2( const v4i a[8], size_t idx ) noexcept
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

static etcpak_force_inline uint64_t CheckSolid_AVX2( const uint8_t* src ) noexcept
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

static etcpak_force_inline __m128i PrepareAverages_AVX2( v4i a[8], const uint8_t* src) noexcept
{
    __m256i sum4 = Sum4_AVX2( src );

    ProcessAverages_AVX2(Average_AVX2( sum4 ), a );

    return CalcErrorBlock_AVX2( sum4, a);
}

static etcpak_force_inline __m128i PrepareAverages_AVX2( v4i a[8], const __m256i sum4) noexcept
{
    ProcessAverages_AVX2(Average_AVX2( sum4 ), a );

    return CalcErrorBlock_AVX2( sum4, a);
}

static etcpak_force_inline void FindBestFit_4x2_AVX2( uint32_t terr[2][8], uint32_t tsel[8], v4i a[8], const uint32_t offset, const uint8_t* data) noexcept
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

static etcpak_force_inline void FindBestFit_2x4_AVX2( uint32_t terr[2][8], uint32_t tsel[8], v4i a[8], const uint32_t offset, const uint8_t* data) noexcept
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

static etcpak_force_inline uint64_t EncodeSelectors_AVX2( uint64_t d, const uint32_t terr[2][8], const uint32_t tsel[8], const bool rotate) noexcept
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

static etcpak_force_inline __m128i r6g7b6_AVX2(__m128 cof, __m128 chf, __m128 cvf) noexcept
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

static etcpak_force_inline Plane Planar_AVX2(const uint8_t* src)
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

    uint32_t rgbv = ( rgbv0 & 0x3F ) | ( ( rgbv0 >> 2 ) & 0x1FC0 ) | ( ( rgbv0 >> 3 ) & 0x7E000 );
    uint64_t rgbho0_ = ( rgbho & 0x3F0000003F ) | ( ( rgbho >> 2 ) & 0x1FC000001FC0 ) | ( ( rgbho >> 3 ) & 0x7E0000007E000 );
    uint64_t rgbho0 = ( rgbho0_ & 0x7FFFF ) | ( ( rgbho0_ >> 13 ) & 0x3FFFF80000 );

    uint32_t hi = rgbv | ((rgbho0 & 0x1FFF) << 19);
    rgbho0 >>= 13;
    uint32_t lo = ( rgbho0 & 0x1 ) | ( ( rgbho0 & 0x1FE ) << 1 ) | ( ( rgbho0 & 0x600 ) << 2 ) | ( ( rgbho0 & 0x3F800 ) << 5 ) | ( ( rgbho0 & 0x1FC0000 ) << 6 );

    uint32_t idx = ( ( rgbho >> 33 ) & 0xF ) | ( ( rgbho >> 41 ) & 0x10 ) | ( ( rgbho >> 48 ) & 0x20 );
    lo |= g_flags[idx];
    uint64_t result = static_cast<uint32_t>(_bswap(lo));
    result |= static_cast<uint64_t>(static_cast<uint32_t>(_bswap(hi))) << 32;

    Plane plane;

    plane.plane = result;
    plane.error = error;
    plane.sum4 = _mm256_permute4x64_epi64(srgb, _MM_SHUFFLE(2, 3, 0, 1));

    return plane;
}

static etcpak_force_inline uint64_t EncodeSelectors_AVX2( uint64_t d, const uint32_t terr[2][8], const uint32_t tsel[8], const bool rotate, const uint64_t value, const uint32_t error) noexcept
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

#endif

static etcpak_force_inline void Average( const uint8_t* data, v4i* a )
{
#ifdef __SSE4_1__
    __m128i d0 = _mm_loadu_si128(((__m128i*)data) + 0);
    __m128i d1 = _mm_loadu_si128(((__m128i*)data) + 1);
    __m128i d2 = _mm_loadu_si128(((__m128i*)data) + 2);
    __m128i d3 = _mm_loadu_si128(((__m128i*)data) + 3);

    __m128i d0l = _mm_unpacklo_epi8(d0, _mm_setzero_si128());
    __m128i d0h = _mm_unpackhi_epi8(d0, _mm_setzero_si128());
    __m128i d1l = _mm_unpacklo_epi8(d1, _mm_setzero_si128());
    __m128i d1h = _mm_unpackhi_epi8(d1, _mm_setzero_si128());
    __m128i d2l = _mm_unpacklo_epi8(d2, _mm_setzero_si128());
    __m128i d2h = _mm_unpackhi_epi8(d2, _mm_setzero_si128());
    __m128i d3l = _mm_unpacklo_epi8(d3, _mm_setzero_si128());
    __m128i d3h = _mm_unpackhi_epi8(d3, _mm_setzero_si128());

    __m128i sum0 = _mm_add_epi16(d0l, d1l);
    __m128i sum1 = _mm_add_epi16(d0h, d1h);
    __m128i sum2 = _mm_add_epi16(d2l, d3l);
    __m128i sum3 = _mm_add_epi16(d2h, d3h);

    __m128i sum0l = _mm_unpacklo_epi16(sum0, _mm_setzero_si128());
    __m128i sum0h = _mm_unpackhi_epi16(sum0, _mm_setzero_si128());
    __m128i sum1l = _mm_unpacklo_epi16(sum1, _mm_setzero_si128());
    __m128i sum1h = _mm_unpackhi_epi16(sum1, _mm_setzero_si128());
    __m128i sum2l = _mm_unpacklo_epi16(sum2, _mm_setzero_si128());
    __m128i sum2h = _mm_unpackhi_epi16(sum2, _mm_setzero_si128());
    __m128i sum3l = _mm_unpacklo_epi16(sum3, _mm_setzero_si128());
    __m128i sum3h = _mm_unpackhi_epi16(sum3, _mm_setzero_si128());

    __m128i b0 = _mm_add_epi32(sum0l, sum0h);
    __m128i b1 = _mm_add_epi32(sum1l, sum1h);
    __m128i b2 = _mm_add_epi32(sum2l, sum2h);
    __m128i b3 = _mm_add_epi32(sum3l, sum3h);

    __m128i a0 = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(b2, b3), _mm_set1_epi32(4)), 3);
    __m128i a1 = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(b0, b1), _mm_set1_epi32(4)), 3);
    __m128i a2 = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(b1, b3), _mm_set1_epi32(4)), 3);
    __m128i a3 = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(b0, b2), _mm_set1_epi32(4)), 3);

    _mm_storeu_si128((__m128i*)&a[0], _mm_packus_epi32(_mm_shuffle_epi32(a0, _MM_SHUFFLE(3, 0, 1, 2)), _mm_shuffle_epi32(a1, _MM_SHUFFLE(3, 0, 1, 2))));
    _mm_storeu_si128((__m128i*)&a[2], _mm_packus_epi32(_mm_shuffle_epi32(a2, _MM_SHUFFLE(3, 0, 1, 2)), _mm_shuffle_epi32(a3, _MM_SHUFFLE(3, 0, 1, 2))));
#elif defined __ARM_NEON
    uint8x16x2_t t0 = vzipq_u8(vld1q_u8(data +  0), uint8x16_t());
    uint8x16x2_t t1 = vzipq_u8(vld1q_u8(data + 16), uint8x16_t());
    uint8x16x2_t t2 = vzipq_u8(vld1q_u8(data + 32), uint8x16_t());
    uint8x16x2_t t3 = vzipq_u8(vld1q_u8(data + 48), uint8x16_t());

    uint16x8x2_t d0 = { vreinterpretq_u16_u8(t0.val[0]), vreinterpretq_u16_u8(t0.val[1]) };
    uint16x8x2_t d1 = { vreinterpretq_u16_u8(t1.val[0]), vreinterpretq_u16_u8(t1.val[1]) };
    uint16x8x2_t d2 = { vreinterpretq_u16_u8(t2.val[0]), vreinterpretq_u16_u8(t2.val[1]) };
    uint16x8x2_t d3 = { vreinterpretq_u16_u8(t3.val[0]), vreinterpretq_u16_u8(t3.val[1]) };

    uint16x8x2_t s0 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d0.val[0] ), vreinterpretq_s16_u16( d1.val[0] ) ) ), uint16x8_t());
    uint16x8x2_t s1 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d0.val[1] ), vreinterpretq_s16_u16( d1.val[1] ) ) ), uint16x8_t());
    uint16x8x2_t s2 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d2.val[0] ), vreinterpretq_s16_u16( d3.val[0] ) ) ), uint16x8_t());
    uint16x8x2_t s3 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d2.val[1] ), vreinterpretq_s16_u16( d3.val[1] ) ) ), uint16x8_t());

    uint32x4x2_t sum0 = { vreinterpretq_u32_u16(s0.val[0]), vreinterpretq_u32_u16(s0.val[1]) };
    uint32x4x2_t sum1 = { vreinterpretq_u32_u16(s1.val[0]), vreinterpretq_u32_u16(s1.val[1]) };
    uint32x4x2_t sum2 = { vreinterpretq_u32_u16(s2.val[0]), vreinterpretq_u32_u16(s2.val[1]) };
    uint32x4x2_t sum3 = { vreinterpretq_u32_u16(s3.val[0]), vreinterpretq_u32_u16(s3.val[1]) };

    uint32x4_t b0 = vaddq_u32(sum0.val[0], sum0.val[1]);
    uint32x4_t b1 = vaddq_u32(sum1.val[0], sum1.val[1]);
    uint32x4_t b2 = vaddq_u32(sum2.val[0], sum2.val[1]);
    uint32x4_t b3 = vaddq_u32(sum3.val[0], sum3.val[1]);

    uint32x4_t a0 = vshrq_n_u32(vqaddq_u32(vqaddq_u32(b2, b3), vdupq_n_u32(4)), 3);
    uint32x4_t a1 = vshrq_n_u32(vqaddq_u32(vqaddq_u32(b0, b1), vdupq_n_u32(4)), 3);
    uint32x4_t a2 = vshrq_n_u32(vqaddq_u32(vqaddq_u32(b1, b3), vdupq_n_u32(4)), 3);
    uint32x4_t a3 = vshrq_n_u32(vqaddq_u32(vqaddq_u32(b0, b2), vdupq_n_u32(4)), 3);

    uint16x8_t o0 = vcombine_u16(vqmovun_s32(vreinterpretq_s32_u32( a0 )), vqmovun_s32(vreinterpretq_s32_u32( a1 )));
    uint16x8_t o1 = vcombine_u16(vqmovun_s32(vreinterpretq_s32_u32( a2 )), vqmovun_s32(vreinterpretq_s32_u32( a3 )));

    a[0] = v4i{o0[2], o0[1], o0[0], 0};
    a[1] = v4i{o0[6], o0[5], o0[4], 0};
    a[2] = v4i{o1[2], o1[1], o1[0], 0};
    a[3] = v4i{o1[6], o1[5], o1[4], 0};
#else
    uint32_t r[4];
    uint32_t g[4];
    uint32_t b[4];

    memset(r, 0, sizeof(r));
    memset(g, 0, sizeof(g));
    memset(b, 0, sizeof(b));

    for( int j=0; j<4; j++ )
    {
        for( int i=0; i<4; i++ )
        {
            int index = (j & 2) + (i >> 1);
            b[index] += *data++;
            g[index] += *data++;
            r[index] += *data++;
            data++;
        }
    }

    a[0] = v4i{ uint16_t( (r[2] + r[3] + 4) / 8 ), uint16_t( (g[2] + g[3] + 4) / 8 ), uint16_t( (b[2] + b[3] + 4) / 8 ), 0};
    a[1] = v4i{ uint16_t( (r[0] + r[1] + 4) / 8 ), uint16_t( (g[0] + g[1] + 4) / 8 ), uint16_t( (b[0] + b[1] + 4) / 8 ), 0};
    a[2] = v4i{ uint16_t( (r[1] + r[3] + 4) / 8 ), uint16_t( (g[1] + g[3] + 4) / 8 ), uint16_t( (b[1] + b[3] + 4) / 8 ), 0};
    a[3] = v4i{ uint16_t( (r[0] + r[2] + 4) / 8 ), uint16_t( (g[0] + g[2] + 4) / 8 ), uint16_t( (b[0] + b[2] + 4) / 8 ), 0};
#endif
}

static etcpak_force_inline void CalcErrorBlock( const uint8_t* data, unsigned int err[4][4] )
{
#ifdef __SSE4_1__
    __m128i d0 = _mm_loadu_si128(((__m128i*)data) + 0);
    __m128i d1 = _mm_loadu_si128(((__m128i*)data) + 1);
    __m128i d2 = _mm_loadu_si128(((__m128i*)data) + 2);
    __m128i d3 = _mm_loadu_si128(((__m128i*)data) + 3);

    __m128i dm0 = _mm_and_si128(d0, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm1 = _mm_and_si128(d1, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm2 = _mm_and_si128(d2, _mm_set1_epi32(0x00FFFFFF));
    __m128i dm3 = _mm_and_si128(d3, _mm_set1_epi32(0x00FFFFFF));

    __m128i d0l = _mm_unpacklo_epi8(dm0, _mm_setzero_si128());
    __m128i d0h = _mm_unpackhi_epi8(dm0, _mm_setzero_si128());
    __m128i d1l = _mm_unpacklo_epi8(dm1, _mm_setzero_si128());
    __m128i d1h = _mm_unpackhi_epi8(dm1, _mm_setzero_si128());
    __m128i d2l = _mm_unpacklo_epi8(dm2, _mm_setzero_si128());
    __m128i d2h = _mm_unpackhi_epi8(dm2, _mm_setzero_si128());
    __m128i d3l = _mm_unpacklo_epi8(dm3, _mm_setzero_si128());
    __m128i d3h = _mm_unpackhi_epi8(dm3, _mm_setzero_si128());

    __m128i sum0 = _mm_add_epi16(d0l, d1l);
    __m128i sum1 = _mm_add_epi16(d0h, d1h);
    __m128i sum2 = _mm_add_epi16(d2l, d3l);
    __m128i sum3 = _mm_add_epi16(d2h, d3h);

    __m128i sum0l = _mm_unpacklo_epi16(sum0, _mm_setzero_si128());
    __m128i sum0h = _mm_unpackhi_epi16(sum0, _mm_setzero_si128());
    __m128i sum1l = _mm_unpacklo_epi16(sum1, _mm_setzero_si128());
    __m128i sum1h = _mm_unpackhi_epi16(sum1, _mm_setzero_si128());
    __m128i sum2l = _mm_unpacklo_epi16(sum2, _mm_setzero_si128());
    __m128i sum2h = _mm_unpackhi_epi16(sum2, _mm_setzero_si128());
    __m128i sum3l = _mm_unpacklo_epi16(sum3, _mm_setzero_si128());
    __m128i sum3h = _mm_unpackhi_epi16(sum3, _mm_setzero_si128());

    __m128i b0 = _mm_add_epi32(sum0l, sum0h);
    __m128i b1 = _mm_add_epi32(sum1l, sum1h);
    __m128i b2 = _mm_add_epi32(sum2l, sum2h);
    __m128i b3 = _mm_add_epi32(sum3l, sum3h);

    __m128i a0 = _mm_add_epi32(b2, b3);
    __m128i a1 = _mm_add_epi32(b0, b1);
    __m128i a2 = _mm_add_epi32(b1, b3);
    __m128i a3 = _mm_add_epi32(b0, b2);

    _mm_storeu_si128((__m128i*)&err[0], a0);
    _mm_storeu_si128((__m128i*)&err[1], a1);
    _mm_storeu_si128((__m128i*)&err[2], a2);
    _mm_storeu_si128((__m128i*)&err[3], a3);
#elif defined __ARM_NEON
    uint8x16x2_t t0 = vzipq_u8(vld1q_u8(data +  0), uint8x16_t());
    uint8x16x2_t t1 = vzipq_u8(vld1q_u8(data + 16), uint8x16_t());
    uint8x16x2_t t2 = vzipq_u8(vld1q_u8(data + 32), uint8x16_t());
    uint8x16x2_t t3 = vzipq_u8(vld1q_u8(data + 48), uint8x16_t());

    uint16x8x2_t d0 = { vreinterpretq_u16_u8(t0.val[0]), vreinterpretq_u16_u8(t0.val[1]) };
    uint16x8x2_t d1 = { vreinterpretq_u16_u8(t1.val[0]), vreinterpretq_u16_u8(t1.val[1]) };
    uint16x8x2_t d2 = { vreinterpretq_u16_u8(t2.val[0]), vreinterpretq_u16_u8(t2.val[1]) };
    uint16x8x2_t d3 = { vreinterpretq_u16_u8(t3.val[0]), vreinterpretq_u16_u8(t3.val[1]) };

    uint16x8x2_t s0 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d0.val[0] ), vreinterpretq_s16_u16( d1.val[0] ))), uint16x8_t());
    uint16x8x2_t s1 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d0.val[1] ), vreinterpretq_s16_u16( d1.val[1] ))), uint16x8_t());
    uint16x8x2_t s2 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d2.val[0] ), vreinterpretq_s16_u16( d3.val[0] ))), uint16x8_t());
    uint16x8x2_t s3 = vzipq_u16(vreinterpretq_u16_s16( vaddq_s16(vreinterpretq_s16_u16( d2.val[1] ), vreinterpretq_s16_u16( d3.val[1] ))), uint16x8_t());

    uint32x4x2_t sum0 = { vreinterpretq_u32_u16(s0.val[0]), vreinterpretq_u32_u16(s0.val[1]) };
    uint32x4x2_t sum1 = { vreinterpretq_u32_u16(s1.val[0]), vreinterpretq_u32_u16(s1.val[1]) };
    uint32x4x2_t sum2 = { vreinterpretq_u32_u16(s2.val[0]), vreinterpretq_u32_u16(s2.val[1]) };
    uint32x4x2_t sum3 = { vreinterpretq_u32_u16(s3.val[0]), vreinterpretq_u32_u16(s3.val[1]) };

    uint32x4_t b0 = vaddq_u32(sum0.val[0], sum0.val[1]);
    uint32x4_t b1 = vaddq_u32(sum1.val[0], sum1.val[1]);
    uint32x4_t b2 = vaddq_u32(sum2.val[0], sum2.val[1]);
    uint32x4_t b3 = vaddq_u32(sum3.val[0], sum3.val[1]);

    uint32x4_t a0 = vreinterpretq_u32_u8( vandq_u8(vreinterpretq_u8_u32( vqaddq_u32(b2, b3) ), vreinterpretq_u8_u32( vdupq_n_u32(0x00FFFFFF)) ) );
    uint32x4_t a1 = vreinterpretq_u32_u8( vandq_u8(vreinterpretq_u8_u32( vqaddq_u32(b0, b1) ), vreinterpretq_u8_u32( vdupq_n_u32(0x00FFFFFF)) ) );
    uint32x4_t a2 = vreinterpretq_u32_u8( vandq_u8(vreinterpretq_u8_u32( vqaddq_u32(b1, b3) ), vreinterpretq_u8_u32( vdupq_n_u32(0x00FFFFFF)) ) );
    uint32x4_t a3 = vreinterpretq_u32_u8( vandq_u8(vreinterpretq_u8_u32( vqaddq_u32(b0, b2) ), vreinterpretq_u8_u32( vdupq_n_u32(0x00FFFFFF)) ) );

    vst1q_u32(err[0], a0);
    vst1q_u32(err[1], a1);
    vst1q_u32(err[2], a2);
    vst1q_u32(err[3], a3);
#else
    unsigned int terr[4][4];

    memset(terr, 0, 16 * sizeof(unsigned int));

    for( int j=0; j<4; j++ )
    {
        for( int i=0; i<4; i++ )
        {
            int index = (j & 2) + (i >> 1);
            unsigned int d = *data++;
            terr[index][0] += d;
            d = *data++;
            terr[index][1] += d;
            d = *data++;
            terr[index][2] += d;
            data++;
        }
    }

    for( int i=0; i<3; i++ )
    {
        err[0][i] = terr[2][i] + terr[3][i];
        err[1][i] = terr[0][i] + terr[1][i];
        err[2][i] = terr[1][i] + terr[3][i];
        err[3][i] = terr[0][i] + terr[2][i];
    }
    for( int i=0; i<4; i++ )
    {
        err[i][3] = 0;
    }
#endif
}

static etcpak_force_inline unsigned int CalcError( const unsigned int block[4], const v4i& average )
{
    unsigned int err = 0x3FFFFFFF; // Big value to prevent negative values, but small enough to prevent overflow
    err -= block[0] * 2 * average[2];
    err -= block[1] * 2 * average[1];
    err -= block[2] * 2 * average[0];
    err += 8 * ( sq( average[0] ) + sq( average[1] ) + sq( average[2] ) );
    return err;
}

static etcpak_force_inline void ProcessAverages( v4i* a )
{
#ifdef __SSE4_1__
    for( int i=0; i<2; i++ )
    {
        __m128i d = _mm_loadu_si128((__m128i*)a[i*2].data());

        __m128i t = _mm_add_epi16(_mm_mullo_epi16(d, _mm_set1_epi16(31)), _mm_set1_epi16(128));

        __m128i c = _mm_srli_epi16(_mm_add_epi16(t, _mm_srli_epi16(t, 8)), 8);

        __m128i c1 = _mm_shuffle_epi32(c, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i diff = _mm_sub_epi16(c, c1);
        diff = _mm_max_epi16(diff, _mm_set1_epi16(-4));
        diff = _mm_min_epi16(diff, _mm_set1_epi16(3));

        __m128i co = _mm_add_epi16(c1, diff);

        c = _mm_blend_epi16(co, c, 0xF0);

        __m128i a0 = _mm_or_si128(_mm_slli_epi16(c, 3), _mm_srli_epi16(c, 2));

        _mm_storeu_si128((__m128i*)a[4+i*2].data(), a0);
    }

    for( int i=0; i<2; i++ )
    {
        __m128i d = _mm_loadu_si128((__m128i*)a[i*2].data());

        __m128i t0 = _mm_add_epi16(_mm_mullo_epi16(d, _mm_set1_epi16(15)), _mm_set1_epi16(128));
        __m128i t1 = _mm_srli_epi16(_mm_add_epi16(t0, _mm_srli_epi16(t0, 8)), 8);

        __m128i t2 = _mm_or_si128(t1, _mm_slli_epi16(t1, 4));

        _mm_storeu_si128((__m128i*)a[i*2].data(), t2);
    }
#elif defined __ARM_NEON
    for( int i=0; i<2; i++ )
    {
        int16x8_t d = vld1q_s16((int16_t*)&a[i*2]);
        int16x8_t t = vaddq_s16(vmulq_s16(d, vdupq_n_s16(31)), vdupq_n_s16(128));
        int16x8_t c = vshrq_n_s16(vaddq_s16(t, vshrq_n_s16(t, 8)), 8);

        int16x8_t c1 = vcombine_s16(vget_high_s16(c), vget_high_s16(c));
        int16x8_t diff = vsubq_s16(c, c1);
        diff = vmaxq_s16(diff, vdupq_n_s16(-4));
        diff = vminq_s16(diff, vdupq_n_s16(3));

        int16x8_t co = vaddq_s16(c1, diff);

        c = vcombine_s16(vget_low_s16(co), vget_high_s16(c));

        int16x8_t a0 = vorrq_s16(vshlq_n_s16(c, 3), vshrq_n_s16(c, 2));

        vst1q_s16((int16_t*)&a[4+i*2], a0);
    }

    for( int i=0; i<2; i++ )
    {
        int16x8_t d = vld1q_s16((int16_t*)&a[i*2]);

        int16x8_t t0 = vaddq_s16(vmulq_s16(d, vdupq_n_s16(15)), vdupq_n_s16(128));
        int16x8_t t1 = vshrq_n_s16(vaddq_s16(t0, vshrq_n_s16(t0, 8)), 8);

        int16x8_t t2 = vorrq_s16(t1, vshlq_n_s16(t1, 4));

        vst1q_s16((int16_t*)&a[i*2], t2);
    }
#else
    for( int i=0; i<2; i++ )
    {
        for( int j=0; j<3; j++ )
        {
            int32_t c1 = mul8bit( a[i*2+1][j], 31 );
            int32_t c2 = mul8bit( a[i*2][j], 31 );

            int32_t diff = c2 - c1;
            if( diff > 3 ) diff = 3;
            else if( diff < -4 ) diff = -4;

            int32_t co = c1 + diff;

            a[5+i*2][j] = ( c1 << 3 ) | ( c1 >> 2 );
            a[4+i*2][j] = ( co << 3 ) | ( co >> 2 );
        }
    }

    for( int i=0; i<4; i++ )
    {
        a[i][0] = g_avg2[mul8bit( a[i][0], 15 )];
        a[i][1] = g_avg2[mul8bit( a[i][1], 15 )];
        a[i][2] = g_avg2[mul8bit( a[i][2], 15 )];
    }
#endif
}

static etcpak_force_inline void EncodeAverages( uint64_t& _d, const v4i* a, size_t idx )
{
    auto d = _d;
    d |= ( idx << 24 );
    size_t base = idx << 1;

    if( ( idx & 0x2 ) == 0 )
    {
        for( int i=0; i<3; i++ )
        {
            d |= uint64_t( a[base+0][i] >> 4 ) << ( i*8 );
            d |= uint64_t( a[base+1][i] >> 4 ) << ( i*8 + 4 );
        }
    }
    else
    {
        for( int i=0; i<3; i++ )
        {
            d |= uint64_t( a[base+1][i] & 0xF8 ) << ( i*8 );
            int32_t c = ( ( a[base+0][i] & 0xF8 ) - ( a[base+1][i] & 0xF8 ) ) >> 3;
            c &= ~0xFFFFFFF8;
            d |= ((uint64_t)c) << ( i*8 );
        }
    }
    _d = d;
}

static etcpak_force_inline uint64_t CheckSolid( const uint8_t* src )
{
#ifdef __SSE4_1__
    __m128i d0 = _mm_loadu_si128(((__m128i*)src) + 0);
    __m128i d1 = _mm_loadu_si128(((__m128i*)src) + 1);
    __m128i d2 = _mm_loadu_si128(((__m128i*)src) + 2);
    __m128i d3 = _mm_loadu_si128(((__m128i*)src) + 3);

    __m128i c = _mm_shuffle_epi32(d0, _MM_SHUFFLE(0, 0, 0, 0));

    __m128i c0 = _mm_cmpeq_epi8(d0, c);
    __m128i c1 = _mm_cmpeq_epi8(d1, c);
    __m128i c2 = _mm_cmpeq_epi8(d2, c);
    __m128i c3 = _mm_cmpeq_epi8(d3, c);

    __m128i m0 = _mm_and_si128(c0, c1);
    __m128i m1 = _mm_and_si128(c2, c3);
    __m128i m = _mm_and_si128(m0, m1);

    if (!_mm_testc_si128(m, _mm_set1_epi32(-1)))
    {
        return 0;
    }
#elif defined __ARM_NEON
    int32x4_t d0 = vld1q_s32((int32_t*)src +  0);
    int32x4_t d1 = vld1q_s32((int32_t*)src +  4);
    int32x4_t d2 = vld1q_s32((int32_t*)src +  8);
    int32x4_t d3 = vld1q_s32((int32_t*)src + 12);

    int32x4_t c = vdupq_n_s32(d0[0]);

    int32x4_t c0 = vreinterpretq_s32_u32(vceqq_s32(d0, c));
    int32x4_t c1 = vreinterpretq_s32_u32(vceqq_s32(d1, c));
    int32x4_t c2 = vreinterpretq_s32_u32(vceqq_s32(d2, c));
    int32x4_t c3 = vreinterpretq_s32_u32(vceqq_s32(d3, c));

    int32x4_t m0 = vandq_s32(c0, c1);
    int32x4_t m1 = vandq_s32(c2, c3);
    int64x2_t m = vreinterpretq_s64_s32(vandq_s32(m0, m1));

    if (m[0] != -1 || m[1] != -1)
    {
        return 0;
    }
#else
    const uint8_t* ptr = src + 4;
    for( int i=1; i<16; i++ )
    {
        if( memcmp( src, ptr, 4 ) != 0 )
        {
            return 0;
        }
        ptr += 4;
    }
#endif
    return 0x02000000 |
        ( (unsigned int)( src[0] & 0xF8 ) << 16 ) |
        ( (unsigned int)( src[1] & 0xF8 ) << 8 ) |
        ( (unsigned int)( src[2] & 0xF8 ) );
}

static etcpak_force_inline void PrepareAverages( v4i a[8], const uint8_t* src, unsigned int err[4] )
{
    Average( src, a );
    ProcessAverages( a );

    unsigned int errblock[4][4];
    CalcErrorBlock( src, errblock );

    for( int i=0; i<4; i++ )
    {
        err[i/2] += CalcError( errblock[i], a[i] );
        err[2+i/2] += CalcError( errblock[i], a[i+4] );
    }
}

static etcpak_force_inline void FindBestFit( uint64_t terr[2][8], uint16_t tsel[16][8], v4i a[8], const uint32_t* id, const uint8_t* data )
{
    for( size_t i=0; i<16; i++ )
    {
        uint16_t* sel = tsel[i];
        unsigned int bid = id[i];
        uint64_t* ter = terr[bid%2];

        uint8_t b = *data++;
        uint8_t g = *data++;
        uint8_t r = *data++;
        data++;

        int dr = a[bid][0] - r;
        int dg = a[bid][1] - g;
        int db = a[bid][2] - b;

#ifdef __SSE4_1__
        // Reference implementation

        __m128i pix = _mm_set1_epi32(dr * 77 + dg * 151 + db * 28);
        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        __m128i error0 = _mm_abs_epi32(_mm_add_epi32(pix, g_table256_SIMD[0]));
        __m128i error1 = _mm_abs_epi32(_mm_add_epi32(pix, g_table256_SIMD[1]));
        __m128i error2 = _mm_abs_epi32(_mm_sub_epi32(pix, g_table256_SIMD[0]));
        __m128i error3 = _mm_abs_epi32(_mm_sub_epi32(pix, g_table256_SIMD[1]));

        __m128i index0 = _mm_and_si128(_mm_cmplt_epi32(error1, error0), _mm_set1_epi32(1));
        __m128i minError0 = _mm_min_epi32(error0, error1);

        __m128i index1 = _mm_sub_epi32(_mm_set1_epi32(2), _mm_cmplt_epi32(error3, error2));
        __m128i minError1 = _mm_min_epi32(error2, error3);

        __m128i minIndex0 = _mm_blendv_epi8(index0, index1, _mm_cmplt_epi32(minError1, minError0));
        __m128i minError = _mm_min_epi32(minError0, minError1);

        // Squaring the minimum error to produce correct values when adding
        __m128i minErrorLow = _mm_shuffle_epi32(minError, _MM_SHUFFLE(1, 1, 0, 0));
        __m128i squareErrorLow = _mm_mul_epi32(minErrorLow, minErrorLow);
        squareErrorLow = _mm_add_epi64(squareErrorLow, _mm_loadu_si128(((__m128i*)ter) + 0));
        _mm_storeu_si128(((__m128i*)ter) + 0, squareErrorLow);
        __m128i minErrorHigh = _mm_shuffle_epi32(minError, _MM_SHUFFLE(3, 3, 2, 2));
        __m128i squareErrorHigh = _mm_mul_epi32(minErrorHigh, minErrorHigh);
        squareErrorHigh = _mm_add_epi64(squareErrorHigh, _mm_loadu_si128(((__m128i*)ter) + 1));
        _mm_storeu_si128(((__m128i*)ter) + 1, squareErrorHigh);

        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        error0 = _mm_abs_epi32(_mm_add_epi32(pix, g_table256_SIMD[2]));
        error1 = _mm_abs_epi32(_mm_add_epi32(pix, g_table256_SIMD[3]));
        error2 = _mm_abs_epi32(_mm_sub_epi32(pix, g_table256_SIMD[2]));
        error3 = _mm_abs_epi32(_mm_sub_epi32(pix, g_table256_SIMD[3]));

        index0 = _mm_and_si128(_mm_cmplt_epi32(error1, error0), _mm_set1_epi32(1));
        minError0 = _mm_min_epi32(error0, error1);

        index1 = _mm_sub_epi32(_mm_set1_epi32(2), _mm_cmplt_epi32(error3, error2));
        minError1 = _mm_min_epi32(error2, error3);

        __m128i minIndex1 = _mm_blendv_epi8(index0, index1, _mm_cmplt_epi32(minError1, minError0));
        minError = _mm_min_epi32(minError0, minError1);

        // Squaring the minimum error to produce correct values when adding
        minErrorLow = _mm_shuffle_epi32(minError, _MM_SHUFFLE(1, 1, 0, 0));
        squareErrorLow = _mm_mul_epi32(minErrorLow, minErrorLow);
        squareErrorLow = _mm_add_epi64(squareErrorLow, _mm_loadu_si128(((__m128i*)ter) + 2));
        _mm_storeu_si128(((__m128i*)ter) + 2, squareErrorLow);
        minErrorHigh = _mm_shuffle_epi32(minError, _MM_SHUFFLE(3, 3, 2, 2));
        squareErrorHigh = _mm_mul_epi32(minErrorHigh, minErrorHigh);
        squareErrorHigh = _mm_add_epi64(squareErrorHigh, _mm_loadu_si128(((__m128i*)ter) + 3));
        _mm_storeu_si128(((__m128i*)ter) + 3, squareErrorHigh);
        __m128i minIndex = _mm_packs_epi32(minIndex0, minIndex1);
        _mm_storeu_si128((__m128i*)sel, minIndex);
#elif defined __ARM_NEON
        int32x4_t pix = vdupq_n_s32(dr * 77 + dg * 151 + db * 28);

        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        uint32x4_t error0 = vreinterpretq_u32_s32(vabsq_s32(vaddq_s32(pix, g_table256_NEON[0])));
        uint32x4_t error1 = vreinterpretq_u32_s32(vabsq_s32(vaddq_s32(pix, g_table256_NEON[1])));
        uint32x4_t error2 = vreinterpretq_u32_s32(vabsq_s32(vsubq_s32(pix, g_table256_NEON[0])));
        uint32x4_t error3 = vreinterpretq_u32_s32(vabsq_s32(vsubq_s32(pix, g_table256_NEON[1])));

        uint32x4_t index0 = vandq_u32(vcltq_u32(error1, error0), vdupq_n_u32(1));
        uint32x4_t minError0 = vminq_u32(error0, error1);

        uint32x4_t index1 = vreinterpretq_u32_s32(vsubq_s32(vdupq_n_s32(2), vreinterpretq_s32_u32(vcltq_u32(error3, error2))));
        uint32x4_t minError1 = vminq_u32(error2, error3);

        uint32x4_t blendMask = vcltq_u32(minError1, minError0);
        uint32x4_t minIndex0 = vorrq_u32(vbicq_u32(index0, blendMask), vandq_u32(index1, blendMask));
        uint32x4_t minError = vminq_u32(minError0, minError1);

        // Squaring the minimum error to produce correct values when adding
        uint32x4_t squareErrorLow = vmulq_u32(minError, minError);
        uint32x4_t squareErrorHigh = vshrq_n_u32(vreinterpretq_u32_s32(vqdmulhq_s32(vreinterpretq_s32_u32(minError), vreinterpretq_s32_u32(minError))), 1);
        uint32x4x2_t squareErrorZip = vzipq_u32(squareErrorLow, squareErrorHigh);
        uint64x2x2_t squareError = { vreinterpretq_u64_u32(squareErrorZip.val[0]), vreinterpretq_u64_u32(squareErrorZip.val[1]) };
        squareError.val[0] = vaddq_u64(squareError.val[0], vld1q_u64(ter + 0));
        squareError.val[1] = vaddq_u64(squareError.val[1], vld1q_u64(ter + 2));
        vst1q_u64(ter + 0, squareError.val[0]);
        vst1q_u64(ter + 2, squareError.val[1]);

        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        error0 = vreinterpretq_u32_s32( vabsq_s32(vaddq_s32(pix, g_table256_NEON[2])));
        error1 = vreinterpretq_u32_s32( vabsq_s32(vaddq_s32(pix, g_table256_NEON[3])));
        error2 = vreinterpretq_u32_s32( vabsq_s32(vsubq_s32(pix, g_table256_NEON[2])));
        error3 = vreinterpretq_u32_s32( vabsq_s32(vsubq_s32(pix, g_table256_NEON[3])));

        index0 = vandq_u32(vcltq_u32(error1, error0), vdupq_n_u32(1));
        minError0 = vminq_u32(error0, error1);

        index1 = vreinterpretq_u32_s32( vsubq_s32(vdupq_n_s32(2), vreinterpretq_s32_u32(vcltq_u32(error3, error2))) );
        minError1 = vminq_u32(error2, error3);

        blendMask = vcltq_u32(minError1, minError0);
        uint32x4_t minIndex1 = vorrq_u32(vbicq_u32(index0, blendMask), vandq_u32(index1, blendMask));
        minError = vminq_u32(minError0, minError1);

        // Squaring the minimum error to produce correct values when adding
        squareErrorLow = vmulq_u32(minError, minError);
        squareErrorHigh = vshrq_n_u32(vreinterpretq_u32_s32( vqdmulhq_s32(vreinterpretq_s32_u32(minError), vreinterpretq_s32_u32(minError)) ), 1 );
        squareErrorZip = vzipq_u32(squareErrorLow, squareErrorHigh);
        squareError.val[0] = vaddq_u64(vreinterpretq_u64_u32( squareErrorZip.val[0] ), vld1q_u64(ter + 4));
        squareError.val[1] = vaddq_u64(vreinterpretq_u64_u32( squareErrorZip.val[1] ), vld1q_u64(ter + 6));
        vst1q_u64(ter + 4, squareError.val[0]);
        vst1q_u64(ter + 6, squareError.val[1]);

        uint16x8_t minIndex = vcombine_u16(vqmovn_u32(minIndex0), vqmovn_u32(minIndex1));
        vst1q_u16(sel, minIndex);
#else
        int pix = dr * 77 + dg * 151 + db * 28;

        for( int t=0; t<8; t++ )
        {
            const int64_t* tab = g_table256[t];
            unsigned int idx = 0;
            uint64_t err = sq( tab[0] + pix );
            for( int j=1; j<4; j++ )
            {
                uint64_t local = sq( tab[j] + pix );
                if( local < err )
                {
                    err = local;
                    idx = j;
                }
            }
            *sel++ = idx;
            *ter++ += err;
        }
#endif
    }
}

#if defined __SSE4_1__ || defined __ARM_NEON
// Non-reference implementation, but faster. Produces same results as the AVX2 version
static etcpak_force_inline void FindBestFit( uint32_t terr[2][8], uint16_t tsel[16][8], v4i a[8], const uint32_t* id, const uint8_t* data )
{
    for( size_t i=0; i<16; i++ )
    {
        uint16_t* sel = tsel[i];
        unsigned int bid = id[i];
        uint32_t* ter = terr[bid%2];

        uint8_t b = *data++;
        uint8_t g = *data++;
        uint8_t r = *data++;
        data++;

        int dr = a[bid][0] - r;
        int dg = a[bid][1] - g;
        int db = a[bid][2] - b;

#ifdef __SSE4_1__
        // The scaling values are divided by two and rounded, to allow the differences to be in the range of signed int16
        // This produces slightly different results, but is significant faster
        __m128i pixel = _mm_set1_epi16(dr * 38 + dg * 76 + db * 14);
        __m128i pix = _mm_abs_epi16(pixel);

        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        // Since the selector table is symmetrical, we need to calculate the difference only for half of the entries.
        __m128i error0 = _mm_abs_epi16(_mm_sub_epi16(pix, g_table128_SIMD[0]));
        __m128i error1 = _mm_abs_epi16(_mm_sub_epi16(pix, g_table128_SIMD[1]));

        __m128i index = _mm_and_si128(_mm_cmplt_epi16(error1, error0), _mm_set1_epi16(1));
        __m128i minError = _mm_min_epi16(error0, error1);

        // Exploiting symmetry of the selector table and use the sign bit
        // This produces slightly different results, but is needed to produce same results as AVX2 implementation
        __m128i indexBit = _mm_andnot_si128(_mm_srli_epi16(pixel, 15), _mm_set1_epi8(-1));
        __m128i minIndex = _mm_or_si128(index, _mm_add_epi16(indexBit, indexBit));

        // Squaring the minimum error to produce correct values when adding
        __m128i squareErrorLo = _mm_mullo_epi16(minError, minError);
        __m128i squareErrorHi = _mm_mulhi_epi16(minError, minError);

        __m128i squareErrorLow = _mm_unpacklo_epi16(squareErrorLo, squareErrorHi);
        __m128i squareErrorHigh = _mm_unpackhi_epi16(squareErrorLo, squareErrorHi);

        squareErrorLow = _mm_add_epi32(squareErrorLow, _mm_loadu_si128(((__m128i*)ter) + 0));
        _mm_storeu_si128(((__m128i*)ter) + 0, squareErrorLow);
        squareErrorHigh = _mm_add_epi32(squareErrorHigh, _mm_loadu_si128(((__m128i*)ter) + 1));
        _mm_storeu_si128(((__m128i*)ter) + 1, squareErrorHigh);

        _mm_storeu_si128((__m128i*)sel, minIndex);
#elif defined __ARM_NEON
        int16x8_t pixel = vdupq_n_s16( dr * 38 + dg * 76 + db * 14 );
        int16x8_t pix = vabsq_s16( pixel );

        int16x8_t error0 = vabsq_s16( vsubq_s16( pix, g_table128_NEON[0] ) );
        int16x8_t error1 = vabsq_s16( vsubq_s16( pix, g_table128_NEON[1] ) );

        int16x8_t index = vandq_s16( vreinterpretq_s16_u16( vcltq_s16( error1, error0 ) ), vdupq_n_s16( 1 ) );
        int16x8_t minError = vminq_s16( error0, error1 );

        int16x8_t indexBit = vandq_s16( vmvnq_s16( vshrq_n_s16( pixel, 15 ) ), vdupq_n_s16( -1 ) );
        int16x8_t minIndex = vorrq_s16( index, vaddq_s16( indexBit, indexBit ) );

        int16x4_t minErrorLow = vget_low_s16( minError );
        int16x4_t minErrorHigh = vget_high_s16( minError );

        int32x4_t squareErrorLow = vmull_s16( minErrorLow, minErrorLow );
        int32x4_t squareErrorHigh = vmull_s16( minErrorHigh, minErrorHigh );

        int32x4_t squareErrorSumLow = vaddq_s32( squareErrorLow, vld1q_s32( (int32_t*)ter ) );
        int32x4_t squareErrorSumHigh = vaddq_s32( squareErrorHigh, vld1q_s32( (int32_t*)ter + 4 ) );

        vst1q_s32( (int32_t*)ter, squareErrorSumLow );
        vst1q_s32( (int32_t*)ter + 4, squareErrorSumHigh );

        vst1q_s16( (int16_t*)sel, minIndex );
#endif
    }
}
#endif

static etcpak_force_inline uint8_t convert6(float f)
{
    int i = (std::min(std::max(static_cast<int>(f), 0), 1023) - 15) >> 1;
    return (i + 11 - ((i + 11) >> 7) - ((i + 4) >> 7)) >> 3;
}

static etcpak_force_inline uint8_t convert7(float f)
{
    int i = (std::min(std::max(static_cast<int>(f), 0), 1023) - 15) >> 1;
    return (i + 9 - ((i + 9) >> 8) - ((i + 6) >> 8)) >> 2;
}

static etcpak_force_inline std::pair<uint64_t, uint64_t> Planar(const uint8_t* src)
{
    int32_t r = 0;
    int32_t g = 0;
    int32_t b = 0;

    for (int i = 0; i < 16; ++i)
    {
        b += src[i * 4 + 0];
        g += src[i * 4 + 1];
        r += src[i * 4 + 2];
    }

    int32_t difRyz = 0;
    int32_t difGyz = 0;
    int32_t difByz = 0;
    int32_t difRxz = 0;
    int32_t difGxz = 0;
    int32_t difBxz = 0;

    const int32_t scaling[] = { -255, -85, 85, 255 };

    for (int i = 0; i < 16; ++i)
    {
        int32_t difB = (static_cast<int>(src[i * 4 + 0]) << 4) - b;
        int32_t difG = (static_cast<int>(src[i * 4 + 1]) << 4) - g;
        int32_t difR = (static_cast<int>(src[i * 4 + 2]) << 4) - r;

        difRyz += difR * scaling[i % 4];
        difGyz += difG * scaling[i % 4];
        difByz += difB * scaling[i % 4];

        difRxz += difR * scaling[i / 4];
        difGxz += difG * scaling[i / 4];
        difBxz += difB * scaling[i / 4];
    }

    const float scale = -4.0f / ((255 * 255 * 8.0f + 85 * 85 * 8.0f) * 16.0f);

    float aR = difRxz * scale;
    float aG = difGxz * scale;
    float aB = difBxz * scale;

    float bR = difRyz * scale;
    float bG = difGyz * scale;
    float bB = difByz * scale;

    float dR = r * (4.0f / 16.0f);
    float dG = g * (4.0f / 16.0f);
    float dB = b * (4.0f / 16.0f);

    // calculating the three colors RGBO, RGBH, and RGBV.  RGB = df - af * x - bf * y;
    float cofR = std::fma(aR,  255.0f, std::fma(bR,  255.0f, dR));
    float cofG = std::fma(aG,  255.0f, std::fma(bG,  255.0f, dG));
    float cofB = std::fma(aB,  255.0f, std::fma(bB,  255.0f, dB));
    float chfR = std::fma(aR, -425.0f, std::fma(bR,  255.0f, dR));
    float chfG = std::fma(aG, -425.0f, std::fma(bG,  255.0f, dG));
    float chfB = std::fma(aB, -425.0f, std::fma(bB,  255.0f, dB));
    float cvfR = std::fma(aR,  255.0f, std::fma(bR, -425.0f, dR));
    float cvfG = std::fma(aG,  255.0f, std::fma(bG, -425.0f, dG));
    float cvfB = std::fma(aB,  255.0f, std::fma(bB, -425.0f, dB));

    // convert to r6g7b6
    int32_t coR = convert6(cofR);
    int32_t coG = convert7(cofG);
    int32_t coB = convert6(cofB);
    int32_t chR = convert6(chfR);
    int32_t chG = convert7(chfG);
    int32_t chB = convert6(chfB);
    int32_t cvR = convert6(cvfR);
    int32_t cvG = convert7(cvfG);
    int32_t cvB = convert6(cvfB);

    // Error calculation
    auto ro0 = coR;
    auto go0 = coG;
    auto bo0 = coB;
    auto ro1 = (ro0 >> 4) | (ro0 << 2);
    auto go1 = (go0 >> 6) | (go0 << 1);
    auto bo1 = (bo0 >> 4) | (bo0 << 2);
    auto ro2 = (ro1 << 2) + 2;
    auto go2 = (go1 << 2) + 2;
    auto bo2 = (bo1 << 2) + 2;

    auto rh0 = chR;
    auto gh0 = chG;
    auto bh0 = chB;
    auto rh1 = (rh0 >> 4) | (rh0 << 2);
    auto gh1 = (gh0 >> 6) | (gh0 << 1);
    auto bh1 = (bh0 >> 4) | (bh0 << 2);

    auto rh2 = rh1 - ro1;
    auto gh2 = gh1 - go1;
    auto bh2 = bh1 - bo1;

    auto rv0 = cvR;
    auto gv0 = cvG;
    auto bv0 = cvB;
    auto rv1 = (rv0 >> 4) | (rv0 << 2);
    auto gv1 = (gv0 >> 6) | (gv0 << 1);
    auto bv1 = (bv0 >> 4) | (bv0 << 2);

    auto rv2 = rv1 - ro1;
    auto gv2 = gv1 - go1;
    auto bv2 = bv1 - bo1;

    uint64_t error = 0;

    for (int i = 0; i < 16; ++i)
    {
        int32_t cR = clampu8((rh2 * (i / 4) + rv2 * (i % 4) + ro2) >> 2);
        int32_t cG = clampu8((gh2 * (i / 4) + gv2 * (i % 4) + go2) >> 2);
        int32_t cB = clampu8((bh2 * (i / 4) + bv2 * (i % 4) + bo2) >> 2);

        int32_t difB = static_cast<int>(src[i * 4 + 0]) - cB;
        int32_t difG = static_cast<int>(src[i * 4 + 1]) - cG;
        int32_t difR = static_cast<int>(src[i * 4 + 2]) - cR;

        int32_t dif = difR * 38 + difG * 76 + difB * 14;

        error += dif * dif;
    }

    /**/
    uint32_t rgbv = cvB | (cvG << 6) | (cvR << 13);
    uint32_t rgbh = chB | (chG << 6) | (chR << 13);
    uint32_t hi = rgbv | ((rgbh & 0x1FFF) << 19);
    uint32_t lo = (chR & 0x1) | 0x2 | ((chR << 1) & 0x7C);
    lo |= ((coB & 0x07) <<  7) | ((coB & 0x18) <<  8) | ((coB & 0x20) << 11);
    lo |= ((coG & 0x3F) << 17) | ((coG & 0x40) << 18);
    lo |= coR << 25;

    const auto idx = (coR & 0x20) | ((coG & 0x20) >> 1) | ((coB & 0x1E) >> 1);

    lo |= g_flags[idx];

    uint64_t result = static_cast<uint32_t>(_bswap(lo));
    result |= static_cast<uint64_t>(static_cast<uint32_t>(_bswap(hi))) << 32;

    return std::make_pair(result, error);
}

#ifdef __ARM_NEON

static etcpak_force_inline int32x2_t Planar_NEON_DifXZ( int16x8_t dif_lo, int16x8_t dif_hi )
{
    int32x4_t dif0 = vmull_n_s16( vget_low_s16( dif_lo ), -255 );
    int32x4_t dif1 = vmull_n_s16( vget_high_s16( dif_lo ), -85 );
    int32x4_t dif2 = vmull_n_s16( vget_low_s16( dif_hi ), 85 );
    int32x4_t dif3 = vmull_n_s16( vget_high_s16( dif_hi ), 255 );
    int32x4_t dif4 = vaddq_s32( vaddq_s32( dif0, dif1 ), vaddq_s32( dif2, dif3 ) );

#ifndef __aarch64__
    int32x2_t dif5 = vpadd_s32( vget_low_s32( dif4 ), vget_high_s32( dif4 ) );
    return vpadd_s32( dif5, dif5 );
#else
    return vdup_n_s32( vaddvq_s32( dif4 ) );
#endif
}

static etcpak_force_inline int32x2_t Planar_NEON_DifYZ( int16x8_t dif_lo, int16x8_t dif_hi )
{
    int16x4_t scaling = { -255, -85, 85, 255 };
    int32x4_t dif0 = vmull_s16( vget_low_s16( dif_lo ), scaling );
    int32x4_t dif1 = vmull_s16( vget_high_s16( dif_lo ), scaling );
    int32x4_t dif2 = vmull_s16( vget_low_s16( dif_hi ), scaling );
    int32x4_t dif3 = vmull_s16( vget_high_s16( dif_hi ), scaling );
    int32x4_t dif4 = vaddq_s32( vaddq_s32( dif0, dif1 ), vaddq_s32( dif2, dif3 ) );

#ifndef __aarch64__
    int32x2_t dif5 = vpadd_s32( vget_low_s32( dif4 ), vget_high_s32( dif4 ) );
    return vpadd_s32( dif5, dif5 );
#else
    return vdup_n_s32( vaddvq_s32( dif4 ) );
#endif
}

static etcpak_force_inline int16x8_t Planar_NEON_SumWide( uint8x16_t src )
{
    uint16x8_t accu8 = vpaddlq_u8( src );
#ifndef __aarch64__
    uint16x4_t accu4 = vpadd_u16( vget_low_u16( accu8 ), vget_high_u16( accu8 ) );
    uint16x4_t accu2 = vpadd_u16( accu4, accu4 );
    uint16x4_t accu1 = vpadd_u16( accu2, accu2 );
    return vreinterpretq_s16_u16( vcombine_u16( accu1, accu1 ) );
#else 
    return vdupq_n_s16( vaddvq_u16( accu8 ) );
#endif
}

static etcpak_force_inline int16x8_t convert6_NEON( int32x4_t lo, int32x4_t hi )
{
    uint16x8_t x = vcombine_u16( vqmovun_s32( lo ), vqmovun_s32( hi ) );
    int16x8_t i = vreinterpretq_s16_u16( vshrq_n_u16( vqshlq_n_u16( x, 6 ), 6) ); // clamp 0-1023
    i = vhsubq_s16( i, vdupq_n_s16( 15 ) );

    int16x8_t ip11 = vaddq_s16( i, vdupq_n_s16( 11 ) );
    int16x8_t ip4 = vaddq_s16( i, vdupq_n_s16( 4 ) );

    return vshrq_n_s16( vsubq_s16( vsubq_s16( ip11, vshrq_n_s16( ip11, 7 ) ), vshrq_n_s16( ip4, 7) ), 3 );
}

static etcpak_force_inline int16x4_t convert7_NEON( int32x4_t x )
{
    int16x4_t i = vreinterpret_s16_u16( vshr_n_u16( vqshl_n_u16( vqmovun_s32( x ), 6 ), 6 ) ); // clamp 0-1023
    i = vhsub_s16( i, vdup_n_s16( 15 ) );

    int16x4_t p9 = vadd_s16( i, vdup_n_s16( 9 ) );
    int16x4_t p6 = vadd_s16( i, vdup_n_s16( 6 ) );
    return vshr_n_s16( vsub_s16( vsub_s16( p9, vshr_n_s16( p9, 8 ) ), vshr_n_s16( p6, 8 ) ), 2 );
}

static etcpak_force_inline std::pair<uint64_t, uint64_t> Planar_NEON( const uint8_t* src )
{
    uint8x16x4_t srcBlock = vld4q_u8( src );

    int16x8_t bSumWide = Planar_NEON_SumWide( srcBlock.val[0] );
    int16x8_t gSumWide = Planar_NEON_SumWide( srcBlock.val[1] );
    int16x8_t rSumWide = Planar_NEON_SumWide( srcBlock.val[2] );

    int16x8_t dif_R_lo = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_low_u8( srcBlock.val[2] ), 4) ), rSumWide );
    int16x8_t dif_R_hi = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_high_u8( srcBlock.val[2] ), 4) ), rSumWide );

    int16x8_t dif_G_lo = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_low_u8( srcBlock.val[1] ), 4 ) ), gSumWide );
    int16x8_t dif_G_hi = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_high_u8( srcBlock.val[1] ), 4 ) ), gSumWide );

    int16x8_t dif_B_lo = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_low_u8( srcBlock.val[0] ), 4) ), bSumWide );
    int16x8_t dif_B_hi = vsubq_s16( vreinterpretq_s16_u16( vshll_n_u8( vget_high_u8( srcBlock.val[0] ), 4) ), bSumWide );

    int32x2x2_t dif_xz_z = vzip_s32( vzip_s32( Planar_NEON_DifXZ( dif_B_lo, dif_B_hi ), Planar_NEON_DifXZ( dif_R_lo, dif_R_hi ) ).val[0], Planar_NEON_DifXZ( dif_G_lo, dif_G_hi ) );
    int32x4_t dif_xz = vcombine_s32( dif_xz_z.val[0], dif_xz_z.val[1] );
    int32x2x2_t dif_yz_z = vzip_s32( vzip_s32( Planar_NEON_DifYZ( dif_B_lo, dif_B_hi ), Planar_NEON_DifYZ( dif_R_lo, dif_R_hi ) ).val[0], Planar_NEON_DifYZ( dif_G_lo, dif_G_hi ) );
    int32x4_t dif_yz = vcombine_s32( dif_yz_z.val[0], dif_yz_z.val[1] );

    const float fscale = -4.0f / ( (255 * 255 * 8.0f + 85 * 85 * 8.0f ) * 16.0f );
    float32x4_t fa = vmulq_n_f32( vcvtq_f32_s32( dif_xz ), fscale );
    float32x4_t fb = vmulq_n_f32( vcvtq_f32_s32( dif_yz ), fscale );
    int16x4_t bgrgSum = vzip_s16( vzip_s16( vget_low_s16( bSumWide ), vget_low_s16( rSumWide ) ).val[0], vget_low_s16( gSumWide ) ).val[0];
    float32x4_t fd = vmulq_n_f32( vcvtq_f32_s32( vmovl_s16( bgrgSum ) ), 4.0f / 16.0f);

    float32x4_t cof = vmlaq_n_f32( vmlaq_n_f32( fd, fb, 255.0f ), fa, 255.0f );
    float32x4_t chf = vmlaq_n_f32( vmlaq_n_f32( fd, fb, 255.0f ), fa, -425.0f );
    float32x4_t cvf = vmlaq_n_f32( vmlaq_n_f32( fd, fb, -425.0f ), fa, 255.0f );

    int32x4_t coi = vcvtq_s32_f32( cof );
    int32x4_t chi = vcvtq_s32_f32( chf );
    int32x4_t cvi = vcvtq_s32_f32( cvf );

    int32x4x2_t tr_hv = vtrnq_s32( chi, cvi );
    int32x4x2_t tr_o = vtrnq_s32( coi, coi );

    int16x8_t c_hvoo_br_6 = convert6_NEON( tr_hv.val[0], tr_o.val[0] );
    int16x4_t c_hvox_g_7 = convert7_NEON( vcombine_s32( vget_low_s32( tr_hv.val[1] ), vget_low_s32( tr_o.val[1] ) ) );
    int16x8_t c_hvoo_br_8 = vorrq_s16( vshrq_n_s16( c_hvoo_br_6, 4 ), vshlq_n_s16( c_hvoo_br_6, 2 ) );
    int16x4_t c_hvox_g_8 = vorr_s16( vshr_n_s16( c_hvox_g_7, 6 ), vshl_n_s16( c_hvox_g_7, 1 ) );

    int16x4_t rec_gxbr_o = vext_s16( c_hvox_g_8, vget_high_s16( c_hvoo_br_8 ), 3 );

    rec_gxbr_o = vadd_s16( vshl_n_s16( rec_gxbr_o, 2 ), vdup_n_s16( 2 ) );
    int16x8_t rec_ro_wide = vdupq_lane_s16( rec_gxbr_o, 3 );
    int16x8_t rec_go_wide = vdupq_lane_s16( rec_gxbr_o, 0 );
    int16x8_t rec_bo_wide = vdupq_lane_s16( rec_gxbr_o, 1 );

    int16x4_t br_hv2 = vsub_s16( vget_low_s16( c_hvoo_br_8 ), vget_high_s16( c_hvoo_br_8 ) );
    int16x4_t gg_hv2 = vsub_s16( c_hvox_g_8, vdup_lane_s16( c_hvox_g_8, 2 ) );

    int16x8_t scaleh_lo = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int16x8_t scaleh_hi = { 2, 2, 2, 2, 3, 3, 3, 3 };
    int16x8_t scalev = { 0, 1, 2, 3, 0, 1, 2, 3 };

    int16x8_t rec_r_1 = vmlaq_lane_s16( rec_ro_wide, scalev, br_hv2, 3 );
    int16x8_t rec_r_lo = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_r_1, scaleh_lo, br_hv2, 2 ), 2 ) ) );
    int16x8_t rec_r_hi = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_r_1, scaleh_hi, br_hv2, 2 ), 2 ) ) );

    int16x8_t rec_b_1 = vmlaq_lane_s16( rec_bo_wide, scalev, br_hv2, 1 );
    int16x8_t rec_b_lo = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_b_1, scaleh_lo, br_hv2, 0 ), 2 ) ) );
    int16x8_t rec_b_hi = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_b_1, scaleh_hi, br_hv2, 0 ), 2 ) ) );

    int16x8_t rec_g_1 = vmlaq_lane_s16( rec_go_wide, scalev, gg_hv2, 1 );
    int16x8_t rec_g_lo = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_g_1, scaleh_lo, gg_hv2, 0 ), 2 ) ) );
    int16x8_t rec_g_hi = vreinterpretq_s16_u16( vmovl_u8( vqshrun_n_s16( vmlaq_lane_s16( rec_g_1, scaleh_hi, gg_hv2, 0 ), 2 ) ) );

    int16x8_t dif_r_lo = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( srcBlock.val[2] ) ) ), rec_r_lo );
    int16x8_t dif_r_hi = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( srcBlock.val[2] ) ) ), rec_r_hi );

    int16x8_t dif_g_lo = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( srcBlock.val[1] ) ) ), rec_g_lo );
    int16x8_t dif_g_hi = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( srcBlock.val[1] ) ) ), rec_g_hi );

    int16x8_t dif_b_lo = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_low_u8( srcBlock.val[0] ) ) ), rec_b_lo );
    int16x8_t dif_b_hi = vsubq_s16( vreinterpretq_s16_u16( vmovl_u8( vget_high_u8( srcBlock.val[0] ) ) ), rec_b_hi );

    int16x8_t dif_lo = vmlaq_n_s16( vmlaq_n_s16( vmulq_n_s16( dif_r_lo, 38 ), dif_g_lo, 76 ), dif_b_lo, 14 );
    int16x8_t dif_hi = vmlaq_n_s16( vmlaq_n_s16( vmulq_n_s16( dif_r_hi, 38 ), dif_g_hi, 76 ), dif_b_hi, 14 );

    int16x4_t tmpDif = vget_low_s16( dif_lo );
    int32x4_t difsq_0 = vmull_s16( tmpDif, tmpDif );
    tmpDif = vget_high_s16( dif_lo );
    int32x4_t difsq_1 = vmull_s16( tmpDif, tmpDif );
    tmpDif = vget_low_s16( dif_hi );
    int32x4_t difsq_2 = vmull_s16( tmpDif, tmpDif );
    tmpDif = vget_high_s16( dif_hi );
    int32x4_t difsq_3 = vmull_s16( tmpDif, tmpDif );

    uint32x4_t difsq_5 = vaddq_u32( vreinterpretq_u32_s32( difsq_0 ), vreinterpretq_u32_s32( difsq_1 ) );
    uint32x4_t difsq_6 = vaddq_u32( vreinterpretq_u32_s32( difsq_2 ), vreinterpretq_u32_s32( difsq_3) );

    uint64x2_t difsq_7 = vaddl_u32( vget_low_u32( difsq_5 ), vget_high_u32( difsq_5 ) );
    uint64x2_t difsq_8 = vaddl_u32( vget_low_u32( difsq_6 ), vget_high_u32( difsq_6 ) );

    uint64x2_t difsq_9 = vaddq_u64( difsq_7, difsq_8 );

#ifdef __aarch64__
    uint64_t error = vaddvq_u64( difsq_9 );
#else
    uint64_t error = vgetq_lane_u64( difsq_9, 0 ) + vgetq_lane_u64( difsq_9, 1 );
#endif

    int32_t coR = c_hvoo_br_6[6];
    int32_t coG = c_hvox_g_7[2];
    int32_t coB = c_hvoo_br_6[4];

    int32_t chR = c_hvoo_br_6[2];
    int32_t chG = c_hvox_g_7[0];
    int32_t chB = c_hvoo_br_6[0];

    int32_t cvR = c_hvoo_br_6[3];
    int32_t cvG = c_hvox_g_7[1];
    int32_t cvB = c_hvoo_br_6[1];

    uint32_t rgbv = cvB | ( cvG << 6 ) | ( cvR << 13 );
    uint32_t rgbh = chB | ( chG << 6 ) | ( chR << 13 );
    uint32_t hi = rgbv | ( ( rgbh & 0x1FFF ) << 19 );
    uint32_t lo = ( chR & 0x1 ) | 0x2 | ( ( chR << 1 ) & 0x7C );
    lo |= ( ( coB & 0x07 ) << 7 ) | ( ( coB & 0x18 ) << 8 ) | ( ( coB & 0x20 ) << 11 );
    lo |= ( ( coG & 0x3F) << 17) | ( (coG & 0x40 ) << 18 );
    lo |= coR << 25;

    const auto idx = ( coR & 0x20 ) | ( ( coG & 0x20 ) >> 1 ) | ( ( coB & 0x1E ) >> 1 );

    lo |= g_flags[idx];

    uint64_t result = static_cast<uint32_t>( _bswap(lo) );
    result |= static_cast<uint64_t>( static_cast<uint32_t>( _bswap( hi ) ) ) << 32;

    return std::make_pair( result, error );
}

#endif

template<class T, class S>
static etcpak_force_inline uint64_t EncodeSelectors( uint64_t d, const T terr[2][8], const S tsel[16][8], const uint32_t* id, const uint64_t value, const uint64_t error)
{
    size_t tidx[2];
    tidx[0] = GetLeastError( terr[0], 8 );
    tidx[1] = GetLeastError( terr[1], 8 );

    if ((terr[0][tidx[0]] + terr[1][tidx[1]]) >= error)
    {
        return value;
    }

    d |= tidx[0] << 26;
    d |= tidx[1] << 29;
    for( int i=0; i<16; i++ )
    {
        uint64_t t = tsel[i][tidx[id[i]%2]];
        d |= ( t & 0x1 ) << ( i + 32 );
        d |= ( t & 0x2 ) << ( i + 47 );
    }

    return FixByteOrder(d);
}

}

static etcpak_force_inline uint64_t ProcessRGB( const uint8_t* src )
{
#ifdef __AVX2__
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
#else
    uint64_t d = CheckSolid( src );
    if( d != 0 ) return d;

    v4i a[8];
    unsigned int err[4] = {};
    PrepareAverages( a, src, err );
    size_t idx = GetLeastError( err, 4 );
    EncodeAverages( d, a, idx );

#if ( defined __SSE4_1__ || defined __ARM_NEON ) && !defined REFERENCE_IMPLEMENTATION
    uint32_t terr[2][8] = {};
#else
    uint64_t terr[2][8] = {};
#endif
    uint16_t tsel[16][8];
    auto id = g_id[idx];
    FindBestFit( terr, tsel, a, id, src );

    return FixByteOrder( EncodeSelectors( d, terr, tsel, id ) );
#endif
}

static etcpak_force_inline uint64_t ProcessRGB_ETC2( const uint8_t* src )
{
#ifdef __AVX2__
    uint64_t d = CheckSolid_AVX2( src );
    if( d != 0 ) return d;

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

    d = EncodeAverages_AVX2( a, idx );

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
#else
    uint64_t d = CheckSolid( src );
    if (d != 0) return d;

#ifdef __ARM_NEON
    auto result = Planar_NEON( src );
#else
    auto result = Planar( src );
#endif

    v4i a[8];
    unsigned int err[4] = {};
    PrepareAverages( a, src, err );
    size_t idx = GetLeastError( err, 4 );
    EncodeAverages( d, a, idx );

#if ( defined __SSE4_1__ || defined __ARM_NEON ) && !defined REFERENCE_IMPLEMENTATION
    uint32_t terr[2][8] = {};
#else
    uint64_t terr[2][8] = {};
#endif
    uint16_t tsel[16][8];
    auto id = g_id[idx];
    FindBestFit( terr, tsel, a, id, src );

    return EncodeSelectors( d, terr, tsel, id, result.first, result.second );
#endif
}

#ifdef __SSE4_1__
template<int K>
static etcpak_force_inline __m128i Widen( const __m128i src )
{
    static_assert( K >= 0 && K <= 7, "Index out of range" );

    __m128i tmp;
    switch( K )
    {
    case 0:
        tmp = _mm_shufflelo_epi16( src, _MM_SHUFFLE( 0, 0, 0, 0 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    case 1:
        tmp = _mm_shufflelo_epi16( src, _MM_SHUFFLE( 1, 1, 1, 1 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    case 2:
        tmp = _mm_shufflelo_epi16( src, _MM_SHUFFLE( 2, 2, 2, 2 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    case 3:
        tmp = _mm_shufflelo_epi16( src, _MM_SHUFFLE( 3, 3, 3, 3 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    case 4:
        tmp = _mm_shufflehi_epi16( src, _MM_SHUFFLE( 0, 0, 0, 0 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    case 5:
        tmp = _mm_shufflehi_epi16( src, _MM_SHUFFLE( 1, 1, 1, 1 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    case 6:
        tmp = _mm_shufflehi_epi16( src, _MM_SHUFFLE( 2, 2, 2, 2 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    case 7:
        tmp = _mm_shufflehi_epi16( src, _MM_SHUFFLE( 3, 3, 3, 3 ) );
        return _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    }
}

static etcpak_force_inline int GetMulSel( int sel )
{
    switch( sel )
    {
    case 0:
        return 0;
    case 1:
    case 2:
    case 3:
        return 1;
    case 4:
        return 2;
    case 5:
    case 6:
    case 7:
        return 3;
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
        return 4;
    case 14:
    case 15:
        return 5;
    }
}

#endif

#ifdef __ARM_NEON

static constexpr etcpak_force_inline int GetMulSel(int sel)
{
    return ( sel < 1 ) ? 0 : ( sel < 4 ) ? 1 : ( sel < 5 ) ? 2 : ( sel < 8 ) ? 3 : ( sel < 14 ) ? 4 : 5;
}

static constexpr int ClampConstant( int x, int min, int max )
{
    return x < min ? min : x > max ? max : x;
}

template <int Index>
etcpak_force_inline static uint16x8_t ErrorProbe_EAC_NEON( uint8x8_t recVal, uint8x16_t alphaBlock )
{
    uint8x8_t srcValWide;
#ifndef __aarch64__
    if( Index < 8 )
        srcValWide = vdup_lane_u8( vget_low_u8( alphaBlock ), ClampConstant( Index, 0, 8 ) );
    else
        srcValWide = vdup_lane_u8( vget_high_u8( alphaBlock ), ClampConstant( Index - 8, 0, 8 ) );
#else
    srcValWide = vdup_laneq_u8( alphaBlock, Index );
#endif

    uint8x8_t deltaVal = vabd_u8( srcValWide, recVal );
    return vmull_u8( deltaVal, deltaVal );
}

etcpak_force_inline static uint16_t MinError_EAC_NEON( uint16x8_t errProbe )
{
#ifndef __aarch64__
    uint16x4_t tmpErr = vpmin_u16( vget_low_u16( errProbe ), vget_high_u16( errProbe ) );
    tmpErr = vpmin_u16( tmpErr, tmpErr );
    return vpmin_u16( tmpErr, tmpErr )[0];
#else
    return vminvq_u16( errProbe );
#endif
}

template <int Index>
etcpak_force_inline static uint64_t MinErrorIndex_EAC_NEON( uint8x8_t recVal, uint8x16_t alphaBlock )
{
    uint16x8_t errProbe = ErrorProbe_EAC_NEON<Index>( recVal, alphaBlock );
    uint16x8_t minErrMask = vceqq_u16( errProbe, vdupq_n_u16( MinError_EAC_NEON( errProbe ) ) );
    uint64_t idx = __builtin_ctzll( vget_lane_u64( vreinterpret_u64_u8( vqmovn_u16( minErrMask ) ), 0 ) );
    idx >>= 3;
    idx <<= 45 - Index * 3;

    return idx;
}

template <int Index>
etcpak_force_inline static int16x8_t WidenMultiplier_EAC_NEON( int16x8_t multipliers )
{
    constexpr int Lane = GetMulSel( Index );
#ifndef __aarch64__
    if( Lane < 4 )
        return vdupq_lane_s16( vget_low_s16( multipliers ), ClampConstant( Lane, 0, 4 ) );
    else
        return vdupq_lane_s16( vget_high_s16( multipliers ), ClampConstant( Lane - 4, 0, 4 ) );
#else
    return vdupq_laneq_s16( multipliers, Lane );
#endif
}

#endif

static etcpak_force_inline uint64_t ProcessAlpha_ETC2( const uint8_t* src )
{
#if defined __SSE4_1__
    // Check solid
    __m128i s = _mm_loadu_si128( (__m128i*)src );
    __m128i solidCmp = _mm_set1_epi8( src[0] );
    __m128i cmpRes = _mm_cmpeq_epi8( s, solidCmp );
    if( _mm_testc_si128( cmpRes, _mm_set1_epi32( -1 ) ) )
    {
        return src[0];
    }

    // Calculate min, max
    __m128i s1 = _mm_shuffle_epi32( s, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    __m128i max1 = _mm_max_epu8( s, s1 );
    __m128i min1 = _mm_min_epu8( s, s1 );
    __m128i smax2 = _mm_shuffle_epi32( max1, _MM_SHUFFLE( 0, 0, 2, 2 ) );
    __m128i smin2 = _mm_shuffle_epi32( min1, _MM_SHUFFLE( 0, 0, 2, 2 ) );
    __m128i max2 = _mm_max_epu8( max1, smax2 );
    __m128i min2 = _mm_min_epu8( min1, smin2 );
    __m128i smax3 = _mm_alignr_epi8( max2, max2, 2 );
    __m128i smin3 = _mm_alignr_epi8( min2, min2, 2 );
    __m128i max3 = _mm_max_epu8( max2, smax3 );
    __m128i min3 = _mm_min_epu8( min2, smin3 );
    __m128i smax4 = _mm_alignr_epi8( max3, max3, 1 );
    __m128i smin4 = _mm_alignr_epi8( min3, min3, 1 );
    __m128i max = _mm_max_epu8( max3, smax4 );
    __m128i min = _mm_min_epu8( min3, smin4 );
    __m128i max16 = _mm_unpacklo_epi8( max, _mm_setzero_si128() );
    __m128i min16 = _mm_unpacklo_epi8( min, _mm_setzero_si128() );

    // src range, mid
    __m128i srcRange = _mm_sub_epi16( max16, min16 );
    __m128i srcRangeHalf = _mm_srli_epi16( srcRange, 1 );
    __m128i srcMid = _mm_add_epi16( min16, srcRangeHalf );

    // multiplier
    __m128i mul1 = _mm_mulhi_epi16( srcRange, g_alphaRange_SIMD );
    __m128i mul = _mm_add_epi16( mul1, _mm_set1_epi16( 1 ) );

    // wide source
    __m128i s16_1 = _mm_shuffle_epi32( s, _MM_SHUFFLE( 3, 2, 3, 2 ) );
    __m128i s16[2] = { _mm_unpacklo_epi8( s, _mm_setzero_si128() ), _mm_unpacklo_epi8( s16_1, _mm_setzero_si128() ) };

    __m128i sr[16] = {
        Widen<0>( s16[0] ),
        Widen<1>( s16[0] ),
        Widen<2>( s16[0] ),
        Widen<3>( s16[0] ),
        Widen<4>( s16[0] ),
        Widen<5>( s16[0] ),
        Widen<6>( s16[0] ),
        Widen<7>( s16[0] ),
        Widen<0>( s16[1] ),
        Widen<1>( s16[1] ),
        Widen<2>( s16[1] ),
        Widen<3>( s16[1] ),
        Widen<4>( s16[1] ),
        Widen<5>( s16[1] ),
        Widen<6>( s16[1] ),
        Widen<7>( s16[1] )
    };

#ifdef __AVX2__
    __m256i srcRangeWide = _mm256_broadcastsi128_si256( srcRange );
    __m256i srcMidWide = _mm256_broadcastsi128_si256( srcMid );

    __m256i mulWide1 = _mm256_mulhi_epi16( srcRangeWide, g_alphaRange_AVX );
    __m256i mulWide = _mm256_add_epi16( mulWide1, _mm256_set1_epi16( 1 ) );

    __m256i modMul[8] = {
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[0] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[0] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[1] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[1] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[2] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[2] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[3] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[3] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[4] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[4] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[5] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[5] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[6] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[6] ) ) ), _mm256_setzero_si256() ),
        _mm256_unpacklo_epi8( _mm256_packus_epi16( _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[7] ) ), _mm256_add_epi16( srcMidWide, _mm256_mullo_epi16( mulWide, g_alpha_AVX[7] ) ) ), _mm256_setzero_si256() ),
    };

    // find selector
    __m256i mulErr = _mm256_setzero_si256();
    for( int j=0; j<16; j++ )
    {
        __m256i s16Wide = _mm256_broadcastsi128_si256( sr[j] );
        __m256i err1, err2;

        err1 = _mm256_sub_epi16( s16Wide, modMul[0] );
        __m256i localErr = _mm256_mullo_epi16( err1, err1 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[1] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[2] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[3] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[4] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[5] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[6] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        err1 = _mm256_sub_epi16( s16Wide, modMul[7] );
        err2 = _mm256_mullo_epi16( err1, err1 );
        localErr = _mm256_min_epu16( localErr, err2 );

        // note that this can overflow, but since we're looking for the smallest error, it shouldn't matter
        mulErr = _mm256_adds_epu16( mulErr, localErr );
    }
    uint64_t minPos1 = _mm_cvtsi128_si64( _mm_minpos_epu16( _mm256_castsi256_si128( mulErr ) ) );
    uint64_t minPos2 = _mm_cvtsi128_si64( _mm_minpos_epu16( _mm256_extracti128_si256( mulErr, 1 ) ) );
    int sel = ( ( minPos1 & 0xFFFF ) < ( minPos2 & 0xFFFF ) ) ? ( minPos1 >> 16 ) : ( 8 + ( minPos2 >> 16 ) );

    __m128i recVal16;
    switch( sel )
    {
    case 0:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<0>( mul ), g_alpha_SIMD[0] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<0>( mul ), g_alpha_SIMD[0] ) ) ), _mm_setzero_si128() );
        break;
    case 1:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[1] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[1] ) ) ), _mm_setzero_si128() );
        break;
    case 2:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[2] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[2] ) ) ), _mm_setzero_si128() );
        break;
    case 3:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[3] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[3] ) ) ), _mm_setzero_si128() );
        break;
    case 4:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<2>( mul ), g_alpha_SIMD[4] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<2>( mul ), g_alpha_SIMD[4] ) ) ), _mm_setzero_si128() );
        break;
    case 5:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[5] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[5] ) ) ), _mm_setzero_si128() );
        break;
    case 6:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[6] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[6] ) ) ), _mm_setzero_si128() );
        break;
    case 7:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[7] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[7] ) ) ), _mm_setzero_si128() );
        break;
    case 8:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[8] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[8] ) ) ), _mm_setzero_si128() );
        break;
    case 9:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[9] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[9] ) ) ), _mm_setzero_si128() );
        break;
    case 10:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[10] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[10] ) ) ), _mm_setzero_si128() );
        break;
    case 11:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[11] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[11] ) ) ), _mm_setzero_si128() );
        break;
    case 12:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[12] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[12] ) ) ), _mm_setzero_si128() );
        break;
    case 13:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[13] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[13] ) ) ), _mm_setzero_si128() );
        break;
    case 14:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[14] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[14] ) ) ), _mm_setzero_si128() );
        break;
    case 15:
        recVal16 = _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[15] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[15] ) ) ), _mm_setzero_si128() );
        break;
    default:
        assert( false );
        break;
    }
#else
    // wide multiplier
    __m128i rangeMul[16] = {
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<0>( mul ), g_alpha_SIMD[0] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<0>( mul ), g_alpha_SIMD[0] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[1] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[1] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[2] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[2] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[3] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<1>( mul ), g_alpha_SIMD[3] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<2>( mul ), g_alpha_SIMD[4] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<2>( mul ), g_alpha_SIMD[4] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[5] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[5] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[6] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[6] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[7] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<3>( mul ), g_alpha_SIMD[7] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[8] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[8] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[9] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[9] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[10] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[10] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[11] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[11] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[12] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[12] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[13] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<4>( mul ), g_alpha_SIMD[13] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[14] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[14] ) ) ), _mm_setzero_si128() ),
        _mm_unpacklo_epi8( _mm_packus_epi16( _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[15] ) ), _mm_add_epi16( srcMid, _mm_mullo_epi16( Widen<5>( mul ), g_alpha_SIMD[15] ) ) ), _mm_setzero_si128() )
    };

    // find selector
    int err = std::numeric_limits<int>::max();
    int sel;
    for( int r=0; r<16; r++ )
    {
        __m128i err1, err2, minerr;
        __m128i recVal16 = rangeMul[r];
        int rangeErr;

        err1 = _mm_sub_epi16( sr[0], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr = _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[1], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[2], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[3], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[4], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[5], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[6], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[7], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[8], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[9], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[10], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[11], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[12], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[13], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[14], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        err1 = _mm_sub_epi16( sr[15], recVal16 );
        err2 = _mm_mullo_epi16( err1, err1 );
        minerr = _mm_minpos_epu16( err2 );
        rangeErr += _mm_cvtsi128_si64( minerr ) & 0xFFFF;

        if( rangeErr < err )
        {
            err = rangeErr;
            sel = r;
            if( err == 0 ) break;
        }
    }

    __m128i recVal16 = rangeMul[sel];
#endif

    // find indices
    __m128i err1, err2, minerr;
    uint64_t idx = 0, tmp;

    err1 = _mm_sub_epi16( sr[0], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 15*3;

    err1 = _mm_sub_epi16( sr[1], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 14*3;

    err1 = _mm_sub_epi16( sr[2], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 13*3;

    err1 = _mm_sub_epi16( sr[3], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 12*3;

    err1 = _mm_sub_epi16( sr[4], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 11*3;

    err1 = _mm_sub_epi16( sr[5], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 10*3;

    err1 = _mm_sub_epi16( sr[6], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 9*3;

    err1 = _mm_sub_epi16( sr[7], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 8*3;

    err1 = _mm_sub_epi16( sr[8], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 7*3;

    err1 = _mm_sub_epi16( sr[9], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 6*3;

    err1 = _mm_sub_epi16( sr[10], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 5*3;

    err1 = _mm_sub_epi16( sr[11], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 4*3;

    err1 = _mm_sub_epi16( sr[12], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 3*3;

    err1 = _mm_sub_epi16( sr[13], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 2*3;

    err1 = _mm_sub_epi16( sr[14], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 1*3;

    err1 = _mm_sub_epi16( sr[15], recVal16 );
    err2 = _mm_mullo_epi16( err1, err1 );
    minerr = _mm_minpos_epu16( err2 );
    tmp = _mm_cvtsi128_si64( minerr );
    idx |= ( tmp >> 16 ) << 0*3;

    uint16_t rm[8];
    _mm_storeu_si128( (__m128i*)rm, mul );
    uint16_t sm = _mm_cvtsi128_si64( srcMid );

    uint64_t d = ( uint64_t( sm ) << 56 ) |
        ( uint64_t( rm[GetMulSel( sel )] ) << 52 ) |
        ( uint64_t( sel ) << 48 ) |
        idx;

    return _bswap64( d );
#elif defined __ARM_NEON

    int16x8_t srcMidWide, multipliers;
    int srcMid;
    uint8x16_t srcAlphaBlock = vld1q_u8( src );
    {
        uint8_t ref = src[0];
        uint8x16_t a0 = vdupq_n_u8( ref );
        uint8x16_t r = vceqq_u8( srcAlphaBlock, a0 );
        int64x2_t m = vreinterpretq_s64_u8( r );
        if( m[0] == -1 && m[1] == -1 )
            return ref;

        // srcRange
#ifdef __aarch64__
        uint8_t min = vminvq_u8( srcAlphaBlock );
        uint8_t max = vmaxvq_u8( srcAlphaBlock );
        uint8_t srcRange = max - min;
        multipliers = vqaddq_s16( vshrq_n_s16( vqdmulhq_n_s16( g_alphaRange_NEON, srcRange ), 1 ), vdupq_n_s16( 1 ) );
        srcMid = min + srcRange / 2;
        srcMidWide = vdupq_n_s16( srcMid );
#else
        uint8x8_t vmin = vpmin_u8( vget_low_u8( srcAlphaBlock ), vget_high_u8( srcAlphaBlock ) );
        vmin = vpmin_u8( vmin, vmin );
        vmin = vpmin_u8( vmin, vmin );
        vmin = vpmin_u8( vmin, vmin );
        uint8x8_t vmax = vpmax_u8( vget_low_u8( srcAlphaBlock ), vget_high_u8( srcAlphaBlock ) );
        vmax = vpmax_u8( vmax, vmax );
        vmax = vpmax_u8( vmax, vmax );
        vmax = vpmax_u8( vmax, vmax );

        int16x8_t srcRangeWide = vreinterpretq_s16_u16( vsubl_u8( vmax, vmin ) );
        multipliers = vqaddq_s16( vshrq_n_s16( vqdmulhq_s16( g_alphaRange_NEON, srcRangeWide ), 1 ), vdupq_n_s16( 1 ) );
        srcMidWide = vsraq_n_s16( vreinterpretq_s16_u16(vmovl_u8(vmin)), srcRangeWide, 1);
        srcMid = vgetq_lane_s16( srcMidWide, 0 );
#endif
    }

    // calculate reconstructed values
#define EAC_APPLY_16X( m ) m( 0 ) m( 1 ) m( 2 ) m( 3 ) m( 4 ) m( 5 ) m( 6 ) m( 7 ) m( 8 ) m( 9 ) m( 10 ) m( 11 ) m( 12 ) m( 13 ) m( 14 ) m( 15 )

#define EAC_RECONSTRUCT_VALUE( n ) vqmovun_s16( vmlaq_s16( srcMidWide, g_alpha_NEON[n], WidenMultiplier_EAC_NEON<n>( multipliers ) ) ),
    uint8x8_t recVals[16] = { EAC_APPLY_16X( EAC_RECONSTRUCT_VALUE ) };

    // find selector
    int err = std::numeric_limits<int>::max();
    int sel = 0;
    for( int r = 0; r < 16; r++ )
    {
        uint8x8_t recVal = recVals[r];

        int rangeErr = 0;
#define EAC_ACCUMULATE_ERROR( n ) rangeErr += MinError_EAC_NEON( ErrorProbe_EAC_NEON<n>( recVal, srcAlphaBlock ) );
        EAC_APPLY_16X( EAC_ACCUMULATE_ERROR )

        if( rangeErr < err )
        {
            err = rangeErr;
            sel = r;
            if ( err == 0 ) break;
        }
    }

    // combine results
    uint64_t d = ( uint64_t( srcMid ) << 56 ) |
        ( uint64_t( multipliers[GetMulSel( sel )] ) << 52 ) |
        ( uint64_t( sel ) << 48);

    // generate indices
    uint8x8_t recVal = recVals[sel];
#define EAC_INSERT_INDEX(n) d |= MinErrorIndex_EAC_NEON<n>( recVal, srcAlphaBlock );
    EAC_APPLY_16X( EAC_INSERT_INDEX )

    return _bswap64( d );

#undef EAC_APPLY_16X
#undef EAC_INSERT_INDEX
#undef EAC_ACCUMULATE_ERROR
#undef EAC_RECONSTRUCT_VALUE

#else
    {
        bool solid = true;
        const uint8_t* ptr = src + 1;
        const uint8_t ref = *src;
        for( int i=1; i<16; i++ )
        {
            if( ref != *ptr++ )
            {
                solid = false;
                break;
            }
        }
        if( solid )
        {
            return ref;
        }
    }

    uint8_t min = src[0];
    uint8_t max = src[0];
    for( int i=1; i<16; i++ )
    {
        if( min > src[i] ) min = src[i];
        else if( max < src[i] ) max = src[i];
    }
    int srcRange = max - min;
    int srcMid = min + srcRange / 2;

    uint8_t buf[16][16];
    int err = std::numeric_limits<int>::max();
    int sel;
    int selmul;
    for( int r=0; r<16; r++ )
    {
        int mul = ( ( srcRange * g_alphaRange[r] ) >> 16 ) + 1;

        int rangeErr = 0;
        for( int i=0; i<16; i++ )
        {
            const auto srcVal = src[i];

            int idx = 0;
            const auto modVal = g_alpha[r][0] * mul;
            const auto recVal = clampu8( srcMid + modVal );
            int localErr = sq( srcVal - recVal );

            if( localErr != 0 )
            {
                for( int j=1; j<8; j++ )
                {
                    const auto modVal = g_alpha[r][j] * mul;
                    const auto recVal = clampu8( srcMid + modVal );
                    const auto errProbe = sq( srcVal - recVal );
                    if( errProbe < localErr )
                    {
                        localErr = errProbe;
                        idx = j;
                    }
                }
            }

            buf[r][i] = idx;
            rangeErr += localErr;
        }

        if( rangeErr < err )
        {
            err = rangeErr;
            sel = r;
            selmul = mul;
            if( err == 0 ) break;
        }
    }

    uint64_t d = ( uint64_t( srcMid ) << 56 ) |
        ( uint64_t( selmul ) << 52 ) |
        ( uint64_t( sel ) << 48 );

    int offset = 45;
    auto ptr = buf[sel];
    for( int i=0; i<16; i++ )
    {
        d |= uint64_t( *ptr++ ) << offset;
        offset -= 3;
    }

    return _bswap64( d );
#endif
}


void CompressEtc1Alpha( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t buf[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

        __m128i c0 = _mm_castps_si128( px0 );
        __m128i c1 = _mm_castps_si128( px1 );
        __m128i c2 = _mm_castps_si128( px2 );
        __m128i c3 = _mm_castps_si128( px3 );

        __m128i mask = _mm_setr_epi32( 0x03030303, 0x07070707, 0x0b0b0b0b, 0x0f0f0f0f );
        __m128i p0 = _mm_shuffle_epi8( c0, mask );
        __m128i p1 = _mm_shuffle_epi8( c1, mask );
        __m128i p2 = _mm_shuffle_epi8( c2, mask );
        __m128i p3 = _mm_shuffle_epi8( c3, mask );

        _mm_store_si128( (__m128i*)(buf + 0),  p0 );
        _mm_store_si128( (__m128i*)(buf + 4),  p1 );
        _mm_store_si128( (__m128i*)(buf + 8),  p2 );
        _mm_store_si128( (__m128i*)(buf + 12), p3 );

        src += 4;
#else
        auto ptr = buf;
        for( int x=0; x<4; x++ )
        {
            unsigned int a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessRGB( (uint8_t*)buf );
    }
    while( --blocks );
}

void CompressEtc2Alpha( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t buf[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

        __m128i c0 = _mm_castps_si128( px0 );
        __m128i c1 = _mm_castps_si128( px1 );
        __m128i c2 = _mm_castps_si128( px2 );
        __m128i c3 = _mm_castps_si128( px3 );

        __m128i mask = _mm_setr_epi32( 0x03030303, 0x07070707, 0x0b0b0b0b, 0x0f0f0f0f );
        __m128i p0 = _mm_shuffle_epi8( c0, mask );
        __m128i p1 = _mm_shuffle_epi8( c1, mask );
        __m128i p2 = _mm_shuffle_epi8( c2, mask );
        __m128i p3 = _mm_shuffle_epi8( c3, mask );

        _mm_store_si128( (__m128i*)(buf + 0),  p0 );
        _mm_store_si128( (__m128i*)(buf + 4),  p1 );
        _mm_store_si128( (__m128i*)(buf + 8),  p2 );
        _mm_store_si128( (__m128i*)(buf + 12), p3 );

        src += 4;
#else
        auto ptr = buf;
        for( int x=0; x<4; x++ )
        {
            unsigned int a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src += width;
            a = *src >> 24;
            *ptr++ = a | ( a << 8 ) | ( a << 16 );
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessRGB_ETC2( (uint8_t*)buf );
    }
    while( --blocks );
}

#include <chrono>
#include <thread>

void CompressEtc1Rgb( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t buf[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

        _mm_store_si128( (__m128i*)(buf + 0),  _mm_castps_si128( px0 ) );
        _mm_store_si128( (__m128i*)(buf + 4),  _mm_castps_si128( px1 ) );
        _mm_store_si128( (__m128i*)(buf + 8),  _mm_castps_si128( px2 ) );
        _mm_store_si128( (__m128i*)(buf + 12), _mm_castps_si128( px3 ) );

        src += 4;
#else
        auto ptr = buf;
        for( int x=0; x<4; x++ )
        {
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessRGB( (uint8_t*)buf );
    }
    while( --blocks );
}

void CompressEtc1RgbDither( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t buf[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

#  ifdef __AVX2__
        DitherAvx2( (uint8_t*)buf, _mm_castps_si128( px0 ), _mm_castps_si128( px1 ), _mm_castps_si128( px2 ), _mm_castps_si128( px3 ) );
#  else
        _mm_store_si128( (__m128i*)(buf + 0),  _mm_castps_si128( px0 ) );
        _mm_store_si128( (__m128i*)(buf + 4),  _mm_castps_si128( px1 ) );
        _mm_store_si128( (__m128i*)(buf + 8),  _mm_castps_si128( px2 ) );
        _mm_store_si128( (__m128i*)(buf + 12), _mm_castps_si128( px3 ) );

        Dither( (uint8_t*)buf );
#  endif

        src += 4;
#else
        auto ptr = buf;
        for( int x=0; x<4; x++ )
        {
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessRGB( (uint8_t*)buf );
    }
    while( --blocks );
}

void CompressEtc2Rgb( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t buf[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

        _mm_store_si128( (__m128i*)(buf + 0),  _mm_castps_si128( px0 ) );
        _mm_store_si128( (__m128i*)(buf + 4),  _mm_castps_si128( px1 ) );
        _mm_store_si128( (__m128i*)(buf + 8),  _mm_castps_si128( px2 ) );
        _mm_store_si128( (__m128i*)(buf + 12), _mm_castps_si128( px3 ) );

        src += 4;
#else
        auto ptr = buf;
        for( int x=0; x<4; x++ )
        {
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src += width;
            *ptr++ = *src;
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessRGB_ETC2( (uint8_t*)buf );
    }
    while( --blocks );
}

void CompressEtc2Rgba( const uint32_t* src, uint64_t* dst, uint32_t blocks, size_t width )
{
    int w = 0;
    uint32_t rgba[4*4];
    uint8_t alpha[4*4];
    do
    {
#ifdef __SSE4_1__
        __m128 px0 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 0 ) ) );
        __m128 px1 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 1 ) ) );
        __m128 px2 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 2 ) ) );
        __m128 px3 = _mm_castsi128_ps( _mm_loadu_si128( (__m128i*)( src + width * 3 ) ) );

        _MM_TRANSPOSE4_PS( px0, px1, px2, px3 );

        __m128i c0 = _mm_castps_si128( px0 );
        __m128i c1 = _mm_castps_si128( px1 );
        __m128i c2 = _mm_castps_si128( px2 );
        __m128i c3 = _mm_castps_si128( px3 );

        _mm_store_si128( (__m128i*)(rgba + 0),  c0 );
        _mm_store_si128( (__m128i*)(rgba + 4),  c1 );
        _mm_store_si128( (__m128i*)(rgba + 8),  c2 );
        _mm_store_si128( (__m128i*)(rgba + 12), c3 );

        __m128i mask = _mm_setr_epi32( 0x0f0b0703, -1, -1, -1 );

        __m128i a0 = _mm_shuffle_epi8( c0, mask );
        __m128i a1 = _mm_shuffle_epi8( c1, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 3, 0, 3 ) ) );
        __m128i a2 = _mm_shuffle_epi8( c2, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 0, 3, 3 ) ) );
        __m128i a3 = _mm_shuffle_epi8( c3, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 0, 3, 3, 3 ) ) );

        __m128i s0 = _mm_or_si128( a0, a1 );
        __m128i s1 = _mm_or_si128( a2, a3 );
        __m128i s2 = _mm_or_si128( s0, s1 );

        _mm_store_si128( (__m128i*)alpha, s2 );

        src += 4;
#else
        auto ptr = rgba;
        auto ptr8 = alpha;
        for( int x=0; x<4; x++ )
        {
            auto v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src += width;
            v = *src;
            *ptr++ = v;
            *ptr8++ = v >> 24;
            src -= width * 3 - 1;
        }
#endif
        if( ++w == width/4 )
        {
            src += width * 3;
            w = 0;
        }
        *dst++ = ProcessAlpha_ETC2( alpha );
        *dst++ = ProcessRGB_ETC2( (uint8_t*)rgba );
    }
    while( --blocks );
}
