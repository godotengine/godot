#include <array>
#include <string.h>

// Godot
#include "core/typedefs.h"

#include "Math.hpp"
#include "ProcessCommon.hpp"
#include "ProcessRGB.hpp"
#include "Tables.hpp"
#include "Vector.hpp"
// Godot
//#ifdef __SSE4_1__
//#  ifdef _MSC_VER
//#    include <intrin.h>
//#    include <Windows.h>
//#    define _bswap(x) _byteswap_ulong(x)
//#  else
//#    include <x86intrin.h>
//#  endif
//#else
//#  ifndef _MSC_VER
//#    include <byteswap.h>
//#    define _bswap(x) bswap_32(x)
//#  endif
//#endif

//#ifndef _bswap
//#  define _bswap(x) __builtin_bswap32(x)
//#endif

namespace
{

typedef std::array<uint16_t, 4> v4i;

void Average( const uint8_t* data, v4i* a )
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

void CalcErrorBlock( const uint8_t* data, unsigned int err[4][4] )
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

unsigned int CalcError( const unsigned int block[4], const v4i& average )
{
    unsigned int err = 0x3FFFFFFF; // Big value to prevent negative values, but small enough to prevent overflow
    err -= block[0] * 2 * average[2];
    err -= block[1] * 2 * average[1];
    err -= block[2] * 2 * average[0];
    err += 8 * ( sq( average[0] ) + sq( average[1] ) + sq( average[2] ) );
    return err;
}

void ProcessAverages( v4i* a )
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

void EncodeAverages( uint64_t& _d, const v4i* a, size_t idx )
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

uint64_t CheckSolid( const uint8_t* src )
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

void PrepareAverages( v4i a[8], const uint8_t* src, unsigned int err[4] )
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

void FindBestFit( uint64_t terr[2][8], uint16_t tsel[16][8], v4i a[8], const uint32_t* id, const uint8_t* data )
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

#ifdef __SSE4_1__
// Non-reference implementation, but faster. Produces same results as the AVX2 version
void FindBestFit( uint32_t terr[2][8], uint16_t tsel[16][8], v4i a[8], const uint32_t* id, const uint8_t* data )
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
    }
}
#endif

uint8_t convert6(float f)
{
    int i = (std::min(std::max(static_cast<int>(f), 0), 1023) - 15) >> 1;
    return (i + 11 - ((i + 11) >> 7) - ((i + 4) >> 7)) >> 3;
}

uint8_t convert7(float f)
{
    int i = (std::min(std::max(static_cast<int>(f), 0), 1023) - 15) >> 1;
    return (i + 9 - ((i + 9) >> 8) - ((i + 6) >> 8)) >> 2;
}

std::pair<uint64_t, uint64_t> Planar(const uint8_t* src)
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

    uint64_t result = static_cast<uint32_t>(BSWAP32(lo));
	result |= static_cast<uint64_t>(static_cast<uint32_t>(BSWAP32(hi))) << 32;

    return std::make_pair(result, error);
}

template<class T, class S>
uint64_t EncodeSelectors( uint64_t d, const T terr[2][8], const S tsel[16][8], const uint32_t* id, const uint64_t value, const uint64_t error)
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

uint64_t ProcessRGB( const uint8_t* src )
{
    uint64_t d = CheckSolid( src );
    if( d != 0 ) return d;

    v4i a[8];
    unsigned int err[4] = {};
    PrepareAverages( a, src, err );
    size_t idx = GetLeastError( err, 4 );
    EncodeAverages( d, a, idx );

#if defined __SSE4_1__ && !defined REFERENCE_IMPLEMENTATION
    uint32_t terr[2][8] = {};
#else
    uint64_t terr[2][8] = {};
#endif
    uint16_t tsel[16][8];
    auto id = g_id[idx];
    FindBestFit( terr, tsel, a, id, src );

    return FixByteOrder( EncodeSelectors( d, terr, tsel, id ) );
}

uint64_t ProcessRGB_ETC2( const uint8_t* src )
{
    auto result = Planar( src );

    uint64_t d = 0;

    v4i a[8];
    unsigned int err[4] = {};
    PrepareAverages( a, src, err );
    size_t idx = GetLeastError( err, 4 );
    EncodeAverages( d, a, idx );

#if defined __SSE4_1__ && !defined REFERENCE_IMPLEMENTATION
    uint32_t terr[2][8] = {};
#else
    uint64_t terr[2][8] = {};
#endif
    uint16_t tsel[16][8];
    auto id = g_id[idx];
    FindBestFit( terr, tsel, a, id, src );

    return EncodeSelectors( d, terr, tsel, id, result.first, result.second );
}

