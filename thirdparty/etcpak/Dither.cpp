#include <algorithm>
#include <string.h>

#include "Dither.hpp"
#include "Math.hpp"
#ifdef __SSE4_1__
#  ifdef _MSC_VER
#    include <intrin.h>
#    include <Windows.h>
#  else
#    include <x86intrin.h>
#  endif
#endif

#ifdef __AVX2__
void DitherAvx2( uint8_t* data, __m128i px0, __m128i px1, __m128i px2, __m128i px3 )
{
    static constexpr uint8_t a31[] = { 0, 0, 0, 1, 2, 0, 4, 0, 0, 2, 0, 0, 4, 0, 3, 0 };
    static constexpr uint8_t a63[] = { 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0 };
    static constexpr uint8_t s31[] = { 5, 0, 4, 0, 0, 2, 0, 1, 3, 0, 4, 0, 0, 0, 0, 2 };
    static constexpr uint8_t s63[] = { 2, 0, 2, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1 };

    const __m256i BayerAdd0 = _mm256_setr_epi8(
        a31[0], a63[0], a31[0], 0, a31[1], a63[1], a31[1], 0, a31[2], a63[2], a31[2], 0, a31[3], a63[3], a31[3], 0,
        a31[4], a63[4], a31[4], 0, a31[5], a63[5], a31[5], 0, a31[6], a63[6], a31[6], 0, a31[7], a63[7], a31[7], 0
    );
    const __m256i BayerAdd1 = _mm256_setr_epi8(
        a31[8],  a63[8],  a31[8],  0, a31[9],  a63[9],  a31[9],  0, a31[10], a63[10], a31[10], 0, a31[11], a63[11], a31[11], 0,
        a31[12], a63[12], a31[12], 0, a31[13], a63[13], a31[13], 0, a31[14], a63[14], a31[14], 0, a31[15], a63[15], a31[15], 0
    );
    const __m256i BayerSub0 = _mm256_setr_epi8(
        s31[0], s63[0], s31[0], 0, s31[1], s63[1], s31[1], 0, s31[2], s63[2], s31[2], 0, s31[3], s63[3], s31[3], 0,
        s31[4], s63[4], s31[4], 0, s31[5], s63[5], s31[5], 0, s31[6], s63[6], s31[6], 0, s31[7], s63[7], s31[7], 0
    );
    const __m256i BayerSub1 = _mm256_setr_epi8(
        s31[8],  s63[8],  s31[8],  0, s31[9],  s63[9],  s31[9],  0, s31[10], s63[10], s31[10], 0, s31[11], s63[11], s31[11], 0,
        s31[12], s63[12], s31[12], 0, s31[13], s63[13], s31[13], 0, s31[14], s63[14], s31[14], 0, s31[15], s63[15], s31[15], 0
    );

    __m256i l0 = _mm256_inserti128_si256( _mm256_castsi128_si256( px0 ), px1, 1 );
    __m256i l1 = _mm256_inserti128_si256( _mm256_castsi128_si256( px2 ), px3, 1 );

    __m256i a0 = _mm256_adds_epu8( l0, BayerAdd0 );
    __m256i a1 = _mm256_adds_epu8( l1, BayerAdd1 );
    __m256i s0 = _mm256_subs_epu8( a0, BayerSub0 );
    __m256i s1 = _mm256_subs_epu8( a1, BayerSub1 );

    _mm256_storeu_si256( (__m256i*)(data   ), s0 );
    _mm256_storeu_si256( (__m256i*)(data+32), s1 );

}
#endif

void Dither( uint8_t* data )
{
#ifdef __AVX2__
    static constexpr uint8_t a31[] = { 0, 0, 0, 1, 2, 0, 4, 0, 0, 2, 0, 0, 4, 0, 3, 0 };
    static constexpr uint8_t a63[] = { 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0 };
    static constexpr uint8_t s31[] = { 5, 0, 4, 0, 0, 2, 0, 1, 3, 0, 4, 0, 0, 0, 0, 2 };
    static constexpr uint8_t s63[] = { 2, 0, 2, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1 };

    const __m256i BayerAdd0 = _mm256_setr_epi8(
        a31[0], a63[0], a31[0], 0, a31[1], a63[1], a31[1], 0, a31[2], a63[2], a31[2], 0, a31[3], a63[3], a31[3], 0,
        a31[4], a63[4], a31[4], 0, a31[5], a63[5], a31[5], 0, a31[6], a63[6], a31[6], 0, a31[7], a63[7], a31[7], 0
    );
    const __m256i BayerAdd1 = _mm256_setr_epi8(
        a31[8],  a63[8],  a31[8],  0, a31[9],  a63[9],  a31[9],  0, a31[10], a63[10], a31[10], 0, a31[11], a63[11], a31[11], 0,
        a31[12], a63[12], a31[12], 0, a31[13], a63[13], a31[13], 0, a31[14], a63[14], a31[14], 0, a31[15], a63[15], a31[15], 0
    );
    const __m256i BayerSub0 = _mm256_setr_epi8(
        s31[0], s63[0], s31[0], 0, s31[1], s63[1], s31[1], 0, s31[2], s63[2], s31[2], 0, s31[3], s63[3], s31[3], 0,
        s31[4], s63[4], s31[4], 0, s31[5], s63[5], s31[5], 0, s31[6], s63[6], s31[6], 0, s31[7], s63[7], s31[7], 0
    );
    const __m256i BayerSub1 = _mm256_setr_epi8(
        s31[8],  s63[8],  s31[8],  0, s31[9],  s63[9],  s31[9],  0, s31[10], s63[10], s31[10], 0, s31[11], s63[11], s31[11], 0,
        s31[12], s63[12], s31[12], 0, s31[13], s63[13], s31[13], 0, s31[14], s63[14], s31[14], 0, s31[15], s63[15], s31[15], 0
    );

    __m256i px0 = _mm256_loadu_si256( (__m256i*)(data   ) );
    __m256i px1 = _mm256_loadu_si256( (__m256i*)(data+32) );

    __m256i a0 = _mm256_adds_epu8( px0, BayerAdd0 );
    __m256i a1 = _mm256_adds_epu8( px1, BayerAdd1 );
    __m256i s0 = _mm256_subs_epu8( a0, BayerSub0 );
    __m256i s1 = _mm256_subs_epu8( a1, BayerSub1 );

    _mm256_storeu_si256( (__m256i*)(data   ), s0 );
    _mm256_storeu_si256( (__m256i*)(data+32), s1 );
#else
    static constexpr int8_t Bayer31[16] = {
        ( 0-8)*2/3, ( 8-8)*2/3, ( 2-8)*2/3, (10-8)*2/3,
        (12-8)*2/3, ( 4-8)*2/3, (14-8)*2/3, ( 6-8)*2/3,
        ( 3-8)*2/3, (11-8)*2/3, ( 1-8)*2/3, ( 9-8)*2/3,
        (15-8)*2/3, ( 7-8)*2/3, (13-8)*2/3, ( 5-8)*2/3
    };
    static constexpr int8_t Bayer63[16] = {
        ( 0-8)*2/6, ( 8-8)*2/6, ( 2-8)*2/6, (10-8)*2/6,
        (12-8)*2/6, ( 4-8)*2/6, (14-8)*2/6, ( 6-8)*2/6,
        ( 3-8)*2/6, (11-8)*2/6, ( 1-8)*2/6, ( 9-8)*2/6,
        (15-8)*2/6, ( 7-8)*2/6, (13-8)*2/6, ( 5-8)*2/6
    };

    for( int i=0; i<16; i++ )
    {
        uint32_t col;
        memcpy( &col, data, 4 );
        uint8_t r = col & 0xFF;
        uint8_t g = ( col >> 8 ) & 0xFF;
        uint8_t b = ( col >> 16 ) & 0xFF;

        r = clampu8( r + Bayer31[i] );
        g = clampu8( g + Bayer63[i] );
        b = clampu8( b + Bayer31[i] );

        col = r | ( g << 8 ) | ( b << 16 );
        memcpy( data, &col, 4 );
        data += 4;
    }
#endif
}
