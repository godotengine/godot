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

static uint8_t e5[32];
static uint8_t e6[64];
static uint8_t qrb[256+16];
static uint8_t qg[256+16];

void InitDither()
{
    for( int i=0; i<32; i++ )
    {
        e5[i] = (i<<3) | (i>>2);
    }
    for( int i=0; i<64; i++ )
    {
        e6[i] = (i<<2) | (i>>4);
    }
    for( int i=0; i<256+16; i++ )
    {
        int v = std::min( std::max( 0, i-8 ), 255 );
        qrb[i] = e5[mul8bit( v, 31 )];
        qg[i] = e6[mul8bit( v, 63 )];
    }
}

void Dither( uint8_t* data )
{
    int err[8];
    int* ep1 = err;
    int* ep2 = err+4;

    for( int ch=0; ch<3; ch++ )
    {
        uint8_t* ptr = data + ch;
        uint8_t* quant = (ch == 1) ? qg + 8 : qrb + 8;
        memset( err, 0, sizeof( err ) );

        for( int y=0; y<4; y++ )
        {
            uint8_t tmp;
            tmp = quant[ptr[0] + ( ( 3 * ep2[1] + 5 * ep2[0] ) >> 4 )];
            ep1[0] = ptr[0] - tmp;
            ptr[0] = tmp;
            tmp = quant[ptr[4] + ( ( 7 * ep1[0] + 3 * ep2[2] + 5 * ep2[1] + ep2[0] ) >> 4 )];
            ep1[1] = ptr[4] - tmp;
            ptr[4] = tmp;
            tmp = quant[ptr[8] + ( ( 7 * ep1[1] + 3 * ep2[3] + 5 * ep2[2] + ep2[1] ) >> 4 )];
            ep1[2] = ptr[8] - tmp;
            ptr[8] = tmp;
            tmp = quant[ptr[12] + ( ( 7 * ep1[2] + 5 * ep2[3] + ep2[2] ) >> 4 )];
            ep1[3] = ptr[12] - tmp;
            ptr[12] = tmp;
            ptr += 16;
            std::swap( ep1, ep2 );
        }
    }
}

void Swizzle(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output)
{
    for (int i = 0; i < 4; ++i)
    {
        uint64_t d0 = *(const uint64_t*)(data + i * pitch + 0);
        uint64_t d1 = *(const uint64_t*)(data + i * pitch + 8);

        *(uint64_t*)(output + i * 16 + 0) = d0;
        *(uint64_t*)(output + i * 16 + 8) = d1;
    }
}

#ifdef __SSE4_1__
// This version uses a 5 bit quantization for each channel to allow SIMD acceleration.
// Tow blocks are processed in parallel
void Dither_SSE41(const uint8_t* data0, const uint8_t* data1, uint8_t* output0, uint8_t* output1)
{
    __m128i ep1[4];
    __m128i ep2[4];

    ep1[0] = _mm_setzero_si128();
    ep1[1] = _mm_setzero_si128();
    ep1[2] = _mm_setzero_si128();
    ep1[3] = _mm_setzero_si128();

    ep2[0] = _mm_setzero_si128();
    ep2[1] = _mm_setzero_si128();
    ep2[2] = _mm_setzero_si128();
    ep2[3] = _mm_setzero_si128();

    for( int y=0; y<4; y++ )
    {
        __m128i d0 = _mm_loadl_epi64((const __m128i*)(data0 + y * 16));
        __m128i d1 = _mm_loadl_epi64((const __m128i*)(data1 + y * 16));
        __m128i d2 = _mm_unpacklo_epi32(d0, d1);

        __m128i o0;
        __m128i o1;

        // tmp = quant[ptr[0] + ( ( 3 * ep2[1] + 5 * ep2[0] ) >> 4 )];
        {
            __m128i d3 = _mm_cvtepu8_epi16(d2);

            __m128i t0 = _mm_mullo_epi16(ep2[1], _mm_set1_epi16(3));
            __m128i t1 = _mm_mullo_epi16(ep2[0], _mm_set1_epi16(5));
            __m128i t2 = _mm_add_epi16(t0, t1);
            __m128i t3 = _mm_srai_epi16(t2, 4);
            __m128i t4 = _mm_add_epi16(t3, d3);
            __m128i t5 = _mm_add_epi16(t4, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t5, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o0 = q2;

            // ep1[0] = ptr[0] - tmp;
            ep1[0] = _mm_sub_epi16(d3, q2);
        }

        // tmp = quant[ptr[4] + ( ( 7 * ep1[0] + 3 * ep2[2] + 5 * ep2[1] + ep2[0] ) >> 4 )];
        {
            __m128i d3 = _mm_unpackhi_epi8(d2, _mm_setzero_si128());

            __m128i t0 = _mm_mullo_epi16(ep1[0], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[2], _mm_set1_epi16(3));
            __m128i t2 = _mm_mullo_epi16(ep2[1], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t2, ep2[0]);
            __m128i t5 = _mm_add_epi16(t3, t4);
            __m128i t6 = _mm_srai_epi16(t5, 4);
            __m128i t7 = _mm_add_epi16(t6, d3);
            __m128i t8 = _mm_add_epi16(t7, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t8, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o1 = q2;

            // ep1[1] = ptr[4] - tmp;
            ep1[1] = _mm_sub_epi16(d3, q2);
        }

        __m128i o2 = _mm_packus_epi16(o0, o1);

        _mm_storel_epi64((__m128i*)(output0 + y * 16), _mm_shuffle_epi32(o2, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_storel_epi64((__m128i*)(output1 + y * 16), _mm_shuffle_epi32(o2, _MM_SHUFFLE(3, 1, 3, 1)));

        d0 = _mm_loadl_epi64((const __m128i*)(data0 + y * 16 + 8));
        d1 = _mm_loadl_epi64((const __m128i*)(data1 + y * 16 + 8));
        d2 = _mm_unpacklo_epi32(d0, d1);

        // tmp = quant[ptr[8] + ( ( 7 * ep1[1] + 3 * ep2[3] + 5 * ep2[2] + ep2[1] ) >> 4 )];
        {
            __m128i d3 = _mm_cvtepu8_epi16(d2);

            __m128i t0 = _mm_mullo_epi16(ep1[1], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[3], _mm_set1_epi16(3));
            __m128i t2 = _mm_mullo_epi16(ep2[2], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t2, ep2[1]);
            __m128i t5 = _mm_add_epi16(t3, t4);
            __m128i t6 = _mm_srai_epi16(t5, 4);
            __m128i t7 = _mm_add_epi16(t6, d3);
            __m128i t8 = _mm_add_epi16(t7, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t8, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o0 = q2;

            // ep1[2] = ptr[8] - tmp;
            ep1[2] = _mm_sub_epi16(d3, q2);
        }

        // tmp = quant[ptr[12] + ( ( 7 * ep1[2] + 5 * ep2[3] + ep2[2] ) >> 4 )];
        {
            __m128i d3 = _mm_unpackhi_epi8(d2, _mm_setzero_si128());

            __m128i t0 = _mm_mullo_epi16(ep1[2], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[3], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t3, ep2[2]);
            __m128i t5 = _mm_srai_epi16(t4, 4);
            __m128i t6 = _mm_add_epi16(t5, d3);
            __m128i t7 = _mm_add_epi16(t6, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t7, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o1 = q2;

            // ep1[3] = ptr[12] - tmp;
            ep1[3] = _mm_sub_epi16(d3, q2);
        }

        o2 = _mm_packus_epi16(o0, o1);

        _mm_storel_epi64((__m128i*)(output0 + y * 16 + 8), _mm_shuffle_epi32(o2, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_storel_epi64((__m128i*)(output1 + y * 16 + 8), _mm_shuffle_epi32(o2, _MM_SHUFFLE(3, 1, 3, 1)));

        for (int i = 0; i < 4; ++i)
        {
            std::swap( ep1[i], ep2[i] );
        }
    }
}

// Tow blocks are processed in parallel
void Swizzle_SSE41(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output0, uint8_t* output1)
{
    for (int i = 0; i < 4; ++i)
    {
        __m128i d0 = _mm_loadu_si128((const __m128i*)(data + i * pitch +  0));
        __m128i d1 = _mm_loadu_si128((const __m128i*)(data + i * pitch + 16));
        _mm_storeu_si128((__m128i*)(output0 + i * 16), d0);
        _mm_storeu_si128((__m128i*)(output1 + i * 16), d1);
    }
}

// This version uses a 5 bit quantization for each channel to allow SIMD acceleration.
// Tow blocks are processed in parallel
void Dither_Swizzle_SSE41(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output0, uint8_t* output1)
{
    __m128i ep1[4];
    __m128i ep2[4];

    ep1[0] = _mm_setzero_si128();
    ep1[1] = _mm_setzero_si128();
    ep1[2] = _mm_setzero_si128();
    ep1[3] = _mm_setzero_si128();

    ep2[0] = _mm_setzero_si128();
    ep2[1] = _mm_setzero_si128();
    ep2[2] = _mm_setzero_si128();
    ep2[3] = _mm_setzero_si128();

    for( int y=0; y<4; y++ )
    {
        __m128i d0 = _mm_loadl_epi64((const __m128i*)(data + y * pitch +  0));
        __m128i d1 = _mm_loadl_epi64((const __m128i*)(data + y * pitch + 16));
        __m128i d2 = _mm_unpacklo_epi32(d0, d1);

        __m128i o0;
        __m128i o1;

        // tmp = quant[ptr[0] + ( ( 3 * ep2[1] + 5 * ep2[0] ) >> 4 )];
        {
            __m128i d3 = _mm_cvtepu8_epi16(d2);

            __m128i t0 = _mm_mullo_epi16(ep2[1], _mm_set1_epi16(3));
            __m128i t1 = _mm_mullo_epi16(ep2[0], _mm_set1_epi16(5));
            __m128i t2 = _mm_add_epi16(t0, t1);
            __m128i t3 = _mm_srai_epi16(t2, 4);
            __m128i t4 = _mm_add_epi16(t3, d3);
            __m128i t5 = _mm_add_epi16(t4, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t5, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o0 = q2;

            // ep1[0] = ptr[0] - tmp;
            ep1[0] = _mm_sub_epi16(d3, q2);
        }

        // tmp = quant[ptr[4] + ( ( 7 * ep1[0] + 3 * ep2[2] + 5 * ep2[1] + ep2[0] ) >> 4 )];
        {
            __m128i d3 = _mm_unpackhi_epi8(d2, _mm_setzero_si128());

            __m128i t0 = _mm_mullo_epi16(ep1[0], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[2], _mm_set1_epi16(3));
            __m128i t2 = _mm_mullo_epi16(ep2[1], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t2, ep2[0]);
            __m128i t5 = _mm_add_epi16(t3, t4);
            __m128i t6 = _mm_srai_epi16(t5, 4);
            __m128i t7 = _mm_add_epi16(t6, d3);
            __m128i t8 = _mm_add_epi16(t7, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t8, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o1 = q2;

            // ep1[1] = ptr[4] - tmp;
            ep1[1] = _mm_sub_epi16(d3, q2);
        }

        __m128i o2 = _mm_packus_epi16(o0, o1);

        _mm_storel_epi64((__m128i*)(output0 + y * 16), _mm_shuffle_epi32(o2, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_storel_epi64((__m128i*)(output1 + y * 16), _mm_shuffle_epi32(o2, _MM_SHUFFLE(3, 1, 3, 1)));

        d0 = _mm_loadl_epi64((const __m128i*)(data + y * pitch +  8));
        d1 = _mm_loadl_epi64((const __m128i*)(data + y * pitch + 24));
        d2 = _mm_unpacklo_epi32(d0, d1);

        // tmp = quant[ptr[8] + ( ( 7 * ep1[1] + 3 * ep2[3] + 5 * ep2[2] + ep2[1] ) >> 4 )];
        {
            __m128i d3 = _mm_cvtepu8_epi16(d2);

            __m128i t0 = _mm_mullo_epi16(ep1[1], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[3], _mm_set1_epi16(3));
            __m128i t2 = _mm_mullo_epi16(ep2[2], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t2, ep2[1]);
            __m128i t5 = _mm_add_epi16(t3, t4);
            __m128i t6 = _mm_srai_epi16(t5, 4);
            __m128i t7 = _mm_add_epi16(t6, d3);
            __m128i t8 = _mm_add_epi16(t7, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t8, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o0 = q2;

            // ep1[2] = ptr[8] - tmp;
            ep1[2] = _mm_sub_epi16(d3, q2);
        }

        // tmp = quant[ptr[12] + ( ( 7 * ep1[2] + 5 * ep2[3] + ep2[2] ) >> 4 )];
        {
            __m128i d3 = _mm_unpackhi_epi8(d2, _mm_setzero_si128());

            __m128i t0 = _mm_mullo_epi16(ep1[2], _mm_set1_epi16(7));
            __m128i t1 = _mm_mullo_epi16(ep2[3], _mm_set1_epi16(5));
            __m128i t3 = _mm_add_epi16(t0, t1);
            __m128i t4 = _mm_add_epi16(t3, ep2[2]);
            __m128i t5 = _mm_srai_epi16(t4, 4);
            __m128i t6 = _mm_add_epi16(t5, d3);
            __m128i t7 = _mm_add_epi16(t6, _mm_set1_epi16(4));

            // clamp to 0..255
            __m128i c0 = _mm_min_epi16(t7, _mm_set1_epi16(255));
            __m128i c1 = _mm_max_epi16(c0, _mm_set1_epi16(0));

            __m128i q0 = _mm_and_si128(c1, _mm_set1_epi16(0xF8));
            __m128i q1 = _mm_srli_epi16(c1, 5);
            __m128i q2 = _mm_or_si128(q0, q1);
            o1 = q2;

            // ep1[3] = ptr[12] - tmp;
            ep1[3] = _mm_sub_epi16(d3, q2);
        }

        o2 = _mm_packus_epi16(o0, o1);

        _mm_storel_epi64((__m128i*)(output0 + y * 16 + 8), _mm_shuffle_epi32(o2, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_storel_epi64((__m128i*)(output1 + y * 16 + 8), _mm_shuffle_epi32(o2, _MM_SHUFFLE(3, 1, 3, 1)));

        for (int i = 0; i < 4; ++i)
        {
            std::swap( ep1[i], ep2[i] );
        }
    }
}
#endif

