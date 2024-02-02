// -- GODOT start --

#include "ForceInline.hpp"
#include "ProcessRgtc.hpp"

#include <assert.h>
#include <string.h>

#if defined __AVX__ && !defined __SSE4_1__
#  define __SSE4_1__
#endif

#if defined __SSE4_1__ || defined __AVX2__
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <x86intrin.h>
#    ifndef _mm256_cvtsi256_si32
#      define _mm256_cvtsi256_si32( v ) ( _mm_cvtsi128_si32( _mm256_castsi256_si128( v ) ) )
#    endif
#  endif
#endif

static const uint8_t AlphaIndexTable[8] = { 1, 7, 6, 5, 4, 3, 2, 0 };

static const uint8_t AlphaIndexTable_SSE[64] = {
    9,      15,     14,     13,     12,     11,     10,     8,      57,     63,     62,     61,     60,     59,     58,     56,
    49,     55,     54,     53,     52,     51,     50,     48,     41,     47,     46,     45,     44,     43,     42,     40,
    33,     39,     38,     37,     36,     35,     34,     32,     25,     31,     30,     29,     28,     27,     26,     24,
    17,     23,     22,     21,     20,     19,     18,     16,     1,      7,      6,      5,      4,      3,      2,      0,
};

static const uint16_t DivTableAlpha[256] = {
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xe38e, 0xcccc, 0xba2e, 0xaaaa, 0x9d89, 0x9249, 0x8888, 0x8000,
    0x7878, 0x71c7, 0x6bca, 0x6666, 0x6186, 0x5d17, 0x590b, 0x5555, 0x51eb, 0x4ec4, 0x4bda, 0x4924, 0x469e, 0x4444, 0x4210, 0x4000,
    0x3e0f, 0x3c3c, 0x3a83, 0x38e3, 0x3759, 0x35e5, 0x3483, 0x3333, 0x31f3, 0x30c3, 0x2fa0, 0x2e8b, 0x2d82, 0x2c85, 0x2b93, 0x2aaa,
    0x29cb, 0x28f5, 0x2828, 0x2762, 0x26a4, 0x25ed, 0x253c, 0x2492, 0x23ee, 0x234f, 0x22b6, 0x2222, 0x2192, 0x2108, 0x2082, 0x2000,
    0x1f81, 0x1f07, 0x1e91, 0x1e1e, 0x1dae, 0x1d41, 0x1cd8, 0x1c71, 0x1c0e, 0x1bac, 0x1b4e, 0x1af2, 0x1a98, 0x1a41, 0x19ec, 0x1999,
    0x1948, 0x18f9, 0x18ac, 0x1861, 0x1818, 0x17d0, 0x178a, 0x1745, 0x1702, 0x16c1, 0x1681, 0x1642, 0x1605, 0x15c9, 0x158e, 0x1555,
    0x151d, 0x14e5, 0x14af, 0x147a, 0x1446, 0x1414, 0x13e2, 0x13b1, 0x1381, 0x1352, 0x1323, 0x12f6, 0x12c9, 0x129e, 0x1273, 0x1249,
    0x121f, 0x11f7, 0x11cf, 0x11a7, 0x1181, 0x115b, 0x1135, 0x1111, 0x10ec, 0x10c9, 0x10a6, 0x1084, 0x1062, 0x1041, 0x1020, 0x1000,
    0x0fe0, 0x0fc0, 0x0fa2, 0x0f83, 0x0f66, 0x0f48, 0x0f2b, 0x0f0f, 0x0ef2, 0x0ed7, 0x0ebb, 0x0ea0, 0x0e86, 0x0e6c, 0x0e52, 0x0e38,
    0x0e1f, 0x0e07, 0x0dee, 0x0dd6, 0x0dbe, 0x0da7, 0x0d90, 0x0d79, 0x0d62, 0x0d4c, 0x0d36, 0x0d20, 0x0d0b, 0x0cf6, 0x0ce1, 0x0ccc,
    0x0cb8, 0x0ca4, 0x0c90, 0x0c7c, 0x0c69, 0x0c56, 0x0c43, 0x0c30, 0x0c1e, 0x0c0c, 0x0bfa, 0x0be8, 0x0bd6, 0x0bc5, 0x0bb3, 0x0ba2,
    0x0b92, 0x0b81, 0x0b70, 0x0b60, 0x0b50, 0x0b40, 0x0b30, 0x0b21, 0x0b11, 0x0b02, 0x0af3, 0x0ae4, 0x0ad6, 0x0ac7, 0x0ab8, 0x0aaa,
    0x0a9c, 0x0a8e, 0x0a80, 0x0a72, 0x0a65, 0x0a57, 0x0a4a, 0x0a3d, 0x0a30, 0x0a23, 0x0a16, 0x0a0a, 0x09fd, 0x09f1, 0x09e4, 0x09d8,
    0x09cc, 0x09c0, 0x09b4, 0x09a9, 0x099d, 0x0991, 0x0986, 0x097b, 0x0970, 0x0964, 0x095a, 0x094f, 0x0944, 0x0939, 0x092f, 0x0924,
    0x091a, 0x090f, 0x0905, 0x08fb, 0x08f1, 0x08e7, 0x08dd, 0x08d3, 0x08ca, 0x08c0, 0x08b7, 0x08ad, 0x08a4, 0x089a, 0x0891, 0x0888,
    0x087f, 0x0876, 0x086d, 0x0864, 0x085b, 0x0853, 0x084a, 0x0842, 0x0839, 0x0831, 0x0828, 0x0820, 0x0818, 0x0810, 0x0808, 0x0800,
};

static etcpak_force_inline uint64_t ProcessAlpha( const uint8_t* src )
{
    uint8_t solid8 = *src;
    uint16_t solid16 = uint16_t( solid8 ) | ( uint16_t( solid8 ) << 8 );
    uint32_t solid32 = uint32_t( solid16 ) | ( uint32_t( solid16 ) << 16 );
    uint64_t solid64 = uint64_t( solid32 ) | ( uint64_t( solid32 ) << 32 );
    if( memcmp( src, &solid64, 8 ) == 0 && memcmp( src+8, &solid64, 8 ) == 0 )
    {
        return solid8;
    }

    uint8_t min = src[0];
    uint8_t max = min;
    for( int i=1; i<16; i++ )
    {
        const auto v = src[i];
        if( v > max ) max = v;
        else if( v < min ) min = v;
    }

    uint32_t range = ( 8 << 13 ) / ( 1 + max - min );
    uint64_t data = 0;
    for( int i=0; i<16; i++ )
    {
        uint8_t a = src[i] - min;
        uint64_t idx = AlphaIndexTable[( a * range ) >> 13];
        data |= idx << (i*3);
    }

    return max | ( min << 8 ) | ( data << 16 );
}

#ifdef __SSE4_1__
static etcpak_force_inline uint64_t Process_Alpha_SSE( __m128i a )
{
    __m128i solidCmp = _mm_shuffle_epi8( a, _mm_setzero_si128() );
    __m128i cmpRes = _mm_cmpeq_epi8( a, solidCmp );
    if( _mm_testc_si128( cmpRes, _mm_set1_epi32( -1 ) ) )
    {
        return _mm_cvtsi128_si32( a ) & 0xFF;
    }

    __m128i a1 = _mm_shuffle_epi32( a, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    __m128i max1 = _mm_max_epu8( a, a1 );
    __m128i min1 = _mm_min_epu8( a, a1 );
    __m128i amax2 = _mm_shuffle_epi32( max1, _MM_SHUFFLE( 0, 0, 2, 2 ) );
    __m128i amin2 = _mm_shuffle_epi32( min1, _MM_SHUFFLE( 0, 0, 2, 2 ) );
    __m128i max2 = _mm_max_epu8( max1, amax2 );
    __m128i min2 = _mm_min_epu8( min1, amin2 );
    __m128i amax3 = _mm_alignr_epi8( max2, max2, 2 );
    __m128i amin3 = _mm_alignr_epi8( min2, min2, 2 );
    __m128i max3 = _mm_max_epu8( max2, amax3 );
    __m128i min3 = _mm_min_epu8( min2, amin3 );
    __m128i amax4 = _mm_alignr_epi8( max3, max3, 1 );
    __m128i amin4 = _mm_alignr_epi8( min3, min3, 1 );
    __m128i max = _mm_max_epu8( max3, amax4 );
    __m128i min = _mm_min_epu8( min3, amin4 );
    __m128i minmax = _mm_unpacklo_epi8( max, min );

    __m128i r = _mm_sub_epi8( max, min );
    int range = _mm_cvtsi128_si32( r ) & 0xFF;
    __m128i rv = _mm_set1_epi16( DivTableAlpha[range] );

    __m128i v = _mm_sub_epi8( a, min );

    __m128i lo16 = _mm_unpacklo_epi8( v, _mm_setzero_si128() );
    __m128i hi16 = _mm_unpackhi_epi8( v, _mm_setzero_si128() );

    __m128i lomul = _mm_mulhi_epu16( lo16, rv );
    __m128i himul = _mm_mulhi_epu16( hi16, rv );

    __m128i p0 = _mm_packus_epi16( lomul, himul );
    __m128i p1 = _mm_or_si128( _mm_and_si128( p0, _mm_set1_epi16( 0x3F ) ), _mm_srai_epi16( _mm_and_si128( p0, _mm_set1_epi16( 0x3F00 ) ), 5 ) );
    __m128i p2 = _mm_packus_epi16( p1, p1 );

    uint64_t pi = _mm_cvtsi128_si64( p2 );
    uint64_t data = 0;
    for( int i=0; i<8; i++ )
    {
        uint64_t idx = AlphaIndexTable_SSE[(pi>>(i*8)) & 0x3F];
        data |= idx << (i*6);
    }
    return (uint64_t)(uint16_t)_mm_cvtsi128_si32( minmax ) | ( data << 16 );
}
#endif

void CompressRgtcR(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width) 
{
	int i = 0;
	auto ptr = dst;
	do 
	{
#ifdef __SSE4_1__
		__m128i px0 = _mm_loadu_si128( (__m128i*)( src + width * 0 ) );
        __m128i px1 = _mm_loadu_si128( (__m128i*)( src + width * 1 ) );
        __m128i px2 = _mm_loadu_si128( (__m128i*)( src + width * 2 ) );
        __m128i px3 = _mm_loadu_si128( (__m128i*)( src + width * 3 ) );

		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		__m128i mask = _mm_setr_epi32( 0x0c080400, -1, -1, -1 );

    	__m128i m0 = _mm_shuffle_epi8( px0, mask );
    	__m128i m1 = _mm_shuffle_epi8( px1, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 3, 0, 3 ) ) );
    	__m128i m2 = _mm_shuffle_epi8( px2, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 0, 3, 3 ) ) );
    	__m128i m3 = _mm_shuffle_epi8( px3, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 0, 3, 3, 3 ) ) );
    	__m128i m4 = _mm_or_si128( m0, m1 );
    	__m128i m5 = _mm_or_si128( m2, m3 );

		*ptr++ = Process_Alpha_SSE(_mm_or_si128( m4, m5 ));
#else
		uint8_t r[4 * 4];
		auto rgba = src;
		for (int i = 0; i < 4; i++) 
		{
			r[i * 4] = rgba[0] & 0xff;
			r[i * 4 + 1] = rgba[1] & 0xff;
			r[i * 4 + 2] = rgba[2] & 0xff;
			r[i * 4 + 3] = rgba[3] & 0xff;

			rgba += width;
		}

		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		*ptr++ = ProcessAlpha(r);
#endif
	} 
	while (--blocks);
}

void CompressRgtcRG(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width) 
{
	int i = 0;
	auto ptr = dst;
	do 
	{
#ifdef __SSE4_1__
		__m128i px0 = _mm_loadu_si128( (__m128i*)( src + width * 0 ) );
        __m128i px1 = _mm_loadu_si128( (__m128i*)( src + width * 1 ) );
        __m128i px2 = _mm_loadu_si128( (__m128i*)( src + width * 2 ) );
        __m128i px3 = _mm_loadu_si128( (__m128i*)( src + width * 3 ) );
		
		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		__m128i mask = _mm_setr_epi32( 0x0c080400, -1, -1, -1 );

    	__m128i m0 = _mm_shuffle_epi8( px0, mask );
    	__m128i m1 = _mm_shuffle_epi8( px1, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 3, 0, 3 ) ) );
    	__m128i m2 = _mm_shuffle_epi8( px2, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 0, 3, 3 ) ) );
    	__m128i m3 = _mm_shuffle_epi8( px3, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 0, 3, 3, 3 ) ) );
    	__m128i m4 = _mm_or_si128( m0, m1 );
    	__m128i m5 = _mm_or_si128( m2, m3 );

		*ptr++ = Process_Alpha_SSE(_mm_or_si128( m4, m5 ));

		mask = _mm_setr_epi32( 0x0d090501, -1, -1, -1 );

		m0 = _mm_shuffle_epi8( px0, mask );
    	m1 = _mm_shuffle_epi8( px1, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 3, 0, 3 ) ) );
    	m2 = _mm_shuffle_epi8( px2, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 3, 0, 3, 3 ) ) );
    	m3 = _mm_shuffle_epi8( px3, _mm_shuffle_epi32( mask, _MM_SHUFFLE( 0, 3, 3, 3 ) ) );
    	m4 = _mm_or_si128( m0, m1 );
    	m5 = _mm_or_si128( m2, m3 );

		*ptr++ = Process_Alpha_SSE(_mm_or_si128( m4, m5 ));
#else
		uint8_t rg[4 * 4 * 2];
		auto rgba = src;
		for (int i = 0; i < 4; i++) 
		{
			rg[i * 4] = rgba[0] & 0xff;
			rg[i * 4 + 1] = rgba[1] & 0xff;
			rg[i * 4 + 2] = rgba[2] & 0xff;
			rg[i * 4 + 3] = rgba[3] & 0xff;

			rg[16 + i * 4] = (rgba[0] & 0xff00) >> 8;
			rg[16 + i * 4 + 1] = (rgba[1] & 0xff00) >> 8;
			rg[16 + i * 4 + 2] = (rgba[2] & 0xff00) >> 8;
			rg[16 + i * 4 + 3] = (rgba[3] & 0xff00) >> 8;

			rgba += width;
		}

		src += 4;
		if (++i == width / 4) 
		{
			src += width * 3;
			i = 0;
		}

		*ptr++ = ProcessAlpha(rg);
		*ptr++ = ProcessAlpha(&rg[16]);
#endif
	} 
	while (--blocks);
}

// -- GODOT end --
