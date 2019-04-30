#ifdef __SSE4_1__

#include <limits>

#include "Math.hpp"
#include "ProcessAlpha.hpp"
#include "Tables.hpp"

#ifdef _MSC_VER
#  include <intrin.h>
#  include <Windows.h>
#  define _bswap(x) _byteswap_ulong(x)
#  define _bswap64(x) _byteswap_uint64(x)
#  define VS_VECTORCALL _vectorcall
#else
#  include <x86intrin.h>
#  pragma GCC push_options
#  pragma GCC target ("avx2,fma,bmi2")
#  define VS_VECTORCALL
#endif

#ifndef _bswap
#  define _bswap(x) __builtin_bswap32(x)
#  define _bswap64(x) __builtin_bswap64(x)
#endif

template<int K>
static inline __m128i VS_VECTORCALL Widen( const __m128i src )
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

static inline int VS_VECTORCALL GetMulSel( int sel )
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

uint64_t ProcessAlpha_AVX2( const uint8_t* src )
{
    // Check solid
    __m128i s = _mm_loadu_si128( (__m128i*)src );
    __m128i solidCmp = _mm_broadcastb_epi8( s );
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

    // find indices
    uint8_t buf[16][16];
    int err = std::numeric_limits<int>::max();
    int sel;
    for( int r=0; r<16; r++ )
    {
        __m128i recVal16 = rangeMul[r];

        int rangeErr = 0;
        for( int i=0; i<16; i++ )
        {
            __m128i err1 = _mm_sub_epi16( sr[i], recVal16 );
            __m128i err = _mm_mullo_epi16( err1, err1 );
            __m128i minerr = _mm_minpos_epu16( err );
            uint64_t tmp = _mm_cvtsi128_si64( minerr );
            buf[r][i] = tmp >> 16;
            rangeErr += tmp & 0xFFFF;
        }

        if( rangeErr < err )
        {
            err = rangeErr;
            sel = r;
            if( err == 0 ) break;
        }
    }

    uint16_t rm[8];
    _mm_storeu_si128( (__m128i*)rm, mul );
    uint16_t sm = _mm_cvtsi128_si64( srcMid );

    uint64_t d = ( uint64_t( sm ) << 56 ) |
        ( uint64_t( rm[GetMulSel( sel )] ) << 52 ) |
        ( uint64_t( sel ) << 48 );

    int offset = 45;
    auto ptr = buf[sel];
    for( int i=0; i<16; i++ )
    {
        d |= uint64_t( *ptr++ ) << offset;
        offset -= 3;
    }

    return _bswap64( d );
}

#ifndef _MSC_VER
#  pragma GCC pop_options
#endif

#endif
