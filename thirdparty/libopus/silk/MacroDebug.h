/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Copyright (C) 2012 Xiph.Org Foundation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef MACRO_DEBUG_H
#define MACRO_DEBUG_H

/* Redefine macro functions with extensive assertion in DEBUG mode.
   As functions can't be undefined, this file can't work with SigProcFIX_MacroCount.h */

#if ( defined (FIXED_DEBUG) || ( 0 && defined (_DEBUG) ) ) && !defined (silk_MACRO_COUNT)

#undef silk_ADD16
#define silk_ADD16(a,b) silk_ADD16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_ADD16_(opus_int16 a, opus_int16 b, char *file, int line){
    opus_int16 ret;

    ret = a + b;
    if ( ret != silk_ADD_SAT16( a, b ) )
    {
        fprintf (stderr, "silk_ADD16(%d, %d) in %s: line %d\n", a, b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_ADD32
#define silk_ADD32(a,b) silk_ADD32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_ADD32_(opus_int32 a, opus_int32 b, char *file, int line){
    opus_int32 ret;

    ret = (opus_int32)((opus_uint32)a + (opus_uint32)b);
    if ( ret != silk_ADD_SAT32( a, b ) )
    {
        fprintf (stderr, "silk_ADD32(%d, %d) in %s: line %d\n", a, b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_ADD64
#define silk_ADD64(a,b) silk_ADD64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_ADD64_(opus_int64 a, opus_int64 b, char *file, int line){
    opus_int64 ret;

    ret = a + b;
    if ( ret != silk_ADD_SAT64( a, b ) )
    {
        fprintf (stderr, "silk_ADD64(%lld, %lld) in %s: line %d\n", (long long)a, (long long)b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SUB16
#define silk_SUB16(a,b) silk_SUB16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_SUB16_(opus_int16 a, opus_int16 b, char *file, int line){
    opus_int16 ret;

    ret = a - b;
    if ( ret != silk_SUB_SAT16( a, b ) )
    {
        fprintf (stderr, "silk_SUB16(%d, %d) in %s: line %d\n", a, b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SUB32
#define silk_SUB32(a,b) silk_SUB32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SUB32_(opus_int32 a, opus_int32 b, char *file, int line){
    opus_int64 ret;

    ret = a - (opus_int64)b;
    if ( ret != silk_SUB_SAT32( a, b ) )
    {
        fprintf (stderr, "silk_SUB32(%d, %d) in %s: line %d\n", a, b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SUB64
#define silk_SUB64(a,b) silk_SUB64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_SUB64_(opus_int64 a, opus_int64 b, char *file, int line){
    opus_int64 ret;

    ret = a - b;
    if ( ret != silk_SUB_SAT64( a, b ) )
    {
        fprintf (stderr, "silk_SUB64(%lld, %lld) in %s: line %d\n", (long long)a, (long long)b, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_ADD_SAT16
#define silk_ADD_SAT16(a,b) silk_ADD_SAT16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_ADD_SAT16_( opus_int16 a16, opus_int16 b16, char *file, int line) {
    opus_int16 res;
    res = (opus_int16)silk_SAT16( silk_ADD32( (opus_int32)(a16), (b16) ) );
    if ( res != silk_SAT16( (opus_int32)a16 + (opus_int32)b16 ) )
    {
        fprintf (stderr, "silk_ADD_SAT16(%d, %d) in %s: line %d\n", a16, b16, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_ADD_SAT32
#define silk_ADD_SAT32(a,b) silk_ADD_SAT32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_ADD_SAT32_(opus_int32 a32, opus_int32 b32, char *file, int line){
    opus_int32 res;
    res =   ((((opus_uint32)(a32) + (opus_uint32)(b32)) & 0x80000000) == 0 ?       \
            ((((a32) & (b32)) & 0x80000000) != 0 ? silk_int32_MIN : (a32)+(b32)) : \
            ((((a32) | (b32)) & 0x80000000) == 0 ? silk_int32_MAX : (a32)+(b32)) );
    if ( res != silk_SAT32( (opus_int64)a32 + (opus_int64)b32 ) )
    {
        fprintf (stderr, "silk_ADD_SAT32(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_ADD_SAT64
#define silk_ADD_SAT64(a,b) silk_ADD_SAT64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_ADD_SAT64_( opus_int64 a64, opus_int64 b64, char *file, int line) {
    opus_int64 res;
    int        fail = 0;
    res =   ((((a64) + (b64)) & 0x8000000000000000LL) == 0 ?                                 \
            ((((a64) & (b64)) & 0x8000000000000000LL) != 0 ? silk_int64_MIN : (a64)+(b64)) : \
            ((((a64) | (b64)) & 0x8000000000000000LL) == 0 ? silk_int64_MAX : (a64)+(b64)) );
    if( res != a64 + b64 ) {
        /* Check that we saturated to the correct extreme value */
        if ( !(( res == silk_int64_MAX && ( ( a64 >> 1 ) + ( b64 >> 1 ) > ( silk_int64_MAX >> 3 ) ) ) ||
               ( res == silk_int64_MIN && ( ( a64 >> 1 ) + ( b64 >> 1 ) < ( silk_int64_MIN >> 3 ) ) ) ) )
        {
            fail = 1;
        }
    } else {
        /* Saturation not necessary */
        fail = res != a64 + b64;
    }
    if ( fail )
    {
        fprintf (stderr, "silk_ADD_SAT64(%lld, %lld) in %s: line %d\n", (long long)a64, (long long)b64, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_SUB_SAT16
#define silk_SUB_SAT16(a,b) silk_SUB_SAT16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_SUB_SAT16_( opus_int16 a16, opus_int16 b16, char *file, int line ) {
    opus_int16 res;
    res = (opus_int16)silk_SAT16( silk_SUB32( (opus_int32)(a16), (b16) ) );
    if ( res != silk_SAT16( (opus_int32)a16 - (opus_int32)b16 ) )
    {
        fprintf (stderr, "silk_SUB_SAT16(%d, %d) in %s: line %d\n", a16, b16, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_SUB_SAT32
#define silk_SUB_SAT32(a,b) silk_SUB_SAT32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SUB_SAT32_( opus_int32 a32, opus_int32 b32, char *file, int line ) {
    opus_int32 res;
    res =   ((((opus_uint32)(a32)-(opus_uint32)(b32)) & 0x80000000) == 0 ?                \
            (( (a32) & ((b32)^0x80000000) & 0x80000000) ? silk_int32_MIN : (a32)-(b32)) : \
            ((((a32)^0x80000000) & (b32)  & 0x80000000) ? silk_int32_MAX : (a32)-(b32)) );
    if ( res != silk_SAT32( (opus_int64)a32 - (opus_int64)b32 ) )
    {
        fprintf (stderr, "silk_SUB_SAT32(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_SUB_SAT64
#define silk_SUB_SAT64(a,b) silk_SUB_SAT64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_SUB_SAT64_( opus_int64 a64, opus_int64 b64, char *file, int line ) {
    opus_int64 res;
    int        fail = 0;
    res =   ((((a64)-(b64)) & 0x8000000000000000LL) == 0 ?                                                    \
            (( (a64) & ((b64)^0x8000000000000000LL) & 0x8000000000000000LL) ? silk_int64_MIN : (a64)-(b64)) : \
            ((((a64)^0x8000000000000000LL) & (b64)  & 0x8000000000000000LL) ? silk_int64_MAX : (a64)-(b64)) );
    if( res != a64 - b64 ) {
        /* Check that we saturated to the correct extreme value */
        if( !(( res == silk_int64_MAX && ( ( a64 >> 1 ) + ( b64 >> 1 ) > ( silk_int64_MAX >> 3 ) ) ) ||
              ( res == silk_int64_MIN && ( ( a64 >> 1 ) + ( b64 >> 1 ) < ( silk_int64_MIN >> 3 ) ) ) ))
        {
            fail = 1;
        }
    } else {
        /* Saturation not necessary */
        fail = res != a64 - b64;
    }
    if ( fail )
    {
        fprintf (stderr, "silk_SUB_SAT64(%lld, %lld) in %s: line %d\n", (long long)a64, (long long)b64, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return res;
}

#undef silk_MUL
#define silk_MUL(a,b) silk_MUL_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_MUL_(opus_int32 a32, opus_int32 b32, char *file, int line){
    opus_int32 ret;
    opus_int64 ret64;
    ret = (opus_int32)((opus_uint32)a32 * (opus_uint32)b32);
    ret64 = (opus_int64)a32 * (opus_int64)b32;
    if ( (opus_int64)ret != ret64 )
    {
        fprintf (stderr, "silk_MUL(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_MUL_uint
#define silk_MUL_uint(a,b) silk_MUL_uint_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_uint32 silk_MUL_uint_(opus_uint32 a32, opus_uint32 b32, char *file, int line){
    opus_uint32 ret;
    ret = a32 * b32;
    if ( (opus_uint64)ret != (opus_uint64)a32 * (opus_uint64)b32 )
    {
        fprintf (stderr, "silk_MUL_uint(%u, %u) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_MLA
#define silk_MLA(a,b,c) silk_MLA_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_MLA_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = a32 + b32 * c32;
    if ( (opus_int64)ret != (opus_int64)a32 + (opus_int64)b32 * (opus_int64)c32 )
    {
        fprintf (stderr, "silk_MLA(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_MLA_uint
#define silk_MLA_uint(a,b,c) silk_MLA_uint_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_MLA_uint_(opus_uint32 a32, opus_uint32 b32, opus_uint32 c32, char *file, int line){
    opus_uint32 ret;
    ret = a32 + b32 * c32;
    if ( (opus_int64)ret != (opus_int64)a32 + (opus_int64)b32 * (opus_int64)c32 )
    {
        fprintf (stderr, "silk_MLA_uint(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMULWB
#define silk_SMULWB(a,b) silk_SMULWB_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMULWB_(opus_int32 a32, opus_int32 b32, char *file, int line){
    opus_int32 ret;
    ret = (a32 >> 16) * (opus_int32)((opus_int16)b32) + (((a32 & 0x0000FFFF) * (opus_int32)((opus_int16)b32)) >> 16);
    if ( (opus_int64)ret != ((opus_int64)a32 * (opus_int16)b32) >> 16 )
    {
        fprintf (stderr, "silk_SMULWB(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMLAWB
#define silk_SMLAWB(a,b,c) silk_SMLAWB_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLAWB_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = silk_ADD32_ovflw( a32, silk_SMULWB( b32, c32 ) );
    if ( ret != silk_ADD_SAT32( a32, silk_SMULWB( b32, c32 ) ) )
    {
        fprintf (stderr, "silk_SMLAWB(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMULWT
#define silk_SMULWT(a,b) silk_SMULWT_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMULWT_(opus_int32 a32, opus_int32 b32, char *file, int line){
    opus_int32 ret;
    ret = (a32 >> 16) * (b32 >> 16) + (((a32 & 0x0000FFFF) * (b32 >> 16)) >> 16);
    if ( (opus_int64)ret != ((opus_int64)a32 * (b32 >> 16)) >> 16 )
    {
        fprintf (stderr, "silk_SMULWT(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMLAWT
#define silk_SMLAWT(a,b,c) silk_SMLAWT_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLAWT_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = a32 + ((b32 >> 16) * (c32 >> 16)) + (((b32 & 0x0000FFFF) * ((c32 >> 16)) >> 16));
    if ( (opus_int64)ret != (opus_int64)a32 + (((opus_int64)b32 * (c32 >> 16)) >> 16) )
    {
        fprintf (stderr, "silk_SMLAWT(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMULL
#define silk_SMULL(a,b) silk_SMULL_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_SMULL_(opus_int64 a64, opus_int64 b64, char *file, int line){
    opus_int64 ret64;
    int        fail = 0;
    ret64 = a64 * b64;
    if( b64 != 0 ) {
        fail = a64 != (ret64 / b64);
    } else if( a64 != 0 ) {
        fail = b64 != (ret64 / a64);
    }
    if ( fail )
    {
        fprintf (stderr, "silk_SMULL(%lld, %lld) in %s: line %d\n", (long long)a64, (long long)b64, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret64;
}

/* no checking needed for silk_SMULBB */
#undef silk_SMLABB
#define silk_SMLABB(a,b,c) silk_SMLABB_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLABB_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = a32 + (opus_int32)((opus_int16)b32) * (opus_int32)((opus_int16)c32);
    if ( (opus_int64)ret != (opus_int64)a32 + (opus_int64)b32 * (opus_int16)c32 )
    {
        fprintf (stderr, "silk_SMLABB(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

/* no checking needed for silk_SMULBT */
#undef silk_SMLABT
#define silk_SMLABT(a,b,c) silk_SMLABT_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLABT_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = a32 + ((opus_int32)((opus_int16)b32)) * (c32 >> 16);
    if ( (opus_int64)ret != (opus_int64)a32 + (opus_int64)b32 * (c32 >> 16) )
    {
        fprintf (stderr, "silk_SMLABT(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

/* no checking needed for silk_SMULTT */
#undef silk_SMLATT
#define silk_SMLATT(a,b,c) silk_SMLATT_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLATT_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret;
    ret = a32 + (b32 >> 16) * (c32 >> 16);
    if ( (opus_int64)ret != (opus_int64)a32 + (b32 >> 16) * (c32 >> 16) )
    {
        fprintf (stderr, "silk_SMLATT(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_SMULWW
#define silk_SMULWW(a,b) silk_SMULWW_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMULWW_(opus_int32 a32, opus_int32 b32, char *file, int line){
    opus_int32 ret, tmp1, tmp2;
    opus_int64 ret64;
    int        fail = 0;

    ret  = silk_SMULWB( a32, b32 );
    tmp1 = silk_RSHIFT_ROUND( b32, 16 );
    tmp2 = silk_MUL( a32, tmp1 );

    fail |= (opus_int64)tmp2 != (opus_int64) a32 * (opus_int64) tmp1;

    tmp1 = ret;
    ret  = silk_ADD32( tmp1, tmp2 );
    fail |= silk_ADD32( tmp1, tmp2 ) != silk_ADD_SAT32( tmp1, tmp2 );

    ret64 = silk_RSHIFT64( silk_SMULL( a32, b32 ), 16 );
    fail |= (opus_int64)ret != ret64;

    if ( fail )
    {
        fprintf (stderr, "silk_SMULWW(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }

    return ret;
}

#undef silk_SMLAWW
#define silk_SMLAWW(a,b,c) silk_SMLAWW_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SMLAWW_(opus_int32 a32, opus_int32 b32, opus_int32 c32, char *file, int line){
    opus_int32 ret, tmp;

    tmp = silk_SMULWW( b32, c32 );
    ret = silk_ADD32( a32, tmp );
    if ( ret != silk_ADD_SAT32( a32, tmp ) )
    {
        fprintf (stderr, "silk_SMLAWW(%d, %d, %d) in %s: line %d\n", a32, b32, c32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

/* no checking needed for silk_SMULL
   no checking needed for silk_SMLAL
   no checking needed for silk_SMLALBB
   no checking needed for SigProcFIX_CLZ16
   no checking needed for SigProcFIX_CLZ32*/

#undef silk_DIV32
#define silk_DIV32(a,b) silk_DIV32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_DIV32_(opus_int32 a32, opus_int32 b32, char *file, int line){
    if ( b32 == 0 )
    {
        fprintf (stderr, "silk_DIV32(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a32 / b32;
}

#undef silk_DIV32_16
#define silk_DIV32_16(a,b) silk_DIV32_16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_DIV32_16_(opus_int32 a32, opus_int32 b32, char *file, int line){
    int fail = 0;
    fail |= b32 == 0;
    fail |= b32 > silk_int16_MAX;
    fail |= b32 < silk_int16_MIN;
    if ( fail )
    {
        fprintf (stderr, "silk_DIV32_16(%d, %d) in %s: line %d\n", a32, b32, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a32 / b32;
}

/* no checking needed for silk_SAT8
   no checking needed for silk_SAT16
   no checking needed for silk_SAT32
   no checking needed for silk_POS_SAT32
   no checking needed for silk_ADD_POS_SAT8
   no checking needed for silk_ADD_POS_SAT16
   no checking needed for silk_ADD_POS_SAT32 */

#undef silk_LSHIFT8
#define silk_LSHIFT8(a,b) silk_LSHIFT8_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int8 silk_LSHIFT8_(opus_int8 a, opus_int32 shift, char *file, int line){
    opus_int8 ret;
    int       fail = 0;
    ret = (opus_int8)((opus_uint8)a << shift);
    fail |= shift < 0;
    fail |= shift >= 8;
    fail |= (opus_int64)ret != (opus_int64)(((opus_uint64)a) << shift);
    if ( fail )
    {
        fprintf (stderr, "silk_LSHIFT8(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_LSHIFT16
#define silk_LSHIFT16(a,b) silk_LSHIFT16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_LSHIFT16_(opus_int16 a, opus_int32 shift, char *file, int line){
    opus_int16 ret;
    int        fail = 0;
    ret = (opus_int16)((opus_uint16)a << shift);
    fail |= shift < 0;
    fail |= shift >= 16;
    fail |= (opus_int64)ret != (opus_int64)(((opus_uint64)a) << shift);
    if ( fail )
    {
        fprintf (stderr, "silk_LSHIFT16(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_LSHIFT32
#define silk_LSHIFT32(a,b) silk_LSHIFT32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_LSHIFT32_(opus_int32 a, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    int        fail = 0;
    ret = (opus_int32)((opus_uint32)a << shift);
    fail |= shift < 0;
    fail |= shift >= 32;
    fail |= (opus_int64)ret != (opus_int64)(((opus_uint64)a) << shift);
    if ( fail )
    {
        fprintf (stderr, "silk_LSHIFT32(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_LSHIFT64
#define silk_LSHIFT64(a,b) silk_LSHIFT64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_LSHIFT64_(opus_int64 a, opus_int shift, char *file, int line){
    opus_int64 ret;
    int        fail = 0;
    ret = (opus_int64)((opus_uint64)a << shift);
    fail |= shift < 0;
    fail |= shift >= 64;
    fail |= (ret>>shift) != ((opus_int64)a);
    if ( fail )
    {
        fprintf (stderr, "silk_LSHIFT64(%lld, %d) in %s: line %d\n", (long long)a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_LSHIFT_ovflw
#define silk_LSHIFT_ovflw(a,b) silk_LSHIFT_ovflw_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_LSHIFT_ovflw_(opus_int32 a, opus_int32 shift, char *file, int line){
    if ( (shift < 0) || (shift >= 32) ) /* no check for overflow */
    {
        fprintf (stderr, "silk_LSHIFT_ovflw(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a << shift;
}

#undef silk_LSHIFT_uint
#define silk_LSHIFT_uint(a,b) silk_LSHIFT_uint_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_uint32 silk_LSHIFT_uint_(opus_uint32 a, opus_int32 shift, char *file, int line){
    opus_uint32 ret;
    ret = a << shift;
    if ( (shift < 0) || ((opus_int64)ret != ((opus_int64)a) << shift))
    {
        fprintf (stderr, "silk_LSHIFT_uint(%u, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_RSHIFT8
#define silk_RSHITF8(a,b) silk_RSHIFT8_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int8 silk_RSHIFT8_(opus_int8 a, opus_int32 shift, char *file, int line){
    if ( (shift < 0) || (shift>=8) )
    {
        fprintf (stderr, "silk_RSHITF8(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a >> shift;
}

#undef silk_RSHIFT16
#define silk_RSHITF16(a,b) silk_RSHIFT16_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_RSHIFT16_(opus_int16 a, opus_int32 shift, char *file, int line){
    if ( (shift < 0) || (shift>=16) )
    {
        fprintf (stderr, "silk_RSHITF16(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a >> shift;
}

#undef silk_RSHIFT32
#define silk_RSHIFT32(a,b) silk_RSHIFT32_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_RSHIFT32_(opus_int32 a, opus_int32 shift, char *file, int line){
    if ( (shift < 0) || (shift>=32) )
    {
        fprintf (stderr, "silk_RSHITF32(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a >> shift;
}

#undef silk_RSHIFT64
#define silk_RSHIFT64(a,b) silk_RSHIFT64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_RSHIFT64_(opus_int64 a, opus_int64 shift, char *file, int line){
    if ( (shift < 0) || (shift>=64) )
    {
        fprintf (stderr, "silk_RSHITF64(%lld, %lld) in %s: line %d\n", (long long)a, (long long)shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a >> shift;
}

#undef silk_RSHIFT_uint
#define silk_RSHIFT_uint(a,b) silk_RSHIFT_uint_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_uint32 silk_RSHIFT_uint_(opus_uint32 a, opus_int32 shift, char *file, int line){
    if ( (shift < 0) || (shift>32) )
    {
        fprintf (stderr, "silk_RSHIFT_uint(%u, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return a >> shift;
}

#undef silk_ADD_LSHIFT
#define silk_ADD_LSHIFT(a,b,c) silk_ADD_LSHIFT_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE int silk_ADD_LSHIFT_(int a, int b, int shift, char *file, int line){
    opus_int16 ret;
    ret = a + (opus_int16)((opus_uint16)b << shift);
    if ( (shift < 0) || (shift>15) || ((opus_int64)ret != (opus_int64)a + (opus_int64)(((opus_uint64)b) << shift)) )
    {
        fprintf (stderr, "silk_ADD_LSHIFT(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift >= 0 */
}

#undef silk_ADD_LSHIFT32
#define silk_ADD_LSHIFT32(a,b,c) silk_ADD_LSHIFT32_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_ADD_LSHIFT32_(opus_int32 a, opus_int32 b, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    ret = silk_ADD32_ovflw(a, (opus_int32)((opus_uint32)b << shift));
    if ( (shift < 0) || (shift>31) || ((opus_int64)ret != (opus_int64)a + (opus_int64)(((opus_uint64)b) << shift)) )
    {
        fprintf (stderr, "silk_ADD_LSHIFT32(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift >= 0 */
}

#undef silk_ADD_LSHIFT_uint
#define silk_ADD_LSHIFT_uint(a,b,c) silk_ADD_LSHIFT_uint_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_uint32 silk_ADD_LSHIFT_uint_(opus_uint32 a, opus_uint32 b, opus_int32 shift, char *file, int line){
    opus_uint32 ret;
    ret = a + (b << shift);
    if ( (shift < 0) || (shift>32) || ((opus_int64)ret != (opus_int64)a + (((opus_int64)b) << shift)) )
    {
        fprintf (stderr, "silk_ADD_LSHIFT_uint(%u, %u, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift >= 0 */
}

#undef silk_ADD_RSHIFT
#define silk_ADD_RSHIFT(a,b,c) silk_ADD_RSHIFT_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE int silk_ADD_RSHIFT_(int a, int b, int shift, char *file, int line){
    opus_int16 ret;
    ret = a + (b >> shift);
    if ( (shift < 0) || (shift>15) || ((opus_int64)ret != (opus_int64)a + (((opus_int64)b) >> shift)) )
    {
        fprintf (stderr, "silk_ADD_RSHIFT(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift  > 0 */
}

#undef silk_ADD_RSHIFT32
#define silk_ADD_RSHIFT32(a,b,c) silk_ADD_RSHIFT32_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_ADD_RSHIFT32_(opus_int32 a, opus_int32 b, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    ret = silk_ADD32_ovflw(a, (b >> shift));
    if ( (shift < 0) || (shift>31) || ((opus_int64)ret != (opus_int64)a + (((opus_int64)b) >> shift)) )
    {
        fprintf (stderr, "silk_ADD_RSHIFT32(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift  > 0 */
}

#undef silk_ADD_RSHIFT_uint
#define silk_ADD_RSHIFT_uint(a,b,c) silk_ADD_RSHIFT_uint_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_uint32 silk_ADD_RSHIFT_uint_(opus_uint32 a, opus_uint32 b, opus_int32 shift, char *file, int line){
    opus_uint32 ret;
    ret = a + (b >> shift);
    if ( (shift < 0) || (shift>32) || ((opus_int64)ret != (opus_int64)a + (((opus_int64)b) >> shift)) )
    {
        fprintf (stderr, "silk_ADD_RSHIFT_uint(%u, %u, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift  > 0 */
}

#undef silk_SUB_LSHIFT32
#define silk_SUB_LSHIFT32(a,b,c) silk_SUB_LSHIFT32_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SUB_LSHIFT32_(opus_int32 a, opus_int32 b, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    ret = silk_SUB32_ovflw(a, (opus_int32)((opus_uint32)b << shift));
    if ( (shift < 0) || (shift>31) || ((opus_int64)ret != (opus_int64)a - (opus_int64)(((opus_uint64)b) << shift)) )
    {
        fprintf (stderr, "silk_SUB_LSHIFT32(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift >= 0 */
}

#undef silk_SUB_RSHIFT32
#define silk_SUB_RSHIFT32(a,b,c) silk_SUB_RSHIFT32_((a), (b), (c), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_SUB_RSHIFT32_(opus_int32 a, opus_int32 b, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    ret = silk_SUB32_ovflw(a, (b >> shift));
    if ( (shift < 0) || (shift>31) || ((opus_int64)ret != (opus_int64)a - (((opus_int64)b) >> shift)) )
    {
        fprintf (stderr, "silk_SUB_RSHIFT32(%d, %d, %d) in %s: line %d\n", a, b, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;                /* shift  > 0 */
}

#undef silk_RSHIFT_ROUND
#define silk_RSHIFT_ROUND(a,b) silk_RSHIFT_ROUND_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_RSHIFT_ROUND_(opus_int32 a, opus_int32 shift, char *file, int line){
    opus_int32 ret;
    ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
    /* the macro definition can't handle a shift of zero */
    if ( (shift <= 0) || (shift>31) || ((opus_int64)ret != ((opus_int64)a + ((opus_int64)1 << (shift - 1))) >> shift) )
    {
        fprintf (stderr, "silk_RSHIFT_ROUND(%d, %d) in %s: line %d\n", a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return ret;
}

#undef silk_RSHIFT_ROUND64
#define silk_RSHIFT_ROUND64(a,b) silk_RSHIFT_ROUND64_((a), (b), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_RSHIFT_ROUND64_(opus_int64 a, opus_int32 shift, char *file, int line){
    opus_int64 ret;
    /* the macro definition can't handle a shift of zero */
    if ( (shift <= 0) || (shift>=64) )
    {
        fprintf (stderr, "silk_RSHIFT_ROUND64(%lld, %d) in %s: line %d\n", (long long)a, shift, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
    return ret;
}

/* silk_abs is used on floats also, so doesn't work... */
/*#undef silk_abs
static OPUS_INLINE opus_int32 silk_abs(opus_int32 a){
    silk_assert(a != 0x80000000);
    return (((a) >  0)  ? (a) : -(a));            // Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN
}*/

#undef silk_abs_int64
#define silk_abs_int64(a) silk_abs_int64_((a), __FILE__, __LINE__)
static OPUS_INLINE opus_int64 silk_abs_int64_(opus_int64 a, char *file, int line){
    if ( a == silk_int64_MIN )
    {
        fprintf (stderr, "silk_abs_int64(%lld) in %s: line %d\n", (long long)a, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return (((a) >  0)  ? (a) : -(a));            /* Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN */
}

#undef silk_abs_int32
#define silk_abs_int32(a) silk_abs_int32_((a), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_abs_int32_(opus_int32 a, char *file, int line){
    if ( a == silk_int32_MIN )
    {
        fprintf (stderr, "silk_abs_int32(%d) in %s: line %d\n", a, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return silk_abs(a);
}

#undef silk_CHECK_FIT8
#define silk_CHECK_FIT8(a) silk_CHECK_FIT8_((a), __FILE__, __LINE__)
static OPUS_INLINE opus_int8 silk_CHECK_FIT8_( opus_int64 a, char *file, int line ){
    opus_int8 ret;
    ret = (opus_int8)a;
    if ( (opus_int64)ret != a )
    {
        fprintf (stderr, "silk_CHECK_FIT8(%lld) in %s: line %d\n", (long long)a, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return( ret );
}

#undef silk_CHECK_FIT16
#define silk_CHECK_FIT16(a) silk_CHECK_FIT16_((a), __FILE__, __LINE__)
static OPUS_INLINE opus_int16 silk_CHECK_FIT16_( opus_int64 a, char *file, int line ){
    opus_int16 ret;
    ret = (opus_int16)a;
    if ( (opus_int64)ret != a )
    {
        fprintf (stderr, "silk_CHECK_FIT16(%lld) in %s: line %d\n", (long long)a, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return( ret );
}

#undef silk_CHECK_FIT32
#define silk_CHECK_FIT32(a) silk_CHECK_FIT32_((a), __FILE__, __LINE__)
static OPUS_INLINE opus_int32 silk_CHECK_FIT32_( opus_int64 a, char *file, int line ){
    opus_int32 ret;
    ret = (opus_int32)a;
    if ( (opus_int64)ret != a )
    {
        fprintf (stderr, "silk_CHECK_FIT32(%lld) in %s: line %d\n", (long long)a, file, line);
#ifdef FIXED_DEBUG_ASSERT
        silk_assert( 0 );
#endif
    }
    return( ret );
}

/* no checking for silk_NSHIFT_MUL_32_32
   no checking for silk_NSHIFT_MUL_16_16
   no checking needed for silk_min
   no checking needed for silk_max
   no checking needed for silk_sign
*/

#endif
#endif /* MACRO_DEBUG_H */
