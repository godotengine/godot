/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
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

#ifndef SIGPROCFIX_API_MACROCOUNT_H
#define SIGPROCFIX_API_MACROCOUNT_H
#include <stdio.h>

#ifdef    silk_MACRO_COUNT
#define varDefine opus_int64 ops_count = 0;

extern opus_int64 ops_count;

static OPUS_INLINE opus_int64 silk_SaveCount(){
    return(ops_count);
}

static OPUS_INLINE opus_int64 silk_SaveResetCount(){
    opus_int64 ret;

    ret = ops_count;
    ops_count = 0;
    return(ret);
}

static OPUS_INLINE silk_PrintCount(){
    printf("ops_count = %d \n ", (opus_int32)ops_count);
}

#undef silk_MUL
static OPUS_INLINE opus_int32 silk_MUL(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    ops_count += 4;
    ret = a32 * b32;
    return ret;
}

#undef silk_MUL_uint
static OPUS_INLINE opus_uint32 silk_MUL_uint(opus_uint32 a32, opus_uint32 b32){
    opus_uint32 ret;
    ops_count += 4;
    ret = a32 * b32;
    return ret;
}
#undef silk_MLA
static OPUS_INLINE opus_int32 silk_MLA(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 4;
    ret = a32 + b32 * c32;
    return ret;
}

#undef silk_MLA_uint
static OPUS_INLINE opus_int32 silk_MLA_uint(opus_uint32 a32, opus_uint32 b32, opus_uint32 c32){
    opus_uint32 ret;
    ops_count += 4;
    ret = a32 + b32 * c32;
    return ret;
}

#undef silk_SMULWB
static OPUS_INLINE opus_int32 silk_SMULWB(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    ops_count += 5;
    ret = (a32 >> 16) * (opus_int32)((opus_int16)b32) + (((a32 & 0x0000FFFF) * (opus_int32)((opus_int16)b32)) >> 16);
    return ret;
}
#undef    silk_SMLAWB
static OPUS_INLINE opus_int32 silk_SMLAWB(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 5;
    ret = ((a32) + ((((b32) >> 16) * (opus_int32)((opus_int16)(c32))) + ((((b32) & 0x0000FFFF) * (opus_int32)((opus_int16)(c32))) >> 16)));
    return ret;
}

#undef silk_SMULWT
static OPUS_INLINE opus_int32 silk_SMULWT(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    ops_count += 4;
    ret = (a32 >> 16) * (b32 >> 16) + (((a32 & 0x0000FFFF) * (b32 >> 16)) >> 16);
    return ret;
}
#undef silk_SMLAWT
static OPUS_INLINE opus_int32 silk_SMLAWT(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 4;
    ret = a32 + ((b32 >> 16) * (c32 >> 16)) + (((b32 & 0x0000FFFF) * ((c32 >> 16)) >> 16));
    return ret;
}

#undef silk_SMULBB
static OPUS_INLINE opus_int32 silk_SMULBB(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    ops_count += 1;
    ret = (opus_int32)((opus_int16)a32) * (opus_int32)((opus_int16)b32);
    return ret;
}
#undef silk_SMLABB
static OPUS_INLINE opus_int32 silk_SMLABB(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 1;
    ret = a32 + (opus_int32)((opus_int16)b32) * (opus_int32)((opus_int16)c32);
    return ret;
}

#undef silk_SMULBT
static OPUS_INLINE opus_int32 silk_SMULBT(opus_int32 a32, opus_int32 b32 ){
    opus_int32 ret;
    ops_count += 4;
    ret = ((opus_int32)((opus_int16)a32)) * (b32 >> 16);
    return ret;
}

#undef silk_SMLABT
static OPUS_INLINE opus_int32 silk_SMLABT(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 1;
    ret = a32 + ((opus_int32)((opus_int16)b32)) * (c32 >> 16);
    return ret;
}

#undef silk_SMULTT
static OPUS_INLINE opus_int32 silk_SMULTT(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    ops_count += 1;
    ret = (a32 >> 16) * (b32 >> 16);
    return ret;
}

#undef    silk_SMLATT
static OPUS_INLINE opus_int32 silk_SMLATT(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    ops_count += 1;
    ret = a32 + (b32 >> 16) * (c32 >> 16);
    return ret;
}


/* multiply-accumulate macros that allow overflow in the addition (ie, no asserts in debug mode)*/
#undef    silk_MLA_ovflw
#define silk_MLA_ovflw silk_MLA

#undef silk_SMLABB_ovflw
#define silk_SMLABB_ovflw silk_SMLABB

#undef silk_SMLABT_ovflw
#define silk_SMLABT_ovflw silk_SMLABT

#undef silk_SMLATT_ovflw
#define silk_SMLATT_ovflw silk_SMLATT

#undef silk_SMLAWB_ovflw
#define silk_SMLAWB_ovflw silk_SMLAWB

#undef silk_SMLAWT_ovflw
#define silk_SMLAWT_ovflw silk_SMLAWT

#undef silk_SMULL
static OPUS_INLINE opus_int64 silk_SMULL(opus_int32 a32, opus_int32 b32){
    opus_int64 ret;
    ops_count += 8;
    ret = ((opus_int64)(a32) * /*(opus_int64)*/(b32));
    return ret;
}

#undef    silk_SMLAL
static OPUS_INLINE opus_int64 silk_SMLAL(opus_int64 a64, opus_int32 b32, opus_int32 c32){
    opus_int64 ret;
    ops_count += 8;
    ret = a64 + ((opus_int64)(b32) * /*(opus_int64)*/(c32));
    return ret;
}
#undef    silk_SMLALBB
static OPUS_INLINE opus_int64 silk_SMLALBB(opus_int64 a64, opus_int16 b16, opus_int16 c16){
    opus_int64 ret;
    ops_count += 4;
    ret = a64 + ((opus_int64)(b16) * /*(opus_int64)*/(c16));
    return ret;
}

#undef    SigProcFIX_CLZ16
static OPUS_INLINE opus_int32 SigProcFIX_CLZ16(opus_int16 in16)
{
    opus_int32 out32 = 0;
    ops_count += 10;
    if( in16 == 0 ) {
        return 16;
    }
    /* test nibbles */
    if( in16 & 0xFF00 ) {
        if( in16 & 0xF000 ) {
            in16 >>= 12;
        } else {
            out32 += 4;
            in16 >>= 8;
        }
    } else {
        if( in16 & 0xFFF0 ) {
            out32 += 8;
            in16 >>= 4;
        } else {
            out32 += 12;
        }
    }
    /* test bits and return */
    if( in16 & 0xC ) {
        if( in16 & 0x8 )
            return out32 + 0;
        else
            return out32 + 1;
    } else {
        if( in16 & 0xE )
            return out32 + 2;
        else
            return out32 + 3;
    }
}

#undef SigProcFIX_CLZ32
static OPUS_INLINE opus_int32 SigProcFIX_CLZ32(opus_int32 in32)
{
    /* test highest 16 bits and convert to opus_int16 */
    ops_count += 2;
    if( in32 & 0xFFFF0000 ) {
        return SigProcFIX_CLZ16((opus_int16)(in32 >> 16));
    } else {
        return SigProcFIX_CLZ16((opus_int16)in32) + 16;
    }
}

#undef silk_DIV32
static OPUS_INLINE opus_int32 silk_DIV32(opus_int32 a32, opus_int32 b32){
    ops_count += 64;
    return a32 / b32;
}

#undef silk_DIV32_16
static OPUS_INLINE opus_int32 silk_DIV32_16(opus_int32 a32, opus_int32 b32){
    ops_count += 32;
    return a32 / b32;
}

#undef silk_SAT8
static OPUS_INLINE opus_int8 silk_SAT8(opus_int64 a){
    opus_int8 tmp;
    ops_count += 1;
    tmp = (opus_int8)((a) > silk_int8_MAX ? silk_int8_MAX  : \
                    ((a) < silk_int8_MIN ? silk_int8_MIN  : (a)));
    return(tmp);
}

#undef silk_SAT16
static OPUS_INLINE opus_int16 silk_SAT16(opus_int64 a){
    opus_int16 tmp;
    ops_count += 1;
    tmp = (opus_int16)((a) > silk_int16_MAX ? silk_int16_MAX  : \
                     ((a) < silk_int16_MIN ? silk_int16_MIN  : (a)));
    return(tmp);
}
#undef silk_SAT32
static OPUS_INLINE opus_int32 silk_SAT32(opus_int64 a){
    opus_int32 tmp;
    ops_count += 1;
    tmp = (opus_int32)((a) > silk_int32_MAX ? silk_int32_MAX  : \
                     ((a) < silk_int32_MIN ? silk_int32_MIN  : (a)));
    return(tmp);
}
#undef silk_POS_SAT32
static OPUS_INLINE opus_int32 silk_POS_SAT32(opus_int64 a){
    opus_int32 tmp;
    ops_count += 1;
    tmp = (opus_int32)((a) > silk_int32_MAX ? silk_int32_MAX : (a));
    return(tmp);
}

#undef silk_ADD_POS_SAT8
static OPUS_INLINE opus_int8 silk_ADD_POS_SAT8(opus_int64 a, opus_int64 b){
    opus_int8 tmp;
    ops_count += 1;
    tmp = (opus_int8)((((a)+(b)) & 0x80) ? silk_int8_MAX  : ((a)+(b)));
    return(tmp);
}
#undef silk_ADD_POS_SAT16
static OPUS_INLINE opus_int16 silk_ADD_POS_SAT16(opus_int64 a, opus_int64 b){
    opus_int16 tmp;
    ops_count += 1;
    tmp = (opus_int16)((((a)+(b)) & 0x8000) ? silk_int16_MAX : ((a)+(b)));
    return(tmp);
}

#undef silk_ADD_POS_SAT32
static OPUS_INLINE opus_int32 silk_ADD_POS_SAT32(opus_int64 a, opus_int64 b){
    opus_int32 tmp;
    ops_count += 1;
    tmp = (opus_int32)((((a)+(b)) & 0x80000000) ? silk_int32_MAX : ((a)+(b)));
    return(tmp);
}

#undef    silk_LSHIFT8
static OPUS_INLINE opus_int8 silk_LSHIFT8(opus_int8 a, opus_int32 shift){
    opus_int8 ret;
    ops_count += 1;
    ret = a << shift;
    return ret;
}
#undef    silk_LSHIFT16
static OPUS_INLINE opus_int16 silk_LSHIFT16(opus_int16 a, opus_int32 shift){
    opus_int16 ret;
    ops_count += 1;
    ret = a << shift;
    return ret;
}
#undef    silk_LSHIFT32
static OPUS_INLINE opus_int32 silk_LSHIFT32(opus_int32 a, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a << shift;
    return ret;
}
#undef    silk_LSHIFT64
static OPUS_INLINE opus_int64 silk_LSHIFT64(opus_int64 a, opus_int shift){
    ops_count += 1;
    return a << shift;
}

#undef    silk_LSHIFT_ovflw
static OPUS_INLINE opus_int32 silk_LSHIFT_ovflw(opus_int32 a, opus_int32 shift){
    ops_count += 1;
    return a << shift;
}

#undef    silk_LSHIFT_uint
static OPUS_INLINE opus_uint32 silk_LSHIFT_uint(opus_uint32 a, opus_int32 shift){
    opus_uint32 ret;
    ops_count += 1;
    ret = a << shift;
    return ret;
}

#undef    silk_RSHIFT8
static OPUS_INLINE opus_int8 silk_RSHIFT8(opus_int8 a, opus_int32 shift){
    ops_count += 1;
    return a >> shift;
}
#undef    silk_RSHIFT16
static OPUS_INLINE opus_int16 silk_RSHIFT16(opus_int16 a, opus_int32 shift){
    ops_count += 1;
    return a >> shift;
}
#undef    silk_RSHIFT32
static OPUS_INLINE opus_int32 silk_RSHIFT32(opus_int32 a, opus_int32 shift){
    ops_count += 1;
    return a >> shift;
}
#undef    silk_RSHIFT64
static OPUS_INLINE opus_int64 silk_RSHIFT64(opus_int64 a, opus_int64 shift){
    ops_count += 1;
    return a >> shift;
}

#undef    silk_RSHIFT_uint
static OPUS_INLINE opus_uint32 silk_RSHIFT_uint(opus_uint32 a, opus_int32 shift){
    ops_count += 1;
    return a >> shift;
}

#undef    silk_ADD_LSHIFT
static OPUS_INLINE opus_int32 silk_ADD_LSHIFT(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a + (b << shift);
    return ret;                /* shift >= 0*/
}
#undef    silk_ADD_LSHIFT32
static OPUS_INLINE opus_int32 silk_ADD_LSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a + (b << shift);
    return ret;                /* shift >= 0*/
}
#undef    silk_ADD_LSHIFT_uint
static OPUS_INLINE opus_uint32 silk_ADD_LSHIFT_uint(opus_uint32 a, opus_uint32 b, opus_int32 shift){
    opus_uint32 ret;
    ops_count += 1;
    ret = a + (b << shift);
    return ret;                /* shift >= 0*/
}
#undef    silk_ADD_RSHIFT
static OPUS_INLINE opus_int32 silk_ADD_RSHIFT(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a + (b >> shift);
    return ret;                /* shift  > 0*/
}
#undef    silk_ADD_RSHIFT32
static OPUS_INLINE opus_int32 silk_ADD_RSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a + (b >> shift);
    return ret;                /* shift  > 0*/
}
#undef    silk_ADD_RSHIFT_uint
static OPUS_INLINE opus_uint32 silk_ADD_RSHIFT_uint(opus_uint32 a, opus_uint32 b, opus_int32 shift){
    opus_uint32 ret;
    ops_count += 1;
    ret = a + (b >> shift);
    return ret;                /* shift  > 0*/
}
#undef    silk_SUB_LSHIFT32
static OPUS_INLINE opus_int32 silk_SUB_LSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a - (b << shift);
    return ret;                /* shift >= 0*/
}
#undef    silk_SUB_RSHIFT32
static OPUS_INLINE opus_int32 silk_SUB_RSHIFT32(opus_int32 a, opus_int32 b, opus_int32 shift){
    opus_int32 ret;
    ops_count += 1;
    ret = a - (b >> shift);
    return ret;                /* shift  > 0*/
}

#undef    silk_RSHIFT_ROUND
static OPUS_INLINE opus_int32 silk_RSHIFT_ROUND(opus_int32 a, opus_int32 shift){
    opus_int32 ret;
    ops_count += 3;
    ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
    return ret;
}

#undef    silk_RSHIFT_ROUND64
static OPUS_INLINE opus_int64 silk_RSHIFT_ROUND64(opus_int64 a, opus_int32 shift){
    opus_int64 ret;
    ops_count += 6;
    ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
    return ret;
}

#undef    silk_abs_int64
static OPUS_INLINE opus_int64 silk_abs_int64(opus_int64 a){
    ops_count += 1;
    return (((a) >  0)  ? (a) : -(a));            /* Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN*/
}

#undef    silk_abs_int32
static OPUS_INLINE opus_int32 silk_abs_int32(opus_int32 a){
    ops_count += 1;
    return silk_abs(a);
}


#undef silk_min
static silk_min(a, b){
    ops_count += 1;
    return (((a) < (b)) ? (a) :  (b));
}
#undef silk_max
static silk_max(a, b){
    ops_count += 1;
    return (((a) > (b)) ? (a) :  (b));
}
#undef silk_sign
static silk_sign(a){
    ops_count += 1;
    return ((a) > 0 ? 1 : ( (a) < 0 ? -1 : 0 ));
}

#undef    silk_ADD16
static OPUS_INLINE opus_int16 silk_ADD16(opus_int16 a, opus_int16 b){
    opus_int16 ret;
    ops_count += 1;
    ret = a + b;
    return ret;
}

#undef    silk_ADD32
static OPUS_INLINE opus_int32 silk_ADD32(opus_int32 a, opus_int32 b){
    opus_int32 ret;
    ops_count += 1;
    ret = a + b;
    return ret;
}

#undef    silk_ADD64
static OPUS_INLINE opus_int64 silk_ADD64(opus_int64 a, opus_int64 b){
    opus_int64 ret;
    ops_count += 2;
    ret = a + b;
    return ret;
}

#undef    silk_SUB16
static OPUS_INLINE opus_int16 silk_SUB16(opus_int16 a, opus_int16 b){
    opus_int16 ret;
    ops_count += 1;
    ret = a - b;
    return ret;
}

#undef    silk_SUB32
static OPUS_INLINE opus_int32 silk_SUB32(opus_int32 a, opus_int32 b){
    opus_int32 ret;
    ops_count += 1;
    ret = a - b;
    return ret;
}

#undef    silk_SUB64
static OPUS_INLINE opus_int64 silk_SUB64(opus_int64 a, opus_int64 b){
    opus_int64 ret;
    ops_count += 2;
    ret = a - b;
    return ret;
}

#undef silk_ADD_SAT16
static OPUS_INLINE opus_int16 silk_ADD_SAT16( opus_int16 a16, opus_int16 b16 ) {
    opus_int16 res;
    /* Nb will be counted in AKP_add32 and silk_SAT16*/
    res = (opus_int16)silk_SAT16( silk_ADD32( (opus_int32)(a16), (b16) ) );
    return res;
}

#undef silk_ADD_SAT32
static OPUS_INLINE opus_int32 silk_ADD_SAT32(opus_int32 a32, opus_int32 b32){
    opus_int32 res;
    ops_count += 1;
    res =    ((((a32) + (b32)) & 0x80000000) == 0 ?                                    \
            ((((a32) & (b32)) & 0x80000000) != 0 ? silk_int32_MIN : (a32)+(b32)) :    \
            ((((a32) | (b32)) & 0x80000000) == 0 ? silk_int32_MAX : (a32)+(b32)) );
    return res;
}

#undef silk_ADD_SAT64
static OPUS_INLINE opus_int64 silk_ADD_SAT64( opus_int64 a64, opus_int64 b64 ) {
    opus_int64 res;
    ops_count += 1;
    res =    ((((a64) + (b64)) & 0x8000000000000000LL) == 0 ?                                \
            ((((a64) & (b64)) & 0x8000000000000000LL) != 0 ? silk_int64_MIN : (a64)+(b64)) :    \
            ((((a64) | (b64)) & 0x8000000000000000LL) == 0 ? silk_int64_MAX : (a64)+(b64)) );
    return res;
}

#undef silk_SUB_SAT16
static OPUS_INLINE opus_int16 silk_SUB_SAT16( opus_int16 a16, opus_int16 b16 ) {
    opus_int16 res;
    silk_assert(0);
    /* Nb will be counted in sub-macros*/
    res = (opus_int16)silk_SAT16( silk_SUB32( (opus_int32)(a16), (b16) ) );
    return res;
}

#undef silk_SUB_SAT32
static OPUS_INLINE opus_int32 silk_SUB_SAT32( opus_int32 a32, opus_int32 b32 ) {
    opus_int32 res;
    ops_count += 1;
    res =     ((((a32)-(b32)) & 0x80000000) == 0 ?                                            \
            (( (a32) & ((b32)^0x80000000) & 0x80000000) ? silk_int32_MIN : (a32)-(b32)) :    \
            ((((a32)^0x80000000) & (b32)  & 0x80000000) ? silk_int32_MAX : (a32)-(b32)) );
    return res;
}

#undef silk_SUB_SAT64
static OPUS_INLINE opus_int64 silk_SUB_SAT64( opus_int64 a64, opus_int64 b64 ) {
    opus_int64 res;
    ops_count += 1;
    res =    ((((a64)-(b64)) & 0x8000000000000000LL) == 0 ?                                                        \
            (( (a64) & ((b64)^0x8000000000000000LL) & 0x8000000000000000LL) ? silk_int64_MIN : (a64)-(b64)) :    \
            ((((a64)^0x8000000000000000LL) & (b64)  & 0x8000000000000000LL) ? silk_int64_MAX : (a64)-(b64)) );

    return res;
}

#undef    silk_SMULWW
static OPUS_INLINE opus_int32 silk_SMULWW(opus_int32 a32, opus_int32 b32){
    opus_int32 ret;
    /* Nb will be counted in sub-macros*/
    ret = silk_MLA(silk_SMULWB((a32), (b32)), (a32), silk_RSHIFT_ROUND((b32), 16));
    return ret;
}

#undef    silk_SMLAWW
static OPUS_INLINE opus_int32 silk_SMLAWW(opus_int32 a32, opus_int32 b32, opus_int32 c32){
    opus_int32 ret;
    /* Nb will be counted in sub-macros*/
    ret = silk_MLA(silk_SMLAWB((a32), (b32), (c32)), (b32), silk_RSHIFT_ROUND((c32), 16));
    return ret;
}

#undef    silk_min_int
static OPUS_INLINE opus_int silk_min_int(opus_int a, opus_int b)
{
    ops_count += 1;
    return (((a) < (b)) ? (a) : (b));
}

#undef    silk_min_16
static OPUS_INLINE opus_int16 silk_min_16(opus_int16 a, opus_int16 b)
{
    ops_count += 1;
    return (((a) < (b)) ? (a) : (b));
}
#undef    silk_min_32
static OPUS_INLINE opus_int32 silk_min_32(opus_int32 a, opus_int32 b)
{
    ops_count += 1;
    return (((a) < (b)) ? (a) : (b));
}
#undef    silk_min_64
static OPUS_INLINE opus_int64 silk_min_64(opus_int64 a, opus_int64 b)
{
    ops_count += 1;
    return (((a) < (b)) ? (a) : (b));
}

/* silk_min() versions with typecast in the function call */
#undef    silk_max_int
static OPUS_INLINE opus_int silk_max_int(opus_int a, opus_int b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}
#undef    silk_max_16
static OPUS_INLINE opus_int16 silk_max_16(opus_int16 a, opus_int16 b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}
#undef    silk_max_32
static OPUS_INLINE opus_int32 silk_max_32(opus_int32 a, opus_int32 b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}

#undef    silk_max_64
static OPUS_INLINE opus_int64 silk_max_64(opus_int64 a, opus_int64 b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}


#undef silk_LIMIT_int
static OPUS_INLINE opus_int silk_LIMIT_int(opus_int a, opus_int limit1, opus_int limit2)
{
    opus_int ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));

    return(ret);
}

#undef silk_LIMIT_16
static OPUS_INLINE opus_int16 silk_LIMIT_16(opus_int16 a, opus_int16 limit1, opus_int16 limit2)
{
    opus_int16 ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));

return(ret);
}


#undef silk_LIMIT_32
static OPUS_INLINE opus_int32 silk_LIMIT_32(opus_int32 a, opus_int32 limit1, opus_int32 limit2)
{
    opus_int32 ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));
    return(ret);
}

#else
#define varDefine
#define silk_SaveCount()

#endif
#endif

