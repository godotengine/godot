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


#ifndef SILK_MACROS_MIPSR1_H__
#define SILK_MACROS_MIPSR1_H__

#define mips_clz(x) __builtin_clz(x)

#undef silk_SMULWB
static inline int silk_SMULWB(int a, int b)
{
    long long ac;
    int c;

    ac = __builtin_mips_mult(a, (opus_int32)(opus_int16)b);
    c = __builtin_mips_extr_w(ac, 16);

    return c;
}

#undef silk_SMLAWB
#define silk_SMLAWB(a32, b32, c32)       ((a32) + silk_SMULWB(b32, c32))

#undef silk_SMULWW
static inline int silk_SMULWW(int a, int b)
{
    long long ac;
    int c;

    ac = __builtin_mips_mult(a, b);
    c = __builtin_mips_extr_w(ac, 16);

    return c;
}

#undef silk_SMLAWW
static inline int silk_SMLAWW(int a, int b, int c)
{
    long long ac;
    int res;

    ac = __builtin_mips_mult(b, c);
    res = __builtin_mips_extr_w(ac, 16);
    res += a;

    return res;
}

#define OVERRIDE_silk_CLZ16
static inline opus_int32 silk_CLZ16(opus_int16 in16)
{
    int re32;
    opus_int32 in32 = (opus_int32 )in16;
    re32 = mips_clz(in32);
    re32-=16;
    return re32;
}

#define OVERRIDE_silk_CLZ32
static inline opus_int32 silk_CLZ32(opus_int32 in32)
{
    int re32;
    re32 = mips_clz(in32);
    return re32;
}

#endif /* SILK_MACROS_MIPSR1_H__ */
