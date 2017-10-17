/* Copyright (C) 2007-2009 Xiph.Org Foundation
   Copyright (C) 2003-2008 Jean-Marc Valin
   Copyright (C) 2007-2008 CSIRO */
/**
   @file fixed_generic.h
   @brief Generic fixed-point operations
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef FIXED_GENERIC_H
#define FIXED_GENERIC_H

/** Multiply a 16-bit signed value by a 16-bit unsigned value. The result is a 32-bit signed value */
#define MULT16_16SU(a,b) ((opus_val32)(opus_val16)(a)*(opus_val32)(opus_uint16)(b))

/** 16x32 multiplication, followed by a 16-bit shift right. Results fits in 32 bits */
#if OPUS_FAST_INT64
#define MULT16_32_Q16(a,b) ((opus_val32)SHR((opus_int64)((opus_val16)(a))*(b),16))
#else
#define MULT16_32_Q16(a,b) ADD32(MULT16_16((a),SHR((b),16)), SHR(MULT16_16SU((a),((b)&0x0000ffff)),16))
#endif

/** 16x32 multiplication, followed by a 16-bit shift right (round-to-nearest). Results fits in 32 bits */
#if OPUS_FAST_INT64
#define MULT16_32_P16(a,b) ((opus_val32)PSHR((opus_int64)((opus_val16)(a))*(b),16))
#else
#define MULT16_32_P16(a,b) ADD32(MULT16_16((a),SHR((b),16)), PSHR(MULT16_16SU((a),((b)&0x0000ffff)),16))
#endif

/** 16x32 multiplication, followed by a 15-bit shift right. Results fits in 32 bits */
#if OPUS_FAST_INT64
#define MULT16_32_Q15(a,b) ((opus_val32)SHR((opus_int64)((opus_val16)(a))*(b),15))
#else
#define MULT16_32_Q15(a,b) ADD32(SHL(MULT16_16((a),SHR((b),16)),1), SHR(MULT16_16SU((a),((b)&0x0000ffff)),15))
#endif

/** 32x32 multiplication, followed by a 31-bit shift right. Results fits in 32 bits */
#if OPUS_FAST_INT64
#define MULT32_32_Q31(a,b) ((opus_val32)SHR((opus_int64)(a)*(opus_int64)(b),31))
#else
#define MULT32_32_Q31(a,b) ADD32(ADD32(SHL(MULT16_16(SHR((a),16),SHR((b),16)),1), SHR(MULT16_16SU(SHR((a),16),((b)&0x0000ffff)),15)), SHR(MULT16_16SU(SHR((b),16),((a)&0x0000ffff)),15))
#endif

/** Compile-time conversion of float constant to 16-bit value */
#define QCONST16(x,bits) ((opus_val16)(.5+(x)*(((opus_val32)1)<<(bits))))

/** Compile-time conversion of float constant to 32-bit value */
#define QCONST32(x,bits) ((opus_val32)(.5+(x)*(((opus_val32)1)<<(bits))))

/** Negate a 16-bit value */
#define NEG16(x) (-(x))
/** Negate a 32-bit value */
#define NEG32(x) (-(x))

/** Change a 32-bit value into a 16-bit value. The value is assumed to fit in 16-bit, otherwise the result is undefined */
#define EXTRACT16(x) ((opus_val16)(x))
/** Change a 16-bit value into a 32-bit value */
#define EXTEND32(x) ((opus_val32)(x))

/** Arithmetic shift-right of a 16-bit value */
#define SHR16(a,shift) ((a) >> (shift))
/** Arithmetic shift-left of a 16-bit value */
#define SHL16(a,shift) ((opus_int16)((opus_uint16)(a)<<(shift)))
/** Arithmetic shift-right of a 32-bit value */
#define SHR32(a,shift) ((a) >> (shift))
/** Arithmetic shift-left of a 32-bit value */
#define SHL32(a,shift) ((opus_int32)((opus_uint32)(a)<<(shift)))

/** 32-bit arithmetic shift right with rounding-to-nearest instead of rounding down */
#define PSHR32(a,shift) (SHR32((a)+((EXTEND32(1)<<((shift))>>1)),shift))
/** 32-bit arithmetic shift right where the argument can be negative */
#define VSHR32(a, shift) (((shift)>0) ? SHR32(a, shift) : SHL32(a, -(shift)))

/** "RAW" macros, should not be used outside of this header file */
#define SHR(a,shift) ((a) >> (shift))
#define SHL(a,shift) SHL32(a,shift)
#define PSHR(a,shift) (SHR((a)+((EXTEND32(1)<<((shift))>>1)),shift))
#define SATURATE(x,a) (((x)>(a) ? (a) : (x)<-(a) ? -(a) : (x)))

#define SATURATE16(x) (EXTRACT16((x)>32767 ? 32767 : (x)<-32768 ? -32768 : (x)))

/** Shift by a and round-to-neareast 32-bit value. Result is a 16-bit value */
#define ROUND16(x,a) (EXTRACT16(PSHR32((x),(a))))
/** Divide by two */
#define HALF16(x)  (SHR16(x,1))
#define HALF32(x)  (SHR32(x,1))

/** Add two 16-bit values */
#define ADD16(a,b) ((opus_val16)((opus_val16)(a)+(opus_val16)(b)))
/** Subtract two 16-bit values */
#define SUB16(a,b) ((opus_val16)(a)-(opus_val16)(b))
/** Add two 32-bit values */
#define ADD32(a,b) ((opus_val32)(a)+(opus_val32)(b))
/** Subtract two 32-bit values */
#define SUB32(a,b) ((opus_val32)(a)-(opus_val32)(b))

/** 16x16 multiplication where the result fits in 16 bits */
#define MULT16_16_16(a,b)     ((((opus_val16)(a))*((opus_val16)(b))))

/* (opus_val32)(opus_val16) gives TI compiler a hint that it's 16x16->32 multiply */
/** 16x16 multiplication where the result fits in 32 bits */
#define MULT16_16(a,b)     (((opus_val32)(opus_val16)(a))*((opus_val32)(opus_val16)(b)))

/** 16x16 multiply-add where the result fits in 32 bits */
#define MAC16_16(c,a,b) (ADD32((c),MULT16_16((a),(b))))
/** 16x32 multiply, followed by a 15-bit shift right and 32-bit add.
    b must fit in 31 bits.
    Result fits in 32 bits. */
#define MAC16_32_Q15(c,a,b) ADD32((c),ADD32(MULT16_16((a),SHR((b),15)), SHR(MULT16_16((a),((b)&0x00007fff)),15)))

/** 16x32 multiplication, followed by a 16-bit shift right and 32-bit add.
    Results fits in 32 bits */
#define MAC16_32_Q16(c,a,b) ADD32((c),ADD32(MULT16_16((a),SHR((b),16)), SHR(MULT16_16SU((a),((b)&0x0000ffff)),16)))

#define MULT16_16_Q11_32(a,b) (SHR(MULT16_16((a),(b)),11))
#define MULT16_16_Q11(a,b) (SHR(MULT16_16((a),(b)),11))
#define MULT16_16_Q13(a,b) (SHR(MULT16_16((a),(b)),13))
#define MULT16_16_Q14(a,b) (SHR(MULT16_16((a),(b)),14))
#define MULT16_16_Q15(a,b) (SHR(MULT16_16((a),(b)),15))

#define MULT16_16_P13(a,b) (SHR(ADD32(4096,MULT16_16((a),(b))),13))
#define MULT16_16_P14(a,b) (SHR(ADD32(8192,MULT16_16((a),(b))),14))
#define MULT16_16_P15(a,b) (SHR(ADD32(16384,MULT16_16((a),(b))),15))

/** Divide a 32-bit value by a 16-bit value. Result fits in 16 bits */
#define DIV32_16(a,b) ((opus_val16)(((opus_val32)(a))/((opus_val16)(b))))

/** Divide a 32-bit value by a 32-bit value. Result fits in 32 bits */
#define DIV32(a,b) (((opus_val32)(a))/((opus_val32)(b)))

#if defined(MIPSr1_ASM)
#include "mips/fixed_generic_mipsr1.h"
#endif

static OPUS_INLINE opus_val16 SIG2WORD16_generic(celt_sig x)
{
   x = PSHR32(x, SIG_SHIFT);
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return EXTRACT16(x);
}
#define SIG2WORD16(x) (SIG2WORD16_generic(x))

#endif
