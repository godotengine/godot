/* Copyright (c) 2003-2008 Jean-Marc Valin
   Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file arch.h
   @brief Various architecture definitions for CELT
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

#ifndef ARCH_H
#define ARCH_H

#include "opus_types.h"
#include "opus_defines.h"

# if !defined(__GNUC_PREREQ)
#  if defined(__GNUC__)&&defined(__GNUC_MINOR__)
#   define __GNUC_PREREQ(_maj,_min) \
 ((__GNUC__<<16)+__GNUC_MINOR__>=((_maj)<<16)+(_min))
#  else
#   define __GNUC_PREREQ(_maj,_min) 0
#  endif
# endif

#if OPUS_GNUC_PREREQ(3, 0)
#define opus_likely(x)       (__builtin_expect(!!(x), 1))
#define opus_unlikely(x)     (__builtin_expect(!!(x), 0))
#else
#define opus_likely(x)       (!!(x))
#define opus_unlikely(x)     (!!(x))
#endif

#define CELT_SIG_SCALE 32768.f

#define CELT_FATAL(str) celt_fatal(str, __FILE__, __LINE__)

#if defined(ENABLE_ASSERTIONS) || defined(ENABLE_HARDENING)
#ifdef __GNUC__
__attribute__((noreturn))
#endif
void celt_fatal(const char *str, const char *file, int line);

#if defined(CELT_C) && !defined(OVERRIDE_celt_fatal)
#include <stdio.h>
#include <stdlib.h>
#ifdef __GNUC__
__attribute__((noreturn))
#endif
void celt_fatal(const char *str, const char *file, int line)
{
   fprintf (stderr, "Fatal (internal) error in %s, line %d: %s\n", file, line, str);
#if defined(_MSC_VER)
   _set_abort_behavior( 0, _WRITE_ABORT_MSG);
#endif
   abort();
}
#endif

#define celt_assert(cond) {if (!(cond)) {CELT_FATAL("assertion failed: " #cond);}}
#define celt_assert2(cond, message) {if (!(cond)) {CELT_FATAL("assertion failed: " #cond "\n" message);}}
#define MUST_SUCCEED(call) celt_assert((call) == OPUS_OK)
#else
#define celt_assert(cond) ((void)(cond))
#define celt_assert2(cond, message) ((void)(cond))
#define MUST_SUCCEED(call) do {if((call) != OPUS_OK) {RESTORE_STACK; return OPUS_INTERNAL_ERROR;} } while (0)
#endif

#if defined(ENABLE_ASSERTIONS)
#define celt_sig_assert(cond) {if (!(cond)) {CELT_FATAL("signal assertion failed: " #cond);}}
#else
#define celt_sig_assert(cond) ((void)(cond))
#endif

#define IMUL32(a,b) ((a)*(b))

#define MIN16(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum 16-bit value.   */
#define MAX16(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum 16-bit value.   */
#define MIN32(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum 32-bit value.   */
#define MAX32(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum 32-bit value.   */
#define IMIN(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum int value.   */
#define IMAX(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum int value.   */
#define FMIN(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum float value.   */
#define FMAX(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum float value.   */
#define UADD32(a,b) ((a)+(b))
#define USUB32(a,b) ((a)-(b))
#define MAXG(a,b) MAX32(a, b)
#define MING(a,b) MIN32(a, b)

/* Throughout the code, we use the following scaling for signals:
   FLOAT: used for float API, normalized to +/-1.
   INT16: used for 16-bit API, normalized to +/- 32768
   RES: internal Opus resolution, defined as +/-1. in float builds, or either 16-bit or 24-bit int for fixed-point builds
   SIG: internal CELT resolution: defined as +/- 32768. in float builds, or Q27 in fixed-point builds (int16 shifted by 12)
*/


/* Set this if opus_int64 is a native type of the CPU. */
/* Assume that all LP64 architectures have fast 64-bit types; also x86_64
   (which can be ILP32 for x32) and Win64 (which is LLP64). */
#if defined(__x86_64__) || defined(__LP64__) || defined(_WIN64) || defined (__mips)
#define OPUS_FAST_INT64 1
#else
#define OPUS_FAST_INT64 0
#endif

#ifdef FIXED_POINT
#define ARG_FIXED(arg) , arg
#else
#define ARG_FIXED(arg)
#endif

#define PRINT_MIPS(file)

#ifdef FIXED_POINT

typedef opus_int16 opus_val16;
typedef opus_int32 opus_val32;
typedef opus_int64 opus_val64;

typedef opus_val32 celt_sig;
typedef opus_val32 celt_norm;
typedef opus_val32 celt_ener;
typedef opus_val32 celt_glog;

#ifdef ENABLE_RES24
typedef opus_val32 opus_res;
#define RES_SHIFT 8
#define SIG2RES(a)      PSHR32(a, SIG_SHIFT-RES_SHIFT)
#define RES2INT16(a)    SAT16(PSHR32(a, RES_SHIFT))
#define RES2INT24(a)    (a)
#define RES2FLOAT(a)    ((1.f/32768.f/256.f)*(a))
#define INT16TORES(a)   SHL32(EXTEND32(a), RES_SHIFT)
#define INT24TORES(a)   (a)
#define ADD_RES(a, b)   ADD32(a, b)
#define FLOAT2RES(a)    FLOAT2INT24(a)
#define RES2SIG(a)      SHL32((a), SIG_SHIFT-RES_SHIFT)
#define MULT16_RES_Q15(a,b) MULT16_32_Q15(a,b)
#define MAX_ENCODING_DEPTH 24
#else
typedef opus_val16 opus_res;
#define RES_SHIFT 0
#define SIG2RES(a)      SIG2WORD16(a)
#define RES2INT16(a)    (a)
#define RES2INT24(a)    SHL32(EXTEND32(a), 8)
#define RES2FLOAT(a)    ((1.f/32768.f)*(a))
#define INT16TORES(a)   (a)
#define INT24TORES(a)   SAT16(PSHR32(a, 8))
#define ADD_RES(a, b)   SAT16(ADD32((a), (b)));
#define FLOAT2RES(a)    FLOAT2INT16(a)
#define RES2SIG(a)      SHL32(EXTEND32(a), SIG_SHIFT)
#define MULT16_RES_Q15(a,b) MULT16_16_Q15(a,b)
#define MAX_ENCODING_DEPTH 16
#endif

#define RES2VAL16(a)    RES2INT16(a)
#define INT16TOSIG(a)   SHL32(EXTEND32(a), SIG_SHIFT)
#define INT24TOSIG(a)   SHL32(a, SIG_SHIFT-8)

#define NORM_SHIFT 24
#ifdef ENABLE_QEXT
typedef opus_val32 celt_coef;
#define COEF_ONE Q31ONE
#define MULT_COEF_32(a, b) MULT32_32_P31(a,b)
#define MAC_COEF_32_ARM(c, a, b) ADD32((c), MULT32_32_Q32(a,b))
#define MULT_COEF(a, b) MULT32_32_Q31(a,b)
#define MULT_COEF_TAPS(a, b) SHL32(MULT16_16(a,b), 1)
#define COEF2VAL16(x) EXTRACT16(SHR32(x, 16))
#else
typedef opus_val16 celt_coef;
#define COEF_ONE Q15ONE
#define MULT_COEF_32(a, b) MULT16_32_Q15(a,b)
#define MAC_COEF_32_ARM(a, b, c) MAC16_32_Q16(a,b,c)
#define MULT_COEF(a, b) MULT16_16_Q15(a,b)
#define MULT_COEF_TAPS(a, b) MULT16_16_P15(a,b)
#define COEF2VAL16(x) (x)
#endif

#define celt_isnan(x) 0

#define Q15ONE 32767
#define Q31ONE 2147483647

#define SIG_SHIFT 12
/* Safe saturation value for 32-bit signals. We need to make sure that we can
   add two sig values and that the first stages of the MDCT don't cause an overflow.
   The most constraining is the ARM_ASM comb filter where we shift left by one
   and then add two values. Because of that, we use 2^29-1. SIG_SAT must be large
   enough to fit a full-scale high-freq tone through the prefilter and comb filter,
   meaning 1.85*1.75*2^(15+SIG_SHIFT) =  434529895.
   so the limit should be about 2^31*sqrt(.5). */
#define SIG_SAT (536870911)

#define NORM_SCALING (1<<NORM_SHIFT)

#define DB_SHIFT 24

#define EPSILON 1
#define VERY_SMALL 0
#define VERY_LARGE16 ((opus_val16)32767)
#define Q15_ONE ((opus_val16)32767)


#define ABS16(x) ((x) < 0 ? (-(x)) : (x))
#define ABS32(x) ((x) < 0 ? (-(x)) : (x))

static OPUS_INLINE opus_int16 SAT16(opus_int32 x) {
   return x > 32767 ? 32767 : x < -32768 ? -32768 : (opus_int16)x;
}

#ifdef FIXED_DEBUG
#include "fixed_debug.h"
#else

#include "fixed_generic.h"

#ifdef OPUS_ARM_PRESUME_AARCH64_NEON_INTR
#include "arm/fixed_arm64.h"
#elif defined (OPUS_ARM_INLINE_EDSP)
#include "arm/fixed_armv5e.h"
#elif defined (OPUS_ARM_INLINE_ASM)
#include "arm/fixed_armv4.h"
#elif defined (BFIN_ASM)
#include "fixed_bfin.h"
#elif defined (TI_C5X_ASM)
#include "fixed_c5x.h"
#elif defined (TI_C6X_ASM)
#include "fixed_c6x.h"
#endif

#endif

#else /* FIXED_POINT */

typedef float opus_val16;
typedef float opus_val32;
typedef float opus_val64;

typedef float celt_sig;
typedef float celt_norm;
typedef float celt_ener;
typedef float celt_glog;

typedef float opus_res;
typedef float celt_coef;

#ifdef FLOAT_APPROX
/* This code should reliably detect NaN/inf even when -ffast-math is used.
   Assumes IEEE 754 format. */
static OPUS_INLINE int celt_isnan(float x)
{
   union {float f; opus_uint32 i;} in;
   in.f = x;
   return ((in.i>>23)&0xFF)==0xFF && (in.i&0x007FFFFF)!=0;
}
#else
#ifdef __FAST_MATH__
#error Cannot build libopus with -ffast-math unless FLOAT_APPROX is defined. This could result in crashes on extreme (e.g. NaN) input
#endif
#define celt_isnan(x) ((x)!=(x))
#endif

#define Q15ONE 1.0f
#define Q31ONE 1.0f
#define COEF_ONE 1.0f
#define COEF2VAL16(x) (x)

#define NORM_SCALING 1.f

#define EPSILON 1e-15f
#define VERY_SMALL 1e-30f
#define VERY_LARGE16 1e15f
#define Q15_ONE ((opus_val16)1.f)

/* This appears to be the same speed as C99's fabsf() but it's more portable. */
#define ABS16(x) ((float)fabs(x))
#define ABS32(x) ((float)fabs(x))

#define QCONST16(x,bits) (x)
#define QCONST32(x,bits) (x)
#define GCONST(x) (x)

#define NEG16(x) (-(x))
#define NEG32(x) (-(x))
#define NEG32_ovflw(x) (-(x))
#define EXTRACT16(x) (x)
#define EXTEND32(x) (x)
#define SHR16(a,shift) (a)
#define SHL16(a,shift) (a)
#define SHR32(a,shift) (a)
#define SHL32(a,shift) (a)
#define PSHR32(a,shift) (a)
#define VSHR32(a,shift) (a)

#define SHR64(a,shift) (a)

#define PSHR(a,shift)   (a)
#define SHR(a,shift)    (a)
#define SHL(a,shift)    (a)
#define SATURATE(x,a)   (x)
#define SATURATE16(x)   (x)

#define ROUND16(a,shift)  (a)
#define SROUND16(a,shift) (a)
#define HALF16(x)       (.5f*(x))
#define HALF32(x)       (.5f*(x))

#define ADD16(a,b) ((a)+(b))
#define SUB16(a,b) ((a)-(b))
#define ADD32(a,b) ((a)+(b))
#define SUB32(a,b) ((a)-(b))
#define ADD32_ovflw(a,b) ((a)+(b))
#define SUB32_ovflw(a,b) ((a)-(b))
#define SHL32_ovflw(a,shift) (a)
#define PSHR32_ovflw(a,shift) (a)

#define MULT16_16_16(a,b)     ((a)*(b))
#define MULT16_16(a,b)     ((opus_val32)(a)*(opus_val32)(b))
#define MAC16_16(c,a,b)     ((c)+(opus_val32)(a)*(opus_val32)(b))

#define MULT16_32_Q15(a,b)     ((a)*(b))
#define MULT16_32_Q16(a,b)     ((a)*(b))

#define MULT32_32_Q16(a,b)     ((a)*(b))
#define MULT32_32_Q31(a,b)     ((a)*(b))
#define MULT32_32_P31(a,b)     ((a)*(b))
#define MULT32_32_P31_ovflw(a,b) ((a)*(b))

#define MAC16_32_Q15(c,a,b)     ((c)+(a)*(b))
#define MAC16_32_Q16(c,a,b)     ((c)+(a)*(b))
#define MAC_COEF_32_ARM(c,a,b)     ((c)+(a)*(b))

#define MULT16_16_Q11_32(a,b)     ((a)*(b))
#define MULT16_16_Q11(a,b)     ((a)*(b))
#define MULT16_16_Q13(a,b)     ((a)*(b))
#define MULT16_16_Q14(a,b)     ((a)*(b))
#define MULT16_16_Q15(a,b)     ((a)*(b))
#define MULT16_16_P15(a,b)     ((a)*(b))
#define MULT16_16_P13(a,b)     ((a)*(b))
#define MULT16_16_P14(a,b)     ((a)*(b))
#define MULT16_32_P16(a,b)     ((a)*(b))

#define MULT_COEF_32(a, b)      ((a)*(b))
#define MULT_COEF(a, b)   ((a)*(b))
#define MULT_COEF_TAPS(a, b)   ((a)*(b))

#define DIV32_16(a,b)     (((opus_val32)(a))/(opus_val16)(b))
#define DIV32(a,b)     (((opus_val32)(a))/(opus_val32)(b))

#define SIG2RES(a)      ((1/CELT_SIG_SCALE)*(a))
#define RES2INT16(a)    FLOAT2INT16(a)
#define RES2INT24(a)    float2int(32768.f*256.f*(a))
#define RES2FLOAT(a)    (a)
#define INT16TORES(a)   ((a)*(1/CELT_SIG_SCALE))
#define INT24TORES(a)   ((1.f/32768.f/256.f)*(a))
#define ADD_RES(a, b)   ADD32(a, b)
#define FLOAT2RES(a)    (a)
#define RES2SIG(a)      (CELT_SIG_SCALE*(a))
#define MULT16_RES_Q15(a,b) MULT16_16_Q15(a,b)

#define RES2VAL16(a)    (a)
#define FLOAT2SIG(a)    ((a)*CELT_SIG_SCALE)
#define INT16TOSIG(a)   ((float)(a))
#define INT24TOSIG(a)   ((float)(a)*(1.f/256.f))
#define MAX_ENCODING_DEPTH 24

#endif /* !FIXED_POINT */

#ifndef GLOBAL_STACK_SIZE
#ifdef FIXED_POINT
#define GLOBAL_STACK_SIZE 120000
#else
#define GLOBAL_STACK_SIZE 120000
#endif
#endif

#endif /* ARCH_H */
