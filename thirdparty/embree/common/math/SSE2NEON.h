#ifndef SSE2NEON_H
#define SSE2NEON_H

// This header file provides a simple API translation layer
// between SSE intrinsics to their corresponding ARM NEON versions
//
// This header file does not (yet) translate *all* of the SSE intrinsics.
// Since this is in support of a specific porting effort, I have only
// included the intrinsics I needed to get my port to work.
//
// Questions/Comments/Feedback send to: jratcliffscarab@gmail.com
//
// If you want to improve or add to this project, send me an
// email and I will probably approve your access to the depot.
//
// Project is located here:
//
//	https://github.com/jratcliff63367/sse2neon
//
// Show your appreciation for open source by sending me a bitcoin tip to the following
// address.
//
// TipJar: 1PzgWDSyq4pmdAXRH8SPUtta4SWGrt4B1p :
// https://blockchain.info/address/1PzgWDSyq4pmdAXRH8SPUtta4SWGrt4B1p
//
//
// Contributors to this project are:
//
// John W. Ratcliff : jratcliffscarab@gmail.com
// Brandon Rowlett  : browlett@nvidia.com
// Ken Fast         : kfast@gdeb.com
// Eric van Beurden : evanbeurden@nvidia.com
//
//
// *********************************************************************************************************************
// Release notes for January 20, 2017 version:
//
// The unit tests have been refactored.  They no longer assert on an error, instead they return a pass/fail condition
// The unit-tests now test 10,000 random float and int values against each intrinsic.
//
// SSE2NEON now supports 95 SSE intrinsics.  39 of them have formal unit tests which have been implemented and
// fully tested on NEON/ARM.  The remaining 56 still need unit tests implemented.
//
// A struct is now defined in this header file called 'SIMDVec' which can be used by applications which
// attempt to access the contents of an _m128 struct directly.  It is important to note that accessing the __m128
// struct directly is bad coding practice by Microsoft: @see: https://msdn.microsoft.com/en-us/library/ayeb3ayc.aspx
//
// However, some legacy source code may try to access the contents of an __m128 struct directly so the developer
// can use the SIMDVec as an alias for it.  Any casting must be done manually by the developer, as you cannot
// cast or otherwise alias the base NEON data type for intrinsic operations.
//
// A bug was found with the _mm_shuffle_ps intrinsic.  If the shuffle permutation was not one of the ones with
// a custom/unique implementation causing it to fall through to the default shuffle implementation it was failing
// to return the correct value.  This is now fixed.
//
// A bug was found with the _mm_cvtps_epi32 intrinsic.  This converts floating point values to integers.
// It was not honoring the correct rounding mode.  In SSE the default rounding mode when converting from float to int
// is to use 'round to even' otherwise known as 'bankers rounding'.  ARMv7 did not support this feature but ARMv8 does.
// As it stands today, this header file assumes ARMv8.  If you are trying to target really old ARM devices, you may get
// a build error.
//
// Support for a number of new intrinsics was added, however, none of them yet have unit-tests to 100% confirm they are
// producing the correct results on NEON.  These unit tests will be added as soon as possible.
//
// Here is the list of new instrinsics which have been added:
//
// _mm_cvtss_f32     :  extracts the lower order floating point value from the parameter
// _mm_add_ss        : adds the scalar single - precision floating point values of a and b
// _mm_div_ps        : Divides the four single - precision, floating - point values of a and b.
// _mm_div_ss        : Divides the scalar single - precision floating point value of a by b.
// _mm_sqrt_ss       : Computes the approximation of the square root of the scalar single - precision floating point value of in.
// _mm_rsqrt_ps      : Computes the approximations of the reciprocal square roots of the four single - precision floating point values of in.
// _mm_comilt_ss     : Compares the lower single - precision floating point scalar values of a and b using a less than operation
// _mm_comigt_ss     : Compares the lower single - precision floating point scalar values of a and b using a greater than operation.
// _mm_comile_ss     :  Compares the lower single - precision floating point scalar values of a and b using a less than or equal operation.
// _mm_comige_ss     : Compares the lower single - precision floating point scalar values of a and b using a greater than or equal operation.
// _mm_comieq_ss     :  Compares the lower single - precision floating point scalar values of a and b using an equality operation.
// _mm_comineq_s     :  Compares the lower single - precision floating point scalar values of a and b using an inequality operation
// _mm_unpackhi_epi8 : Interleaves the upper 8 signed or unsigned 8 - bit integers in a with the upper 8 signed or unsigned 8 - bit integers in b.
// _mm_unpackhi_epi16:  Interleaves the upper 4 signed or unsigned 16 - bit integers in a with the upper 4 signed or unsigned 16 - bit integers in b.
//
// *********************************************************************************************************************
/*
** The MIT license:
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is furnished
** to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#define GCC 1
#define ENABLE_CPP_VERSION 0

// enable precise emulation of _mm_min_ps and _mm_max_ps?
// This would slow down the computation a bit, but gives consistent result with x86 SSE2.
// (e.g. would solve a hole or NaN pixel in the rendering result)
#define USE_PRECISE_MINMAX_IMPLEMENTATION (1)

#if GCC
#define FORCE_INLINE					inline __attribute__((always_inline))
#define ALIGN_STRUCT(x)					__attribute__((aligned(x)))
#else
#define FORCE_INLINE					inline
#define ALIGN_STRUCT(x)					__declspec(align(x))
#endif

#include <stdint.h>
#include "arm_neon.h"
#if defined(__aarch64__)
#include "constants.h"
#endif


#if !defined(__has_builtin)
#define __has_builtin(x) (0)
#endif

/*******************************************************/
/* MACRO for shuffle parameter for _mm_shuffle_ps().   */
/* Argument fp3 is a digit[0123] that represents the fp*/
/* from argument "b" of mm_shuffle_ps that will be     */
/* placed in fp3 of result. fp2 is the same for fp2 in */
/* result. fp1 is a digit[0123] that represents the fp */
/* from argument "a" of mm_shuffle_ps that will be     */
/* places in fp1 of result. fp0 is the same for fp0 of */
/* result                                              */
/*******************************************************/
#if defined(__aarch64__)
#define _MN_SHUFFLE(fp3,fp2,fp1,fp0) ( (uint8x16_t){ (((fp3)*4)+0), (((fp3)*4)+1), (((fp3)*4)+2), (((fp3)*4)+3),  (((fp2)*4)+0), (((fp2)*4)+1), (((fp2)*4)+2), (((fp2)*4)+3),  (((fp1)*4)+0), (((fp1)*4)+1), (((fp1)*4)+2), (((fp1)*4)+3),  (((fp0)*4)+0), (((fp0)*4)+1), (((fp0)*4)+2), (((fp0)*4)+3) } )
#define _MF_SHUFFLE(fp3,fp2,fp1,fp0) ( (uint8x16_t){ (((fp3)*4)+0), (((fp3)*4)+1), (((fp3)*4)+2), (((fp3)*4)+3),  (((fp2)*4)+0), (((fp2)*4)+1), (((fp2)*4)+2), (((fp2)*4)+3),  (((fp1)*4)+16+0), (((fp1)*4)+16+1), (((fp1)*4)+16+2), (((fp1)*4)+16+3),  (((fp0)*4)+16+0), (((fp0)*4)+16+1), (((fp0)*4)+16+2), (((fp0)*4)+16+3) } )
#endif

#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3) << 6) | ((fp2) << 4) | \
  ((fp1) << 2) | ((fp0)))

typedef float32x4_t __m128;
typedef int32x4_t __m128i;

// union intended to allow direct access to an __m128 variable using the names that the MSVC
// compiler provides.  This union should really only be used when trying to access the members
// of the vector as integer values.  GCC/clang allow native access to the float members through
// a simple array access operator (in C since 4.6, in C++ since 4.8).
//
// Ideally direct accesses to SIMD vectors should not be used since it can cause a performance
// hit.  If it really is needed however, the original __m128 variable can be aliased with a
// pointer to this union and used to access individual components.  The use of this union should
// be hidden behind a macro that is used throughout the codebase to access the members instead
// of always declaring this type of variable.
typedef union ALIGN_STRUCT(16) SIMDVec
{
  float       m128_f32[4];    // as floats - do not to use this.  Added for convenience.
  int8_t      m128_i8[16];    // as signed 8-bit integers.
  int16_t     m128_i16[8];    // as signed 16-bit integers.
  int32_t     m128_i32[4];    // as signed 32-bit integers.
  int64_t     m128_i64[2];    // as signed 64-bit integers.
  uint8_t     m128_u8[16];    // as unsigned 8-bit integers.
  uint16_t    m128_u16[8];    // as unsigned 16-bit integers.
  uint32_t    m128_u32[4];    // as unsigned 32-bit integers.
  uint64_t    m128_u64[2];    // as unsigned 64-bit integers.
  double	    m128_f64[2];    // as signed double
} SIMDVec;

// ******************************************
// CPU stuff
// ******************************************

typedef SIMDVec __m128d;

#include <stdlib.h>

#ifndef _MM_MASK_MASK
#define _MM_MASK_MASK 0x1f80
#define _MM_MASK_DIV_ZERO 0x200
#define _MM_FLUSH_ZERO_ON 0x8000
#define _MM_DENORMALS_ZERO_ON 0x40
#define _MM_MASK_DENORM 0x100
#endif
#define _MM_SET_EXCEPTION_MASK(x)
#define _MM_SET_FLUSH_ZERO_MODE(x)
#define _MM_SET_DENORMALS_ZERO_MODE(x)

FORCE_INLINE void _mm_pause()
{
}

FORCE_INLINE void _mm_mfence()
{
    __sync_synchronize();
}

#define _MM_HINT_T0 3
#define _MM_HINT_T1 2
#define _MM_HINT_T2 1
#define _MM_HINT_NTA 0

FORCE_INLINE void _mm_prefetch(const void* ptr, unsigned int level)
{
   __builtin_prefetch(ptr);
 
}

FORCE_INLINE void* _mm_malloc(int size, int align)
{
    void *ptr;
    // align must be multiple of sizeof(void *) for posix_memalign.
    if (align < sizeof(void *)) {
        align = sizeof(void *);
    }

    if ((align % sizeof(void *)) != 0) {
        // fallback to malloc
        ptr = malloc(size);
    } else {
        if (posix_memalign(&ptr, align, size)) {
          return 0;
        }
    }

    return ptr;
}

FORCE_INLINE void _mm_free(void* ptr)
{
        free(ptr);
}

FORCE_INLINE int _mm_getcsr()
{
        return 0;
}

FORCE_INLINE void _mm_setcsr(int val)
{
        return;
}

// ******************************************
// Set/get methods
// ******************************************

// extracts the lower order floating point value from the parameter : https://msdn.microsoft.com/en-us/library/bb514059%28v=vs.120%29.aspx?f=255&MSPPError=-2147217396
#if defined(__aarch64__)
FORCE_INLINE float _mm_cvtss_f32(const __m128& x)
{
    return x[0];
}
#else
FORCE_INLINE float _mm_cvtss_f32(__m128 a)
{
    return vgetq_lane_f32(a, 0);
}
#endif

// Sets the 128-bit value to zero https://msdn.microsoft.com/en-us/library/vstudio/ys7dw0kh(v=vs.100).aspx
FORCE_INLINE __m128i _mm_setzero_si128()
{
  return vdupq_n_s32(0);
}

// Clears the four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/tk1t2tbz(v=vs.100).aspx
FORCE_INLINE __m128 _mm_setzero_ps(void)
{
  return vdupq_n_f32(0);
}

// Sets the four single-precision, floating-point values to w. https://msdn.microsoft.com/en-us/library/vstudio/2x1se8ha(v=vs.100).aspx
FORCE_INLINE __m128 _mm_set1_ps(float _w)
{
  return vdupq_n_f32(_w);
}

// Sets the four single-precision, floating-point values to w. https://msdn.microsoft.com/en-us/library/vstudio/2x1se8ha(v=vs.100).aspx
FORCE_INLINE __m128 _mm_set_ps1(float _w)
{
  return vdupq_n_f32(_w);
}

// Sets the four single-precision, floating-point values to the four inputs. https://msdn.microsoft.com/en-us/library/vstudio/afh0zf75(v=vs.100).aspx
#if defined(__aarch64__) 
FORCE_INLINE __m128 _mm_set_ps(const float w, const float z, const float y, const float x)
{
    float32x4_t t = { x, y, z, w };
    return t;
}

// Sets the four single-precision, floating-point values to the four inputs in reverse order. https://msdn.microsoft.com/en-us/library/vstudio/d2172ct3(v=vs.100).aspx
FORCE_INLINE __m128 _mm_setr_ps(const float w, const float z , const float y , const float x )
{
    float32x4_t t = { w, z, y, x };
    return t;
}
#else
FORCE_INLINE __m128 _mm_set_ps(float w, float z, float y, float x)
{
    float __attribute__((aligned(16))) data[4] = { x, y, z, w };
    return vld1q_f32(data);
}

// Sets the four single-precision, floating-point values to the four inputs in reverse order. https://msdn.microsoft.com/en-us/library/vstudio/d2172ct3(v=vs.100).aspx
FORCE_INLINE __m128 _mm_setr_ps(float w, float z , float y , float x )
{
    float __attribute__ ((aligned (16))) data[4] = { w, z, y, x };
    return vld1q_f32(data);
}
#endif

// Sets the 4 signed 32-bit integer values to i. https://msdn.microsoft.com/en-us/library/vstudio/h4xscxat(v=vs.100).aspx
FORCE_INLINE __m128i _mm_set1_epi32(int _i)
{
  return vdupq_n_s32(_i);
}

//Set the first lane to of 4 signed single-position, floating-point number to w
#if defined(__aarch64__)
FORCE_INLINE __m128 _mm_set_ss(float _w)
{
    float32x4_t res = {_w, 0, 0, 0};
    return res;
}

// Sets the 4 signed 32-bit integer values. https://msdn.microsoft.com/en-us/library/vstudio/019beekt(v=vs.100).aspx
FORCE_INLINE __m128i _mm_set_epi32(int i3, int i2, int i1, int i0)
{
    int32x4_t t = {i0,i1,i2,i3};
    return t;
}
#else
FORCE_INLINE __m128 _mm_set_ss(float _w)
{
    __m128 val = _mm_setzero_ps();
    return vsetq_lane_f32(_w, val, 0);
}

// Sets the 4 signed 32-bit integer values. https://msdn.microsoft.com/en-us/library/vstudio/019beekt(v=vs.100).aspx
FORCE_INLINE __m128i _mm_set_epi32(int i3, int i2, int i1, int i0)
{
    int32_t __attribute__((aligned(16))) data[4] = { i0, i1, i2, i3 };
    return vld1q_s32(data);
}
#endif

// Stores four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/s3h4ay6y(v=vs.100).aspx
FORCE_INLINE void _mm_store_ps(float *p, __m128 a)
{
  vst1q_f32(p, a);
}

// Stores four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/44e30x22(v=vs.100).aspx
FORCE_INLINE void _mm_storeu_ps(float *p, __m128 a)
{
  vst1q_f32(p, a);
}

FORCE_INLINE void _mm_storeu_si128(__m128i *p, __m128i a)
{
  vst1q_s32((int32_t*) p,a);
}

// Stores four 32-bit integer values as (as a __m128i value) at the address p. https://msdn.microsoft.com/en-us/library/vstudio/edk11s13(v=vs.100).aspx
FORCE_INLINE void _mm_store_si128(__m128i *p, __m128i a )
{
  vst1q_s32((int32_t*) p,a);
}

// Stores the lower single - precision, floating - point value. https://msdn.microsoft.com/en-us/library/tzz10fbx(v=vs.100).aspx
FORCE_INLINE void _mm_store_ss(float *p, __m128 a)
{
  vst1q_lane_f32(p, a, 0);
}

// Reads the lower 64 bits of b and stores them into the lower 64 bits of a.  https://msdn.microsoft.com/en-us/library/hhwf428f%28v=vs.90%29.aspx
FORCE_INLINE void _mm_storel_epi64(__m128i* a, __m128i b)
{
  *a = (__m128i)vsetq_lane_s64((int64_t)vget_low_s32(b), *(int64x2_t*)a, 0);
}

// Loads a single single-precision, floating-point value, copying it into all four words https://msdn.microsoft.com/en-us/library/vstudio/5cdkf716(v=vs.100).aspx
FORCE_INLINE __m128 _mm_load1_ps(const float * p)
{
  return vld1q_dup_f32(p);
}

// Loads four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/zzd50xxt(v=vs.100).aspx
FORCE_INLINE __m128 _mm_load_ps(const float * p)
{
  return vld1q_f32(p);
}

// Loads four single-precision, floating-point values.  https://msdn.microsoft.com/en-us/library/x1b16s7z%28v=vs.90%29.aspx
FORCE_INLINE __m128 _mm_loadu_ps(const float * p)
{
  // for neon, alignment doesn't matter, so _mm_load_ps and _mm_loadu_ps are equivalent for neon
  return vld1q_f32(p);
}

// Loads an single - precision, floating - point value into the low word and clears the upper three words.  https://msdn.microsoft.com/en-us/library/548bb9h4%28v=vs.90%29.aspx
FORCE_INLINE __m128 _mm_load_ss(const float * p)
{
  __m128 result = vdupq_n_f32(0);
  return vsetq_lane_f32(*p, result, 0);
}

FORCE_INLINE __m128i _mm_loadu_si128(__m128i *p)
{
  return (__m128i)vld1q_s32((const int32_t*) p);
}


// ******************************************
// Logic/Binary operations
// ******************************************

// Compares for inequality.  https://msdn.microsoft.com/en-us/library/sf44thbx(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmpneq_ps(__m128 a, __m128 b)
{
  return (__m128)vmvnq_s32((__m128i)vceqq_f32(a, b));
}

// Computes the bitwise AND-NOT of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/68h7wd02(v=vs.100).aspx
FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
  return (__m128)vbicq_s32((__m128i)b, (__m128i)a); // *NOTE* argument swap
}

// Computes the bitwise AND of the 128-bit value in b and the bitwise NOT of the 128-bit value in a. https://msdn.microsoft.com/en-us/library/vstudio/1beaceh8(v=vs.100).aspx
FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
  return (__m128i)vbicq_s32(b, a); // *NOTE* argument swap
}

// Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b. https://msdn.microsoft.com/en-us/library/vstudio/6d1txsa8(v=vs.100).aspx
FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b)
{
  return (__m128i)vandq_s32(a, b);
}

// Computes the bitwise AND of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/73ck1xc5(v=vs.100).aspx
FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b)
{
  return (__m128)vandq_s32((__m128i)a, (__m128i)b);
}

// Computes the bitwise OR of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/7ctdsyy0(v=vs.100).aspx
FORCE_INLINE __m128 _mm_or_ps(__m128 a, __m128 b)
{
  return (__m128)vorrq_s32((__m128i)a, (__m128i)b);
}

// Computes bitwise EXOR (exclusive-or) of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/ss6k3wk8(v=vs.100).aspx
FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b)
{
  return (__m128)veorq_s32((__m128i)a, (__m128i)b);
}

// Computes the bitwise OR of the 128-bit value in a and the 128-bit value in b. https://msdn.microsoft.com/en-us/library/vstudio/ew8ty0db(v=vs.100).aspx
FORCE_INLINE __m128i _mm_or_si128(__m128i a, __m128i b)
{
  return (__m128i)vorrq_s32(a, b);
}

// Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.  https://msdn.microsoft.com/en-us/library/fzt08www(v=vs.100).aspx
FORCE_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b)
{
  return veorq_s32(a, b);
}

// NEON does not provide this method
// Creates a 4-bit mask from the most significant bits of the four single-precision, floating-point values. https://msdn.microsoft.com/en-us/library/vstudio/4490ys29(v=vs.100).aspx
FORCE_INLINE int _mm_movemask_ps(__m128 a)
{
#if ENABLE_CPP_VERSION // I am not yet convinced that the NEON version is faster than the C version of this
  uint32x4_t &ia = *(uint32x4_t *)&a;
  return (ia[0] >> 31) | ((ia[1] >> 30) & 2) | ((ia[2] >> 29) & 4) | ((ia[3] >> 28) & 8);
#else
    
#if defined(__aarch64__)
    uint32x4_t t2 = vandq_u32(vreinterpretq_u32_f32(a), embree::movemask_mask);
    return vaddvq_u32(t2);
#else
  static const uint32x4_t movemask = { 1, 2, 4, 8 };
  static const uint32x4_t highbit = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
  uint32x4_t t0 = vreinterpretq_u32_f32(a);
  uint32x4_t t1 = vtstq_u32(t0, highbit);
  uint32x4_t t2 = vandq_u32(t1, movemask);
  uint32x2_t t3 = vorr_u32(vget_low_u32(t2), vget_high_u32(t2));
  return vget_lane_u32(t3, 0) | vget_lane_u32(t3, 1);
#endif
    
#endif
}

#if defined(__aarch64__)
FORCE_INLINE int _mm_movemask_popcnt_ps(__m128 a)
{
    uint32x4_t t2 = vandq_u32(vreinterpretq_u32_f32(a), embree::movemask_mask);
    t2 = vreinterpretq_u32_u8(vcntq_u8(vreinterpretq_u8_u32(t2)));
    return vaddvq_u32(t2);
    
}
#endif

// Takes the upper 64 bits of a and places it in the low end of the result
// Takes the lower 64 bits of b and places it into the high end of the result.
FORCE_INLINE __m128 _mm_shuffle_ps_1032(__m128 a, __m128 b)
{
  return vcombine_f32(vget_high_f32(a), vget_low_f32(b));
}

// takes the lower two 32-bit values from a and swaps them and places in high end of result
// takes the higher two 32 bit values from b and swaps them and places in low end of result.
FORCE_INLINE __m128 _mm_shuffle_ps_2301(__m128 a, __m128 b)
{
  return vcombine_f32(vrev64_f32(vget_low_f32(a)), vrev64_f32(vget_high_f32(b)));
}

// keeps the low 64 bits of b in the low and puts the high 64 bits of a in the high
FORCE_INLINE __m128 _mm_shuffle_ps_3210(__m128 a, __m128 b)
{
  return vcombine_f32(vget_low_f32(a), vget_high_f32(b));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0011(__m128 a, __m128 b)
{
  return vcombine_f32(vdup_n_f32(vgetq_lane_f32(a, 1)), vdup_n_f32(vgetq_lane_f32(b, 0)));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0022(__m128 a, __m128 b)
{
  return vcombine_f32(vdup_n_f32(vgetq_lane_f32(a, 2)), vdup_n_f32(vgetq_lane_f32(b, 0)));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2200(__m128 a, __m128 b)
{
  return vcombine_f32(vdup_n_f32(vgetq_lane_f32(a, 0)), vdup_n_f32(vgetq_lane_f32(b, 2)));
}

FORCE_INLINE __m128 _mm_shuffle_ps_3202(__m128 a, __m128 b)
{
  float32_t a0 = vgetq_lane_f32(a, 0);
  float32_t a2 = vgetq_lane_f32(a, 2);
  float32x2_t aVal = vdup_n_f32(a2);
  aVal = vset_lane_f32(a0, aVal, 1);
  return vcombine_f32(aVal, vget_high_f32(b));
}

FORCE_INLINE __m128 _mm_shuffle_ps_1133(__m128 a, __m128 b)
{
  return vcombine_f32(vdup_n_f32(vgetq_lane_f32(a, 3)), vdup_n_f32(vgetq_lane_f32(b, 1)));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2010(__m128 a, __m128 b)
{
  float32_t b0 = vgetq_lane_f32(b, 0);
  float32_t b2 = vgetq_lane_f32(b, 2);
  float32x2_t bVal = vdup_n_f32(b0);
  bVal = vset_lane_f32(b2, bVal, 1);
  return vcombine_f32(vget_low_f32(a), bVal);
}

FORCE_INLINE __m128 _mm_shuffle_ps_2001(__m128 a, __m128 b)
{
  float32_t b0 = vgetq_lane_f32(b, 0);
  float32_t b2 = vgetq_lane_f32(b, 2);
  float32x2_t bVal = vdup_n_f32(b0);
  bVal = vset_lane_f32(b2, bVal, 1);
  return vcombine_f32(vrev64_f32(vget_low_f32(a)), bVal);
}

FORCE_INLINE __m128 _mm_shuffle_ps_2032(__m128 a, __m128 b)
{
  float32_t b0 = vgetq_lane_f32(b, 0);
  float32_t b2 = vgetq_lane_f32(b, 2);
  float32x2_t bVal = vdup_n_f32(b0);
  bVal = vset_lane_f32(b2, bVal, 1);
  return vcombine_f32(vget_high_f32(a), bVal);
}

FORCE_INLINE __m128 _mm_shuffle_ps_0321(__m128 a, __m128 b)
{
  float32x2_t a21 = vget_high_f32(vextq_f32(a, a, 3));
  float32x2_t b03 = vget_low_f32(vextq_f32(b, b, 3));
  return vcombine_f32(a21, b03);
}

FORCE_INLINE __m128 _mm_shuffle_ps_2103(__m128 a, __m128 b)
{
  float32x2_t a03 = vget_low_f32(vextq_f32(a, a, 3));
  float32x2_t b21 = vget_high_f32(vextq_f32(b, b, 3));
  return vcombine_f32(a03, b21);
}

FORCE_INLINE __m128 _mm_shuffle_ps_1010(__m128 a, __m128 b)
{
  float32x2_t a10 = vget_low_f32(a);
  float32x2_t b10 = vget_low_f32(b);
  return vcombine_f32(a10, b10);
}

FORCE_INLINE __m128 _mm_shuffle_ps_1001(__m128 a, __m128 b)
{
  float32x2_t a01 = vrev64_f32(vget_low_f32(a));
  float32x2_t b10 = vget_low_f32(b);
  return vcombine_f32(a01, b10);
}

FORCE_INLINE __m128 _mm_shuffle_ps_0101(__m128 a, __m128 b)
{
  float32x2_t a01 = vrev64_f32(vget_low_f32(a));
  float32x2_t b01 = vrev64_f32(vget_low_f32(b));
  return vcombine_f32(a01, b01);
}

// NEON does not support a general purpose permute intrinsic
// Currently I am not sure whether the C implementation is faster or slower than the NEON version.
// Note, this has to be expanded as a template because the shuffle value must be an immediate value.
// The same is true on SSE as well.
// Selects four specific single-precision, floating-point values from a and b, based on the mask i. https://msdn.microsoft.com/en-us/library/vstudio/5f0858x0(v=vs.100).aspx
template <int i>
FORCE_INLINE __m128 _mm_shuffle_ps_default(const __m128& a, const __m128& b)
{
#if ENABLE_CPP_VERSION // I am not convinced that the NEON version is faster than the C version yet.
  __m128 ret;
  ret[0] = a[i & 0x3];
  ret[1] = a[(i >> 2) & 0x3];
  ret[2] = b[(i >> 4) & 0x03];
  ret[3] = b[(i >> 6) & 0x03];
  return ret;
#else
# if __has_builtin(__builtin_shufflevector)
    return __builtin_shufflevector(             \
        a, b, (i) & (0x3), ((i) >> 2) & 0x3,
        (((i) >> 4) & 0x3) + 4, (((i) >> 6) & 0x3) + 4);
# else
    const int i0 = (i >> 0)&0x3;
    const int i1 = (i >> 2)&0x3;
    const int i2 = (i >> 4)&0x3;
    const int i3 = (i >> 6)&0x3;

    if (&a == &b)
     {
         if (i0 == i1 && i0 == i2 && i0 == i3)
         {
             return (float32x4_t)vdupq_laneq_f32(a,i0);
         }
         static const uint8_t tbl[16] = {
             (i0*4) + 0,(i0*4) + 1,(i0*4) + 2,(i0*4) + 3,
             (i1*4) + 0,(i1*4) + 1,(i1*4) + 2,(i1*4) + 3,
             (i2*4) + 0,(i2*4) + 1,(i2*4) + 2,(i2*4) + 3,
             (i3*4) + 0,(i3*4) + 1,(i3*4) + 2,(i3*4) + 3
         };
         
         return (float32x4_t)vqtbl1q_s8(int8x16_t(b),*(uint8x16_t *)tbl);
         
     }
     else
     {
         
         static const uint8_t tbl[16] = {
             (i0*4) + 0,(i0*4) + 1,(i0*4) + 2,(i0*4) + 3,
             (i1*4) + 0,(i1*4) + 1,(i1*4) + 2,(i1*4) + 3,
             (i2*4) + 0 + 16,(i2*4) + 1 + 16,(i2*4) + 2 + 16,(i2*4) + 3 + 16,
             (i3*4) + 0 + 16,(i3*4) + 1 + 16,(i3*4) + 2 + 16,(i3*4) + 3 + 16
         };
         
         return float32x4_t(vqtbl2q_s8((int8x16x2_t){int8x16_t(a),int8x16_t(b)},*(uint8x16_t *)tbl));
     }
# endif //builtin(shufflevector)
#endif
}

template <int i >
FORCE_INLINE __m128 _mm_shuffle_ps_function(const __m128& a, const __m128& b)
{
  switch (i)
  {
    case _MM_SHUFFLE(1, 0, 3, 2):
      return _mm_shuffle_ps_1032(a, b);
      break;
    case _MM_SHUFFLE(2, 3, 0, 1):
      return _mm_shuffle_ps_2301(a, b);
      break;
    case _MM_SHUFFLE(3, 2, 1, 0):
      return _mm_shuffle_ps_3210(a, b);
      break;
    case _MM_SHUFFLE(0, 0, 1, 1):
      return _mm_shuffle_ps_0011(a, b);
      break;
    case _MM_SHUFFLE(0, 0, 2, 2):
      return _mm_shuffle_ps_0022(a, b);
      break;
    case _MM_SHUFFLE(2, 2, 0, 0):
      return _mm_shuffle_ps_2200(a, b);
      break;
    case _MM_SHUFFLE(3, 2, 0, 2):
      return _mm_shuffle_ps_3202(a, b);
      break;
    case _MM_SHUFFLE(1, 1, 3, 3):
      return _mm_shuffle_ps_1133(a, b);
      break;
    case _MM_SHUFFLE(2, 0, 1, 0):
      return _mm_shuffle_ps_2010(a, b);
      break;
    case _MM_SHUFFLE(2, 0, 0, 1):
      return _mm_shuffle_ps_2001(a, b);
      break;
    case _MM_SHUFFLE(2, 0, 3, 2):
      return _mm_shuffle_ps_2032(a, b);
      break;
    case _MM_SHUFFLE(0, 3, 2, 1):
      return _mm_shuffle_ps_0321(a, b);
      break;
    case _MM_SHUFFLE(2, 1, 0, 3):
      return _mm_shuffle_ps_2103(a, b);
      break;
    case _MM_SHUFFLE(1, 0, 1, 0):
      return _mm_shuffle_ps_1010(a, b);
      break;
    case _MM_SHUFFLE(1, 0, 0, 1):
      return _mm_shuffle_ps_1001(a, b);
      break;
    case _MM_SHUFFLE(0, 1, 0, 1):
      return _mm_shuffle_ps_0101(a, b);
      break;
  }
  return _mm_shuffle_ps_default<i>(a, b);
}

# if __has_builtin(__builtin_shufflevector)
#define _mm_shuffle_ps(a,b,i) _mm_shuffle_ps_default<i>(a,b)
# else
#define _mm_shuffle_ps(a,b,i) _mm_shuffle_ps_function<i>(a,b)
#endif

// Takes the upper 64 bits of a and places it in the low end of the result
// Takes the lower 64 bits of b and places it into the high end of the result.
FORCE_INLINE __m128i _mm_shuffle_epi_1032(__m128i a, __m128i b)
{
  return vcombine_s32(vget_high_s32(a), vget_low_s32(b));
}

// takes the lower two 32-bit values from a and swaps them and places in low end of result
// takes the higher two 32 bit values from b and swaps them and places in high end of result.
FORCE_INLINE __m128i _mm_shuffle_epi_2301(__m128i a, __m128i b)
{
  return vcombine_s32(vrev64_s32(vget_low_s32(a)), vrev64_s32(vget_high_s32(b)));
}

// shift a right by 32 bits, and put the lower 32 bits of a into the upper 32 bits of b
// when a and b are the same, rotates the least significant 32 bits into the most signficant 32 bits, and shifts the rest down
FORCE_INLINE __m128i _mm_shuffle_epi_0321(__m128i a, __m128i b)
{
  return vextq_s32(a, b, 1);
}

// shift a left by 32 bits, and put the upper 32 bits of b into the lower 32 bits of a
// when a and b are the same, rotates the most significant 32 bits into the least signficant 32 bits, and shifts the rest up
FORCE_INLINE __m128i _mm_shuffle_epi_2103(__m128i a, __m128i b)
{
  return vextq_s32(a, b, 3);
}

// gets the lower 64 bits of a, and places it in the upper 64 bits
// gets the lower 64 bits of b and places it in the lower 64 bits
FORCE_INLINE __m128i _mm_shuffle_epi_1010(__m128i a, __m128i b)
{
  return vcombine_s32(vget_low_s32(a), vget_low_s32(a));
}

// gets the lower 64 bits of a, and places it in the upper 64 bits
// gets the lower 64 bits of b, swaps the 0 and 1 elements, and places it in the lower 64 bits
FORCE_INLINE __m128i _mm_shuffle_epi_1001(__m128i a, __m128i b)
{
  return vcombine_s32(vrev64_s32(vget_low_s32(a)), vget_low_s32(b));
}

// gets the lower 64 bits of a, swaps the 0 and 1 elements and places it in the upper 64 bits
// gets the lower 64 bits of b, swaps the 0 and 1 elements, and places it in the lower 64 bits
FORCE_INLINE __m128i _mm_shuffle_epi_0101(__m128i a, __m128i b)
{
  return vcombine_s32(vrev64_s32(vget_low_s32(a)), vrev64_s32(vget_low_s32(b)));
}

FORCE_INLINE __m128i _mm_shuffle_epi_2211(__m128i a, __m128i b)
{
  return vcombine_s32(vdup_n_s32(vgetq_lane_s32(a, 1)), vdup_n_s32(vgetq_lane_s32(b, 2)));
}

FORCE_INLINE __m128i _mm_shuffle_epi_0122(__m128i a, __m128i b)
{
  return vcombine_s32(vdup_n_s32(vgetq_lane_s32(a, 2)), vrev64_s32(vget_low_s32(b)));
}

FORCE_INLINE __m128i _mm_shuffle_epi_3332(__m128i a, __m128i b)
{
  return vcombine_s32(vget_high_s32(a), vdup_n_s32(vgetq_lane_s32(b, 3)));
}

template <int i >
FORCE_INLINE __m128i _mm_shuffle_epi32_default(__m128i a, __m128i b)
{
#if ENABLE_CPP_VERSION
  __m128i ret;
  ret[0] = a[i & 0x3];
  ret[1] = a[(i >> 2) & 0x3];
  ret[2] = b[(i >> 4) & 0x03];
  ret[3] = b[(i >> 6) & 0x03];
  return ret;
#else
  __m128i ret = vmovq_n_s32(vgetq_lane_s32(a, i & 0x3));
  ret = vsetq_lane_s32(vgetq_lane_s32(a, (i >> 2) & 0x3), ret, 1);
  ret = vsetq_lane_s32(vgetq_lane_s32(b, (i >> 4) & 0x3), ret, 2);
  ret = vsetq_lane_s32(vgetq_lane_s32(b, (i >> 6) & 0x3), ret, 3);
  return ret;
#endif
}

template <int i >
FORCE_INLINE __m128i _mm_shuffle_epi32_function(__m128i a, __m128i b)
{
  switch (i)
  {
    case _MM_SHUFFLE(1, 0, 3, 2): return _mm_shuffle_epi_1032(a, b); break;
    case _MM_SHUFFLE(2, 3, 0, 1): return _mm_shuffle_epi_2301(a, b); break;
    case _MM_SHUFFLE(0, 3, 2, 1): return _mm_shuffle_epi_0321(a, b); break;
    case _MM_SHUFFLE(2, 1, 0, 3): return _mm_shuffle_epi_2103(a, b); break;
    case _MM_SHUFFLE(1, 0, 1, 0): return _mm_shuffle_epi_1010(a, b); break;
    case _MM_SHUFFLE(1, 0, 0, 1): return _mm_shuffle_epi_1001(a, b); break;
    case _MM_SHUFFLE(0, 1, 0, 1): return _mm_shuffle_epi_0101(a, b); break;
    case _MM_SHUFFLE(2, 2, 1, 1): return _mm_shuffle_epi_2211(a, b); break;
    case _MM_SHUFFLE(0, 1, 2, 2): return _mm_shuffle_epi_0122(a, b); break;
    case _MM_SHUFFLE(3, 3, 3, 2): return _mm_shuffle_epi_3332(a, b); break;
    default: return _mm_shuffle_epi32_default<i>(a, b);
  }
}

template <int i >
FORCE_INLINE __m128i _mm_shuffle_epi32_splat(__m128i a)
{
  return vdupq_n_s32(vgetq_lane_s32(a, i));
}

template <int i>
FORCE_INLINE __m128i _mm_shuffle_epi32_single(__m128i a)
{
  switch (i)
  {
    case _MM_SHUFFLE(0, 0, 0, 0): return _mm_shuffle_epi32_splat<0>(a); break;
    case _MM_SHUFFLE(1, 1, 1, 1): return _mm_shuffle_epi32_splat<1>(a); break;
    case _MM_SHUFFLE(2, 2, 2, 2): return _mm_shuffle_epi32_splat<2>(a); break;
    case _MM_SHUFFLE(3, 3, 3, 3): return _mm_shuffle_epi32_splat<3>(a); break;
    default: return _mm_shuffle_epi32_function<i>(a, a);
  }
}

// Shuffles the 4 signed or unsigned 32-bit integers in a as specified by imm.	https://msdn.microsoft.com/en-us/library/56f67xbk%28v=vs.90%29.aspx
#define _mm_shuffle_epi32(a,i) _mm_shuffle_epi32_single<i>(a)

template <int i>
FORCE_INLINE __m128i _mm_shufflehi_epi16_function(__m128i a)
{
  int16x8_t ret = (int16x8_t)a;
  int16x4_t highBits = vget_high_s16(ret);
  ret = vsetq_lane_s16(vget_lane_s16(highBits, i & 0x3), ret, 4);
  ret = vsetq_lane_s16(vget_lane_s16(highBits, (i >> 2) & 0x3), ret, 5);
  ret = vsetq_lane_s16(vget_lane_s16(highBits, (i >> 4) & 0x3), ret, 6);
  ret = vsetq_lane_s16(vget_lane_s16(highBits, (i >> 6) & 0x3), ret, 7);
  return (__m128i)ret;
}

// Shuffles the upper 4 signed or unsigned 16 - bit integers in a as specified by imm.  https://msdn.microsoft.com/en-us/library/13ywktbs(v=vs.100).aspx
#define _mm_shufflehi_epi16(a,i) _mm_shufflehi_epi16_function<i>(a)

// Shifts the 4 signed or unsigned 32-bit integers in a left by count bits while shifting in zeros. : https://msdn.microsoft.com/en-us/library/z2k3bbtb%28v=vs.90%29.aspx
//#define _mm_slli_epi32(a, imm) (__m128i)vshlq_n_s32(a,imm)

// Based on SIMDe
FORCE_INLINE __m128i _mm_slli_epi32(__m128i a, const int imm8)
{
#if defined(__aarch64__)
    const int32x4_t s = vdupq_n_s32(imm8);
    return vshlq_s32(a, s);
#else
  int32_t __attribute__((aligned(16))) data[4];
  vst1q_s32(data, a);
  const int s = (imm8 > 31) ? 0 : imm8;
  data[0] = data[0] << s;
  data[1] = data[1] << s;
  data[2] = data[2] << s;
  data[3] = data[3] << s;

  return vld1q_s32(data);
#endif
}


//Shifts the 4 signed or unsigned 32-bit integers in a right by count bits while shifting in zeros.  https://msdn.microsoft.com/en-us/library/w486zcfa(v=vs.100).aspx
//#define _mm_srli_epi32( a, imm ) (__m128i)vshrq_n_u32((uint32x4_t)a, imm)

// Based on SIMDe
FORCE_INLINE __m128i _mm_srli_epi32(__m128i a, const int imm8)
{
#if defined(__aarch64__)
    const int shift = (imm8 > 31) ? 0 : imm8;  // Unfortunately, we need to check for this case for embree.
    const int32x4_t s = vdupq_n_s32(-shift);
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), s));
#else
  int32_t __attribute__((aligned(16))) data[4];
  vst1q_s32(data, a);

  const int s = (imm8 > 31) ? 0 : imm8;

  data[0] = data[0] >> s;
  data[1] = data[1] >> s;
  data[2] = data[2] >> s;
  data[3] = data[3] >> s;

  return vld1q_s32(data);
#endif
}


// Shifts the 4 signed 32 - bit integers in a right by count bits while shifting in the sign bit.  https://msdn.microsoft.com/en-us/library/z1939387(v=vs.100).aspx
//#define _mm_srai_epi32( a, imm ) vshrq_n_s32(a, imm)

// Based on SIMDe
FORCE_INLINE __m128i _mm_srai_epi32(__m128i a, const int imm8)
{
#if defined(__aarch64__)
    const int32x4_t s = vdupq_n_s32(-imm8);
    return vshlq_s32(a, s);
#else
  int32_t __attribute__((aligned(16))) data[4];
  vst1q_s32(data, a);
  const uint32_t m = (uint32_t) ((~0U) << (32 - imm8));

  for (int i = 0; i < 4; i++) {
    uint32_t is_neg = ((uint32_t) (((data[i]) >> 31)));
    data[i] = (data[i] >> imm8) | (m * is_neg);
  }

  return vld1q_s32(data);
#endif
}

// Shifts the 128 - bit value in a right by imm bytes while shifting in zeros.imm must be an immediate. https://msdn.microsoft.com/en-us/library/305w28yz(v=vs.100).aspx
//#define _mm_srli_si128( a, imm ) (__m128i)vmaxq_s8((int8x16_t)a, vextq_s8((int8x16_t)a, vdupq_n_s8(0), imm))
#define _mm_srli_si128( a, imm ) (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), (imm))

// Shifts the 128-bit value in a left by imm bytes while shifting in zeros. imm must be an immediate.  https://msdn.microsoft.com/en-us/library/34d3k2kt(v=vs.100).aspx
#define _mm_slli_si128( a, imm ) (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 16 - (imm))

// NEON does not provide a version of this function, here is an article about some ways to repro the results.
// http://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
// Creates a 16-bit mask from the most significant bits of the 16 signed or unsigned 8-bit integers in a and zero extends the upper bits. https://msdn.microsoft.com/en-us/library/vstudio/s090c8fk(v=vs.100).aspx
FORCE_INLINE int _mm_movemask_epi8(__m128i _a)
{
  uint8x16_t input = (uint8x16_t)_a;
  const int8_t __attribute__((aligned(16))) xr[8] = { -7, -6, -5, -4, -3, -2, -1, 0 };
  uint8x8_t mask_and = vdup_n_u8(0x80);
  int8x8_t mask_shift = vld1_s8(xr);

  uint8x8_t lo = vget_low_u8(input);
  uint8x8_t hi = vget_high_u8(input);

  lo = vand_u8(lo, mask_and);
  lo = vshl_u8(lo, mask_shift);

  hi = vand_u8(hi, mask_and);
  hi = vshl_u8(hi, mask_shift);

  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);

  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);

  return ((hi[0] << 8) | (lo[0] & 0xFF));
}


// ******************************************
// Math operations
// ******************************************

// Subtracts the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/1zad2k61(v=vs.100).aspx
FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b)
{
  return vsubq_f32(a, b);
}

FORCE_INLINE __m128 _mm_sub_ss(__m128 a, __m128 b)
{
  return vsubq_f32(a, b);
}

// Subtracts the 4 signed or unsigned 32-bit integers of b from the 4 signed or unsigned 32-bit integers of a. https://msdn.microsoft.com/en-us/library/vstudio/fhh866h0(v=vs.100).aspx
FORCE_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b)
{
  return vsubq_s32(a, b);
}

// Adds the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/c9848chc(v=vs.100).aspx
FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
{
  return vaddq_f32(a, b);
}

// adds the scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/be94x2y6(v=vs.100).aspx
FORCE_INLINE __m128 _mm_add_ss(__m128 a, __m128 b)
{
  const float32_t     b0 = vgetq_lane_f32(b, 0);
  float32x4_t         value = vdupq_n_f32(0);

  //the upper values in the result must be the remnants of <a>.
  value = vsetq_lane_f32(b0, value, 0);
  return vaddq_f32(a, value);
}

// Adds the 4 signed or unsigned 32-bit integers in a to the 4 signed or unsigned 32-bit integers in b. https://msdn.microsoft.com/en-us/library/vstudio/09xs4fkk(v=vs.100).aspx
FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b)
{
  return vaddq_s32(a, b);
}

// Adds the 8 signed or unsigned 16-bit integers in a to the 8 signed or unsigned 16-bit integers in b. https://msdn.microsoft.com/en-us/library/fceha5k4(v=vs.100).aspx
FORCE_INLINE __m128i _mm_add_epi16(__m128i a, __m128i b)
{
  return (__m128i)vaddq_s16((int16x8_t)a, (int16x8_t)b);
}

// Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/9ks1472s(v=vs.100).aspx
FORCE_INLINE __m128i _mm_mullo_epi16(__m128i a, __m128i b)
{
  return (__m128i)vmulq_s16((int16x8_t)a, (int16x8_t)b);
}

// Multiplies the 4 signed or unsigned 32-bit integers from a by the 4 signed or unsigned 32-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/bb531409(v=vs.100).aspx
FORCE_INLINE __m128i _mm_mullo_epi32 (__m128i a, __m128i b)
{
  return (__m128i)vmulq_s32((int32x4_t)a,(int32x4_t)b);
}

// Multiplies the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/22kbk6t9(v=vs.100).aspx
FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b)
{
  return vmulq_f32(a, b);
}

FORCE_INLINE __m128 _mm_mul_ss(__m128 a, __m128 b)
{
  return vmulq_f32(a, b);
}

// Computes the approximations of reciprocals of the four single-precision, floating-point values of a. https://msdn.microsoft.com/en-us/library/vstudio/796k1tty(v=vs.100).aspx
FORCE_INLINE __m128 _mm_rcp_ps(__m128 in)
{
#if defined(BUILD_IOS)
  return vdivq_f32(vdupq_n_f32(1.0f),in);
    
#endif
    // Get an initial estimate of 1/in.
  float32x4_t reciprocal = vrecpeq_f32(in);

  // We only return estimated 1/in.
  // Newton-Raphon iteration shold be done in the outside of _mm_rcp_ps().

  // TODO(LTE): We could delete these ifdef?
  reciprocal = vmulq_f32(vrecpsq_f32(in, reciprocal), reciprocal);
  reciprocal = vmulq_f32(vrecpsq_f32(in, reciprocal), reciprocal);
  return reciprocal;

}

FORCE_INLINE __m128 _mm_rcp_ss(__m128 in)
{
  float32x4_t value;
  float32x4_t result = in;

  value = _mm_rcp_ps(in);
  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}

// Divides the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/edaw8147(v=vs.100).aspx
FORCE_INLINE __m128 _mm_div_ps(__m128 a, __m128 b)
{
#if defined(BUILD_IOS) 
  return vdivq_f32(a,b);
#else
  float32x4_t reciprocal = _mm_rcp_ps(b);
    
  reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
  reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);

  // Add one more round of newton-raphson since NEON's reciprocal estimation has less accuracy compared to SSE2's rcp.
  reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);

  // Another round for safety
  reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);

    
  return vmulq_f32(a, reciprocal);
#endif
}

// Divides the scalar single-precision floating point value of a by b.  https://msdn.microsoft.com/en-us/library/4y73xa49(v=vs.100).aspx
FORCE_INLINE __m128 _mm_div_ss(__m128 a, __m128 b)
{
  float32x4_t value;
  float32x4_t result = a;
  value = _mm_div_ps(a, b);
  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}

// Computes the approximations of the reciprocal square roots of the four single-precision floating point values of in.  https://msdn.microsoft.com/en-us/library/22hfsh53(v=vs.100).aspx
FORCE_INLINE __m128 _mm_rsqrt_ps(__m128 in)
{
	
  float32x4_t value = vrsqrteq_f32(in);
  
  // TODO: We must debug and ensure that rsqrt(0) and rsqrt(-0) yield proper values.
  // Related code snippets can be found here: https://cpp.hotexamples.com/examples/-/-/vrsqrteq_f32/cpp-vrsqrteq_f32-function-examples.html
  // If we adapt this function, we might be able to avoid special zero treatment in _mm_sqrt_ps
  
  value = vmulq_f32(value, vrsqrtsq_f32(vmulq_f32(in, value), value));
  value = vmulq_f32(value, vrsqrtsq_f32(vmulq_f32(in, value), value));

  // one more round to get better precision
  value = vmulq_f32(value, vrsqrtsq_f32(vmulq_f32(in, value), value));

  // another round for safety
  value = vmulq_f32(value, vrsqrtsq_f32(vmulq_f32(in, value), value));

  return value;
}

FORCE_INLINE __m128 _mm_rsqrt_ss(__m128 in)
{
  float32x4_t result = in;
  
  __m128 value = _mm_rsqrt_ps(in);

  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}


// Computes the approximations of square roots of the four single-precision, floating-point values of a. First computes reciprocal square roots and then reciprocals of the four values. https://msdn.microsoft.com/en-us/library/vstudio/8z67bwwk(v=vs.100).aspx
FORCE_INLINE __m128 _mm_sqrt_ps(__m128 in)
{
#if defined(BUILD_IOS)
  return vsqrtq_f32(in);
#else
  __m128 reciprocal = _mm_rsqrt_ps(in);
  
  // We must treat sqrt(in == 0) in a special way. At this point reciprocal contains gargabe due to vrsqrteq_f32(0) returning +inf.
  // We assign 0 to reciprocal wherever required.
  const float32x4_t vzero = vdupq_n_f32(0.0f);
  const uint32x4_t mask = vceqq_f32(in, vzero);
  reciprocal = vbslq_f32(mask, vzero, reciprocal);
  
  // sqrt(x) = x * (1 / sqrt(x))
  return vmulq_f32(in, reciprocal);
#endif
}

// Computes the approximation of the square root of the scalar single-precision floating point value of in.  https://msdn.microsoft.com/en-us/library/ahfsc22d(v=vs.100).aspx
FORCE_INLINE __m128 _mm_sqrt_ss(__m128 in)
{
  float32x4_t value;
  float32x4_t result = in;

  value = _mm_sqrt_ps(in);
  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}


// Computes the maximums of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/ff5d607a(v=vs.100).aspx
FORCE_INLINE __m128 _mm_max_ps(__m128 a, __m128 b)
{
#if USE_PRECISE_MINMAX_IMPLEMENTATION
  return vbslq_f32(vcltq_f32(b,a),a,b);
#else
  // Faster, but would give inconsitent rendering(e.g. holes, NaN pixels)
  return vmaxq_f32(a, b);
#endif
}

// Computes the minima of the four single-precision, floating-point values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/wh13kadz(v=vs.100).aspx
FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b)
{
#if USE_PRECISE_MINMAX_IMPLEMENTATION
  return vbslq_f32(vcltq_f32(a,b),a,b);
#else
  // Faster, but would give inconsitent rendering(e.g. holes, NaN pixels)
  return vminq_f32(a, b);
#endif
}

// Computes the maximum of the two lower scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/s6db5esz(v=vs.100).aspx
FORCE_INLINE __m128 _mm_max_ss(__m128 a, __m128 b)
{
  float32x4_t value;
  float32x4_t result = a;
 
  value = _mm_max_ps(a, b);
  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}

// Computes the minimum of the two lower scalar single-precision floating point values of a and b.  https://msdn.microsoft.com/en-us/library/0a9y7xaa(v=vs.100).aspx
FORCE_INLINE __m128 _mm_min_ss(__m128 a, __m128 b)
{
  float32x4_t value;
  float32x4_t result = a;

    
  value = _mm_min_ps(a, b);
  return vsetq_lane_f32(vgetq_lane_f32(value, 0), result, 0);
}

// Computes the pairwise minima of the 8 signed 16-bit integers from a and the 8 signed 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/6te997ew(v=vs.100).aspx
FORCE_INLINE __m128i _mm_min_epi16(__m128i a, __m128i b)
{
  return (__m128i)vminq_s16((int16x8_t)a, (int16x8_t)b);
}

// epi versions of min/max
// Computes the pariwise maximums of the four signed 32-bit integer values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/bb514055(v=vs.100).aspx
FORCE_INLINE __m128i _mm_max_epi32(__m128i a, __m128i b )
{
  return vmaxq_s32(a,b);
}

// Computes the pariwise minima of the four signed 32-bit integer values of a and b. https://msdn.microsoft.com/en-us/library/vstudio/bb531476(v=vs.100).aspx
FORCE_INLINE __m128i _mm_min_epi32(__m128i a, __m128i b )
{
  return vminq_s32(a,b);
}

// Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b. https://msdn.microsoft.com/en-us/library/vstudio/59hddw1d(v=vs.100).aspx
FORCE_INLINE __m128i _mm_mulhi_epi16(__m128i a, __m128i b)
{
  int16x8_t ret = vqdmulhq_s16((int16x8_t)a, (int16x8_t)b);
  ret = vshrq_n_s16(ret, 1);
  return (__m128i)ret;
}

// Computes pairwise add of each argument as single-precision, floating-point values a and b.
//https://msdn.microsoft.com/en-us/library/yd9wecaa.aspx
FORCE_INLINE __m128 _mm_hadd_ps(__m128 a, __m128 b )
{
#if defined(__aarch64__)
    return vpaddq_f32(a,b);
#else
// This does not work, no vpaddq...
//	return (__m128) vpaddq_f32(a,b);
        //
        // get two f32x2_t values from a
        // do vpadd
        // put result in low half of f32x4 result
        //
        // get two f32x2_t values from b
        // do vpadd
        // put result in high half of f32x4 result
        //
        // combine
        return vcombine_f32( vpadd_f32( vget_low_f32(a), vget_high_f32(a) ), vpadd_f32( vget_low_f32(b), vget_high_f32(b) ) );
#endif
}

// ******************************************
// Compare operations
// ******************************************

// Compares for less than https://msdn.microsoft.com/en-us/library/vstudio/f330yhc8(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmplt_ps(__m128 a, __m128 b)
{
  return (__m128)vcltq_f32(a, b);
}

FORCE_INLINE __m128 _mm_cmpnlt_ps(__m128 a, __m128 b)
{
  return (__m128) vmvnq_s32((__m128i)_mm_cmplt_ps(a,b));
}

// Compares for greater than. https://msdn.microsoft.com/en-us/library/vstudio/11dy102s(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
  return (__m128)vcgtq_f32(a, b);
}

FORCE_INLINE __m128 _mm_cmpnle_ps(__m128 a, __m128 b)
{
  return (__m128) _mm_cmpgt_ps(a,b);
}


// Compares for greater than or equal. https://msdn.microsoft.com/en-us/library/vstudio/fs813y2t(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmpge_ps(__m128 a, __m128 b)
{
  return (__m128)vcgeq_f32(a, b);
}

// Compares for less than or equal. https://msdn.microsoft.com/en-us/library/vstudio/1s75w83z(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
  return (__m128)vcleq_f32(a, b);
}

// Compares for equality. https://msdn.microsoft.com/en-us/library/vstudio/36aectz5(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cmpeq_ps(__m128 a, __m128 b)
{
  return (__m128)vceqq_f32(a, b);
}

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for less than. https://msdn.microsoft.com/en-us/library/vstudio/4ak0bf5d(v=vs.100).aspx
FORCE_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b)
{
  return (__m128i)vcltq_s32(a, b);
}

FORCE_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
{
  return (__m128i) vceqq_s32(a,b);
}

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for greater than. https://msdn.microsoft.com/en-us/library/vstudio/1s9f2z0y(v=vs.100).aspx
FORCE_INLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)
{
  return (__m128i)vcgtq_s32(a, b);
}

// Compares the four 32-bit floats in a and b to check if any values are NaN. Ordered compare between each value returns true for "orderable" and false for "not orderable" (NaN). https://msdn.microsoft.com/en-us/library/vstudio/0h9w00fx(v=vs.100).aspx
// see also:
// http://stackoverflow.com/questions/8627331/what-does-ordered-unordered-comparison-mean
// http://stackoverflow.com/questions/29349621/neon-isnanval-intrinsics
FORCE_INLINE __m128 _mm_cmpord_ps(__m128 a, __m128 b )
{
  // Note: NEON does not have ordered compare builtin
  // Need to compare a eq a and b eq b to check for NaN
  // Do AND of results to get final
  return (__m128) vreinterpretq_f32_u32( vandq_u32( vceqq_f32(a,a), vceqq_f32(b,b) ) );
}

// Compares the lower single-precision floating point scalar values of a and b using a less than operation. : https://msdn.microsoft.com/en-us/library/2kwe606b(v=vs.90).aspx
FORCE_INLINE int _mm_comilt_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vcltq_f32(a, b);
  return vgetq_lane_u32(value, 0);
}

// Compares the lower single-precision floating point scalar values of a and b using a greater than operation. : https://msdn.microsoft.com/en-us/library/b0738e0t(v=vs.100).aspx
FORCE_INLINE int _mm_comigt_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vcgtq_f32(a, b);
  return vgetq_lane_u32(value, 0);
}

// Compares the lower single-precision floating point scalar values of a and b using a less than or equal operation. : https://msdn.microsoft.com/en-us/library/1w4t7c57(v=vs.90).aspx
FORCE_INLINE int _mm_comile_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vcleq_f32(a, b);
  return vgetq_lane_u32(value, 0);
}

// Compares the lower single-precision floating point scalar values of a and b using a greater than or equal operation. : https://msdn.microsoft.com/en-us/library/8t80des6(v=vs.100).aspx
FORCE_INLINE int _mm_comige_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vcgeq_f32(a, b);
  return vgetq_lane_u32(value, 0);
}

// Compares the lower single-precision floating point scalar values of a and b using an equality operation. : https://msdn.microsoft.com/en-us/library/93yx2h2b(v=vs.100).aspx
FORCE_INLINE int _mm_comieq_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vceqq_f32(a, b);
  return vgetq_lane_u32(value, 0);
}

// Compares the lower single-precision floating point scalar values of a and b using an inequality operation. : https://msdn.microsoft.com/en-us/library/bafh5e0a(v=vs.90).aspx
FORCE_INLINE int _mm_comineq_ss(__m128 a, __m128 b)
{
  uint32x4_t value;

  value = vceqq_f32(a, b);
  return !vgetq_lane_u32(value, 0);
}

// according to the documentation, these intrinsics behave the same as the non-'u' versions.  We'll just alias them here.
#define _mm_ucomilt_ss      _mm_comilt_ss
#define _mm_ucomile_ss      _mm_comile_ss
#define _mm_ucomigt_ss      _mm_comigt_ss
#define _mm_ucomige_ss      _mm_comige_ss
#define _mm_ucomieq_ss      _mm_comieq_ss
#define _mm_ucomineq_ss     _mm_comineq_ss

// ******************************************
// Conversions
// ******************************************

// Converts the four single-precision, floating-point values of a to signed 32-bit integer values using truncate. https://msdn.microsoft.com/en-us/library/vstudio/1h005y6x(v=vs.100).aspx
FORCE_INLINE __m128i _mm_cvttps_epi32(__m128 a)
{
  return vcvtq_s32_f32(a);
}

// Converts the four signed 32-bit integer values of a to single-precision, floating-point values https://msdn.microsoft.com/en-us/library/vstudio/36bwxcx5(v=vs.100).aspx
FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
  return vcvtq_f32_s32(a);
}

// Converts the four single-precision, floating-point values of a to signed 32-bit integer values. https://msdn.microsoft.com/en-us/library/vstudio/xdc42k5e(v=vs.100).aspx
// *NOTE*. The default rounding mode on SSE is 'round to even', which ArmV7 does not support!
// It is supported on ARMv8 however.
FORCE_INLINE __m128i _mm_cvtps_epi32(__m128 a)
{
#if 1
  return vcvtnq_s32_f32(a);
#else
  __m128 half = vdupq_n_f32(0.5f);
  const __m128 sign = vcvtq_f32_u32((vshrq_n_u32(vreinterpretq_u32_f32(a), 31)));
  const __m128 aPlusHalf = vaddq_f32(a, half);
  const __m128 aRound = vsubq_f32(aPlusHalf, sign);
  return vcvtq_s32_f32(aRound);
#endif
}

// Moves the least significant 32 bits of a to a 32-bit integer. https://msdn.microsoft.com/en-us/library/5z7a9642%28v=vs.90%29.aspx
FORCE_INLINE int _mm_cvtsi128_si32(__m128i a)
{
  return vgetq_lane_s32(a, 0);
}

// Moves 32-bit integer a to the least significant 32 bits of an __m128 object, zero extending the upper bits. https://msdn.microsoft.com/en-us/library/ct3539ha%28v=vs.90%29.aspx
FORCE_INLINE __m128i _mm_cvtsi32_si128(int a)
{
  __m128i result = vdupq_n_s32(0);
  return vsetq_lane_s32(a, result, 0);
}


// Applies a type cast to reinterpret four 32-bit floating point values passed in as a 128-bit parameter as packed 32-bit integers. https://msdn.microsoft.com/en-us/library/bb514099.aspx
FORCE_INLINE __m128i _mm_castps_si128(__m128 a)
{
#if defined(__aarch64__)
    return (__m128i)a;
#else
  return *(const __m128i *)&a;
#endif
}

// Applies a type cast to reinterpret four 32-bit integers passed in as a 128-bit parameter as packed 32-bit floating point values. https://msdn.microsoft.com/en-us/library/bb514029.aspx
FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a)
{
#if defined(__aarch64__)
    return (__m128)a;
#else
  return *(const __m128 *)&a;
#endif
}

// Loads 128-bit value. : https://msdn.microsoft.com/en-us/library/atzzad1h(v=vs.80).aspx
FORCE_INLINE __m128i _mm_load_si128(const __m128i *p)
{
  return vld1q_s32((int32_t *)p);
}

FORCE_INLINE __m128d _mm_castps_pd(const __m128 a)
{
  return *(const __m128d *)&a;
}

FORCE_INLINE __m128d _mm_castsi128_pd(__m128i a)
{
  return *(const __m128d *)&a;
}
// ******************************************
// Miscellaneous Operations
// ******************************************

// Packs the 16 signed 16-bit integers from a and b into 8-bit integers and saturates. https://msdn.microsoft.com/en-us/library/k4y4f7w5%28v=vs.90%29.aspx
FORCE_INLINE __m128i _mm_packs_epi16(__m128i a, __m128i b)
{
  return (__m128i)vcombine_s8(vqmovn_s16((int16x8_t)a), vqmovn_s16((int16x8_t)b));
}

// Packs the 16 signed 16 - bit integers from a and b into 8 - bit unsigned integers and saturates. https://msdn.microsoft.com/en-us/library/07ad1wx4(v=vs.100).aspx
FORCE_INLINE __m128i _mm_packus_epi16(const __m128i a, const __m128i b)
{
  return (__m128i)vcombine_u8(vqmovun_s16((int16x8_t)a), vqmovun_s16((int16x8_t)b));
}

// Packs the 8 signed 32-bit integers from a and b into signed 16-bit integers and saturates. https://msdn.microsoft.com/en-us/library/393t56f9%28v=vs.90%29.aspx
FORCE_INLINE __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
  return (__m128i)vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
}

// Interleaves the lower 8 signed or unsigned 8-bit integers in a with the lower 8 signed or unsigned 8-bit integers in b.  https://msdn.microsoft.com/en-us/library/xf7k860c%28v=vs.90%29.aspx
FORCE_INLINE __m128i _mm_unpacklo_epi8(__m128i a, __m128i b)
{
  int8x8_t a1 = (int8x8_t)vget_low_s16((int16x8_t)a);
  int8x8_t b1 = (int8x8_t)vget_low_s16((int16x8_t)b);

  int8x8x2_t result = vzip_s8(a1, b1);

  return (__m128i)vcombine_s8(result.val[0], result.val[1]);
}

// Interleaves the lower 4 signed or unsigned 16-bit integers in a with the lower 4 signed or unsigned 16-bit integers in b.  https://msdn.microsoft.com/en-us/library/btxb17bw%28v=vs.90%29.aspx
FORCE_INLINE __m128i _mm_unpacklo_epi16(__m128i a, __m128i b)
{
  int16x4_t a1 = vget_low_s16((int16x8_t)a);
  int16x4_t b1 = vget_low_s16((int16x8_t)b);

  int16x4x2_t result = vzip_s16(a1, b1);

  return (__m128i)vcombine_s16(result.val[0], result.val[1]);
}

// Interleaves the lower 2 signed or unsigned 32 - bit integers in a with the lower 2 signed or unsigned 32 - bit integers in b.  https://msdn.microsoft.com/en-us/library/x8atst9d(v=vs.100).aspx
FORCE_INLINE __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)
{
  int32x2_t a1 = vget_low_s32(a);
  int32x2_t b1 = vget_low_s32(b);

  int32x2x2_t result = vzip_s32(a1, b1);

  return vcombine_s32(result.val[0], result.val[1]);
}

// Selects and interleaves the lower two single-precision, floating-point values from a and b. https://msdn.microsoft.com/en-us/library/25st103b%28v=vs.90%29.aspx
FORCE_INLINE __m128 _mm_unpacklo_ps(__m128 a, __m128 b)
{
  float32x2x2_t result = vzip_f32(vget_low_f32(a), vget_low_f32(b));
  return vcombine_f32(result.val[0], result.val[1]);
}

// Selects and interleaves the upper two single-precision, floating-point values from a and b. https://msdn.microsoft.com/en-us/library/skccxx7d%28v=vs.90%29.aspx
FORCE_INLINE __m128 _mm_unpackhi_ps(__m128 a, __m128 b)
{
  float32x2x2_t result = vzip_f32(vget_high_f32(a), vget_high_f32(b));
  return vcombine_f32(result.val[0], result.val[1]);
}

// Interleaves the upper 8 signed or unsigned 8-bit integers in a with the upper 8 signed or unsigned 8-bit integers in b.  https://msdn.microsoft.com/en-us/library/t5h7783k(v=vs.100).aspx
FORCE_INLINE __m128i _mm_unpackhi_epi8(__m128i a, __m128i b)
{
  int8x8_t a1 = (int8x8_t)vget_high_s16((int16x8_t)a);
  int8x8_t b1 = (int8x8_t)vget_high_s16((int16x8_t)b);

  int8x8x2_t result = vzip_s8(a1, b1);

  return (__m128i)vcombine_s8(result.val[0], result.val[1]);
}

// Interleaves the upper 4 signed or unsigned 16-bit integers in a with the upper 4 signed or unsigned 16-bit integers in b.  https://msdn.microsoft.com/en-us/library/03196cz7(v=vs.100).aspx
FORCE_INLINE __m128i _mm_unpackhi_epi16(__m128i a, __m128i b)
{
  int16x4_t a1 = vget_high_s16((int16x8_t)a);
  int16x4_t b1 = vget_high_s16((int16x8_t)b);

  int16x4x2_t result = vzip_s16(a1, b1);

  return (__m128i)vcombine_s16(result.val[0], result.val[1]);
}

// Interleaves the upper 2 signed or unsigned 32-bit integers in a with the upper 2 signed or unsigned 32-bit integers in b.  https://msdn.microsoft.com/en-us/library/65sa7cbs(v=vs.100).aspx
FORCE_INLINE __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)
{
  int32x2_t a1 = vget_high_s32(a);
  int32x2_t b1 = vget_high_s32(b);

  int32x2x2_t result = vzip_s32(a1, b1);

  return vcombine_s32(result.val[0], result.val[1]);
}

// Extracts the selected signed or unsigned 16-bit integer from a and zero extends.  https://msdn.microsoft.com/en-us/library/6dceta0c(v=vs.100).aspx
#define _mm_extract_epi16( a, imm ) vgetq_lane_s16((int16x8_t)a, imm)

// ******************************************
// Streaming Extensions
// ******************************************

// Guarantees that every preceding store is globally visible before any subsequent store.  https://msdn.microsoft.com/en-us/library/5h2w73d1%28v=vs.90%29.aspx
FORCE_INLINE void _mm_sfence(void)
{
  __sync_synchronize();
}

// Stores the data in a to the address p without polluting the caches.  If the cache line containing address p is already in the cache, the cache will be updated.Address p must be 16 - byte aligned.  https://msdn.microsoft.com/en-us/library/ba08y07y%28v=vs.90%29.aspx
FORCE_INLINE void _mm_stream_si128(__m128i *p, __m128i a)
{
  *p = a;
}

// Cache line containing p is flushed and invalidated from all caches in the coherency domain. : https://msdn.microsoft.com/en-us/library/ba08y07y(v=vs.100).aspx
FORCE_INLINE void _mm_clflush(void const*p)
{
  // no corollary for Neon?
}

FORCE_INLINE __m128i _mm_set_epi64x(int64_t a, int64_t b)
{
  // Stick to the flipped behavior of x86.
  int64_t __attribute__((aligned(16))) data[2] = { b, a };
  return (__m128i)vld1q_s64(data);
}

FORCE_INLINE __m128i _mm_set1_epi64x(int64_t _i)
{
  return (__m128i)vmovq_n_s64(_i);
}

#if defined(__aarch64__)
FORCE_INLINE __m128 _mm_blendv_ps(__m128 a, __m128 b, __m128 c)
{
    int32x4_t mask = vshrq_n_s32(__m128i(c),31);
    return vbslq_f32( uint32x4_t(mask), b, a);
}

FORCE_INLINE __m128i _mm_load4epu8_epi32(__m128i *ptr)
{
    uint8x8_t  t0 = vld1_u8((uint8_t*)ptr);
    uint16x8_t t1 = vmovl_u8(t0);
    uint32x4_t t2 = vmovl_u16(vget_low_u16(t1));
    return vreinterpretq_s32_u32(t2);
}

FORCE_INLINE __m128i _mm_load4epu16_epi32(__m128i *ptr)
{
    uint16x8_t t0 = vld1q_u16((uint16_t*)ptr);
    uint32x4_t t1 = vmovl_u16(vget_low_u16(t0));
    return vreinterpretq_s32_u32(t1);
}

FORCE_INLINE __m128i _mm_load4epi8_f32(__m128i *ptr)
{
    int8x8_t    t0 = vld1_s8((int8_t*)ptr);
    int16x8_t   t1 = vmovl_s8(t0);
    int32x4_t   t2 = vmovl_s16(vget_low_s16(t1));
    float32x4_t t3 = vcvtq_f32_s32(t2);
    return vreinterpretq_s32_f32(t3);
}

FORCE_INLINE __m128i _mm_load4epu8_f32(__m128i *ptr)
{
    uint8x8_t   t0 = vld1_u8((uint8_t*)ptr);
    uint16x8_t  t1 = vmovl_u8(t0);
    uint32x4_t  t2 = vmovl_u16(vget_low_u16(t1));
    return vreinterpretq_s32_u32(t2);
}

FORCE_INLINE __m128i _mm_load4epi16_f32(__m128i *ptr)
{
    int16x8_t   t0 = vld1q_s16((int16_t*)ptr);
    int32x4_t   t1 = vmovl_s16(vget_low_s16(t0));
    float32x4_t t2 = vcvtq_f32_s32(t1);
    return vreinterpretq_s32_f32(t2);
}

FORCE_INLINE __m128i _mm_packus_epi32(__m128i a, __m128i b)
{
    return (__m128i)vcombine_u8(vqmovun_s16((int16x8_t)a), vqmovun_s16((int16x8_t)b));
}

FORCE_INLINE __m128i _mm_stream_load_si128(__m128i* ptr)
{
    // No non-temporal load on a single register on ARM.
    return vreinterpretq_s32_u8(vld1q_u8((uint8_t*)ptr));
}

FORCE_INLINE void _mm_stream_ps(float* ptr, __m128i a)
{
    // No non-temporal store on a single register on ARM.
    vst1q_f32((float*)ptr, vreinterpretq_f32_s32(a));
}

FORCE_INLINE __m128i _mm_min_epu32(__m128i a, __m128i b)
{
    return vreinterpretq_s32_u32(vminq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b)));
}

FORCE_INLINE __m128i _mm_max_epu32(__m128i a, __m128i b)
{
    return vreinterpretq_s32_u32(vmaxq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b)));
}

FORCE_INLINE __m128 _mm_abs_ps(__m128 a)
{
    return vabsq_f32(a);
}

FORCE_INLINE __m128 _mm_madd_ps(__m128 a, __m128 b, __m128 c)
{
    return vmlaq_f32(c, a, b);
}

FORCE_INLINE __m128 _mm_msub_ps(__m128 a, __m128 b, __m128 c)
{
    return vmlsq_f32(c, a, b);
}

FORCE_INLINE __m128i _mm_abs_epi32(__m128i a)
{
  return vabsq_s32(a);
}
#endif  //defined(__aarch64__)

// Count the number of bits set to 1 in unsigned 32-bit integer a, and
// return that count in dst.
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_popcnt_u32
FORCE_INLINE int _mm_popcnt_u32(unsigned int a)
{
  return (int)vaddlv_u8(vcnt_u8(vcreate_u8((uint64_t)a)));
}

// Count the number of bits set to 1 in unsigned 64-bit integer a, and
// return that count in dst.
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_popcnt_u64
FORCE_INLINE int64_t _mm_popcnt_u64(uint64_t a)
{
  return (int64_t)vaddlv_u8(vcnt_u8(vcreate_u8(a)));
}

#endif
