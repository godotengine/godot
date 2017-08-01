#ifndef CLEW_HPP_INCLUDED
#define CLEW_HPP_INCLUDED

//////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009-2011 Organic Vectory B.V., KindDragon
//  Written by George van Venrooij
//
//  Distributed under the MIT License.
//////////////////////////////////////////////////////////////////////////

//! \file clew.h
//! \brief OpenCL run-time loader header
//!
//! This file contains a copy of the contents of CL.H and CL_PLATFORM.H from the 
//! official OpenCL spec. The purpose of this code is to load the OpenCL dynamic
//! library at run-time and thus allow the executable to function on many
//! platforms regardless of the vendor of the OpenCL driver actually installed.
//! Some of the techniques used here were inspired by work done in the GLEW
//! library (http://glew.sourceforge.net/)

//  Run-time dynamic linking functionality based on concepts used in GLEW
#ifdef  __OPENCL_CL_H
#error cl.h included before clew.h
#endif

#ifdef  __OPENCL_CL_PLATFORM_H
#error cl_platform.h included before clew.h
#endif

//  Prevent cl.h inclusion
#define __OPENCL_CL_H
//  Prevent cl_platform.h inclusion
#define __CL_PLATFORM_H

/*******************************************************************************
* Copyright (c) 2008-2010 The Khronos Group Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and/or associated documentation files (the
* "Materials"), to deal in the Materials without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Materials, and to
* permit persons to whom the Materials are furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Materials.
*
* THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
******************************************************************************/
#ifdef __APPLE__
    /* Contains #defines for AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER below */
    #include <AvailabilityMacros.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
    #define CL_API_ENTRY
    #define CL_API_CALL     __stdcall
    #define CL_CALLBACK     __stdcall
#else
    #define CL_API_ENTRY
    #define CL_API_CALL
    #define CL_CALLBACK
#endif
//disabled the APPLE thing, don't know why it is there, is just causes tons of warnings

#ifdef __APPLE1__
    #define CL_EXTENSION_WEAK_LINK                  __attribute__((weak_import))       
    #define CL_API_SUFFIX__VERSION_1_0              AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER
    #define CL_EXT_SUFFIX__VERSION_1_0              CL_EXTENSION_WEAK_LINK AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER
    #define CL_API_SUFFIX__VERSION_1_1              CL_EXTENSION_WEAK_LINK
    #define CL_EXT_SUFFIX__VERSION_1_1              CL_EXTENSION_WEAK_LINK
    #define CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED   CL_EXTENSION_WEAK_LINK AVAILABLE_MAC_OS_X_VERSION_10_6_AND_LATER
#else
    #define CL_EXTENSION_WEAK_LINK                         
    #define CL_API_SUFFIX__VERSION_1_0
    #define CL_EXT_SUFFIX__VERSION_1_0
    #define CL_API_SUFFIX__VERSION_1_1
    #define CL_EXT_SUFFIX__VERSION_1_1
    #define CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED
#endif

#if (defined (_WIN32) && defined(_MSC_VER))

/* scalar types  */
typedef signed   __int8         cl_char;
typedef unsigned __int8         cl_uchar;
typedef signed   __int16        cl_short;
typedef unsigned __int16        cl_ushort;
typedef signed   __int32        cl_int;
typedef unsigned __int32        cl_uint;
typedef signed   __int64        cl_long;
typedef unsigned __int64        cl_ulong;

typedef unsigned __int16        cl_half;
typedef float                   cl_float;
typedef double                  cl_double;

/* Macro names and corresponding values defined by OpenCL */
#define CL_CHAR_BIT         8
#define CL_SCHAR_MAX        127
#define CL_SCHAR_MIN        (-127-1)
#define CL_CHAR_MAX         CL_SCHAR_MAX
#define CL_CHAR_MIN         CL_SCHAR_MIN
#define CL_UCHAR_MAX        255
#define CL_SHRT_MAX         32767
#define CL_SHRT_MIN         (-32767-1)
#define CL_USHRT_MAX        65535
#define CL_INT_MAX          2147483647
#define CL_INT_MIN          (-2147483647-1)
#define CL_UINT_MAX         0xffffffffU
#define CL_LONG_MAX         ((cl_long) 0x7FFFFFFFFFFFFFFFLL)
#define CL_LONG_MIN         ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define CL_ULONG_MAX        ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL)

#define CL_FLT_DIG          6
#define CL_FLT_MANT_DIG     24
#define CL_FLT_MAX_10_EXP   +38
#define CL_FLT_MAX_EXP      +128
#define CL_FLT_MIN_10_EXP   -37
#define CL_FLT_MIN_EXP      -125
#define CL_FLT_RADIX        2
#define CL_FLT_MAX          340282346638528859811704183484516925440.0f
#define CL_FLT_MIN          1.175494350822287507969e-38f
#define CL_FLT_EPSILON      0x1.0p-23f

#define CL_DBL_DIG          15
#define CL_DBL_MANT_DIG     53
#define CL_DBL_MAX_10_EXP   +308
#define CL_DBL_MAX_EXP      +1024
#define CL_DBL_MIN_10_EXP   -307
#define CL_DBL_MIN_EXP      -1021
#define CL_DBL_RADIX        2
#define CL_DBL_MAX          179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0
#define CL_DBL_MIN          2.225073858507201383090e-308
#define CL_DBL_EPSILON      2.220446049250313080847e-16

#define  CL_M_E             2.718281828459045090796
#define  CL_M_LOG2E         1.442695040888963387005
#define  CL_M_LOG10E        0.434294481903251816668
#define  CL_M_LN2           0.693147180559945286227
#define  CL_M_LN10          2.302585092994045901094
#define  CL_M_PI            3.141592653589793115998
#define  CL_M_PI_2          1.570796326794896557999
#define  CL_M_PI_4          0.785398163397448278999
#define  CL_M_1_PI          0.318309886183790691216
#define  CL_M_2_PI          0.636619772367581382433
#define  CL_M_2_SQRTPI      1.128379167095512558561
#define  CL_M_SQRT2         1.414213562373095145475
#define  CL_M_SQRT1_2       0.707106781186547572737

#define  CL_M_E_F           2.71828174591064f
#define  CL_M_LOG2E_F       1.44269502162933f
#define  CL_M_LOG10E_F      0.43429449200630f
#define  CL_M_LN2_F         0.69314718246460f
#define  CL_M_LN10_F        2.30258512496948f
#define  CL_M_PI_F          3.14159274101257f
#define  CL_M_PI_2_F        1.57079637050629f
#define  CL_M_PI_4_F        0.78539818525314f
#define  CL_M_1_PI_F        0.31830987334251f
#define  CL_M_2_PI_F        0.63661974668503f
#define  CL_M_2_SQRTPI_F    1.12837922573090f
#define  CL_M_SQRT2_F       1.41421353816986f
#define  CL_M_SQRT1_2_F     0.70710676908493f

#define CL_NAN              (CL_INFINITY - CL_INFINITY)
#define CL_HUGE_VALF        ((cl_float) 1e50)
#define CL_HUGE_VAL         ((cl_double) 1e500)
#define CL_MAXFLOAT         CL_FLT_MAX
#define CL_INFINITY         CL_HUGE_VALF

#else

#include <stdint.h>

/* scalar types  */
typedef int8_t          cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t         cl_short    __attribute__((aligned(2)));
typedef uint16_t        cl_ushort   __attribute__((aligned(2)));
typedef int32_t         cl_int      __attribute__((aligned(4)));
typedef uint32_t        cl_uint     __attribute__((aligned(4)));
typedef int64_t         cl_long     __attribute__((aligned(8)));
typedef uint64_t        cl_ulong    __attribute__((aligned(8)));

typedef uint16_t        cl_half     __attribute__((aligned(2)));
typedef float           cl_float    __attribute__((aligned(4)));
typedef double          cl_double   __attribute__((aligned(8)));

/* Macro names and corresponding values defined by OpenCL */
#define CL_CHAR_BIT         8
#define CL_SCHAR_MAX        127
#define CL_SCHAR_MIN        (-127-1)
#define CL_CHAR_MAX         CL_SCHAR_MAX
#define CL_CHAR_MIN         CL_SCHAR_MIN
#define CL_UCHAR_MAX        255
#define CL_SHRT_MAX         32767
#define CL_SHRT_MIN         (-32767-1)
#define CL_USHRT_MAX        65535
#define CL_INT_MAX          2147483647
#define CL_INT_MIN          (-2147483647-1)
#define CL_UINT_MAX         0xffffffffU
#define CL_LONG_MAX         ((cl_long) 0x7FFFFFFFFFFFFFFFLL)
#define CL_LONG_MIN         ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define CL_ULONG_MAX        ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL)

#define CL_FLT_DIG          6
#define CL_FLT_MANT_DIG     24
#define CL_FLT_MAX_10_EXP   +38
#define CL_FLT_MAX_EXP      +128
#define CL_FLT_MIN_10_EXP   -37
#define CL_FLT_MIN_EXP      -125
#define CL_FLT_RADIX        2
#define CL_FLT_MAX          0x1.fffffep127f
#define CL_FLT_MIN          0x1.0p-126f
#define CL_FLT_EPSILON      0x1.0p-23f

#define CL_DBL_DIG          15
#define CL_DBL_MANT_DIG     53
#define CL_DBL_MAX_10_EXP   +308
#define CL_DBL_MAX_EXP      +1024
#define CL_DBL_MIN_10_EXP   -307
#define CL_DBL_MIN_EXP      -1021
#define CL_DBL_RADIX        2
#define CL_DBL_MAX          0x1.fffffffffffffp1023
#define CL_DBL_MIN          0x1.0p-1022
#define CL_DBL_EPSILON      0x1.0p-52

#define  CL_M_E             2.718281828459045090796
#define  CL_M_LOG2E         1.442695040888963387005
#define  CL_M_LOG10E        0.434294481903251816668
#define  CL_M_LN2           0.693147180559945286227
#define  CL_M_LN10          2.302585092994045901094
#define  CL_M_PI            3.141592653589793115998
#define  CL_M_PI_2          1.570796326794896557999
#define  CL_M_PI_4          0.785398163397448278999
#define  CL_M_1_PI          0.318309886183790691216
#define  CL_M_2_PI          0.636619772367581382433
#define  CL_M_2_SQRTPI      1.128379167095512558561
#define  CL_M_SQRT2         1.414213562373095145475
#define  CL_M_SQRT1_2       0.707106781186547572737

#define  CL_M_E_F           2.71828174591064f
#define  CL_M_LOG2E_F       1.44269502162933f
#define  CL_M_LOG10E_F      0.43429449200630f
#define  CL_M_LN2_F         0.69314718246460f
#define  CL_M_LN10_F        2.30258512496948f
#define  CL_M_PI_F          3.14159274101257f
#define  CL_M_PI_2_F        1.57079637050629f
#define  CL_M_PI_4_F        0.78539818525314f
#define  CL_M_1_PI_F        0.31830987334251f
#define  CL_M_2_PI_F        0.63661974668503f
#define  CL_M_2_SQRTPI_F    1.12837922573090f
#define  CL_M_SQRT2_F       1.41421353816986f
#define  CL_M_SQRT1_2_F     0.70710676908493f

#if defined( __GNUC__ )
   #define CL_HUGE_VALF     __builtin_huge_valf()
   #define CL_HUGE_VAL      __builtin_huge_val()
   #define CL_NAN           __builtin_nanf( "" )
#else
   #define CL_HUGE_VALF     ((cl_float) 1e50)
   #define CL_HUGE_VAL      ((cl_double) 1e500)
   float nanf( const char * );
   #define CL_NAN           nanf( "" )  
#endif
#define CL_MAXFLOAT         CL_FLT_MAX
#define CL_INFINITY         CL_HUGE_VALF

#endif

#include <stddef.h>

/* Mirror types to GL types. Mirror types allow us to avoid deciding which headers to load based on whether we are using GL or GLES here. */
typedef unsigned int cl_GLuint;
typedef int          cl_GLint;
typedef unsigned int cl_GLenum;

/*
 * Vector types 
 *
 *  Note:   OpenCL requires that all types be naturally aligned. 
 *          This means that vector types must be naturally aligned.
 *          For example, a vector of four floats must be aligned to
 *          a 16 byte boundary (calculated as 4 * the natural 4-byte 
 *          alignment of the float).  The alignment qualifiers here
 *          will only function properly if your compiler supports them
 *          and if you don't actively work to defeat them.  For example,
 *          in order for a cl_float4 to be 16 byte aligned in a struct,
 *          the start of the struct must itself be 16-byte aligned. 
 *
 *          Maintaining proper alignment is the user's responsibility.
 */


#ifdef _MSC_VER
#if defined(_M_IX86)
#if _M_IX86_FP >= 0
#define __SSE__
#endif
#if _M_IX86_FP >= 1
#define __SSE2__
#endif
#elif defined(_M_X64)
#define __SSE__
#define __SSE2__
#endif
#endif

/* Define basic vector types */
#if defined( __VEC__ )
   #include <altivec.h>   /* may be omitted depending on compiler. AltiVec spec provides no way to detect whether the header is required. */
   typedef vector unsigned char     __cl_uchar16;
   typedef vector signed char       __cl_char16;
   typedef vector unsigned short    __cl_ushort8;
   typedef vector signed short      __cl_short8;
   typedef vector unsigned int      __cl_uint4;
   typedef vector signed int        __cl_int4;
   typedef vector float             __cl_float4;
   #define  __CL_UCHAR16__  1
   #define  __CL_CHAR16__   1
   #define  __CL_USHORT8__  1
   #define  __CL_SHORT8__   1
   #define  __CL_UINT4__    1
   #define  __CL_INT4__     1
   #define  __CL_FLOAT4__   1
#endif

#if defined( __SSE__ )
    #if defined( __MINGW64__ )
        #include <intrin.h>
    #else
        #include <xmmintrin.h>
    #endif
    #if defined( __GNUC__ ) && !defined( __ICC )
        typedef float __cl_float4   __attribute__((vector_size(16)));
    #else
        typedef __m128 __cl_float4;
    #endif
    #define __CL_FLOAT4__   1
#endif

#if defined( __SSE2__ )
    #if defined( __MINGW64__ )
        #include <intrin.h>
    #else
        #include <emmintrin.h>
    #endif
    #if defined( __GNUC__ ) && !defined( __ICC )
        typedef cl_uchar    __cl_uchar16    __attribute__((vector_size(16)));
        typedef cl_char     __cl_char16     __attribute__((vector_size(16)));
        typedef cl_ushort   __cl_ushort8    __attribute__((vector_size(16)));
        typedef cl_short    __cl_short8     __attribute__((vector_size(16)));
        typedef cl_uint     __cl_uint4      __attribute__((vector_size(16)));
        typedef cl_int      __cl_int4       __attribute__((vector_size(16)));
        typedef cl_ulong    __cl_ulong2     __attribute__((vector_size(16)));
        typedef cl_long     __cl_long2      __attribute__((vector_size(16)));
        typedef cl_double   __cl_double2    __attribute__((vector_size(16)));
    #else
        typedef __m128i __cl_uchar16;
        typedef __m128i __cl_char16;
        typedef __m128i __cl_ushort8;
        typedef __m128i __cl_short8;
        typedef __m128i __cl_uint4;
        typedef __m128i __cl_int4;
        typedef __m128i __cl_ulong2;
        typedef __m128i __cl_long2;
        typedef __m128d __cl_double2;
    #endif
    #define __CL_UCHAR16__  1
    #define __CL_CHAR16__   1
    #define __CL_USHORT8__  1
    #define __CL_SHORT8__   1
    #define __CL_INT4__     1
    #define __CL_UINT4__    1
    #define __CL_ULONG2__   1
    #define __CL_LONG2__    1
    #define __CL_DOUBLE2__  1
#endif

#if defined( __MMX__ )
    #include <mmintrin.h>
    #if defined( __GNUC__ ) && !defined( __ICC )
        typedef cl_uchar    __cl_uchar8     __attribute__((vector_size(8)));
        typedef cl_char     __cl_char8      __attribute__((vector_size(8)));
        typedef cl_ushort   __cl_ushort4    __attribute__((vector_size(8)));
        typedef cl_short    __cl_short4     __attribute__((vector_size(8)));
        typedef cl_uint     __cl_uint2      __attribute__((vector_size(8)));
        typedef cl_int      __cl_int2       __attribute__((vector_size(8)));
        typedef cl_ulong    __cl_ulong1     __attribute__((vector_size(8)));
        typedef cl_long     __cl_long1      __attribute__((vector_size(8)));
        typedef cl_float    __cl_float2     __attribute__((vector_size(8)));
    #else
        typedef __m64       __cl_uchar8;
        typedef __m64       __cl_char8;
        typedef __m64       __cl_ushort4;
        typedef __m64       __cl_short4;
        typedef __m64       __cl_uint2;
        typedef __m64       __cl_int2;
        typedef __m64       __cl_ulong1;
        typedef __m64       __cl_long1;
        typedef __m64       __cl_float2;
    #endif
    #define __CL_UCHAR8__   1
    #define __CL_CHAR8__    1
    #define __CL_USHORT4__  1
    #define __CL_SHORT4__   1
    #define __CL_INT2__     1
    #define __CL_UINT2__    1
    #define __CL_ULONG1__   1
    #define __CL_LONG1__    1
    #define __CL_FLOAT2__   1
#endif

#if defined( __AVX__ )
    #if defined( __MINGW64__ )
        #include <intrin.h>
    #else
        #include <immintrin.h> 
    #endif
    #if defined( __GNUC__ ) && !defined( __ICC )
        typedef cl_float    __cl_float8     __attribute__((vector_size(32)));
        typedef cl_double   __cl_double4    __attribute__((vector_size(32)));
    #else
        typedef __m256      __cl_float8;
        typedef __m256d     __cl_double4;
    #endif
    #define __CL_FLOAT8__   1
    #define __CL_DOUBLE4__  1
#endif

/* Define alignment keys */
#if defined( __GNUC__ )
    #define CL_ALIGNED(_x)          __attribute__ ((aligned(_x)))
#elif defined( _WIN32) && (_MSC_VER)
    /* Alignment keys neutered on windows because MSVC can't swallow function arguments with alignment requirements     */
    /* http://msdn.microsoft.com/en-us/library/373ak2y1%28VS.71%29.aspx                                                 */
    /* #include <crtdefs.h>                                                                                             */
    /* #define CL_ALIGNED(_x)          _CRT_ALIGN(_x)                                                                   */
    #define CL_ALIGNED(_x)
#else
   #warning  Need to implement some method to align data here
   #define  CL_ALIGNED(_x)
#endif

/* Indicate whether .xyzw, .s0123 and .hi.lo are supported */
#if (defined( __GNUC__) && ! defined( __STRICT_ANSI__ )) || (defined( _MSC_VER ) && ! defined( __STDC__ ))
    /* .xyzw and .s0123...{f|F} are supported */
    #define CL_HAS_NAMED_VECTOR_FIELDS 1
    /* .hi and .lo are supported */
    #define CL_HAS_HI_LO_VECTOR_FIELDS 1

    #define CL_NAMED_STRUCT_SUPPORTED
#endif

#if defined( CL_NAMED_STRUCT_SUPPORTED) && defined( _MSC_VER )
#define __extension__ __pragma(warning(suppress:4201))
#endif

/* Define cl_vector types */

/* ---- cl_charn ---- */
typedef union
{
    cl_char  CL_ALIGNED(2) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_char  x, y; };
   __extension__ struct{ cl_char  s0, s1; };
   __extension__ struct{ cl_char  lo, hi; };
#endif
#if defined( __CL_CHAR2__) 
    __cl_char2     v2;
#endif
}cl_char2;

typedef union
{
    cl_char  CL_ALIGNED(4) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_char  x, y, z, w; };
   __extension__ struct{ cl_char  s0, s1, s2, s3; };
   __extension__ struct{ cl_char2 lo, hi; };
#endif
#if defined( __CL_CHAR2__) 
    __cl_char2     v2[2];
#endif
#if defined( __CL_CHAR4__) 
    __cl_char4     v4;
#endif
}cl_char4;

/* cl_char3 is identical in size, alignment and behavior to cl_char4. See section 6.1.5. */
typedef  cl_char4  cl_char3;

typedef union
{
    cl_char   CL_ALIGNED(8) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_char  x, y, z, w; };
   __extension__ struct{ cl_char  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_char4 lo, hi; };
#endif
#if defined( __CL_CHAR2__) 
    __cl_char2     v2[4];
#endif
#if defined( __CL_CHAR4__) 
    __cl_char4     v4[2];
#endif
#if defined( __CL_CHAR8__ )
    __cl_char8     v8;
#endif
}cl_char8;

typedef union
{
    cl_char  CL_ALIGNED(16) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_char  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_char  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_char8 lo, hi; };
#endif
#if defined( __CL_CHAR2__) 
    __cl_char2     v2[8];
#endif
#if defined( __CL_CHAR4__) 
    __cl_char4     v4[4];
#endif
#if defined( __CL_CHAR8__ )
    __cl_char8     v8[2];
#endif
#if defined( __CL_CHAR16__ )
    __cl_char16    v16;
#endif
}cl_char16;


/* ---- cl_ucharn ---- */
typedef union
{
    cl_uchar  CL_ALIGNED(2) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uchar  x, y; };
   __extension__ struct{ cl_uchar  s0, s1; };
   __extension__ struct{ cl_uchar  lo, hi; };
#endif
#if defined( __cl_uchar2__) 
    __cl_uchar2     v2;
#endif
}cl_uchar2;

typedef union
{
    cl_uchar  CL_ALIGNED(4) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uchar  x, y, z, w; };
   __extension__ struct{ cl_uchar  s0, s1, s2, s3; };
   __extension__ struct{ cl_uchar2 lo, hi; };
#endif
#if defined( __CL_UCHAR2__) 
    __cl_uchar2     v2[2];
#endif
#if defined( __CL_UCHAR4__) 
    __cl_uchar4     v4;
#endif
}cl_uchar4;

/* cl_uchar3 is identical in size, alignment and behavior to cl_uchar4. See section 6.1.5. */
typedef  cl_uchar4  cl_uchar3;

typedef union
{
    cl_uchar   CL_ALIGNED(8) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uchar  x, y, z, w; };
   __extension__ struct{ cl_uchar  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_uchar4 lo, hi; };
#endif
#if defined( __CL_UCHAR2__) 
    __cl_uchar2     v2[4];
#endif
#if defined( __CL_UCHAR4__) 
    __cl_uchar4     v4[2];
#endif
#if defined( __CL_UCHAR8__ )
    __cl_uchar8     v8;
#endif
}cl_uchar8;

typedef union
{
    cl_uchar  CL_ALIGNED(16) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uchar  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_uchar  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_uchar8 lo, hi; };
#endif
#if defined( __CL_UCHAR2__) 
    __cl_uchar2     v2[8];
#endif
#if defined( __CL_UCHAR4__) 
    __cl_uchar4     v4[4];
#endif
#if defined( __CL_UCHAR8__ )
    __cl_uchar8     v8[2];
#endif
#if defined( __CL_UCHAR16__ )
    __cl_uchar16    v16;
#endif
}cl_uchar16;


/* ---- cl_shortn ---- */
typedef union
{
    cl_short  CL_ALIGNED(4) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_short  x, y; };
   __extension__ struct{ cl_short  s0, s1; };
   __extension__ struct{ cl_short  lo, hi; };
#endif
#if defined( __CL_SHORT2__) 
    __cl_short2     v2;
#endif
}cl_short2;

typedef union
{
    cl_short  CL_ALIGNED(8) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_short  x, y, z, w; };
   __extension__ struct{ cl_short  s0, s1, s2, s3; };
   __extension__ struct{ cl_short2 lo, hi; };
#endif
#if defined( __CL_SHORT2__) 
    __cl_short2     v2[2];
#endif
#if defined( __CL_SHORT4__) 
    __cl_short4     v4;
#endif
}cl_short4;

/* cl_short3 is identical in size, alignment and behavior to cl_short4. See section 6.1.5. */
typedef  cl_short4  cl_short3;

typedef union
{
    cl_short   CL_ALIGNED(16) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_short  x, y, z, w; };
   __extension__ struct{ cl_short  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_short4 lo, hi; };
#endif
#if defined( __CL_SHORT2__) 
    __cl_short2     v2[4];
#endif
#if defined( __CL_SHORT4__) 
    __cl_short4     v4[2];
#endif
#if defined( __CL_SHORT8__ )
    __cl_short8     v8;
#endif
}cl_short8;

typedef union
{
    cl_short  CL_ALIGNED(32) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_short  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_short  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_short8 lo, hi; };
#endif
#if defined( __CL_SHORT2__) 
    __cl_short2     v2[8];
#endif
#if defined( __CL_SHORT4__) 
    __cl_short4     v4[4];
#endif
#if defined( __CL_SHORT8__ )
    __cl_short8     v8[2];
#endif
#if defined( __CL_SHORT16__ )
    __cl_short16    v16;
#endif
}cl_short16;


/* ---- cl_ushortn ---- */
typedef union
{
    cl_ushort  CL_ALIGNED(4) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ushort  x, y; };
   __extension__ struct{ cl_ushort  s0, s1; };
   __extension__ struct{ cl_ushort  lo, hi; };
#endif
#if defined( __CL_USHORT2__) 
    __cl_ushort2     v2;
#endif
}cl_ushort2;

typedef union
{
    cl_ushort  CL_ALIGNED(8) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ushort  x, y, z, w; };
   __extension__ struct{ cl_ushort  s0, s1, s2, s3; };
   __extension__ struct{ cl_ushort2 lo, hi; };
#endif
#if defined( __CL_USHORT2__) 
    __cl_ushort2     v2[2];
#endif
#if defined( __CL_USHORT4__) 
    __cl_ushort4     v4;
#endif
}cl_ushort4;

/* cl_ushort3 is identical in size, alignment and behavior to cl_ushort4. See section 6.1.5. */
typedef  cl_ushort4  cl_ushort3;

typedef union
{
    cl_ushort   CL_ALIGNED(16) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ushort  x, y, z, w; };
   __extension__ struct{ cl_ushort  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_ushort4 lo, hi; };
#endif
#if defined( __CL_USHORT2__) 
    __cl_ushort2     v2[4];
#endif
#if defined( __CL_USHORT4__) 
    __cl_ushort4     v4[2];
#endif
#if defined( __CL_USHORT8__ )
    __cl_ushort8     v8;
#endif
}cl_ushort8;

typedef union
{
    cl_ushort  CL_ALIGNED(32) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ushort  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_ushort  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_ushort8 lo, hi; };
#endif
#if defined( __CL_USHORT2__) 
    __cl_ushort2     v2[8];
#endif
#if defined( __CL_USHORT4__) 
    __cl_ushort4     v4[4];
#endif
#if defined( __CL_USHORT8__ )
    __cl_ushort8     v8[2];
#endif
#if defined( __CL_USHORT16__ )
    __cl_ushort16    v16;
#endif
}cl_ushort16;

/* ---- cl_intn ---- */
typedef union
{
    cl_int  CL_ALIGNED(8) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_int  x, y; };
   __extension__ struct{ cl_int  s0, s1; };
   __extension__ struct{ cl_int  lo, hi; };
#endif
#if defined( __CL_INT2__) 
    __cl_int2     v2;
#endif
}cl_int2;

typedef union
{
    cl_int  CL_ALIGNED(16) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_int  x, y, z, w; };
   __extension__ struct{ cl_int  s0, s1, s2, s3; };
   __extension__ struct{ cl_int2 lo, hi; };
#endif
#if defined( __CL_INT2__) 
    __cl_int2     v2[2];
#endif
#if defined( __CL_INT4__) 
    __cl_int4     v4;
#endif
}cl_int4;

/* cl_int3 is identical in size, alignment and behavior to cl_int4. See section 6.1.5. */
typedef  cl_int4  cl_int3;

typedef union
{
    cl_int   CL_ALIGNED(32) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_int  x, y, z, w; };
   __extension__ struct{ cl_int  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_int4 lo, hi; };
#endif
#if defined( __CL_INT2__) 
    __cl_int2     v2[4];
#endif
#if defined( __CL_INT4__) 
    __cl_int4     v4[2];
#endif
#if defined( __CL_INT8__ )
    __cl_int8     v8;
#endif
}cl_int8;

typedef union
{
    cl_int  CL_ALIGNED(64) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_int  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_int  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_int8 lo, hi; };
#endif
#if defined( __CL_INT2__) 
    __cl_int2     v2[8];
#endif
#if defined( __CL_INT4__) 
    __cl_int4     v4[4];
#endif
#if defined( __CL_INT8__ )
    __cl_int8     v8[2];
#endif
#if defined( __CL_INT16__ )
    __cl_int16    v16;
#endif
}cl_int16;


/* ---- cl_uintn ---- */
typedef union
{
    cl_uint  CL_ALIGNED(8) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uint  x, y; };
   __extension__ struct{ cl_uint  s0, s1; };
   __extension__ struct{ cl_uint  lo, hi; };
#endif
#if defined( __CL_UINT2__) 
    __cl_uint2     v2;
#endif
}cl_uint2;

typedef union
{
    cl_uint  CL_ALIGNED(16) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uint  x, y, z, w; };
   __extension__ struct{ cl_uint  s0, s1, s2, s3; };
   __extension__ struct{ cl_uint2 lo, hi; };
#endif
#if defined( __CL_UINT2__) 
    __cl_uint2     v2[2];
#endif
#if defined( __CL_UINT4__) 
    __cl_uint4     v4;
#endif
}cl_uint4;

/* cl_uint3 is identical in size, alignment and behavior to cl_uint4. See section 6.1.5. */
typedef  cl_uint4  cl_uint3;

typedef union
{
    cl_uint   CL_ALIGNED(32) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uint  x, y, z, w; };
   __extension__ struct{ cl_uint  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_uint4 lo, hi; };
#endif
#if defined( __CL_UINT2__) 
    __cl_uint2     v2[4];
#endif
#if defined( __CL_UINT4__) 
    __cl_uint4     v4[2];
#endif
#if defined( __CL_UINT8__ )
    __cl_uint8     v8;
#endif
}cl_uint8;

typedef union
{
    cl_uint  CL_ALIGNED(64) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_uint  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_uint  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_uint8 lo, hi; };
#endif
#if defined( __CL_UINT2__) 
    __cl_uint2     v2[8];
#endif
#if defined( __CL_UINT4__) 
    __cl_uint4     v4[4];
#endif
#if defined( __CL_UINT8__ )
    __cl_uint8     v8[2];
#endif
#if defined( __CL_UINT16__ )
    __cl_uint16    v16;
#endif
}cl_uint16;

/* ---- cl_longn ---- */
typedef union
{
    cl_long  CL_ALIGNED(16) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_long  x, y; };
   __extension__ struct{ cl_long  s0, s1; };
   __extension__ struct{ cl_long  lo, hi; };
#endif
#if defined( __CL_LONG2__) 
    __cl_long2     v2;
#endif
}cl_long2;

typedef union
{
    cl_long  CL_ALIGNED(32) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_long  x, y, z, w; };
   __extension__ struct{ cl_long  s0, s1, s2, s3; };
   __extension__ struct{ cl_long2 lo, hi; };
#endif
#if defined( __CL_LONG2__) 
    __cl_long2     v2[2];
#endif
#if defined( __CL_LONG4__) 
    __cl_long4     v4;
#endif
}cl_long4;

/* cl_long3 is identical in size, alignment and behavior to cl_long4. See section 6.1.5. */
typedef  cl_long4  cl_long3;

typedef union
{
    cl_long   CL_ALIGNED(64) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_long  x, y, z, w; };
   __extension__ struct{ cl_long  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_long4 lo, hi; };
#endif
#if defined( __CL_LONG2__) 
    __cl_long2     v2[4];
#endif
#if defined( __CL_LONG4__) 
    __cl_long4     v4[2];
#endif
#if defined( __CL_LONG8__ )
    __cl_long8     v8;
#endif
}cl_long8;

typedef union
{
    cl_long  CL_ALIGNED(128) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_long  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_long  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_long8 lo, hi; };
#endif
#if defined( __CL_LONG2__) 
    __cl_long2     v2[8];
#endif
#if defined( __CL_LONG4__) 
    __cl_long4     v4[4];
#endif
#if defined( __CL_LONG8__ )
    __cl_long8     v8[2];
#endif
#if defined( __CL_LONG16__ )
    __cl_long16    v16;
#endif
}cl_long16;


/* ---- cl_ulongn ---- */
typedef union
{
    cl_ulong  CL_ALIGNED(16) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ulong  x, y; };
   __extension__ struct{ cl_ulong  s0, s1; };
   __extension__ struct{ cl_ulong  lo, hi; };
#endif
#if defined( __CL_ULONG2__) 
    __cl_ulong2     v2;
#endif
}cl_ulong2;

typedef union
{
    cl_ulong  CL_ALIGNED(32) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ulong  x, y, z, w; };
   __extension__ struct{ cl_ulong  s0, s1, s2, s3; };
   __extension__ struct{ cl_ulong2 lo, hi; };
#endif
#if defined( __CL_ULONG2__) 
    __cl_ulong2     v2[2];
#endif
#if defined( __CL_ULONG4__) 
    __cl_ulong4     v4;
#endif
}cl_ulong4;

/* cl_ulong3 is identical in size, alignment and behavior to cl_ulong4. See section 6.1.5. */
typedef  cl_ulong4  cl_ulong3;

typedef union
{
    cl_ulong   CL_ALIGNED(64) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ulong  x, y, z, w; };
   __extension__ struct{ cl_ulong  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_ulong4 lo, hi; };
#endif
#if defined( __CL_ULONG2__) 
    __cl_ulong2     v2[4];
#endif
#if defined( __CL_ULONG4__) 
    __cl_ulong4     v4[2];
#endif
#if defined( __CL_ULONG8__ )
    __cl_ulong8     v8;
#endif
}cl_ulong8;

typedef union
{
    cl_ulong  CL_ALIGNED(128) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_ulong  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_ulong  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_ulong8 lo, hi; };
#endif
#if defined( __CL_ULONG2__) 
    __cl_ulong2     v2[8];
#endif
#if defined( __CL_ULONG4__) 
    __cl_ulong4     v4[4];
#endif
#if defined( __CL_ULONG8__ )
    __cl_ulong8     v8[2];
#endif
#if defined( __CL_ULONG16__ )
    __cl_ulong16    v16;
#endif
}cl_ulong16;


/* --- cl_floatn ---- */

typedef union
{
    cl_float  CL_ALIGNED(8) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_float  x, y; };
   __extension__ struct{ cl_float  s0, s1; };
   __extension__ struct{ cl_float  lo, hi; };
#endif
#if defined( __CL_FLOAT2__) 
    __cl_float2     v2;
#endif
}cl_float2;

typedef union
{
    cl_float  CL_ALIGNED(16) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_float   x, y, z, w; };
   __extension__ struct{ cl_float   s0, s1, s2, s3; };
   __extension__ struct{ cl_float2  lo, hi; };
#endif
#if defined( __CL_FLOAT2__) 
    __cl_float2     v2[2];
#endif
#if defined( __CL_FLOAT4__) 
    __cl_float4     v4;
#endif
}cl_float4;

/* cl_float3 is identical in size, alignment and behavior to cl_float4. See section 6.1.5. */
typedef  cl_float4  cl_float3;

typedef union
{
    cl_float   CL_ALIGNED(32) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_float   x, y, z, w; };
   __extension__ struct{ cl_float   s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_float4  lo, hi; };
#endif
#if defined( __CL_FLOAT2__) 
    __cl_float2     v2[4];
#endif
#if defined( __CL_FLOAT4__) 
    __cl_float4     v4[2];
#endif
#if defined( __CL_FLOAT8__ )
    __cl_float8     v8;
#endif
}cl_float8;

typedef union
{
    cl_float  CL_ALIGNED(64) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_float  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_float  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_float8 lo, hi; };
#endif
#if defined( __CL_FLOAT2__) 
    __cl_float2     v2[8];
#endif
#if defined( __CL_FLOAT4__) 
    __cl_float4     v4[4];
#endif
#if defined( __CL_FLOAT8__ )
    __cl_float8     v8[2];
#endif
#if defined( __CL_FLOAT16__ )
    __cl_float16    v16;
#endif
}cl_float16;

/* --- cl_doublen ---- */

typedef union
{
    cl_double  CL_ALIGNED(16) s[2];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_double  x, y; };
   __extension__ struct{ cl_double s0, s1; };
   __extension__ struct{ cl_double lo, hi; };
#endif
#if defined( __CL_DOUBLE2__) 
    __cl_double2     v2;
#endif
}cl_double2;

typedef union
{
    cl_double  CL_ALIGNED(32) s[4];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_double  x, y, z, w; };
   __extension__ struct{ cl_double  s0, s1, s2, s3; };
   __extension__ struct{ cl_double2 lo, hi; };
#endif
#if defined( __CL_DOUBLE2__) 
    __cl_double2     v2[2];
#endif
#if defined( __CL_DOUBLE4__) 
    __cl_double4     v4;
#endif
}cl_double4;

/* cl_double3 is identical in size, alignment and behavior to cl_double4. See section 6.1.5. */
typedef  cl_double4  cl_double3;

typedef union
{
    cl_double   CL_ALIGNED(64) s[8];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_double  x, y, z, w; };
   __extension__ struct{ cl_double  s0, s1, s2, s3, s4, s5, s6, s7; };
   __extension__ struct{ cl_double4 lo, hi; };
#endif
#if defined( __CL_DOUBLE2__) 
    __cl_double2     v2[4];
#endif
#if defined( __CL_DOUBLE4__) 
    __cl_double4     v4[2];
#endif
#if defined( __CL_DOUBLE8__ )
    __cl_double8     v8;
#endif
}cl_double8;

typedef union
{
    cl_double  CL_ALIGNED(128) s[16];
#if defined( CL_NAMED_STRUCT_SUPPORTED )
   __extension__ struct{ cl_double  x, y, z, w, __spacer4, __spacer5, __spacer6, __spacer7, __spacer8, __spacer9, sa, sb, sc, sd, se, sf; };
   __extension__ struct{ cl_double  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF; };
   __extension__ struct{ cl_double8 lo, hi; };
#endif
#if defined( __CL_DOUBLE2__) 
    __cl_double2     v2[8];
#endif
#if defined( __CL_DOUBLE4__) 
    __cl_double4     v4[4];
#endif
#if defined( __CL_DOUBLE8__ )
    __cl_double8     v8[2];
#endif
#if defined( __CL_DOUBLE16__ )
    __cl_double16    v16;
#endif
}cl_double16;

/* Macro to facilitate debugging 
 * Usage:
 *   Place CL_PROGRAM_STRING_DEBUG_INFO on the line before the first line of your source. 
 *   The first line ends with:   CL_PROGRAM_STRING_BEGIN \"
 *   Each line thereafter of OpenCL C source must end with: \n\
 *   The last line ends in ";
 *
 *   Example:
 *
 *   const char *my_program = CL_PROGRAM_STRING_BEGIN "\
 *   kernel void foo( int a, float * b )             \n\
 *   {                                               \n\
 *      // my comment                                \n\
 *      *b[ get_global_id(0)] = a;                   \n\
 *   }                                               \n\
 *   ";
 *
 * This should correctly set up the line, (column) and file information for your source 
 * string so you can do source level debugging.
 */
#define  __CL_STRINGIFY( _x )               # _x
#define  _CL_STRINGIFY( _x )                __CL_STRINGIFY( _x )
#define  CL_PROGRAM_STRING_DEBUG_INFO       "#line "  _CL_STRINGIFY(__LINE__) " \"" __FILE__ "\" \n\n" 

//  CL.h contents
/******************************************************************************/

typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

typedef cl_uint             cl_bool;                     /* WARNING!  Unlike cl_ types in cl_platform.h, cl_bool is not guaranteed to be the same size as the bool in kernels. */ 
typedef cl_ulong            cl_bitfield;
typedef cl_bitfield         cl_device_type;
typedef cl_uint             cl_platform_info;
typedef cl_uint             cl_device_info;
typedef cl_bitfield         cl_device_fp_config;
typedef cl_uint             cl_device_mem_cache_type;
typedef cl_uint             cl_device_local_mem_type;
typedef cl_bitfield         cl_device_exec_capabilities;
typedef cl_bitfield         cl_command_queue_properties;

typedef intptr_t			cl_context_properties;
typedef cl_uint             cl_context_info;
typedef cl_uint             cl_command_queue_info;
typedef cl_uint             cl_channel_order;
typedef cl_uint             cl_channel_type;
typedef cl_bitfield         cl_mem_flags;
typedef cl_uint             cl_mem_object_type;
typedef cl_uint             cl_mem_info;
typedef cl_uint             cl_image_info;
typedef cl_uint             cl_buffer_create_type;
typedef cl_uint             cl_addressing_mode;
typedef cl_uint             cl_filter_mode;
typedef cl_uint             cl_sampler_info;
typedef cl_bitfield         cl_map_flags;
typedef cl_uint             cl_program_info;
typedef cl_uint             cl_program_build_info;
typedef cl_int              cl_build_status;
typedef cl_uint             cl_kernel_info;
typedef cl_uint             cl_kernel_work_group_info;
typedef cl_uint             cl_event_info;
typedef cl_uint             cl_command_type;
typedef cl_uint             cl_profiling_info;

typedef struct _cl_image_format {
    cl_channel_order        image_channel_order;
    cl_channel_type         image_channel_data_type;
} cl_image_format;


typedef struct _cl_buffer_region {
    size_t                  origin;
    size_t                  size;
} cl_buffer_region;

/******************************************************************************/

/* Error Codes */
#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14

#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64

/* OpenCL Version */
#define CL_VERSION_1_0                              1
#define CL_VERSION_1_1                              1

/* cl_bool */
#define CL_FALSE                                    0
#define CL_TRUE                                     1

/* cl_platform_info */
#define CL_PLATFORM_PROFILE                         0x0900
#define CL_PLATFORM_VERSION                         0x0901
#define CL_PLATFORM_NAME                            0x0902
#define CL_PLATFORM_VENDOR                          0x0903
#define CL_PLATFORM_EXTENSIONS                      0x0904

/* cl_device_type - bitfield */
#define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF

/* cl_device_info */
#define CL_DEVICE_TYPE                              0x1000
#define CL_DEVICE_VENDOR_ID                         0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS                 0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES               0x1005
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR       0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT      0x1007
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT        0x1008
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG       0x1009
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT      0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE     0x100B
#define CL_DEVICE_MAX_CLOCK_FREQUENCY               0x100C
#define CL_DEVICE_ADDRESS_BITS                      0x100D
#define CL_DEVICE_MAX_READ_IMAGE_ARGS               0x100E
#define CL_DEVICE_MAX_WRITE_IMAGE_ARGS              0x100F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE                0x1010
#define CL_DEVICE_IMAGE2D_MAX_WIDTH                 0x1011
#define CL_DEVICE_IMAGE2D_MAX_HEIGHT                0x1012
#define CL_DEVICE_IMAGE3D_MAX_WIDTH                 0x1013
#define CL_DEVICE_IMAGE3D_MAX_HEIGHT                0x1014
#define CL_DEVICE_IMAGE3D_MAX_DEPTH                 0x1015
#define CL_DEVICE_IMAGE_SUPPORT                     0x1016
#define CL_DEVICE_MAX_PARAMETER_SIZE                0x1017
#define CL_DEVICE_MAX_SAMPLERS                      0x1018
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN               0x1019
#define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          0x101A
#define CL_DEVICE_SINGLE_FP_CONFIG                  0x101B
#define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             0x101C
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         0x101D
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             0x101E
#define CL_DEVICE_GLOBAL_MEM_SIZE                   0x101F
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          0x1020
#define CL_DEVICE_MAX_CONSTANT_ARGS                 0x1021
#define CL_DEVICE_LOCAL_MEM_TYPE                    0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE                    0x1023
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT          0x1024
#define CL_DEVICE_PROFILING_TIMER_RESOLUTION        0x1025
#define CL_DEVICE_ENDIAN_LITTLE                     0x1026
#define CL_DEVICE_AVAILABLE                         0x1027
#define CL_DEVICE_COMPILER_AVAILABLE                0x1028
#define CL_DEVICE_EXECUTION_CAPABILITIES            0x1029
#define CL_DEVICE_QUEUE_PROPERTIES                  0x102A
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_VENDOR                            0x102C
#define CL_DRIVER_VERSION                           0x102D
#define CL_DEVICE_PROFILE                           0x102E
#define CL_DEVICE_VERSION                           0x102F
#define CL_DEVICE_EXTENSIONS                        0x1030
#define CL_DEVICE_PLATFORM                          0x1031
/* 0x1032 reserved for CL_DEVICE_DOUBLE_FP_CONFIG */
/* 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG */
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       0x1034
#define CL_DEVICE_HOST_UNIFIED_MEMORY               0x1035
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          0x1036
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         0x1037
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           0x1038
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          0x1039
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         0x103A
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        0x103B
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          0x103C
#define CL_DEVICE_OPENCL_C_VERSION                  0x103D

/* cl_device_fp_config - bitfield */
#define CL_FP_DENORM                                (1 << 0)
#define CL_FP_INF_NAN                               (1 << 1)
#define CL_FP_ROUND_TO_NEAREST                      (1 << 2)
#define CL_FP_ROUND_TO_ZERO                         (1 << 3)
#define CL_FP_ROUND_TO_INF                          (1 << 4)
#define CL_FP_FMA                                   (1 << 5)
#define CL_FP_SOFT_FLOAT                            (1 << 6)

/* cl_device_mem_cache_type */
#define CL_NONE                                     0x0
#define CL_READ_ONLY_CACHE                          0x1
#define CL_READ_WRITE_CACHE                         0x2

/* cl_device_local_mem_type */
#define CL_LOCAL                                    0x1
#define CL_GLOBAL                                   0x2

/* cl_device_exec_capabilities - bitfield */
#define CL_EXEC_KERNEL                              (1 << 0)
#define CL_EXEC_NATIVE_KERNEL                       (1 << 1)

/* cl_command_queue_properties - bitfield */
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)

/* cl_context_info  */
#define CL_CONTEXT_REFERENCE_COUNT                  0x1080
#define CL_CONTEXT_DEVICES                          0x1081
#define CL_CONTEXT_PROPERTIES                       0x1082
#define CL_CONTEXT_NUM_DEVICES                      0x1083

/* cl_context_info + cl_context_properties */
#define CL_CONTEXT_PLATFORM                         0x1084

/* cl_command_queue_info */
#define CL_QUEUE_CONTEXT                            0x1090
#define CL_QUEUE_DEVICE                             0x1091
#define CL_QUEUE_REFERENCE_COUNT                    0x1092
#define CL_QUEUE_PROPERTIES                         0x1093

/* cl_mem_flags - bitfield */
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_USE_HOST_PTR                         (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)

/* cl_channel_order */
#define CL_R                                        0x10B0
#define CL_A                                        0x10B1
#define CL_RG                                       0x10B2
#define CL_RA                                       0x10B3
#define CL_RGB                                      0x10B4
#define CL_RGBA                                     0x10B5
#define CL_BGRA                                     0x10B6
#define CL_ARGB                                     0x10B7
#define CL_INTENSITY                                0x10B8
#define CL_LUMINANCE                                0x10B9
#define CL_Rx                                       0x10BA
#define CL_RGx                                      0x10BB
#define CL_RGBx                                     0x10BC

/* cl_channel_type */
#define CL_SNORM_INT8                               0x10D0
#define CL_SNORM_INT16                              0x10D1
#define CL_UNORM_INT8                               0x10D2
#define CL_UNORM_INT16                              0x10D3
#define CL_UNORM_SHORT_565                          0x10D4
#define CL_UNORM_SHORT_555                          0x10D5
#define CL_UNORM_INT_101010                         0x10D6
#define CL_SIGNED_INT8                              0x10D7
#define CL_SIGNED_INT16                             0x10D8
#define CL_SIGNED_INT32                             0x10D9
#define CL_UNSIGNED_INT8                            0x10DA
#define CL_UNSIGNED_INT16                           0x10DB
#define CL_UNSIGNED_INT32                           0x10DC
#define CL_HALF_FLOAT                               0x10DD
#define CL_FLOAT                                    0x10DE

/* cl_mem_object_type */
#define CL_MEM_OBJECT_BUFFER                        0x10F0
#define CL_MEM_OBJECT_IMAGE2D                       0x10F1
#define CL_MEM_OBJECT_IMAGE3D                       0x10F2

/* cl_mem_info */
#define CL_MEM_TYPE                                 0x1100
#define CL_MEM_FLAGS                                0x1101
#define CL_MEM_SIZE                                 0x1102
#define CL_MEM_HOST_PTR                             0x1103
#define CL_MEM_MAP_COUNT                            0x1104
#define CL_MEM_REFERENCE_COUNT                      0x1105
#define CL_MEM_CONTEXT                              0x1106
#define CL_MEM_ASSOCIATED_MEMOBJECT                 0x1107
#define CL_MEM_OFFSET                               0x1108

/* cl_image_info */
#define CL_IMAGE_FORMAT                             0x1110
#define CL_IMAGE_ELEMENT_SIZE                       0x1111
#define CL_IMAGE_ROW_PITCH                          0x1112
#define CL_IMAGE_SLICE_PITCH                        0x1113
#define CL_IMAGE_WIDTH                              0x1114
#define CL_IMAGE_HEIGHT                             0x1115
#define CL_IMAGE_DEPTH                              0x1116

/* cl_addressing_mode */
#define CL_ADDRESS_NONE                             0x1130
#define CL_ADDRESS_CLAMP_TO_EDGE                    0x1131
#define CL_ADDRESS_CLAMP                            0x1132
#define CL_ADDRESS_REPEAT                           0x1133
#define CL_ADDRESS_MIRRORED_REPEAT                  0x1134

/* cl_filter_mode */
#define CL_FILTER_NEAREST                           0x1140
#define CL_FILTER_LINEAR                            0x1141

/* cl_sampler_info */
#define CL_SAMPLER_REFERENCE_COUNT                  0x1150
#define CL_SAMPLER_CONTEXT                          0x1151
#define CL_SAMPLER_NORMALIZED_COORDS                0x1152
#define CL_SAMPLER_ADDRESSING_MODE                  0x1153
#define CL_SAMPLER_FILTER_MODE                      0x1154

/* cl_map_flags - bitfield */
#define CL_MAP_READ                                 (1 << 0)
#define CL_MAP_WRITE                                (1 << 1)

/* cl_program_info */
#define CL_PROGRAM_REFERENCE_COUNT                  0x1160
#define CL_PROGRAM_CONTEXT                          0x1161
#define CL_PROGRAM_NUM_DEVICES                      0x1162
#define CL_PROGRAM_DEVICES                          0x1163
#define CL_PROGRAM_SOURCE                           0x1164
#define CL_PROGRAM_BINARY_SIZES                     0x1165
#define CL_PROGRAM_BINARIES                         0x1166

/* cl_program_build_info */
#define CL_PROGRAM_BUILD_STATUS                     0x1181
#define CL_PROGRAM_BUILD_OPTIONS                    0x1182
#define CL_PROGRAM_BUILD_LOG                        0x1183

/* cl_build_status */
#define CL_BUILD_SUCCESS                            0
#define CL_BUILD_NONE                               -1
#define CL_BUILD_ERROR                              -2
#define CL_BUILD_IN_PROGRESS                        -3

/* cl_kernel_info */
#define CL_KERNEL_FUNCTION_NAME                     0x1190
#define CL_KERNEL_NUM_ARGS                          0x1191
#define CL_KERNEL_REFERENCE_COUNT                   0x1192
#define CL_KERNEL_CONTEXT                           0x1193
#define CL_KERNEL_PROGRAM                           0x1194

/* cl_kernel_work_group_info */
#define CL_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define CL_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define CL_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define CL_KERNEL_PRIVATE_MEM_SIZE                  0x11B4

/* cl_event_info  */
#define CL_EVENT_COMMAND_QUEUE                      0x11D0
#define CL_EVENT_COMMAND_TYPE                       0x11D1
#define CL_EVENT_REFERENCE_COUNT                    0x11D2
#define CL_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define CL_EVENT_CONTEXT                            0x11D4

/* cl_command_type */
#define CL_COMMAND_NDRANGE_KERNEL                   0x11F0
#define CL_COMMAND_TASK                             0x11F1
#define CL_COMMAND_NATIVE_KERNEL                    0x11F2
#define CL_COMMAND_READ_BUFFER                      0x11F3
#define CL_COMMAND_WRITE_BUFFER                     0x11F4
#define CL_COMMAND_COPY_BUFFER                      0x11F5
#define CL_COMMAND_READ_IMAGE                       0x11F6
#define CL_COMMAND_WRITE_IMAGE                      0x11F7
#define CL_COMMAND_COPY_IMAGE                       0x11F8
#define CL_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define CL_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define CL_COMMAND_MAP_BUFFER                       0x11FB
#define CL_COMMAND_MAP_IMAGE                        0x11FC
#define CL_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define CL_COMMAND_MARKER                           0x11FE
#define CL_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define CL_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define CL_COMMAND_READ_BUFFER_RECT                 0x1201
#define CL_COMMAND_WRITE_BUFFER_RECT                0x1202
#define CL_COMMAND_COPY_BUFFER_RECT                 0x1203
#define CL_COMMAND_USER                             0x1204

/* command execution status */
#define CL_COMPLETE                                 0x0
#define CL_RUNNING                                  0x1
#define CL_SUBMITTED                                0x2
#define CL_QUEUED                                   0x3
  
/* cl_buffer_create_type  */
#define CL_BUFFER_CREATE_TYPE_REGION                0x1220

/* cl_profiling_info  */
#define CL_PROFILING_COMMAND_QUEUED                 0x1280
#define CL_PROFILING_COMMAND_SUBMIT                 0x1281
#define CL_PROFILING_COMMAND_START                  0x1282
#define CL_PROFILING_COMMAND_END                    0x1283

/********************************************************************************************************/

/********************************************************************************************************/

/*  Function signature typedef's */

/* Platform API */
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETPLATFORMIDS)(cl_uint          /* num_entries */,
                 cl_platform_id * /* platforms */,
                 cl_uint *        /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL * 
PFNCLGETPLATFORMINFO)(cl_platform_id   /* platform */, 
                  cl_platform_info /* param_name */,
                  size_t           /* param_value_size */, 
                  void *           /* param_value */,
                  size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Device APIs */
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETDEVICEIDS)(cl_platform_id   /* platform */,
               cl_device_type   /* device_type */, 
               cl_uint          /* num_entries */, 
               cl_device_id *   /* devices */, 
               cl_uint *        /* num_devices */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETDEVICEINFO)(cl_device_id    /* device */,
                cl_device_info  /* param_name */, 
                size_t          /* param_value_size */, 
                void *          /* param_value */,
                size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

// Context APIs  
typedef CL_API_ENTRY cl_context (CL_API_CALL *
PFNCLCREATECONTEXT)(const cl_context_properties * /* properties */,
                cl_uint                       /* num_devices */,
                const cl_device_id *          /* devices */,
                void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                void *                        /* user_data */,
                cl_int *                      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_context (CL_API_CALL *
PFNCLCREATECONTEXTFROMTYPE)(const cl_context_properties * /* properties */,
                        cl_device_type                /* device_type */,
                        void (CL_CALLBACK *     /* pfn_notify*/ )(const char *, const void *, size_t, void *),
                        void *                        /* user_data */,
                        cl_int *                      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINCONTEXT)(cl_context /* context */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASECONTEXT)(cl_context /* context */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETCONTEXTINFO)(cl_context         /* context */, 
                 cl_context_info    /* param_name */, 
                 size_t             /* param_value_size */, 
                 void *             /* param_value */, 
                 size_t *           /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Command Queue APIs */
typedef CL_API_ENTRY cl_command_queue (CL_API_CALL *
PFNCLCREATECOMMANDQUEUE)(cl_context                     /* context */, 
                     cl_device_id                   /* device */, 
                     cl_command_queue_properties    /* properties */,
                     cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINCOMMANDQUEUE)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASECOMMANDQUEUE)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETCOMMANDQUEUEINFO)(cl_command_queue      /* command_queue */,
                      cl_command_queue_info /* param_name */,
                      size_t                /* param_value_size */,
                      void *                /* param_value */,
                      size_t *              /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLSETCOMMANDQUEUEPROPERTY)(cl_command_queue              /* command_queue */,
                          cl_command_queue_properties   /* properties */, 
                          cl_bool                        /* enable */,
                          cl_command_queue_properties * /* old_properties */) CL_API_SUFFIX__VERSION_1_0;

/* Memory Object APIs */
typedef CL_API_ENTRY cl_mem (CL_API_CALL *
PFNCLCREATEBUFFER)(cl_context   /* context */,
               cl_mem_flags /* flags */,
               size_t       /* size */,
               void *       /* host_ptr */,
               cl_int *     /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *
PFNCLCREATESUBBUFFER)(cl_mem   			/* buffer */,
               cl_mem_flags 			/* flags */,
               cl_buffer_create_type    /* buffer_create_type */,
               const void *       		/* buffer_create_info */,
               cl_int *     			/* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *
PFNCLCREATEIMAGE2D)(cl_context              /* context */,
                cl_mem_flags            /* flags */,
                const cl_image_format * /* image_format */,
                size_t                  /* image_width */,
                size_t                  /* image_height */,
                size_t                  /* image_row_pitch */, 
                void *                  /* host_ptr */,
                cl_int *                /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *
PFNCLCREATEIMAGE3D)(cl_context              /* context */,
                cl_mem_flags            /* flags */,
                const cl_image_format * /* image_format */,
                size_t                  /* image_width */, 
                size_t                  /* image_height */,
                size_t                  /* image_depth */, 
                size_t                  /* image_row_pitch */, 
                size_t                  /* image_slice_pitch */, 
                void *                  /* host_ptr */,
                cl_int *                /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINMEMOBJECT)(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASEMEMOBJECT)(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETSUPPORTEDIMAGEFORMATS)(cl_context           /* context */,
                           cl_mem_flags         /* flags */,
                           cl_mem_object_type   /* image_type */,
                           cl_uint              /* num_entries */,
                           cl_image_format *    /* image_formats */,
                           cl_uint *            /* num_image_formats */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETMEMOBJECTINFO)(cl_mem           /* memobj */,
                   cl_mem_info      /* param_name */, 
                   size_t           /* param_value_size */,
                   void *           /* param_value */,
                   size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETIMAGEINFO)(cl_mem           /* image */,
               cl_image_info    /* param_name */, 
               size_t           /* param_value_size */,
               void *           /* param_value */,
               size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLSETMEMOBJECTDESTRUCTORCALLBACK)(  cl_mem /* memobj */, 
                                    void (CL_CALLBACK * /*pfn_notify*/)( cl_mem /* memobj */, void* /*user_data*/), 
                                    void * /*user_data */ )             CL_API_SUFFIX__VERSION_1_1;  

/* Sampler APIs  */
typedef CL_API_ENTRY cl_sampler (CL_API_CALL *
PFNCLCREATESAMPLER)(cl_context          /* context */,
                cl_bool             /* normalized_coords */, 
                cl_addressing_mode  /* addressing_mode */, 
                cl_filter_mode      /* filter_mode */,
                cl_int *            /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINSAMPLER)(cl_sampler /* sampler */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASESAMPLER)(cl_sampler /* sampler */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETSAMPLERINFO)(cl_sampler         /* sampler */,
                 cl_sampler_info    /* param_name */,
                 size_t             /* param_value_size */,
                 void *             /* param_value */,
                 size_t *           /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
 
/* Program Object APIs  */
typedef CL_API_ENTRY cl_program (CL_API_CALL *
PFNCLCREATEPROGRAMWITHSOURCE)(cl_context        /* context */,
                          cl_uint           /* count */,
                          const char **     /* strings */,
                          const size_t *    /* lengths */,
                          cl_int *          /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_program (CL_API_CALL *
PFNCLCREATEPROGRAMWITHBINARY)(cl_context                     /* context */,
                          cl_uint                        /* num_devices */,
                          const cl_device_id *           /* device_list */,
                          const size_t *                 /* lengths */,
                          const unsigned char **         /* binaries */,
                          cl_int *                       /* binary_status */,
                          cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINPROGRAM)(cl_program /* program */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASEPROGRAM)(cl_program /* program */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLBUILDPROGRAM)(cl_program           /* program */,
               cl_uint              /* num_devices */,
               const cl_device_id * /* device_list */,
               const char *         /* options */, 
               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
               void *               /* user_data */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLUNLOADCOMPILER)(void) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETPROGRAMINFO)(cl_program         /* program */,
                 cl_program_info    /* param_name */,
                 size_t             /* param_value_size */,
                 void *             /* param_value */,
                 size_t *           /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETPROGRAMBUILDINFO)(cl_program            /* program */,
                      cl_device_id          /* device */,
                      cl_program_build_info /* param_name */,
                      size_t                /* param_value_size */,
                      void *                /* param_value */,
                      size_t *              /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Kernel Object APIs */
typedef CL_API_ENTRY cl_kernel (CL_API_CALL *
PFNCLCREATEKERNEL)(cl_program      /* program */,
               const char *    /* kernel_name */,
               cl_int *        /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLCREATEKERNELSINPROGRAM)(cl_program     /* program */,
                         cl_uint        /* num_kernels */,
                         cl_kernel *    /* kernels */,
                         cl_uint *      /* num_kernels_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINKERNEL)(cl_kernel    /* kernel */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASEKERNEL)(cl_kernel   /* kernel */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLSETKERNELARG)(cl_kernel    /* kernel */,
               cl_uint      /* arg_index */,
               size_t       /* arg_size */,
               const void * /* arg_value */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETKERNELINFO)(cl_kernel       /* kernel */,
                cl_kernel_info  /* param_name */,
                size_t          /* param_value_size */,
                void *          /* param_value */,
                size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETKERNELWORKGROUPINFO)(cl_kernel                  /* kernel */,
                         cl_device_id               /* device */,
                         cl_kernel_work_group_info  /* param_name */,
                         size_t                     /* param_value_size */,
                         void *                     /* param_value */,
                         size_t *                   /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

// Event Object APIs
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLWAITFOREVENTS)(cl_uint             /* num_events */,
                const cl_event *    /* event_list */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETEVENTINFO)(cl_event         /* event */,
               cl_event_info    /* param_name */,
               size_t           /* param_value_size */,
               void *           /* param_value */,
               size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_event (CL_API_CALL *
PFNCLCREATEUSEREVENT)(cl_context    /* context */,
                  	  cl_int *      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;               

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRETAINEVENT)(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLRELEASEEVENT)(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLSETUSEREVENTSTATUS)(cl_event   /* event */,
                     cl_int     /* execution_status */) CL_API_SUFFIX__VERSION_1_1;
                     
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLSETEVENTCALLBACK)( cl_event    /* event */,
                    cl_int      /* command_exec_callback_type */,
                    void (CL_CALLBACK * /* pfn_notify */)(cl_event, cl_int, void *),
                    void *      /* user_data */) CL_API_SUFFIX__VERSION_1_1;

/* Profiling APIs  */
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLGETEVENTPROFILINGINFO)(cl_event            /* event */,
                        cl_profiling_info   /* param_name */,
                        size_t              /* param_value_size */,
                        void *              /* param_value */,
                        size_t *            /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

// Flush and Finish APIs
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLFLUSH)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLFINISH)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

/* Enqueued Commands APIs */
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEREADBUFFER)(cl_command_queue    /* command_queue */,
                    cl_mem              /* buffer */,
                    cl_bool             /* blocking_read */,
                    size_t              /* offset */,
                    size_t              /* cb */, 
                    void *              /* ptr */,
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEREADBUFFERRECT)(cl_command_queue    /* command_queue */,
                        cl_mem              /* buffer */,
                        cl_bool             /* blocking_read */,
                        const size_t *      /* buffer_origin */,
                        const size_t *      /* host_origin */, 
                        const size_t *      /* region */,
                        size_t              /* buffer_row_pitch */,
                        size_t              /* buffer_slice_pitch */,
                        size_t              /* host_row_pitch */,
                        size_t              /* host_slice_pitch */,                        
                        void *              /* ptr */,
                        cl_uint             /* num_events_in_wait_list */,
                        const cl_event *    /* event_wait_list */,
                        cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEWRITEBUFFER)(cl_command_queue   /* command_queue */, 
                     cl_mem             /* buffer */, 
                     cl_bool            /* blocking_write */, 
                     size_t             /* offset */, 
                     size_t             /* cb */, 
                     const void *       /* ptr */, 
                     cl_uint            /* num_events_in_wait_list */, 
                     const cl_event *   /* event_wait_list */, 
                     cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_0;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEWRITEBUFFERRECT)(cl_command_queue    /* command_queue */,
                         cl_mem              /* buffer */,
                         cl_bool             /* blocking_write */,
                         const size_t *      /* buffer_origin */,
                         const size_t *      /* host_origin */, 
                         const size_t *      /* region */,
                         size_t              /* buffer_row_pitch */,
                         size_t              /* buffer_slice_pitch */,
                         size_t              /* host_row_pitch */,
                         size_t              /* host_slice_pitch */,                        
                         const void *        /* ptr */,
                         cl_uint             /* num_events_in_wait_list */,
                         const cl_event *    /* event_wait_list */,
                         cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUECOPYBUFFER)(cl_command_queue    /* command_queue */, 
                    cl_mem              /* src_buffer */,
                    cl_mem              /* dst_buffer */, 
                    size_t              /* src_offset */,
                    size_t              /* dst_offset */,
                    size_t              /* cb */, 
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUECOPYBUFFERRECT)(cl_command_queue    /* command_queue */, 
                        cl_mem              /* src_buffer */,
                        cl_mem              /* dst_buffer */, 
                        const size_t *      /* src_origin */,
                        const size_t *      /* dst_origin */,
                        const size_t *      /* region */, 
                        size_t              /* src_row_pitch */,
                        size_t              /* src_slice_pitch */,
                        size_t              /* dst_row_pitch */,
                        size_t              /* dst_slice_pitch */,
                        cl_uint             /* num_events_in_wait_list */,
                        const cl_event *    /* event_wait_list */,
                        cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_1;
                            
typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEREADIMAGE)(cl_command_queue     /* command_queue */,
                   cl_mem               /* image */,
                   cl_bool              /* blocking_read */, 
                   const size_t *       /* origin[3] */,
                   const size_t *       /* region[3] */,
                   size_t               /* row_pitch */,
                   size_t               /* slice_pitch */, 
                   void *               /* ptr */,
                   cl_uint              /* num_events_in_wait_list */,
                   const cl_event *     /* event_wait_list */,
                   cl_event *           /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEWRITEIMAGE)(cl_command_queue    /* command_queue */,
                    cl_mem              /* image */,
                    cl_bool             /* blocking_write */, 
                    const size_t *      /* origin[3] */,
                    const size_t *      /* region[3] */,
                    size_t              /* input_row_pitch */,
                    size_t              /* input_slice_pitch */, 
                    const void *        /* ptr */,
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUECOPYIMAGE)(cl_command_queue     /* command_queue */,
                   cl_mem               /* src_image */,
                   cl_mem               /* dst_image */, 
                   const size_t *       /* src_origin[3] */,
                   const size_t *       /* dst_origin[3] */,
                   const size_t *       /* region[3] */, 
                   cl_uint              /* num_events_in_wait_list */,
                   const cl_event *     /* event_wait_list */,
                   cl_event *           /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUECOPYIMAGETOBUFFER)(cl_command_queue /* command_queue */,
                           cl_mem           /* src_image */,
                           cl_mem           /* dst_buffer */, 
                           const size_t *   /* src_origin[3] */,
                           const size_t *   /* region[3] */, 
                           size_t           /* dst_offset */,
                           cl_uint          /* num_events_in_wait_list */,
                           const cl_event * /* event_wait_list */,
                           cl_event *       /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUECOPYBUFFERTOIMAGE)(cl_command_queue /* command_queue */,
                           cl_mem           /* src_buffer */,
                           cl_mem           /* dst_image */, 
                           size_t           /* src_offset */,
                           const size_t *   /* dst_origin[3] */,
                           const size_t *   /* region[3] */, 
                           cl_uint          /* num_events_in_wait_list */,
                           const cl_event * /* event_wait_list */,
                           cl_event *       /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY void * (CL_API_CALL *
PFNCLENQUEUEMAPBUFFER)(cl_command_queue /* command_queue */,
                   cl_mem           /* buffer */,
                   cl_bool          /* blocking_map */, 
                   cl_map_flags     /* map_flags */,
                   size_t           /* offset */,
                   size_t           /* cb */,
                   cl_uint          /* num_events_in_wait_list */,
                   const cl_event * /* event_wait_list */,
                   cl_event *       /* event */,
                   cl_int *         /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY void * (CL_API_CALL *
PFNCLENQUEUEMAPIMAGE)(cl_command_queue  /* command_queue */,
                  cl_mem            /* image */, 
                  cl_bool           /* blocking_map */, 
                  cl_map_flags      /* map_flags */, 
                  const size_t *    /* origin[3] */,
                  const size_t *    /* region[3] */,
                  size_t *          /* image_row_pitch */,
                  size_t *          /* image_slice_pitch */,
                  cl_uint           /* num_events_in_wait_list */,
                  const cl_event *  /* event_wait_list */,
                  cl_event *        /* event */,
                  cl_int *          /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEUNMAPMEMOBJECT)(cl_command_queue /* command_queue */,
                        cl_mem           /* memobj */,
                        void *           /* mapped_ptr */,
                        cl_uint          /* num_events_in_wait_list */,
                        const cl_event *  /* event_wait_list */,
                        cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUENDRANGEKERNEL)(cl_command_queue /* command_queue */,
                       cl_kernel        /* kernel */,
                       cl_uint          /* work_dim */,
                       const size_t *   /* global_work_offset */,
                       const size_t *   /* global_work_size */,
                       const size_t *   /* local_work_size */,
                       cl_uint          /* num_events_in_wait_list */,
                       const cl_event * /* event_wait_list */,
                       cl_event *       /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUETASK)(cl_command_queue  /* command_queue */,
              cl_kernel         /* kernel */,
              cl_uint           /* num_events_in_wait_list */,
              const cl_event *  /* event_wait_list */,
              cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUENATIVEKERNEL)(cl_command_queue  /* command_queue */,
                      void (*user_func)(void *), 
                      void *            /* args */,
                      size_t            /* cb_args */, 
                      cl_uint           /* num_mem_objects */,
                      const cl_mem *    /* mem_list */,
                      const void **     /* args_mem_loc */,
                      cl_uint           /* num_events_in_wait_list */,
                      const cl_event *  /* event_wait_list */,
                      cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEMARKER)(cl_command_queue    /* command_queue */,
                cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEWAITFOREVENTS)(cl_command_queue /* command_queue */,
                       cl_uint          /* num_events */,
                       const cl_event * /* event_list */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *
PFNCLENQUEUEBARRIER)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

// Extension function access
//
// Returns the extension function address for the given function name,
// or NULL if a valid function can not be found.  The client must
// check to make sure the address is not NULL, before using or 
// calling the returned function address.
//
typedef CL_API_ENTRY void * (CL_API_CALL * PFNCLGETEXTENSIONFUNCTIONADDRESS)(const char * /* func_name */) CL_API_SUFFIX__VERSION_1_0;


#define CLEW_STATIC

#ifdef CLEW_STATIC
#  define CLEWAPI extern
#else
#  ifdef CLEW_BUILD
#    define CLEWAPI extern __declspec(dllexport)
#  else
#    define CLEWAPI extern __declspec(dllimport)
#  endif
#endif

#if defined(_WIN32)
#define CLEW_FUN_EXPORT extern
#else
#define CLEW_FUN_EXPORT CLEWAPI
#endif

#define CLEW_GET_FUN(x) x


//  Variables holding function entry points
CLEW_FUN_EXPORT     PFNCLGETPLATFORMIDS                 __clewGetPlatformIDs                ;
CLEW_FUN_EXPORT     PFNCLGETPLATFORMINFO                __clewGetPlatformInfo               ;
CLEW_FUN_EXPORT     PFNCLGETDEVICEIDS                   __clewGetDeviceIDs                  ;
CLEW_FUN_EXPORT     PFNCLGETDEVICEINFO                  __clewGetDeviceInfo                 ;
CLEW_FUN_EXPORT     PFNCLCREATECONTEXT                  __clewCreateContext                 ;
CLEW_FUN_EXPORT     PFNCLCREATECONTEXTFROMTYPE          __clewCreateContextFromType         ;
CLEW_FUN_EXPORT     PFNCLRETAINCONTEXT                  __clewRetainContext                 ;
CLEW_FUN_EXPORT     PFNCLRELEASECONTEXT                 __clewReleaseContext                ;
CLEW_FUN_EXPORT     PFNCLGETCONTEXTINFO                 __clewGetContextInfo                ;
CLEW_FUN_EXPORT     PFNCLCREATECOMMANDQUEUE             __clewCreateCommandQueue            ;
CLEW_FUN_EXPORT     PFNCLRETAINCOMMANDQUEUE             __clewRetainCommandQueue            ;
CLEW_FUN_EXPORT     PFNCLRELEASECOMMANDQUEUE            __clewReleaseCommandQueue           ;
CLEW_FUN_EXPORT     PFNCLGETCOMMANDQUEUEINFO            __clewGetCommandQueueInfo           ;
#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
CLEW_FUN_EXPORT     PFNCLSETCOMMANDQUEUEPROPERTY        __clewSetCommandQueueProperty       ;
#endif
CLEW_FUN_EXPORT     PFNCLCREATEBUFFER                   __clewCreateBuffer                  ;
CLEW_FUN_EXPORT     PFNCLCREATESUBBUFFER                __clewCreateSubBuffer               ;
CLEW_FUN_EXPORT     PFNCLCREATEIMAGE2D                  __clewCreateImage2D                 ;
CLEW_FUN_EXPORT     PFNCLCREATEIMAGE3D                  __clewCreateImage3D                 ;
CLEW_FUN_EXPORT     PFNCLRETAINMEMOBJECT                __clewRetainMemObject               ;
CLEW_FUN_EXPORT     PFNCLRELEASEMEMOBJECT               __clewReleaseMemObject              ;
CLEW_FUN_EXPORT     PFNCLGETSUPPORTEDIMAGEFORMATS       __clewGetSupportedImageFormats      ;
CLEW_FUN_EXPORT     PFNCLGETMEMOBJECTINFO               __clewGetMemObjectInfo              ;
CLEW_FUN_EXPORT     PFNCLGETIMAGEINFO                   __clewGetImageInfo                  ;
CLEW_FUN_EXPORT     PFNCLSETMEMOBJECTDESTRUCTORCALLBACK __clewSetMemObjectDestructorCallback;
CLEW_FUN_EXPORT     PFNCLCREATESAMPLER                  __clewCreateSampler                 ;
CLEW_FUN_EXPORT     PFNCLRETAINSAMPLER                  __clewRetainSampler                 ;
CLEW_FUN_EXPORT     PFNCLRELEASESAMPLER                 __clewReleaseSampler                ;
CLEW_FUN_EXPORT     PFNCLGETSAMPLERINFO                 __clewGetSamplerInfo                ;
CLEW_FUN_EXPORT     PFNCLCREATEPROGRAMWITHSOURCE        __clewCreateProgramWithSource       ;
CLEW_FUN_EXPORT     PFNCLCREATEPROGRAMWITHBINARY        __clewCreateProgramWithBinary       ;
CLEW_FUN_EXPORT     PFNCLRETAINPROGRAM                  __clewRetainProgram                 ;
CLEW_FUN_EXPORT     PFNCLRELEASEPROGRAM                 __clewReleaseProgram                ;
CLEW_FUN_EXPORT     PFNCLBUILDPROGRAM                   __clewBuildProgram                  ;
CLEW_FUN_EXPORT     PFNCLUNLOADCOMPILER                 __clewUnloadCompiler                ;
CLEW_FUN_EXPORT     PFNCLGETPROGRAMINFO                 __clewGetProgramInfo                ;
CLEW_FUN_EXPORT     PFNCLGETPROGRAMBUILDINFO            __clewGetProgramBuildInfo           ;
CLEW_FUN_EXPORT     PFNCLCREATEKERNEL                   __clewCreateKernel                  ;
CLEW_FUN_EXPORT     PFNCLCREATEKERNELSINPROGRAM         __clewCreateKernelsInProgram        ;
CLEW_FUN_EXPORT     PFNCLRETAINKERNEL                   __clewRetainKernel                  ;
CLEW_FUN_EXPORT     PFNCLRELEASEKERNEL                  __clewReleaseKernel                 ;
CLEW_FUN_EXPORT     PFNCLSETKERNELARG                   __clewSetKernelArg                  ;
CLEW_FUN_EXPORT     PFNCLGETKERNELINFO                  __clewGetKernelInfo                 ;
CLEW_FUN_EXPORT     PFNCLGETKERNELWORKGROUPINFO         __clewGetKernelWorkGroupInfo        ;
CLEW_FUN_EXPORT     PFNCLWAITFOREVENTS                  __clewWaitForEvents                 ;
CLEW_FUN_EXPORT     PFNCLGETEVENTINFO                   __clewGetEventInfo                  ;
CLEW_FUN_EXPORT     PFNCLCREATEUSEREVENT                __clewCreateUserEvent               ;
CLEW_FUN_EXPORT     PFNCLRETAINEVENT                    __clewRetainEvent                   ;
CLEW_FUN_EXPORT     PFNCLRELEASEEVENT                   __clewReleaseEvent                  ;
CLEW_FUN_EXPORT     PFNCLSETUSEREVENTSTATUS             __clewSetUserEventStatus            ;
CLEW_FUN_EXPORT     PFNCLSETEVENTCALLBACK               __clewSetEventCallback              ;
CLEW_FUN_EXPORT     PFNCLGETEVENTPROFILINGINFO          __clewGetEventProfilingInfo         ;
CLEW_FUN_EXPORT     PFNCLFLUSH                          __clewFlush                         ;
CLEW_FUN_EXPORT     PFNCLFINISH                         __clewFinish                        ;
CLEW_FUN_EXPORT     PFNCLENQUEUEREADBUFFER              __clewEnqueueReadBuffer             ;
CLEW_FUN_EXPORT     PFNCLENQUEUEREADBUFFERRECT          __clewEnqueueReadBufferRect         ;
CLEW_FUN_EXPORT     PFNCLENQUEUEWRITEBUFFER             __clewEnqueueWriteBuffer            ;
CLEW_FUN_EXPORT     PFNCLENQUEUEWRITEBUFFERRECT         __clewEnqueueWriteBufferRect        ;
CLEW_FUN_EXPORT     PFNCLENQUEUECOPYBUFFER              __clewEnqueueCopyBuffer             ;
CLEW_FUN_EXPORT     PFNCLENQUEUECOPYBUFFERRECT          __clewEnqueueCopyBufferRect         ;
CLEW_FUN_EXPORT     PFNCLENQUEUEREADIMAGE               __clewEnqueueReadImage              ;
CLEW_FUN_EXPORT     PFNCLENQUEUEWRITEIMAGE              __clewEnqueueWriteImage             ;
CLEW_FUN_EXPORT     PFNCLENQUEUECOPYIMAGE               __clewEnqueueCopyImage              ;
CLEW_FUN_EXPORT     PFNCLENQUEUECOPYIMAGETOBUFFER       __clewEnqueueCopyImageToBuffer      ;
CLEW_FUN_EXPORT     PFNCLENQUEUECOPYBUFFERTOIMAGE       __clewEnqueueCopyBufferToImage      ;
CLEW_FUN_EXPORT     PFNCLENQUEUEMAPBUFFER               __clewEnqueueMapBuffer              ;
CLEW_FUN_EXPORT     PFNCLENQUEUEMAPIMAGE                __clewEnqueueMapImage               ;
CLEW_FUN_EXPORT     PFNCLENQUEUEUNMAPMEMOBJECT          __clewEnqueueUnmapMemObject         ;
CLEW_FUN_EXPORT     PFNCLENQUEUENDRANGEKERNEL           __clewEnqueueNDRangeKernel          ;
CLEW_FUN_EXPORT     PFNCLENQUEUETASK                    __clewEnqueueTask                   ;
CLEW_FUN_EXPORT     PFNCLENQUEUENATIVEKERNEL            __clewEnqueueNativeKernel           ;
CLEW_FUN_EXPORT     PFNCLENQUEUEMARKER                  __clewEnqueueMarker                 ;
CLEW_FUN_EXPORT     PFNCLENQUEUEWAITFOREVENTS           __clewEnqueueWaitForEvents          ;
CLEW_FUN_EXPORT     PFNCLENQUEUEBARRIER                 __clewEnqueueBarrier                ;
CLEW_FUN_EXPORT     PFNCLGETEXTENSIONFUNCTIONADDRESS    __clewGetExtensionFunctionAddress   ;


#define	clGetPlatformIDs                CLEW_GET_FUN(__clewGetPlatformIDs                )
#define	clGetPlatformInfo               CLEW_GET_FUN(__clewGetPlatformInfo               )
#define	clGetDeviceIDs                  CLEW_GET_FUN(__clewGetDeviceIDs                  )
#define	clGetDeviceInfo                 CLEW_GET_FUN(__clewGetDeviceInfo                 )
#define	clCreateContext                 CLEW_GET_FUN(__clewCreateContext                 )
#define	clCreateContextFromType         CLEW_GET_FUN(__clewCreateContextFromType         )
#define	clRetainContext                 CLEW_GET_FUN(__clewRetainContext                 )
#define	clReleaseContext                CLEW_GET_FUN(__clewReleaseContext                )
#define	clGetContextInfo                CLEW_GET_FUN(__clewGetContextInfo                )
#define	clCreateCommandQueue            CLEW_GET_FUN(__clewCreateCommandQueue            )
#define	clRetainCommandQueue            CLEW_GET_FUN(__clewRetainCommandQueue            )
#define	clReleaseCommandQueue           CLEW_GET_FUN(__clewReleaseCommandQueue           )
#define	clGetCommandQueueInfo           CLEW_GET_FUN(__clewGetCommandQueueInfo           )
#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#warning CL_USE_DEPRECATED_OPENCL_1_0_APIS is defined. These APIs are unsupported and untested in OpenCL 1.1!
/* 
 *  WARNING:
 *     This API introduces mutable state into the OpenCL implementation. It has been REMOVED
 *  to better facilitate thread safety.  The 1.0 API is not thread safe. It is not tested by the
 *  OpenCL 1.1 conformance test, and consequently may not work or may not work dependably.
 *  It is likely to be non-performant. Use of this API is not advised. Use at your own risk.
 *
 *  Software developers previously relying on this API are instructed to set the command queue 
 *  properties when creating the queue, instead. 
 */
#define	clSetCommandQueueProperty       CLEW_GET_FUN(__clewSetCommandQueueProperty       )
#endif /* CL_USE_DEPRECATED_OPENCL_1_0_APIS */
#define	clCreateBuffer                  CLEW_GET_FUN(__clewCreateBuffer                  )
#define	clCreateSubBuffer               CLEW_GET_FUN(__clewCreateSubBuffer               )
#define	clCreateImage2D                 CLEW_GET_FUN(__clewCreateImage2D                 )
#define	clCreateImage3D                 CLEW_GET_FUN(__clewCreateImage3D                 )
#define	clRetainMemObject               CLEW_GET_FUN(__clewRetainMemObject               )
#define	clReleaseMemObject              CLEW_GET_FUN(__clewReleaseMemObject              )
#define	clGetSupportedImageFormats      CLEW_GET_FUN(__clewGetSupportedImageFormats      )
#define	clGetMemObjectInfo              CLEW_GET_FUN(__clewGetMemObjectInfo              )
#define	clGetImageInfo                  CLEW_GET_FUN(__clewGetImageInfo                  )
#define	clSetMemObjectDestructorCallback CLEW_GET_FUN(__clewSetMemObjectDestructorCallback)
#define	clCreateSampler                 CLEW_GET_FUN(__clewCreateSampler                 )
#define	clRetainSampler                 CLEW_GET_FUN(__clewRetainSampler                 )
#define	clReleaseSampler                CLEW_GET_FUN(__clewReleaseSampler                )
#define	clGetSamplerInfo                CLEW_GET_FUN(__clewGetSamplerInfo                )
#define	clCreateProgramWithSource       CLEW_GET_FUN(__clewCreateProgramWithSource       )
#define	clCreateProgramWithBinary       CLEW_GET_FUN(__clewCreateProgramWithBinary       )
#define	clRetainProgram                 CLEW_GET_FUN(__clewRetainProgram                 )
#define	clReleaseProgram                CLEW_GET_FUN(__clewReleaseProgram                )
#define	clBuildProgram                  CLEW_GET_FUN(__clewBuildProgram                  )
#define	clUnloadCompiler                CLEW_GET_FUN(__clewUnloadCompiler                )
#define	clGetProgramInfo                CLEW_GET_FUN(__clewGetProgramInfo                )
#define	clGetProgramBuildInfo           CLEW_GET_FUN(__clewGetProgramBuildInfo           )
#define	clCreateKernel                  CLEW_GET_FUN(__clewCreateKernel                  )
#define	clCreateKernelsInProgram        CLEW_GET_FUN(__clewCreateKernelsInProgram        )
#define	clRetainKernel                  CLEW_GET_FUN(__clewRetainKernel                  )
#define	clReleaseKernel                 CLEW_GET_FUN(__clewReleaseKernel                 )
#define	clSetKernelArg                  CLEW_GET_FUN(__clewSetKernelArg                  )
#define	clGetKernelInfo                 CLEW_GET_FUN(__clewGetKernelInfo                 )
#define	clGetKernelWorkGroupInfo        CLEW_GET_FUN(__clewGetKernelWorkGroupInfo        )
#define	clWaitForEvents                 CLEW_GET_FUN(__clewWaitForEvents                 )
#define	clGetEventInfo                  CLEW_GET_FUN(__clewGetEventInfo                  )
#define	clCreateUserEvent               CLEW_GET_FUN(__clewCreateUserEvent               )
#define	clRetainEvent                   CLEW_GET_FUN(__clewRetainEvent                   )
#define	clReleaseEvent                  CLEW_GET_FUN(__clewReleaseEvent                  )
#define	clSetUserEventStatus            CLEW_GET_FUN(__clewSetUserEventStatus            )
#define	clSetEventCallback              CLEW_GET_FUN(__clewSetEventCallback              )
#define	clGetEventProfilingInfo         CLEW_GET_FUN(__clewGetEventProfilingInfo         )
#define	clFlush                         CLEW_GET_FUN(__clewFlush                         )
#define	clFinish                        CLEW_GET_FUN(__clewFinish                        )
#define	clEnqueueReadBuffer             CLEW_GET_FUN(__clewEnqueueReadBuffer             )
#define	clEnqueueReadBufferRect         CLEW_GET_FUN(__clewEnqueueReadBufferRect         )
#define	clEnqueueWriteBuffer            CLEW_GET_FUN(__clewEnqueueWriteBuffer            )
#define	clEnqueueWriteBufferRect        CLEW_GET_FUN(__clewEnqueueWriteBufferRect        )
#define	clEnqueueCopyBuffer             CLEW_GET_FUN(__clewEnqueueCopyBuffer             )
#define	clEnqueueCopyBufferRect         CLEW_GET_FUN(__clewEnqueueCopyBufferRect         )
#define	clEnqueueReadImage              CLEW_GET_FUN(__clewEnqueueReadImage              )
#define	clEnqueueWriteImage             CLEW_GET_FUN(__clewEnqueueWriteImage             )
#define	clEnqueueCopyImage              CLEW_GET_FUN(__clewEnqueueCopyImage              )
#define	clEnqueueCopyImageToBuffer      CLEW_GET_FUN(__clewEnqueueCopyImageToBuffer      )
#define	clEnqueueCopyBufferToImage      CLEW_GET_FUN(__clewEnqueueCopyBufferToImage      )
#define	clEnqueueMapBuffer              CLEW_GET_FUN(__clewEnqueueMapBuffer              )
#define	clEnqueueMapImage               CLEW_GET_FUN(__clewEnqueueMapImage               )
#define	clEnqueueUnmapMemObject         CLEW_GET_FUN(__clewEnqueueUnmapMemObject         )
#define	clEnqueueNDRangeKernel          CLEW_GET_FUN(__clewEnqueueNDRangeKernel          )
#define	clEnqueueTask                   CLEW_GET_FUN(__clewEnqueueTask                   )
#define	clEnqueueNativeKernel           CLEW_GET_FUN(__clewEnqueueNativeKernel           )
#define	clEnqueueMarker                 CLEW_GET_FUN(__clewEnqueueMarker                 )
#define	clEnqueueWaitForEvents          CLEW_GET_FUN(__clewEnqueueWaitForEvents          )
#define	clEnqueueBarrier                CLEW_GET_FUN(__clewEnqueueBarrier                )
#define	clGetExtensionFunctionAddress   CLEW_GET_FUN(__clewGetExtensionFunctionAddress   )


#define CLEW_SUCCESS                0       //!<    Success error code
#define CLEW_ERROR_OPEN_FAILED      -1      //!<    Error code for failing to open the dynamic library
#define CLEW_ERROR_ATEXIT_FAILED    -2      //!<    Error code for failing to queue the closing of the dynamic library to atexit()

//! \brief Load OpenCL dynamic library and set function entry points
int         clewInit        (const char*);

//! \brief Exit clew and unload OpenCL dynamic library
void		clewExit();

//! \brief Convert an OpenCL error code to its string equivalent
const char* clewErrorString (cl_int error);

#ifdef __cplusplus
}
#endif

#endif  //  CLEW_HPP_INCLUDED
