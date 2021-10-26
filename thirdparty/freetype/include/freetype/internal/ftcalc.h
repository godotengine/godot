/****************************************************************************
 *
 * ftcalc.h
 *
 *   Arithmetic computations (specification).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTCALC_H_
#define FTCALC_H_


#include <freetype/freetype.h>

#include "compiler-macros.h"

FT_BEGIN_HEADER


  /**************************************************************************
   *
   * FT_MulDiv() and FT_MulFix() are declared in freetype.h.
   *
   */

#ifndef  FT_CONFIG_OPTION_NO_ASSEMBLER
  /* Provide assembler fragments for performance-critical functions. */
  /* These must be defined `static __inline__' with GCC.             */

#if defined( __CC_ARM ) || defined( __ARMCC__ )  /* RVCT */

#define FT_MULFIX_ASSEMBLER  FT_MulFix_arm

  /* documentation is in freetype.h */

  static __inline FT_Int32
  FT_MulFix_arm( FT_Int32  a,
                 FT_Int32  b )
  {
    FT_Int32  t, t2;


    __asm
    {
      smull t2, t,  b,  a           /* (lo=t2,hi=t) = a*b */
      mov   a,  t,  asr #31         /* a   = (hi >> 31) */
      add   a,  a,  #0x8000         /* a  += 0x8000 */
      adds  t2, t2, a               /* t2 += a */
      adc   t,  t,  #0              /* t  += carry */
      mov   a,  t2, lsr #16         /* a   = t2 >> 16 */
      orr   a,  a,  t,  lsl #16     /* a  |= t << 16 */
    }
    return a;
  }

#endif /* __CC_ARM || __ARMCC__ */


#ifdef __GNUC__

#if defined( __arm__ )                                 && \
    ( !defined( __thumb__ ) || defined( __thumb2__ ) ) && \
    !( defined( __CC_ARM ) || defined( __ARMCC__ ) )

#define FT_MULFIX_ASSEMBLER  FT_MulFix_arm

  /* documentation is in freetype.h */

  static __inline__ FT_Int32
  FT_MulFix_arm( FT_Int32  a,
                 FT_Int32  b )
  {
    FT_Int32  t, t2;


    __asm__ __volatile__ (
      "smull  %1, %2, %4, %3\n\t"       /* (lo=%1,hi=%2) = a*b */
      "mov    %0, %2, asr #31\n\t"      /* %0  = (hi >> 31) */
#if defined( __clang__ ) && defined( __thumb2__ )
      "add.w  %0, %0, #0x8000\n\t"      /* %0 += 0x8000 */
#else
      "add    %0, %0, #0x8000\n\t"      /* %0 += 0x8000 */
#endif
      "adds   %1, %1, %0\n\t"           /* %1 += %0 */
      "adc    %2, %2, #0\n\t"           /* %2 += carry */
      "mov    %0, %1, lsr #16\n\t"      /* %0  = %1 >> 16 */
      "orr    %0, %0, %2, lsl #16\n\t"  /* %0 |= %2 << 16 */
      : "=r"(a), "=&r"(t2), "=&r"(t)
      : "r"(a), "r"(b)
      : "cc" );
    return a;
  }

#endif /* __arm__                      && */
       /* ( __thumb2__ || !__thumb__ ) && */
       /* !( __CC_ARM || __ARMCC__ )      */


#if defined( __i386__ )

#define FT_MULFIX_ASSEMBLER  FT_MulFix_i386

  /* documentation is in freetype.h */

  static __inline__ FT_Int32
  FT_MulFix_i386( FT_Int32  a,
                  FT_Int32  b )
  {
    FT_Int32  result;


    __asm__ __volatile__ (
      "imul  %%edx\n"
      "movl  %%edx, %%ecx\n"
      "sarl  $31, %%ecx\n"
      "addl  $0x8000, %%ecx\n"
      "addl  %%ecx, %%eax\n"
      "adcl  $0, %%edx\n"
      "shrl  $16, %%eax\n"
      "shll  $16, %%edx\n"
      "addl  %%edx, %%eax\n"
      : "=a"(result), "=d"(b)
      : "a"(a), "d"(b)
      : "%ecx", "cc" );
    return result;
  }

#endif /* i386 */

#endif /* __GNUC__ */


#ifdef _MSC_VER /* Visual C++ */

#ifdef _M_IX86

#define FT_MULFIX_ASSEMBLER  FT_MulFix_i386

  /* documentation is in freetype.h */

  static __inline FT_Int32
  FT_MulFix_i386( FT_Int32  a,
                  FT_Int32  b )
  {
    FT_Int32  result;

    __asm
    {
      mov eax, a
      mov edx, b
      imul edx
      mov ecx, edx
      sar ecx, 31
      add ecx, 8000h
      add eax, ecx
      adc edx, 0
      shr eax, 16
      shl edx, 16
      add eax, edx
      mov result, eax
    }
    return result;
  }

#endif /* _M_IX86 */

#endif /* _MSC_VER */


#if defined( __GNUC__ ) && defined( __x86_64__ )

#define FT_MULFIX_ASSEMBLER  FT_MulFix_x86_64

  static __inline__ FT_Int32
  FT_MulFix_x86_64( FT_Int32  a,
                    FT_Int32  b )
  {
    /* Temporarily disable the warning that C90 doesn't support */
    /* `long long'.                                             */
#if __GNUC__ > 4 || ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlong-long"
#endif

#if 1
    /* Technically not an assembly fragment, but GCC does a really good */
    /* job at inlining it and generating good machine code for it.      */
    long long  ret, tmp;


    ret  = (long long)a * b;
    tmp  = ret >> 63;
    ret += 0x8000 + tmp;

    return (FT_Int32)( ret >> 16 );
#else

    /* For some reason, GCC 4.6 on Ubuntu 12.04 generates invalid machine  */
    /* code from the lines below.  The main issue is that `wide_a' is not  */
    /* properly initialized by sign-extending `a'.  Instead, the generated */
    /* machine code assumes that the register that contains `a' on input   */
    /* can be used directly as a 64-bit value, which is wrong most of the  */
    /* time.                                                               */
    long long  wide_a = (long long)a;
    long long  wide_b = (long long)b;
    long long  result;


    __asm__ __volatile__ (
      "imul %2, %1\n"
      "mov %1, %0\n"
      "sar $63, %0\n"
      "lea 0x8000(%1, %0), %0\n"
      "sar $16, %0\n"
      : "=&r"(result), "=&r"(wide_a)
      : "r"(wide_b)
      : "cc" );

    return (FT_Int32)result;
#endif

#if __GNUC__ > 4 || ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 )
#pragma GCC diagnostic pop
#endif
  }

#endif /* __GNUC__ && __x86_64__ */

#endif /* !FT_CONFIG_OPTION_NO_ASSEMBLER */


#ifdef FT_CONFIG_OPTION_INLINE_MULFIX
#ifdef FT_MULFIX_ASSEMBLER
#define FT_MulFix( a, b )  FT_MULFIX_ASSEMBLER( (FT_Int32)(a), (FT_Int32)(b) )
#endif
#endif


  /**************************************************************************
   *
   * @function:
   *   FT_MulDiv_No_Round
   *
   * @description:
   *   A very simple function used to perform the computation '(a*b)/c'
   *   (without rounding) with maximum accuracy (it uses a 64-bit
   *   intermediate integer whenever necessary).
   *
   *   This function isn't necessarily as fast as some processor-specific
   *   operations, but is at least completely portable.
   *
   * @input:
   *   a ::
   *     The first multiplier.
   *   b ::
   *     The second multiplier.
   *   c ::
   *     The divisor.
   *
   * @return:
   *   The result of '(a*b)/c'.  This function never traps when trying to
   *   divide by zero; it simply returns 'MaxInt' or 'MinInt' depending on
   *   the signs of 'a' and 'b'.
   */
  FT_BASE( FT_Long )
  FT_MulDiv_No_Round( FT_Long  a,
                      FT_Long  b,
                      FT_Long  c );


  /*
   * A variant of FT_Matrix_Multiply which scales its result afterwards.  The
   * idea is that both `a' and `b' are scaled by factors of 10 so that the
   * values are as precise as possible to get a correct result during the
   * 64bit multiplication.  Let `sa' and `sb' be the scaling factors of `a'
   * and `b', respectively, then the scaling factor of the result is `sa*sb'.
   */
  FT_BASE( void )
  FT_Matrix_Multiply_Scaled( const FT_Matrix*  a,
                             FT_Matrix        *b,
                             FT_Long           scaling );


  /*
   * Check a matrix.  If the transformation would lead to extreme shear or
   * extreme scaling, for example, return 0.  If everything is OK, return 1.
   *
   * Based on geometric considerations we use the following inequality to
   * identify a degenerate matrix.
   *
   *   50 * abs(xx*yy - xy*yx) < xx^2 + xy^2 + yx^2 + yy^2
   *
   * Value 50 is heuristic.
   */
  FT_BASE( FT_Bool )
  FT_Matrix_Check( const FT_Matrix*  matrix );


  /*
   * A variant of FT_Vector_Transform.  See comments for
   * FT_Matrix_Multiply_Scaled.
   */
  FT_BASE( void )
  FT_Vector_Transform_Scaled( FT_Vector*        vector,
                              const FT_Matrix*  matrix,
                              FT_Long           scaling );


  /*
   * This function normalizes a vector and returns its original length.  The
   * normalized vector is a 16.16 fixed-point unit vector with length close
   * to 0x10000.  The accuracy of the returned length is limited to 16 bits
   * also.  The function utilizes quick inverse square root approximation
   * without divisions and square roots relying on Newton's iterations
   * instead.
   */
  FT_BASE( FT_UInt32 )
  FT_Vector_NormLen( FT_Vector*  vector );


  /*
   * Return -1, 0, or +1, depending on the orientation of a given corner.  We
   * use the Cartesian coordinate system, with positive vertical values going
   * upwards.  The function returns +1 if the corner turns to the left, -1 to
   * the right, and 0 for undecidable cases.
   */
  FT_BASE( FT_Int )
  ft_corner_orientation( FT_Pos  in_x,
                         FT_Pos  in_y,
                         FT_Pos  out_x,
                         FT_Pos  out_y );


  /*
   * Return TRUE if a corner is flat or nearly flat.  This is equivalent to
   * saying that the corner point is close to its neighbors, or inside an
   * ellipse defined by the neighbor focal points to be more precise.
   */
  FT_BASE( FT_Int )
  ft_corner_is_flat( FT_Pos  in_x,
                     FT_Pos  in_y,
                     FT_Pos  out_x,
                     FT_Pos  out_y );


  /*
   * Return the most significant bit index.
   */

#ifndef  FT_CONFIG_OPTION_NO_ASSEMBLER

#if defined( __GNUC__ )                                          && \
    ( __GNUC__ > 3 || ( __GNUC__ == 3 && __GNUC_MINOR__ >= 4 ) )

#if FT_SIZEOF_INT == 4

#define FT_MSB( x )  ( 31 - __builtin_clz( x ) )

#elif FT_SIZEOF_LONG == 4

#define FT_MSB( x )  ( 31 - __builtin_clzl( x ) )

#endif /* __GNUC__ */


#elif defined( _MSC_VER ) && ( _MSC_VER >= 1400 )

#if FT_SIZEOF_INT == 4

#include <intrin.h>
#pragma intrinsic( _BitScanReverse )

  static __inline FT_Int32
  FT_MSB_i386( FT_UInt32  x )
  {
    unsigned long  where;


    _BitScanReverse( &where, x );

    return (FT_Int32)where;
  }

#define FT_MSB( x )  ( FT_MSB_i386( x ) )

#endif

#endif /* _MSC_VER */


#endif /* !FT_CONFIG_OPTION_NO_ASSEMBLER */

#ifndef FT_MSB

  FT_BASE( FT_Int )
  FT_MSB( FT_UInt32  z );

#endif


  /*
   * Return sqrt(x*x+y*y), which is the same as `FT_Vector_Length' but uses
   * two fixed-point arguments instead.
   */
  FT_BASE( FT_Fixed )
  FT_Hypot( FT_Fixed  x,
            FT_Fixed  y );


#if 0

  /**************************************************************************
   *
   * @function:
   *   FT_SqrtFixed
   *
   * @description:
   *   Computes the square root of a 16.16 fixed-point value.
   *
   * @input:
   *   x ::
   *     The value to compute the root for.
   *
   * @return:
   *   The result of 'sqrt(x)'.
   *
   * @note:
   *   This function is not very fast.
   */
  FT_BASE( FT_Int32 )
  FT_SqrtFixed( FT_Int32  x );

#endif /* 0 */


#define INT_TO_F26DOT6( x )    ( (FT_Long)(x) * 64  )    /* << 6  */
#define INT_TO_F2DOT14( x )    ( (FT_Long)(x) * 16384 )  /* << 14 */
#define INT_TO_FIXED( x )      ( (FT_Long)(x) * 65536 )  /* << 16 */
#define F2DOT14_TO_FIXED( x )  ( (FT_Long)(x) * 4 )      /* << 2  */
#define FIXED_TO_INT( x )      ( FT_RoundFix( x ) >> 16 )

#define ROUND_F26DOT6( x )     ( ( (x) + 32 - ( x < 0 ) ) & -64 )

  /*
   * The following macros have two purposes.
   *
   * - Tag places where overflow is expected and harmless.
   *
   * - Avoid run-time sanitizer errors.
   *
   * Use with care!
   */
#define ADD_INT( a, b )                           \
          (FT_Int)( (FT_UInt)(a) + (FT_UInt)(b) )
#define SUB_INT( a, b )                           \
          (FT_Int)( (FT_UInt)(a) - (FT_UInt)(b) )
#define MUL_INT( a, b )                           \
          (FT_Int)( (FT_UInt)(a) * (FT_UInt)(b) )
#define NEG_INT( a )                              \
          (FT_Int)( (FT_UInt)0 - (FT_UInt)(a) )

#define ADD_LONG( a, b )                             \
          (FT_Long)( (FT_ULong)(a) + (FT_ULong)(b) )
#define SUB_LONG( a, b )                             \
          (FT_Long)( (FT_ULong)(a) - (FT_ULong)(b) )
#define MUL_LONG( a, b )                             \
          (FT_Long)( (FT_ULong)(a) * (FT_ULong)(b) )
#define NEG_LONG( a )                                \
          (FT_Long)( (FT_ULong)0 - (FT_ULong)(a) )

#define ADD_INT32( a, b )                               \
          (FT_Int32)( (FT_UInt32)(a) + (FT_UInt32)(b) )
#define SUB_INT32( a, b )                               \
          (FT_Int32)( (FT_UInt32)(a) - (FT_UInt32)(b) )
#define MUL_INT32( a, b )                               \
          (FT_Int32)( (FT_UInt32)(a) * (FT_UInt32)(b) )
#define NEG_INT32( a )                                  \
          (FT_Int32)( (FT_UInt32)0 - (FT_UInt32)(a) )

#ifdef FT_LONG64

#define ADD_INT64( a, b )                               \
          (FT_Int64)( (FT_UInt64)(a) + (FT_UInt64)(b) )
#define SUB_INT64( a, b )                               \
          (FT_Int64)( (FT_UInt64)(a) - (FT_UInt64)(b) )
#define MUL_INT64( a, b )                               \
          (FT_Int64)( (FT_UInt64)(a) * (FT_UInt64)(b) )
#define NEG_INT64( a )                                  \
          (FT_Int64)( (FT_UInt64)0 - (FT_UInt64)(a) )

#endif /* FT_LONG64 */


FT_END_HEADER

#endif /* FTCALC_H_ */


/* END */
