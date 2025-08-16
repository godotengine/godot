/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * from: @(#)fdlibm.h 5.1 93/09/24
 * $Id: math_private.h,v 1.3 2004/02/09 07:10:38 andersen Exp $
 */

#ifndef _MATH_PRIVATE_H_
#define _MATH_PRIVATE_H_

/* #include <endian.h> */
/* #include <sys/types.h> */

#define _IEEE_LIBM
#define attribute_hidden
#define libm_hidden_proto(x)
#define libm_hidden_def(x)
#define strong_alias(x, y)
#define weak_alias(x, y)

#if !defined(SDL_PLATFORM_HAIKU) && !defined(SDL_PLATFORM_PSP) && !defined(SDL_PLATFORM_3DS) && !defined(SDL_PLATFORM_PS2) /* already defined in a system header. */
typedef unsigned int u_int32_t;
#endif

#define atan            SDL_uclibc_atan
#define __ieee754_atan2 SDL_uclibc_atan2
#define copysign        SDL_uclibc_copysign
#define cos             SDL_uclibc_cos
#define __ieee754_exp   SDL_uclibc_exp
#define fabs            SDL_uclibc_fabs
#define floor           SDL_uclibc_floor
#define __ieee754_fmod  SDL_uclibc_fmod
#undef __isinf
#define __isinf         SDL_uclibc_isinf
#undef __isinff
#define __isinff        SDL_uclibc_isinff
#undef __isnan
#define __isnan         SDL_uclibc_isnan
#undef __isnanf
#define __isnanf        SDL_uclibc_isnanf
#define __ieee754_log   SDL_uclibc_log
#define __ieee754_log10 SDL_uclibc_log10
#define modf            SDL_uclibc_modf
#define __ieee754_pow   SDL_uclibc_pow
#define scalbln         SDL_uclibc_scalbln
#define scalbn          SDL_uclibc_scalbn
#define sin             SDL_uclibc_sin
#define __ieee754_sqrt  SDL_uclibc_sqrt
#define tan             SDL_uclibc_tan

/* The original fdlibm code used statements like:
	n0 = ((*(int*)&one)>>29)^1;		* index of high word *
	ix0 = *(n0+(int*)&x);			* high word of x *
	ix1 = *((1-n0)+(int*)&x);		* low word of x *
   to dig two 32 bit words out of the 64 bit IEEE floating point
   value.  That is non-ANSI, and, moreover, the gcc instruction
   scheduler gets it wrong.  We instead use the following macros.
   Unlike the original code, we determine the endianness at compile
   time, not at run time; I don't see much benefit to selecting
   endianness at run time.  */

/* A union which permits us to convert between a double and two 32 bit
   ints.  */

/*
 * Math on arm is special:
 * For FPA, float words are always big-endian.
 * For VFP, floats words follow the memory system mode.
 * For Maverick, float words are always little-endian.
 */

#if (SDL_FLOATWORDORDER == SDL_BIG_ENDIAN)

typedef union
{
    double value;
    struct
    {
        u_int32_t msw;
        u_int32_t lsw;
    } parts;
} ieee_double_shape_type;

#else

typedef union
{
    double value;
    struct
    {
        u_int32_t lsw;
        u_int32_t msw;
    } parts;
} ieee_double_shape_type;

#endif

/* Get two 32 bit ints from a double.  */

#define EXTRACT_WORDS(ix0,ix1,d)				\
do {								\
  ieee_double_shape_type ew_u;					\
  ew_u.value = (d);						\
  (ix0) = ew_u.parts.msw;					\
  (ix1) = ew_u.parts.lsw;					\
} while (0)

/* Get the more significant 32 bit int from a double.  */

#define GET_HIGH_WORD(i,d)					\
do {								\
  ieee_double_shape_type gh_u;					\
  gh_u.value = (d);						\
  (i) = gh_u.parts.msw;						\
} while (0)

/* Get the less significant 32 bit int from a double.  */

#define GET_LOW_WORD(i,d)					\
do {								\
  ieee_double_shape_type gl_u;					\
  gl_u.value = (d);						\
  (i) = gl_u.parts.lsw;						\
} while (0)

/* Set a double from two 32 bit ints.  */

#define INSERT_WORDS(d,ix0,ix1)					\
do {								\
  ieee_double_shape_type iw_u;					\
  iw_u.parts.msw = (ix0);					\
  iw_u.parts.lsw = (ix1);					\
  (d) = iw_u.value;						\
} while (0)

/* Set the more significant 32 bits of a double from an int.  */

#define SET_HIGH_WORD(d,v)					\
do {								\
  ieee_double_shape_type sh_u;					\
  sh_u.value = (d);						\
  sh_u.parts.msw = (v);						\
  (d) = sh_u.value;						\
} while (0)

/* Set the less significant 32 bits of a double from an int.  */

#define SET_LOW_WORD(d,v)					\
do {								\
  ieee_double_shape_type sl_u;					\
  sl_u.value = (d);						\
  sl_u.parts.lsw = (v);						\
  (d) = sl_u.value;						\
} while (0)

/* A union which permits us to convert between a float and a 32 bit
   int.  */

typedef union
{
    float value;
    u_int32_t word;
} ieee_float_shape_type;

/* Get a 32 bit int from a float.  */

#define GET_FLOAT_WORD(i,d)					\
do {								\
  ieee_float_shape_type gf_u;					\
  gf_u.value = (d);						\
  (i) = gf_u.word;						\
} while (0)

/* Set a float from a 32 bit int.  */

#define SET_FLOAT_WORD(d,i)					\
do {								\
  ieee_float_shape_type sf_u;					\
  sf_u.word = (i);						\
  (d) = sf_u.value;						\
} while (0)

/* ieee style elementary functions */
extern double __ieee754_sqrt(double) attribute_hidden;
extern double __ieee754_acos(double) attribute_hidden;
extern double __ieee754_acosh(double) attribute_hidden;
extern double __ieee754_log(double) attribute_hidden;
extern double __ieee754_atanh(double) attribute_hidden;
extern double __ieee754_asin(double) attribute_hidden;
extern double __ieee754_atan2(double, double) attribute_hidden;
extern double __ieee754_exp(double) attribute_hidden;
extern double __ieee754_cosh(double) attribute_hidden;
extern double __ieee754_fmod(double, double) attribute_hidden;
extern double __ieee754_pow(double, double) attribute_hidden;
extern double __ieee754_lgamma_r(double, int *) attribute_hidden;
extern double __ieee754_gamma_r(double, int *) attribute_hidden;
extern double __ieee754_lgamma(double) attribute_hidden;
extern double __ieee754_gamma(double) attribute_hidden;
extern double __ieee754_log10(double) attribute_hidden;
extern double __ieee754_sinh(double) attribute_hidden;
extern double __ieee754_hypot(double, double) attribute_hidden;
extern double __ieee754_j0(double) attribute_hidden;
extern double __ieee754_j1(double) attribute_hidden;
extern double __ieee754_y0(double) attribute_hidden;
extern double __ieee754_y1(double) attribute_hidden;
extern double __ieee754_jn(int, double) attribute_hidden;
extern double __ieee754_yn(int, double) attribute_hidden;
extern double __ieee754_remainder(double, double) attribute_hidden;
extern int32_t __ieee754_rem_pio2(double, double *) attribute_hidden;
#if defined(_SCALB_INT)
extern double __ieee754_scalb(double, int) attribute_hidden;
#else
extern double __ieee754_scalb(double, double) attribute_hidden;
#endif

/* fdlibm kernel function */
#ifndef _IEEE_LIBM
extern double __kernel_standard(double, double, int) attribute_hidden;
#endif
extern double __kernel_sin(double, double, int) attribute_hidden;
extern double __kernel_cos(double, double) attribute_hidden;
extern double __kernel_tan(double, double, int) attribute_hidden;
extern int32_t __kernel_rem_pio2(const double *, double *, int, int, const unsigned int, const int32_t *) attribute_hidden;

#endif /* _MATH_PRIVATE_H_ */
