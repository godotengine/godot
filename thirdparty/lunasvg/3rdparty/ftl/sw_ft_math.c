/***************************************************************************/
/*                                                                         */
/*  fttrigon.c                                                             */
/*                                                                         */
/*    FreeType trigonometric functions (body).                             */
/*                                                                         */
/*  Copyright 2001-2005, 2012-2013 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include "sw_ft_math.h"
#include <math.h>

//form https://github.com/chromium/chromium/blob/59afd8336009c9d97c22854c52e0382b62b3aa5e/third_party/abseil-cpp/absl/base/internal/bits.h

#if defined(_MSC_VER)
#include <intrin.h>
static unsigned int __inline clz(unsigned int x) {
	unsigned long r = 0;
	if (x != 0)
	{
		_BitScanReverse(&r, x);
	}
	return  r;
}
#define SW_FT_MSB(x)  (clz(x))
#elif defined(__GNUC__)
#define SW_FT_MSB(x)  (31 - __builtin_clz(x))
#else
static unsigned int __inline clz(unsigned int x) {
	int c = 31;
	x &= ~x + 1;
	if (n & 0x0000FFFF) c -= 16;
	if (n & 0x00FF00FF) c -= 8;
	if (n & 0x0F0F0F0F) c -= 4;
	if (n & 0x33333333) c -= 2;
	if (n & 0x55555555) c -= 1;
	return c;
}
#define SW_FT_MSB(x)  (clz(x))
#endif





#define SW_FT_PAD_FLOOR(x, n) ((x) & ~((n)-1))
#define SW_FT_PAD_ROUND(x, n) SW_FT_PAD_FLOOR((x) + ((n) / 2), n)
#define SW_FT_PAD_CEIL(x, n) SW_FT_PAD_FLOOR((x) + ((n)-1), n)

#define SW_FT_BEGIN_STMNT do {
#define SW_FT_END_STMNT \
    }                   \
    while (0)
/* transfer sign leaving a positive number */
#define SW_FT_MOVE_SIGN(x, s) \
    SW_FT_BEGIN_STMNT         \
    if (x < 0) {              \
        x = -x;               \
        s = -s;               \
    }                         \
    SW_FT_END_STMNT

SW_FT_Long SW_FT_MulFix(SW_FT_Long a, SW_FT_Long b)
{
    SW_FT_Int  s = 1;
    SW_FT_Long c;

    SW_FT_MOVE_SIGN(a, s);
    SW_FT_MOVE_SIGN(b, s);

    c = (SW_FT_Long)(((SW_FT_Int64)a * b + 0x8000L) >> 16);

    return (s > 0) ? c : -c;
}

SW_FT_Long SW_FT_MulDiv(SW_FT_Long a, SW_FT_Long b, SW_FT_Long c)
{
    SW_FT_Int  s = 1;
    SW_FT_Long d;

    SW_FT_MOVE_SIGN(a, s);
    SW_FT_MOVE_SIGN(b, s);
    SW_FT_MOVE_SIGN(c, s);

    d = (SW_FT_Long)(c > 0 ? ((SW_FT_Int64)a * b + (c >> 1)) / c : 0x7FFFFFFFL);

    return (s > 0) ? d : -d;
}

SW_FT_Long SW_FT_DivFix(SW_FT_Long a, SW_FT_Long b)
{
    SW_FT_Int  s = 1;
    SW_FT_Long q;

    SW_FT_MOVE_SIGN(a, s);
    SW_FT_MOVE_SIGN(b, s);

    q = (SW_FT_Long)(b > 0 ? (((SW_FT_UInt64)a << 16) + (b >> 1)) / b
                           : 0x7FFFFFFFL);

    return (s < 0 ? -q : q);
}

/*************************************************************************/
/*                                                                       */
/* This is a fixed-point CORDIC implementation of trigonometric          */
/* functions as well as transformations between Cartesian and polar      */
/* coordinates.  The angles are represented as 16.16 fixed-point values  */
/* in degrees, i.e., the angular resolution is 2^-16 degrees.  Note that */
/* only vectors longer than 2^16*180/pi (or at least 22 bits) on a       */
/* discrete Cartesian grid can have the same or better angular           */
/* resolution.  Therefore, to maintain this precision, some functions    */
/* require an interim upscaling of the vectors, whereas others operate   */
/* with 24-bit long vectors directly.                                    */
/*                                                                       */
/*************************************************************************/

/* the Cordic shrink factor 0.858785336480436 * 2^32 */
#define SW_FT_TRIG_SCALE 0xDBD95B16UL

/* the highest bit in overflow-safe vector components, */
/* MSB of 0.858785336480436 * sqrt(0.5) * 2^30         */
#define SW_FT_TRIG_SAFE_MSB 29

/* this table was generated for SW_FT_PI = 180L << 16, i.e. degrees */
#define SW_FT_TRIG_MAX_ITERS 23

static const SW_FT_Fixed ft_trig_arctan_table[] = {
    1740967L, 919879L, 466945L, 234379L, 117304L, 58666L, 29335L, 14668L,
    7334L,    3667L,   1833L,   917L,    458L,    229L,   115L,   57L,
    29L,      14L,     7L,      4L,      2L,      1L};

/* multiply a given value by the CORDIC shrink factor */
static SW_FT_Fixed ft_trig_downscale(SW_FT_Fixed val)
{
    SW_FT_Fixed s;
    SW_FT_Int64 v;

    s = val;
    val = SW_FT_ABS(val);

    v = (val * (SW_FT_Int64)SW_FT_TRIG_SCALE) + 0x100000000UL;
    val = (SW_FT_Fixed)(v >> 32);

    return (s >= 0) ? val : -val;
}

/* undefined and never called for zero vector */
static SW_FT_Int ft_trig_prenorm(SW_FT_Vector* vec)
{
    SW_FT_Pos x, y;
    SW_FT_Int shift;

    x = vec->x;
    y = vec->y;

    shift = SW_FT_MSB(SW_FT_ABS(x) | SW_FT_ABS(y));

    if (shift <= SW_FT_TRIG_SAFE_MSB) {
        shift = SW_FT_TRIG_SAFE_MSB - shift;
        vec->x = (SW_FT_Pos)((SW_FT_ULong)x << shift);
        vec->y = (SW_FT_Pos)((SW_FT_ULong)y << shift);
    } else {
        shift -= SW_FT_TRIG_SAFE_MSB;
        vec->x = x >> shift;
        vec->y = y >> shift;
        shift = -shift;
    }

    return shift;
}

static void ft_trig_pseudo_rotate(SW_FT_Vector* vec, SW_FT_Angle theta)
{
    SW_FT_Int          i;
    SW_FT_Fixed        x, y, xtemp, b;
    const SW_FT_Fixed* arctanptr;

    x = vec->x;
    y = vec->y;

    /* Rotate inside [-PI/4,PI/4] sector */
    while (theta < -SW_FT_ANGLE_PI4) {
        xtemp = y;
        y = -x;
        x = xtemp;
        theta += SW_FT_ANGLE_PI2;
    }

    while (theta > SW_FT_ANGLE_PI4) {
        xtemp = -y;
        y = x;
        x = xtemp;
        theta -= SW_FT_ANGLE_PI2;
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for (i = 1, b = 1; i < SW_FT_TRIG_MAX_ITERS; b <<= 1, i++) {
        SW_FT_Fixed v1 = ((y + b) >> i);
        SW_FT_Fixed v2 = ((x + b) >> i);
        if (theta < 0) {
            xtemp = x + v1;
            y = y - v2;
            x = xtemp;
            theta += *arctanptr++;
        } else {
            xtemp = x - v1;
            y = y + v2;
            x = xtemp;
            theta -= *arctanptr++;
        }
    }

    vec->x = x;
    vec->y = y;
}

static void ft_trig_pseudo_polarize(SW_FT_Vector* vec)
{
    SW_FT_Angle        theta;
    SW_FT_Int          i;
    SW_FT_Fixed        x, y, xtemp, b;
    const SW_FT_Fixed* arctanptr;

    x = vec->x;
    y = vec->y;

    /* Get the vector into [-PI/4,PI/4] sector */
    if (y > x) {
        if (y > -x) {
            theta = SW_FT_ANGLE_PI2;
            xtemp = y;
            y = -x;
            x = xtemp;
        } else {
            theta = y > 0 ? SW_FT_ANGLE_PI : -SW_FT_ANGLE_PI;
            x = -x;
            y = -y;
        }
    } else {
        if (y < -x) {
            theta = -SW_FT_ANGLE_PI2;
            xtemp = -y;
            y = x;
            x = xtemp;
        } else {
            theta = 0;
        }
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for (i = 1, b = 1; i < SW_FT_TRIG_MAX_ITERS; b <<= 1, i++) {
        SW_FT_Fixed v1 = ((y + b) >> i);
        SW_FT_Fixed v2 = ((x + b) >> i);
        if (y > 0) {
            xtemp = x + v1;
            y = y - v2;
            x = xtemp;
            theta += *arctanptr++;
        } else {
            xtemp = x - v1;
            y = y + v2;
            x = xtemp;
            theta -= *arctanptr++;
        }
    }

    /* round theta */
    if (theta >= 0)
        theta = SW_FT_PAD_ROUND(theta, 32);
    else
        theta = -SW_FT_PAD_ROUND(-theta, 32);

    vec->x = x;
    vec->y = theta;
}

/* documentation is in fttrigon.h */

SW_FT_Fixed SW_FT_Cos(SW_FT_Angle angle)
{
    SW_FT_Vector v;

    v.x = SW_FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate(&v, angle);

    return (v.x + 0x80L) >> 8;
}

/* documentation is in fttrigon.h */

SW_FT_Fixed SW_FT_Sin(SW_FT_Angle angle)
{
    return SW_FT_Cos(SW_FT_ANGLE_PI2 - angle);
}

/* documentation is in fttrigon.h */

SW_FT_Fixed SW_FT_Tan(SW_FT_Angle angle)
{
    SW_FT_Vector v;

    v.x = SW_FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate(&v, angle);

    return SW_FT_DivFix(v.y, v.x);
}

/* documentation is in fttrigon.h */

SW_FT_Angle SW_FT_Atan2(SW_FT_Fixed dx, SW_FT_Fixed dy)
{
    SW_FT_Vector v;

    if (dx == 0 && dy == 0) return 0;

    v.x = dx;
    v.y = dy;
    ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    return v.y;
}

/* documentation is in fttrigon.h */

void SW_FT_Vector_Unit(SW_FT_Vector* vec, SW_FT_Angle angle)
{
    vec->x = SW_FT_TRIG_SCALE >> 8;
    vec->y = 0;
    ft_trig_pseudo_rotate(vec, angle);
    vec->x = (vec->x + 0x80L) >> 8;
    vec->y = (vec->y + 0x80L) >> 8;
}

/* these macros return 0 for positive numbers,
   and -1 for negative ones */
#define SW_FT_SIGN_LONG(x) ((x) >> (SW_FT_SIZEOF_LONG * 8 - 1))
#define SW_FT_SIGN_INT(x) ((x) >> (SW_FT_SIZEOF_INT * 8 - 1))
#define SW_FT_SIGN_INT32(x) ((x) >> 31)
#define SW_FT_SIGN_INT16(x) ((x) >> 15)

/* documentation is in fttrigon.h */

void SW_FT_Vector_Rotate(SW_FT_Vector* vec, SW_FT_Angle angle)
{
    SW_FT_Int    shift;
    SW_FT_Vector v;

    v.x = vec->x;
    v.y = vec->y;

    if (angle && (v.x != 0 || v.y != 0)) {
        shift = ft_trig_prenorm(&v);
        ft_trig_pseudo_rotate(&v, angle);
        v.x = ft_trig_downscale(v.x);
        v.y = ft_trig_downscale(v.y);

        if (shift > 0) {
            SW_FT_Int32 half = (SW_FT_Int32)1L << (shift - 1);

            vec->x = (v.x + half + SW_FT_SIGN_LONG(v.x)) >> shift;
            vec->y = (v.y + half + SW_FT_SIGN_LONG(v.y)) >> shift;
        } else {
            shift = -shift;
            vec->x = (SW_FT_Pos)((SW_FT_ULong)v.x << shift);
            vec->y = (SW_FT_Pos)((SW_FT_ULong)v.y << shift);
        }
    }
}

/* documentation is in fttrigon.h */

SW_FT_Fixed SW_FT_Vector_Length(SW_FT_Vector* vec)
{
    SW_FT_Int    shift;
    SW_FT_Vector v;

    v = *vec;

    /* handle trivial cases */
    if (v.x == 0) {
        return SW_FT_ABS(v.y);
    } else if (v.y == 0) {
        return SW_FT_ABS(v.x);
    }

    /* general case */
    shift = ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    v.x = ft_trig_downscale(v.x);

    if (shift > 0) return (v.x + (1 << (shift - 1))) >> shift;

    return (SW_FT_Fixed)((SW_FT_UInt32)v.x << -shift);
}

/* documentation is in fttrigon.h */

void SW_FT_Vector_Polarize(SW_FT_Vector* vec, SW_FT_Fixed* length,
                           SW_FT_Angle* angle)
{
    SW_FT_Int    shift;
    SW_FT_Vector v;

    v = *vec;

    if (v.x == 0 && v.y == 0) return;

    shift = ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    v.x = ft_trig_downscale(v.x);

    *length = (shift >= 0) ? (v.x >> shift)
                           : (SW_FT_Fixed)((SW_FT_UInt32)v.x << -shift);
    *angle = v.y;
}

/* documentation is in fttrigon.h */

void SW_FT_Vector_From_Polar(SW_FT_Vector* vec, SW_FT_Fixed length,
                             SW_FT_Angle angle)
{
    vec->x = length;
    vec->y = 0;

    SW_FT_Vector_Rotate(vec, angle);
}

/* documentation is in fttrigon.h */

SW_FT_Angle SW_FT_Angle_Diff( SW_FT_Angle  angle1, SW_FT_Angle  angle2 )
{
  SW_FT_Angle  delta = angle2 - angle1;

  while ( delta <= -SW_FT_ANGLE_PI )
    delta += SW_FT_ANGLE_2PI;

  while ( delta > SW_FT_ANGLE_PI )
    delta -= SW_FT_ANGLE_2PI;

  return delta;
}

/* END */
