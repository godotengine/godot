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
/*  license, FTL.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include "plutovg-ft-math.h"

#if defined(_MSC_VER)
#include <intrin.h>
static inline int clz(unsigned int x) {
    unsigned long r = 0;
    if (_BitScanReverse(&r, x))
        return 31 - r;
    return 32;
}
#define PVG_FT_MSB(x)  (31 - clz(x))
#elif defined(__GNUC__)
#define PVG_FT_MSB(x)  (31 - __builtin_clz(x))
#else
static inline int clz(unsigned int x) {
    int n = 0;
    if (x == 0) return 32;
    if (x <= 0x0000FFFFU) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFU) { n +=  8; x <<=  8; }
    if (x <= 0x0FFFFFFFU) { n +=  4; x <<=  4; }
    if (x <= 0x3FFFFFFFU) { n +=  2; x <<=  2; }
    if (x <= 0x7FFFFFFFU) { n +=  1; }
    return n;
}
#define PVG_FT_MSB(x)  (31 - clz(x))
#endif

#define PVG_FT_PAD_FLOOR(x, n) ((x) & ~((n)-1))
#define PVG_FT_PAD_ROUND(x, n) PVG_FT_PAD_FLOOR((x) + ((n) / 2), n)
#define PVG_FT_PAD_CEIL(x, n) PVG_FT_PAD_FLOOR((x) + ((n)-1), n)

#define PVG_FT_BEGIN_STMNT do {
#define PVG_FT_END_STMNT } while (0)

/* transfer sign leaving a positive number */
#define PVG_FT_MOVE_SIGN(x, s) \
    PVG_FT_BEGIN_STMNT         \
    if (x < 0) {              \
        x = -x;               \
        s = -s;               \
    }                         \
    PVG_FT_END_STMNT

PVG_FT_Long PVG_FT_MulFix(PVG_FT_Long a, PVG_FT_Long b)
{
    PVG_FT_Int  s = 1;
    PVG_FT_Long c;

    PVG_FT_MOVE_SIGN(a, s);
    PVG_FT_MOVE_SIGN(b, s);

    c = (PVG_FT_Long)(((PVG_FT_Int64)a * b + 0x8000L) >> 16);

    return (s > 0) ? c : -c;
}

PVG_FT_Long PVG_FT_MulDiv(PVG_FT_Long a, PVG_FT_Long b, PVG_FT_Long c)
{
    PVG_FT_Int  s = 1;
    PVG_FT_Long d;

    PVG_FT_MOVE_SIGN(a, s);
    PVG_FT_MOVE_SIGN(b, s);
    PVG_FT_MOVE_SIGN(c, s);

    d = (PVG_FT_Long)(c > 0 ? ((PVG_FT_Int64)a * b + (c >> 1)) / c : 0x7FFFFFFFL);

    return (s > 0) ? d : -d;
}

PVG_FT_Long PVG_FT_DivFix(PVG_FT_Long a, PVG_FT_Long b)
{
    PVG_FT_Int  s = 1;
    PVG_FT_Long q;

    PVG_FT_MOVE_SIGN(a, s);
    PVG_FT_MOVE_SIGN(b, s);

    q = (PVG_FT_Long)(b > 0 ? (((PVG_FT_UInt64)a << 16) + (b >> 1)) / b
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
#define PVG_FT_TRIG_SCALE 0xDBD95B16UL

/* the highest bit in overflow-safe vector components, */
/* MSB of 0.858785336480436 * sqrt(0.5) * 2^30         */
#define PVG_FT_TRIG_SAFE_MSB 29

/* this table was generated for PVG_FT_PI = 180L << 16, i.e. degrees */
#define PVG_FT_TRIG_MAX_ITERS 23

static const PVG_FT_Fixed ft_trig_arctan_table[] = {
    1740967L, 919879L, 466945L, 234379L, 117304L, 58666L, 29335L, 14668L,
    7334L,    3667L,   1833L,   917L,    458L,    229L,   115L,   57L,
    29L,      14L,     7L,      4L,      2L,      1L};

/* multiply a given value by the CORDIC shrink factor */
static PVG_FT_Fixed ft_trig_downscale(PVG_FT_Fixed val)
{
    PVG_FT_Fixed s;
    PVG_FT_Int64 v;

    s = val;
    val = PVG_FT_ABS(val);

    v = (val * (PVG_FT_Int64)PVG_FT_TRIG_SCALE) + 0x100000000UL;
    val = (PVG_FT_Fixed)(v >> 32);

    return (s >= 0) ? val : -val;
}

/* undefined and never called for zero vector */
static PVG_FT_Int ft_trig_prenorm(PVG_FT_Vector* vec)
{
    PVG_FT_Pos x, y;
    PVG_FT_Int shift;

    x = vec->x;
    y = vec->y;

    shift = PVG_FT_MSB(PVG_FT_ABS(x) | PVG_FT_ABS(y));

    if (shift <= PVG_FT_TRIG_SAFE_MSB) {
        shift = PVG_FT_TRIG_SAFE_MSB - shift;
        vec->x = (PVG_FT_Pos)((PVG_FT_ULong)x << shift);
        vec->y = (PVG_FT_Pos)((PVG_FT_ULong)y << shift);
    } else {
        shift -= PVG_FT_TRIG_SAFE_MSB;
        vec->x = x >> shift;
        vec->y = y >> shift;
        shift = -shift;
    }

    return shift;
}

static void ft_trig_pseudo_rotate(PVG_FT_Vector* vec, PVG_FT_Angle theta)
{
    PVG_FT_Int          i;
    PVG_FT_Fixed        x, y, xtemp, b;
    const PVG_FT_Fixed* arctanptr;

    x = vec->x;
    y = vec->y;

    /* Rotate inside [-PI/4,PI/4] sector */
    while (theta < -PVG_FT_ANGLE_PI4) {
        xtemp = y;
        y = -x;
        x = xtemp;
        theta += PVG_FT_ANGLE_PI2;
    }

    while (theta > PVG_FT_ANGLE_PI4) {
        xtemp = -y;
        y = x;
        x = xtemp;
        theta -= PVG_FT_ANGLE_PI2;
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for (i = 1, b = 1; i < PVG_FT_TRIG_MAX_ITERS; b <<= 1, i++) {
        PVG_FT_Fixed v1 = ((y + b) >> i);
        PVG_FT_Fixed v2 = ((x + b) >> i);
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

static void ft_trig_pseudo_polarize(PVG_FT_Vector* vec)
{
    PVG_FT_Angle        theta;
    PVG_FT_Int          i;
    PVG_FT_Fixed        x, y, xtemp, b;
    const PVG_FT_Fixed* arctanptr;

    x = vec->x;
    y = vec->y;

    /* Get the vector into [-PI/4,PI/4] sector */
    if (y > x) {
        if (y > -x) {
            theta = PVG_FT_ANGLE_PI2;
            xtemp = y;
            y = -x;
            x = xtemp;
        } else {
            theta = y > 0 ? PVG_FT_ANGLE_PI : -PVG_FT_ANGLE_PI;
            x = -x;
            y = -y;
        }
    } else {
        if (y < -x) {
            theta = -PVG_FT_ANGLE_PI2;
            xtemp = -y;
            y = x;
            x = xtemp;
        } else {
            theta = 0;
        }
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for (i = 1, b = 1; i < PVG_FT_TRIG_MAX_ITERS; b <<= 1, i++) {
        PVG_FT_Fixed v1 = ((y + b) >> i);
        PVG_FT_Fixed v2 = ((x + b) >> i);
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
        theta = PVG_FT_PAD_ROUND(theta, 32);
    else
        theta = -PVG_FT_PAD_ROUND(-theta, 32);

    vec->x = x;
    vec->y = theta;
}

/* documentation is in fttrigon.h */

PVG_FT_Fixed PVG_FT_Cos(PVG_FT_Angle angle)
{
    PVG_FT_Vector v;

    v.x = PVG_FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate(&v, angle);

    return (v.x + 0x80L) >> 8;
}

/* documentation is in fttrigon.h */

PVG_FT_Fixed PVG_FT_Sin(PVG_FT_Angle angle)
{
    return PVG_FT_Cos(PVG_FT_ANGLE_PI2 - angle);
}

/* documentation is in fttrigon.h */

PVG_FT_Fixed PVG_FT_Tan(PVG_FT_Angle angle)
{
    PVG_FT_Vector v;

    v.x = PVG_FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate(&v, angle);

    return PVG_FT_DivFix(v.y, v.x);
}

/* documentation is in fttrigon.h */

PVG_FT_Angle PVG_FT_Atan2(PVG_FT_Fixed dx, PVG_FT_Fixed dy)
{
    PVG_FT_Vector v;

    if (dx == 0 && dy == 0) return 0;

    v.x = dx;
    v.y = dy;
    ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    return v.y;
}

/* documentation is in fttrigon.h */

void PVG_FT_Vector_Unit(PVG_FT_Vector* vec, PVG_FT_Angle angle)
{
    vec->x = PVG_FT_TRIG_SCALE >> 8;
    vec->y = 0;
    ft_trig_pseudo_rotate(vec, angle);
    vec->x = (vec->x + 0x80L) >> 8;
    vec->y = (vec->y + 0x80L) >> 8;
}

void PVG_FT_Vector_Rotate(PVG_FT_Vector* vec, PVG_FT_Angle angle)
{
    PVG_FT_Int     shift;
    PVG_FT_Vector  v = *vec;

    if ( v.x == 0 && v.y == 0 )
        return;

    shift = ft_trig_prenorm( &v );
    ft_trig_pseudo_rotate( &v, angle );
    v.x = ft_trig_downscale( v.x );
    v.y = ft_trig_downscale( v.y );

    if ( shift > 0 )
    {
        PVG_FT_Int32  half = (PVG_FT_Int32)1L << ( shift - 1 );


        vec->x = ( v.x + half - ( v.x < 0 ) ) >> shift;
        vec->y = ( v.y + half - ( v.y < 0 ) ) >> shift;
    }
    else
    {
        shift  = -shift;
        vec->x = (PVG_FT_Pos)( (PVG_FT_ULong)v.x << shift );
        vec->y = (PVG_FT_Pos)( (PVG_FT_ULong)v.y << shift );
    }
}

/* documentation is in fttrigon.h */

PVG_FT_Fixed PVG_FT_Vector_Length(PVG_FT_Vector* vec)
{
    PVG_FT_Int    shift;
    PVG_FT_Vector v;

    v = *vec;

    /* handle trivial cases */
    if (v.x == 0) {
        return PVG_FT_ABS(v.y);
    } else if (v.y == 0) {
        return PVG_FT_ABS(v.x);
    }

    /* general case */
    shift = ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    v.x = ft_trig_downscale(v.x);

    if (shift > 0) return (v.x + (1 << (shift - 1))) >> shift;

    return (PVG_FT_Fixed)((PVG_FT_UInt32)v.x << -shift);
}

/* documentation is in fttrigon.h */

void PVG_FT_Vector_Polarize(PVG_FT_Vector* vec, PVG_FT_Fixed* length,
    PVG_FT_Angle* angle)
{
    PVG_FT_Int    shift;
    PVG_FT_Vector v;

    v = *vec;

    if (v.x == 0 && v.y == 0) return;

    shift = ft_trig_prenorm(&v);
    ft_trig_pseudo_polarize(&v);

    v.x = ft_trig_downscale(v.x);

    *length = (shift >= 0) ? (v.x >> shift)
                           : (PVG_FT_Fixed)((PVG_FT_UInt32)v.x << -shift);
    *angle = v.y;
}

/* documentation is in fttrigon.h */

void PVG_FT_Vector_From_Polar(PVG_FT_Vector* vec, PVG_FT_Fixed length,
    PVG_FT_Angle angle)
{
    vec->x = length;
    vec->y = 0;

    PVG_FT_Vector_Rotate(vec, angle);
}

/* documentation is in fttrigon.h */

PVG_FT_Angle PVG_FT_Angle_Diff( PVG_FT_Angle  angle1, PVG_FT_Angle  angle2 )
{
    PVG_FT_Angle  delta = angle2 - angle1;

    while ( delta <= -PVG_FT_ANGLE_PI )
        delta += PVG_FT_ANGLE_2PI;

    while ( delta > PVG_FT_ANGLE_PI )
        delta -= PVG_FT_ANGLE_2PI;

    return delta;
}

/* END */
