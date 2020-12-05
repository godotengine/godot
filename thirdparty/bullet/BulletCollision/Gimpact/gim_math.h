#ifndef GIM_MATH_H_INCLUDED
#define GIM_MATH_H_INCLUDED
/*! \file gim_math.h
\author Francisco Leon Najera
*/
/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/

#include "LinearMath/btScalar.h"

#define GREAL btScalar
#define GREAL2 double
#define GINT int
#define GUINT unsigned int
#define GSHORT short
#define GUSHORT unsigned short
#define GINT64 long long
#define GUINT64 unsigned long long

#define G_PI 3.14159265358979f
#define G_HALF_PI 1.5707963f
//267948966
#define G_TWO_PI 6.28318530f
//71795864
#define G_ROOT3 1.73205f
#define G_ROOT2 1.41421f
#define G_UINT_INFINITY 0xffffffff  //!< A very very high value
#define G_REAL_INFINITY FLT_MAX
#define G_SIGN_BITMASK 0x80000000
#define G_EPSILON SIMD_EPSILON

enum GIM_SCALAR_TYPES
{
	G_STYPE_REAL = 0,
	G_STYPE_REAL2,
	G_STYPE_SHORT,
	G_STYPE_USHORT,
	G_STYPE_INT,
	G_STYPE_UINT,
	G_STYPE_INT64,
	G_STYPE_UINT64
};

#define G_DEGTORAD(X) ((X)*3.1415926f / 180.0f)
#define G_RADTODEG(X) ((X)*180.0f / 3.1415926f)

//! Integer representation of a floating-point value.
#define GIM_IR(x) ((GUINT&)(x))

//! Signed integer representation of a floating-point value.
#define GIM_SIR(x) ((GINT&)(x))

//! Absolute integer representation of a floating-point value
#define GIM_AIR(x) (GIM_IR(x) & 0x7fffffff)

//! Floating-point representation of an integer value.
#define GIM_FR(x) ((GREAL&)(x))

#define GIM_MAX(a, b) (a < b ? b : a)
#define GIM_MIN(a, b) (a > b ? b : a)

#define GIM_MAX3(a, b, c) GIM_MAX(a, GIM_MAX(b, c))
#define GIM_MIN3(a, b, c) GIM_MIN(a, GIM_MIN(b, c))

#define GIM_IS_ZERO(value) (value < G_EPSILON && value > -G_EPSILON)

#define GIM_IS_NEGATIVE(value) (value <= -G_EPSILON)

#define GIM_IS_POSISITVE(value) (value >= G_EPSILON)

#define GIM_NEAR_EQUAL(v1, v2) GIM_IS_ZERO((v1 - v2))

///returns a clamped number
#define GIM_CLAMP(number, minval, maxval) (number < minval ? minval : (number > maxval ? maxval : number))

#define GIM_GREATER(x, y) btFabs(x) > (y)

///Swap numbers
#define GIM_SWAP_NUMBERS(a, b) \
	{                          \
		a = a + b;             \
		b = a - b;             \
		a = a - b;             \
	}

#define GIM_INV_SQRT(va, isva)                         \
	{                                                  \
		if (va <= 0.0000001f)                          \
		{                                              \
			isva = G_REAL_INFINITY;                    \
		}                                              \
		else                                           \
		{                                              \
			GREAL _x = va * 0.5f;                      \
			GUINT _y = 0x5f3759df - (GIM_IR(va) >> 1); \
			isva = GIM_FR(_y);                         \
			isva = isva * (1.5f - (_x * isva * isva)); \
		}                                              \
	}

#define GIM_SQRT(va, sva)      \
	{                          \
		GIM_INV_SQRT(va, sva); \
		sva = 1.0f / sva;      \
	}

//! Computes 1.0f / sqrtf(x). Comes from Quake3. See http://www.magic-software.com/3DGEDInvSqrt.html
inline GREAL gim_inv_sqrt(GREAL f)
{
	GREAL r;
	GIM_INV_SQRT(f, r);
	return r;
}

inline GREAL gim_sqrt(GREAL f)
{
	GREAL r;
	GIM_SQRT(f, r);
	return r;
}

#endif  // GIM_MATH_H_INCLUDED
