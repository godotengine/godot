/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2024, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jsimd.h"
#include "../../src/jdct.h"
#include "../../src/jsimddct.h"
#include "../jsimd.h"
#include <altivec.h>


/* Common code */

#define __4X(a)      a, a, a, a
#define __4X2(a, b)  a, b, a, b, a, b, a, b
#define __8X(a)      __4X(a), __4X(a)
#define __16X(a)     __8X(a), __8X(a)

#define TRANSPOSE(row, col) { \
  __vector short row04l, row04h, row15l, row15h, \
                 row26l, row26h, row37l, row37h; \
  __vector short col01e, col01o, col23e, col23o, \
                 col45e, col45o, col67e, col67o; \
  \
                                       /* transpose coefficients (phase 1) */ \
  row04l = vec_mergeh(row##0, row##4); /* row04l=(00 40 01 41 02 42 03 43) */ \
  row04h = vec_mergel(row##0, row##4); /* row04h=(04 44 05 45 06 46 07 47) */ \
  row15l = vec_mergeh(row##1, row##5); /* row15l=(10 50 11 51 12 52 13 53) */ \
  row15h = vec_mergel(row##1, row##5); /* row15h=(14 54 15 55 16 56 17 57) */ \
  row26l = vec_mergeh(row##2, row##6); /* row26l=(20 60 21 61 22 62 23 63) */ \
  row26h = vec_mergel(row##2, row##6); /* row26h=(24 64 25 65 26 66 27 67) */ \
  row37l = vec_mergeh(row##3, row##7); /* row37l=(30 70 31 71 32 72 33 73) */ \
  row37h = vec_mergel(row##3, row##7); /* row37h=(34 74 35 75 36 76 37 77) */ \
  \
                                       /* transpose coefficients (phase 2) */ \
  col01e = vec_mergeh(row04l, row26l); /* col01e=(00 20 40 60 01 21 41 61) */ \
  col23e = vec_mergel(row04l, row26l); /* col23e=(02 22 42 62 03 23 43 63) */ \
  col45e = vec_mergeh(row04h, row26h); /* col45e=(04 24 44 64 05 25 45 65) */ \
  col67e = vec_mergel(row04h, row26h); /* col67e=(06 26 46 66 07 27 47 67) */ \
  col01o = vec_mergeh(row15l, row37l); /* col01o=(10 30 50 70 11 31 51 71) */ \
  col23o = vec_mergel(row15l, row37l); /* col23o=(12 32 52 72 13 33 53 73) */ \
  col45o = vec_mergeh(row15h, row37h); /* col45o=(14 34 54 74 15 35 55 75) */ \
  col67o = vec_mergel(row15h, row37h); /* col67o=(16 36 56 76 17 37 57 77) */ \
  \
                                       /* transpose coefficients (phase 3) */ \
  col##0 = vec_mergeh(col01e, col01o); /* col0=(00 10 20 30 40 50 60 70) */ \
  col##1 = vec_mergel(col01e, col01o); /* col1=(01 11 21 31 41 51 61 71) */ \
  col##2 = vec_mergeh(col23e, col23o); /* col2=(02 12 22 32 42 52 62 72) */ \
  col##3 = vec_mergel(col23e, col23o); /* col3=(03 13 23 33 43 53 63 73) */ \
  col##4 = vec_mergeh(col45e, col45o); /* col4=(04 14 24 34 44 54 64 74) */ \
  col##5 = vec_mergel(col45e, col45o); /* col5=(05 15 25 35 45 55 65 75) */ \
  col##6 = vec_mergeh(col67e, col67o); /* col6=(06 16 26 36 46 56 66 76) */ \
  col##7 = vec_mergel(col67e, col67o); /* col7=(07 17 27 37 47 57 67 77) */ \
}

#ifndef min
#define min(a, b)  ((a) < (b) ? (a) : (b))
#endif


/* Macros to abstract big/little endian bit twiddling */

#ifdef __BIG_ENDIAN__

#define VEC_LD(a, b)     vec_ld(a, b)
#define VEC_ST(a, b, c)  vec_st(a, b, c)
#define VEC_UNPACKHU(a)  vec_mergeh(pb_zero, a)
#define VEC_UNPACKLU(a)  vec_mergel(pb_zero, a)

#else

#define VEC_LD(a, b)     vec_vsx_ld(a, b)
#define VEC_ST(a, b, c)  vec_vsx_st(a, b, c)
#define VEC_UNPACKHU(a)  vec_mergeh(a, pb_zero)
#define VEC_UNPACKLU(a)  vec_mergel(a, pb_zero)

#endif


#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
