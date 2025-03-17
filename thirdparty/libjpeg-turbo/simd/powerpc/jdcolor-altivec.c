/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2015, D. R. Commander.  All Rights Reserved.
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

/* YCC --> RGB CONVERSION */

#include "jsimd_altivec.h"


#define F_0_344  22554              /* FIX(0.34414) */
#define F_0_714  46802              /* FIX(0.71414) */
#define F_1_402  91881              /* FIX(1.40200) */
#define F_1_772  116130             /* FIX(1.77200) */
#define F_0_402  (F_1_402 - 65536)  /* FIX(1.40200) - FIX(1) */
#define F_0_285  (65536 - F_0_714)  /* FIX(1) - FIX(0.71414) */
#define F_0_228  (131072 - F_1_772) /* FIX(2) - FIX(1.77200) */

#define SCALEBITS  16
#define ONE_HALF  (1 << (SCALEBITS - 1))

#define RGB_INDEX0 \
  {  0,  1,  8,  2,  3, 10,  4,  5, 12,  6,  7, 14, 16, 17, 24, 18 }
#define RGB_INDEX1 \
  {  3, 10,  4,  5, 12,  6,  7, 14, 16, 17, 24, 18, 19, 26, 20, 21 }
#define RGB_INDEX2 \
  { 12,  6,  7, 14, 16, 17, 24, 18, 19, 26, 20, 21, 28, 22, 23, 30 }
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE

#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extrgb_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX0
#undef RGB_INDEX1
#undef RGB_INDEX2
#undef jsimd_ycc_rgb_convert_altivec

#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define RGB_INDEX \
  {  0,  1,  8,  9,  2,  3, 10, 11,  4,  5, 12, 13,  6,  7, 14, 15 }
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extrgbx_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX
#undef jsimd_ycc_rgb_convert_altivec

#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define RGB_INDEX0 \
  {  8,  1,  0, 10,  3,  2, 12,  5,  4, 14,  7,  6, 24, 17, 16, 26 }
#define RGB_INDEX1 \
  {  3,  2, 12,  5,  4, 14,  7,  6, 24, 17, 16, 26, 19, 18, 28, 21 }
#define RGB_INDEX2 \
  {  4, 14,  7,  6, 24, 17, 16, 26, 19, 18, 28, 21, 20, 30, 23, 22 }
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extbgr_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX0
#undef RGB_INDEX1
#undef RGB_INDEX2
#undef jsimd_ycc_rgb_convert_altivec

#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define RGB_INDEX \
  {  8,  1,  0,  9, 10,  3,  2, 11, 12,  5,  4, 13, 14,  7,  6, 15 }
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extbgrx_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX
#undef jsimd_ycc_rgb_convert_altivec

#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define RGB_INDEX \
  {  9,  8,  1,  0, 11, 10,  3,  2, 13, 12,  5,  4, 15, 14,  7,  6 }
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extxbgr_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX
#undef jsimd_ycc_rgb_convert_altivec

#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define RGB_INDEX \
  {  9,  0,  1,  8, 11,  2,  3, 10, 13,  4,  5, 12, 15,  6,  7, 14 }
#define jsimd_ycc_rgb_convert_altivec  jsimd_ycc_extxrgb_convert_altivec
#include "jdcolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGB_INDEX
#undef jsimd_ycc_rgb_convert_altivec
