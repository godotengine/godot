/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014, D. R. Commander.  All Rights Reserved.
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

/* RGB --> YCC CONVERSION */

#include "jsimd_altivec.h"


#define F_0_081  5329                 /* FIX(0.08131) */
#define F_0_114  7471                 /* FIX(0.11400) */
#define F_0_168  11059                /* FIX(0.16874) */
#define F_0_250  16384                /* FIX(0.25000) */
#define F_0_299  19595                /* FIX(0.29900) */
#define F_0_331  21709                /* FIX(0.33126) */
#define F_0_418  27439                /* FIX(0.41869) */
#define F_0_500  32768                /* FIX(0.50000) */
#define F_0_587  38470                /* FIX(0.58700) */
#define F_0_337  (F_0_587 - F_0_250)  /* FIX(0.58700) - FIX(0.25000) */

#define SCALEBITS  16
#define ONE_HALF  (1 << (SCALEBITS - 1))


#define RGBG_INDEX0 \
  {  0,  1,  3,  4,  6,  7,  9, 10,  2,  1,  5,  4,  8,  7, 11, 10 }
#define RGBG_INDEX1 \
  { 12, 13, 15, 16, 18, 19, 21, 22, 14, 13, 17, 16, 20, 19, 23, 22 }
#define RGBG_INDEX2 \
  {  8,  9, 11, 12, 14, 15, 17, 18, 10,  9, 13, 12, 16, 15, 19, 18 }
#define RGBG_INDEX3 \
  {  4,  5,  7,  8, 10, 11, 13, 14,  6,  5,  9,  8, 12, 11, 15, 14 }
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE

#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_altivec  jsimd_extrgb_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX0
#undef RGBG_INDEX1
#undef RGBG_INDEX2
#undef RGBG_INDEX3
#undef jsimd_rgb_ycc_convert_altivec

#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define RGBG_INDEX \
  {  0,  1,  4,  5,  8,  9, 12, 13,  2,  1,  6,  5, 10,  9, 14, 13 }
#define jsimd_rgb_ycc_convert_altivec  jsimd_extrgbx_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX
#undef jsimd_rgb_ycc_convert_altivec

#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define RGBG_INDEX0 \
  {  2,  1,  5,  4,  8,  7, 11, 10,  0,  1,  3,  4,  6,  7,  9, 10 }
#define RGBG_INDEX1 \
  { 14, 13, 17, 16, 20, 19, 23, 22, 12, 13, 15, 16, 18, 19, 21, 22 }
#define RGBG_INDEX2 \
  { 10,  9, 13, 12, 16, 15, 19, 18,  8,  9, 11, 12, 14, 15, 17, 18 }
#define RGBG_INDEX3 \
  {  6,  5,  9,  8, 12, 11, 15, 14,  4,  5,  7,  8, 10, 11, 13, 14 }
#define jsimd_rgb_ycc_convert_altivec  jsimd_extbgr_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX0
#undef RGBG_INDEX1
#undef RGBG_INDEX2
#undef RGBG_INDEX3
#undef jsimd_rgb_ycc_convert_altivec

#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define RGBG_INDEX \
  {  2,  1,  6,  5, 10,  9, 14, 13,  0,  1,  4,  5,  8,  9, 12, 13 }
#define jsimd_rgb_ycc_convert_altivec  jsimd_extbgrx_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX
#undef jsimd_rgb_ycc_convert_altivec

#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define RGBG_INDEX \
  {  3,  2,  7,  6, 11, 10, 15, 14,  1,  2,  5,  6,  9, 10, 13, 14 }
#define jsimd_rgb_ycc_convert_altivec  jsimd_extxbgr_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX
#undef jsimd_rgb_ycc_convert_altivec

#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define RGBG_INDEX \
  {  1,  2,  5,  6,  9, 10, 13, 14,  3,  2,  7,  6, 11, 10, 15, 14 }
#define jsimd_rgb_ycc_convert_altivec  jsimd_extxrgb_ycc_convert_altivec
#include "jccolext-altivec.c"
#undef RGB_PIXELSIZE
#undef RGBG_INDEX
#undef jsimd_rgb_ycc_convert_altivec
