/*
 * jccolor-neon.c - colorspace conversion (Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, 2024, D. R. Commander.  All Rights Reserved.
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
#include "align.h"
#include "neon-compat.h"

#include <arm_neon.h>


/* RGB -> YCbCr conversion constants */

#define F_0_298  19595
#define F_0_587  38470
#define F_0_113  7471
#define F_0_168  11059
#define F_0_331  21709
#define F_0_500  32768
#define F_0_418  27439
#define F_0_081  5329

ALIGN(16) static const uint16_t jsimd_rgb_ycc_neon_consts[] = {
  F_0_298, F_0_587, F_0_113, F_0_168,
  F_0_331, F_0_500, F_0_418, F_0_081
};


/* Include inline routines for colorspace extensions. */

#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extrgb_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extrgbx_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extbgr_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extbgrx_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extxbgr_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_neon  jsimd_extxrgb_ycc_convert_neon
#if defined(__aarch64__) || defined(_M_ARM64)
#include "aarch64/jccolext-neon.c"
#else
#include "aarch32/jccolext-neon.c"
#endif
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_neon
