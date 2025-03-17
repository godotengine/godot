/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2011, 2014, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2016-2017, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Authors:  ZhuChen     <zhuchen@loongson.cn>
 *           CaiWanwei   <caiwanwei@loongson.cn>
 *           SunZhangzhi <sunzhangzhi-cq@loongson.cn>
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

#include "jsimd_mmi.h"


#define F_0_081  ((short)5329)                /* FIX(0.08131) */
#define F_0_114  ((short)7471)                /* FIX(0.11400) */
#define F_0_168  ((short)11059)               /* FIX(0.16874) */
#define F_0_250  ((short)16384)               /* FIX(0.25000) */
#define F_0_299  ((short)19595)               /* FIX(0.29900) */
#define F_0_331  ((short)21709)               /* FIX(0.33126) */
#define F_0_418  ((short)27439)               /* FIX(0.41869) */
#define F_0_587  ((short)38470)               /* FIX(0.58700) */
#define F_0_337  ((short)(F_0_587 - F_0_250)) /* FIX(0.58700) - FIX(0.25000) */

enum const_index {
  index_PD_ONEHALF,
  index_PW_F0299_F0337,
  index_PW_F0114_F0250,
  index_PW_MF016_MF033,
  index_PW_MF008_MF041,
  index_PD_ONEHALFM1_CJ
};

static uint64_t const_value[] = {
  _uint64_set_pi32((int)(1 << (SCALEBITS - 1)), (int)(1 << (SCALEBITS - 1))),
  _uint64_set_pi16(F_0_337, F_0_299, F_0_337, F_0_299),
  _uint64_set_pi16(F_0_250, F_0_114, F_0_250, F_0_114),
  _uint64_set_pi16(-F_0_331, -F_0_168, -F_0_331, -F_0_168),
  _uint64_set_pi16(-F_0_418, -F_0_081, -F_0_418, -F_0_081),
  _uint64_set_pi32(((1 << (SCALEBITS - 1)) - 1 + (CENTERJSAMPLE << SCALEBITS)),
                   ((1 << (SCALEBITS - 1)) - 1 + (CENTERJSAMPLE << SCALEBITS)))
};

#define get_const_value(index)  (*(__m64 *)&const_value[index])

#define PD_ONEHALF       get_const_value(index_PD_ONEHALF)
#define PW_F0299_F0337   get_const_value(index_PW_F0299_F0337)
#define PW_F0114_F0250   get_const_value(index_PW_F0114_F0250)
#define PW_MF016_MF033   get_const_value(index_PW_MF016_MF033)
#define PW_MF008_MF041   get_const_value(index_PW_MF008_MF041)
#define PD_ONEHALFM1_CJ  get_const_value(index_PD_ONEHALFM1_CJ)


#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extrgb_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extrgbx_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extbgr_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extbgrx_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extxbgr_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_mmi  jsimd_extxrgb_ycc_convert_mmi
#include "jccolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_mmi
