/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2011, 2015, D. R. Commander.  All Rights Reserved.
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

/* YCC --> RGB CONVERSION */

#include "jsimd_mmi.h"


#define F_0_344  ((short)22554)  /* FIX(0.34414) */
#define F_0_402  ((short)26345)  /* FIX(1.40200) - FIX(1) */
#define F_0_285  ((short)18734)  /* FIX(1) - FIX(0.71414) */
#define F_0_228  ((short)14942)  /* FIX(2) - FIX(1.77200) */

enum const_index {
  index_PW_ONE,
  index_PW_F0402,
  index_PW_MF0228,
  index_PW_MF0344_F0285,
  index_PD_ONEHALF
};

static uint64_t const_value[] = {
  _uint64_set_pi16(1, 1, 1, 1),
  _uint64_set_pi16(F_0_402, F_0_402, F_0_402, F_0_402),
  _uint64_set_pi16(-F_0_228, -F_0_228, -F_0_228, -F_0_228),
  _uint64_set_pi16(F_0_285, -F_0_344, F_0_285, -F_0_344),
  _uint64_set_pi32((int)(1 << (SCALEBITS - 1)), (int)(1 << (SCALEBITS - 1)))
};

#define PW_ONE           get_const_value(index_PW_ONE)
#define PW_F0402         get_const_value(index_PW_F0402)
#define PW_MF0228        get_const_value(index_PW_MF0228)
#define PW_MF0344_F0285  get_const_value(index_PW_MF0344_F0285)
#define PD_ONEHALF       get_const_value(index_PD_ONEHALF)

#define RGBX_FILLER_0XFF  1


#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extrgb_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extrgbx_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extbgr_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extbgrx_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extxbgr_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define jsimd_ycc_rgb_convert_mmi  jsimd_ycc_extxrgb_convert_mmi
#include "jdcolext-mmi.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_ycc_rgb_convert_mmi
