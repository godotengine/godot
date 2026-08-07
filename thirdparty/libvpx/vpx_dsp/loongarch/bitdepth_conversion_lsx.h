/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_util/loongson_intrinsics.h"

static INLINE __m128i load_tran_low(const tran_low_t *s) {
#if CONFIG_VP9_HIGHBITDEPTH
  __m128i v0_m = __lsx_vld(s, 0);
  __m128i v1_m = __lsx_vld(s + 4, 0);
  return __lsx_vsrlni_h_w(v0_m, v1_m, 0);
#else
  return __lsx_vld(s, 0);
#endif
}

static INLINE void store_tran_low(__m128i v, tran_low_t *s, int32_t c) {
#if CONFIG_VP9_HIGHBITDEPTH
  __m128i v0_m, v1_m;
  v1_m = __lsx_vexth_w_h(v);
  v0_m = __lsx_vsllwil_w_h(v, 0);
  __lsx_vst(v0_m, s + c, 0);
  __lsx_vst(v1_m, s + c + 4, 0);
#else
  __lsx_vst(v, s + c, 0);
#endif
}

#endif  // VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_
