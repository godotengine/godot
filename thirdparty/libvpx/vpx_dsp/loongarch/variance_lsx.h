/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_

#include "vpx_util/loongson_intrinsics.h"

#define HADD_SW_S32(in0, in1)                  \
  do {                                         \
    __m128i res0_m;                            \
                                               \
    res0_m = __lsx_vhaddw_d_w(in0, in0);       \
    res0_m = __lsx_vhaddw_q_d(res0_m, res0_m); \
    in1 = __lsx_vpickve2gr_w(res0_m, 0);       \
  } while (0)

#define HORIZ_2TAP_FILT_UH(in0, in1, mask, coeff, shift, in2) \
  do {                                                        \
    __m128i tmp0_m, tmp1_m;                                   \
                                                              \
    tmp0_m = __lsx_vshuf_b(in1, in0, mask);                   \
    tmp1_m = __lsx_vdp2_h_bu(tmp0_m, coeff);                  \
    in2 = __lsx_vsrari_h(tmp1_m, shift);                      \
  } while (0)

#define CALC_MSE_B(src, ref, var)                                         \
  do {                                                                    \
    __m128i src_l0_m, src_l1_m;                                           \
    __m128i res_l0_m, res_l1_m;                                           \
                                                                          \
    src_l0_m = __lsx_vilvl_b(src, ref);                                   \
    src_l1_m = __lsx_vilvh_b(src, ref);                                   \
    DUP2_ARG2(__lsx_vhsubw_hu_bu, src_l0_m, src_l0_m, src_l1_m, src_l1_m, \
              res_l0_m, res_l1_m);                                        \
    var = __lsx_vdp2add_w_h(var, res_l0_m, res_l0_m);                     \
    var = __lsx_vdp2add_w_h(var, res_l1_m, res_l1_m);                     \
  } while (0)

#define CALC_MSE_AVG_B(src, ref, var, sub)                                \
  do {                                                                    \
    __m128i src_l0_m, src_l1_m;                                           \
    __m128i res_l0_m, res_l1_m;                                           \
                                                                          \
    src_l0_m = __lsx_vilvl_b(src, ref);                                   \
    src_l1_m = __lsx_vilvh_b(src, ref);                                   \
    DUP2_ARG2(__lsx_vhsubw_hu_bu, src_l0_m, src_l0_m, src_l1_m, src_l1_m, \
              res_l0_m, res_l1_m);                                        \
    var = __lsx_vdp2add_w_h(var, res_l0_m, res_l0_m);                     \
    var = __lsx_vdp2add_w_h(var, res_l1_m, res_l1_m);                     \
    sub = __lsx_vadd_h(sub, res_l0_m);                                    \
    sub = __lsx_vadd_h(sub, res_l1_m);                                    \
  } while (0)

#endif  // VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_
