/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdint.h>
#include "./vp8_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"
#include "vp8/encoder/block.h"

#define BOOST_QUANT1(_in0, _in1, _in2, _ui)               \
  {                                                       \
    if (boost_temp[0] <= __lsx_vpickve2gr_h(_in0, _ui)) { \
      if (__lsx_vpickve2gr_h(_in1, _ui)) {                \
        eob = _ui;                                        \
        boost_temp = zbin_boost;                          \
      } else {                                            \
        boost_temp++;                                     \
      }                                                   \
    } else {                                              \
      _in2 = __lsx_vinsgr2vr_h(_in2, 0, _ui);             \
      boost_temp++;                                       \
    }                                                     \
  }

#define BOOST_QUANT2(_in0, _in1, _in2, _ui)               \
  {                                                       \
    if (boost_temp[0] <= __lsx_vpickve2gr_h(_in0, _ui)) { \
      if (__lsx_vpickve2gr_h(_in1, _ui)) {                \
        eob = _ui + 8;                                    \
        boost_temp = zbin_boost;                          \
      } else {                                            \
        boost_temp++;                                     \
      }                                                   \
    } else {                                              \
      _in2 = __lsx_vinsgr2vr_h(_in2, 0, _ui);             \
      boost_temp++;                                       \
    }                                                     \
  }

static int8_t exact_regular_quantize_b_lsx(
    int16_t *zbin_boost, int16_t *coeff_ptr, int16_t *zbin, int16_t *round,
    int16_t *quant, int16_t *quant_shift, int16_t *de_quant, int16_t zbin_oq_in,
    int16_t *q_coeff, int16_t *dq_coeff) {
  int32_t eob;
  int16_t *boost_temp = zbin_boost;
  __m128i inv_zig_zag = { 0x0C07040206050100, 0x0F0E0A090D0B0803 };
  __m128i sign_z0, sign_z1, q_coeff0, q_coeff1;
  __m128i z_bin0, z_bin1, zbin_o_q, x0, x1, sign_x0, sign_x1, de_quant0,
      de_quant1;
  __m128i z0, z1, round0, round1, quant0, quant2;
  __m128i inv_zig_zag0, inv_zig_zag1;
  __m128i zigzag_mask0 = { 0x0008000400010000, 0x0006000300020005 };
  __m128i zigzag_mask1 = { 0x000A000D000C0009, 0X000F000E000B0007 };
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i zero = __lsx_vldi(0);

  zbin_o_q = __lsx_vreplgr2vr_h(zbin_oq_in);
  inv_zig_zag0 = __lsx_vilvl_b(zero, inv_zig_zag);
  inv_zig_zag1 = __lsx_vilvh_b(zero, inv_zig_zag);
  eob = -1;
  DUP4_ARG2(__lsx_vld, coeff_ptr, 0, coeff_ptr, 16, round, 0, round, 16, tmp0,
            tmp1, tmp2, tmp3);
  DUP4_ARG3(__lsx_vshuf_h, zigzag_mask0, tmp1, tmp0, zigzag_mask1, tmp1, tmp0,
            zigzag_mask0, tmp3, tmp2, zigzag_mask1, tmp3, tmp2, z0, z1, round0,
            round1);
  DUP4_ARG2(__lsx_vld, quant, 0, quant, 16, zbin, 0, zbin, 16, tmp0, tmp1, tmp2,
            tmp3);
  DUP4_ARG3(__lsx_vshuf_h, zigzag_mask0, tmp1, tmp0, zigzag_mask1, tmp1, tmp0,
            zigzag_mask0, tmp3, tmp2, zigzag_mask1, tmp3, tmp2, quant0, quant2,
            z_bin0, z_bin1);
  DUP2_ARG2(__lsx_vsrai_h, z0, 15, z1, 15, sign_z0, sign_z1);
  DUP2_ARG2(__lsx_vadda_h, z0, zero, z1, zero, x0, x1);
  DUP2_ARG2(__lsx_vsub_h, x0, z_bin0, x1, z_bin1, z_bin0, z_bin1);
  DUP2_ARG2(__lsx_vsub_h, z_bin0, zbin_o_q, z_bin1, zbin_o_q, z_bin0, z_bin1);
  DUP2_ARG2(__lsx_vmulwev_w_h, quant0, round0, quant2, round1, tmp0, tmp2);
  DUP2_ARG2(__lsx_vmulwod_w_h, quant0, round0, quant2, round1, tmp1, tmp3);
  DUP2_ARG3(__lsx_vmaddwev_w_h, tmp0, quant0, x0, tmp2, quant2, x1, tmp0, tmp2);
  DUP2_ARG3(__lsx_vmaddwod_w_h, tmp1, quant0, x0, tmp3, quant2, x1, tmp1, tmp3);
  DUP2_ARG2(__lsx_vpackod_h, tmp1, tmp0, tmp3, tmp2, q_coeff0, q_coeff1);

  DUP2_ARG2(__lsx_vld, quant_shift, 0, quant_shift, 16, tmp1, tmp3);
  DUP2_ARG3(__lsx_vshuf_h, zigzag_mask0, tmp3, tmp1, zigzag_mask1, tmp3, tmp1,
            quant0, quant2);
  DUP2_ARG2(__lsx_vadd_h, x0, round0, x1, round1, x0, x1);
  DUP2_ARG2(__lsx_vmulwev_w_h, quant0, q_coeff0, quant2, q_coeff1, tmp0, tmp2);
  DUP2_ARG2(__lsx_vmulwod_w_h, quant0, q_coeff0, quant2, q_coeff1, tmp1, tmp3);
  DUP2_ARG3(__lsx_vmaddwev_w_h, tmp0, quant0, x0, tmp2, quant2, x1, tmp0, tmp2);
  DUP2_ARG3(__lsx_vmaddwod_w_h, tmp1, quant0, x0, tmp3, quant2, x1, tmp1, tmp3);
  DUP2_ARG2(__lsx_vpackod_h, tmp1, tmp0, tmp3, tmp2, x0, x1);
  DUP2_ARG2(__lsx_vxor_v, x0, sign_z0, x1, sign_z1, sign_x0, sign_x1);
  DUP2_ARG2(__lsx_vsub_h, sign_x0, sign_z0, sign_x1, sign_z1, sign_x0, sign_x1);

  BOOST_QUANT1(z_bin0, x0, sign_x0, 0);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 1);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 2);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 3);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 4);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 5);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 6);
  BOOST_QUANT1(z_bin0, x0, sign_x0, 7);

  BOOST_QUANT2(z_bin1, x1, sign_x1, 0);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 1);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 2);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 3);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 4);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 5);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 6);
  BOOST_QUANT2(z_bin1, x1, sign_x1, 7);

  DUP2_ARG2(__lsx_vld, de_quant, 0, de_quant, 16, de_quant0, de_quant1);
  DUP2_ARG3(__lsx_vshuf_h, inv_zig_zag0, sign_x1, sign_x0, inv_zig_zag1,
            sign_x1, sign_x0, q_coeff0, q_coeff1);
  DUP2_ARG2(__lsx_vmul_h, de_quant0, q_coeff0, de_quant1, q_coeff1, de_quant0,
            de_quant1);
  __lsx_vst(q_coeff0, q_coeff, 0);
  __lsx_vst(q_coeff1, q_coeff, 16);
  __lsx_vst(de_quant0, dq_coeff, 0);
  __lsx_vst(de_quant1, dq_coeff, 16);

  return (int8_t)(eob + 1);
}

void vp8_regular_quantize_b_lsx(BLOCK *b, BLOCKD *d) {
  int16_t *zbin_boost_ptr = b->zrun_zbin_boost;
  int16_t *coeff_ptr = b->coeff;
  int16_t *zbin_ptr = b->zbin;
  int16_t *round_ptr = b->round;
  int16_t *quant_ptr = b->quant;
  int16_t *quant_shift_ptr = b->quant_shift;
  int16_t *qcoeff_ptr = d->qcoeff;
  int16_t *dqcoeff_ptr = d->dqcoeff;
  int16_t *dequant_ptr = d->dequant;
  int16_t zbin_oq_value = b->zbin_extra;

  *d->eob = exact_regular_quantize_b_lsx(
      zbin_boost_ptr, coeff_ptr, zbin_ptr, round_ptr, quant_ptr,
      quant_shift_ptr, dequant_ptr, zbin_oq_value, qcoeff_ptr, dqcoeff_ptr);
}
