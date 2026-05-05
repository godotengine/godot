/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

static void temporal_filter_apply_16size_msa(
    uint8_t *frame1_ptr, uint32_t stride, uint8_t *frame2_ptr,
    int32_t strength_in, int32_t filter_wt_in, uint32_t *acc, uint16_t *cnt) {
  uint32_t row;
  v16i8 frame1_0_b, frame1_1_b, frame2_0_b, frame2_1_b;
  v16u8 frame_l, frame_h;
  v16i8 zero = { 0 };
  v8i16 frame2_0_h, frame2_1_h, mod0_h, mod1_h;
  v8i16 diff0, diff1, cnt0, cnt1;
  v4i32 const3, const16, filter_wt, strength;
  v4i32 mod0_w, mod1_w, mod2_w, mod3_w;
  v4i32 diff0_r, diff0_l, diff1_r, diff1_l;
  v4i32 frame2_0, frame2_1, frame2_2, frame2_3;
  v4i32 acc0, acc1, acc2, acc3;

  filter_wt = __msa_fill_w(filter_wt_in);
  strength = __msa_fill_w(strength_in);
  const3 = __msa_ldi_w(3);
  const16 = __msa_ldi_w(16);

  for (row = 8; row--;) {
    frame1_0_b = LD_SB(frame1_ptr);
    frame2_0_b = LD_SB(frame2_ptr);
    frame1_ptr += stride;
    frame2_ptr += 16;
    frame1_1_b = LD_SB(frame1_ptr);
    frame2_1_b = LD_SB(frame2_ptr);
    LD_SW2(acc, 4, acc0, acc1);
    LD_SW2(acc + 8, 4, acc2, acc3);
    LD_SH2(cnt, 8, cnt0, cnt1);
    ILVRL_B2_UB(frame1_0_b, frame2_0_b, frame_l, frame_h);
    HSUB_UB2_SH(frame_l, frame_h, diff0, diff1);
    UNPCK_SH_SW(diff0, diff0_r, diff0_l);
    UNPCK_SH_SW(diff1, diff1_r, diff1_l);
    MUL4(diff0_r, diff0_r, diff0_l, diff0_l, diff1_r, diff1_r, diff1_l, diff1_l,
         mod0_w, mod1_w, mod2_w, mod3_w);
    MUL4(mod0_w, const3, mod1_w, const3, mod2_w, const3, mod3_w, const3, mod0_w,
         mod1_w, mod2_w, mod3_w);
    SRAR_W4_SW(mod0_w, mod1_w, mod2_w, mod3_w, strength);
    diff0_r = (mod0_w < const16);
    diff0_l = (mod1_w < const16);
    diff1_r = (mod2_w < const16);
    diff1_l = (mod3_w < const16);
    SUB4(const16, mod0_w, const16, mod1_w, const16, mod2_w, const16, mod3_w,
         mod0_w, mod1_w, mod2_w, mod3_w);
    mod0_w = diff0_r & mod0_w;
    mod1_w = diff0_l & mod1_w;
    mod2_w = diff1_r & mod2_w;
    mod3_w = diff1_l & mod3_w;
    MUL4(mod0_w, filter_wt, mod1_w, filter_wt, mod2_w, filter_wt, mod3_w,
         filter_wt, mod0_w, mod1_w, mod2_w, mod3_w);
    PCKEV_H2_SH(mod1_w, mod0_w, mod3_w, mod2_w, mod0_h, mod1_h)
    ADD2(mod0_h, cnt0, mod1_h, cnt1, mod0_h, mod1_h);
    ST_SH2(mod0_h, mod1_h, cnt, 8);
    cnt += 16;
    ILVRL_B2_SH(zero, frame2_0_b, frame2_0_h, frame2_1_h);
    UNPCK_SH_SW(frame2_0_h, frame2_0, frame2_1);
    UNPCK_SH_SW(frame2_1_h, frame2_2, frame2_3);
    MUL4(mod0_w, frame2_0, mod1_w, frame2_1, mod2_w, frame2_2, mod3_w, frame2_3,
         mod0_w, mod1_w, mod2_w, mod3_w);
    ADD4(mod0_w, acc0, mod1_w, acc1, mod2_w, acc2, mod3_w, acc3, mod0_w, mod1_w,
         mod2_w, mod3_w);
    ST_SW2(mod0_w, mod1_w, acc, 4);
    ST_SW2(mod2_w, mod3_w, acc + 8, 4);
    acc += 16;
    LD_SW2(acc, 4, acc0, acc1);
    LD_SW2(acc + 8, 4, acc2, acc3);
    LD_SH2(cnt, 8, cnt0, cnt1);
    ILVRL_B2_UB(frame1_1_b, frame2_1_b, frame_l, frame_h);
    HSUB_UB2_SH(frame_l, frame_h, diff0, diff1);
    UNPCK_SH_SW(diff0, diff0_r, diff0_l);
    UNPCK_SH_SW(diff1, diff1_r, diff1_l);
    MUL4(diff0_r, diff0_r, diff0_l, diff0_l, diff1_r, diff1_r, diff1_l, diff1_l,
         mod0_w, mod1_w, mod2_w, mod3_w);
    MUL4(mod0_w, const3, mod1_w, const3, mod2_w, const3, mod3_w, const3, mod0_w,
         mod1_w, mod2_w, mod3_w);
    SRAR_W4_SW(mod0_w, mod1_w, mod2_w, mod3_w, strength);
    diff0_r = (mod0_w < const16);
    diff0_l = (mod1_w < const16);
    diff1_r = (mod2_w < const16);
    diff1_l = (mod3_w < const16);
    SUB4(const16, mod0_w, const16, mod1_w, const16, mod2_w, const16, mod3_w,
         mod0_w, mod1_w, mod2_w, mod3_w);
    mod0_w = diff0_r & mod0_w;
    mod1_w = diff0_l & mod1_w;
    mod2_w = diff1_r & mod2_w;
    mod3_w = diff1_l & mod3_w;
    MUL4(mod0_w, filter_wt, mod1_w, filter_wt, mod2_w, filter_wt, mod3_w,
         filter_wt, mod0_w, mod1_w, mod2_w, mod3_w);
    PCKEV_H2_SH(mod1_w, mod0_w, mod3_w, mod2_w, mod0_h, mod1_h);
    ADD2(mod0_h, cnt0, mod1_h, cnt1, mod0_h, mod1_h);
    ST_SH2(mod0_h, mod1_h, cnt, 8);
    cnt += 16;

    UNPCK_UB_SH(frame2_1_b, frame2_0_h, frame2_1_h);
    UNPCK_SH_SW(frame2_0_h, frame2_0, frame2_1);
    UNPCK_SH_SW(frame2_1_h, frame2_2, frame2_3);
    MUL4(mod0_w, frame2_0, mod1_w, frame2_1, mod2_w, frame2_2, mod3_w, frame2_3,
         mod0_w, mod1_w, mod2_w, mod3_w);
    ADD4(mod0_w, acc0, mod1_w, acc1, mod2_w, acc2, mod3_w, acc3, mod0_w, mod1_w,
         mod2_w, mod3_w);
    ST_SW2(mod0_w, mod1_w, acc, 4);
    ST_SW2(mod2_w, mod3_w, acc + 8, 4);
    acc += 16;
    frame1_ptr += stride;
    frame2_ptr += 16;
  }
}

static void temporal_filter_apply_8size_msa(
    uint8_t *frame1_ptr, uint32_t stride, uint8_t *frame2_ptr,
    int32_t strength_in, int32_t filter_wt_in, uint32_t *acc, uint16_t *cnt) {
  uint32_t row;
  uint64_t f0, f1, f2, f3, f4, f5, f6, f7;
  v16i8 frame1 = { 0 };
  v16i8 frame2 = { 0 };
  v16i8 frame3 = { 0 };
  v16i8 frame4 = { 0 };
  v16u8 frame_l, frame_h;
  v8i16 frame2_0_h, frame2_1_h, mod0_h, mod1_h;
  v8i16 diff0, diff1, cnt0, cnt1;
  v4i32 const3, const16;
  v4i32 filter_wt, strength;
  v4i32 mod0_w, mod1_w, mod2_w, mod3_w;
  v4i32 diff0_r, diff0_l, diff1_r, diff1_l;
  v4i32 frame2_0, frame2_1, frame2_2, frame2_3;
  v4i32 acc0, acc1, acc2, acc3;

  filter_wt = __msa_fill_w(filter_wt_in);
  strength = __msa_fill_w(strength_in);
  const3 = __msa_ldi_w(3);
  const16 = __msa_ldi_w(16);

  for (row = 2; row--;) {
    LD2(frame1_ptr, stride, f0, f1);
    frame1_ptr += (2 * stride);
    LD2(frame2_ptr, 8, f2, f3);
    frame2_ptr += 16;
    LD2(frame1_ptr, stride, f4, f5);
    frame1_ptr += (2 * stride);
    LD2(frame2_ptr, 8, f6, f7);
    frame2_ptr += 16;

    LD_SW2(acc, 4, acc0, acc1);
    LD_SW2(acc + 8, 4, acc2, acc3);
    LD_SH2(cnt, 8, cnt0, cnt1);
    INSERT_D2_SB(f0, f1, frame1);
    INSERT_D2_SB(f2, f3, frame2);
    INSERT_D2_SB(f4, f5, frame3);
    INSERT_D2_SB(f6, f7, frame4);
    ILVRL_B2_UB(frame1, frame2, frame_l, frame_h);
    HSUB_UB2_SH(frame_l, frame_h, diff0, diff1);
    UNPCK_SH_SW(diff0, diff0_r, diff0_l);
    UNPCK_SH_SW(diff1, diff1_r, diff1_l);
    MUL4(diff0_r, diff0_r, diff0_l, diff0_l, diff1_r, diff1_r, diff1_l, diff1_l,
         mod0_w, mod1_w, mod2_w, mod3_w);
    MUL4(mod0_w, const3, mod1_w, const3, mod2_w, const3, mod3_w, const3, mod0_w,
         mod1_w, mod2_w, mod3_w);
    SRAR_W4_SW(mod0_w, mod1_w, mod2_w, mod3_w, strength);
    diff0_r = (mod0_w < const16);
    diff0_l = (mod1_w < const16);
    diff1_r = (mod2_w < const16);
    diff1_l = (mod3_w < const16);
    SUB4(const16, mod0_w, const16, mod1_w, const16, mod2_w, const16, mod3_w,
         mod0_w, mod1_w, mod2_w, mod3_w);
    mod0_w = diff0_r & mod0_w;
    mod1_w = diff0_l & mod1_w;
    mod2_w = diff1_r & mod2_w;
    mod3_w = diff1_l & mod3_w;
    MUL4(mod0_w, filter_wt, mod1_w, filter_wt, mod2_w, filter_wt, mod3_w,
         filter_wt, mod0_w, mod1_w, mod2_w, mod3_w);
    PCKEV_H2_SH(mod1_w, mod0_w, mod3_w, mod2_w, mod0_h, mod1_h);
    ADD2(mod0_h, cnt0, mod1_h, cnt1, mod0_h, mod1_h);
    ST_SH2(mod0_h, mod1_h, cnt, 8);
    cnt += 16;

    UNPCK_UB_SH(frame2, frame2_0_h, frame2_1_h);
    UNPCK_SH_SW(frame2_0_h, frame2_0, frame2_1);
    UNPCK_SH_SW(frame2_1_h, frame2_2, frame2_3);
    MUL4(mod0_w, frame2_0, mod1_w, frame2_1, mod2_w, frame2_2, mod3_w, frame2_3,
         mod0_w, mod1_w, mod2_w, mod3_w);
    ADD4(mod0_w, acc0, mod1_w, acc1, mod2_w, acc2, mod3_w, acc3, mod0_w, mod1_w,
         mod2_w, mod3_w);
    ST_SW2(mod0_w, mod1_w, acc, 4);
    ST_SW2(mod2_w, mod3_w, acc + 8, 4);
    acc += 16;

    LD_SW2(acc, 4, acc0, acc1);
    LD_SW2(acc + 8, 4, acc2, acc3);
    LD_SH2(cnt, 8, cnt0, cnt1);
    ILVRL_B2_UB(frame3, frame4, frame_l, frame_h);
    HSUB_UB2_SH(frame_l, frame_h, diff0, diff1);
    UNPCK_SH_SW(diff0, diff0_r, diff0_l);
    UNPCK_SH_SW(diff1, diff1_r, diff1_l);
    MUL4(diff0_r, diff0_r, diff0_l, diff0_l, diff1_r, diff1_r, diff1_l, diff1_l,
         mod0_w, mod1_w, mod2_w, mod3_w);
    MUL4(mod0_w, const3, mod1_w, const3, mod2_w, const3, mod3_w, const3, mod0_w,
         mod1_w, mod2_w, mod3_w);
    SRAR_W4_SW(mod0_w, mod1_w, mod2_w, mod3_w, strength);
    diff0_r = (mod0_w < const16);
    diff0_l = (mod1_w < const16);
    diff1_r = (mod2_w < const16);
    diff1_l = (mod3_w < const16);
    SUB4(const16, mod0_w, const16, mod1_w, const16, mod2_w, const16, mod3_w,
         mod0_w, mod1_w, mod2_w, mod3_w);
    mod0_w = diff0_r & mod0_w;
    mod1_w = diff0_l & mod1_w;
    mod2_w = diff1_r & mod2_w;
    mod3_w = diff1_l & mod3_w;
    MUL4(mod0_w, filter_wt, mod1_w, filter_wt, mod2_w, filter_wt, mod3_w,
         filter_wt, mod0_w, mod1_w, mod2_w, mod3_w);
    PCKEV_H2_SH(mod1_w, mod0_w, mod3_w, mod2_w, mod0_h, mod1_h);
    ADD2(mod0_h, cnt0, mod1_h, cnt1, mod0_h, mod1_h);
    ST_SH2(mod0_h, mod1_h, cnt, 8);
    cnt += 16;

    UNPCK_UB_SH(frame4, frame2_0_h, frame2_1_h);
    UNPCK_SH_SW(frame2_0_h, frame2_0, frame2_1);
    UNPCK_SH_SW(frame2_1_h, frame2_2, frame2_3);
    MUL4(mod0_w, frame2_0, mod1_w, frame2_1, mod2_w, frame2_2, mod3_w, frame2_3,
         mod0_w, mod1_w, mod2_w, mod3_w);
    ADD4(mod0_w, acc0, mod1_w, acc1, mod2_w, acc2, mod3_w, acc3, mod0_w, mod1_w,
         mod2_w, mod3_w);
    ST_SW2(mod0_w, mod1_w, acc, 4);
    ST_SW2(mod2_w, mod3_w, acc + 8, 4);
    acc += 16;
  }
}

void vp8_temporal_filter_apply_msa(uint8_t *frame1, uint32_t stride,
                                   uint8_t *frame2, uint32_t block_size,
                                   int32_t strength, int32_t filter_weight,
                                   uint32_t *accumulator, uint16_t *count) {
  if (8 == block_size) {
    temporal_filter_apply_8size_msa(frame1, stride, frame2, strength,
                                    filter_weight, accumulator, count);
  } else if (16 == block_size) {
    temporal_filter_apply_16size_msa(frame1, stride, frame2, strength,
                                     filter_weight, accumulator, count);
  } else {
    uint32_t i, j, k;
    int32_t modifier;
    int32_t byte = 0;
    const int32_t rounding = strength > 0 ? 1 << (strength - 1) : 0;

    for (i = 0, k = 0; i < block_size; ++i) {
      for (j = 0; j < block_size; ++j, ++k) {
        int src_byte = frame1[byte];
        int pixel_value = *frame2++;

        modifier = src_byte - pixel_value;
        modifier *= modifier;
        modifier *= 3;
        modifier += rounding;
        modifier >>= strength;

        if (modifier > 16) modifier = 16;

        modifier = 16 - modifier;
        modifier *= filter_weight;

        count[k] += modifier;
        accumulator[k] += modifier * pixel_value;

        byte++;
      }

      byte += stride - block_size;
    }
  }
}
