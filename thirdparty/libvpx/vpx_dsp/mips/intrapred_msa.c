/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

#define IPRED_SUBS_UH2_UH(in0, in1, out0, out1) \
  {                                             \
    out0 = __msa_subs_u_h(out0, in0);           \
    out1 = __msa_subs_u_h(out1, in1);           \
  }

static void intra_predict_vert_4x4_msa(const uint8_t *src, uint8_t *dst,
                                       int32_t dst_stride) {
  uint32_t src_data;

  src_data = LW(src);

  SW4(src_data, src_data, src_data, src_data, dst, dst_stride);
}

static void intra_predict_vert_8x8_msa(const uint8_t *src, uint8_t *dst,
                                       int32_t dst_stride) {
  uint32_t row;
  uint32_t src_data1, src_data2;

  src_data1 = LW(src);
  src_data2 = LW(src + 4);

  for (row = 8; row--;) {
    SW(src_data1, dst);
    SW(src_data2, (dst + 4));
    dst += dst_stride;
  }
}

static void intra_predict_vert_16x16_msa(const uint8_t *src, uint8_t *dst,
                                         int32_t dst_stride) {
  uint32_t row;
  v16u8 src0;

  src0 = LD_UB(src);

  for (row = 16; row--;) {
    ST_UB(src0, dst);
    dst += dst_stride;
  }
}

static void intra_predict_vert_32x32_msa(const uint8_t *src, uint8_t *dst,
                                         int32_t dst_stride) {
  uint32_t row;
  v16u8 src1, src2;

  src1 = LD_UB(src);
  src2 = LD_UB(src + 16);

  for (row = 32; row--;) {
    ST_UB2(src1, src2, dst, 16);
    dst += dst_stride;
  }
}

static void intra_predict_horiz_4x4_msa(const uint8_t *src, uint8_t *dst,
                                        int32_t dst_stride) {
  uint32_t out0, out1, out2, out3;

  out0 = src[0] * 0x01010101;
  out1 = src[1] * 0x01010101;
  out2 = src[2] * 0x01010101;
  out3 = src[3] * 0x01010101;

  SW4(out0, out1, out2, out3, dst, dst_stride);
}

static void intra_predict_horiz_8x8_msa(const uint8_t *src, uint8_t *dst,
                                        int32_t dst_stride) {
  uint64_t out0, out1, out2, out3, out4, out5, out6, out7;

  out0 = src[0] * 0x0101010101010101ull;
  out1 = src[1] * 0x0101010101010101ull;
  out2 = src[2] * 0x0101010101010101ull;
  out3 = src[3] * 0x0101010101010101ull;
  out4 = src[4] * 0x0101010101010101ull;
  out5 = src[5] * 0x0101010101010101ull;
  out6 = src[6] * 0x0101010101010101ull;
  out7 = src[7] * 0x0101010101010101ull;

  SD4(out0, out1, out2, out3, dst, dst_stride);
  dst += (4 * dst_stride);
  SD4(out4, out5, out6, out7, dst, dst_stride);
}

static void intra_predict_horiz_16x16_msa(const uint8_t *src, uint8_t *dst,
                                          int32_t dst_stride) {
  uint32_t row;
  uint8_t inp0, inp1, inp2, inp3;
  v16u8 src0, src1, src2, src3;

  for (row = 4; row--;) {
    inp0 = src[0];
    inp1 = src[1];
    inp2 = src[2];
    inp3 = src[3];
    src += 4;

    src0 = (v16u8)__msa_fill_b(inp0);
    src1 = (v16u8)__msa_fill_b(inp1);
    src2 = (v16u8)__msa_fill_b(inp2);
    src3 = (v16u8)__msa_fill_b(inp3);

    ST_UB4(src0, src1, src2, src3, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void intra_predict_horiz_32x32_msa(const uint8_t *src, uint8_t *dst,
                                          int32_t dst_stride) {
  uint32_t row;
  uint8_t inp0, inp1, inp2, inp3;
  v16u8 src0, src1, src2, src3;

  for (row = 8; row--;) {
    inp0 = src[0];
    inp1 = src[1];
    inp2 = src[2];
    inp3 = src[3];
    src += 4;

    src0 = (v16u8)__msa_fill_b(inp0);
    src1 = (v16u8)__msa_fill_b(inp1);
    src2 = (v16u8)__msa_fill_b(inp2);
    src3 = (v16u8)__msa_fill_b(inp3);

    ST_UB2(src0, src0, dst, 16);
    dst += dst_stride;
    ST_UB2(src1, src1, dst, 16);
    dst += dst_stride;
    ST_UB2(src2, src2, dst, 16);
    dst += dst_stride;
    ST_UB2(src3, src3, dst, 16);
    dst += dst_stride;
  }
}

static void intra_predict_dc_4x4_msa(const uint8_t *src_top,
                                     const uint8_t *src_left, uint8_t *dst,
                                     int32_t dst_stride) {
  uint32_t val0, val1;
  v16i8 store, src = { 0 };
  v8u16 sum_h;
  v4u32 sum_w;
  v2u64 sum_d;

  val0 = LW(src_top);
  val1 = LW(src_left);
  INSERT_W2_SB(val0, val1, src);
  sum_h = __msa_hadd_u_h((v16u8)src, (v16u8)src);
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 3);
  store = __msa_splati_b((v16i8)sum_w, 0);
  val0 = __msa_copy_u_w((v4i32)store, 0);

  SW4(val0, val0, val0, val0, dst, dst_stride);
}

static void intra_predict_dc_tl_4x4_msa(const uint8_t *src, uint8_t *dst,
                                        int32_t dst_stride) {
  uint32_t val0;
  v16i8 store, data = { 0 };
  v8u16 sum_h;
  v4u32 sum_w;

  val0 = LW(src);
  data = (v16i8)__msa_insert_w((v4i32)data, 0, val0);
  sum_h = __msa_hadd_u_h((v16u8)data, (v16u8)data);
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_w, 2);
  store = __msa_splati_b((v16i8)sum_w, 0);
  val0 = __msa_copy_u_w((v4i32)store, 0);

  SW4(val0, val0, val0, val0, dst, dst_stride);
}

static void intra_predict_128dc_4x4_msa(uint8_t *dst, int32_t dst_stride) {
  uint32_t out;
  const v16i8 store = __msa_ldi_b(128);

  out = __msa_copy_u_w((v4i32)store, 0);

  SW4(out, out, out, out, dst, dst_stride);
}

static void intra_predict_dc_8x8_msa(const uint8_t *src_top,
                                     const uint8_t *src_left, uint8_t *dst,
                                     int32_t dst_stride) {
  uint64_t val0, val1;
  v16i8 store;
  v16u8 src = { 0 };
  v8u16 sum_h;
  v4u32 sum_w;
  v2u64 sum_d;

  val0 = LD(src_top);
  val1 = LD(src_left);
  INSERT_D2_UB(val0, val1, src);
  sum_h = __msa_hadd_u_h(src, src);
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_pckev_w((v4i32)sum_d, (v4i32)sum_d);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 4);
  store = __msa_splati_b((v16i8)sum_w, 0);
  val0 = __msa_copy_u_d((v2i64)store, 0);

  SD4(val0, val0, val0, val0, dst, dst_stride);
  dst += (4 * dst_stride);
  SD4(val0, val0, val0, val0, dst, dst_stride);
}

static void intra_predict_dc_tl_8x8_msa(const uint8_t *src, uint8_t *dst,
                                        int32_t dst_stride) {
  uint64_t val0;
  v16i8 store;
  v16u8 data = { 0 };
  v8u16 sum_h;
  v4u32 sum_w;
  v2u64 sum_d;

  val0 = LD(src);
  data = (v16u8)__msa_insert_d((v2i64)data, 0, val0);
  sum_h = __msa_hadd_u_h(data, data);
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 3);
  store = __msa_splati_b((v16i8)sum_w, 0);
  val0 = __msa_copy_u_d((v2i64)store, 0);

  SD4(val0, val0, val0, val0, dst, dst_stride);
  dst += (4 * dst_stride);
  SD4(val0, val0, val0, val0, dst, dst_stride);
}

static void intra_predict_128dc_8x8_msa(uint8_t *dst, int32_t dst_stride) {
  uint64_t out;
  const v16i8 store = __msa_ldi_b(128);

  out = __msa_copy_u_d((v2i64)store, 0);

  SD4(out, out, out, out, dst, dst_stride);
  dst += (4 * dst_stride);
  SD4(out, out, out, out, dst, dst_stride);
}

static void intra_predict_dc_16x16_msa(const uint8_t *src_top,
                                       const uint8_t *src_left, uint8_t *dst,
                                       int32_t dst_stride) {
  v16u8 top, left, out;
  v8u16 sum_h, sum_top, sum_left;
  v4u32 sum_w;
  v2u64 sum_d;

  top = LD_UB(src_top);
  left = LD_UB(src_left);
  HADD_UB2_UH(top, left, sum_top, sum_left);
  sum_h = sum_top + sum_left;
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_pckev_w((v4i32)sum_d, (v4i32)sum_d);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 5);
  out = (v16u8)__msa_splati_b((v16i8)sum_w, 0);

  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
  dst += (8 * dst_stride);
  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
}

static void intra_predict_dc_tl_16x16_msa(const uint8_t *src, uint8_t *dst,
                                          int32_t dst_stride) {
  v16u8 data, out;
  v8u16 sum_h;
  v4u32 sum_w;
  v2u64 sum_d;

  data = LD_UB(src);
  sum_h = __msa_hadd_u_h(data, data);
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_pckev_w((v4i32)sum_d, (v4i32)sum_d);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 4);
  out = (v16u8)__msa_splati_b((v16i8)sum_w, 0);

  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
  dst += (8 * dst_stride);
  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
}

static void intra_predict_128dc_16x16_msa(uint8_t *dst, int32_t dst_stride) {
  const v16u8 out = (v16u8)__msa_ldi_b(128);

  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
  dst += (8 * dst_stride);
  ST_UB8(out, out, out, out, out, out, out, out, dst, dst_stride);
}

static void intra_predict_dc_32x32_msa(const uint8_t *src_top,
                                       const uint8_t *src_left, uint8_t *dst,
                                       int32_t dst_stride) {
  uint32_t row;
  v16u8 top0, top1, left0, left1, out;
  v8u16 sum_h, sum_top0, sum_top1, sum_left0, sum_left1;
  v4u32 sum_w;
  v2u64 sum_d;

  LD_UB2(src_top, 16, top0, top1);
  LD_UB2(src_left, 16, left0, left1);
  HADD_UB2_UH(top0, top1, sum_top0, sum_top1);
  HADD_UB2_UH(left0, left1, sum_left0, sum_left1);
  sum_h = sum_top0 + sum_top1;
  sum_h += sum_left0 + sum_left1;
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_pckev_w((v4i32)sum_d, (v4i32)sum_d);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 6);
  out = (v16u8)__msa_splati_b((v16i8)sum_w, 0);

  for (row = 16; row--;) {
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
  }
}

static void intra_predict_dc_tl_32x32_msa(const uint8_t *src, uint8_t *dst,
                                          int32_t dst_stride) {
  uint32_t row;
  v16u8 data0, data1, out;
  v8u16 sum_h, sum_data0, sum_data1;
  v4u32 sum_w;
  v2u64 sum_d;

  LD_UB2(src, 16, data0, data1);
  HADD_UB2_UH(data0, data1, sum_data0, sum_data1);
  sum_h = sum_data0 + sum_data1;
  sum_w = __msa_hadd_u_w(sum_h, sum_h);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_pckev_w((v4i32)sum_d, (v4i32)sum_d);
  sum_d = __msa_hadd_u_d(sum_w, sum_w);
  sum_w = (v4u32)__msa_srari_w((v4i32)sum_d, 5);
  out = (v16u8)__msa_splati_b((v16i8)sum_w, 0);

  for (row = 16; row--;) {
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
  }
}

static void intra_predict_128dc_32x32_msa(uint8_t *dst, int32_t dst_stride) {
  uint32_t row;
  const v16u8 out = (v16u8)__msa_ldi_b(128);

  for (row = 16; row--;) {
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
    ST_UB2(out, out, dst, 16);
    dst += dst_stride;
  }
}

static void intra_predict_tm_4x4_msa(const uint8_t *src_top_ptr,
                                     const uint8_t *src_left, uint8_t *dst,
                                     int32_t dst_stride) {
  uint32_t val;
  uint8_t top_left = src_top_ptr[-1];
  v16i8 src_left0, src_left1, src_left2, src_left3, tmp0, tmp1, src_top = { 0 };
  v16u8 src0, src1, src2, src3;
  v8u16 src_top_left, vec0, vec1, vec2, vec3;

  src_top_left = (v8u16)__msa_fill_h(top_left);
  val = LW(src_top_ptr);
  src_top = (v16i8)__msa_insert_w((v4i32)src_top, 0, val);

  src_left0 = __msa_fill_b(src_left[0]);
  src_left1 = __msa_fill_b(src_left[1]);
  src_left2 = __msa_fill_b(src_left[2]);
  src_left3 = __msa_fill_b(src_left[3]);

  ILVR_B4_UB(src_left0, src_top, src_left1, src_top, src_left2, src_top,
             src_left3, src_top, src0, src1, src2, src3);
  HADD_UB4_UH(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
  IPRED_SUBS_UH2_UH(src_top_left, src_top_left, vec0, vec1);
  IPRED_SUBS_UH2_UH(src_top_left, src_top_left, vec2, vec3);
  SAT_UH4_UH(vec0, vec1, vec2, vec3, 7);
  PCKEV_B2_SB(vec1, vec0, vec3, vec2, tmp0, tmp1);
  ST4x4_UB(tmp0, tmp1, 0, 2, 0, 2, dst, dst_stride);
}

static void intra_predict_tm_8x8_msa(const uint8_t *src_top_ptr,
                                     const uint8_t *src_left, uint8_t *dst,
                                     int32_t dst_stride) {
  uint64_t val;
  uint8_t top_left = src_top_ptr[-1];
  uint32_t loop_cnt;
  v16i8 src_left0, src_left1, src_left2, src_left3, tmp0, tmp1, src_top = { 0 };
  v8u16 src_top_left, vec0, vec1, vec2, vec3;
  v16u8 src0, src1, src2, src3;

  val = LD(src_top_ptr);
  src_top = (v16i8)__msa_insert_d((v2i64)src_top, 0, val);
  src_top_left = (v8u16)__msa_fill_h(top_left);

  for (loop_cnt = 2; loop_cnt--;) {
    src_left0 = __msa_fill_b(src_left[0]);
    src_left1 = __msa_fill_b(src_left[1]);
    src_left2 = __msa_fill_b(src_left[2]);
    src_left3 = __msa_fill_b(src_left[3]);
    src_left += 4;

    ILVR_B4_UB(src_left0, src_top, src_left1, src_top, src_left2, src_top,
               src_left3, src_top, src0, src1, src2, src3);
    HADD_UB4_UH(src0, src1, src2, src3, vec0, vec1, vec2, vec3);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, vec0, vec1);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, vec2, vec3);
    SAT_UH4_UH(vec0, vec1, vec2, vec3, 7);
    PCKEV_B2_SB(vec1, vec0, vec3, vec2, tmp0, tmp1);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void intra_predict_tm_16x16_msa(const uint8_t *src_top_ptr,
                                       const uint8_t *src_left, uint8_t *dst,
                                       int32_t dst_stride) {
  uint8_t top_left = src_top_ptr[-1];
  uint32_t loop_cnt;
  v16i8 src_top, src_left0, src_left1, src_left2, src_left3;
  v8u16 src_top_left, res_r, res_l;

  src_top = LD_SB(src_top_ptr);
  src_top_left = (v8u16)__msa_fill_h(top_left);

  for (loop_cnt = 4; loop_cnt--;) {
    src_left0 = __msa_fill_b(src_left[0]);
    src_left1 = __msa_fill_b(src_left[1]);
    src_left2 = __msa_fill_b(src_left[2]);
    src_left3 = __msa_fill_b(src_left[3]);
    src_left += 4;

    ILVRL_B2_UH(src_left0, src_top, res_r, res_l);
    HADD_UB2_UH(res_r, res_l, res_r, res_l);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r, res_l);

    SAT_UH2_UH(res_r, res_l, 7);
    PCKEV_ST_SB(res_r, res_l, dst);
    dst += dst_stride;

    ILVRL_B2_UH(src_left1, src_top, res_r, res_l);
    HADD_UB2_UH(res_r, res_l, res_r, res_l);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r, res_l);
    SAT_UH2_UH(res_r, res_l, 7);
    PCKEV_ST_SB(res_r, res_l, dst);
    dst += dst_stride;

    ILVRL_B2_UH(src_left2, src_top, res_r, res_l);
    HADD_UB2_UH(res_r, res_l, res_r, res_l);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r, res_l);
    SAT_UH2_UH(res_r, res_l, 7);
    PCKEV_ST_SB(res_r, res_l, dst);
    dst += dst_stride;

    ILVRL_B2_UH(src_left3, src_top, res_r, res_l);
    HADD_UB2_UH(res_r, res_l, res_r, res_l);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r, res_l);
    SAT_UH2_UH(res_r, res_l, 7);
    PCKEV_ST_SB(res_r, res_l, dst);
    dst += dst_stride;
  }
}

static void intra_predict_tm_32x32_msa(const uint8_t *src_top,
                                       const uint8_t *src_left, uint8_t *dst,
                                       int32_t dst_stride) {
  uint8_t top_left = src_top[-1];
  uint32_t loop_cnt;
  v16i8 src_top0, src_top1, src_left0, src_left1, src_left2, src_left3;
  v8u16 src_top_left, res_r0, res_r1, res_l0, res_l1;

  LD_SB2(src_top, 16, src_top0, src_top1);
  src_top_left = (v8u16)__msa_fill_h(top_left);

  for (loop_cnt = 8; loop_cnt--;) {
    src_left0 = __msa_fill_b(src_left[0]);
    src_left1 = __msa_fill_b(src_left[1]);
    src_left2 = __msa_fill_b(src_left[2]);
    src_left3 = __msa_fill_b(src_left[3]);
    src_left += 4;

    ILVR_B2_UH(src_left0, src_top0, src_left0, src_top1, res_r0, res_r1);
    ILVL_B2_UH(src_left0, src_top0, src_left0, src_top1, res_l0, res_l1);
    HADD_UB4_UH(res_r0, res_l0, res_r1, res_l1, res_r0, res_l0, res_r1, res_l1);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r0, res_l0);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r1, res_l1);
    SAT_UH4_UH(res_r0, res_l0, res_r1, res_l1, 7);
    PCKEV_ST_SB(res_r0, res_l0, dst);
    PCKEV_ST_SB(res_r1, res_l1, dst + 16);
    dst += dst_stride;

    ILVR_B2_UH(src_left1, src_top0, src_left1, src_top1, res_r0, res_r1);
    ILVL_B2_UH(src_left1, src_top0, src_left1, src_top1, res_l0, res_l1);
    HADD_UB4_UH(res_r0, res_l0, res_r1, res_l1, res_r0, res_l0, res_r1, res_l1);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r0, res_l0);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r1, res_l1);
    SAT_UH4_UH(res_r0, res_l0, res_r1, res_l1, 7);
    PCKEV_ST_SB(res_r0, res_l0, dst);
    PCKEV_ST_SB(res_r1, res_l1, dst + 16);
    dst += dst_stride;

    ILVR_B2_UH(src_left2, src_top0, src_left2, src_top1, res_r0, res_r1);
    ILVL_B2_UH(src_left2, src_top0, src_left2, src_top1, res_l0, res_l1);
    HADD_UB4_UH(res_r0, res_l0, res_r1, res_l1, res_r0, res_l0, res_r1, res_l1);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r0, res_l0);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r1, res_l1);
    SAT_UH4_UH(res_r0, res_l0, res_r1, res_l1, 7);
    PCKEV_ST_SB(res_r0, res_l0, dst);
    PCKEV_ST_SB(res_r1, res_l1, dst + 16);
    dst += dst_stride;

    ILVR_B2_UH(src_left3, src_top0, src_left3, src_top1, res_r0, res_r1);
    ILVL_B2_UH(src_left3, src_top0, src_left3, src_top1, res_l0, res_l1);
    HADD_UB4_UH(res_r0, res_l0, res_r1, res_l1, res_r0, res_l0, res_r1, res_l1);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r0, res_l0);
    IPRED_SUBS_UH2_UH(src_top_left, src_top_left, res_r1, res_l1);
    SAT_UH4_UH(res_r0, res_l0, res_r1, res_l1, 7);
    PCKEV_ST_SB(res_r0, res_l0, dst);
    PCKEV_ST_SB(res_r1, res_l1, dst + 16);
    dst += dst_stride;
  }
}

void vpx_v_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                             const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_vert_4x4_msa(above, dst, y_stride);
}

void vpx_v_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                             const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_vert_8x8_msa(above, dst, y_stride);
}

void vpx_v_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                               const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_vert_16x16_msa(above, dst, y_stride);
}

void vpx_v_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                               const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_vert_32x32_msa(above, dst, y_stride);
}

void vpx_h_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                             const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_horiz_4x4_msa(left, dst, y_stride);
}

void vpx_h_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                             const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_horiz_8x8_msa(left, dst, y_stride);
}

void vpx_h_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                               const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_horiz_16x16_msa(left, dst, y_stride);
}

void vpx_h_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                               const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_horiz_32x32_msa(left, dst, y_stride);
}

void vpx_dc_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                              const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_4x4_msa(above, left, dst, y_stride);
}

void vpx_dc_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                              const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_8x8_msa(above, left, dst, y_stride);
}

void vpx_dc_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                                const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_16x16_msa(above, left, dst, y_stride);
}

void vpx_dc_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                                const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_32x32_msa(above, left, dst, y_stride);
}

void vpx_dc_top_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_dc_tl_4x4_msa(above, dst, y_stride);
}

void vpx_dc_top_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_dc_tl_8x8_msa(above, dst, y_stride);
}

void vpx_dc_top_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_dc_tl_16x16_msa(above, dst, y_stride);
}

void vpx_dc_top_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)left;

  intra_predict_dc_tl_32x32_msa(above, dst, y_stride);
}

void vpx_dc_left_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_dc_tl_4x4_msa(left, dst, y_stride);
}

void vpx_dc_left_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)above;

  intra_predict_dc_tl_8x8_msa(left, dst, y_stride);
}

void vpx_dc_left_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;

  intra_predict_dc_tl_16x16_msa(left, dst, y_stride);
}

void vpx_dc_left_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;

  intra_predict_dc_tl_32x32_msa(left, dst, y_stride);
}

void vpx_dc_128_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;

  intra_predict_128dc_4x4_msa(dst, y_stride);
}

void vpx_dc_128_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;

  intra_predict_128dc_8x8_msa(dst, y_stride);
}

void vpx_dc_128_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;

  intra_predict_128dc_16x16_msa(dst, y_stride);
}

void vpx_dc_128_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;

  intra_predict_128dc_32x32_msa(dst, y_stride);
}

void vpx_tm_predictor_4x4_msa(uint8_t *dst, ptrdiff_t y_stride,
                              const uint8_t *above, const uint8_t *left) {
  intra_predict_tm_4x4_msa(above, left, dst, y_stride);
}

void vpx_tm_predictor_8x8_msa(uint8_t *dst, ptrdiff_t y_stride,
                              const uint8_t *above, const uint8_t *left) {
  intra_predict_tm_8x8_msa(above, left, dst, y_stride);
}

void vpx_tm_predictor_16x16_msa(uint8_t *dst, ptrdiff_t y_stride,
                                const uint8_t *above, const uint8_t *left) {
  intra_predict_tm_16x16_msa(above, left, dst, y_stride);
}

void vpx_tm_predictor_32x32_msa(uint8_t *dst, ptrdiff_t y_stride,
                                const uint8_t *above, const uint8_t *left) {
  intra_predict_tm_32x32_msa(above, left, dst, y_stride);
}
