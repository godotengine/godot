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
#include "vp8/common/loopfilter.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

#define VP8_SIMPLE_MASK(p1, p0, q0, q1, b_limit, mask)        \
  {                                                           \
    v16u8 p1_a_sub_q1, p0_a_sub_q0;                           \
                                                              \
    p0_a_sub_q0 = __msa_asub_u_b(p0, q0);                     \
    p1_a_sub_q1 = __msa_asub_u_b(p1, q1);                     \
    p1_a_sub_q1 = (v16u8)__msa_srli_b((v16i8)p1_a_sub_q1, 1); \
    p0_a_sub_q0 = __msa_adds_u_b(p0_a_sub_q0, p0_a_sub_q0);   \
    mask = __msa_adds_u_b(p0_a_sub_q0, p1_a_sub_q1);          \
    mask = ((v16u8)mask <= b_limit);                          \
  }

#define VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev)      \
  {                                                        \
    v16i8 p1_m, p0_m, q0_m, q1_m, filt, q0_sub_p0, t1, t2; \
    const v16i8 cnst4b = __msa_ldi_b(4);                   \
    const v16i8 cnst3b = __msa_ldi_b(3);                   \
                                                           \
    p1_m = (v16i8)__msa_xori_b(p1, 0x80);                  \
    p0_m = (v16i8)__msa_xori_b(p0, 0x80);                  \
    q0_m = (v16i8)__msa_xori_b(q0, 0x80);                  \
    q1_m = (v16i8)__msa_xori_b(q1, 0x80);                  \
                                                           \
    filt = __msa_subs_s_b(p1_m, q1_m);                     \
    filt &= hev;                                           \
    q0_sub_p0 = __msa_subs_s_b(q0_m, p0_m);                \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                \
    filt &= mask;                                          \
    t1 = __msa_adds_s_b(filt, cnst4b);                     \
    t1 >>= cnst3b;                                         \
    t2 = __msa_adds_s_b(filt, cnst3b);                     \
    t2 >>= cnst3b;                                         \
    q0_m = __msa_subs_s_b(q0_m, t1);                       \
    q0 = __msa_xori_b((v16u8)q0_m, 0x80);                  \
    p0_m = __msa_adds_s_b(p0_m, t2);                       \
    p0 = __msa_xori_b((v16u8)p0_m, 0x80);                  \
    filt = __msa_srari_b(t1, 1);                           \
    hev = __msa_xori_b(hev, 0xff);                         \
    filt &= hev;                                           \
    q1_m = __msa_subs_s_b(q1_m, filt);                     \
    q1 = __msa_xori_b((v16u8)q1_m, 0x80);                  \
    p1_m = __msa_adds_s_b(p1_m, filt);                     \
    p1 = __msa_xori_b((v16u8)p1_m, 0x80);                  \
  }

#define VP8_SIMPLE_FILT(p1_in, p0_in, q0_in, q1_in, mask) \
  {                                                       \
    v16i8 p1_m, p0_m, q0_m, q1_m, filt, filt1, filt2;     \
    v16i8 q0_sub_p0;                                      \
    const v16i8 cnst4b = __msa_ldi_b(4);                  \
    const v16i8 cnst3b = __msa_ldi_b(3);                  \
                                                          \
    p1_m = (v16i8)__msa_xori_b(p1_in, 0x80);              \
    p0_m = (v16i8)__msa_xori_b(p0_in, 0x80);              \
    q0_m = (v16i8)__msa_xori_b(q0_in, 0x80);              \
    q1_m = (v16i8)__msa_xori_b(q1_in, 0x80);              \
                                                          \
    filt = __msa_subs_s_b(p1_m, q1_m);                    \
    q0_sub_p0 = __msa_subs_s_b(q0_m, p0_m);               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);               \
    filt &= mask;                                         \
    filt1 = __msa_adds_s_b(filt, cnst4b);                 \
    filt1 >>= cnst3b;                                     \
    filt2 = __msa_adds_s_b(filt, cnst3b);                 \
    filt2 >>= cnst3b;                                     \
    q0_m = __msa_subs_s_b(q0_m, filt1);                   \
    p0_m = __msa_adds_s_b(p0_m, filt2);                   \
    q0_in = __msa_xori_b((v16u8)q0_m, 0x80);              \
    p0_in = __msa_xori_b((v16u8)p0_m, 0x80);              \
  }

#define VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev) \
  {                                                     \
    v16i8 p2_m, p1_m, p0_m, q2_m, q1_m, q0_m;           \
    v16i8 u, filt, t1, t2, filt_sign, q0_sub_p0;        \
    v8i16 filt_r, filt_l, u_r, u_l;                     \
    v8i16 temp0, temp1, temp2, temp3;                   \
    const v16i8 cnst4b = __msa_ldi_b(4);                \
    const v16i8 cnst3b = __msa_ldi_b(3);                \
    const v8i16 cnst9h = __msa_ldi_h(9);                \
    const v8i16 cnst63h = __msa_ldi_h(63);              \
                                                        \
    p2_m = (v16i8)__msa_xori_b(p2, 0x80);               \
    p1_m = (v16i8)__msa_xori_b(p1, 0x80);               \
    p0_m = (v16i8)__msa_xori_b(p0, 0x80);               \
    q0_m = (v16i8)__msa_xori_b(q0, 0x80);               \
    q1_m = (v16i8)__msa_xori_b(q1, 0x80);               \
    q2_m = (v16i8)__msa_xori_b(q2, 0x80);               \
                                                        \
    filt = __msa_subs_s_b(p1_m, q1_m);                  \
    q0_sub_p0 = __msa_subs_s_b(q0_m, p0_m);             \
    filt = __msa_adds_s_b(filt, q0_sub_p0);             \
    filt = __msa_adds_s_b(filt, q0_sub_p0);             \
    filt = __msa_adds_s_b(filt, q0_sub_p0);             \
    filt &= mask;                                       \
                                                        \
    t2 = filt & hev;                                    \
    hev = __msa_xori_b(hev, 0xff);                      \
    filt &= hev;                                        \
    t1 = __msa_adds_s_b(t2, cnst4b);                    \
    t1 >>= cnst3b;                                      \
    t2 = __msa_adds_s_b(t2, cnst3b);                    \
    t2 >>= cnst3b;                                      \
    q0_m = __msa_subs_s_b(q0_m, t1);                    \
    p0_m = __msa_adds_s_b(p0_m, t2);                    \
    filt_sign = __msa_clti_s_b(filt, 0);                \
    ILVRL_B2_SH(filt_sign, filt, filt_r, filt_l);       \
    temp0 = filt_r * cnst9h;                            \
    temp1 = temp0 + cnst63h;                            \
    temp2 = filt_l * cnst9h;                            \
    temp3 = temp2 + cnst63h;                            \
                                                        \
    u_r = temp1 >> 7;                                   \
    u_r = __msa_sat_s_h(u_r, 7);                        \
    u_l = temp3 >> 7;                                   \
    u_l = __msa_sat_s_h(u_l, 7);                        \
    u = __msa_pckev_b((v16i8)u_l, (v16i8)u_r);          \
    q2_m = __msa_subs_s_b(q2_m, u);                     \
    p2_m = __msa_adds_s_b(p2_m, u);                     \
    q2 = __msa_xori_b((v16u8)q2_m, 0x80);               \
    p2 = __msa_xori_b((v16u8)p2_m, 0x80);               \
                                                        \
    temp1 += temp0;                                     \
    temp3 += temp2;                                     \
                                                        \
    u_r = temp1 >> 7;                                   \
    u_r = __msa_sat_s_h(u_r, 7);                        \
    u_l = temp3 >> 7;                                   \
    u_l = __msa_sat_s_h(u_l, 7);                        \
    u = __msa_pckev_b((v16i8)u_l, (v16i8)u_r);          \
    q1_m = __msa_subs_s_b(q1_m, u);                     \
    p1_m = __msa_adds_s_b(p1_m, u);                     \
    q1 = __msa_xori_b((v16u8)q1_m, 0x80);               \
    p1 = __msa_xori_b((v16u8)p1_m, 0x80);               \
                                                        \
    temp1 += temp0;                                     \
    temp3 += temp2;                                     \
                                                        \
    u_r = temp1 >> 7;                                   \
    u_r = __msa_sat_s_h(u_r, 7);                        \
    u_l = temp3 >> 7;                                   \
    u_l = __msa_sat_s_h(u_l, 7);                        \
    u = __msa_pckev_b((v16i8)u_l, (v16i8)u_r);          \
    q0_m = __msa_subs_s_b(q0_m, u);                     \
    p0_m = __msa_adds_s_b(p0_m, u);                     \
    q0 = __msa_xori_b((v16u8)q0_m, 0x80);               \
    p0 = __msa_xori_b((v16u8)p0_m, 0x80);               \
  }

#define LPF_MASK_HEV(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                     limit_in, b_limit_in, thresh_in, hev_out, mask_out,     \
                     flat_out)                                               \
  {                                                                          \
    v16u8 p3_asub_p2_m, p2_asub_p1_m, p1_asub_p0_m, q1_asub_q0_m;            \
    v16u8 p1_asub_q1_m, p0_asub_q0_m, q3_asub_q2_m, q2_asub_q1_m;            \
                                                                             \
    p3_asub_p2_m = __msa_asub_u_b((p3_in), (p2_in));                         \
    p2_asub_p1_m = __msa_asub_u_b((p2_in), (p1_in));                         \
    p1_asub_p0_m = __msa_asub_u_b((p1_in), (p0_in));                         \
    q1_asub_q0_m = __msa_asub_u_b((q1_in), (q0_in));                         \
    q2_asub_q1_m = __msa_asub_u_b((q2_in), (q1_in));                         \
    q3_asub_q2_m = __msa_asub_u_b((q3_in), (q2_in));                         \
    p0_asub_q0_m = __msa_asub_u_b((p0_in), (q0_in));                         \
    p1_asub_q1_m = __msa_asub_u_b((p1_in), (q1_in));                         \
    flat_out = __msa_max_u_b(p1_asub_p0_m, q1_asub_q0_m);                    \
    hev_out = (thresh_in) < (v16u8)flat_out;                                 \
    p0_asub_q0_m = __msa_adds_u_b(p0_asub_q0_m, p0_asub_q0_m);               \
    p1_asub_q1_m >>= 1;                                                      \
    p0_asub_q0_m = __msa_adds_u_b(p0_asub_q0_m, p1_asub_q1_m);               \
    mask_out = (b_limit_in) < p0_asub_q0_m;                                  \
    mask_out = __msa_max_u_b(flat_out, mask_out);                            \
    p3_asub_p2_m = __msa_max_u_b(p3_asub_p2_m, p2_asub_p1_m);                \
    mask_out = __msa_max_u_b(p3_asub_p2_m, mask_out);                        \
    q2_asub_q1_m = __msa_max_u_b(q2_asub_q1_m, q3_asub_q2_m);                \
    mask_out = __msa_max_u_b(q2_asub_q1_m, mask_out);                        \
    mask_out = (limit_in) < (v16u8)mask_out;                                 \
    mask_out = __msa_xori_b(mask_out, 0xff);                                 \
  }

#define VP8_ST6x1_UB(in0, in0_idx, in1, in1_idx, pdst, stride) \
  {                                                            \
    uint16_t tmp0_h;                                           \
    uint32_t tmp0_w;                                           \
                                                               \
    tmp0_w = __msa_copy_u_w((v4i32)in0, in0_idx);              \
    tmp0_h = __msa_copy_u_h((v8i16)in1, in1_idx);              \
    SW(tmp0_w, pdst);                                          \
    SH(tmp0_h, pdst + stride);                                 \
  }

static void loop_filter_horizontal_4_dual_msa(uint8_t *src, int32_t pitch,
                                              const uint8_t *b_limit0_ptr,
                                              const uint8_t *limit0_ptr,
                                              const uint8_t *thresh0_ptr,
                                              const uint8_t *b_limit1_ptr,
                                              const uint8_t *limit1_ptr,
                                              const uint8_t *thresh1_ptr) {
  v16u8 mask, hev, flat;
  v16u8 thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;

  LD_UB8((src - 4 * pitch), pitch, p3, p2, p1, p0, q0, q1, q2, q3);
  thresh0 = (v16u8)__msa_fill_b(*thresh0_ptr);
  thresh1 = (v16u8)__msa_fill_b(*thresh1_ptr);
  thresh0 = (v16u8)__msa_ilvr_d((v2i64)thresh1, (v2i64)thresh0);

  b_limit0 = (v16u8)__msa_fill_b(*b_limit0_ptr);
  b_limit1 = (v16u8)__msa_fill_b(*b_limit1_ptr);
  b_limit0 = (v16u8)__msa_ilvr_d((v2i64)b_limit1, (v2i64)b_limit0);

  limit0 = (v16u8)__msa_fill_b(*limit0_ptr);
  limit1 = (v16u8)__msa_fill_b(*limit1_ptr);
  limit0 = (v16u8)__msa_ilvr_d((v2i64)limit1, (v2i64)limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  ST_UB4(p1, p0, q0, q1, (src - 2 * pitch), pitch);
}

static void loop_filter_vertical_4_dual_msa(uint8_t *src, int32_t pitch,
                                            const uint8_t *b_limit0_ptr,
                                            const uint8_t *limit0_ptr,
                                            const uint8_t *thresh0_ptr,
                                            const uint8_t *b_limit1_ptr,
                                            const uint8_t *limit1_ptr,
                                            const uint8_t *thresh1_ptr) {
  v16u8 mask, hev, flat;
  v16u8 thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7;
  v16u8 row8, row9, row10, row11, row12, row13, row14, row15;
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

  LD_UB8(src - 4, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  LD_UB8(src - 4 + (8 * pitch), pitch, row8, row9, row10, row11, row12, row13,
         row14, row15);
  TRANSPOSE16x8_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  thresh0 = (v16u8)__msa_fill_b(*thresh0_ptr);
  thresh1 = (v16u8)__msa_fill_b(*thresh1_ptr);
  thresh0 = (v16u8)__msa_ilvr_d((v2i64)thresh1, (v2i64)thresh0);

  b_limit0 = (v16u8)__msa_fill_b(*b_limit0_ptr);
  b_limit1 = (v16u8)__msa_fill_b(*b_limit1_ptr);
  b_limit0 = (v16u8)__msa_ilvr_d((v2i64)b_limit1, (v2i64)b_limit0);

  limit0 = (v16u8)__msa_fill_b(*limit0_ptr);
  limit1 = (v16u8)__msa_fill_b(*limit1_ptr);
  limit0 = (v16u8)__msa_ilvr_d((v2i64)limit1, (v2i64)limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);
  ILVR_B2_SH(p0, p1, q1, q0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp2, tmp3);
  ILVL_B2_SH(p0, p1, q1, q0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp4, tmp5);

  src -= 2;
  ST4x8_UB(tmp2, tmp3, src, pitch);
  src += (8 * pitch);
  ST4x8_UB(tmp4, tmp5, src, pitch);
}

static void mbloop_filter_horizontal_edge_y_msa(uint8_t *src, int32_t pitch,
                                                const uint8_t b_limit_in,
                                                const uint8_t limit_in,
                                                const uint8_t thresh_in) {
  uint8_t *temp_src;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;

  b_limit = (v16u8)__msa_fill_b(b_limit_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  thresh = (v16u8)__msa_fill_b(thresh_in);
  temp_src = src - (pitch << 2);
  LD_UB8(temp_src, pitch, p3, p2, p1, p0, q0, q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);
  temp_src = src - 3 * pitch;
  ST_UB4(p2, p1, p0, q0, temp_src, pitch);
  temp_src += (4 * pitch);
  ST_UB2(q1, q2, temp_src, pitch);
}

static void mbloop_filter_horizontal_edge_uv_msa(uint8_t *src_u, uint8_t *src_v,
                                                 int32_t pitch,
                                                 const uint8_t b_limit_in,
                                                 const uint8_t limit_in,
                                                 const uint8_t thresh_in) {
  uint8_t *temp_src;
  uint64_t p2_d, p1_d, p0_d, q0_d, q1_d, q2_d;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;
  v16u8 p3_u, p2_u, p1_u, p0_u, q3_u, q2_u, q1_u, q0_u;
  v16u8 p3_v, p2_v, p1_v, p0_v, q3_v, q2_v, q1_v, q0_v;

  b_limit = (v16u8)__msa_fill_b(b_limit_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  thresh = (v16u8)__msa_fill_b(thresh_in);

  temp_src = src_u - (pitch << 2);
  LD_UB8(temp_src, pitch, p3_u, p2_u, p1_u, p0_u, q0_u, q1_u, q2_u, q3_u);
  temp_src = src_v - (pitch << 2);
  LD_UB8(temp_src, pitch, p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);

  ILVR_D4_UB(p3_v, p3_u, p2_v, p2_u, p1_v, p1_u, p0_v, p0_u, p3, p2, p1, p0);
  ILVR_D4_UB(q0_v, q0_u, q1_v, q1_u, q2_v, q2_u, q3_v, q3_u, q0, q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);

  p2_d = __msa_copy_u_d((v2i64)p2, 0);
  p1_d = __msa_copy_u_d((v2i64)p1, 0);
  p0_d = __msa_copy_u_d((v2i64)p0, 0);
  q0_d = __msa_copy_u_d((v2i64)q0, 0);
  q1_d = __msa_copy_u_d((v2i64)q1, 0);
  q2_d = __msa_copy_u_d((v2i64)q2, 0);
  src_u -= (pitch * 3);
  SD4(p2_d, p1_d, p0_d, q0_d, src_u, pitch);
  src_u += 4 * pitch;
  SD(q1_d, src_u);
  src_u += pitch;
  SD(q2_d, src_u);

  p2_d = __msa_copy_u_d((v2i64)p2, 1);
  p1_d = __msa_copy_u_d((v2i64)p1, 1);
  p0_d = __msa_copy_u_d((v2i64)p0, 1);
  q0_d = __msa_copy_u_d((v2i64)q0, 1);
  q1_d = __msa_copy_u_d((v2i64)q1, 1);
  q2_d = __msa_copy_u_d((v2i64)q2, 1);
  src_v -= (pitch * 3);
  SD4(p2_d, p1_d, p0_d, q0_d, src_v, pitch);
  src_v += 4 * pitch;
  SD(q1_d, src_v);
  src_v += pitch;
  SD(q2_d, src_v);
}

static void mbloop_filter_vertical_edge_y_msa(uint8_t *src, int32_t pitch,
                                              const uint8_t b_limit_in,
                                              const uint8_t limit_in,
                                              const uint8_t thresh_in) {
  uint8_t *temp_src;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7, row8;
  v16u8 row9, row10, row11, row12, row13, row14, row15;
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  b_limit = (v16u8)__msa_fill_b(b_limit_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  thresh = (v16u8)__msa_fill_b(thresh_in);
  temp_src = src - 4;
  LD_UB8(temp_src, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  temp_src += (8 * pitch);
  LD_UB8(temp_src, pitch, row8, row9, row10, row11, row12, row13, row14, row15);
  TRANSPOSE16x8_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);
  ILVR_B2_SH(p1, p2, q0, p0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp3, tmp4);
  ILVL_B2_SH(p1, p2, q0, p0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp6, tmp7);
  ILVRL_B2_SH(q2, q1, tmp2, tmp5);

  temp_src = src - 3;
  VP8_ST6x1_UB(tmp3, 0, tmp2, 0, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp3, 1, tmp2, 1, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp3, 2, tmp2, 2, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp3, 3, tmp2, 3, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp4, 0, tmp2, 4, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp4, 1, tmp2, 5, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp4, 2, tmp2, 6, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp4, 3, tmp2, 7, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp6, 0, tmp5, 0, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp6, 1, tmp5, 1, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp6, 2, tmp5, 2, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp6, 3, tmp5, 3, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp7, 0, tmp5, 4, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp7, 1, tmp5, 5, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp7, 2, tmp5, 6, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_UB(tmp7, 3, tmp5, 7, temp_src, 4);
}

static void mbloop_filter_vertical_edge_uv_msa(uint8_t *src_u, uint8_t *src_v,
                                               int32_t pitch,
                                               const uint8_t b_limit_in,
                                               const uint8_t limit_in,
                                               const uint8_t thresh_in) {
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7, row8;
  v16u8 row9, row10, row11, row12, row13, row14, row15;
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  b_limit = (v16u8)__msa_fill_b(b_limit_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  thresh = (v16u8)__msa_fill_b(thresh_in);

  LD_UB8(src_u - 4, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  LD_UB8(src_v - 4, pitch, row8, row9, row10, row11, row12, row13, row14,
         row15);
  TRANSPOSE16x8_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);

  ILVR_B2_SH(p1, p2, q0, p0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp3, tmp4);
  ILVL_B2_SH(p1, p2, q0, p0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp6, tmp7);
  ILVRL_B2_SH(q2, q1, tmp2, tmp5);

  src_u -= 3;
  VP8_ST6x1_UB(tmp3, 0, tmp2, 0, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp3, 1, tmp2, 1, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp3, 2, tmp2, 2, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp3, 3, tmp2, 3, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp4, 0, tmp2, 4, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp4, 1, tmp2, 5, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp4, 2, tmp2, 6, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_UB(tmp4, 3, tmp2, 7, src_u, 4);

  src_v -= 3;
  VP8_ST6x1_UB(tmp6, 0, tmp5, 0, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp6, 1, tmp5, 1, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp6, 2, tmp5, 2, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp6, 3, tmp5, 3, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp7, 0, tmp5, 4, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp7, 1, tmp5, 5, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp7, 2, tmp5, 6, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_UB(tmp7, 3, tmp5, 7, src_v, 4);
}

void vp8_loop_filter_simple_horizontal_edge_msa(uint8_t *src, int32_t pitch,
                                                const uint8_t *b_limit_ptr) {
  v16u8 p1, p0, q1, q0;
  v16u8 mask, b_limit;

  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  LD_UB4(src - (pitch << 1), pitch, p1, p0, q0, q1);
  VP8_SIMPLE_MASK(p1, p0, q0, q1, b_limit, mask);
  VP8_SIMPLE_FILT(p1, p0, q0, q1, mask);
  ST_UB2(p0, q0, (src - pitch), pitch);
}

void vp8_loop_filter_simple_vertical_edge_msa(uint8_t *src, int32_t pitch,
                                              const uint8_t *b_limit_ptr) {
  uint8_t *temp_src;
  v16u8 p1, p0, q1, q0;
  v16u8 mask, b_limit;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7, row8;
  v16u8 row9, row10, row11, row12, row13, row14, row15;
  v8i16 tmp0, tmp1;

  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  temp_src = src - 2;
  LD_UB8(temp_src, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  temp_src += (8 * pitch);
  LD_UB8(temp_src, pitch, row8, row9, row10, row11, row12, row13, row14, row15);
  TRANSPOSE16x4_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p1, p0,
                      q0, q1);
  VP8_SIMPLE_MASK(p1, p0, q0, q1, b_limit, mask);
  VP8_SIMPLE_FILT(p1, p0, q0, q1, mask);
  ILVRL_B2_SH(q0, p0, tmp1, tmp0);

  src -= 1;
  ST2x4_UB(tmp1, 0, src, pitch);
  src += 4 * pitch;
  ST2x4_UB(tmp1, 4, src, pitch);
  src += 4 * pitch;
  ST2x4_UB(tmp0, 0, src, pitch);
  src += 4 * pitch;
  ST2x4_UB(tmp0, 4, src, pitch);
  src += 4 * pitch;
}

static void loop_filter_horizontal_edge_uv_msa(uint8_t *src_u, uint8_t *src_v,
                                               int32_t pitch,
                                               const uint8_t b_limit_in,
                                               const uint8_t limit_in,
                                               const uint8_t thresh_in) {
  uint64_t p1_d, p0_d, q0_d, q1_d;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;
  v16u8 p3_u, p2_u, p1_u, p0_u, q3_u, q2_u, q1_u, q0_u;
  v16u8 p3_v, p2_v, p1_v, p0_v, q3_v, q2_v, q1_v, q0_v;

  thresh = (v16u8)__msa_fill_b(thresh_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  b_limit = (v16u8)__msa_fill_b(b_limit_in);

  src_u = src_u - (pitch << 2);
  LD_UB8(src_u, pitch, p3_u, p2_u, p1_u, p0_u, q0_u, q1_u, q2_u, q3_u);
  src_u += (5 * pitch);
  src_v = src_v - (pitch << 2);
  LD_UB8(src_v, pitch, p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);
  src_v += (5 * pitch);

  /* right 8 element of p3 are u pixel and
     left 8 element of p3 are v pixel */
  ILVR_D4_UB(p3_v, p3_u, p2_v, p2_u, p1_v, p1_u, p0_v, p0_u, p3, p2, p1, p0);
  ILVR_D4_UB(q0_v, q0_u, q1_v, q1_u, q2_v, q2_u, q3_v, q3_u, q0, q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  p1_d = __msa_copy_u_d((v2i64)p1, 0);
  p0_d = __msa_copy_u_d((v2i64)p0, 0);
  q0_d = __msa_copy_u_d((v2i64)q0, 0);
  q1_d = __msa_copy_u_d((v2i64)q1, 0);
  SD4(q1_d, q0_d, p0_d, p1_d, src_u, (-pitch));

  p1_d = __msa_copy_u_d((v2i64)p1, 1);
  p0_d = __msa_copy_u_d((v2i64)p0, 1);
  q0_d = __msa_copy_u_d((v2i64)q0, 1);
  q1_d = __msa_copy_u_d((v2i64)q1, 1);
  SD4(q1_d, q0_d, p0_d, p1_d, src_v, (-pitch));
}

static void loop_filter_vertical_edge_uv_msa(uint8_t *src_u, uint8_t *src_v,
                                             int32_t pitch,
                                             const uint8_t b_limit_in,
                                             const uint8_t limit_in,
                                             const uint8_t thresh_in) {
  uint8_t *temp_src_u, *temp_src_v;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 mask, hev, flat, thresh, limit, b_limit;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7, row8;
  v16u8 row9, row10, row11, row12, row13, row14, row15;
  v4i32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

  thresh = (v16u8)__msa_fill_b(thresh_in);
  limit = (v16u8)__msa_fill_b(limit_in);
  b_limit = (v16u8)__msa_fill_b(b_limit_in);

  LD_UB8(src_u - 4, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  LD_UB8(src_v - 4, pitch, row8, row9, row10, row11, row12, row13, row14,
         row15);
  TRANSPOSE16x8_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);
  ILVR_B2_SW(p0, p1, q1, q0, tmp0, tmp1);
  ILVRL_H2_SW(tmp1, tmp0, tmp2, tmp3);
  tmp0 = (v4i32)__msa_ilvl_b((v16i8)p0, (v16i8)p1);
  tmp1 = (v4i32)__msa_ilvl_b((v16i8)q1, (v16i8)q0);
  ILVRL_H2_SW(tmp1, tmp0, tmp4, tmp5);

  temp_src_u = src_u - 2;
  ST4x4_UB(tmp2, tmp2, 0, 1, 2, 3, temp_src_u, pitch);
  temp_src_u += 4 * pitch;
  ST4x4_UB(tmp3, tmp3, 0, 1, 2, 3, temp_src_u, pitch);

  temp_src_v = src_v - 2;
  ST4x4_UB(tmp4, tmp4, 0, 1, 2, 3, temp_src_v, pitch);
  temp_src_v += 4 * pitch;
  ST4x4_UB(tmp5, tmp5, 0, 1, 2, 3, temp_src_v, pitch);
}

void vp8_loop_filter_mbh_msa(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                             int32_t pitch_y, int32_t pitch_u_v,
                             loop_filter_info *lpf_info_ptr) {
  mbloop_filter_horizontal_edge_y_msa(src_y, pitch_y, *lpf_info_ptr->mblim,
                                      *lpf_info_ptr->lim,
                                      *lpf_info_ptr->hev_thr);
  if (src_u) {
    mbloop_filter_horizontal_edge_uv_msa(
        src_u, src_v, pitch_u_v, *lpf_info_ptr->mblim, *lpf_info_ptr->lim,
        *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_mbv_msa(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                             int32_t pitch_y, int32_t pitch_u_v,
                             loop_filter_info *lpf_info_ptr) {
  mbloop_filter_vertical_edge_y_msa(src_y, pitch_y, *lpf_info_ptr->mblim,
                                    *lpf_info_ptr->lim, *lpf_info_ptr->hev_thr);
  if (src_u) {
    mbloop_filter_vertical_edge_uv_msa(src_u, src_v, pitch_u_v,
                                       *lpf_info_ptr->mblim, *lpf_info_ptr->lim,
                                       *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_bh_msa(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                            int32_t pitch_y, int32_t pitch_u_v,
                            loop_filter_info *lpf_info_ptr) {
  loop_filter_horizontal_4_dual_msa(src_y + 4 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  loop_filter_horizontal_4_dual_msa(src_y + 8 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  loop_filter_horizontal_4_dual_msa(src_y + 12 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  if (src_u) {
    loop_filter_horizontal_edge_uv_msa(
        src_u + (4 * pitch_u_v), src_v + (4 * pitch_u_v), pitch_u_v,
        *lpf_info_ptr->blim, *lpf_info_ptr->lim, *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_bv_msa(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                            int32_t pitch_y, int32_t pitch_u_v,
                            loop_filter_info *lpf_info_ptr) {
  loop_filter_vertical_4_dual_msa(src_y + 4, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  loop_filter_vertical_4_dual_msa(src_y + 8, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  loop_filter_vertical_4_dual_msa(src_y + 12, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  if (src_u) {
    loop_filter_vertical_edge_uv_msa(src_u + 4, src_v + 4, pitch_u_v,
                                     *lpf_info_ptr->blim, *lpf_info_ptr->lim,
                                     *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_bhs_msa(uint8_t *src_y, int32_t pitch_y,
                             const uint8_t *b_limit_ptr) {
  vp8_loop_filter_simple_horizontal_edge_msa(src_y + (4 * pitch_y), pitch_y,
                                             b_limit_ptr);
  vp8_loop_filter_simple_horizontal_edge_msa(src_y + (8 * pitch_y), pitch_y,
                                             b_limit_ptr);
  vp8_loop_filter_simple_horizontal_edge_msa(src_y + (12 * pitch_y), pitch_y,
                                             b_limit_ptr);
}

void vp8_loop_filter_bvs_msa(uint8_t *src_y, int32_t pitch_y,
                             const uint8_t *b_limit_ptr) {
  vp8_loop_filter_simple_vertical_edge_msa(src_y + 4, pitch_y, b_limit_ptr);
  vp8_loop_filter_simple_vertical_edge_msa(src_y + 8, pitch_y, b_limit_ptr);
  vp8_loop_filter_simple_vertical_edge_msa(src_y + 12, pitch_y, b_limit_ptr);
}
