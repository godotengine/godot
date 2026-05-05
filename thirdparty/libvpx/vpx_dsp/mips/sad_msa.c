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

#define SAD_INSVE_W4(RTYPE, in0, in1, in2, in3, out)       \
  {                                                        \
    out = (RTYPE)__msa_insve_w((v4i32)out, 0, (v4i32)in0); \
    out = (RTYPE)__msa_insve_w((v4i32)out, 1, (v4i32)in1); \
    out = (RTYPE)__msa_insve_w((v4i32)out, 2, (v4i32)in2); \
    out = (RTYPE)__msa_insve_w((v4i32)out, 3, (v4i32)in3); \
  }
#define SAD_INSVE_W4_UB(...) SAD_INSVE_W4(v16u8, __VA_ARGS__)

static uint32_t sad_4width_msa(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *ref_ptr, int32_t ref_stride,
                               int32_t height) {
  int32_t ht_cnt;
  uint32_t src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  v16u8 src = { 0 };
  v16u8 ref = { 0 };
  v16u8 diff;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);

    INSERT_W4_UB(src0, src1, src2, src3, src);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);

    diff = __msa_asub_u_b(src, ref);
    sad += __msa_hadd_u_h(diff, diff);
  }

  return HADD_UH_U32(sad);
}

static uint32_t sad_8width_msa(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(ref, ref_stride, ref0, ref1, ref2, ref3);
    ref += (4 * ref_stride);

    PCKEV_D4_UB(src1, src0, src3, src2, ref1, ref0, ref3, ref2, src0, src1,
                ref0, ref1);
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t sad_16width_msa(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(src, src_stride, src0, src1);
    src += (2 * src_stride);
    LD_UB2(ref, ref_stride, ref0, ref1);
    ref += (2 * ref_stride);
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(src, src_stride, src0, src1);
    src += (2 * src_stride);
    LD_UB2(ref, ref_stride, ref0, ref1);
    ref += (2 * ref_stride);
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t sad_32width_msa(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB2(src, 16, src0, src1);
    src += src_stride;
    LD_UB2(ref, 16, ref0, ref1);
    ref += ref_stride;
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(src, 16, src0, src1);
    src += src_stride;
    LD_UB2(ref, 16, ref0, ref1);
    ref += ref_stride;
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(src, 16, src0, src1);
    src += src_stride;
    LD_UB2(ref, 16, ref0, ref1);
    ref += ref_stride;
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(src, 16, src0, src1);
    src += src_stride;
    LD_UB2(ref, 16, ref0, ref1);
    ref += ref_stride;
    sad += SAD_UB2_UH(src0, src1, ref0, ref1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t sad_64width_msa(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt;
  uint32_t sad = 0;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };

  for (ht_cnt = (height >> 1); ht_cnt--;) {
    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    sad0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad1 += SAD_UB2_UH(src2, src3, ref2, ref3);

    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    sad0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad1 += SAD_UB2_UH(src2, src3, ref2, ref3);
  }

  sad = HADD_UH_U32(sad0);
  sad += HADD_UH_U32(sad1);

  return sad;
}

static void sad_4width_x4d_msa(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *const aref_ptr[],
                               int32_t ref_stride, int32_t height,
                               uint32_t *sad_array) {
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  int32_t ht_cnt;
  uint32_t src0, src1, src2, src3;
  uint32_t ref0, ref1, ref2, ref3;
  v16u8 src = { 0 };
  v16u8 ref = { 0 };
  v16u8 diff;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };
  v8u16 sad2 = { 0 };
  v8u16 sad3 = { 0 };

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    INSERT_W4_UB(src0, src1, src2, src3, src);
    src_ptr += (4 * src_stride);

    LW4(ref0_ptr, ref_stride, ref0, ref1, ref2, ref3);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ref0_ptr += (4 * ref_stride);

    diff = __msa_asub_u_b(src, ref);
    sad0 += __msa_hadd_u_h(diff, diff);

    LW4(ref1_ptr, ref_stride, ref0, ref1, ref2, ref3);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ref1_ptr += (4 * ref_stride);

    diff = __msa_asub_u_b(src, ref);
    sad1 += __msa_hadd_u_h(diff, diff);

    LW4(ref2_ptr, ref_stride, ref0, ref1, ref2, ref3);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ref2_ptr += (4 * ref_stride);

    diff = __msa_asub_u_b(src, ref);
    sad2 += __msa_hadd_u_h(diff, diff);

    LW4(ref3_ptr, ref_stride, ref0, ref1, ref2, ref3);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);
    ref3_ptr += (4 * ref_stride);

    diff = __msa_asub_u_b(src, ref);
    sad3 += __msa_hadd_u_h(diff, diff);
  }

  sad_array[0] = HADD_UH_U32(sad0);
  sad_array[1] = HADD_UH_U32(sad1);
  sad_array[2] = HADD_UH_U32(sad2);
  sad_array[3] = HADD_UH_U32(sad3);
}

static void sad_8width_x4d_msa(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *const aref_ptr[],
                               int32_t ref_stride, int32_t height,
                               uint32_t *sad_array) {
  int32_t ht_cnt;
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7;
  v16u8 ref8, ref9, ref10, ref11, ref12, ref13, ref14, ref15;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };
  v8u16 sad2 = { 0 };
  v8u16 sad3 = { 0 };

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LD_UB4(ref0_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref0_ptr += (4 * ref_stride);
    LD_UB4(ref1_ptr, ref_stride, ref4, ref5, ref6, ref7);
    ref1_ptr += (4 * ref_stride);
    LD_UB4(ref2_ptr, ref_stride, ref8, ref9, ref10, ref11);
    ref2_ptr += (4 * ref_stride);
    LD_UB4(ref3_ptr, ref_stride, ref12, ref13, ref14, ref15);
    ref3_ptr += (4 * ref_stride);

    PCKEV_D2_UB(src1, src0, src3, src2, src0, src1);
    PCKEV_D2_UB(ref1, ref0, ref3, ref2, ref0, ref1);
    sad0 += SAD_UB2_UH(src0, src1, ref0, ref1);

    PCKEV_D2_UB(ref5, ref4, ref7, ref6, ref0, ref1);
    sad1 += SAD_UB2_UH(src0, src1, ref0, ref1);

    PCKEV_D2_UB(ref9, ref8, ref11, ref10, ref0, ref1);
    sad2 += SAD_UB2_UH(src0, src1, ref0, ref1);

    PCKEV_D2_UB(ref13, ref12, ref15, ref14, ref0, ref1);
    sad3 += SAD_UB2_UH(src0, src1, ref0, ref1);
  }

  sad_array[0] = HADD_UH_U32(sad0);
  sad_array[1] = HADD_UH_U32(sad1);
  sad_array[2] = HADD_UH_U32(sad2);
  sad_array[3] = HADD_UH_U32(sad3);
}

static void sad_16width_x4d_msa(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  int32_t ht_cnt;
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  v16u8 src, ref0, ref1, ref2, ref3, diff;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };
  v8u16 sad2 = { 0 };
  v8u16 sad3 = { 0 };

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (ht_cnt = (height >> 1); ht_cnt--;) {
    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref0 = LD_UB(ref0_ptr);
    ref0_ptr += ref_stride;
    ref1 = LD_UB(ref1_ptr);
    ref1_ptr += ref_stride;
    ref2 = LD_UB(ref2_ptr);
    ref2_ptr += ref_stride;
    ref3 = LD_UB(ref3_ptr);
    ref3_ptr += ref_stride;

    diff = __msa_asub_u_b(src, ref0);
    sad0 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref1);
    sad1 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref2);
    sad2 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref3);
    sad3 += __msa_hadd_u_h(diff, diff);

    src = LD_UB(src_ptr);
    src_ptr += src_stride;
    ref0 = LD_UB(ref0_ptr);
    ref0_ptr += ref_stride;
    ref1 = LD_UB(ref1_ptr);
    ref1_ptr += ref_stride;
    ref2 = LD_UB(ref2_ptr);
    ref2_ptr += ref_stride;
    ref3 = LD_UB(ref3_ptr);
    ref3_ptr += ref_stride;

    diff = __msa_asub_u_b(src, ref0);
    sad0 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref1);
    sad1 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref2);
    sad2 += __msa_hadd_u_h(diff, diff);
    diff = __msa_asub_u_b(src, ref3);
    sad3 += __msa_hadd_u_h(diff, diff);
  }

  sad_array[0] = HADD_UH_U32(sad0);
  sad_array[1] = HADD_UH_U32(sad1);
  sad_array[2] = HADD_UH_U32(sad2);
  sad_array[3] = HADD_UH_U32(sad3);
}

static void sad_32width_x4d_msa(const uint8_t *src, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  int32_t ht_cnt;
  v16u8 src0, src1, ref0, ref1;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };
  v8u16 sad2 = { 0 };
  v8u16 sad3 = { 0 };

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (ht_cnt = height; ht_cnt--;) {
    LD_UB2(src, 16, src0, src1);
    src += src_stride;

    LD_UB2(ref0_ptr, 16, ref0, ref1);
    ref0_ptr += ref_stride;
    sad0 += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(ref1_ptr, 16, ref0, ref1);
    ref1_ptr += ref_stride;
    sad1 += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(ref2_ptr, 16, ref0, ref1);
    ref2_ptr += ref_stride;
    sad2 += SAD_UB2_UH(src0, src1, ref0, ref1);

    LD_UB2(ref3_ptr, 16, ref0, ref1);
    ref3_ptr += ref_stride;
    sad3 += SAD_UB2_UH(src0, src1, ref0, ref1);
  }

  sad_array[0] = HADD_UH_U32(sad0);
  sad_array[1] = HADD_UH_U32(sad1);
  sad_array[2] = HADD_UH_U32(sad2);
  sad_array[3] = HADD_UH_U32(sad3);
}

static void sad_64width_x4d_msa(const uint8_t *src, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v8u16 sad0_0 = { 0 };
  v8u16 sad0_1 = { 0 };
  v8u16 sad1_0 = { 0 };
  v8u16 sad1_1 = { 0 };
  v8u16 sad2_0 = { 0 };
  v8u16 sad2_1 = { 0 };
  v8u16 sad3_0 = { 0 };
  v8u16 sad3_1 = { 0 };
  v4u32 sad;

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (ht_cnt = height; ht_cnt--;) {
    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;

    LD_UB4(ref0_ptr, 16, ref0, ref1, ref2, ref3);
    ref0_ptr += ref_stride;
    sad0_0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad0_1 += SAD_UB2_UH(src2, src3, ref2, ref3);

    LD_UB4(ref1_ptr, 16, ref0, ref1, ref2, ref3);
    ref1_ptr += ref_stride;
    sad1_0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad1_1 += SAD_UB2_UH(src2, src3, ref2, ref3);

    LD_UB4(ref2_ptr, 16, ref0, ref1, ref2, ref3);
    ref2_ptr += ref_stride;
    sad2_0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad2_1 += SAD_UB2_UH(src2, src3, ref2, ref3);

    LD_UB4(ref3_ptr, 16, ref0, ref1, ref2, ref3);
    ref3_ptr += ref_stride;
    sad3_0 += SAD_UB2_UH(src0, src1, ref0, ref1);
    sad3_1 += SAD_UB2_UH(src2, src3, ref2, ref3);
  }

  sad = __msa_hadd_u_w(sad0_0, sad0_0);
  sad += __msa_hadd_u_w(sad0_1, sad0_1);
  sad_array[0] = HADD_UW_U32(sad);

  sad = __msa_hadd_u_w(sad1_0, sad1_0);
  sad += __msa_hadd_u_w(sad1_1, sad1_1);
  sad_array[1] = HADD_UW_U32(sad);

  sad = __msa_hadd_u_w(sad2_0, sad2_0);
  sad += __msa_hadd_u_w(sad2_1, sad2_1);
  sad_array[2] = HADD_UW_U32(sad);

  sad = __msa_hadd_u_w(sad3_0, sad3_0);
  sad += __msa_hadd_u_w(sad3_1, sad3_1);
  sad_array[3] = HADD_UW_U32(sad);
}

static uint32_t avgsad_4width_msa(const uint8_t *src_ptr, int32_t src_stride,
                                  const uint8_t *ref_ptr, int32_t ref_stride,
                                  int32_t height, const uint8_t *sec_pred) {
  int32_t ht_cnt;
  uint32_t src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  v16u8 src = { 0 };
  v16u8 ref = { 0 };
  v16u8 diff, pred, comp;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LW4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LW4(ref_ptr, ref_stride, ref0, ref1, ref2, ref3);
    ref_ptr += (4 * ref_stride);
    pred = LD_UB(sec_pred);
    sec_pred += 16;

    INSERT_W4_UB(src0, src1, src2, src3, src);
    INSERT_W4_UB(ref0, ref1, ref2, ref3, ref);

    comp = __msa_aver_u_b(pred, ref);
    diff = __msa_asub_u_b(src, comp);
    sad += __msa_hadd_u_h(diff, diff);
  }

  return HADD_UH_U32(sad);
}

static uint32_t avgsad_8width_msa(const uint8_t *src, int32_t src_stride,
                                  const uint8_t *ref, int32_t ref_stride,
                                  int32_t height, const uint8_t *sec_pred) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  v16u8 diff0, diff1, pred0, pred1;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(ref, ref_stride, ref0, ref1, ref2, ref3);
    ref += (4 * ref_stride);
    LD_UB2(sec_pred, 16, pred0, pred1);
    sec_pred += 32;
    PCKEV_D4_UB(src1, src0, src3, src2, ref1, ref0, ref3, ref2, src0, src1,
                ref0, ref1);
    AVER_UB2_UB(pred0, ref0, pred1, ref1, diff0, diff1);
    sad += SAD_UB2_UH(src0, src1, diff0, diff1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t avgsad_16width_msa(const uint8_t *src, int32_t src_stride,
                                   const uint8_t *ref, int32_t ref_stride,
                                   int32_t height, const uint8_t *sec_pred) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  v16u8 pred0, pred1, pred2, pred3, comp0, comp1;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 3); ht_cnt--;) {
    LD_UB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(ref, ref_stride, ref0, ref1, ref2, ref3);
    ref += (4 * ref_stride);
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += (4 * 16);
    AVER_UB2_UB(pred0, ref0, pred1, ref1, comp0, comp1);
    sad += SAD_UB2_UH(src0, src1, comp0, comp1);
    AVER_UB2_UB(pred2, ref2, pred3, ref3, comp0, comp1);
    sad += SAD_UB2_UH(src2, src3, comp0, comp1);

    LD_UB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);
    LD_UB4(ref, ref_stride, ref0, ref1, ref2, ref3);
    ref += (4 * ref_stride);
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += (4 * 16);
    AVER_UB2_UB(pred0, ref0, pred1, ref1, comp0, comp1);
    sad += SAD_UB2_UH(src0, src1, comp0, comp1);
    AVER_UB2_UB(pred2, ref2, pred3, ref3, comp0, comp1);
    sad += SAD_UB2_UH(src2, src3, comp0, comp1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t avgsad_32width_msa(const uint8_t *src, int32_t src_stride,
                                   const uint8_t *ref, int32_t ref_stride,
                                   int32_t height, const uint8_t *sec_pred) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7;
  v16u8 pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  v16u8 comp0, comp1;
  v8u16 sad = { 0 };

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src, src_stride, src0, src2, src4, src6);
    LD_UB4(src + 16, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);

    LD_UB4(ref, ref_stride, ref0, ref2, ref4, ref6);
    LD_UB4(ref + 16, ref_stride, ref1, ref3, ref5, ref7);
    ref += (4 * ref_stride);

    LD_UB4(sec_pred, 32, pred0, pred2, pred4, pred6);
    LD_UB4(sec_pred + 16, 32, pred1, pred3, pred5, pred7);
    sec_pred += (4 * 32);

    AVER_UB2_UB(pred0, ref0, pred1, ref1, comp0, comp1);
    sad += SAD_UB2_UH(src0, src1, comp0, comp1);
    AVER_UB2_UB(pred2, ref2, pred3, ref3, comp0, comp1);
    sad += SAD_UB2_UH(src2, src3, comp0, comp1);
    AVER_UB2_UB(pred4, ref4, pred5, ref5, comp0, comp1);
    sad += SAD_UB2_UH(src4, src5, comp0, comp1);
    AVER_UB2_UB(pred6, ref6, pred7, ref7, comp0, comp1);
    sad += SAD_UB2_UH(src6, src7, comp0, comp1);
  }

  return HADD_UH_U32(sad);
}

static uint32_t avgsad_64width_msa(const uint8_t *src, int32_t src_stride,
                                   const uint8_t *ref, int32_t ref_stride,
                                   int32_t height, const uint8_t *sec_pred) {
  int32_t ht_cnt;
  v16u8 src0, src1, src2, src3;
  v16u8 ref0, ref1, ref2, ref3;
  v16u8 comp0, comp1, comp2, comp3;
  v16u8 pred0, pred1, pred2, pred3;
  v8u16 sad0 = { 0 };
  v8u16 sad1 = { 0 };
  v4u32 sad;

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    AVER_UB4_UB(pred0, ref0, pred1, ref1, pred2, ref2, pred3, ref3, comp0,
                comp1, comp2, comp3);
    sad0 += SAD_UB2_UH(src0, src1, comp0, comp1);
    sad1 += SAD_UB2_UH(src2, src3, comp2, comp3);

    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    AVER_UB4_UB(pred0, ref0, pred1, ref1, pred2, ref2, pred3, ref3, comp0,
                comp1, comp2, comp3);
    sad0 += SAD_UB2_UH(src0, src1, comp0, comp1);
    sad1 += SAD_UB2_UH(src2, src3, comp2, comp3);

    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    AVER_UB4_UB(pred0, ref0, pred1, ref1, pred2, ref2, pred3, ref3, comp0,
                comp1, comp2, comp3);
    sad0 += SAD_UB2_UH(src0, src1, comp0, comp1);
    sad1 += SAD_UB2_UH(src2, src3, comp2, comp3);

    LD_UB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_UB4(ref, 16, ref0, ref1, ref2, ref3);
    ref += ref_stride;
    LD_UB4(sec_pred, 16, pred0, pred1, pred2, pred3);
    sec_pred += 64;
    AVER_UB4_UB(pred0, ref0, pred1, ref1, pred2, ref2, pred3, ref3, comp0,
                comp1, comp2, comp3);
    sad0 += SAD_UB2_UH(src0, src1, comp0, comp1);
    sad1 += SAD_UB2_UH(src2, src3, comp2, comp3);
  }

  sad = __msa_hadd_u_w(sad0, sad0);
  sad += __msa_hadd_u_w(sad1, sad1);

  return HADD_SW_S32(sad);
}

#define VPX_SAD_4xHEIGHT_MSA(height)                                         \
  uint32_t vpx_sad4x##height##_msa(const uint8_t *src, int32_t src_stride,   \
                                   const uint8_t *ref, int32_t ref_stride) { \
    return sad_4width_msa(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_8xHEIGHT_MSA(height)                                         \
  uint32_t vpx_sad8x##height##_msa(const uint8_t *src, int32_t src_stride,   \
                                   const uint8_t *ref, int32_t ref_stride) { \
    return sad_8width_msa(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_16xHEIGHT_MSA(height)                                         \
  uint32_t vpx_sad16x##height##_msa(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_16width_msa(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_32xHEIGHT_MSA(height)                                         \
  uint32_t vpx_sad32x##height##_msa(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_32width_msa(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_64xHEIGHT_MSA(height)                                         \
  uint32_t vpx_sad64x##height##_msa(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_64width_msa(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_4xHEIGHTx4D_MSA(height)                                   \
  void vpx_sad4x##height##x4d_msa(const uint8_t *src, int32_t src_stride, \
                                  const uint8_t *const refs[4],           \
                                  int32_t ref_stride, uint32_t sads[4]) { \
    sad_4width_x4d_msa(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_8xHEIGHTx4D_MSA(height)                                   \
  void vpx_sad8x##height##x4d_msa(const uint8_t *src, int32_t src_stride, \
                                  const uint8_t *const refs[4],           \
                                  int32_t ref_stride, uint32_t sads[4]) { \
    sad_8width_x4d_msa(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_16xHEIGHTx4D_MSA(height)                                   \
  void vpx_sad16x##height##x4d_msa(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[4],           \
                                   int32_t ref_stride, uint32_t sads[4]) { \
    sad_16width_x4d_msa(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_32xHEIGHTx4D_MSA(height)                                   \
  void vpx_sad32x##height##x4d_msa(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[4],           \
                                   int32_t ref_stride, uint32_t sads[4]) { \
    sad_32width_x4d_msa(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_64xHEIGHTx4D_MSA(height)                                   \
  void vpx_sad64x##height##x4d_msa(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[4],           \
                                   int32_t ref_stride, uint32_t sads[4]) { \
    sad_64width_x4d_msa(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_AVGSAD_4xHEIGHT_MSA(height)                                        \
  uint32_t vpx_sad4x##height##_avg_msa(const uint8_t *src, int32_t src_stride, \
                                       const uint8_t *ref, int32_t ref_stride, \
                                       const uint8_t *second_pred) {           \
    return avgsad_4width_msa(src, src_stride, ref, ref_stride, height,         \
                             second_pred);                                     \
  }

#define VPX_AVGSAD_8xHEIGHT_MSA(height)                                        \
  uint32_t vpx_sad8x##height##_avg_msa(const uint8_t *src, int32_t src_stride, \
                                       const uint8_t *ref, int32_t ref_stride, \
                                       const uint8_t *second_pred) {           \
    return avgsad_8width_msa(src, src_stride, ref, ref_stride, height,         \
                             second_pred);                                     \
  }

#define VPX_AVGSAD_16xHEIGHT_MSA(height)                                \
  uint32_t vpx_sad16x##height##_avg_msa(                                \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,       \
      int32_t ref_stride, const uint8_t *second_pred) {                 \
    return avgsad_16width_msa(src, src_stride, ref, ref_stride, height, \
                              second_pred);                             \
  }

#define VPX_AVGSAD_32xHEIGHT_MSA(height)                                \
  uint32_t vpx_sad32x##height##_avg_msa(                                \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,       \
      int32_t ref_stride, const uint8_t *second_pred) {                 \
    return avgsad_32width_msa(src, src_stride, ref, ref_stride, height, \
                              second_pred);                             \
  }

#define VPX_AVGSAD_64xHEIGHT_MSA(height)                                \
  uint32_t vpx_sad64x##height##_avg_msa(                                \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,       \
      int32_t ref_stride, const uint8_t *second_pred) {                 \
    return avgsad_64width_msa(src, src_stride, ref, ref_stride, height, \
                              second_pred);                             \
  }

// 64x64
VPX_SAD_64xHEIGHT_MSA(64);
VPX_SAD_64xHEIGHTx4D_MSA(64);
VPX_AVGSAD_64xHEIGHT_MSA(64);

// 64x32
VPX_SAD_64xHEIGHT_MSA(32);
VPX_SAD_64xHEIGHTx4D_MSA(32);
VPX_AVGSAD_64xHEIGHT_MSA(32);

// 32x64
VPX_SAD_32xHEIGHT_MSA(64);
VPX_SAD_32xHEIGHTx4D_MSA(64);
VPX_AVGSAD_32xHEIGHT_MSA(64);

// 32x32
VPX_SAD_32xHEIGHT_MSA(32);
VPX_SAD_32xHEIGHTx4D_MSA(32);
VPX_AVGSAD_32xHEIGHT_MSA(32);

// 32x16
VPX_SAD_32xHEIGHT_MSA(16);
VPX_SAD_32xHEIGHTx4D_MSA(16);
VPX_AVGSAD_32xHEIGHT_MSA(16);

// 16x32
VPX_SAD_16xHEIGHT_MSA(32);
VPX_SAD_16xHEIGHTx4D_MSA(32);
VPX_AVGSAD_16xHEIGHT_MSA(32);

// 16x16
VPX_SAD_16xHEIGHT_MSA(16);
VPX_SAD_16xHEIGHTx4D_MSA(16);
VPX_AVGSAD_16xHEIGHT_MSA(16);

// 16x8
VPX_SAD_16xHEIGHT_MSA(8);
VPX_SAD_16xHEIGHTx4D_MSA(8);
VPX_AVGSAD_16xHEIGHT_MSA(8);

// 8x16
VPX_SAD_8xHEIGHT_MSA(16);
VPX_SAD_8xHEIGHTx4D_MSA(16);
VPX_AVGSAD_8xHEIGHT_MSA(16);

// 8x8
VPX_SAD_8xHEIGHT_MSA(8);
VPX_SAD_8xHEIGHTx4D_MSA(8);
VPX_AVGSAD_8xHEIGHT_MSA(8);

// 8x4
VPX_SAD_8xHEIGHT_MSA(4);
VPX_SAD_8xHEIGHTx4D_MSA(4);
VPX_AVGSAD_8xHEIGHT_MSA(4);

// 4x8
VPX_SAD_4xHEIGHT_MSA(8);
VPX_SAD_4xHEIGHTx4D_MSA(8);
VPX_AVGSAD_4xHEIGHT_MSA(8);

// 4x4
VPX_SAD_4xHEIGHT_MSA(4);
VPX_SAD_4xHEIGHTx4D_MSA(4);
VPX_AVGSAD_4xHEIGHT_MSA(4);
