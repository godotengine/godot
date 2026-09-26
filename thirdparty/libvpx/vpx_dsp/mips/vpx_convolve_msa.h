/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_VPX_CONVOLVE_MSA_H_
#define VPX_VPX_DSP_MIPS_VPX_CONVOLVE_MSA_H_

#include "vpx_dsp/mips/macros_msa.h"
#include "vpx_dsp/vpx_filter.h"

extern const uint8_t mc_filt_mask_arr[16 * 3];

#define FILT_8TAP_DPADD_S_H(vec0, vec1, vec2, vec3, filt0, filt1, filt2,   \
                            filt3)                                         \
  ({                                                                       \
    v8i16 tmp_dpadd_0, tmp_dpadd_1;                                        \
                                                                           \
    tmp_dpadd_0 = __msa_dotp_s_h((v16i8)vec0, (v16i8)filt0);               \
    tmp_dpadd_0 = __msa_dpadd_s_h(tmp_dpadd_0, (v16i8)vec1, (v16i8)filt1); \
    tmp_dpadd_1 = __msa_dotp_s_h((v16i8)vec2, (v16i8)filt2);               \
    tmp_dpadd_1 = __msa_dpadd_s_h(tmp_dpadd_1, (v16i8)vec3, (v16i8)filt3); \
    tmp_dpadd_0 = __msa_adds_s_h(tmp_dpadd_0, tmp_dpadd_1);                \
                                                                           \
    tmp_dpadd_0;                                                           \
  })

#define HORIZ_8TAP_FILT(src0, src1, mask0, mask1, mask2, mask3, filt_h0,       \
                        filt_h1, filt_h2, filt_h3)                             \
  ({                                                                           \
    v16i8 vec0_m, vec1_m, vec2_m, vec3_m;                                      \
    v8i16 hz_out_m;                                                            \
                                                                               \
    VSHF_B4_SB(src0, src1, mask0, mask1, mask2, mask3, vec0_m, vec1_m, vec2_m, \
               vec3_m);                                                        \
    hz_out_m = FILT_8TAP_DPADD_S_H(vec0_m, vec1_m, vec2_m, vec3_m, filt_h0,    \
                                   filt_h1, filt_h2, filt_h3);                 \
                                                                               \
    hz_out_m = __msa_srari_h(hz_out_m, FILTER_BITS);                           \
    hz_out_m = __msa_sat_s_h(hz_out_m, 7);                                     \
                                                                               \
    hz_out_m;                                                                  \
  })

#define HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,     \
                                   mask2, mask3, filt0, filt1, filt2, filt3, \
                                   out0, out1)                               \
  {                                                                          \
    v16i8 vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m;    \
    v8i16 res0_m, res1_m, res2_m, res3_m;                                    \
                                                                             \
    VSHF_B2_SB(src0, src1, src2, src3, mask0, mask0, vec0_m, vec1_m);        \
    DOTP_SB2_SH(vec0_m, vec1_m, filt0, filt0, res0_m, res1_m);               \
    VSHF_B2_SB(src0, src1, src2, src3, mask1, mask1, vec2_m, vec3_m);        \
    DPADD_SB2_SH(vec2_m, vec3_m, filt1, filt1, res0_m, res1_m);              \
    VSHF_B2_SB(src0, src1, src2, src3, mask2, mask2, vec4_m, vec5_m);        \
    DOTP_SB2_SH(vec4_m, vec5_m, filt2, filt2, res2_m, res3_m);               \
    VSHF_B2_SB(src0, src1, src2, src3, mask3, mask3, vec6_m, vec7_m);        \
    DPADD_SB2_SH(vec6_m, vec7_m, filt3, filt3, res2_m, res3_m);              \
    ADDS_SH2_SH(res0_m, res2_m, res1_m, res3_m, out0, out1);                 \
  }

#define HORIZ_8TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,     \
                                   mask2, mask3, filt0, filt1, filt2, filt3, \
                                   out0, out1, out2, out3)                   \
  {                                                                          \
    v16i8 vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m;    \
    v8i16 res0_m, res1_m, res2_m, res3_m, res4_m, res5_m, res6_m, res7_m;    \
                                                                             \
    VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, vec0_m, vec1_m);        \
    VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, vec2_m, vec3_m);        \
    DOTP_SB4_SH(vec0_m, vec1_m, vec2_m, vec3_m, filt0, filt0, filt0, filt0,  \
                res0_m, res1_m, res2_m, res3_m);                             \
    VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, vec0_m, vec1_m);        \
    VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, vec2_m, vec3_m);        \
    DOTP_SB4_SH(vec0_m, vec1_m, vec2_m, vec3_m, filt2, filt2, filt2, filt2,  \
                res4_m, res5_m, res6_m, res7_m);                             \
    VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, vec4_m, vec5_m);        \
    VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, vec6_m, vec7_m);        \
    DPADD_SB4_SH(vec4_m, vec5_m, vec6_m, vec7_m, filt1, filt1, filt1, filt1, \
                 res0_m, res1_m, res2_m, res3_m);                            \
    VSHF_B2_SB(src0, src0, src1, src1, mask3, mask3, vec4_m, vec5_m);        \
    VSHF_B2_SB(src2, src2, src3, src3, mask3, mask3, vec6_m, vec7_m);        \
    DPADD_SB4_SH(vec4_m, vec5_m, vec6_m, vec7_m, filt3, filt3, filt3, filt3, \
                 res4_m, res5_m, res6_m, res7_m);                            \
    ADDS_SH4_SH(res0_m, res4_m, res1_m, res5_m, res2_m, res6_m, res3_m,      \
                res7_m, out0, out1, out2, out3);                             \
  }

#define PCKEV_XORI128_AVG_ST_UB(in0, in1, dst, pdst) \
  {                                                  \
    v16u8 tmp_m;                                     \
                                                     \
    tmp_m = PCKEV_XORI128_UB(in1, in0);              \
    tmp_m = __msa_aver_u_b(tmp_m, (v16u8)dst);       \
    ST_UB(tmp_m, (pdst));                            \
  }

#define PCKEV_AVG_ST_UB(in0, in1, dst, pdst)              \
  {                                                       \
    v16u8 tmp_m;                                          \
                                                          \
    tmp_m = (v16u8)__msa_pckev_b((v16i8)in0, (v16i8)in1); \
    tmp_m = __msa_aver_u_b(tmp_m, (v16u8)dst);            \
    ST_UB(tmp_m, (pdst));                                 \
  }

#define PCKEV_AVG_ST8x4_UB(in0, in1, in2, in3, dst0, dst1, pdst, stride) \
  {                                                                      \
    v16u8 tmp0_m, tmp1_m;                                                \
    uint8_t *pdst_m = (uint8_t *)(pdst);                                 \
                                                                         \
    PCKEV_B2_UB(in1, in0, in3, in2, tmp0_m, tmp1_m);                     \
    AVER_UB2_UB(tmp0_m, dst0, tmp1_m, dst1, tmp0_m, tmp1_m);             \
    ST8x4_UB(tmp0_m, tmp1_m, pdst_m, stride);                            \
  }
#endif  // VPX_VPX_DSP_MIPS_VPX_CONVOLVE_MSA_H_
