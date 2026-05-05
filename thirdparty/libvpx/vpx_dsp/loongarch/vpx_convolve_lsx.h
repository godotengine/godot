/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_LOONGARCH_VPX_CONVOLVE_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_VPX_CONVOLVE_LSX_H_

#include "./vpx_config.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_util/loongson_intrinsics.h"

static INLINE __m128i filt_8tap_dpadd_s_h(__m128i _reg0, __m128i _reg1,
                                          __m128i _reg2, __m128i _reg3,
                                          __m128i _filter0, __m128i _filter1,
                                          __m128i _filter2, __m128i _filter3) {
  __m128i _vec0, _vec1;

  _vec0 = __lsx_vdp2_h_b(_reg0, _filter0);
  _vec0 = __lsx_vdp2add_h_b(_vec0, _reg1, _filter1);
  _vec1 = __lsx_vdp2_h_b(_reg2, _filter2);
  _vec1 = __lsx_vdp2add_h_b(_vec1, _reg3, _filter3);
  return __lsx_vsadd_h(_vec0, _vec1);
}

static INLINE __m128i horiz_8tap_filt(__m128i _src0, __m128i _src1,
                                      __m128i _mask0, __m128i _mask1,
                                      __m128i _mask2, __m128i _mask3,
                                      __m128i _filt_h0, __m128i _filt_h1,
                                      __m128i _filt_h2, __m128i _filt_h3) {
  __m128i _tmp0, _tmp1, _tmp2, _tmp3;
  __m128i _out;

  DUP4_ARG3(__lsx_vshuf_b, _src1, _src0, _mask0, _src1, _src0, _mask1, _src1,
            _src0, _mask2, _src1, _src0, _mask3, _tmp0, _tmp1, _tmp2, _tmp3);
  _out = filt_8tap_dpadd_s_h(_tmp0, _tmp1, _tmp2, _tmp3, _filt_h0, _filt_h1,
                             _filt_h2, _filt_h3);
  _out = __lsx_vsrari_h(_out, FILTER_BITS);
  return __lsx_vsat_h(_out, 7);
}

static INLINE __m128i horiz_2tap_filt_uh(__m128i in0, __m128i in1, __m128i mask,
                                         __m128i coeff) {
  __m128i tmp0_m, tmp1_m;

  tmp0_m = __lsx_vshuf_b(in1, in0, mask);
  tmp1_m = __lsx_vdp2_h_bu(tmp0_m, coeff);
  return __lsx_vsrari_h(tmp1_m, FILTER_BITS);
}

#define LSX_LD_4(_src, _stride, _src0, _src1, _src2, _src3) \
  do {                                                      \
    _src0 = __lsx_vld(_src, 0);                             \
    _src += _stride;                                        \
    _src1 = __lsx_vld(_src, 0);                             \
    _src += _stride;                                        \
    _src2 = __lsx_vld(_src, 0);                             \
    _src += _stride;                                        \
    _src3 = __lsx_vld(_src, 0);                             \
  } while (0)

#define HORIZ_8TAP_4WID_4VECS_FILT(_src0, _src1, _src2, _src3, _mask0, _mask1, \
                                   _mask2, _mask3, _filter0, _filter1,         \
                                   _filter2, _filter3, _out0, _out1)           \
  do {                                                                         \
    __m128i _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7;            \
    __m128i _reg0, _reg1, _reg2, _reg3;                                        \
                                                                               \
    DUP2_ARG3(__lsx_vshuf_b, _src1, _src0, _mask0, _src3, _src2, _mask0,       \
              _tmp0, _tmp1);                                                   \
    DUP2_ARG2(__lsx_vdp2_h_b, _tmp0, _filter0, _tmp1, _filter0, _reg0, _reg1); \
    DUP2_ARG3(__lsx_vshuf_b, _src1, _src0, _mask1, _src3, _src2, _mask1,       \
              _tmp2, _tmp3);                                                   \
    DUP2_ARG3(__lsx_vdp2add_h_b, _reg0, _tmp2, _filter1, _reg1, _tmp3,         \
              _filter1, _reg0, _reg1);                                         \
    DUP2_ARG3(__lsx_vshuf_b, _src1, _src0, _mask2, _src3, _src2, _mask2,       \
              _tmp4, _tmp5);                                                   \
    DUP2_ARG2(__lsx_vdp2_h_b, _tmp4, _filter2, _tmp5, _filter2, _reg2, _reg3); \
    DUP2_ARG3(__lsx_vshuf_b, _src1, _src0, _mask3, _src3, _src2, _mask3,       \
              _tmp6, _tmp7);                                                   \
    DUP2_ARG3(__lsx_vdp2add_h_b, _reg2, _tmp6, _filter3, _reg3, _tmp7,         \
              _filter3, _reg2, _reg3);                                         \
    DUP2_ARG2(__lsx_vsadd_h, _reg0, _reg2, _reg1, _reg3, _out0, _out1);        \
  } while (0)

#define HORIZ_8TAP_8WID_4VECS_FILT(                                            \
    _src0, _src1, _src2, _src3, _mask0, _mask1, _mask2, _mask3, _filter0,      \
    _filter1, _filter2, _filter3, _out0, _out1, _out2, _out3)                  \
  do {                                                                         \
    __m128i _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7;            \
    __m128i _reg0, _reg1, _reg2, _reg3, _reg4, _reg5, _reg6, _reg7;            \
                                                                               \
    DUP4_ARG3(__lsx_vshuf_b, _src0, _src0, _mask0, _src1, _src1, _mask0,       \
              _src2, _src2, _mask0, _src3, _src3, _mask0, _tmp0, _tmp1, _tmp2, \
              _tmp3);                                                          \
    DUP4_ARG2(__lsx_vdp2_h_b, _tmp0, _filter0, _tmp1, _filter0, _tmp2,         \
              _filter0, _tmp3, _filter0, _reg0, _reg1, _reg2, _reg3);          \
    DUP4_ARG3(__lsx_vshuf_b, _src0, _src0, _mask2, _src1, _src1, _mask2,       \
              _src2, _src2, _mask2, _src3, _src3, _mask2, _tmp0, _tmp1, _tmp2, \
              _tmp3);                                                          \
    DUP4_ARG2(__lsx_vdp2_h_b, _tmp0, _filter2, _tmp1, _filter2, _tmp2,         \
              _filter2, _tmp3, _filter2, _reg4, _reg5, _reg6, _reg7);          \
    DUP4_ARG3(__lsx_vshuf_b, _src0, _src0, _mask1, _src1, _src1, _mask1,       \
              _src2, _src2, _mask1, _src3, _src3, _mask1, _tmp4, _tmp5, _tmp6, \
              _tmp7);                                                          \
    DUP4_ARG3(__lsx_vdp2add_h_b, _reg0, _tmp4, _filter1, _reg1, _tmp5,         \
              _filter1, _reg2, _tmp6, _filter1, _reg3, _tmp7, _filter1, _reg0, \
              _reg1, _reg2, _reg3);                                            \
    DUP4_ARG3(__lsx_vshuf_b, _src0, _src0, _mask3, _src1, _src1, _mask3,       \
              _src2, _src2, _mask3, _src3, _src3, _mask3, _tmp4, _tmp5, _tmp6, \
              _tmp7);                                                          \
    DUP4_ARG3(__lsx_vdp2add_h_b, _reg4, _tmp4, _filter3, _reg5, _tmp5,         \
              _filter3, _reg6, _tmp6, _filter3, _reg7, _tmp7, _filter3, _reg4, \
              _reg5, _reg6, _reg7);                                            \
    DUP4_ARG2(__lsx_vsadd_h, _reg0, _reg4, _reg1, _reg5, _reg2, _reg6, _reg3,  \
              _reg7, _out0, _out1, _out2, _out3);                              \
  } while (0)

#define AVG_ST4_D(in0, in1, dst0, dst1, pdst, stride)                \
  do {                                                               \
    __m128i tmp0_m, tmp1_m;                                          \
                                                                     \
    DUP2_ARG2(__lsx_vavgr_bu, in0, dst0, in1, dst1, tmp0_m, tmp1_m); \
    __lsx_vstelm_d(tmp0_m, pdst, 0, 0);                              \
    pdst += stride;                                                  \
    __lsx_vstelm_d(tmp0_m, pdst, 0, 1);                              \
    pdst += stride;                                                  \
    __lsx_vstelm_d(tmp1_m, pdst, 0, 0);                              \
    pdst += stride;                                                  \
    __lsx_vstelm_d(tmp1_m, pdst, 0, 1);                              \
  } while (0)

#endif  // VPX_VPX_DSP_LOONGARCH_VPX_CONVOLVE_LSX_H_
