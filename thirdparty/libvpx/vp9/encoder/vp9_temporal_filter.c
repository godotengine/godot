/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <math.h>
#include <limits.h>

#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/encoder/vp9_encodeframe.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vp9/encoder/vp9_extend.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_mcomp.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_quantize.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_segmentation.h"
#include "vp9/encoder/vp9_temporal_filter.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_scale/vpx_scale.h"

static int fixed_divide[512];
static unsigned int index_mult[14] = { 0,     0,     0,     0,     49152,
                                       39322, 32768, 28087, 24576, 21846,
                                       19661, 17874, 0,     15124 };
#if CONFIG_VP9_HIGHBITDEPTH
static int64_t highbd_index_mult[14] = { 0U,          0U,          0U,
                                         0U,          3221225472U, 2576980378U,
                                         2147483648U, 1840700270U, 1610612736U,
                                         1431655766U, 1288490189U, 1171354718U,
                                         0U,          991146300U };
#endif  // CONFIG_VP9_HIGHBITDEPTH

static const MV kZeroMv = { 0, 0 };
#define TF_INTERP_EXTEND 6

// Prediction function using 12-tap interpolation filter.
void vpx_convolve12_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel12 *filter, int x0_q4,
                            int x_step_q4, int y0_q4, int y_step_q4, int w,
                            int h) {
  (void)y0_q4;
  (void)y_step_q4;
  int x, y;
  src -= MAX_FILTER_TAP / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = filter[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < MAX_FILTER_TAP; ++k) sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve12_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel12 *filter, int x0_q4,
                           int x_step_q4, int y0_q4, int y_step_q4, int w,
                           int h) {
  (void)x0_q4;
  (void)x_step_q4;
  int x, y;
  src -= src_stride * (MAX_FILTER_TAP / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = filter[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < MAX_FILTER_TAP; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

// Copied from vpx_convolve8_c(). Possible block sizes are 32x32, 16x16, 8x8.
void vpx_convolve12_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                      ptrdiff_t dst_stride, const InterpKernel12 *filter,
                      int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                      int h) {
  uint8_t temp[BW * (BH + MAX_FILTER_TAP - 1)];
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;

  vpx_convolve12_horiz_c(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                         src_stride, temp, temp_stride, filter, x0_q4,
                         x_step_q4, y0_q4, y_step_q4, w, intermediate_height);
  vpx_convolve12_vert_c(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                        temp_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                        y0_q4, y_step_q4, w, h);
}

static void vp9_build_inter_predictor_12(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride,
    const MV *src_mv, const struct scale_factors *sf, int w, int h, int ref,
    const InterpKernel12 *kernel, enum mv_precision precision, int x, int y) {
  (void)ref;
  const int is_q4 = precision == MV_PRECISION_Q4;
  const MV mv_q4 = { is_q4 ? src_mv->row : src_mv->row * 2,
                     is_q4 ? src_mv->col : src_mv->col * 2 };
  MV32 mv = vp9_scale_mv(&mv_q4, x, y, sf);
  const int subpel_x = mv.col & SUBPEL_MASK;
  const int subpel_y = mv.row & SUBPEL_MASK;

  src += (mv.row >> SUBPEL_BITS) * src_stride + (mv.col >> SUBPEL_BITS);

  if (subpel_x == 0 && subpel_y == 0) {
    vpx_convolve_copy(src, src_stride, dst, dst_stride, NULL, subpel_x,
                      sf->x_step_q4, subpel_y, sf->y_step_q4, w, h);
  } else if (subpel_x == 0 && subpel_y != 0) {
    vpx_convolve12_vert(src, src_stride, dst, dst_stride, kernel, subpel_x,
                        sf->x_step_q4, subpel_y, sf->y_step_q4, w, h);
  } else if (subpel_x != 0 && subpel_y == 0) {
    vpx_convolve12_horiz(src, src_stride, dst, dst_stride, kernel, subpel_x,
                         sf->x_step_q4, subpel_y, sf->y_step_q4, w, h);
  } else {
    vpx_convolve12(src, src_stride, dst, dst_stride, kernel, subpel_x,
                   sf->x_step_q4, subpel_y, sf->y_step_q4, w, h);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_convolve12_horiz_c(const uint16_t *src, ptrdiff_t src_stride,
                                   uint16_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel12 *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h, int bd) {
  (void)y0_q4;
  (void)y_step_q4;
  int x, y;
  src -= MAX_FILTER_TAP / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint16_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = filter[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < MAX_FILTER_TAP; ++k) sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_highbd_convolve12_vert_c(const uint16_t *src, ptrdiff_t src_stride,
                                  uint16_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel12 *filter, int x0_q4,
                                  int x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h, int bd) {
  (void)x0_q4;
  (void)x_step_q4;
  int x, y;
  src -= src_stride * (MAX_FILTER_TAP / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint16_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = filter[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < MAX_FILTER_TAP; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] =
          clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void highbd_convolve12(const uint16_t *src, ptrdiff_t src_stride,
                              uint16_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel12 *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4, int w,
                              int h, int bd) {
  uint16_t temp[BW * (BH + MAX_FILTER_TAP - 1)];
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;

  vpx_highbd_convolve12_horiz_c(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                                src_stride, temp, temp_stride, filter, x0_q4,
                                x_step_q4, y0_q4, y_step_q4, w,
                                intermediate_height, bd);
  vpx_highbd_convolve12_vert_c(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                               temp_stride, dst, dst_stride, filter, x0_q4,
                               x_step_q4, y0_q4, y_step_q4, w, h, bd);
}

// Copied from vpx_highbd_convolve8_c()
void vpx_highbd_convolve12_c(const uint16_t *src, ptrdiff_t src_stride,
                             uint16_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel12 *filter, int x0_q4,
                             int x_step_q4, int y0_q4, int y_step_q4, int w,
                             int h, int bd) {
  highbd_convolve12(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                    y0_q4, y_step_q4, w, h, bd);
}

static void vp9_highbd_build_inter_predictor_12(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
    const MV *src_mv, const struct scale_factors *sf, int w, int h, int ref,
    const InterpKernel12 *kernel, enum mv_precision precision, int x, int y,
    int bd) {
  (void)ref;
  const int is_q4 = precision == MV_PRECISION_Q4;
  const MV mv_q4 = { is_q4 ? src_mv->row : src_mv->row * 2,
                     is_q4 ? src_mv->col : src_mv->col * 2 };
  MV32 mv = vp9_scale_mv(&mv_q4, x, y, sf);
  const int subpel_x = mv.col & SUBPEL_MASK;
  const int subpel_y = mv.row & SUBPEL_MASK;

  src += (mv.row >> SUBPEL_BITS) * src_stride + (mv.col >> SUBPEL_BITS);

  if (subpel_x == 0 && subpel_y == 0) {
    vpx_highbd_convolve_copy(src, src_stride, dst, dst_stride, NULL, subpel_x,
                             sf->x_step_q4, subpel_y, sf->y_step_q4, w, h, bd);
  } else if (subpel_x == 0 && subpel_y != 0) {
    vpx_highbd_convolve12_vert(src, src_stride, dst, dst_stride, kernel,
                               subpel_x, sf->x_step_q4, subpel_y, sf->y_step_q4,
                               w, h, bd);
  } else if (subpel_x != 0 && subpel_y == 0) {
    vpx_highbd_convolve12_horiz(src, src_stride, dst, dst_stride, kernel,
                                subpel_x, sf->x_step_q4, subpel_y,
                                sf->y_step_q4, w, h, bd);
  } else {
    vpx_highbd_convolve12(src, src_stride, dst, dst_stride, kernel, subpel_x,
                          sf->x_step_q4, subpel_y, sf->y_step_q4, w, h, bd);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void temporal_filter_predictors_mb_c(
    MACROBLOCKD *xd, uint8_t *y_mb_ptr, uint8_t *u_mb_ptr, uint8_t *v_mb_ptr,
    int stride, int uv_block_width, int uv_block_height, int mv_row, int mv_col,
    uint8_t *pred, struct scale_factors *scale, int x, int y, MV *blk_mvs,
    int use_32x32) {
  const int which_mv = 0;
  const InterpKernel12 *const kernel = sub_pel_filters_12;
  int i, j, k = 0, ys = (BH >> 1), xs = (BW >> 1);

  enum mv_precision mv_precision_uv;
  int uv_stride;
  if (uv_block_width == (BW >> 1)) {
    uv_stride = (stride + 1) >> 1;
    mv_precision_uv = MV_PRECISION_Q4;
  } else {
    uv_stride = stride;
    mv_precision_uv = MV_PRECISION_Q3;
  }
#if !CONFIG_VP9_HIGHBITDEPTH
  (void)xd;
#endif

  if (use_32x32) {
    const MV mv = { mv_row, mv_col };
#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      vp9_highbd_build_inter_predictor_12(CONVERT_TO_SHORTPTR(y_mb_ptr), stride,
                                          CONVERT_TO_SHORTPTR(&pred[0]), BW,
                                          &mv, scale, BW, BH, which_mv, kernel,
                                          MV_PRECISION_Q3, x, y, xd->bd);

      vp9_highbd_build_inter_predictor_12(
          CONVERT_TO_SHORTPTR(u_mb_ptr), uv_stride,
          CONVERT_TO_SHORTPTR(&pred[BLK_PELS]), uv_block_width, &mv, scale,
          uv_block_width, uv_block_height, which_mv, kernel, mv_precision_uv, x,
          y, xd->bd);

      vp9_highbd_build_inter_predictor_12(
          CONVERT_TO_SHORTPTR(v_mb_ptr), uv_stride,
          CONVERT_TO_SHORTPTR(&pred[(BLK_PELS << 1)]), uv_block_width, &mv,
          scale, uv_block_width, uv_block_height, which_mv, kernel,
          mv_precision_uv, x, y, xd->bd);
      return;
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    vp9_build_inter_predictor_12(y_mb_ptr, stride, &pred[0], BW, &mv, scale, BW,
                                 BH, which_mv, kernel, MV_PRECISION_Q3, x, y);

    vp9_build_inter_predictor_12(u_mb_ptr, uv_stride, &pred[BLK_PELS],
                                 uv_block_width, &mv, scale, uv_block_width,
                                 uv_block_height, which_mv, kernel,
                                 mv_precision_uv, x, y);

    vp9_build_inter_predictor_12(v_mb_ptr, uv_stride, &pred[(BLK_PELS << 1)],
                                 uv_block_width, &mv, scale, uv_block_width,
                                 uv_block_height, which_mv, kernel,
                                 mv_precision_uv, x, y);
    return;
  }

  // While use_32x32 = 0, construct the 32x32 predictor using 4 16x16
  // predictors.
  // Y predictor
  for (i = 0; i < BH; i += ys) {
    for (j = 0; j < BW; j += xs) {
      const MV mv = blk_mvs[k];
      const int y_offset = i * stride + j;
      const int p_offset = i * BW + j;

#if CONFIG_VP9_HIGHBITDEPTH
      if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
        vp9_highbd_build_inter_predictor_12(
            CONVERT_TO_SHORTPTR(y_mb_ptr + y_offset), stride,
            CONVERT_TO_SHORTPTR(&pred[p_offset]), BW, &mv, scale, xs, ys,
            which_mv, kernel, MV_PRECISION_Q3, x, y, xd->bd);
      } else {
        vp9_build_inter_predictor_12(y_mb_ptr + y_offset, stride,
                                     &pred[p_offset], BW, &mv, scale, xs, ys,
                                     which_mv, kernel, MV_PRECISION_Q3, x, y);
      }
#else
      vp9_build_inter_predictor_12(y_mb_ptr + y_offset, stride, &pred[p_offset],
                                   BW, &mv, scale, xs, ys, which_mv, kernel,
                                   MV_PRECISION_Q3, x, y);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      k++;
    }
  }

  // U and V predictors
  ys = (uv_block_height >> 1);
  xs = (uv_block_width >> 1);
  k = 0;

  for (i = 0; i < uv_block_height; i += ys) {
    for (j = 0; j < uv_block_width; j += xs) {
      const MV mv = blk_mvs[k];
      const int uv_offset = i * uv_stride + j;
      const int p_offset = i * uv_block_width + j;

#if CONFIG_VP9_HIGHBITDEPTH
      if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
        vp9_highbd_build_inter_predictor_12(
            CONVERT_TO_SHORTPTR(u_mb_ptr + uv_offset), uv_stride,
            CONVERT_TO_SHORTPTR(&pred[BLK_PELS + p_offset]), uv_block_width,
            &mv, scale, xs, ys, which_mv, kernel, mv_precision_uv, x, y,
            xd->bd);

        vp9_highbd_build_inter_predictor_12(
            CONVERT_TO_SHORTPTR(v_mb_ptr + uv_offset), uv_stride,
            CONVERT_TO_SHORTPTR(&pred[(BLK_PELS << 1) + p_offset]),
            uv_block_width, &mv, scale, xs, ys, which_mv, kernel,
            mv_precision_uv, x, y, xd->bd);
      } else {
        vp9_build_inter_predictor_12(u_mb_ptr + uv_offset, uv_stride,
                                     &pred[BLK_PELS + p_offset], uv_block_width,
                                     &mv, scale, xs, ys, which_mv, kernel,
                                     mv_precision_uv, x, y);

        vp9_build_inter_predictor_12(v_mb_ptr + uv_offset, uv_stride,
                                     &pred[(BLK_PELS << 1) + p_offset],
                                     uv_block_width, &mv, scale, xs, ys,
                                     which_mv, kernel, mv_precision_uv, x, y);
      }
#else
      vp9_build_inter_predictor_12(u_mb_ptr + uv_offset, uv_stride,
                                   &pred[BLK_PELS + p_offset], uv_block_width,
                                   &mv, scale, xs, ys, which_mv, kernel,
                                   mv_precision_uv, x, y);

      vp9_build_inter_predictor_12(v_mb_ptr + uv_offset, uv_stride,
                                   &pred[(BLK_PELS << 1) + p_offset],
                                   uv_block_width, &mv, scale, xs, ys, which_mv,
                                   kernel, mv_precision_uv, x, y);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      k++;
    }
  }
}

void vp9_temporal_filter_init(void) {
  int i;

  fixed_divide[0] = 0;
  for (i = 1; i < 512; ++i) fixed_divide[i] = 0x80000 / i;
}

static INLINE int mod_index(int sum_dist, int index, int rounding, int strength,
                            int filter_weight) {
  int mod;

  assert(index >= 0 && index <= 13);
  assert(index_mult[index] != 0);

  mod =
      ((unsigned int)clamp(sum_dist, 0, UINT16_MAX) * index_mult[index]) >> 16;
  mod += rounding;
  mod >>= strength;

  mod = VPXMIN(16, mod);

  mod = 16 - mod;
  mod *= filter_weight;

  return mod;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE int highbd_mod_index(int sum_dist, int index, int rounding,
                                   int strength, int filter_weight) {
  int mod;

  assert(index >= 0 && index <= 13);
  assert(highbd_index_mult[index] != 0);

  mod = (int)((clamp(sum_dist, 0, INT32_MAX) * highbd_index_mult[index]) >> 32);
  mod += rounding;
  mod >>= strength;

  mod = VPXMIN(16, mod);

  mod = 16 - mod;
  mod *= filter_weight;

  return mod;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static INLINE int get_filter_weight(unsigned int i, unsigned int j,
                                    unsigned int block_height,
                                    unsigned int block_width,
                                    const int *const blk_fw, int use_32x32) {
  // blk_fw[0] ~ blk_fw[3] are the same.
  if (use_32x32) {
    return blk_fw[0];
  }

  if (i < block_height / 2) {
    if (j < block_width / 2) {
      return blk_fw[0];
    }

    return blk_fw[1];
  }

  if (j < block_width / 2) {
    return blk_fw[2];
  }

  return blk_fw[3];
}

void vp9_apply_temporal_filter_c(
    const uint8_t *y_frame1, int y_stride, const uint8_t *y_pred,
    int y_buf_stride, const uint8_t *u_frame1, const uint8_t *v_frame1,
    int uv_stride, const uint8_t *u_pred, const uint8_t *v_pred,
    int uv_buf_stride, unsigned int block_width, unsigned int block_height,
    int ss_x, int ss_y, int strength, const int *const blk_fw, int use_32x32,
    uint32_t *y_accumulator, uint16_t *y_count, uint32_t *u_accumulator,
    uint16_t *u_count, uint32_t *v_accumulator, uint16_t *v_count) {
  unsigned int i, j, k, m;
  int modifier;
  const int rounding = (1 << strength) >> 1;
  const unsigned int uv_block_width = block_width >> ss_x;
  const unsigned int uv_block_height = block_height >> ss_y;
  DECLARE_ALIGNED(16, uint16_t, y_diff_sse[BLK_PELS]);
  DECLARE_ALIGNED(16, uint16_t, u_diff_sse[BLK_PELS]);
  DECLARE_ALIGNED(16, uint16_t, v_diff_sse[BLK_PELS]);

  int idx = 0, idy;

  assert(strength >= 0);
  assert(strength <= 6);

  memset(y_diff_sse, 0, BLK_PELS * sizeof(uint16_t));
  memset(u_diff_sse, 0, BLK_PELS * sizeof(uint16_t));
  memset(v_diff_sse, 0, BLK_PELS * sizeof(uint16_t));

  // Calculate diff^2 for each pixel of the 16x16 block.
  // TODO(yunqing): the following code needs to be optimized.
  for (i = 0; i < block_height; i++) {
    for (j = 0; j < block_width; j++) {
      const int16_t diff =
          y_frame1[i * (int)y_stride + j] - y_pred[i * (int)block_width + j];
      y_diff_sse[idx++] = diff * diff;
    }
  }
  idx = 0;
  for (i = 0; i < uv_block_height; i++) {
    for (j = 0; j < uv_block_width; j++) {
      const int16_t diffu =
          u_frame1[i * uv_stride + j] - u_pred[i * uv_buf_stride + j];
      const int16_t diffv =
          v_frame1[i * uv_stride + j] - v_pred[i * uv_buf_stride + j];
      u_diff_sse[idx] = diffu * diffu;
      v_diff_sse[idx] = diffv * diffv;
      idx++;
    }
  }

  for (i = 0, k = 0, m = 0; i < block_height; i++) {
    for (j = 0; j < block_width; j++) {
      const int pixel_value = y_pred[i * y_buf_stride + j];
      const int filter_weight =
          get_filter_weight(i, j, block_height, block_width, blk_fw, use_32x32);

      // non-local mean approach
      int y_index = 0;

      const int uv_r = i >> ss_y;
      const int uv_c = j >> ss_x;
      modifier = 0;

      for (idy = -1; idy <= 1; ++idy) {
        for (idx = -1; idx <= 1; ++idx) {
          const int row = (int)i + idy;
          const int col = (int)j + idx;

          if (row >= 0 && row < (int)block_height && col >= 0 &&
              col < (int)block_width) {
            modifier += y_diff_sse[row * (int)block_width + col];
            ++y_index;
          }
        }
      }

      assert(y_index > 0);

      modifier += u_diff_sse[uv_r * uv_block_width + uv_c];
      modifier += v_diff_sse[uv_r * uv_block_width + uv_c];

      y_index += 2;

      modifier =
          mod_index(modifier, y_index, rounding, strength, filter_weight);

      y_count[k] += modifier;
      y_accumulator[k] += modifier * pixel_value;

      ++k;

      // Process chroma component
      if (!(i & ss_y) && !(j & ss_x)) {
        const int u_pixel_value = u_pred[uv_r * uv_buf_stride + uv_c];
        const int v_pixel_value = v_pred[uv_r * uv_buf_stride + uv_c];

        // non-local mean approach
        int cr_index = 0;
        int u_mod = 0, v_mod = 0;
        int y_diff = 0;

        for (idy = -1; idy <= 1; ++idy) {
          for (idx = -1; idx <= 1; ++idx) {
            const int row = uv_r + idy;
            const int col = uv_c + idx;

            if (row >= 0 && row < (int)uv_block_height && col >= 0 &&
                col < (int)uv_block_width) {
              u_mod += u_diff_sse[row * uv_block_width + col];
              v_mod += v_diff_sse[row * uv_block_width + col];
              ++cr_index;
            }
          }
        }

        assert(cr_index > 0);

        for (idy = 0; idy < 1 + ss_y; ++idy) {
          for (idx = 0; idx < 1 + ss_x; ++idx) {
            const int row = (uv_r << ss_y) + idy;
            const int col = (uv_c << ss_x) + idx;
            y_diff += y_diff_sse[row * (int)block_width + col];
            ++cr_index;
          }
        }

        u_mod += y_diff;
        v_mod += y_diff;

        u_mod = mod_index(u_mod, cr_index, rounding, strength, filter_weight);
        v_mod = mod_index(v_mod, cr_index, rounding, strength, filter_weight);

        u_count[m] += u_mod;
        u_accumulator[m] += u_mod * u_pixel_value;
        v_count[m] += v_mod;
        v_accumulator[m] += v_mod * v_pixel_value;

        ++m;
      }  // Complete YUV pixel
    }
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_apply_temporal_filter_c(
    const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre,
    int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src,
    int uv_src_stride, const uint16_t *u_pre, const uint16_t *v_pre,
    int uv_pre_stride, unsigned int block_width, unsigned int block_height,
    int ss_x, int ss_y, int strength, const int *const blk_fw, int use_32x32,
    uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count,
    uint32_t *v_accum, uint16_t *v_count) {
  const int uv_block_width = block_width >> ss_x;
  const int uv_block_height = block_height >> ss_y;
  const int y_diff_stride = BW;
  const int uv_diff_stride = BW;

  DECLARE_ALIGNED(16, uint32_t, y_diff_sse[BLK_PELS]);
  DECLARE_ALIGNED(16, uint32_t, u_diff_sse[BLK_PELS]);
  DECLARE_ALIGNED(16, uint32_t, v_diff_sse[BLK_PELS]);

  const int rounding = (1 << strength) >> 1;

  // Loop variables
  int row, col;
  int uv_row, uv_col;
  int row_step, col_step;

  memset(y_diff_sse, 0, BLK_PELS * sizeof(uint32_t));
  memset(u_diff_sse, 0, BLK_PELS * sizeof(uint32_t));
  memset(v_diff_sse, 0, BLK_PELS * sizeof(uint32_t));

  // Get the square diffs
  for (row = 0; row < (int)block_height; row++) {
    for (col = 0; col < (int)block_width; col++) {
      const int diff =
          y_src[row * y_src_stride + col] - y_pre[row * y_pre_stride + col];
      y_diff_sse[row * y_diff_stride + col] = diff * diff;
    }
  }

  for (row = 0; row < uv_block_height; row++) {
    for (col = 0; col < uv_block_width; col++) {
      const int u_diff =
          u_src[row * uv_src_stride + col] - u_pre[row * uv_pre_stride + col];
      const int v_diff =
          v_src[row * uv_src_stride + col] - v_pre[row * uv_pre_stride + col];
      u_diff_sse[row * uv_diff_stride + col] = u_diff * u_diff;
      v_diff_sse[row * uv_diff_stride + col] = v_diff * v_diff;
    }
  }

  // Apply the filter to luma
  for (row = 0; row < (int)block_height; row++) {
    for (col = 0; col < (int)block_width; col++) {
      const int filter_weight = get_filter_weight(
          row, col, block_height, block_width, blk_fw, use_32x32);

      // First we get the modifier for the current y pixel
      const int y_pixel = y_pre[row * y_pre_stride + col];
      int y_num_used = 0;
      int y_mod = 0;

      // Sum the neighboring 3x3 y pixels
      for (row_step = -1; row_step <= 1; row_step++) {
        for (col_step = -1; col_step <= 1; col_step++) {
          const int sub_row = row + row_step;
          const int sub_col = col + col_step;

          if (sub_row >= 0 && sub_row < (int)block_height && sub_col >= 0 &&
              sub_col < (int)block_width) {
            y_mod += y_diff_sse[sub_row * y_diff_stride + sub_col];
            y_num_used++;
          }
        }
      }

      // Sum the corresponding uv pixels to the current y modifier
      // Note we are rounding down instead of rounding to the nearest pixel.
      uv_row = row >> ss_y;
      uv_col = col >> ss_x;
      y_mod += u_diff_sse[uv_row * uv_diff_stride + uv_col];
      y_mod += v_diff_sse[uv_row * uv_diff_stride + uv_col];

      y_num_used += 2;

      // Set the modifier
      y_mod = highbd_mod_index(y_mod, y_num_used, rounding, strength,
                               filter_weight);

      // Accumulate the result
      y_count[row * block_width + col] += y_mod;
      y_accum[row * block_width + col] += y_mod * y_pixel;
    }
  }

  // Apply the filter to chroma
  for (uv_row = 0; uv_row < uv_block_height; uv_row++) {
    for (uv_col = 0; uv_col < uv_block_width; uv_col++) {
      const int y_row = uv_row << ss_y;
      const int y_col = uv_col << ss_x;
      const int filter_weight = get_filter_weight(
          uv_row, uv_col, uv_block_height, uv_block_width, blk_fw, use_32x32);

      const int u_pixel = u_pre[uv_row * uv_pre_stride + uv_col];
      const int v_pixel = v_pre[uv_row * uv_pre_stride + uv_col];

      int uv_num_used = 0;
      int u_mod = 0, v_mod = 0;

      // Sum the neighboring 3x3 chromal pixels to the chroma modifier
      for (row_step = -1; row_step <= 1; row_step++) {
        for (col_step = -1; col_step <= 1; col_step++) {
          const int sub_row = uv_row + row_step;
          const int sub_col = uv_col + col_step;

          if (sub_row >= 0 && sub_row < uv_block_height && sub_col >= 0 &&
              sub_col < uv_block_width) {
            u_mod += u_diff_sse[sub_row * uv_diff_stride + sub_col];
            v_mod += v_diff_sse[sub_row * uv_diff_stride + sub_col];
            uv_num_used++;
          }
        }
      }

      // Sum all the luma pixels associated with the current luma pixel
      for (row_step = 0; row_step < 1 + ss_y; row_step++) {
        for (col_step = 0; col_step < 1 + ss_x; col_step++) {
          const int sub_row = y_row + row_step;
          const int sub_col = y_col + col_step;
          const int y_diff = y_diff_sse[sub_row * y_diff_stride + sub_col];

          u_mod += y_diff;
          v_mod += y_diff;
          uv_num_used++;
        }
      }

      // Set the modifier
      u_mod = highbd_mod_index(u_mod, uv_num_used, rounding, strength,
                               filter_weight);
      v_mod = highbd_mod_index(v_mod, uv_num_used, rounding, strength,
                               filter_weight);

      // Accumulate the result
      u_count[uv_row * uv_block_width + uv_col] += u_mod;
      u_accum[uv_row * uv_block_width + uv_col] += u_mod * u_pixel;
      v_count[uv_row * uv_block_width + uv_col] += v_mod;
      v_accum[uv_row * uv_block_width + uv_col] += v_mod * v_pixel;
    }
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static uint32_t temporal_filter_find_matching_mb_c(
    VP9_COMP *cpi, ThreadData *td, uint8_t *arf_frame_buf,
    uint8_t *frame_ptr_buf, int stride, MV *ref_mv, MV *blk_mvs,
    int *blk_bestsme, int *is_dc_diff_large) {
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MV_SPEED_FEATURES *const mv_sf = &cpi->sf.mv;
  const SEARCH_METHODS search_method = MESH;
  const SEARCH_METHODS search_method_16 = cpi->sf.temporal_filter_search_method;
  int step_param;
  int sadpb = x->sadperbit16;
  uint32_t bestsme = UINT_MAX;
  uint32_t distortion;
  uint32_t sse;
  int cost_list[5];
  const MvLimits tmp_mv_limits = x->mv_limits;

  MV best_ref_mv1 = { 0, 0 };
  MV best_ref_mv1_full; /* full-pixel value of best_ref_mv1 */

  // Save input state
  struct buf_2d src = x->plane[0].src;
  struct buf_2d pre = xd->plane[0].pre[0];
  int i, j, k = 0;

  best_ref_mv1_full.col = best_ref_mv1.col >> 3;
  best_ref_mv1_full.row = best_ref_mv1.row >> 3;

  // Setup frame pointers
  x->plane[0].src.buf = arf_frame_buf;
  x->plane[0].src.stride = stride;
  xd->plane[0].pre[0].buf = frame_ptr_buf;
  xd->plane[0].pre[0].stride = stride;
  *is_dc_diff_large = 0;

  step_param = mv_sf->reduce_first_step_size;
  step_param = VPXMIN(step_param, MAX_MVSEARCH_STEPS - 2);

  vp9_set_mv_search_range(&x->mv_limits, &best_ref_mv1);

  vp9_full_pixel_search(cpi, x, TF_BLOCK, &best_ref_mv1_full, step_param,
                        search_method, sadpb, cond_cost_list(cpi, cost_list),
                        &best_ref_mv1, ref_mv, 0, 0);

  /* restore UMV window */
  x->mv_limits = tmp_mv_limits;

  // find_fractional_mv_step parameters: best_ref_mv1 is for mv rate cost
  // calculation. The start full mv and the search result are stored in
  // ref_mv.
  bestsme = cpi->find_fractional_mv_step(
      x, ref_mv, &best_ref_mv1, cpi->common.allow_high_precision_mv,
      x->errorperbit, &cpi->fn_ptr[TF_BLOCK], 0, mv_sf->subpel_search_level,
      cond_cost_list(cpi, cost_list), NULL, NULL, &distortion, &sse, NULL, BW,
      BH, USE_8_TAPS_SHARP);
  *is_dc_diff_large = 50 * bestsme < sse;

  // DO motion search on 4 16x16 sub_blocks.
  best_ref_mv1.row = ref_mv->row;
  best_ref_mv1.col = ref_mv->col;
  best_ref_mv1_full.col = best_ref_mv1.col >> 3;
  best_ref_mv1_full.row = best_ref_mv1.row >> 3;

  for (i = 0; i < BH; i += SUB_BH) {
    for (j = 0; j < BW; j += SUB_BW) {
      // Setup frame pointers
      x->plane[0].src.buf = arf_frame_buf + i * stride + j;
      x->plane[0].src.stride = stride;
      xd->plane[0].pre[0].buf = frame_ptr_buf + i * stride + j;
      xd->plane[0].pre[0].stride = stride;

      vp9_set_mv_search_range(&x->mv_limits, &best_ref_mv1);
      vp9_full_pixel_search(cpi, x, TF_SUB_BLOCK, &best_ref_mv1_full,
                            step_param, search_method_16, sadpb,
                            cond_cost_list(cpi, cost_list), &best_ref_mv1,
                            &blk_mvs[k], 0, 0);
      /* restore UMV window */
      x->mv_limits = tmp_mv_limits;

      blk_bestsme[k] = cpi->find_fractional_mv_step(
          x, &blk_mvs[k], &best_ref_mv1, cpi->common.allow_high_precision_mv,
          x->errorperbit, &cpi->fn_ptr[TF_SUB_BLOCK], 0,
          mv_sf->subpel_search_level, cond_cost_list(cpi, cost_list), NULL,
          NULL, &distortion, &sse, NULL, SUB_BW, SUB_BH, USE_8_TAPS_SHARP);
      k++;
    }
  }

  // Restore input state
  x->plane[0].src = src;
  xd->plane[0].pre[0] = pre;

  return bestsme;
}

void vp9_temporal_filter_iterate_row_c(VP9_COMP *cpi, ThreadData *td,
                                       int mb_row, int mb_col_start,
                                       int mb_col_end) {
  ARNRFilterData *arnr_filter_data = &cpi->arnr_filter_data;
  YV12_BUFFER_CONFIG **frames = arnr_filter_data->frames;
  int frame_count = arnr_filter_data->frame_count;
  int alt_ref_index = arnr_filter_data->alt_ref_index;
  int strength = arnr_filter_data->strength;
  struct scale_factors *scale = &arnr_filter_data->sf;
  int byte;
  int frame;
  int mb_col;
  int mb_cols = (frames[alt_ref_index]->y_crop_width + BW - 1) >> BW_LOG2;
  int mb_rows = (frames[alt_ref_index]->y_crop_height + BH - 1) >> BH_LOG2;
  DECLARE_ALIGNED(16, uint32_t, accumulator[BLK_PELS * 3]);
  DECLARE_ALIGNED(16, uint16_t, count[BLK_PELS * 3]);
  MACROBLOCKD *mbd = &td->mb.e_mbd;
  YV12_BUFFER_CONFIG *f = frames[alt_ref_index];
  YV12_BUFFER_CONFIG *dst = arnr_filter_data->dst;
  uint8_t *dst1, *dst2;
#if CONFIG_VP9_HIGHBITDEPTH
  DECLARE_ALIGNED(16, uint16_t, predictor16[BLK_PELS * 3]);
  DECLARE_ALIGNED(16, uint8_t, predictor8[BLK_PELS * 3]);
  uint8_t *predictor;
#else
  DECLARE_ALIGNED(16, uint8_t, predictor[BLK_PELS * 3]);
#endif
  const int mb_uv_height = BH >> mbd->plane[1].subsampling_y;
  const int mb_uv_width = BW >> mbd->plane[1].subsampling_x;
  // Addition of the tile col level offsets
  int mb_y_offset = mb_row * BH * (f->y_stride) + BW * mb_col_start;
  int mb_uv_offset =
      mb_row * mb_uv_height * f->uv_stride + mb_uv_width * mb_col_start;

#if CONFIG_VP9_HIGHBITDEPTH
  if (mbd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    predictor = CONVERT_TO_BYTEPTR(predictor16);
  } else {
    predictor = predictor8;
  }
#endif

  // Source frames are extended to 16 pixels. This is different than
  //  L/A/G reference frames that have a border of 32 (VP9ENCBORDERINPIXELS)
  // A 6/8/12 tap filter is used for motion search and prediction. So the
  // largest Y mv on a border would then be 16 - TF_INTERP_EXTEND. The UV
  // blocks are half the size of the Y and therefore only extended by 8.
  // The largest mv that a UV block can support is 8 - TF_INTERP_EXTEND.
  // A UV mv is half of a Y mv. (16 - TF_INTERP_EXTEND) >> 1 is greater than
  // 8 - TF_INTERP_EXTEND. To keep the mv in play for both Y and UV planes,
  // the max that it can be on a border is therefore 16 - (2 * TF_INTERP_EXTEND
  // + 1).
  td->mb.mv_limits.row_min = -((mb_row * BH) + (17 - 2 * TF_INTERP_EXTEND));
  td->mb.mv_limits.row_max =
      ((mb_rows - 1 - mb_row) * BH) + (17 - 2 * TF_INTERP_EXTEND);

  for (mb_col = mb_col_start; mb_col < mb_col_end; mb_col++) {
    int i, j, k;
    int stride;
    MV ref_mv;

    vp9_zero_array(accumulator, BLK_PELS * 3);
    vp9_zero_array(count, BLK_PELS * 3);

    td->mb.mv_limits.col_min = -((mb_col * BW) + (17 - 2 * TF_INTERP_EXTEND));
    td->mb.mv_limits.col_max =
        ((mb_cols - 1 - mb_col) * BW) + (17 - 2 * TF_INTERP_EXTEND);

    if (cpi->oxcf.content == VP9E_CONTENT_FILM) {
      unsigned int src_variance;
      struct buf_2d src;

      src.buf = f->y_buffer + mb_y_offset;
      src.stride = f->y_stride;

#if CONFIG_VP9_HIGHBITDEPTH
      if (mbd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
        src_variance =
            vp9_high_get_sby_perpixel_variance(cpi, &src, TF_BLOCK, mbd->bd);
      } else {
        src_variance = vp9_get_sby_perpixel_variance(cpi, &src, TF_BLOCK);
      }
#else
      src_variance = vp9_get_sby_perpixel_variance(cpi, &src, TF_BLOCK);
#endif  // CONFIG_VP9_HIGHBITDEPTH

      if (src_variance <= 2) {
        strength = VPXMAX(0, arnr_filter_data->strength - 2);
      }
    }

    for (frame = 0; frame < frame_count; frame++) {
      // MVs for 4 16x16 sub blocks.
      MV blk_mvs[4];
      // Filter weights for 4 16x16 sub blocks.
      int blk_fw[4] = { 0, 0, 0, 0 };
      int use_32x32 = 0;

      if (frames[frame] == NULL) continue;

      ref_mv.row = 0;
      ref_mv.col = 0;
      blk_mvs[0] = kZeroMv;
      blk_mvs[1] = kZeroMv;
      blk_mvs[2] = kZeroMv;
      blk_mvs[3] = kZeroMv;

      if (frame == alt_ref_index) {
        blk_fw[0] = blk_fw[1] = blk_fw[2] = blk_fw[3] = 2;
        use_32x32 = 1;
      } else {
        const int thresh_low = 10000;
        const int thresh_high = 20000;
        int blk_bestsme[4] = { INT_MAX, INT_MAX, INT_MAX, INT_MAX };
        int is_dc_diff_large = 0;

        // Find best match in this frame by MC
        int err = temporal_filter_find_matching_mb_c(
            cpi, td, frames[alt_ref_index]->y_buffer + mb_y_offset,
            frames[frame]->y_buffer + mb_y_offset, frames[frame]->y_stride,
            &ref_mv, blk_mvs, blk_bestsme, &is_dc_diff_large);

        if (cpi->oxcf.enable_keyframe_filtering == 1 &&
            cpi->common.frame_type == KEY_FRAME && is_dc_diff_large)
          strength = VPXMIN(strength, 1);

        int err16 =
            blk_bestsme[0] + blk_bestsme[1] + blk_bestsme[2] + blk_bestsme[3];
        int max_err = INT_MIN, min_err = INT_MAX;
        for (k = 0; k < 4; k++) {
          if (min_err > blk_bestsme[k]) min_err = blk_bestsme[k];
          if (max_err < blk_bestsme[k]) max_err = blk_bestsme[k];
        }

        if (((err * 15 < (err16 << 4)) && max_err - min_err < 10000) ||
            ((err * 14 < (err16 << 4)) && max_err - min_err < 5000)) {
          use_32x32 = 1;
          // Assign higher weight to matching MB if it's error
          // score is lower. If not applying MC default behavior
          // is to weight all MBs equal.
          blk_fw[0] = err < (thresh_low << THR_SHIFT)    ? 2
                      : err < (thresh_high << THR_SHIFT) ? 1
                                                         : 0;
          blk_fw[1] = blk_fw[2] = blk_fw[3] = blk_fw[0];
        } else {
          use_32x32 = 0;
          for (k = 0; k < 4; k++)
            blk_fw[k] = blk_bestsme[k] < thresh_low    ? 2
                        : blk_bestsme[k] < thresh_high ? 1
                                                       : 0;
        }

        for (k = 0; k < 4; k++) {
          switch (abs(frame - alt_ref_index)) {
            case 1: blk_fw[k] = VPXMIN(blk_fw[k], 2); break;
            case 2:
            case 3: blk_fw[k] = VPXMIN(blk_fw[k], 1); break;
            default: break;
          }
        }
      }

      if (blk_fw[0] | blk_fw[1] | blk_fw[2] | blk_fw[3]) {
        // Construct the predictors
        temporal_filter_predictors_mb_c(
            mbd, frames[frame]->y_buffer + mb_y_offset,
            frames[frame]->u_buffer + mb_uv_offset,
            frames[frame]->v_buffer + mb_uv_offset, frames[frame]->y_stride,
            mb_uv_width, mb_uv_height, ref_mv.row, ref_mv.col, predictor, scale,
            mb_col * BW, mb_row * BH, blk_mvs, use_32x32);

#if CONFIG_VP9_HIGHBITDEPTH
        if (mbd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
          int adj_strength = strength + 2 * (mbd->bd - 8);
          // Apply the filter (YUV)
          vp9_highbd_apply_temporal_filter(
              CONVERT_TO_SHORTPTR(f->y_buffer + mb_y_offset), f->y_stride,
              CONVERT_TO_SHORTPTR(predictor), BW,
              CONVERT_TO_SHORTPTR(f->u_buffer + mb_uv_offset),
              CONVERT_TO_SHORTPTR(f->v_buffer + mb_uv_offset), f->uv_stride,
              CONVERT_TO_SHORTPTR(predictor + BLK_PELS),
              CONVERT_TO_SHORTPTR(predictor + (BLK_PELS << 1)), mb_uv_width, BW,
              BH, mbd->plane[1].subsampling_x, mbd->plane[1].subsampling_y,
              adj_strength, blk_fw, use_32x32, accumulator, count,
              accumulator + BLK_PELS, count + BLK_PELS,
              accumulator + (BLK_PELS << 1), count + (BLK_PELS << 1));
        } else {
          // Apply the filter (YUV)
          vp9_apply_temporal_filter(
              f->y_buffer + mb_y_offset, f->y_stride, predictor, BW,
              f->u_buffer + mb_uv_offset, f->v_buffer + mb_uv_offset,
              f->uv_stride, predictor + BLK_PELS, predictor + (BLK_PELS << 1),
              mb_uv_width, BW, BH, mbd->plane[1].subsampling_x,
              mbd->plane[1].subsampling_y, strength, blk_fw, use_32x32,
              accumulator, count, accumulator + BLK_PELS, count + BLK_PELS,
              accumulator + (BLK_PELS << 1), count + (BLK_PELS << 1));
        }
#else
        // Apply the filter (YUV)
        vp9_apply_temporal_filter(
            f->y_buffer + mb_y_offset, f->y_stride, predictor, BW,
            f->u_buffer + mb_uv_offset, f->v_buffer + mb_uv_offset,
            f->uv_stride, predictor + BLK_PELS, predictor + (BLK_PELS << 1),
            mb_uv_width, BW, BH, mbd->plane[1].subsampling_x,
            mbd->plane[1].subsampling_y, strength, blk_fw, use_32x32,
            accumulator, count, accumulator + BLK_PELS, count + BLK_PELS,
            accumulator + (BLK_PELS << 1), count + (BLK_PELS << 1));
#endif  // CONFIG_VP9_HIGHBITDEPTH
      }
    }

#if CONFIG_VP9_HIGHBITDEPTH
    if (mbd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      uint16_t *dst1_16;
      uint16_t *dst2_16;
      // Normalize filter output to produce AltRef frame
      dst1 = dst->y_buffer;
      dst1_16 = CONVERT_TO_SHORTPTR(dst1);
      stride = dst->y_stride;
      byte = mb_y_offset;
      for (i = 0, k = 0; i < BH; i++) {
        for (j = 0; j < BW; j++, k++) {
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= fixed_divide[count[k]];
          pval >>= 19;

          dst1_16[byte] = (uint16_t)pval;

          // move to next pixel
          byte++;
        }

        byte += stride - BW;
      }

      dst1 = dst->u_buffer;
      dst2 = dst->v_buffer;
      dst1_16 = CONVERT_TO_SHORTPTR(dst1);
      dst2_16 = CONVERT_TO_SHORTPTR(dst2);
      stride = dst->uv_stride;
      byte = mb_uv_offset;
      for (i = 0, k = BLK_PELS; i < mb_uv_height; i++) {
        for (j = 0; j < mb_uv_width; j++, k++) {
          int m = k + BLK_PELS;

          // U
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= fixed_divide[count[k]];
          pval >>= 19;
          dst1_16[byte] = (uint16_t)pval;

          // V
          pval = accumulator[m] + (count[m] >> 1);
          pval *= fixed_divide[count[m]];
          pval >>= 19;
          dst2_16[byte] = (uint16_t)pval;

          // move to next pixel
          byte++;
        }

        byte += stride - mb_uv_width;
      }
    } else {
      // Normalize filter output to produce AltRef frame
      dst1 = dst->y_buffer;
      stride = dst->y_stride;
      byte = mb_y_offset;
      for (i = 0, k = 0; i < BH; i++) {
        for (j = 0; j < BW; j++, k++) {
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= fixed_divide[count[k]];
          pval >>= 19;

          dst1[byte] = (uint8_t)pval;

          // move to next pixel
          byte++;
        }
        byte += stride - BW;
      }

      dst1 = dst->u_buffer;
      dst2 = dst->v_buffer;
      stride = dst->uv_stride;
      byte = mb_uv_offset;
      for (i = 0, k = BLK_PELS; i < mb_uv_height; i++) {
        for (j = 0; j < mb_uv_width; j++, k++) {
          int m = k + BLK_PELS;

          // U
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= fixed_divide[count[k]];
          pval >>= 19;
          dst1[byte] = (uint8_t)pval;

          // V
          pval = accumulator[m] + (count[m] >> 1);
          pval *= fixed_divide[count[m]];
          pval >>= 19;
          dst2[byte] = (uint8_t)pval;

          // move to next pixel
          byte++;
        }
        byte += stride - mb_uv_width;
      }
    }
#else
    // Normalize filter output to produce AltRef frame
    dst1 = dst->y_buffer;
    stride = dst->y_stride;
    byte = mb_y_offset;
    for (i = 0, k = 0; i < BH; i++) {
      for (j = 0; j < BW; j++, k++) {
        unsigned int pval = accumulator[k] + (count[k] >> 1);
        pval *= fixed_divide[count[k]];
        pval >>= 19;

        dst1[byte] = (uint8_t)pval;

        // move to next pixel
        byte++;
      }
      byte += stride - BW;
    }

    dst1 = dst->u_buffer;
    dst2 = dst->v_buffer;
    stride = dst->uv_stride;
    byte = mb_uv_offset;
    for (i = 0, k = BLK_PELS; i < mb_uv_height; i++) {
      for (j = 0; j < mb_uv_width; j++, k++) {
        int m = k + BLK_PELS;

        // U
        unsigned int pval = accumulator[k] + (count[k] >> 1);
        pval *= fixed_divide[count[k]];
        pval >>= 19;
        dst1[byte] = (uint8_t)pval;

        // V
        pval = accumulator[m] + (count[m] >> 1);
        pval *= fixed_divide[count[m]];
        pval >>= 19;
        dst2[byte] = (uint8_t)pval;

        // move to next pixel
        byte++;
      }
      byte += stride - mb_uv_width;
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    mb_y_offset += BW;
    mb_uv_offset += mb_uv_width;
  }
}

static void temporal_filter_iterate_tile_c(VP9_COMP *cpi, int tile_row,
                                           int tile_col) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  TileInfo *tile_info =
      &cpi->tile_data[tile_row * tile_cols + tile_col].tile_info;
  const int mb_row_start = (tile_info->mi_row_start) >> TF_SHIFT;
  const int mb_row_end = (tile_info->mi_row_end + TF_ROUND) >> TF_SHIFT;
  const int mb_col_start = (tile_info->mi_col_start) >> TF_SHIFT;
  const int mb_col_end = (tile_info->mi_col_end + TF_ROUND) >> TF_SHIFT;
  int mb_row;

  for (mb_row = mb_row_start; mb_row < mb_row_end; mb_row++) {
    vp9_temporal_filter_iterate_row_c(cpi, &cpi->td, mb_row, mb_col_start,
                                      mb_col_end);
  }
}

static void temporal_filter_iterate_c(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  int tile_row, tile_col;
  vp9_init_tile_data(cpi);

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      temporal_filter_iterate_tile_c(cpi, tile_row, tile_col);
    }
  }
}

// Apply buffer limits and context specific adjustments to arnr filter.
static void adjust_arnr_filter(VP9_COMP *cpi, int distance, int group_boost,
                               int *arnr_frames, int *frames_backward,
                               int *frames_forward, int *arnr_strength) {
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;

  int max_fwd =
      VPXMAX((int)vp9_lookahead_depth(cpi->lookahead) - distance - 1, 0);
  int max_bwd = VPXMAX(distance, 0);
  int frames = VPXMAX(oxcf->arnr_max_frames, 1);
  int q, base_strength, strength;

  // Context dependent two pass adjustment to strength.
  if (oxcf->pass == 2) {
    base_strength = oxcf->arnr_strength + cpi->twopass.arnr_strength_adjustment;
    // Clip to allowed range.
    base_strength = clamp(base_strength, 0, 6);
  } else {
    base_strength = oxcf->arnr_strength;
  }

  // Adjust the strength based on active max q.
  if (cpi->common.current_video_frame > 1)
    q = ((int)vp9_convert_qindex_to_q(cpi->rc.avg_frame_qindex[INTER_FRAME],
                                      cpi->common.bit_depth));
  else
    q = ((int)vp9_convert_qindex_to_q(cpi->rc.avg_frame_qindex[KEY_FRAME],
                                      cpi->common.bit_depth));
  if (q > 16) {
    strength = base_strength;
  } else {
    strength = base_strength - ((16 - q) / 2);
    if (strength < 0) strength = 0;
  }

  // Adjust number of frames in filter and strength based on gf boost level.
  frames = VPXMIN(frames, group_boost / 150);

  if (strength > group_boost / 300) {
    strength = group_boost / 300;
  }

  if (VPXMIN(max_fwd, max_bwd) >= frames / 2) {
    // Handle the even/odd case.
    *frames_backward = frames / 2;
    *frames_forward = (frames - 1) / 2;
  } else {
    if (max_fwd < frames / 2) {
      *frames_forward = max_fwd;
      *frames_backward = VPXMIN(frames - 1 - *frames_forward, max_bwd);
    } else {
      *frames_backward = max_bwd;
      *frames_forward = VPXMIN(frames - 1 - *frames_backward, max_fwd);
    }
  }

  // Set the baseline active filter size.
  frames = *frames_backward + 1 + *frames_forward;

  // Adjustments for second level arf in multi arf case.
  // Leave commented out place holder for possible filtering adjustment with
  // new multi-layer arf code.
  // if (cpi->oxcf.pass == 2 && cpi->multi_arf_allowed)
  //   if (gf_group->rf_level[gf_group->index] != GF_ARF_STD) strength >>= 1;

  // TODO(jingning): Skip temporal filtering for intermediate frames that will
  // be used as show_existing_frame. Need to further explore the possibility to
  // apply certain filter.
  if (frames <= 1) {
    frames = 1;
    *frames_backward = 0;
    *frames_forward = 0;
  }

  *arnr_frames = frames;
  *arnr_strength = strength;
}

void vp9_temporal_filter(VP9_COMP *cpi, int distance) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  ARNRFilterData *arnr_filter_data = &cpi->arnr_filter_data;
  int frame;
  int frames_to_blur;
  int start_frame;
  int strength;
  int frames_to_blur_backward;
  int frames_to_blur_forward;
  struct scale_factors *sf = &arnr_filter_data->sf;
  YV12_BUFFER_CONFIG **frames = arnr_filter_data->frames;
  int rdmult;

  // Apply context specific adjustments to the arnr filter parameters.
  adjust_arnr_filter(cpi, distance, rc->gfu_boost, &frames_to_blur,
                     &frames_to_blur_backward, &frames_to_blur_forward,
                     &strength);
  start_frame = distance + frames_to_blur_forward;

  arnr_filter_data->strength = strength;
  arnr_filter_data->frame_count = frames_to_blur;
  arnr_filter_data->alt_ref_index = frames_to_blur_backward;
  arnr_filter_data->dst = &cpi->tf_buffer;

  // Setup frame pointers, NULL indicates frame not included in filter.
  for (frame = 0; frame < frames_to_blur; ++frame) {
    const int which_buffer = start_frame - frame;
    struct lookahead_entry *buf =
        vp9_lookahead_peek(cpi->lookahead, which_buffer);
    frames[frames_to_blur - 1 - frame] = &buf->img;
  }

  YV12_BUFFER_CONFIG *f = frames[arnr_filter_data->alt_ref_index];
  xd->cur_buf = f;
  xd->plane[1].subsampling_y = f->subsampling_y;
  xd->plane[1].subsampling_x = f->subsampling_x;

  if (frames_to_blur > 0) {
    // Setup scaling factors. Scaling on each of the arnr frames is not
    // supported.
    if (cpi->use_svc) {
      // In spatial svc the scaling factors might be less then 1/2.
      // So we will use non-normative scaling.
      int frame_used = 0;
#if CONFIG_VP9_HIGHBITDEPTH
      vp9_setup_scale_factors_for_frame(
          sf, get_frame_new_buffer(cm)->y_crop_width,
          get_frame_new_buffer(cm)->y_crop_height,
          get_frame_new_buffer(cm)->y_crop_width,
          get_frame_new_buffer(cm)->y_crop_height, cm->use_highbitdepth);
#else
      vp9_setup_scale_factors_for_frame(
          sf, get_frame_new_buffer(cm)->y_crop_width,
          get_frame_new_buffer(cm)->y_crop_height,
          get_frame_new_buffer(cm)->y_crop_width,
          get_frame_new_buffer(cm)->y_crop_height);
#endif  // CONFIG_VP9_HIGHBITDEPTH

      for (frame = 0; frame < frames_to_blur; ++frame) {
        if (cm->mi_cols * MI_SIZE != frames[frame]->y_width ||
            cm->mi_rows * MI_SIZE != frames[frame]->y_height) {
          if (vpx_realloc_frame_buffer(&cpi->svc.scaled_frames[frame_used],
                                       cm->width, cm->height, cm->subsampling_x,
                                       cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                       cm->use_highbitdepth,
#endif
                                       VP9_ENC_BORDER_IN_PIXELS,
                                       cm->byte_alignment, NULL, NULL, NULL)) {
            vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                               "Failed to reallocate alt_ref_buffer");
          }
          frames[frame] = vp9_scale_if_required(
              cm, frames[frame], &cpi->svc.scaled_frames[frame_used], 0,
              EIGHTTAP, 0);
          ++frame_used;
        }
      }
      cm->mi = cm->mip + cm->mi_stride + 1;
      xd->mi = cm->mi_grid_visible;
      xd->mi[0] = cm->mi;
    } else {
// ARF is produced at the native frame size and resized when coded.
#if CONFIG_VP9_HIGHBITDEPTH
      vp9_setup_scale_factors_for_frame(
          sf, frames[0]->y_crop_width, frames[0]->y_crop_height,
          frames[0]->y_crop_width, frames[0]->y_crop_height,
          cm->use_highbitdepth);
#else
      vp9_setup_scale_factors_for_frame(
          sf, frames[0]->y_crop_width, frames[0]->y_crop_height,
          frames[0]->y_crop_width, frames[0]->y_crop_height);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    }
  }

  // Initialize errorperbit and sabperbit.
  rdmult = vp9_compute_rd_mult_based_on_qindex(cpi, ARNR_FILT_QINDEX);
  set_error_per_bit(&cpi->td.mb, rdmult);
  vp9_initialize_me_consts(cpi, &cpi->td.mb, ARNR_FILT_QINDEX);

  if (!cpi->row_mt)
    temporal_filter_iterate_c(cpi);
  else
    vp9_temporal_filter_row_mt(cpi);
}
