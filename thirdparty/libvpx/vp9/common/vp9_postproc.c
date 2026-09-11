/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "./vp9_rtcd.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/postproc.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/system_state.h"
#include "vpx_scale/vpx_scale.h"
#include "vpx_scale/yv12config.h"

#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_postproc.h"

#if CONFIG_VP9_POSTPROC

static const uint8_t q_diff_thresh = 20;
static const uint8_t last_q_thresh = 170;
extern const int16_t vpx_rv[];

#if CONFIG_VP9_HIGHBITDEPTH
static const int16_t kernel5[] = { 1, 1, 4, 1, 1 };

void vp9_highbd_post_proc_down_and_across_c(const uint16_t *src_ptr,
                                            uint16_t *dst_ptr,
                                            int src_pixels_per_line,
                                            int dst_pixels_per_line, int rows,
                                            int cols, int flimit) {
  uint16_t const *p_src;
  uint16_t *p_dst;
  int row, col, i, v, kernel;
  int pitch = src_pixels_per_line;
  uint16_t d[8];

  for (row = 0; row < rows; row++) {
    // post_proc_down for one row.
    p_src = src_ptr;
    p_dst = dst_ptr;

    for (col = 0; col < cols; col++) {
      kernel = 4;
      v = p_src[col];

      for (i = -2; i <= 2; i++) {
        if (abs(v - p_src[col + i * pitch]) > flimit) goto down_skip_convolve;

        kernel += kernel5[2 + i] * p_src[col + i * pitch];
      }

      v = (kernel >> 3);

    down_skip_convolve:
      p_dst[col] = v;
    }

    /* now post_proc_across */
    p_src = dst_ptr;
    p_dst = dst_ptr;

    for (i = 0; i < 8; i++) d[i] = p_src[i];

    for (col = 0; col < cols; col++) {
      kernel = 4;
      v = p_src[col];

      d[col & 7] = v;

      for (i = -2; i <= 2; i++) {
        if (abs(v - p_src[col + i]) > flimit) goto across_skip_convolve;

        kernel += kernel5[2 + i] * p_src[col + i];
      }

      d[col & 7] = (kernel >> 3);

    across_skip_convolve:
      if (col >= 2) p_dst[col - 2] = d[(col - 2) & 7];
    }

    /* handle the last two pixels */
    p_dst[col - 2] = d[(col - 2) & 7];
    p_dst[col - 1] = d[(col - 1) & 7];

    /* next row */
    src_ptr += pitch;
    dst_ptr += dst_pixels_per_line;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static int q2mbl(int x) {
  if (x < 20) x = 20;

  x = 50 + (x - 50) * 10 / 8;
  return x * x / 3;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_mbpost_proc_across_ip_c(uint16_t *src, int pitch, int rows,
                                        int cols, int flimit) {
  int r, c, i;

  uint16_t *s = src;
  uint16_t d[16];

  for (r = 0; r < rows; r++) {
    int64_t sumsq = 0;
    int64_t sum = 0;

    for (i = -8; i <= 6; i++) {
      sumsq += s[i] * s[i];
      sum += s[i];
      d[i + 8] = 0;
    }

    for (c = 0; c < cols + 8; c++) {
      int x = s[c + 7] - s[c - 8];
      int y = s[c + 7] + s[c - 8];

      sum += x;
      sumsq += x * y;

      d[c & 15] = s[c];

      if (sumsq * 15 - sum * sum < flimit) {
        d[c & 15] = (8 + sum + s[c]) >> 4;
      }

      s[c - 8] = d[(c - 8) & 15];
    }

    s += pitch;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_mbpost_proc_down_c(uint16_t *dst, int pitch, int rows, int cols,
                                   int flimit) {
  int r, c, i;
  const int16_t *rv3 = &vpx_rv[63 & rand()];  // NOLINT

  for (c = 0; c < cols; c++) {
    uint16_t *s = &dst[c];
    int64_t sumsq = 0;
    int64_t sum = 0;
    uint16_t d[16];
    const int16_t *rv2 = rv3 + ((c * 17) & 127);

    for (i = -8; i <= 6; i++) {
      sumsq += s[i * pitch] * s[i * pitch];
      sum += s[i * pitch];
    }

    for (r = 0; r < rows + 8; r++) {
      sumsq += s[7 * pitch] * s[7 * pitch] - s[-8 * pitch] * s[-8 * pitch];
      sum += s[7 * pitch] - s[-8 * pitch];
      d[r & 15] = s[0];

      if (sumsq * 15 - sum * sum < flimit) {
        d[r & 15] = (rv2[r & 127] + sum + s[0]) >> 4;
      }

      s[-8 * pitch] = d[(r - 8) & 15];
      s += pitch;
    }
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void deblock_and_de_macro_block(VP9_COMMON *cm,
                                       YV12_BUFFER_CONFIG *source,
                                       YV12_BUFFER_CONFIG *post, int q,
                                       int low_var_thresh, int flag,
                                       uint8_t *limits) {
  (void)low_var_thresh;
  (void)flag;
#if CONFIG_VP9_HIGHBITDEPTH
  if (source->flags & YV12_FLAG_HIGHBITDEPTH) {
    double level = 6.0e-05 * q * q * q - .0067 * q * q + .306 * q + .0065;
    int ppl = (int)(level + .5);
    vp9_highbd_post_proc_down_and_across(
        CONVERT_TO_SHORTPTR(source->y_buffer),
        CONVERT_TO_SHORTPTR(post->y_buffer), source->y_stride, post->y_stride,
        source->y_height, source->y_width, ppl);

    vp9_highbd_mbpost_proc_across_ip(CONVERT_TO_SHORTPTR(post->y_buffer),
                                     post->y_stride, post->y_height,
                                     post->y_width, q2mbl(q));

    vp9_highbd_mbpost_proc_down(CONVERT_TO_SHORTPTR(post->y_buffer),
                                post->y_stride, post->y_height, post->y_width,
                                q2mbl(q));

    vp9_highbd_post_proc_down_and_across(
        CONVERT_TO_SHORTPTR(source->u_buffer),
        CONVERT_TO_SHORTPTR(post->u_buffer), source->uv_stride, post->uv_stride,
        source->uv_height, source->uv_width, ppl);
    vp9_highbd_post_proc_down_and_across(
        CONVERT_TO_SHORTPTR(source->v_buffer),
        CONVERT_TO_SHORTPTR(post->v_buffer), source->uv_stride, post->uv_stride,
        source->uv_height, source->uv_width, ppl);
  } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
    vp9_deblock(cm, source, post, q, limits);
    vpx_mbpost_proc_across_ip(post->y_buffer, post->y_stride, post->y_height,
                              post->y_width, q2mbl(q));
    vpx_mbpost_proc_down(post->y_buffer, post->y_stride, post->y_height,
                         post->y_width, q2mbl(q));
#if CONFIG_VP9_HIGHBITDEPTH
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

void vp9_deblock(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits) {
  const int ppl =
      (int)(6.0e-05 * q * q * q - 0.0067 * q * q + 0.306 * q + 0.0065 + 0.5);
#if CONFIG_VP9_HIGHBITDEPTH
  if (src->flags & YV12_FLAG_HIGHBITDEPTH) {
    int i;
    const uint8_t *const srcs[3] = { src->y_buffer, src->u_buffer,
                                     src->v_buffer };
    const int src_strides[3] = { src->y_stride, src->uv_stride,
                                 src->uv_stride };
    const int src_widths[3] = { src->y_width, src->uv_width, src->uv_width };
    const int src_heights[3] = { src->y_height, src->uv_height,
                                 src->uv_height };

    uint8_t *const dsts[3] = { dst->y_buffer, dst->u_buffer, dst->v_buffer };
    const int dst_strides[3] = { dst->y_stride, dst->uv_stride,
                                 dst->uv_stride };
    for (i = 0; i < MAX_MB_PLANE; ++i) {
      vp9_highbd_post_proc_down_and_across(
          CONVERT_TO_SHORTPTR(srcs[i]), CONVERT_TO_SHORTPTR(dsts[i]),
          src_strides[i], dst_strides[i], src_heights[i], src_widths[i], ppl);
    }
  } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
    int mbr;
    const int mb_rows = cm->mb_rows;
    memset(limits, (unsigned char)ppl, cm->postproc_state.limits_size);

    for (mbr = 0; mbr < mb_rows; mbr++) {
      vpx_post_proc_down_and_across_mb_row(
          src->y_buffer + 16 * mbr * src->y_stride,
          dst->y_buffer + 16 * mbr * dst->y_stride, src->y_stride,
          dst->y_stride, src->y_width, limits, 16);
      vpx_post_proc_down_and_across_mb_row(
          src->u_buffer + 8 * mbr * src->uv_stride,
          dst->u_buffer + 8 * mbr * dst->uv_stride, src->uv_stride,
          dst->uv_stride, src->uv_width, limits, 8);
      vpx_post_proc_down_and_across_mb_row(
          src->v_buffer + 8 * mbr * src->uv_stride,
          dst->v_buffer + 8 * mbr * dst->uv_stride, src->uv_stride,
          dst->uv_stride, src->uv_width, limits, 8);
    }
#if CONFIG_VP9_HIGHBITDEPTH
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

void vp9_denoise(struct VP9Common *cm, const YV12_BUFFER_CONFIG *src,
                 YV12_BUFFER_CONFIG *dst, int q, uint8_t *limits) {
  vp9_deblock(cm, src, dst, q, limits);
}

static void swap_mi_and_prev_mi(VP9_COMMON *cm) {
  // Current mip will be the prev_mip for the next frame.
  MODE_INFO *temp = cm->postproc_state.prev_mip;
  cm->postproc_state.prev_mip = cm->mip;
  cm->mip = temp;

  // Update the upper left visible macroblock ptrs.
  cm->mi = cm->mip + cm->mi_stride + 1;
  cm->postproc_state.prev_mi = cm->postproc_state.prev_mip + cm->mi_stride + 1;
}

int vp9_post_proc_frame(struct VP9Common *cm, YV12_BUFFER_CONFIG *dest,
                        vp9_ppflags_t *ppflags, int unscaled_width) {
  const int q = VPXMIN(105, cm->lf.filter_level * 2);
  const int flags = ppflags->post_proc_flag;
  YV12_BUFFER_CONFIG *const ppbuf = &cm->post_proc_buffer;
  struct postproc_state *const ppstate = &cm->postproc_state;
  ppstate->limits_size = unscaled_width;

  if (!cm->frame_to_show) return -1;

  if (!flags) {
    *dest = *cm->frame_to_show;
    return 0;
  }

  vpx_clear_system_state();

  // Alloc memory for prev_mip in the first frame.
  if (cm->current_video_frame == 1) {
    ppstate->last_base_qindex = cm->base_qindex;
    ppstate->last_frame_valid = 1;
  }

  if ((flags & VP9D_MFQE) && ppstate->prev_mip == NULL) {
    ppstate->prev_mip = vpx_calloc(cm->mi_alloc_size, sizeof(*cm->mip));
    if (!ppstate->prev_mip) {
      return 1;
    }
    ppstate->prev_mi = ppstate->prev_mip + cm->mi_stride + 1;
  }

  // Allocate post_proc_buffer_int if needed.
  if ((flags & VP9D_MFQE) && !cm->post_proc_buffer_int.buffer_alloc) {
    if ((flags & VP9D_DEMACROBLOCK) || (flags & VP9D_DEBLOCK)) {
      const int width = ALIGN_POWER_OF_TWO(cm->width, 4);
      const int height = ALIGN_POWER_OF_TWO(cm->height, 4);

      if (vpx_alloc_frame_buffer(&cm->post_proc_buffer_int, width, height,
                                 cm->subsampling_x, cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                 cm->use_highbitdepth,
#endif  // CONFIG_VP9_HIGHBITDEPTH
                                 VP9_ENC_BORDER_IN_PIXELS,
                                 cm->byte_alignment) < 0) {
        vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate MFQE framebuffer");
      }

      // Ensure that postproc is set to flat image so that post proc
      // doesn't pull random data in from edge.
      memset(cm->post_proc_buffer_int.buffer_alloc, 128,
             cm->post_proc_buffer_int.frame_size);
    }
  }

  if (vpx_realloc_frame_buffer(&cm->post_proc_buffer, cm->width, cm->height,
                               cm->subsampling_x, cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
                               cm->use_highbitdepth,
#endif
                               VP9_DEC_BORDER_IN_PIXELS, cm->byte_alignment,
                               NULL, NULL, NULL) < 0) {
    vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate post-processing buffer");
  }
  memset(cm->post_proc_buffer.buffer_alloc, 128,
         cm->post_proc_buffer.frame_size);

  if (flags & (VP9D_DEMACROBLOCK | VP9D_DEBLOCK)) {
    if (!cm->postproc_state.limits) {
      cm->postproc_state.limits =
          vpx_calloc(ppstate->limits_size, sizeof(*cm->postproc_state.limits));
      if (!cm->postproc_state.limits) return 1;
    }
  }

  if (flags & VP9D_ADDNOISE) {
    if (!cm->postproc_state.generated_noise) {
      cm->postproc_state.generated_noise = vpx_calloc(
          cm->width + 256, sizeof(*cm->postproc_state.generated_noise));
      if (!cm->postproc_state.generated_noise) return 1;
    }
  }

  if ((flags & VP9D_MFQE) && cm->current_video_frame >= 2 &&
      ppstate->last_frame_valid && cm->bit_depth == 8 &&
      ppstate->last_base_qindex <= last_q_thresh &&
      cm->base_qindex - ppstate->last_base_qindex >= q_diff_thresh) {
    vp9_mfqe(cm);
    // TODO(jackychen): Consider whether enable deblocking by default
    // if mfqe is enabled. Need to take both the quality and the speed
    // into consideration.
    if ((flags & VP9D_DEMACROBLOCK) || (flags & VP9D_DEBLOCK)) {
      vpx_yv12_copy_frame(ppbuf, &cm->post_proc_buffer_int);
    }
    if ((flags & VP9D_DEMACROBLOCK) && cm->post_proc_buffer_int.buffer_alloc) {
      deblock_and_de_macro_block(cm, &cm->post_proc_buffer_int, ppbuf,
                                 q + (ppflags->deblocking_level - 5) * 10, 1, 0,
                                 cm->postproc_state.limits);
    } else if (flags & VP9D_DEBLOCK) {
      vp9_deblock(cm, &cm->post_proc_buffer_int, ppbuf, q,
                  cm->postproc_state.limits);
    } else {
      vpx_yv12_copy_frame(&cm->post_proc_buffer_int, ppbuf);
    }
  } else if (flags & VP9D_DEMACROBLOCK) {
    deblock_and_de_macro_block(cm, cm->frame_to_show, ppbuf,
                               q + (ppflags->deblocking_level - 5) * 10, 1, 0,
                               cm->postproc_state.limits);
  } else if (flags & VP9D_DEBLOCK) {
    vp9_deblock(cm, cm->frame_to_show, ppbuf, q, cm->postproc_state.limits);
  } else {
    vpx_yv12_copy_frame(cm->frame_to_show, ppbuf);
  }

  ppstate->last_base_qindex = cm->base_qindex;
  ppstate->last_frame_valid = 1;
  if (flags & VP9D_ADDNOISE) {
    const int noise_level = ppflags->noise_level;
    if (ppstate->last_q != q || ppstate->last_noise != noise_level) {
      double sigma;
      vpx_clear_system_state();
      sigma = noise_level + .5 + .6 * q / 63.0;
      ppstate->clamp =
          vpx_setup_noise(sigma, ppstate->generated_noise, cm->width + 256);
      ppstate->last_q = q;
      ppstate->last_noise = noise_level;
    }
    vpx_plane_add_noise(ppbuf->y_buffer, ppstate->generated_noise,
                        ppstate->clamp, ppstate->clamp, ppbuf->y_width,
                        ppbuf->y_height, ppbuf->y_stride);
  }

  *dest = *ppbuf;

  /* handle problem with extending borders */
  dest->y_width = cm->width;
  dest->y_height = cm->height;
  dest->uv_width = dest->y_width >> cm->subsampling_x;
  dest->uv_height = dest->y_height >> cm->subsampling_y;

  if (flags & VP9D_MFQE) swap_mi_and_prev_mi(cm);
  return 0;
}
#endif  // CONFIG_VP9_POSTPROC
