/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>

#include "denoising.h"

#include "vp8/common/reconinter.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8_rtcd.h"

static const unsigned int NOISE_MOTION_THRESHOLD = 25 * 25;
/* SSE_DIFF_THRESHOLD is selected as ~95% confidence assuming
 * var(noise) ~= 100.
 */
static const unsigned int SSE_DIFF_THRESHOLD = 16 * 16 * 20;
static const unsigned int SSE_THRESHOLD = 16 * 16 * 40;
static const unsigned int SSE_THRESHOLD_HIGH = 16 * 16 * 80;

/*
 * The filter function was modified to reduce the computational complexity.
 * Step 1:
 * Instead of applying tap coefficients for each pixel, we calculated the
 * pixel adjustments vs. pixel diff value ahead of time.
 *     adjustment = filtered_value - current_raw
 *                = (filter_coefficient * diff + 128) >> 8
 * where
 *     filter_coefficient = (255 << 8) / (256 + ((absdiff * 330) >> 3));
 *     filter_coefficient += filter_coefficient /
 *                           (3 + motion_magnitude_adjustment);
 *     filter_coefficient is clamped to 0 ~ 255.
 *
 * Step 2:
 * The adjustment vs. diff curve becomes flat very quick when diff increases.
 * This allowed us to use only several levels to approximate the curve without
 * changing the filtering algorithm too much.
 * The adjustments were further corrected by checking the motion magnitude.
 * The levels used are:
 * diff       adjustment w/o motion correction   adjustment w/ motion correction
 * [-255, -16]           -6                                   -7
 * [-15, -8]             -4                                   -5
 * [-7, -4]              -3                                   -4
 * [-3, 3]               diff                                 diff
 * [4, 7]                 3                                    4
 * [8, 15]                4                                    5
 * [16, 255]              6                                    7
 */

int vp8_denoiser_filter_c(unsigned char *mc_running_avg_y, int mc_avg_y_stride,
                          unsigned char *running_avg_y, int avg_y_stride,
                          unsigned char *sig, int sig_stride,
                          unsigned int motion_magnitude,
                          int increase_denoising) {
  unsigned char *running_avg_y_start = running_avg_y;
  unsigned char *sig_start = sig;
  int sum_diff_thresh;
  int r, c;
  int sum_diff = 0;
  int adj_val[3] = { 3, 4, 6 };
  int shift_inc1 = 0;
  int shift_inc2 = 1;
  int col_sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  /* If motion_magnitude is small, making the denoiser more aggressive by
   * increasing the adjustment for each level. Add another increment for
   * blocks that are labeled for increase denoising. */
  if (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) {
    if (increase_denoising) {
      shift_inc1 = 1;
      shift_inc2 = 2;
    }
    adj_val[0] += shift_inc2;
    adj_val[1] += shift_inc2;
    adj_val[2] += shift_inc2;
  }

  for (r = 0; r < 16; ++r) {
    for (c = 0; c < 16; ++c) {
      int diff = 0;
      int adjustment = 0;
      int absdiff = 0;

      diff = mc_running_avg_y[c] - sig[c];
      absdiff = abs(diff);

      // When |diff| <= |3 + shift_inc1|, use pixel value from
      // last denoised raw.
      if (absdiff <= 3 + shift_inc1) {
        running_avg_y[c] = mc_running_avg_y[c];
        col_sum[c] += diff;
      } else {
        if (absdiff >= 4 + shift_inc1 && absdiff <= 7) {
          adjustment = adj_val[0];
        } else if (absdiff >= 8 && absdiff <= 15) {
          adjustment = adj_val[1];
        } else {
          adjustment = adj_val[2];
        }

        if (diff > 0) {
          if ((sig[c] + adjustment) > 255) {
            running_avg_y[c] = 255;
          } else {
            running_avg_y[c] = sig[c] + adjustment;
          }

          col_sum[c] += adjustment;
        } else {
          if ((sig[c] - adjustment) < 0) {
            running_avg_y[c] = 0;
          } else {
            running_avg_y[c] = sig[c] - adjustment;
          }

          col_sum[c] -= adjustment;
        }
      }
    }

    /* Update pointers for next iteration. */
    sig += sig_stride;
    mc_running_avg_y += mc_avg_y_stride;
    running_avg_y += avg_y_stride;
  }

  for (c = 0; c < 16; ++c) {
    // Below we clip the value in the same way which SSE code use.
    // When adopting aggressive denoiser, the adj_val for each pixel
    // could be at most 8 (this is current max adjustment of the map).
    // In SSE code, we calculate the sum of adj_val for
    // the columns, so the sum could be up to 128(16 rows). However,
    // the range of the value is -128 ~ 127 in SSE code, that's why
    // we do this change in C code.
    // We don't do this for UV denoiser, since there are only 8 rows,
    // and max adjustments <= 8, so the sum of the columns will not
    // exceed 64.
    if (col_sum[c] >= 128) {
      col_sum[c] = 127;
    }
    sum_diff += col_sum[c];
  }

  sum_diff_thresh = SUM_DIFF_THRESHOLD;
  if (increase_denoising) sum_diff_thresh = SUM_DIFF_THRESHOLD_HIGH;
  if (abs(sum_diff) > sum_diff_thresh) {
    // Before returning to copy the block (i.e., apply no denoising), check
    // if we can still apply some (weaker) temporal filtering to this block,
    // that would otherwise not be denoised at all. Simplest is to apply
    // an additional adjustment to running_avg_y to bring it closer to sig.
    // The adjustment is capped by a maximum delta, and chosen such that
    // in most cases the resulting sum_diff will be within the
    // accceptable range given by sum_diff_thresh.

    // The delta is set by the excess of absolute pixel diff over threshold.
    int delta = ((abs(sum_diff) - sum_diff_thresh) >> 8) + 1;
    // Only apply the adjustment for max delta up to 3.
    if (delta < 4) {
      sig -= sig_stride * 16;
      mc_running_avg_y -= mc_avg_y_stride * 16;
      running_avg_y -= avg_y_stride * 16;
      for (r = 0; r < 16; ++r) {
        for (c = 0; c < 16; ++c) {
          int diff = mc_running_avg_y[c] - sig[c];
          int adjustment = abs(diff);
          if (adjustment > delta) adjustment = delta;
          if (diff > 0) {
            // Bring denoised signal down.
            if (running_avg_y[c] - adjustment < 0) {
              running_avg_y[c] = 0;
            } else {
              running_avg_y[c] = running_avg_y[c] - adjustment;
            }
            col_sum[c] -= adjustment;
          } else if (diff < 0) {
            // Bring denoised signal up.
            if (running_avg_y[c] + adjustment > 255) {
              running_avg_y[c] = 255;
            } else {
              running_avg_y[c] = running_avg_y[c] + adjustment;
            }
            col_sum[c] += adjustment;
          }
        }
        // TODO(marpan): Check here if abs(sum_diff) has gone below the
        // threshold sum_diff_thresh, and if so, we can exit the row loop.
        sig += sig_stride;
        mc_running_avg_y += mc_avg_y_stride;
        running_avg_y += avg_y_stride;
      }

      sum_diff = 0;
      for (c = 0; c < 16; ++c) {
        if (col_sum[c] >= 128) {
          col_sum[c] = 127;
        }
        sum_diff += col_sum[c];
      }

      if (abs(sum_diff) > sum_diff_thresh) return COPY_BLOCK;
    } else {
      return COPY_BLOCK;
    }
  }

  vp8_copy_mem16x16(running_avg_y_start, avg_y_stride, sig_start, sig_stride);
  return FILTER_BLOCK;
}

int vp8_denoiser_filter_uv_c(unsigned char *mc_running_avg, int mc_avg_stride,
                             unsigned char *running_avg, int avg_stride,
                             unsigned char *sig, int sig_stride,
                             unsigned int motion_magnitude,
                             int increase_denoising) {
  unsigned char *running_avg_start = running_avg;
  unsigned char *sig_start = sig;
  int sum_diff_thresh;
  int r, c;
  int sum_diff = 0;
  int sum_block = 0;
  int adj_val[3] = { 3, 4, 6 };
  int shift_inc1 = 0;
  int shift_inc2 = 1;
  /* If motion_magnitude is small, making the denoiser more aggressive by
   * increasing the adjustment for each level. Add another increment for
   * blocks that are labeled for increase denoising. */
  if (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD_UV) {
    if (increase_denoising) {
      shift_inc1 = 1;
      shift_inc2 = 2;
    }
    adj_val[0] += shift_inc2;
    adj_val[1] += shift_inc2;
    adj_val[2] += shift_inc2;
  }

  // Avoid denoising color signal if its close to average level.
  for (r = 0; r < 8; ++r) {
    for (c = 0; c < 8; ++c) {
      sum_block += sig[c];
    }
    sig += sig_stride;
  }
  if (abs(sum_block - (128 * 8 * 8)) < SUM_DIFF_FROM_AVG_THRESH_UV) {
    return COPY_BLOCK;
  }

  sig -= sig_stride * 8;
  for (r = 0; r < 8; ++r) {
    for (c = 0; c < 8; ++c) {
      int diff = 0;
      int adjustment = 0;
      int absdiff = 0;

      diff = mc_running_avg[c] - sig[c];
      absdiff = abs(diff);

      // When |diff| <= |3 + shift_inc1|, use pixel value from
      // last denoised raw.
      if (absdiff <= 3 + shift_inc1) {
        running_avg[c] = mc_running_avg[c];
        sum_diff += diff;
      } else {
        if (absdiff >= 4 && absdiff <= 7) {
          adjustment = adj_val[0];
        } else if (absdiff >= 8 && absdiff <= 15) {
          adjustment = adj_val[1];
        } else {
          adjustment = adj_val[2];
        }
        if (diff > 0) {
          if ((sig[c] + adjustment) > 255) {
            running_avg[c] = 255;
          } else {
            running_avg[c] = sig[c] + adjustment;
          }
          sum_diff += adjustment;
        } else {
          if ((sig[c] - adjustment) < 0) {
            running_avg[c] = 0;
          } else {
            running_avg[c] = sig[c] - adjustment;
          }
          sum_diff -= adjustment;
        }
      }
    }
    /* Update pointers for next iteration. */
    sig += sig_stride;
    mc_running_avg += mc_avg_stride;
    running_avg += avg_stride;
  }

  sum_diff_thresh = SUM_DIFF_THRESHOLD_UV;
  if (increase_denoising) sum_diff_thresh = SUM_DIFF_THRESHOLD_HIGH_UV;
  if (abs(sum_diff) > sum_diff_thresh) {
    // Before returning to copy the block (i.e., apply no denoising), check
    // if we can still apply some (weaker) temporal filtering to this block,
    // that would otherwise not be denoised at all. Simplest is to apply
    // an additional adjustment to running_avg_y to bring it closer to sig.
    // The adjustment is capped by a maximum delta, and chosen such that
    // in most cases the resulting sum_diff will be within the
    // accceptable range given by sum_diff_thresh.

    // The delta is set by the excess of absolute pixel diff over threshold.
    int delta = ((abs(sum_diff) - sum_diff_thresh) >> 8) + 1;
    // Only apply the adjustment for max delta up to 3.
    if (delta < 4) {
      sig -= sig_stride * 8;
      mc_running_avg -= mc_avg_stride * 8;
      running_avg -= avg_stride * 8;
      for (r = 0; r < 8; ++r) {
        for (c = 0; c < 8; ++c) {
          int diff = mc_running_avg[c] - sig[c];
          int adjustment = abs(diff);
          if (adjustment > delta) adjustment = delta;
          if (diff > 0) {
            // Bring denoised signal down.
            if (running_avg[c] - adjustment < 0) {
              running_avg[c] = 0;
            } else {
              running_avg[c] = running_avg[c] - adjustment;
            }
            sum_diff -= adjustment;
          } else if (diff < 0) {
            // Bring denoised signal up.
            if (running_avg[c] + adjustment > 255) {
              running_avg[c] = 255;
            } else {
              running_avg[c] = running_avg[c] + adjustment;
            }
            sum_diff += adjustment;
          }
        }
        // TODO(marpan): Check here if abs(sum_diff) has gone below the
        // threshold sum_diff_thresh, and if so, we can exit the row loop.
        sig += sig_stride;
        mc_running_avg += mc_avg_stride;
        running_avg += avg_stride;
      }
      if (abs(sum_diff) > sum_diff_thresh) return COPY_BLOCK;
    } else {
      return COPY_BLOCK;
    }
  }

  vp8_copy_mem8x8(running_avg_start, avg_stride, sig_start, sig_stride);
  return FILTER_BLOCK;
}

void vp8_denoiser_set_parameters(VP8_DENOISER *denoiser, int mode) {
  assert(mode > 0);  // Denoiser is allocated only if mode > 0.
  if (mode == 1) {
    denoiser->denoiser_mode = kDenoiserOnYOnly;
  } else if (mode == 2) {
    denoiser->denoiser_mode = kDenoiserOnYUV;
  } else if (mode == 3) {
    denoiser->denoiser_mode = kDenoiserOnYUVAggressive;
  } else {
    denoiser->denoiser_mode = kDenoiserOnYUV;
  }
  if (denoiser->denoiser_mode != kDenoiserOnYUVAggressive) {
    denoiser->denoise_pars.scale_sse_thresh = 1;
    denoiser->denoise_pars.scale_motion_thresh = 8;
    denoiser->denoise_pars.scale_increase_filter = 0;
    denoiser->denoise_pars.denoise_mv_bias = 95;
    denoiser->denoise_pars.pickmode_mv_bias = 100;
    denoiser->denoise_pars.qp_thresh = 0;
    denoiser->denoise_pars.consec_zerolast = UINT_MAX;
    denoiser->denoise_pars.spatial_blur = 0;
  } else {
    denoiser->denoise_pars.scale_sse_thresh = 2;
    denoiser->denoise_pars.scale_motion_thresh = 16;
    denoiser->denoise_pars.scale_increase_filter = 1;
    denoiser->denoise_pars.denoise_mv_bias = 60;
    denoiser->denoise_pars.pickmode_mv_bias = 75;
    denoiser->denoise_pars.qp_thresh = 80;
    denoiser->denoise_pars.consec_zerolast = 15;
    denoiser->denoise_pars.spatial_blur = 0;
  }
}

int vp8_denoiser_allocate(VP8_DENOISER *denoiser, int width, int height,
                          int num_mb_rows, int num_mb_cols, int mode) {
  int i;
  assert(denoiser);
  denoiser->num_mb_cols = num_mb_cols;

  for (i = 0; i < MAX_REF_FRAMES; ++i) {
    denoiser->yv12_running_avg[i].flags = 0;

    if (vp8_yv12_alloc_frame_buffer(&(denoiser->yv12_running_avg[i]), width,
                                    height, VP8BORDERINPIXELS) < 0) {
      vp8_denoiser_free(denoiser);
      return 1;
    }
    memset(denoiser->yv12_running_avg[i].buffer_alloc, 0,
           denoiser->yv12_running_avg[i].frame_size);
  }
  denoiser->yv12_mc_running_avg.flags = 0;

  if (vp8_yv12_alloc_frame_buffer(&(denoiser->yv12_mc_running_avg), width,
                                  height, VP8BORDERINPIXELS) < 0) {
    vp8_denoiser_free(denoiser);
    return 1;
  }

  memset(denoiser->yv12_mc_running_avg.buffer_alloc, 0,
         denoiser->yv12_mc_running_avg.frame_size);

  if (vp8_yv12_alloc_frame_buffer(&denoiser->yv12_last_source, width, height,
                                  VP8BORDERINPIXELS) < 0) {
    vp8_denoiser_free(denoiser);
    return 1;
  }
  memset(denoiser->yv12_last_source.buffer_alloc, 0,
         denoiser->yv12_last_source.frame_size);

  denoiser->denoise_state = vpx_calloc((num_mb_rows * num_mb_cols), 1);
  if (!denoiser->denoise_state) {
    vp8_denoiser_free(denoiser);
    return 1;
  }
  memset(denoiser->denoise_state, 0, (num_mb_rows * num_mb_cols));
  vp8_denoiser_set_parameters(denoiser, mode);
  denoiser->nmse_source_diff = 0;
  denoiser->nmse_source_diff_count = 0;
  denoiser->qp_avg = 0;
  // QP threshold below which we can go up to aggressive mode.
  denoiser->qp_threshold_up = 80;
  // QP threshold above which we can go back down to normal mode.
  // For now keep this second threshold high, so not used currently.
  denoiser->qp_threshold_down = 128;
  // Bitrate thresholds and noise metric (nmse) thresholds for switching to
  // aggressive mode.
  // TODO(marpan): Adjust thresholds, including effect on resolution.
  denoiser->bitrate_threshold = 400000;  // (bits/sec).
  denoiser->threshold_aggressive_mode = 80;
  if (width * height > 1280 * 720) {
    denoiser->bitrate_threshold = 3000000;
    denoiser->threshold_aggressive_mode = 200;
  } else if (width * height > 960 * 540) {
    denoiser->bitrate_threshold = 1200000;
    denoiser->threshold_aggressive_mode = 120;
  } else if (width * height > 640 * 480) {
    denoiser->bitrate_threshold = 600000;
    denoiser->threshold_aggressive_mode = 100;
  }
  return 0;
}

void vp8_denoiser_free(VP8_DENOISER *denoiser) {
  int i;
  assert(denoiser);

  for (i = 0; i < MAX_REF_FRAMES; ++i) {
    vp8_yv12_de_alloc_frame_buffer(&denoiser->yv12_running_avg[i]);
  }
  vp8_yv12_de_alloc_frame_buffer(&denoiser->yv12_mc_running_avg);
  vp8_yv12_de_alloc_frame_buffer(&denoiser->yv12_last_source);
  vpx_free(denoiser->denoise_state);
}

void vp8_denoiser_denoise_mb(VP8_DENOISER *denoiser, MACROBLOCK *x,
                             unsigned int best_sse, unsigned int zero_mv_sse,
                             int recon_yoffset, int recon_uvoffset,
                             loop_filter_info_n *lfi_n, int mb_row, int mb_col,
                             int block_index, int consec_zero_last)

{
  int mv_row;
  int mv_col;
  unsigned int motion_threshold;
  unsigned int motion_magnitude2;
  unsigned int sse_thresh;
  int sse_diff_thresh = 0;
  // Spatial loop filter: only applied selectively based on
  // temporal filter state of block relative to top/left neighbors.
  int apply_spatial_loop_filter = 1;
  MV_REFERENCE_FRAME frame = x->best_reference_frame;
  MV_REFERENCE_FRAME zero_frame = x->best_zeromv_reference_frame;

  enum vp8_denoiser_decision decision = FILTER_BLOCK;
  enum vp8_denoiser_decision decision_u = COPY_BLOCK;
  enum vp8_denoiser_decision decision_v = COPY_BLOCK;

  if (zero_frame) {
    YV12_BUFFER_CONFIG *src = &denoiser->yv12_running_avg[frame];
    YV12_BUFFER_CONFIG *dst = &denoiser->yv12_mc_running_avg;
    YV12_BUFFER_CONFIG saved_pre, saved_dst;
    MB_MODE_INFO saved_mbmi;
    MACROBLOCKD *filter_xd = &x->e_mbd;
    MB_MODE_INFO *mbmi = &filter_xd->mode_info_context->mbmi;
    int sse_diff = 0;
    // Bias on zero motion vector sse.
    const int zero_bias = denoiser->denoise_pars.denoise_mv_bias;
    zero_mv_sse = (unsigned int)((int64_t)zero_mv_sse * zero_bias / 100);
    sse_diff = (int)zero_mv_sse - (int)best_sse;

    saved_mbmi = *mbmi;

    /* Use the best MV for the compensation. */
    mbmi->ref_frame = x->best_reference_frame;
    mbmi->mode = x->best_sse_inter_mode;
    mbmi->mv = x->best_sse_mv;
    mbmi->need_to_clamp_mvs = x->need_to_clamp_best_mvs;
    mv_col = x->best_sse_mv.as_mv.col;
    mv_row = x->best_sse_mv.as_mv.row;
    // Bias to zero_mv if small amount of motion.
    // Note sse_diff_thresh is intialized to zero, so this ensures
    // we will always choose zero_mv for denoising if
    // zero_mv_see <= best_sse (i.e., sse_diff <= 0).
    if ((unsigned int)(mv_row * mv_row + mv_col * mv_col) <=
        NOISE_MOTION_THRESHOLD) {
      sse_diff_thresh = (int)SSE_DIFF_THRESHOLD;
    }

    if (frame == INTRA_FRAME || sse_diff <= sse_diff_thresh) {
      /*
       * Handle intra blocks as referring to last frame with zero motion
       * and let the absolute pixel difference affect the filter factor.
       * Also consider small amount of motion as being random walk due
       * to noise, if it doesn't mean that we get a much bigger error.
       * Note that any changes to the mode info only affects the
       * denoising.
       */
      x->denoise_zeromv = 1;
      mbmi->ref_frame = x->best_zeromv_reference_frame;

      src = &denoiser->yv12_running_avg[zero_frame];

      mbmi->mode = ZEROMV;
      mbmi->mv.as_int = 0;
      x->best_sse_inter_mode = ZEROMV;
      x->best_sse_mv.as_int = 0;
      best_sse = zero_mv_sse;
    }

    mv_row = x->best_sse_mv.as_mv.row;
    mv_col = x->best_sse_mv.as_mv.col;
    motion_magnitude2 = mv_row * mv_row + mv_col * mv_col;
    motion_threshold =
        denoiser->denoise_pars.scale_motion_thresh * NOISE_MOTION_THRESHOLD;

    if (motion_magnitude2 <
        denoiser->denoise_pars.scale_increase_filter * NOISE_MOTION_THRESHOLD) {
      x->increase_denoising = 1;
    }

    sse_thresh = denoiser->denoise_pars.scale_sse_thresh * SSE_THRESHOLD;
    if (x->increase_denoising) {
      sse_thresh = denoiser->denoise_pars.scale_sse_thresh * SSE_THRESHOLD_HIGH;
    }

    if (best_sse > sse_thresh || motion_magnitude2 > motion_threshold) {
      decision = COPY_BLOCK;
    }

    // If block is considered skin, don't denoise if the block
    // (1) is selected as non-zero motion for current frame, or
    // (2) has not been selected as ZERO_LAST mode at least x past frames
    // in a row.
    // TODO(marpan): Parameter "x" should be varied with framerate.
    // In particualar, should be reduced for layers (base layer/LAST).
    if (x->is_skin && (consec_zero_last < 2 || motion_magnitude2 > 0)) {
      decision = COPY_BLOCK;
    }

    if (decision == FILTER_BLOCK) {
      saved_pre = filter_xd->pre;
      saved_dst = filter_xd->dst;

      /* Compensate the running average. */
      filter_xd->pre.y_buffer = src->y_buffer + recon_yoffset;
      filter_xd->pre.u_buffer = src->u_buffer + recon_uvoffset;
      filter_xd->pre.v_buffer = src->v_buffer + recon_uvoffset;
      /* Write the compensated running average to the destination buffer. */
      filter_xd->dst.y_buffer = dst->y_buffer + recon_yoffset;
      filter_xd->dst.u_buffer = dst->u_buffer + recon_uvoffset;
      filter_xd->dst.v_buffer = dst->v_buffer + recon_uvoffset;

      if (!x->skip) {
        vp8_build_inter_predictors_mb(filter_xd);
      } else {
        vp8_build_inter16x16_predictors_mb(
            filter_xd, filter_xd->dst.y_buffer, filter_xd->dst.u_buffer,
            filter_xd->dst.v_buffer, filter_xd->dst.y_stride,
            filter_xd->dst.uv_stride);
      }
      filter_xd->pre = saved_pre;
      filter_xd->dst = saved_dst;
      *mbmi = saved_mbmi;
    }
  } else {
    // zero_frame should always be 1 for real-time mode, as the
    // ZEROMV mode is always checked, so we should never go into this branch.
    // If case ZEROMV is not checked, then we will force no denoise (COPY).
    decision = COPY_BLOCK;
  }

  if (decision == FILTER_BLOCK) {
    unsigned char *mc_running_avg_y =
        denoiser->yv12_mc_running_avg.y_buffer + recon_yoffset;
    int mc_avg_y_stride = denoiser->yv12_mc_running_avg.y_stride;
    unsigned char *running_avg_y =
        denoiser->yv12_running_avg[INTRA_FRAME].y_buffer + recon_yoffset;
    int avg_y_stride = denoiser->yv12_running_avg[INTRA_FRAME].y_stride;

    /* Filter. */
    decision = vp8_denoiser_filter(mc_running_avg_y, mc_avg_y_stride,
                                   running_avg_y, avg_y_stride, x->thismb, 16,
                                   motion_magnitude2, x->increase_denoising);
    denoiser->denoise_state[block_index] =
        motion_magnitude2 > 0 ? kFilterNonZeroMV : kFilterZeroMV;
    // Only denoise UV for zero motion, and if y channel was denoised.
    if (denoiser->denoiser_mode != kDenoiserOnYOnly && motion_magnitude2 == 0 &&
        decision == FILTER_BLOCK) {
      unsigned char *mc_running_avg_u =
          denoiser->yv12_mc_running_avg.u_buffer + recon_uvoffset;
      unsigned char *running_avg_u =
          denoiser->yv12_running_avg[INTRA_FRAME].u_buffer + recon_uvoffset;
      unsigned char *mc_running_avg_v =
          denoiser->yv12_mc_running_avg.v_buffer + recon_uvoffset;
      unsigned char *running_avg_v =
          denoiser->yv12_running_avg[INTRA_FRAME].v_buffer + recon_uvoffset;
      int mc_avg_uv_stride = denoiser->yv12_mc_running_avg.uv_stride;
      int avg_uv_stride = denoiser->yv12_running_avg[INTRA_FRAME].uv_stride;
      int signal_stride = x->block[16].src_stride;
      decision_u = vp8_denoiser_filter_uv(
          mc_running_avg_u, mc_avg_uv_stride, running_avg_u, avg_uv_stride,
          x->block[16].src + *x->block[16].base_src, signal_stride,
          motion_magnitude2, 0);
      decision_v = vp8_denoiser_filter_uv(
          mc_running_avg_v, mc_avg_uv_stride, running_avg_v, avg_uv_stride,
          x->block[20].src + *x->block[20].base_src, signal_stride,
          motion_magnitude2, 0);
    }
  }
  if (decision == COPY_BLOCK) {
    /* No filtering of this block; it differs too much from the predictor,
     * or the motion vector magnitude is considered too big.
     */
    x->denoise_zeromv = 0;
    vp8_copy_mem16x16(
        x->thismb, 16,
        denoiser->yv12_running_avg[INTRA_FRAME].y_buffer + recon_yoffset,
        denoiser->yv12_running_avg[INTRA_FRAME].y_stride);
    denoiser->denoise_state[block_index] = kNoFilter;
  }
  if (denoiser->denoiser_mode != kDenoiserOnYOnly) {
    if (decision_u == COPY_BLOCK) {
      vp8_copy_mem8x8(
          x->block[16].src + *x->block[16].base_src, x->block[16].src_stride,
          denoiser->yv12_running_avg[INTRA_FRAME].u_buffer + recon_uvoffset,
          denoiser->yv12_running_avg[INTRA_FRAME].uv_stride);
    }
    if (decision_v == COPY_BLOCK) {
      vp8_copy_mem8x8(
          x->block[20].src + *x->block[20].base_src, x->block[16].src_stride,
          denoiser->yv12_running_avg[INTRA_FRAME].v_buffer + recon_uvoffset,
          denoiser->yv12_running_avg[INTRA_FRAME].uv_stride);
    }
  }
  // Option to selectively deblock the denoised signal, for y channel only.
  if (apply_spatial_loop_filter) {
    loop_filter_info lfi;
    int apply_filter_col = 0;
    int apply_filter_row = 0;
    int apply_filter = 0;
    int y_stride = denoiser->yv12_running_avg[INTRA_FRAME].y_stride;
    int uv_stride = denoiser->yv12_running_avg[INTRA_FRAME].uv_stride;

    // Fix filter level to some nominal value for now.
    int filter_level = 48;

    int hev_index = lfi_n->hev_thr_lut[INTER_FRAME][filter_level];
    lfi.mblim = lfi_n->mblim[filter_level];
    lfi.blim = lfi_n->blim[filter_level];
    lfi.lim = lfi_n->lim[filter_level];
    lfi.hev_thr = lfi_n->hev_thr[hev_index];

    // Apply filter if there is a difference in the denoiser filter state
    // between the current and left/top block, or if non-zero motion vector
    // is used for the motion-compensated filtering.
    if (mb_col > 0) {
      apply_filter_col =
          !((denoiser->denoise_state[block_index] ==
             denoiser->denoise_state[block_index - 1]) &&
            denoiser->denoise_state[block_index] != kFilterNonZeroMV);
      if (apply_filter_col) {
        // Filter left vertical edge.
        apply_filter = 1;
        vp8_loop_filter_mbv(
            denoiser->yv12_running_avg[INTRA_FRAME].y_buffer + recon_yoffset,
            NULL, NULL, y_stride, uv_stride, &lfi);
      }
    }
    if (mb_row > 0) {
      apply_filter_row =
          !((denoiser->denoise_state[block_index] ==
             denoiser->denoise_state[block_index - denoiser->num_mb_cols]) &&
            denoiser->denoise_state[block_index] != kFilterNonZeroMV);
      if (apply_filter_row) {
        // Filter top horizontal edge.
        apply_filter = 1;
        vp8_loop_filter_mbh(
            denoiser->yv12_running_avg[INTRA_FRAME].y_buffer + recon_yoffset,
            NULL, NULL, y_stride, uv_stride, &lfi);
      }
    }
    if (apply_filter) {
      // Update the signal block |x|. Pixel changes are only to top and/or
      // left boundary pixels: can we avoid full block copy here.
      vp8_copy_mem16x16(
          denoiser->yv12_running_avg[INTRA_FRAME].y_buffer + recon_yoffset,
          y_stride, x->thismb, 16);
    }
  }
}
