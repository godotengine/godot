/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_config.h"
#include "./vp9_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/encoder/vp9_context_tree.h"
#include "vp9/encoder/vp9_denoiser.h"
#include "vpx_mem/vpx_mem.h"

// Compute the sum of all pixel differences of this MB.
static INLINE int horizontal_add_s8x16(const int8x16_t v_sum_diff_total) {
#if VPX_ARCH_AARCH64
  return vaddlvq_s8(v_sum_diff_total);
#else
  const int16x8_t fe_dc_ba_98_76_54_32_10 = vpaddlq_s8(v_sum_diff_total);
  const int32x4_t fedc_ba98_7654_3210 = vpaddlq_s16(fe_dc_ba_98_76_54_32_10);
  const int64x2_t fedcba98_76543210 = vpaddlq_s32(fedc_ba98_7654_3210);
  const int64x1_t x = vqadd_s64(vget_high_s64(fedcba98_76543210),
                                vget_low_s64(fedcba98_76543210));
  const int sum_diff = vget_lane_s32(vreinterpret_s32_s64(x), 0);
  return sum_diff;
#endif
}

// Denoise a 16x1 vector.
static INLINE int8x16_t denoiser_16x1_neon(
    const uint8_t *sig, const uint8_t *mc_running_avg_y, uint8_t *running_avg_y,
    const uint8x16_t v_level1_threshold, const uint8x16_t v_level2_threshold,
    const uint8x16_t v_level3_threshold, const uint8x16_t v_level1_adjustment,
    const uint8x16_t v_delta_level_1_and_2,
    const uint8x16_t v_delta_level_2_and_3, int8x16_t v_sum_diff_total) {
  const uint8x16_t v_sig = vld1q_u8(sig);
  const uint8x16_t v_mc_running_avg_y = vld1q_u8(mc_running_avg_y);

  /* Calculate absolute difference and sign masks. */
  const uint8x16_t v_abs_diff = vabdq_u8(v_sig, v_mc_running_avg_y);
  const uint8x16_t v_diff_pos_mask = vcltq_u8(v_sig, v_mc_running_avg_y);
  const uint8x16_t v_diff_neg_mask = vcgtq_u8(v_sig, v_mc_running_avg_y);

  /* Figure out which level that put us in. */
  const uint8x16_t v_level1_mask = vcleq_u8(v_level1_threshold, v_abs_diff);
  const uint8x16_t v_level2_mask = vcleq_u8(v_level2_threshold, v_abs_diff);
  const uint8x16_t v_level3_mask = vcleq_u8(v_level3_threshold, v_abs_diff);

  /* Calculate absolute adjustments for level 1, 2 and 3. */
  const uint8x16_t v_level2_adjustment =
      vandq_u8(v_level2_mask, v_delta_level_1_and_2);
  const uint8x16_t v_level3_adjustment =
      vandq_u8(v_level3_mask, v_delta_level_2_and_3);
  const uint8x16_t v_level1and2_adjustment =
      vaddq_u8(v_level1_adjustment, v_level2_adjustment);
  const uint8x16_t v_level1and2and3_adjustment =
      vaddq_u8(v_level1and2_adjustment, v_level3_adjustment);

  /* Figure adjustment absolute value by selecting between the absolute
   * difference if in level0 or the value for level 1, 2 and 3.
   */
  const uint8x16_t v_abs_adjustment =
      vbslq_u8(v_level1_mask, v_level1and2and3_adjustment, v_abs_diff);

  /* Calculate positive and negative adjustments. Apply them to the signal
   * and accumulate them. Adjustments are less than eight and the maximum
   * sum of them (7 * 16) can fit in a signed char.
   */
  const uint8x16_t v_pos_adjustment =
      vandq_u8(v_diff_pos_mask, v_abs_adjustment);
  const uint8x16_t v_neg_adjustment =
      vandq_u8(v_diff_neg_mask, v_abs_adjustment);

  uint8x16_t v_running_avg_y = vqaddq_u8(v_sig, v_pos_adjustment);
  v_running_avg_y = vqsubq_u8(v_running_avg_y, v_neg_adjustment);

  /* Store results. */
  vst1q_u8(running_avg_y, v_running_avg_y);

  /* Sum all the accumulators to have the sum of all pixel differences
   * for this macroblock.
   */
  {
    const int8x16_t v_sum_diff =
        vqsubq_s8(vreinterpretq_s8_u8(v_pos_adjustment),
                  vreinterpretq_s8_u8(v_neg_adjustment));
    v_sum_diff_total = vaddq_s8(v_sum_diff_total, v_sum_diff);
  }
  return v_sum_diff_total;
}

static INLINE int8x16_t denoiser_adjust_16x1_neon(
    const uint8_t *sig, const uint8_t *mc_running_avg_y, uint8_t *running_avg_y,
    const uint8x16_t k_delta, int8x16_t v_sum_diff_total) {
  uint8x16_t v_running_avg_y = vld1q_u8(running_avg_y);
  const uint8x16_t v_sig = vld1q_u8(sig);
  const uint8x16_t v_mc_running_avg_y = vld1q_u8(mc_running_avg_y);

  /* Calculate absolute difference and sign masks. */
  const uint8x16_t v_abs_diff = vabdq_u8(v_sig, v_mc_running_avg_y);
  const uint8x16_t v_diff_pos_mask = vcltq_u8(v_sig, v_mc_running_avg_y);
  const uint8x16_t v_diff_neg_mask = vcgtq_u8(v_sig, v_mc_running_avg_y);
  // Clamp absolute difference to delta to get the adjustment.
  const uint8x16_t v_abs_adjustment = vminq_u8(v_abs_diff, (k_delta));

  const uint8x16_t v_pos_adjustment =
      vandq_u8(v_diff_pos_mask, v_abs_adjustment);
  const uint8x16_t v_neg_adjustment =
      vandq_u8(v_diff_neg_mask, v_abs_adjustment);

  v_running_avg_y = vqsubq_u8(v_running_avg_y, v_pos_adjustment);
  v_running_avg_y = vqaddq_u8(v_running_avg_y, v_neg_adjustment);

  /* Store results. */
  vst1q_u8(running_avg_y, v_running_avg_y);

  {
    const int8x16_t v_sum_diff =
        vqsubq_s8(vreinterpretq_s8_u8(v_neg_adjustment),
                  vreinterpretq_s8_u8(v_pos_adjustment));
    v_sum_diff_total = vaddq_s8(v_sum_diff_total, v_sum_diff);
  }
  return v_sum_diff_total;
}

// Denoise 8x8 and 8x16 blocks.
static int vp9_denoiser_8xN_neon(const uint8_t *sig, int sig_stride,
                                 const uint8_t *mc_running_avg_y,
                                 int mc_avg_y_stride, uint8_t *running_avg_y,
                                 int avg_y_stride, int increase_denoising,
                                 BLOCK_SIZE bs, int motion_magnitude,
                                 int width) {
  int sum_diff_thresh, r, sum_diff = 0;
  const int shift_inc =
      (increase_denoising && motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD)
          ? 1
          : 0;
  uint8_t sig_buffer[8][16], mc_running_buffer[8][16], running_buffer[8][16];

  const uint8x16_t v_level1_adjustment = vmovq_n_u8(
      (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) ? 4 + shift_inc : 3);
  const uint8x16_t v_delta_level_1_and_2 = vdupq_n_u8(1);
  const uint8x16_t v_delta_level_2_and_3 = vdupq_n_u8(2);
  const uint8x16_t v_level1_threshold = vdupq_n_u8(4 + shift_inc);
  const uint8x16_t v_level2_threshold = vdupq_n_u8(8);
  const uint8x16_t v_level3_threshold = vdupq_n_u8(16);

  const int b_height = (4 << b_height_log2_lookup[bs]) >> 1;

  int8x16_t v_sum_diff_total = vdupq_n_s8(0);

  for (r = 0; r < b_height; ++r) {
    memcpy(sig_buffer[r], sig, width);
    memcpy(sig_buffer[r] + width, sig + sig_stride, width);
    memcpy(mc_running_buffer[r], mc_running_avg_y, width);
    memcpy(mc_running_buffer[r] + width, mc_running_avg_y + mc_avg_y_stride,
           width);
    memcpy(running_buffer[r], running_avg_y, width);
    memcpy(running_buffer[r] + width, running_avg_y + avg_y_stride, width);
    v_sum_diff_total = denoiser_16x1_neon(
        sig_buffer[r], mc_running_buffer[r], running_buffer[r],
        v_level1_threshold, v_level2_threshold, v_level3_threshold,
        v_level1_adjustment, v_delta_level_1_and_2, v_delta_level_2_and_3,
        v_sum_diff_total);
    {
      const uint8x16_t v_running_buffer = vld1q_u8(running_buffer[r]);
      const uint8x8_t v_running_buffer_high = vget_high_u8(v_running_buffer);
      const uint8x8_t v_running_buffer_low = vget_low_u8(v_running_buffer);
      vst1_u8(running_avg_y, v_running_buffer_low);
      vst1_u8(running_avg_y + avg_y_stride, v_running_buffer_high);
    }
    // Update pointers for next iteration.
    sig += (sig_stride << 1);
    mc_running_avg_y += (mc_avg_y_stride << 1);
    running_avg_y += (avg_y_stride << 1);
  }

  {
    sum_diff = horizontal_add_s8x16(v_sum_diff_total);
    sum_diff_thresh = total_adj_strong_thresh(bs, increase_denoising);
    if (abs(sum_diff) > sum_diff_thresh) {
      // Before returning to copy the block (i.e., apply no denoising),
      // check if we can still apply some (weaker) temporal filtering to
      // this block, that would otherwise not be denoised at all. Simplest
      // is to apply an additional adjustment to running_avg_y to bring it
      // closer to sig. The adjustment is capped by a maximum delta, and
      // chosen such that in most cases the resulting sum_diff will be
      // within the acceptable range given by sum_diff_thresh.

      // The delta is set by the excess of absolute pixel diff over the
      // threshold.
      const int delta =
          ((abs(sum_diff) - sum_diff_thresh) >> num_pels_log2_lookup[bs]) + 1;
      // Only apply the adjustment for max delta up to 3.
      if (delta < 4) {
        const uint8x16_t k_delta = vmovq_n_u8(delta);
        running_avg_y -= avg_y_stride * (b_height << 1);
        for (r = 0; r < b_height; ++r) {
          v_sum_diff_total = denoiser_adjust_16x1_neon(
              sig_buffer[r], mc_running_buffer[r], running_buffer[r], k_delta,
              v_sum_diff_total);
          {
            const uint8x16_t v_running_buffer = vld1q_u8(running_buffer[r]);
            const uint8x8_t v_running_buffer_high =
                vget_high_u8(v_running_buffer);
            const uint8x8_t v_running_buffer_low =
                vget_low_u8(v_running_buffer);
            vst1_u8(running_avg_y, v_running_buffer_low);
            vst1_u8(running_avg_y + avg_y_stride, v_running_buffer_high);
          }
          // Update pointers for next iteration.
          running_avg_y += (avg_y_stride << 1);
        }
        sum_diff = horizontal_add_s8x16(v_sum_diff_total);
        if (abs(sum_diff) > sum_diff_thresh) {
          return COPY_BLOCK;
        }
      } else {
        return COPY_BLOCK;
      }
    }
  }

  return FILTER_BLOCK;
}

// Denoise 16x16, 16x32, 32x16, 32x32, 32x64, 64x32 and 64x64 blocks.
static int vp9_denoiser_NxM_neon(const uint8_t *sig, int sig_stride,
                                 const uint8_t *mc_running_avg_y,
                                 int mc_avg_y_stride, uint8_t *running_avg_y,
                                 int avg_y_stride, int increase_denoising,
                                 BLOCK_SIZE bs, int motion_magnitude) {
  const int shift_inc =
      (increase_denoising && motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD)
          ? 1
          : 0;
  const uint8x16_t v_level1_adjustment = vmovq_n_u8(
      (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) ? 4 + shift_inc : 3);
  const uint8x16_t v_delta_level_1_and_2 = vdupq_n_u8(1);
  const uint8x16_t v_delta_level_2_and_3 = vdupq_n_u8(2);
  const uint8x16_t v_level1_threshold = vmovq_n_u8(4 + shift_inc);
  const uint8x16_t v_level2_threshold = vdupq_n_u8(8);
  const uint8x16_t v_level3_threshold = vdupq_n_u8(16);

  const int b_width = (4 << b_width_log2_lookup[bs]);
  const int b_height = (4 << b_height_log2_lookup[bs]);
  const int b_width_shift4 = b_width >> 4;

  int8x16_t v_sum_diff_total[4][4];
  int r, c, sum_diff = 0;

  for (r = 0; r < 4; ++r) {
    for (c = 0; c < b_width_shift4; ++c) {
      v_sum_diff_total[c][r] = vdupq_n_s8(0);
    }
  }

  for (r = 0; r < b_height; ++r) {
    for (c = 0; c < b_width_shift4; ++c) {
      v_sum_diff_total[c][r >> 4] = denoiser_16x1_neon(
          sig, mc_running_avg_y, running_avg_y, v_level1_threshold,
          v_level2_threshold, v_level3_threshold, v_level1_adjustment,
          v_delta_level_1_and_2, v_delta_level_2_and_3,
          v_sum_diff_total[c][r >> 4]);

      // Update pointers for next iteration.
      sig += 16;
      mc_running_avg_y += 16;
      running_avg_y += 16;
    }

    if ((r & 0xf) == 0xf || (bs == BLOCK_16X8 && r == 7)) {
      for (c = 0; c < b_width_shift4; ++c) {
        sum_diff += horizontal_add_s8x16(v_sum_diff_total[c][r >> 4]);
      }
    }

    // Update pointers for next iteration.
    sig = sig - b_width + sig_stride;
    mc_running_avg_y = mc_running_avg_y - b_width + mc_avg_y_stride;
    running_avg_y = running_avg_y - b_width + avg_y_stride;
  }

  {
    const int sum_diff_thresh = total_adj_strong_thresh(bs, increase_denoising);
    if (abs(sum_diff) > sum_diff_thresh) {
      const int delta =
          ((abs(sum_diff) - sum_diff_thresh) >> num_pels_log2_lookup[bs]) + 1;
      // Only apply the adjustment for max delta up to 3.
      if (delta < 4) {
        const uint8x16_t k_delta = vdupq_n_u8(delta);
        sig -= sig_stride * b_height;
        mc_running_avg_y -= mc_avg_y_stride * b_height;
        running_avg_y -= avg_y_stride * b_height;
        sum_diff = 0;

        for (r = 0; r < b_height; ++r) {
          for (c = 0; c < b_width_shift4; ++c) {
            v_sum_diff_total[c][r >> 4] =
                denoiser_adjust_16x1_neon(sig, mc_running_avg_y, running_avg_y,
                                          k_delta, v_sum_diff_total[c][r >> 4]);

            // Update pointers for next iteration.
            sig += 16;
            mc_running_avg_y += 16;
            running_avg_y += 16;
          }
          if ((r & 0xf) == 0xf || (bs == BLOCK_16X8 && r == 7)) {
            for (c = 0; c < b_width_shift4; ++c) {
              sum_diff += horizontal_add_s8x16(v_sum_diff_total[c][r >> 4]);
            }
          }

          sig = sig - b_width + sig_stride;
          mc_running_avg_y = mc_running_avg_y - b_width + mc_avg_y_stride;
          running_avg_y = running_avg_y - b_width + avg_y_stride;
        }

        if (abs(sum_diff) > sum_diff_thresh) {
          return COPY_BLOCK;
        }
      } else {
        return COPY_BLOCK;
      }
    }
  }
  return FILTER_BLOCK;
}

int vp9_denoiser_filter_neon(const uint8_t *sig, int sig_stride,
                             const uint8_t *mc_avg, int mc_avg_stride,
                             uint8_t *avg, int avg_stride,
                             int increase_denoising, BLOCK_SIZE bs,
                             int motion_magnitude) {
  // Rank by frequency of the block type to have an early termination.
  if (bs == BLOCK_16X16 || bs == BLOCK_32X32 || bs == BLOCK_64X64 ||
      bs == BLOCK_16X32 || bs == BLOCK_16X8 || bs == BLOCK_32X16 ||
      bs == BLOCK_32X64 || bs == BLOCK_64X32) {
    return vp9_denoiser_NxM_neon(sig, sig_stride, mc_avg, mc_avg_stride, avg,
                                 avg_stride, increase_denoising, bs,
                                 motion_magnitude);
  } else if (bs == BLOCK_8X8 || bs == BLOCK_8X16) {
    return vp9_denoiser_8xN_neon(sig, sig_stride, mc_avg, mc_avg_stride, avg,
                                 avg_stride, increase_denoising, bs,
                                 motion_magnitude, 8);
  }
  return COPY_BLOCK;
}
