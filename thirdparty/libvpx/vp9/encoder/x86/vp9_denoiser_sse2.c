/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>

#include "./vpx_config.h"
#include "./vp9_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/encoder/vp9_context_tree.h"
#include "vp9/encoder/vp9_denoiser.h"
#include "vpx_mem/vpx_mem.h"

// Compute the sum of all pixel differences of this MB.
static INLINE int sum_diff_16x1(__m128i acc_diff) {
  const __m128i k_1 = _mm_set1_epi16(1);
  const __m128i acc_diff_lo =
      _mm_srai_epi16(_mm_unpacklo_epi8(acc_diff, acc_diff), 8);
  const __m128i acc_diff_hi =
      _mm_srai_epi16(_mm_unpackhi_epi8(acc_diff, acc_diff), 8);
  const __m128i acc_diff_16 = _mm_add_epi16(acc_diff_lo, acc_diff_hi);
  const __m128i hg_fe_dc_ba = _mm_madd_epi16(acc_diff_16, k_1);
  const __m128i hgfe_dcba =
      _mm_add_epi32(hg_fe_dc_ba, _mm_srli_si128(hg_fe_dc_ba, 8));
  const __m128i hgfedcba =
      _mm_add_epi32(hgfe_dcba, _mm_srli_si128(hgfe_dcba, 4));
  return _mm_cvtsi128_si32(hgfedcba);
}

// Denoise a 16x1 vector.
static INLINE __m128i vp9_denoiser_16x1_sse2(
    const uint8_t *sig, const uint8_t *mc_running_avg_y, uint8_t *running_avg_y,
    const __m128i *k_0, const __m128i *k_4, const __m128i *k_8,
    const __m128i *k_16, const __m128i *l3, const __m128i *l32,
    const __m128i *l21, __m128i acc_diff) {
  // Calculate differences
  const __m128i v_sig = _mm_loadu_si128((const __m128i *)(&sig[0]));
  const __m128i v_mc_running_avg_y =
      _mm_loadu_si128((const __m128i *)(&mc_running_avg_y[0]));
  __m128i v_running_avg_y;
  const __m128i pdiff = _mm_subs_epu8(v_mc_running_avg_y, v_sig);
  const __m128i ndiff = _mm_subs_epu8(v_sig, v_mc_running_avg_y);
  // Obtain the sign. FF if diff is negative.
  const __m128i diff_sign = _mm_cmpeq_epi8(pdiff, *k_0);
  // Clamp absolute difference to 16 to be used to get mask. Doing this
  // allows us to use _mm_cmpgt_epi8, which operates on signed byte.
  const __m128i clamped_absdiff =
      _mm_min_epu8(_mm_or_si128(pdiff, ndiff), *k_16);
  // Get masks for l2 l1 and l0 adjustments.
  const __m128i mask2 = _mm_cmpgt_epi8(*k_16, clamped_absdiff);
  const __m128i mask1 = _mm_cmpgt_epi8(*k_8, clamped_absdiff);
  const __m128i mask0 = _mm_cmpgt_epi8(*k_4, clamped_absdiff);
  // Get adjustments for l2, l1, and l0.
  __m128i adj2 = _mm_and_si128(mask2, *l32);
  const __m128i adj1 = _mm_and_si128(mask1, *l21);
  const __m128i adj0 = _mm_and_si128(mask0, clamped_absdiff);
  __m128i adj, padj, nadj;

  // Combine the adjustments and get absolute adjustments.
  adj2 = _mm_add_epi8(adj2, adj1);
  adj = _mm_sub_epi8(*l3, adj2);
  adj = _mm_andnot_si128(mask0, adj);
  adj = _mm_or_si128(adj, adj0);

  // Restore the sign and get positive and negative adjustments.
  padj = _mm_andnot_si128(diff_sign, adj);
  nadj = _mm_and_si128(diff_sign, adj);

  // Calculate filtered value.
  v_running_avg_y = _mm_adds_epu8(v_sig, padj);
  v_running_avg_y = _mm_subs_epu8(v_running_avg_y, nadj);
  _mm_storeu_si128((__m128i *)running_avg_y, v_running_avg_y);

  // Adjustments <=7, and each element in acc_diff can fit in signed
  // char.
  acc_diff = _mm_adds_epi8(acc_diff, padj);
  acc_diff = _mm_subs_epi8(acc_diff, nadj);
  return acc_diff;
}

// Denoise a 16x1 vector with a weaker filter.
static INLINE __m128i vp9_denoiser_adj_16x1_sse2(
    const uint8_t *sig, const uint8_t *mc_running_avg_y, uint8_t *running_avg_y,
    const __m128i k_0, const __m128i k_delta, __m128i acc_diff) {
  __m128i v_running_avg_y = _mm_loadu_si128((__m128i *)(&running_avg_y[0]));
  // Calculate differences.
  const __m128i v_sig = _mm_loadu_si128((const __m128i *)(&sig[0]));
  const __m128i v_mc_running_avg_y =
      _mm_loadu_si128((const __m128i *)(&mc_running_avg_y[0]));
  const __m128i pdiff = _mm_subs_epu8(v_mc_running_avg_y, v_sig);
  const __m128i ndiff = _mm_subs_epu8(v_sig, v_mc_running_avg_y);
  // Obtain the sign. FF if diff is negative.
  const __m128i diff_sign = _mm_cmpeq_epi8(pdiff, k_0);
  // Clamp absolute difference to delta to get the adjustment.
  const __m128i adj = _mm_min_epu8(_mm_or_si128(pdiff, ndiff), k_delta);
  // Restore the sign and get positive and negative adjustments.
  __m128i padj, nadj;
  padj = _mm_andnot_si128(diff_sign, adj);
  nadj = _mm_and_si128(diff_sign, adj);
  // Calculate filtered value.
  v_running_avg_y = _mm_subs_epu8(v_running_avg_y, padj);
  v_running_avg_y = _mm_adds_epu8(v_running_avg_y, nadj);
  _mm_storeu_si128((__m128i *)running_avg_y, v_running_avg_y);

  // Accumulate the adjustments.
  acc_diff = _mm_subs_epi8(acc_diff, padj);
  acc_diff = _mm_adds_epi8(acc_diff, nadj);
  return acc_diff;
}

// Denoise 8x8 and 8x16 blocks.
static int vp9_denoiser_NxM_sse2_small(const uint8_t *sig, int sig_stride,
                                       const uint8_t *mc_running_avg_y,
                                       int mc_avg_y_stride,
                                       uint8_t *running_avg_y, int avg_y_stride,
                                       int increase_denoising, BLOCK_SIZE bs,
                                       int motion_magnitude, int width) {
  int sum_diff_thresh, r, sum_diff = 0;
  const int shift_inc =
      (increase_denoising && motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD)
          ? 1
          : 0;
  uint8_t sig_buffer[8][16], mc_running_buffer[8][16], running_buffer[8][16];
  __m128i acc_diff = _mm_setzero_si128();
  const __m128i k_0 = _mm_setzero_si128();
  const __m128i k_4 = _mm_set1_epi8(4 + shift_inc);
  const __m128i k_8 = _mm_set1_epi8(8);
  const __m128i k_16 = _mm_set1_epi8(16);
  // Modify each level's adjustment according to motion_magnitude.
  const __m128i l3 = _mm_set1_epi8(
      (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) ? 7 + shift_inc : 6);
  // Difference between level 3 and level 2 is 2.
  const __m128i l32 = _mm_set1_epi8(2);
  // Difference between level 2 and level 1 is 1.
  const __m128i l21 = _mm_set1_epi8(1);
  const int b_height = (4 << b_height_log2_lookup[bs]) >> 1;

  for (r = 0; r < b_height; ++r) {
    memcpy(sig_buffer[r], sig, width);
    memcpy(sig_buffer[r] + width, sig + sig_stride, width);
    memcpy(mc_running_buffer[r], mc_running_avg_y, width);
    memcpy(mc_running_buffer[r] + width, mc_running_avg_y + mc_avg_y_stride,
           width);
    memcpy(running_buffer[r], running_avg_y, width);
    memcpy(running_buffer[r] + width, running_avg_y + avg_y_stride, width);
    acc_diff = vp9_denoiser_16x1_sse2(sig_buffer[r], mc_running_buffer[r],
                                      running_buffer[r], &k_0, &k_4, &k_8,
                                      &k_16, &l3, &l32, &l21, acc_diff);
    memcpy(running_avg_y, running_buffer[r], width);
    memcpy(running_avg_y + avg_y_stride, running_buffer[r] + width, width);
    // Update pointers for next iteration.
    sig += (sig_stride << 1);
    mc_running_avg_y += (mc_avg_y_stride << 1);
    running_avg_y += (avg_y_stride << 1);
  }

  {
    sum_diff = sum_diff_16x1(acc_diff);
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
        const __m128i k_delta = _mm_set1_epi8(delta);
        running_avg_y -= avg_y_stride * (b_height << 1);
        for (r = 0; r < b_height; ++r) {
          acc_diff = vp9_denoiser_adj_16x1_sse2(
              sig_buffer[r], mc_running_buffer[r], running_buffer[r], k_0,
              k_delta, acc_diff);
          memcpy(running_avg_y, running_buffer[r], width);
          memcpy(running_avg_y + avg_y_stride, running_buffer[r] + width,
                 width);
          // Update pointers for next iteration.
          running_avg_y += (avg_y_stride << 1);
        }
        sum_diff = sum_diff_16x1(acc_diff);
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
static int vp9_denoiser_NxM_sse2_big(const uint8_t *sig, int sig_stride,
                                     const uint8_t *mc_running_avg_y,
                                     int mc_avg_y_stride,
                                     uint8_t *running_avg_y, int avg_y_stride,
                                     int increase_denoising, BLOCK_SIZE bs,
                                     int motion_magnitude) {
  int sum_diff_thresh, r, c, sum_diff = 0;
  const int shift_inc =
      (increase_denoising && motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD)
          ? 1
          : 0;
  __m128i acc_diff[4][4];
  const __m128i k_0 = _mm_setzero_si128();
  const __m128i k_4 = _mm_set1_epi8(4 + shift_inc);
  const __m128i k_8 = _mm_set1_epi8(8);
  const __m128i k_16 = _mm_set1_epi8(16);
  // Modify each level's adjustment according to motion_magnitude.
  const __m128i l3 = _mm_set1_epi8(
      (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) ? 7 + shift_inc : 6);
  // Difference between level 3 and level 2 is 2.
  const __m128i l32 = _mm_set1_epi8(2);
  // Difference between level 2 and level 1 is 1.
  const __m128i l21 = _mm_set1_epi8(1);
  const int b_width = (4 << b_width_log2_lookup[bs]);
  const int b_height = (4 << b_height_log2_lookup[bs]);
  const int b_width_shift4 = b_width >> 4;

  for (r = 0; r < 4; ++r) {
    for (c = 0; c < b_width_shift4; ++c) {
      acc_diff[c][r] = _mm_setzero_si128();
    }
  }

  for (r = 0; r < b_height; ++r) {
    for (c = 0; c < b_width_shift4; ++c) {
      acc_diff[c][r >> 4] = vp9_denoiser_16x1_sse2(
          sig, mc_running_avg_y, running_avg_y, &k_0, &k_4, &k_8, &k_16, &l3,
          &l32, &l21, acc_diff[c][r >> 4]);
      // Update pointers for next iteration.
      sig += 16;
      mc_running_avg_y += 16;
      running_avg_y += 16;
    }

    if ((r & 0xf) == 0xf || (bs == BLOCK_16X8 && r == 7)) {
      for (c = 0; c < b_width_shift4; ++c) {
        sum_diff += sum_diff_16x1(acc_diff[c][r >> 4]);
      }
    }

    // Update pointers for next iteration.
    sig = sig - b_width + sig_stride;
    mc_running_avg_y = mc_running_avg_y - b_width + mc_avg_y_stride;
    running_avg_y = running_avg_y - b_width + avg_y_stride;
  }

  {
    sum_diff_thresh = total_adj_strong_thresh(bs, increase_denoising);
    if (abs(sum_diff) > sum_diff_thresh) {
      const int delta =
          ((abs(sum_diff) - sum_diff_thresh) >> num_pels_log2_lookup[bs]) + 1;

      // Only apply the adjustment for max delta up to 3.
      if (delta < 4) {
        const __m128i k_delta = _mm_set1_epi8(delta);
        sig -= sig_stride * b_height;
        mc_running_avg_y -= mc_avg_y_stride * b_height;
        running_avg_y -= avg_y_stride * b_height;
        sum_diff = 0;
        for (r = 0; r < b_height; ++r) {
          for (c = 0; c < b_width_shift4; ++c) {
            acc_diff[c][r >> 4] =
                vp9_denoiser_adj_16x1_sse2(sig, mc_running_avg_y, running_avg_y,
                                           k_0, k_delta, acc_diff[c][r >> 4]);
            // Update pointers for next iteration.
            sig += 16;
            mc_running_avg_y += 16;
            running_avg_y += 16;
          }

          if ((r & 0xf) == 0xf || (bs == BLOCK_16X8 && r == 7)) {
            for (c = 0; c < b_width_shift4; ++c) {
              sum_diff += sum_diff_16x1(acc_diff[c][r >> 4]);
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

int vp9_denoiser_filter_sse2(const uint8_t *sig, int sig_stride,
                             const uint8_t *mc_avg, int mc_avg_stride,
                             uint8_t *avg, int avg_stride,
                             int increase_denoising, BLOCK_SIZE bs,
                             int motion_magnitude) {
  // Rank by frequency of the block type to have an early termination.
  if (bs == BLOCK_16X16 || bs == BLOCK_32X32 || bs == BLOCK_64X64 ||
      bs == BLOCK_16X32 || bs == BLOCK_16X8 || bs == BLOCK_32X16 ||
      bs == BLOCK_32X64 || bs == BLOCK_64X32) {
    return vp9_denoiser_NxM_sse2_big(sig, sig_stride, mc_avg, mc_avg_stride,
                                     avg, avg_stride, increase_denoising, bs,
                                     motion_magnitude);
  } else if (bs == BLOCK_8X8 || bs == BLOCK_8X16) {
    return vp9_denoiser_NxM_sse2_small(sig, sig_stride, mc_avg, mc_avg_stride,
                                       avg, avg_stride, increase_denoising, bs,
                                       motion_magnitude, 8);
  } else {
    return COPY_BLOCK;
  }
}
