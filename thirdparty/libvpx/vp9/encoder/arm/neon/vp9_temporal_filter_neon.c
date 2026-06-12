/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <arm_neon.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_temporal_filter.h"
#include "vp9/encoder/vp9_temporal_filter_constants.h"

// Read in 8 pixels from a and b as 8-bit unsigned integers, compute the
// difference squared, and store as unsigned 16-bit integer to dst.
static INLINE void store_dist_8(const uint8_t *a, const uint8_t *b,
                                uint16_t *dst) {
  const uint8x8_t a_reg = vld1_u8(a);
  const uint8x8_t b_reg = vld1_u8(b);

  uint16x8_t dist_first = vabdl_u8(a_reg, b_reg);
  dist_first = vmulq_u16(dist_first, dist_first);

  vst1q_u16(dst, dist_first);
}

static INLINE void store_dist_16(const uint8_t *a, const uint8_t *b,
                                 uint16_t *dst) {
  const uint8x16_t a_reg = vld1q_u8(a);
  const uint8x16_t b_reg = vld1q_u8(b);

  uint16x8_t dist_first = vabdl_u8(vget_low_u8(a_reg), vget_low_u8(b_reg));
  uint16x8_t dist_second = vabdl_u8(vget_high_u8(a_reg), vget_high_u8(b_reg));
  dist_first = vmulq_u16(dist_first, dist_first);
  dist_second = vmulq_u16(dist_second, dist_second);

  vst1q_u16(dst, dist_first);
  vst1q_u16(dst + 8, dist_second);
}

static INLINE void read_dist_8(const uint16_t *dist, uint16x8_t *dist_reg) {
  *dist_reg = vld1q_u16(dist);
}

static INLINE void read_dist_16(const uint16_t *dist, uint16x8_t *reg_first,
                                uint16x8_t *reg_second) {
  read_dist_8(dist, reg_first);
  read_dist_8(dist + 8, reg_second);
}

// Average the value based on the number of values summed (9 for pixels away
// from the border, 4 for pixels in corners, and 6 for other edge values).
//
// Add in the rounding factor and shift, clamp to 16, invert and shift. Multiply
// by weight.
static INLINE uint16x8_t average_8(uint16x8_t sum,
                                   const uint16x8_t *mul_constants,
                                   const int strength, const int rounding,
                                   const uint16x8_t *weight) {
  const uint32x4_t rounding_u32 = vdupq_n_u32(rounding << 16);
  const uint16x8_t weight_u16 = *weight;
  const uint16x8_t sixteen = vdupq_n_u16(16);
  const int32x4_t strength_u32 = vdupq_n_s32(-strength - 16);

  // modifier * 3 / index;
  uint32x4_t sum_hi =
      vmull_u16(vget_low_u16(sum), vget_low_u16(*mul_constants));
  uint32x4_t sum_lo =
      vmull_u16(vget_high_u16(sum), vget_high_u16(*mul_constants));

  sum_lo = vqaddq_u32(sum_lo, rounding_u32);
  sum_hi = vqaddq_u32(sum_hi, rounding_u32);

  // we cannot use vshrn_n_u32 as strength is not known at compile time.
  sum_lo = vshlq_u32(sum_lo, strength_u32);
  sum_hi = vshlq_u32(sum_hi, strength_u32);

  sum = vcombine_u16(vmovn_u32(sum_hi), vmovn_u32(sum_lo));

  // The maximum input to this comparison is UINT16_MAX * NEIGHBOR_CONSTANT_4
  // >> 16 (also NEIGHBOR_CONSTANT_4 -1) which is 49151 / 0xbfff / -16385
  // So this needs to use the epu16 version which did not come until SSE4.
  sum = vminq_u16(sum, sixteen);
  sum = vsubq_u16(sixteen, sum);
  return vmulq_u16(sum, weight_u16);
}

// Add 'sum_u16' to 'count'. Multiply by 'pred' and add to 'accumulator.'
static void accumulate_and_store_8(const uint16x8_t sum_u16,
                                   const uint8_t *pred, uint16_t *count,
                                   uint32_t *accumulator) {
  uint16x8_t pred_u16 = vmovl_u8(vld1_u8(pred));
  uint16x8_t count_u16 = vld1q_u16(count);
  uint32x4_t accum_0_u32, accum_1_u32;

  count_u16 = vqaddq_u16(count_u16, sum_u16);
  vst1q_u16(count, count_u16);

  accum_0_u32 = vld1q_u32(accumulator);
  accum_1_u32 = vld1q_u32(accumulator + 4);

  accum_0_u32 =
      vmlal_u16(accum_0_u32, vget_low_u16(sum_u16), vget_low_u16(pred_u16));
  accum_1_u32 =
      vmlal_u16(accum_1_u32, vget_high_u16(sum_u16), vget_high_u16(pred_u16));

  vst1q_u32(accumulator, accum_0_u32);
  vst1q_u32(accumulator + 4, accum_1_u32);
}

static INLINE void accumulate_and_store_16(const uint16x8_t sum_0_u16,
                                           const uint16x8_t sum_1_u16,
                                           const uint8_t *pred, uint16_t *count,
                                           uint32_t *accumulator) {
  uint8x16_t pred_u8 = vld1q_u8(pred);
  uint16x8_t pred_0_u16 = vmovl_u8(vget_low_u8(pred_u8));
  uint16x8_t pred_1_u16 = vmovl_u8(vget_high_u8(pred_u8));
  uint16x8_t count_0_u16 = vld1q_u16(count);
  uint16x8_t count_1_u16 = vld1q_u16(count + 8);
  uint32x4_t accum_0_u32, accum_1_u32, accum_2_u32, accum_3_u32;

  count_0_u16 = vqaddq_u16(count_0_u16, sum_0_u16);
  vst1q_u16(count, count_0_u16);
  count_1_u16 = vqaddq_u16(count_1_u16, sum_1_u16);
  vst1q_u16(count + 8, count_1_u16);

  accum_0_u32 = vld1q_u32(accumulator);
  accum_1_u32 = vld1q_u32(accumulator + 4);
  accum_2_u32 = vld1q_u32(accumulator + 8);
  accum_3_u32 = vld1q_u32(accumulator + 12);

  accum_0_u32 =
      vmlal_u16(accum_0_u32, vget_low_u16(sum_0_u16), vget_low_u16(pred_0_u16));
  accum_1_u32 = vmlal_u16(accum_1_u32, vget_high_u16(sum_0_u16),
                          vget_high_u16(pred_0_u16));
  accum_2_u32 =
      vmlal_u16(accum_2_u32, vget_low_u16(sum_1_u16), vget_low_u16(pred_1_u16));
  accum_3_u32 = vmlal_u16(accum_3_u32, vget_high_u16(sum_1_u16),
                          vget_high_u16(pred_1_u16));

  vst1q_u32(accumulator, accum_0_u32);
  vst1q_u32(accumulator + 4, accum_1_u32);
  vst1q_u32(accumulator + 8, accum_2_u32);
  vst1q_u32(accumulator + 12, accum_3_u32);
}

// Read in 8 pixels from y_dist. For each index i, compute y_dist[i-1] +
// y_dist[i] + y_dist[i+1] and store in sum as 16-bit unsigned int.
static INLINE void get_sum_8(const uint16_t *y_dist, uint16x8_t *sum) {
  uint16x8_t dist_reg, dist_left, dist_right;

  dist_reg = vld1q_u16(y_dist);
  dist_left = vld1q_u16(y_dist - 1);
  dist_right = vld1q_u16(y_dist + 1);

  *sum = vqaddq_u16(dist_reg, dist_left);
  *sum = vqaddq_u16(*sum, dist_right);
}

// Read in 16 pixels from y_dist. For each index i, compute y_dist[i-1] +
// y_dist[i] + y_dist[i+1]. Store the result for first 8 pixels in sum_first and
// the rest in sum_second.
static INLINE void get_sum_16(const uint16_t *y_dist, uint16x8_t *sum_first,
                              uint16x8_t *sum_second) {
  get_sum_8(y_dist, sum_first);
  get_sum_8(y_dist + 8, sum_second);
}

// Read in a row of chroma values corresponds to a row of 16 luma values.
static INLINE void read_chroma_dist_row_16(int ss_x, const uint16_t *u_dist,
                                           const uint16_t *v_dist,
                                           uint16x8_t *u_first,
                                           uint16x8_t *u_second,
                                           uint16x8_t *v_first,
                                           uint16x8_t *v_second) {
  if (!ss_x) {
    // If there is no chroma subsampling in the horizontal direction, then we
    // need to load 16 entries from chroma.
    read_dist_16(u_dist, u_first, u_second);
    read_dist_16(v_dist, v_first, v_second);
  } else {  // ss_x == 1
    // Otherwise, we only need to load 8 entries
    uint16x8_t u_reg, v_reg;
    uint16x8x2_t pair;

    read_dist_8(u_dist, &u_reg);

    pair = vzipq_u16(u_reg, u_reg);
    *u_first = pair.val[0];
    *u_second = pair.val[1];

    read_dist_8(v_dist, &v_reg);

    pair = vzipq_u16(v_reg, v_reg);
    *v_first = pair.val[0];
    *v_second = pair.val[1];
  }
}

// Add a row of luma distortion to 8 corresponding chroma mods.
static INLINE void add_luma_dist_to_8_chroma_mod(const uint16_t *y_dist,
                                                 int ss_x, int ss_y,
                                                 uint16x8_t *u_mod,
                                                 uint16x8_t *v_mod) {
  uint16x8_t y_reg;
  if (!ss_x) {
    read_dist_8(y_dist, &y_reg);
    if (ss_y == 1) {
      uint16x8_t y_tmp;
      read_dist_8(y_dist + DIST_STRIDE, &y_tmp);

      y_reg = vqaddq_u16(y_reg, y_tmp);
    }
  } else {
    uint16x8_t y_first, y_second;
    uint32x4_t y_first32, y_second32;

    read_dist_16(y_dist, &y_first, &y_second);
    if (ss_y == 1) {
      uint16x8_t y_tmp_0, y_tmp_1;
      read_dist_16(y_dist + DIST_STRIDE, &y_tmp_0, &y_tmp_1);

      y_first = vqaddq_u16(y_first, y_tmp_0);
      y_second = vqaddq_u16(y_second, y_tmp_1);
    }

    y_first32 = vpaddlq_u16(y_first);
    y_second32 = vpaddlq_u16(y_second);

    y_reg = vcombine_u16(vqmovn_u32(y_first32), vqmovn_u32(y_second32));
  }

  *u_mod = vqaddq_u16(*u_mod, y_reg);
  *v_mod = vqaddq_u16(*v_mod, y_reg);
}

// Apply temporal filter to the luma components. This performs temporal
// filtering on a luma block of 16 X block_height. Use blk_fw as an array of
// size 4 for the weights for each of the 4 subblocks if blk_fw is not NULL,
// else use top_weight for top half, and bottom weight for bottom half.
static void apply_temporal_filter_luma_16(
    const uint8_t *y_pre, int y_pre_stride, unsigned int block_width,
    unsigned int block_height, int ss_x, int ss_y, int strength,
    int use_whole_blk, uint32_t *y_accum, uint16_t *y_count,
    const uint16_t *y_dist, const uint16_t *u_dist, const uint16_t *v_dist,
    const int16_t *const *neighbors_first,
    const int16_t *const *neighbors_second, int top_weight, int bottom_weight,
    const int *blk_fw) {
  const int rounding = (1 << strength) >> 1;
  uint16x8_t weight_first, weight_second;

  uint16x8_t mul_first, mul_second;

  uint16x8_t sum_row_1_first, sum_row_1_second;
  uint16x8_t sum_row_2_first, sum_row_2_second;
  uint16x8_t sum_row_3_first, sum_row_3_second;

  uint16x8_t u_first, u_second;
  uint16x8_t v_first, v_second;

  uint16x8_t sum_row_first;
  uint16x8_t sum_row_second;

  // Loop variables
  unsigned int h;

  assert(strength >= 0);
  assert(strength <= 6);

  assert(block_width == 16);
  (void)block_width;

  // Initialize the weights
  if (blk_fw) {
    weight_first = vdupq_n_u16(blk_fw[0]);
    weight_second = vdupq_n_u16(blk_fw[1]);
  } else {
    weight_first = vdupq_n_u16(top_weight);
    weight_second = weight_first;
  }

  // First row
  mul_first = vld1q_u16((const uint16_t *)neighbors_first[0]);
  mul_second = vld1q_u16((const uint16_t *)neighbors_second[0]);

  // Add luma values
  get_sum_16(y_dist, &sum_row_2_first, &sum_row_2_second);
  get_sum_16(y_dist + DIST_STRIDE, &sum_row_3_first, &sum_row_3_second);

  sum_row_first = vqaddq_u16(sum_row_2_first, sum_row_3_first);
  sum_row_second = vqaddq_u16(sum_row_2_second, sum_row_3_second);

  // Add chroma values
  read_chroma_dist_row_16(ss_x, u_dist, v_dist, &u_first, &u_second, &v_first,
                          &v_second);

  sum_row_first = vqaddq_u16(sum_row_first, u_first);
  sum_row_second = vqaddq_u16(sum_row_second, u_second);

  sum_row_first = vqaddq_u16(sum_row_first, v_first);
  sum_row_second = vqaddq_u16(sum_row_second, v_second);

  // Get modifier and store result
  sum_row_first =
      average_8(sum_row_first, &mul_first, strength, rounding, &weight_first);

  sum_row_second = average_8(sum_row_second, &mul_second, strength, rounding,
                             &weight_second);

  accumulate_and_store_16(sum_row_first, sum_row_second, y_pre, y_count,
                          y_accum);

  y_pre += y_pre_stride;
  y_count += y_pre_stride;
  y_accum += y_pre_stride;
  y_dist += DIST_STRIDE;

  u_dist += DIST_STRIDE;
  v_dist += DIST_STRIDE;

  // Then all the rows except the last one
  mul_first = vld1q_u16((const uint16_t *)neighbors_first[1]);
  mul_second = vld1q_u16((const uint16_t *)neighbors_second[1]);

  for (h = 1; h < block_height - 1; ++h) {
    // Move the weight to bottom half
    if (!use_whole_blk && h == block_height / 2) {
      if (blk_fw) {
        weight_first = vdupq_n_u16(blk_fw[2]);
        weight_second = vdupq_n_u16(blk_fw[3]);
      } else {
        weight_first = vdupq_n_u16(bottom_weight);
        weight_second = weight_first;
      }
    }
    // Shift the rows up
    sum_row_1_first = sum_row_2_first;
    sum_row_1_second = sum_row_2_second;
    sum_row_2_first = sum_row_3_first;
    sum_row_2_second = sum_row_3_second;

    // Add luma values to the modifier
    sum_row_first = vqaddq_u16(sum_row_1_first, sum_row_2_first);
    sum_row_second = vqaddq_u16(sum_row_1_second, sum_row_2_second);

    get_sum_16(y_dist + DIST_STRIDE, &sum_row_3_first, &sum_row_3_second);

    sum_row_first = vqaddq_u16(sum_row_first, sum_row_3_first);
    sum_row_second = vqaddq_u16(sum_row_second, sum_row_3_second);

    // Add chroma values to the modifier
    if (ss_y == 0 || h % 2 == 0) {
      // Only calculate the new chroma distortion if we are at a pixel that
      // corresponds to a new chroma row
      read_chroma_dist_row_16(ss_x, u_dist, v_dist, &u_first, &u_second,
                              &v_first, &v_second);
      u_dist += DIST_STRIDE;
      v_dist += DIST_STRIDE;
    }

    sum_row_first = vqaddq_u16(sum_row_first, u_first);
    sum_row_second = vqaddq_u16(sum_row_second, u_second);
    sum_row_first = vqaddq_u16(sum_row_first, v_first);
    sum_row_second = vqaddq_u16(sum_row_second, v_second);

    // Get modifier and store result
    sum_row_first =
        average_8(sum_row_first, &mul_first, strength, rounding, &weight_first);
    sum_row_second = average_8(sum_row_second, &mul_second, strength, rounding,
                               &weight_second);
    accumulate_and_store_16(sum_row_first, sum_row_second, y_pre, y_count,
                            y_accum);
    y_pre += y_pre_stride;
    y_count += y_pre_stride;
    y_accum += y_pre_stride;
    y_dist += DIST_STRIDE;
  }

  // The last row
  mul_first = vld1q_u16((const uint16_t *)neighbors_first[0]);
  mul_second = vld1q_u16((const uint16_t *)neighbors_second[0]);

  // Shift the rows up
  sum_row_1_first = sum_row_2_first;
  sum_row_1_second = sum_row_2_second;
  sum_row_2_first = sum_row_3_first;
  sum_row_2_second = sum_row_3_second;

  // Add luma values to the modifier
  sum_row_first = vqaddq_u16(sum_row_1_first, sum_row_2_first);
  sum_row_second = vqaddq_u16(sum_row_1_second, sum_row_2_second);

  // Add chroma values to the modifier
  if (ss_y == 0) {
    // Only calculate the new chroma distortion if we are at a pixel that
    // corresponds to a new chroma row
    read_chroma_dist_row_16(ss_x, u_dist, v_dist, &u_first, &u_second, &v_first,
                            &v_second);
  }

  sum_row_first = vqaddq_u16(sum_row_first, u_first);
  sum_row_second = vqaddq_u16(sum_row_second, u_second);
  sum_row_first = vqaddq_u16(sum_row_first, v_first);
  sum_row_second = vqaddq_u16(sum_row_second, v_second);

  // Get modifier and store result
  sum_row_first =
      average_8(sum_row_first, &mul_first, strength, rounding, &weight_first);
  sum_row_second = average_8(sum_row_second, &mul_second, strength, rounding,
                             &weight_second);
  accumulate_and_store_16(sum_row_first, sum_row_second, y_pre, y_count,
                          y_accum);
}

// Perform temporal filter for the luma component.
static void apply_temporal_filter_luma(
    const uint8_t *y_pre, int y_pre_stride, unsigned int block_width,
    unsigned int block_height, int ss_x, int ss_y, int strength,
    const int *blk_fw, int use_whole_blk, uint32_t *y_accum, uint16_t *y_count,
    const uint16_t *y_dist, const uint16_t *u_dist, const uint16_t *v_dist) {
  unsigned int blk_col = 0, uv_blk_col = 0;
  const unsigned int blk_col_step = 16, uv_blk_col_step = 16 >> ss_x;
  const unsigned int mid_width = block_width >> 1,
                     last_width = block_width - blk_col_step;
  int top_weight = blk_fw[0],
      bottom_weight = use_whole_blk ? blk_fw[0] : blk_fw[2];
  const int16_t *const *neighbors_first;
  const int16_t *const *neighbors_second;

  if (block_width == 16) {
    // Special Case: The block width is 16 and we are operating on a row of 16
    // chroma pixels. In this case, we can't use the usual left-middle-right
    // pattern. We also don't support splitting now.
    neighbors_first = LUMA_LEFT_COLUMN_NEIGHBORS;
    neighbors_second = LUMA_RIGHT_COLUMN_NEIGHBORS;
    if (use_whole_blk) {
      apply_temporal_filter_luma_16(
          y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
          use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
          u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
          neighbors_second, top_weight, bottom_weight, NULL);
    } else {
      apply_temporal_filter_luma_16(
          y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
          use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
          u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
          neighbors_second, 0, 0, blk_fw);
    }

    return;
  }

  // Left
  neighbors_first = LUMA_LEFT_COLUMN_NEIGHBORS;
  neighbors_second = LUMA_MIDDLE_COLUMN_NEIGHBORS;
  apply_temporal_filter_luma_16(
      y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
      use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
      u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
      neighbors_second, top_weight, bottom_weight, NULL);

  blk_col += blk_col_step;
  uv_blk_col += uv_blk_col_step;

  // Middle First
  neighbors_first = LUMA_MIDDLE_COLUMN_NEIGHBORS;
  for (; blk_col < mid_width;
       blk_col += blk_col_step, uv_blk_col += uv_blk_col_step) {
    apply_temporal_filter_luma_16(
        y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
        use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
        u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
        neighbors_second, top_weight, bottom_weight, NULL);
  }

  if (!use_whole_blk) {
    top_weight = blk_fw[1];
    bottom_weight = blk_fw[3];
  }

  // Middle Second
  for (; blk_col < last_width;
       blk_col += blk_col_step, uv_blk_col += uv_blk_col_step) {
    apply_temporal_filter_luma_16(
        y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
        use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
        u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
        neighbors_second, top_weight, bottom_weight, NULL);
  }

  // Right
  neighbors_second = LUMA_RIGHT_COLUMN_NEIGHBORS;
  apply_temporal_filter_luma_16(
      y_pre + blk_col, y_pre_stride, 16, block_height, ss_x, ss_y, strength,
      use_whole_blk, y_accum + blk_col, y_count + blk_col, y_dist + blk_col,
      u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors_first,
      neighbors_second, top_weight, bottom_weight, NULL);
}

// Apply temporal filter to the chroma components. This performs temporal
// filtering on a chroma block of 8 X uv_height. If blk_fw is not NULL, use
// blk_fw as an array of size 4 for the weights for each of the 4 subblocks,
// else use top_weight for top half, and bottom weight for bottom half.
static void apply_temporal_filter_chroma_8(
    const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride,
    unsigned int uv_block_height, int ss_x, int ss_y, int strength,
    uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count,
    const uint16_t *y_dist, const uint16_t *u_dist, const uint16_t *v_dist,
    const int16_t *const *neighbors, int top_weight, int bottom_weight,
    const int *blk_fw) {
  const int rounding = (1 << strength) >> 1;

  uint16x8_t weight;

  uint16x8_t mul;

  uint16x8_t u_sum_row_1, u_sum_row_2, u_sum_row_3;
  uint16x8_t v_sum_row_1, v_sum_row_2, v_sum_row_3;

  uint16x8_t u_sum_row, v_sum_row;

  // Loop variable
  unsigned int h;

  // Initialize weight
  if (blk_fw) {
    weight = vcombine_u16(vdup_n_u16(blk_fw[0]), vdup_n_u16(blk_fw[1]));
  } else {
    weight = vdupq_n_u16(top_weight);
  }

  // First row
  mul = vld1q_u16((const uint16_t *)neighbors[0]);

  // Add chroma values
  get_sum_8(u_dist, &u_sum_row_2);
  get_sum_8(u_dist + DIST_STRIDE, &u_sum_row_3);

  u_sum_row = vqaddq_u16(u_sum_row_2, u_sum_row_3);

  get_sum_8(v_dist, &v_sum_row_2);
  get_sum_8(v_dist + DIST_STRIDE, &v_sum_row_3);

  v_sum_row = vqaddq_u16(v_sum_row_2, v_sum_row_3);

  // Add luma values
  add_luma_dist_to_8_chroma_mod(y_dist, ss_x, ss_y, &u_sum_row, &v_sum_row);

  // Get modifier and store result
  u_sum_row = average_8(u_sum_row, &mul, strength, rounding, &weight);
  v_sum_row = average_8(v_sum_row, &mul, strength, rounding, &weight);

  accumulate_and_store_8(u_sum_row, u_pre, u_count, u_accum);
  accumulate_and_store_8(v_sum_row, v_pre, v_count, v_accum);

  u_pre += uv_pre_stride;
  u_dist += DIST_STRIDE;
  v_pre += uv_pre_stride;
  v_dist += DIST_STRIDE;
  u_count += uv_pre_stride;
  u_accum += uv_pre_stride;
  v_count += uv_pre_stride;
  v_accum += uv_pre_stride;

  y_dist += DIST_STRIDE * (1 + ss_y);

  // Then all the rows except the last one
  mul = vld1q_u16((const uint16_t *)neighbors[1]);

  for (h = 1; h < uv_block_height - 1; ++h) {
    // Move the weight pointer to the bottom half of the blocks
    if (h == uv_block_height / 2) {
      if (blk_fw) {
        weight = vcombine_u16(vdup_n_u16(blk_fw[2]), vdup_n_u16(blk_fw[3]));
      } else {
        weight = vdupq_n_u16(bottom_weight);
      }
    }

    // Shift the rows up
    u_sum_row_1 = u_sum_row_2;
    u_sum_row_2 = u_sum_row_3;

    v_sum_row_1 = v_sum_row_2;
    v_sum_row_2 = v_sum_row_3;

    // Add chroma values
    u_sum_row = vqaddq_u16(u_sum_row_1, u_sum_row_2);
    get_sum_8(u_dist + DIST_STRIDE, &u_sum_row_3);
    u_sum_row = vqaddq_u16(u_sum_row, u_sum_row_3);

    v_sum_row = vqaddq_u16(v_sum_row_1, v_sum_row_2);
    get_sum_8(v_dist + DIST_STRIDE, &v_sum_row_3);
    v_sum_row = vqaddq_u16(v_sum_row, v_sum_row_3);

    // Add luma values
    add_luma_dist_to_8_chroma_mod(y_dist, ss_x, ss_y, &u_sum_row, &v_sum_row);

    // Get modifier and store result
    u_sum_row = average_8(u_sum_row, &mul, strength, rounding, &weight);
    v_sum_row = average_8(v_sum_row, &mul, strength, rounding, &weight);

    accumulate_and_store_8(u_sum_row, u_pre, u_count, u_accum);
    accumulate_and_store_8(v_sum_row, v_pre, v_count, v_accum);

    u_pre += uv_pre_stride;
    u_dist += DIST_STRIDE;
    v_pre += uv_pre_stride;
    v_dist += DIST_STRIDE;
    u_count += uv_pre_stride;
    u_accum += uv_pre_stride;
    v_count += uv_pre_stride;
    v_accum += uv_pre_stride;

    y_dist += DIST_STRIDE * (1 + ss_y);
  }

  // The last row
  mul = vld1q_u16((const uint16_t *)neighbors[0]);

  // Shift the rows up
  u_sum_row_1 = u_sum_row_2;
  u_sum_row_2 = u_sum_row_3;

  v_sum_row_1 = v_sum_row_2;
  v_sum_row_2 = v_sum_row_3;

  // Add chroma values
  u_sum_row = vqaddq_u16(u_sum_row_1, u_sum_row_2);
  v_sum_row = vqaddq_u16(v_sum_row_1, v_sum_row_2);

  // Add luma values
  add_luma_dist_to_8_chroma_mod(y_dist, ss_x, ss_y, &u_sum_row, &v_sum_row);

  // Get modifier and store result
  u_sum_row = average_8(u_sum_row, &mul, strength, rounding, &weight);
  v_sum_row = average_8(v_sum_row, &mul, strength, rounding, &weight);

  accumulate_and_store_8(u_sum_row, u_pre, u_count, u_accum);
  accumulate_and_store_8(v_sum_row, v_pre, v_count, v_accum);
}

// Perform temporal filter for the chroma components.
static void apply_temporal_filter_chroma(
    const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride,
    unsigned int block_width, unsigned int block_height, int ss_x, int ss_y,
    int strength, const int *blk_fw, int use_whole_blk, uint32_t *u_accum,
    uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count,
    const uint16_t *y_dist, const uint16_t *u_dist, const uint16_t *v_dist) {
  const unsigned int uv_width = block_width >> ss_x,
                     uv_height = block_height >> ss_y;

  unsigned int blk_col = 0, uv_blk_col = 0;
  const unsigned int uv_blk_col_step = 8, blk_col_step = 8 << ss_x;
  const unsigned int uv_mid_width = uv_width >> 1,
                     uv_last_width = uv_width - uv_blk_col_step;
  int top_weight = blk_fw[0],
      bottom_weight = use_whole_blk ? blk_fw[0] : blk_fw[2];
  const int16_t *const *neighbors;

  if (uv_width == 8) {
    // Special Case: We are subsampling in x direction on a 16x16 block. Since
    // we are operating on a row of 8 chroma pixels, we can't use the usual
    // left-middle-right pattern.
    assert(ss_x);

    if (ss_y) {
      neighbors = CHROMA_DOUBLE_SS_SINGLE_COLUMN_NEIGHBORS;
    } else {
      neighbors = CHROMA_SINGLE_SS_SINGLE_COLUMN_NEIGHBORS;
    }

    if (use_whole_blk) {
      apply_temporal_filter_chroma_8(
          u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height,
          ss_x, ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
          v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
          u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, top_weight,
          bottom_weight, NULL);
    } else {
      apply_temporal_filter_chroma_8(
          u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height,
          ss_x, ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
          v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
          u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, 0, 0, blk_fw);
    }

    return;
  }

  // Left
  if (ss_x && ss_y) {
    neighbors = CHROMA_DOUBLE_SS_LEFT_COLUMN_NEIGHBORS;
  } else if (ss_x || ss_y) {
    neighbors = CHROMA_SINGLE_SS_LEFT_COLUMN_NEIGHBORS;
  } else {
    neighbors = CHROMA_NO_SS_LEFT_COLUMN_NEIGHBORS;
  }

  apply_temporal_filter_chroma_8(
      u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height, ss_x,
      ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
      v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
      u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, top_weight,
      bottom_weight, NULL);

  blk_col += blk_col_step;
  uv_blk_col += uv_blk_col_step;

  // Middle First
  if (ss_x && ss_y) {
    neighbors = CHROMA_DOUBLE_SS_MIDDLE_COLUMN_NEIGHBORS;
  } else if (ss_x || ss_y) {
    neighbors = CHROMA_SINGLE_SS_MIDDLE_COLUMN_NEIGHBORS;
  } else {
    neighbors = CHROMA_NO_SS_MIDDLE_COLUMN_NEIGHBORS;
  }

  for (; uv_blk_col < uv_mid_width;
       blk_col += blk_col_step, uv_blk_col += uv_blk_col_step) {
    apply_temporal_filter_chroma_8(
        u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height, ss_x,
        ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
        v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
        u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, top_weight,
        bottom_weight, NULL);
  }

  if (!use_whole_blk) {
    top_weight = blk_fw[1];
    bottom_weight = blk_fw[3];
  }

  // Middle Second
  for (; uv_blk_col < uv_last_width;
       blk_col += blk_col_step, uv_blk_col += uv_blk_col_step) {
    apply_temporal_filter_chroma_8(
        u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height, ss_x,
        ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
        v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
        u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, top_weight,
        bottom_weight, NULL);
  }

  // Right
  if (ss_x && ss_y) {
    neighbors = CHROMA_DOUBLE_SS_RIGHT_COLUMN_NEIGHBORS;
  } else if (ss_x || ss_y) {
    neighbors = CHROMA_SINGLE_SS_RIGHT_COLUMN_NEIGHBORS;
  } else {
    neighbors = CHROMA_NO_SS_RIGHT_COLUMN_NEIGHBORS;
  }

  apply_temporal_filter_chroma_8(
      u_pre + uv_blk_col, v_pre + uv_blk_col, uv_pre_stride, uv_height, ss_x,
      ss_y, strength, u_accum + uv_blk_col, u_count + uv_blk_col,
      v_accum + uv_blk_col, v_count + uv_blk_col, y_dist + blk_col,
      u_dist + uv_blk_col, v_dist + uv_blk_col, neighbors, top_weight,
      bottom_weight, NULL);
}

void vp9_apply_temporal_filter_neon(
    const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre,
    int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src,
    int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre,
    int uv_pre_stride, unsigned int block_width, unsigned int block_height,
    int ss_x, int ss_y, int strength, const int *const blk_fw,
    int use_whole_blk, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum,
    uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count) {
  const unsigned int chroma_height = block_height >> ss_y,
                     chroma_width = block_width >> ss_x;

  DECLARE_ALIGNED(16, uint16_t, y_dist[BH * DIST_STRIDE]) = { 0 };
  DECLARE_ALIGNED(16, uint16_t, u_dist[BH * DIST_STRIDE]) = { 0 };
  DECLARE_ALIGNED(16, uint16_t, v_dist[BH * DIST_STRIDE]) = { 0 };
  const int *blk_fw_ptr = blk_fw;

  uint16_t *y_dist_ptr = y_dist + 1, *u_dist_ptr = u_dist + 1,
           *v_dist_ptr = v_dist + 1;
  const uint8_t *y_src_ptr = y_src, *u_src_ptr = u_src, *v_src_ptr = v_src;
  const uint8_t *y_pre_ptr = y_pre, *u_pre_ptr = u_pre, *v_pre_ptr = v_pre;

  // Loop variables
  unsigned int row, blk_col;

  assert(block_width <= BW && "block width too large");
  assert(block_height <= BH && "block height too large");
  assert(block_width % 16 == 0 && "block width must be multiple of 16");
  assert(block_height % 2 == 0 && "block height must be even");
  assert((ss_x == 0 || ss_x == 1) && (ss_y == 0 || ss_y == 1) &&
         "invalid chroma subsampling");
  assert(strength >= 0 && strength <= 6 && "invalid temporal filter strength");
  assert(blk_fw[0] >= 0 && "filter weight must be positive");
  assert(
      (use_whole_blk || (blk_fw[1] >= 0 && blk_fw[2] >= 0 && blk_fw[3] >= 0)) &&
      "subblock filter weight must be positive");
  assert(blk_fw[0] <= 2 && "subblock filter weight must be less than 2");
  assert(
      (use_whole_blk || (blk_fw[1] <= 2 && blk_fw[2] <= 2 && blk_fw[3] <= 2)) &&
      "subblock filter weight must be less than 2");

  // Precompute the difference squared
  for (row = 0; row < block_height; row++) {
    for (blk_col = 0; blk_col < block_width; blk_col += 16) {
      store_dist_16(y_src_ptr + blk_col, y_pre_ptr + blk_col,
                    y_dist_ptr + blk_col);
    }
    y_src_ptr += y_src_stride;
    y_pre_ptr += y_pre_stride;
    y_dist_ptr += DIST_STRIDE;
  }

  for (row = 0; row < chroma_height; row++) {
    for (blk_col = 0; blk_col < chroma_width; blk_col += 8) {
      store_dist_8(u_src_ptr + blk_col, u_pre_ptr + blk_col,
                   u_dist_ptr + blk_col);
      store_dist_8(v_src_ptr + blk_col, v_pre_ptr + blk_col,
                   v_dist_ptr + blk_col);
    }

    u_src_ptr += uv_src_stride;
    u_pre_ptr += uv_pre_stride;
    u_dist_ptr += DIST_STRIDE;
    v_src_ptr += uv_src_stride;
    v_pre_ptr += uv_pre_stride;
    v_dist_ptr += DIST_STRIDE;
  }

  y_dist_ptr = y_dist + 1;
  u_dist_ptr = u_dist + 1;
  v_dist_ptr = v_dist + 1;

  apply_temporal_filter_luma(y_pre, y_pre_stride, block_width, block_height,
                             ss_x, ss_y, strength, blk_fw_ptr, use_whole_blk,
                             y_accum, y_count, y_dist_ptr, u_dist_ptr,
                             v_dist_ptr);

  apply_temporal_filter_chroma(u_pre, v_pre, uv_pre_stride, block_width,
                               block_height, ss_x, ss_y, strength, blk_fw_ptr,
                               use_whole_blk, u_accum, u_count, v_accum,
                               v_count, y_dist_ptr, u_dist_ptr, v_dist_ptr);
}

static INLINE uint8x8_t convolve12_8(const int16x8_t s0, const int16x8_t s1,
                                     const int16x8_t s2, const int16x8_t s3,
                                     const int16x8_t s4, const int16x8_t s5,
                                     const int16x8_t s6, const int16x8_t s7,
                                     const int16x8_t s8, const int16x8_t s9,
                                     const int16x8_t sA, const int16x8_t sB,
                                     const int16x8_t filter_0_7,
                                     const int16x4_t filter_8_11) {
  const int16x4_t filter_0_3 = vget_low_s16(filter_0_7);
  const int16x4_t filter_4_7 = vget_high_s16(filter_0_7);

  int16x8_t sum = vmulq_lane_s16(s0, filter_0_3, 0);
  sum = vmlaq_lane_s16(sum, s1, filter_0_3, 1);
  sum = vmlaq_lane_s16(sum, s2, filter_0_3, 2);
  sum = vmlaq_lane_s16(sum, s3, filter_0_3, 3);
  sum = vmlaq_lane_s16(sum, s4, filter_4_7, 0);

  sum = vmlaq_lane_s16(sum, s7, filter_4_7, 3);
  sum = vmlaq_lane_s16(sum, s8, filter_8_11, 0);
  sum = vmlaq_lane_s16(sum, s9, filter_8_11, 1);
  sum = vmlaq_lane_s16(sum, sA, filter_8_11, 2);
  sum = vmlaq_lane_s16(sum, sB, filter_8_11, 3);

  // Saturating addition is required for the largest filter taps to avoid
  // overflow (while staying in 16-bit elements.)
  sum = vqaddq_s16(sum, vmulq_lane_s16(s5, filter_4_7, 1));
  sum = vqaddq_s16(sum, vmulq_lane_s16(s6, filter_4_7, 2));

  return vqrshrun_n_s16(sum, FILTER_BITS);
}

void vpx_convolve12_horiz_neon(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel12 *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4, int w,
                               int h) {
  // Scaling not supported by Neon implementation.
  if (x_step_q4 != 16) {
    vpx_convolve12_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h % 4 == 0);

  const int16x8_t filter_0_7 = vld1q_s16(filter[x0_q4]);
  const int16x4_t filter_8_11 = vld1_s16(filter[x0_q4] + 8);

  src -= MAX_FILTER_TAP / 2 - 1;

  do {
    const uint8_t *s = src;
    uint8_t *d = dst;
    int width = w;

    uint8x8_t t0, t1, t2, t3;
    load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
    transpose_u8_8x4(&t0, &t1, &t2, &t3);

    int16x4_t s0 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s1 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s2 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    int16x4_t s3 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));
    int16x4_t s4 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s5 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t s6 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
    int16x4_t s7 = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

    int16x8_t s0s1 = vcombine_s16(s0, s1);
    int16x8_t s1s2 = vcombine_s16(s1, s2);
    int16x8_t s2s3 = vcombine_s16(s2, s3);
    int16x8_t s3s4 = vcombine_s16(s3, s4);
    int16x8_t s4s5 = vcombine_s16(s4, s5);
    int16x8_t s5s6 = vcombine_s16(s5, s6);
    int16x8_t s6s7 = vcombine_s16(s6, s7);

    load_u8_8x4(s + 8, src_stride, &t0, &t1, &t2, &t3);
    transpose_u8_8x4(&t0, &t1, &t2, &t3);

    int16x4_t s8 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
    int16x4_t s9 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
    int16x4_t sA = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));

    int16x8_t s7s8 = vcombine_s16(s7, s8);
    int16x8_t s8s9 = vcombine_s16(s8, s9);
    int16x8_t s9sA = vcombine_s16(s9, sA);

    s += 11;

    do {
      load_u8_8x4(s, src_stride, &t0, &t1, &t2, &t3);
      transpose_u8_8x4(&t0, &t1, &t2, &t3);

      int16x4_t sB = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t0)));
      int16x4_t sC = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t1)));
      int16x4_t sD = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t2)));
      int16x4_t sE = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(t3)));

      int16x8_t sAsB = vcombine_s16(sA, sB);
      int16x8_t sBsC = vcombine_s16(sB, sC);
      int16x8_t sCsD = vcombine_s16(sC, sD);
      int16x8_t sDsE = vcombine_s16(sD, sE);

      uint8x8_t d01 =
          convolve12_8(s0s1, s1s2, s2s3, s3s4, s4s5, s5s6, s6s7, s7s8, s8s9,
                       s9sA, sAsB, sBsC, filter_0_7, filter_8_11);
      uint8x8_t d23 =
          convolve12_8(s2s3, s3s4, s4s5, s5s6, s6s7, s7s8, s8s9, s9sA, sAsB,
                       sBsC, sCsD, sDsE, filter_0_7, filter_8_11);

      transpose_u8_4x4(&d01, &d23);

      store_u8(d + 0 * dst_stride, 2 * dst_stride, d01);
      store_u8(d + 1 * dst_stride, 2 * dst_stride, d23);

      s0s1 = s4s5;
      s1s2 = s5s6;
      s2s3 = s6s7;
      s3s4 = s7s8;
      s4s5 = s8s9;
      s5s6 = s9sA;
      s6s7 = sAsB;
      s7s8 = sBsC;
      s8s9 = sCsD;
      s9sA = sDsE;
      sA = sE;
      s += 4;
      d += 4;
      width -= 4;
    } while (width != 0);
    src += 4 * src_stride;
    dst += 4 * dst_stride;
    h -= 4;
  } while (h != 0);
}

void vpx_convolve12_vert_neon(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel12 *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4, int w,
                              int h) {
  // Scaling not supported by Neon implementation.
  if (y_step_q4 != 16) {
    vpx_convolve12_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                          x_step_q4, y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  const int16x8_t filter_0_7 = vld1q_s16(filter[y0_q4]);
  const int16x4_t filter_8_11 = vld1_s16(filter[y0_q4] + 8);

  src -= src_stride * (MAX_FILTER_TAP / 2 - 1);

  do {
    const uint8_t *s = src;
    uint8_t *d = dst;
    int height = h;

    uint8x8_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, tA;
    load_u8_8x11(s, src_stride, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8,
                 &t9, &tA);
    int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(t0));
    int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(t1));
    int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(t2));
    int16x8_t s3 = vreinterpretq_s16_u16(vmovl_u8(t3));
    int16x8_t s4 = vreinterpretq_s16_u16(vmovl_u8(t4));
    int16x8_t s5 = vreinterpretq_s16_u16(vmovl_u8(t5));
    int16x8_t s6 = vreinterpretq_s16_u16(vmovl_u8(t6));
    int16x8_t s7 = vreinterpretq_s16_u16(vmovl_u8(t7));
    int16x8_t s8 = vreinterpretq_s16_u16(vmovl_u8(t8));
    int16x8_t s9 = vreinterpretq_s16_u16(vmovl_u8(t9));
    int16x8_t sA = vreinterpretq_s16_u16(vmovl_u8(tA));

    s += 11 * src_stride;

    do {
      uint8x8_t tB, tC, tD, tE;
      load_u8_8x4(s, src_stride, &tB, &tC, &tD, &tE);

      int16x8_t sB = vreinterpretq_s16_u16(vmovl_u8(tB));
      int16x8_t sC = vreinterpretq_s16_u16(vmovl_u8(tC));
      int16x8_t sD = vreinterpretq_s16_u16(vmovl_u8(tD));
      int16x8_t sE = vreinterpretq_s16_u16(vmovl_u8(tE));

      uint8x8_t d0 = convolve12_8(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA,
                                  sB, filter_0_7, filter_8_11);
      uint8x8_t d1 = convolve12_8(s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB,
                                  sC, filter_0_7, filter_8_11);
      uint8x8_t d2 = convolve12_8(s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC,
                                  sD, filter_0_7, filter_8_11);
      uint8x8_t d3 = convolve12_8(s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD,
                                  sE, filter_0_7, filter_8_11);

      store_u8_8x4(d, dst_stride, d0, d1, d2, d3);

      s0 = s4;
      s1 = s5;
      s2 = s6;
      s3 = s7;
      s4 = s8;
      s5 = s9;
      s6 = sA;
      s7 = sB;
      s8 = sC;
      s9 = sD;
      sA = sE;
      s += 4 * src_stride;
      d += 4 * dst_stride;
      height -= 4;
    } while (height != 0);
    src += 8;
    dst += 8;
    w -= 8;
  } while (w != 0);
}

void vpx_convolve12_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                         ptrdiff_t dst_stride, const InterpKernel12 *filter,
                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                         int w, int h) {
  // Scaling not supported by Neon implementation.
  if (x_step_q4 != 16 || y_step_q4 != 16) {
    vpx_convolve12_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                     y0_q4, y_step_q4, w, h);
    return;
  }

  assert(w == 32 || w == 16 || w == 8);
  assert(h == 32 || h == 16 || h == 8);

  DECLARE_ALIGNED(32, uint8_t, im_block[BW * (BH + MAX_FILTER_TAP)]);

  const int im_stride = BW;
  // Account for the vertical pass needing MAX_FILTER_TAP / 2 - 1 lines prior
  // and MAX_FILTER_TAP / 2 lines post. (+1 to make total divisible by 4.)
  const int im_height = h + MAX_FILTER_TAP;
  const ptrdiff_t border_offset = MAX_FILTER_TAP / 2 - 1;

  // Filter starting border_offset rows up.
  vpx_convolve12_horiz_neon(src - src_stride * border_offset, src_stride,
                            im_block, im_stride, filter, x0_q4, x_step_q4,
                            y0_q4, y_step_q4, w, im_height);

  vpx_convolve12_vert_neon(im_block + im_stride * border_offset, im_stride, dst,
                           dst_stride, filter, x0_q4, x_step_q4, y0_q4,
                           y_step_q4, w, h);
}
