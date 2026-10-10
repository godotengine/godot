/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <arm_neon.h>

#include "vpx_dsp/vpx_dsp_common.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vpx_ports/mem.h"

#ifdef __GNUC__
#define LIKELY(v) __builtin_expect(v, 1)
#define UNLIKELY(v) __builtin_expect(v, 0)
#else
#define LIKELY(v) (v)
#define UNLIKELY(v) (v)
#endif

static INLINE int_mv pack_int_mv(int16_t row, int16_t col) {
  int_mv result;
  result.as_mv.row = row;
  result.as_mv.col = col;
  return result;
}

/*****************************************************************************
 * This function utilizes 3 properties of the cost function lookup tables,   *
 * constructed in using 'cal_nmvjointsadcost' and 'cal_nmvsadcosts' in       *
 * vp9_encoder.c.                                                            *
 * For the joint cost:                                                       *
 *   - mvjointsadcost[1] == mvjointsadcost[2] == mvjointsadcost[3]           *
 * For the component costs:                                                  *
 *   - For all i: mvsadcost[0][i] == mvsadcost[1][i]                         *
 *         (Equal costs for both components)                                 *
 *   - For all i: mvsadcost[0][i] == mvsadcost[0][-i]                        *
 *         (Cost function is even)                                           *
 * If these do not hold, then this function cannot be used without           *
 * modification, in which case you can revert to using the C implementation, *
 * which does not rely on these properties.                                  *
 *****************************************************************************/
int vp9_diamond_search_sad_neon(const MACROBLOCK *x,
                                const search_site_config *cfg, MV *ref_mv,
                                uint32_t start_mv_sad, MV *best_mv,
                                int search_param, int sad_per_bit, int *num00,
                                const vp9_sad_fn_ptr_t *sad_fn_ptr,
                                const MV *center_mv) {
  static const uint32_t data[4] = { 0, 1, 2, 3 };
  const uint32x4_t v_idx_d = vld1q_u32((const uint32_t *)data);

  const int32x4_t zero_s32 = vdupq_n_s32(0);
  const int_mv maxmv = pack_int_mv(x->mv_limits.row_max, x->mv_limits.col_max);
  const int16x8_t v_max_mv_w = vreinterpretq_s16_s32(vdupq_n_s32(maxmv.as_int));
  const int_mv minmv = pack_int_mv(x->mv_limits.row_min, x->mv_limits.col_min);
  const int16x8_t v_min_mv_w = vreinterpretq_s16_s32(vdupq_n_s32(minmv.as_int));

  const int32x4_t v_spb_d = vdupq_n_s32(sad_per_bit);

  const int32x4_t v_joint_cost_0_d = vdupq_n_s32(x->nmvjointsadcost[0]);
  const int32x4_t v_joint_cost_1_d = vdupq_n_s32(x->nmvjointsadcost[1]);

  // search_param determines the length of the initial step and hence the number
  // of iterations.
  // 0 = initial step (MAX_FIRST_STEP) pel
  // 1 = (MAX_FIRST_STEP/2) pel,
  // 2 = (MAX_FIRST_STEP/4) pel...
  const MV *ss_mv = &cfg->ss_mv[cfg->searches_per_step * search_param];
  const intptr_t *ss_os = &cfg->ss_os[cfg->searches_per_step * search_param];
  const int tot_steps = cfg->total_steps - search_param;

  const int_mv fcenter_mv =
      pack_int_mv(center_mv->row >> 3, center_mv->col >> 3);
  const int16x8_t vfcmv = vreinterpretq_s16_s32(vdupq_n_s32(fcenter_mv.as_int));

  const int ref_row = ref_mv->row;
  const int ref_col = ref_mv->col;

  int_mv bmv = pack_int_mv(ref_row, ref_col);
  int_mv new_bmv = bmv;
  int16x8_t v_bmv_w = vreinterpretq_s16_s32(vdupq_n_s32(bmv.as_int));

  const int what_stride = x->plane[0].src.stride;
  const int in_what_stride = x->e_mbd.plane[0].pre[0].stride;
  const uint8_t *const what = x->plane[0].src.buf;
  const uint8_t *const in_what =
      x->e_mbd.plane[0].pre[0].buf + ref_row * in_what_stride + ref_col;

  // Work out the start point for the search
  const uint8_t *best_address = in_what;
  const uint8_t *new_best_address = best_address;
#if VPX_ARCH_AARCH64
  int64x2_t v_ba_q = vdupq_n_s64((intptr_t)best_address);
#else
  int32x4_t v_ba_d = vdupq_n_s32((intptr_t)best_address);
#endif
  // Starting position
  unsigned int best_sad = start_mv_sad;
  int i, j, step;

  // Check the prerequisite cost function properties that are easy to check
  // in an assert. See the function-level documentation for details on all
  // prerequisites.
  assert(x->nmvjointsadcost[1] == x->nmvjointsadcost[2]);
  assert(x->nmvjointsadcost[1] == x->nmvjointsadcost[3]);

  *num00 = 0;

  for (i = 0, step = 0; step < tot_steps; step++) {
    for (j = 0; j < cfg->searches_per_step; j += 4, i += 4) {
      int16x8_t v_diff_mv_w;
      int8x16_t v_inside_d;
      uint32x4_t v_outside_d;
      int32x4_t v_cost_d, v_sad_d;
#if VPX_ARCH_AARCH64
      int64x2_t v_blocka[2];
#else
      int32x4_t v_blocka[1];
      uint32x2_t horiz_max_0, horiz_max_1;
#endif

      uint32_t horiz_max;
      // Compute the candidate motion vectors
      const int16x8_t v_ss_mv_w = vld1q_s16((const int16_t *)&ss_mv[i]);
      const int16x8_t v_these_mv_w = vaddq_s16(v_bmv_w, v_ss_mv_w);
      // Clamp them to the search bounds
      int16x8_t v_these_mv_clamp_w = v_these_mv_w;
      v_these_mv_clamp_w = vminq_s16(v_these_mv_clamp_w, v_max_mv_w);
      v_these_mv_clamp_w = vmaxq_s16(v_these_mv_clamp_w, v_min_mv_w);
      // The ones that did not change are inside the search area
      v_inside_d = vreinterpretq_s8_u32(
          vceqq_s32(vreinterpretq_s32_s16(v_these_mv_clamp_w),
                    vreinterpretq_s32_s16(v_these_mv_w)));

      // If none of them are inside, then move on
#if VPX_ARCH_AARCH64
      horiz_max = vmaxvq_u32(vreinterpretq_u32_s8(v_inside_d));
#else
      horiz_max_0 = vmax_u32(vget_low_u32(vreinterpretq_u32_s8(v_inside_d)),
                             vget_high_u32(vreinterpretq_u32_s8(v_inside_d)));
      horiz_max_1 = vpmax_u32(horiz_max_0, horiz_max_0);
      vst1_lane_u32(&horiz_max, horiz_max_1, 0);
#endif
      if (LIKELY(horiz_max == 0)) {
        continue;
      }

      // The inverse mask indicates which of the MVs are outside
      v_outside_d =
          vreinterpretq_u32_s8(veorq_s8(v_inside_d, vdupq_n_s8((int8_t)0xff)));
      // Shift right to keep the sign bit clear, we will use this later
      // to set the cost to the maximum value.
      v_outside_d = vshrq_n_u32(v_outside_d, 1);

      // Compute the difference MV
      v_diff_mv_w = vsubq_s16(v_these_mv_clamp_w, vfcmv);
      // We utilise the fact that the cost function is even, and use the
      // absolute difference. This allows us to use unsigned indexes later
      // and reduces cache pressure somewhat as only a half of the table
      // is ever referenced.
      v_diff_mv_w = vabsq_s16(v_diff_mv_w);

      // Compute the SIMD pointer offsets.
      {
#if VPX_ARCH_AARCH64  //  sizeof(intptr_t) == 8
        // Load the offsets
        int64x2_t v_bo10_q = vld1q_s64((const int64_t *)&ss_os[i + 0]);
        int64x2_t v_bo32_q = vld1q_s64((const int64_t *)&ss_os[i + 2]);
        // Set the ones falling outside to zero
        v_bo10_q = vandq_s64(
            v_bo10_q,
            vmovl_s32(vget_low_s32(vreinterpretq_s32_s8(v_inside_d))));
        v_bo32_q = vandq_s64(
            v_bo32_q,
            vmovl_s32(vget_high_s32(vreinterpretq_s32_s8(v_inside_d))));
        // Compute the candidate addresses
        v_blocka[0] = vaddq_s64(v_ba_q, v_bo10_q);
        v_blocka[1] = vaddq_s64(v_ba_q, v_bo32_q);
#else  // sizeof(intptr_t) == 4
        int32x4_t v_bo_d = vld1q_s32((const int32_t *)&ss_os[i]);
        v_bo_d = vandq_s32(v_bo_d, vreinterpretq_s32_s8(v_inside_d));
        v_blocka[0] = vaddq_s32(v_ba_d, v_bo_d);
#endif
      }

      sad_fn_ptr->sdx4df(what, what_stride, (const uint8_t **)&v_blocka[0],
                         in_what_stride, (uint32_t *)&v_sad_d);

      // Look up the component cost of the residual motion vector
      {
        uint32_t cost[4];
        DECLARE_ALIGNED(16, int16_t, rowcol[8]);
        vst1q_s16(rowcol, v_diff_mv_w);

        // Note: This is a use case for gather instruction
        cost[0] = x->nmvsadcost[0][rowcol[0]] + x->nmvsadcost[0][rowcol[1]];
        cost[1] = x->nmvsadcost[0][rowcol[2]] + x->nmvsadcost[0][rowcol[3]];
        cost[2] = x->nmvsadcost[0][rowcol[4]] + x->nmvsadcost[0][rowcol[5]];
        cost[3] = x->nmvsadcost[0][rowcol[6]] + x->nmvsadcost[0][rowcol[7]];

        v_cost_d = vld1q_s32((int32_t *)cost);
      }

      // Now add in the joint cost
      {
        const uint32x4_t v_sel_d =
            vceqq_s32(vreinterpretq_s32_s16(v_diff_mv_w), zero_s32);
        const int32x4_t v_joint_cost_d = vreinterpretq_s32_u8(
            vbslq_u8(vreinterpretq_u8_u32(v_sel_d),
                     vreinterpretq_u8_s32(v_joint_cost_0_d),
                     vreinterpretq_u8_s32(v_joint_cost_1_d)));
        v_cost_d = vaddq_s32(v_cost_d, v_joint_cost_d);
      }

      // Multiply by sad_per_bit
      v_cost_d = vmulq_s32(v_cost_d, v_spb_d);
      // ROUND_POWER_OF_TWO(v_cost_d, VP9_PROB_COST_SHIFT)
      v_cost_d =
          vaddq_s32(v_cost_d, vdupq_n_s32(1 << (VP9_PROB_COST_SHIFT - 1)));
      v_cost_d = vshrq_n_s32(v_cost_d, VP9_PROB_COST_SHIFT);
      // Add the cost to the sad
      v_sad_d = vaddq_s32(v_sad_d, v_cost_d);

      // Make the motion vectors outside the search area have max cost
      // by or'ing in the comparison mask, this way the minimum search won't
      // pick them.
      v_sad_d = vorrq_s32(v_sad_d, vreinterpretq_s32_u32(v_outside_d));

      // Find the minimum value and index horizontally in v_sad_d
      {
        uint32_t local_best_sad;
#if VPX_ARCH_AARCH64
        local_best_sad = vminvq_u32(vreinterpretq_u32_s32(v_sad_d));
#else
        uint32x2_t horiz_min_0 =
            vmin_u32(vget_low_u32(vreinterpretq_u32_s32(v_sad_d)),
                     vget_high_u32(vreinterpretq_u32_s32(v_sad_d)));
        uint32x2_t horiz_min_1 = vpmin_u32(horiz_min_0, horiz_min_0);
        vst1_lane_u32(&local_best_sad, horiz_min_1, 0);
#endif

        // Update the global minimum if the local minimum is smaller
        if (LIKELY(local_best_sad < best_sad)) {
#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
          uint32_t local_best_idx;
          const uint32x4_t v_sel_d =
              vceqq_s32(v_sad_d, vdupq_n_s32(local_best_sad));
          uint32x4_t v_mask_d = vandq_u32(v_sel_d, v_idx_d);
          v_mask_d = vbslq_u32(v_sel_d, v_mask_d, vdupq_n_u32(0xffffffff));

#if VPX_ARCH_AARCH64
          local_best_idx = vminvq_u32(v_mask_d);
#else
          horiz_min_0 =
              vmin_u32(vget_low_u32(v_mask_d), vget_high_u32(v_mask_d));
          horiz_min_1 = vpmin_u32(horiz_min_0, horiz_min_0);
          vst1_lane_u32(&local_best_idx, horiz_min_1, 0);
#endif

          new_bmv = ((const int_mv *)&v_these_mv_w)[local_best_idx];
#if defined(__GNUC__) && __GNUC__ >= 4 && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
          new_best_address = ((const uint8_t **)v_blocka)[local_best_idx];

          best_sad = local_best_sad;
        }
      }
    }

    bmv = new_bmv;
    best_address = new_best_address;

    v_bmv_w = vreinterpretq_s16_s32(vdupq_n_s32(bmv.as_int));
#if VPX_ARCH_AARCH64
    v_ba_q = vdupq_n_s64((intptr_t)best_address);
#else
    v_ba_d = vdupq_n_s32((intptr_t)best_address);
#endif

    if (UNLIKELY(best_address == in_what)) {
      (*num00)++;
    }
  }

  *best_mv = bmv.as_mv;
  return best_sad;
}
