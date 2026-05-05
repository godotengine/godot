/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/ppc/types_vsx.h"

static VPX_FORCE_INLINE void subtract_block4x4(
    int16_t *diff, ptrdiff_t diff_stride, const uint8_t *src,
    ptrdiff_t src_stride, const uint8_t *pred, ptrdiff_t pred_stride) {
  int16_t *diff1 = diff + 2 * diff_stride;
  const uint8_t *src1 = src + 2 * src_stride;
  const uint8_t *pred1 = pred + 2 * pred_stride;

  const int16x8_t d0 = vec_vsx_ld(0, diff);
  const int16x8_t d1 = vec_vsx_ld(0, diff + diff_stride);
  const int16x8_t d2 = vec_vsx_ld(0, diff1);
  const int16x8_t d3 = vec_vsx_ld(0, diff1 + diff_stride);

  const uint8x16_t s0 = read4x2(src, (int)src_stride);
  const uint8x16_t p0 = read4x2(pred, (int)pred_stride);
  const uint8x16_t s1 = read4x2(src1, (int)src_stride);
  const uint8x16_t p1 = read4x2(pred1, (int)pred_stride);

  const int16x8_t da = vec_sub(unpack_to_s16_h(s0), unpack_to_s16_h(p0));
  const int16x8_t db = vec_sub(unpack_to_s16_h(s1), unpack_to_s16_h(p1));

  vec_vsx_st(xxpermdi(da, d0, 1), 0, diff);
  vec_vsx_st(xxpermdi(da, d1, 3), 0, diff + diff_stride);
  vec_vsx_st(xxpermdi(db, d2, 1), 0, diff1);
  vec_vsx_st(xxpermdi(db, d3, 3), 0, diff1 + diff_stride);
}

void vpx_subtract_block_vsx(int rows, int cols, int16_t *diff,
                            ptrdiff_t diff_stride, const uint8_t *src,
                            ptrdiff_t src_stride, const uint8_t *pred,
                            ptrdiff_t pred_stride) {
  int r = rows, c;

  switch (cols) {
    case 64:
    case 32:
      do {
        for (c = 0; c < cols; c += 32) {
          const uint8x16_t s0 = vec_vsx_ld(0, src + c);
          const uint8x16_t s1 = vec_vsx_ld(16, src + c);
          const uint8x16_t p0 = vec_vsx_ld(0, pred + c);
          const uint8x16_t p1 = vec_vsx_ld(16, pred + c);
          const int16x8_t d0l =
              vec_sub(unpack_to_s16_l(s0), unpack_to_s16_l(p0));
          const int16x8_t d0h =
              vec_sub(unpack_to_s16_h(s0), unpack_to_s16_h(p0));
          const int16x8_t d1l =
              vec_sub(unpack_to_s16_l(s1), unpack_to_s16_l(p1));
          const int16x8_t d1h =
              vec_sub(unpack_to_s16_h(s1), unpack_to_s16_h(p1));
          vec_vsx_st(d0h, 0, diff + c);
          vec_vsx_st(d0l, 16, diff + c);
          vec_vsx_st(d1h, 0, diff + c + 16);
          vec_vsx_st(d1l, 16, diff + c + 16);
        }
        diff += diff_stride;
        pred += pred_stride;
        src += src_stride;
      } while (--r);
      break;
    case 16:
      do {
        const uint8x16_t s0 = vec_vsx_ld(0, src);
        const uint8x16_t p0 = vec_vsx_ld(0, pred);
        const int16x8_t d0l = vec_sub(unpack_to_s16_l(s0), unpack_to_s16_l(p0));
        const int16x8_t d0h = vec_sub(unpack_to_s16_h(s0), unpack_to_s16_h(p0));
        vec_vsx_st(d0h, 0, diff);
        vec_vsx_st(d0l, 16, diff);
        diff += diff_stride;
        pred += pred_stride;
        src += src_stride;
      } while (--r);
      break;
    case 8:
      do {
        const uint8x16_t s0 = vec_vsx_ld(0, src);
        const uint8x16_t p0 = vec_vsx_ld(0, pred);
        const int16x8_t d0h = vec_sub(unpack_to_s16_h(s0), unpack_to_s16_h(p0));
        vec_vsx_st(d0h, 0, diff);
        diff += diff_stride;
        pred += pred_stride;
        src += src_stride;
      } while (--r);
      break;
    case 4:
      subtract_block4x4(diff, diff_stride, src, src_stride, pred, pred_stride);
      if (r > 4) {
        diff += 4 * diff_stride;
        pred += 4 * pred_stride;
        src += 4 * src_stride;

        subtract_block4x4(diff, diff_stride,

                          src, src_stride,

                          pred, pred_stride);
      }
      break;
    default: assert(0);  // unreachable
  }
}
