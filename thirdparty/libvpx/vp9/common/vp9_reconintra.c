/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#if CONFIG_VP9_HIGHBITDEPTH
#include "vpx_dsp/vpx_dsp_common.h"
#endif  // CONFIG_VP9_HIGHBITDEPTH
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_once.h"

#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_onyxc_int.h"

const TX_TYPE intra_mode_to_tx_type_lookup[INTRA_MODES] = {
  DCT_DCT,    // DC
  ADST_DCT,   // V
  DCT_ADST,   // H
  DCT_DCT,    // D45
  ADST_ADST,  // D135
  ADST_DCT,   // D117
  DCT_ADST,   // D153
  DCT_ADST,   // D207
  ADST_DCT,   // D63
  ADST_ADST,  // TM
};

enum {
  NEED_LEFT = 1 << 1,
  NEED_ABOVE = 1 << 2,
  NEED_ABOVERIGHT = 1 << 3,
};

static const uint8_t extend_modes[INTRA_MODES] = {
  NEED_ABOVE | NEED_LEFT,       // DC
  NEED_ABOVE,                   // V
  NEED_LEFT,                    // H
  NEED_ABOVERIGHT,              // D45
  NEED_LEFT | NEED_ABOVE,       // D135
  NEED_LEFT | NEED_ABOVE,       // D117
  NEED_LEFT | NEED_ABOVE,       // D153
  NEED_LEFT,                    // D207
  NEED_ABOVERIGHT,              // D63
  NEED_LEFT | NEED_ABOVE,       // TM
};

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[INTRA_MODES][TX_SIZES];
static intra_pred_fn dc_pred[2][2][TX_SIZES];

#if CONFIG_VP9_HIGHBITDEPTH
typedef void (*intra_high_pred_fn)(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd);
static intra_high_pred_fn pred_high[INTRA_MODES][4];
static intra_high_pred_fn dc_pred_high[2][2][4];
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void vp9_init_intra_predictors_internal(void) {
#define INIT_ALL_SIZES(p, type) \
  p[TX_4X4] = vpx_##type##_predictor_4x4; \
  p[TX_8X8] = vpx_##type##_predictor_8x8; \
  p[TX_16X16] = vpx_##type##_predictor_16x16; \
  p[TX_32X32] = vpx_##type##_predictor_32x32

  INIT_ALL_SIZES(pred[V_PRED], v);
  INIT_ALL_SIZES(pred[H_PRED], h);
  INIT_ALL_SIZES(pred[D207_PRED], d207);
  INIT_ALL_SIZES(pred[D45_PRED], d45);
  INIT_ALL_SIZES(pred[D63_PRED], d63);
  INIT_ALL_SIZES(pred[D117_PRED], d117);
  INIT_ALL_SIZES(pred[D135_PRED], d135);
  INIT_ALL_SIZES(pred[D153_PRED], d153);
  INIT_ALL_SIZES(pred[TM_PRED], tm);

  INIT_ALL_SIZES(dc_pred[0][0], dc_128);
  INIT_ALL_SIZES(dc_pred[0][1], dc_top);
  INIT_ALL_SIZES(dc_pred[1][0], dc_left);
  INIT_ALL_SIZES(dc_pred[1][1], dc);

#if CONFIG_VP9_HIGHBITDEPTH
  INIT_ALL_SIZES(pred_high[V_PRED], highbd_v);
  INIT_ALL_SIZES(pred_high[H_PRED], highbd_h);
  INIT_ALL_SIZES(pred_high[D207_PRED], highbd_d207);
  INIT_ALL_SIZES(pred_high[D45_PRED], highbd_d45);
  INIT_ALL_SIZES(pred_high[D63_PRED], highbd_d63);
  INIT_ALL_SIZES(pred_high[D117_PRED], highbd_d117);
  INIT_ALL_SIZES(pred_high[D135_PRED], highbd_d135);
  INIT_ALL_SIZES(pred_high[D153_PRED], highbd_d153);
  INIT_ALL_SIZES(pred_high[TM_PRED], highbd_tm);

  INIT_ALL_SIZES(dc_pred_high[0][0], highbd_dc_128);
  INIT_ALL_SIZES(dc_pred_high[0][1], highbd_dc_top);
  INIT_ALL_SIZES(dc_pred_high[1][0], highbd_dc_left);
  INIT_ALL_SIZES(dc_pred_high[1][1], highbd_dc);
#endif  // CONFIG_VP9_HIGHBITDEPTH

#undef intra_pred_allsizes
}

#if CONFIG_VP9_HIGHBITDEPTH
static void build_intra_predictors_high(const MACROBLOCKD *xd,
                                        const uint8_t *ref8,
                                        int ref_stride,
                                        uint8_t *dst8,
                                        int dst_stride,
                                        PREDICTION_MODE mode,
                                        TX_SIZE tx_size,
                                        int up_available,
                                        int left_available,
                                        int right_available,
                                        int x, int y,
                                        int plane, int bd) {
  int i;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  DECLARE_ALIGNED(16, uint16_t, left_col[32]);
  DECLARE_ALIGNED(16, uint16_t, above_data[64 + 16]);
  uint16_t *above_row = above_data + 16;
  const uint16_t *const_above_row = above_row;
  const int bs = 4 << tx_size;
  int frame_width, frame_height;
  int x0, y0;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int need_left = extend_modes[mode] & NEED_LEFT;
  const int need_above = extend_modes[mode] & NEED_ABOVE;
  const int need_aboveright = extend_modes[mode] & NEED_ABOVERIGHT;
  int base = 128 << (bd - 8);
  // 127 127 127 .. 127 127 127 127 127 127
  // 129  A   B  ..  Y   Z
  // 129  C   D  ..  W   X
  // 129  E   F  ..  U   V
  // 129  G   H  ..  S   T   T   T   T   T
  // For 10 bit and 12 bit, 127 and 129 are replaced by base -1 and base + 1.

  // Get current frame pointer, width and height.
  if (plane == 0) {
    frame_width = xd->cur_buf->y_width;
    frame_height = xd->cur_buf->y_height;
  } else {
    frame_width = xd->cur_buf->uv_width;
    frame_height = xd->cur_buf->uv_height;
  }

  // Get block position in current frame.
  x0 = (-xd->mb_to_left_edge >> (3 + pd->subsampling_x)) + x;
  y0 = (-xd->mb_to_top_edge >> (3 + pd->subsampling_y)) + y;

  // NEED_LEFT
  if (need_left) {
    if (left_available) {
      if (xd->mb_to_bottom_edge < 0) {
        /* slower path if the block needs border extension */
        if (y0 + bs <= frame_height) {
          for (i = 0; i < bs; ++i)
            left_col[i] = ref[i * ref_stride - 1];
        } else {
          const int extend_bottom = frame_height - y0;
          for (i = 0; i < extend_bottom; ++i)
            left_col[i] = ref[i * ref_stride - 1];
          for (; i < bs; ++i)
            left_col[i] = ref[(extend_bottom - 1) * ref_stride - 1];
        }
      } else {
        /* faster path if the block does not need extension */
        for (i = 0; i < bs; ++i)
          left_col[i] = ref[i * ref_stride - 1];
      }
    } else {
      vpx_memset16(left_col, base + 1, bs);
    }
  }

  // NEED_ABOVE
  if (need_above) {
    if (up_available) {
      const uint16_t *above_ref = ref - ref_stride;
      if (xd->mb_to_right_edge < 0) {
        /* slower path if the block needs border extension */
        if (x0 + bs <= frame_width) {
          memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
        } else if (x0 <= frame_width) {
          const int r = frame_width - x0;
          memcpy(above_row, above_ref, r * sizeof(above_row[0]));
          vpx_memset16(above_row + r, above_row[r - 1], x0 + bs - frame_width);
        }
      } else {
        /* faster path if the block does not need extension */
        if (bs == 4 && right_available && left_available) {
          const_above_row = above_ref;
        } else {
          memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
        }
      }
      above_row[-1] = left_available ? above_ref[-1] : (base + 1);
    } else {
      vpx_memset16(above_row, base - 1, bs);
      above_row[-1] = base - 1;
    }
  }

  // NEED_ABOVERIGHT
  if (need_aboveright) {
    if (up_available) {
      const uint16_t *above_ref = ref - ref_stride;
      if (xd->mb_to_right_edge < 0) {
        /* slower path if the block needs border extension */
        if (x0 + 2 * bs <= frame_width) {
          if (right_available && bs == 4) {
            memcpy(above_row, above_ref, 2 * bs * sizeof(above_row[0]));
          } else {
            memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
            vpx_memset16(above_row + bs, above_row[bs - 1], bs);
          }
        } else if (x0 + bs <= frame_width) {
          const int r = frame_width - x0;
          if (right_available && bs == 4) {
            memcpy(above_row, above_ref, r * sizeof(above_row[0]));
            vpx_memset16(above_row + r, above_row[r - 1],
                         x0 + 2 * bs - frame_width);
          } else {
            memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
            vpx_memset16(above_row + bs, above_row[bs - 1], bs);
          }
        } else if (x0 <= frame_width) {
          const int r = frame_width - x0;
          memcpy(above_row, above_ref, r * sizeof(above_row[0]));
          vpx_memset16(above_row + r, above_row[r - 1],
                       x0 + 2 * bs - frame_width);
        }
        above_row[-1] = left_available ? above_ref[-1] : (base + 1);
      } else {
        /* faster path if the block does not need extension */
        if (bs == 4 && right_available && left_available) {
          const_above_row = above_ref;
        } else {
          memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
          if (bs == 4 && right_available)
            memcpy(above_row + bs, above_ref + bs, bs * sizeof(above_row[0]));
          else
            vpx_memset16(above_row + bs, above_row[bs - 1], bs);
          above_row[-1] = left_available ? above_ref[-1] : (base + 1);
        }
      }
    } else {
      vpx_memset16(above_row, base - 1, bs * 2);
      above_row[-1] = base - 1;
    }
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred_high[left_available][up_available][tx_size](dst, dst_stride,
                                                        const_above_row,
                                                        left_col, xd->bd);
  } else {
    pred_high[mode][tx_size](dst, dst_stride, const_above_row, left_col,
                             xd->bd);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void build_intra_predictors(const MACROBLOCKD *xd, const uint8_t *ref,
                                   int ref_stride, uint8_t *dst, int dst_stride,
                                   PREDICTION_MODE mode, TX_SIZE tx_size,
                                   int up_available, int left_available,
                                   int right_available, int x, int y,
                                   int plane) {
  int i;
  DECLARE_ALIGNED(16, uint8_t, left_col[32]);
  DECLARE_ALIGNED(16, uint8_t, above_data[64 + 16]);
  uint8_t *above_row = above_data + 16;
  const uint8_t *const_above_row = above_row;
  const int bs = 4 << tx_size;
  int frame_width, frame_height;
  int x0, y0;
  const struct macroblockd_plane *const pd = &xd->plane[plane];

  // 127 127 127 .. 127 127 127 127 127 127
  // 129  A   B  ..  Y   Z
  // 129  C   D  ..  W   X
  // 129  E   F  ..  U   V
  // 129  G   H  ..  S   T   T   T   T   T
  // ..

  // Get current frame pointer, width and height.
  if (plane == 0) {
    frame_width = xd->cur_buf->y_width;
    frame_height = xd->cur_buf->y_height;
  } else {
    frame_width = xd->cur_buf->uv_width;
    frame_height = xd->cur_buf->uv_height;
  }

  // Get block position in current frame.
  x0 = (-xd->mb_to_left_edge >> (3 + pd->subsampling_x)) + x;
  y0 = (-xd->mb_to_top_edge >> (3 + pd->subsampling_y)) + y;

  // NEED_LEFT
  if (extend_modes[mode] & NEED_LEFT) {
    if (left_available) {
      if (xd->mb_to_bottom_edge < 0) {
        /* slower path if the block needs border extension */
        if (y0 + bs <= frame_height) {
          for (i = 0; i < bs; ++i)
            left_col[i] = ref[i * ref_stride - 1];
        } else {
          const int extend_bottom = frame_height - y0;
          for (i = 0; i < extend_bottom; ++i)
            left_col[i] = ref[i * ref_stride - 1];
          for (; i < bs; ++i)
            left_col[i] = ref[(extend_bottom - 1) * ref_stride - 1];
        }
      } else {
        /* faster path if the block does not need extension */
        for (i = 0; i < bs; ++i)
          left_col[i] = ref[i * ref_stride - 1];
      }
    } else {
      memset(left_col, 129, bs);
    }
  }

  // NEED_ABOVE
  if (extend_modes[mode] & NEED_ABOVE) {
    if (up_available) {
      const uint8_t *above_ref = ref - ref_stride;
      if (xd->mb_to_right_edge < 0) {
        /* slower path if the block needs border extension */
        if (x0 + bs <= frame_width) {
          memcpy(above_row, above_ref, bs);
        } else if (x0 <= frame_width) {
          const int r = frame_width - x0;
          memcpy(above_row, above_ref, r);
          memset(above_row + r, above_row[r - 1], x0 + bs - frame_width);
        }
      } else {
        /* faster path if the block does not need extension */
        if (bs == 4 && right_available && left_available) {
          const_above_row = above_ref;
        } else {
          memcpy(above_row, above_ref, bs);
        }
      }
      above_row[-1] = left_available ? above_ref[-1] : 129;
    } else {
      memset(above_row, 127, bs);
      above_row[-1] = 127;
    }
  }

  // NEED_ABOVERIGHT
  if (extend_modes[mode] & NEED_ABOVERIGHT) {
    if (up_available) {
      const uint8_t *above_ref = ref - ref_stride;
      if (xd->mb_to_right_edge < 0) {
        /* slower path if the block needs border extension */
        if (x0 + 2 * bs <= frame_width) {
          if (right_available && bs == 4) {
            memcpy(above_row, above_ref, 2 * bs);
          } else {
            memcpy(above_row, above_ref, bs);
            memset(above_row + bs, above_row[bs - 1], bs);
          }
        } else if (x0 + bs <= frame_width) {
          const int r = frame_width - x0;
          if (right_available && bs == 4) {
            memcpy(above_row, above_ref, r);
            memset(above_row + r, above_row[r - 1], x0 + 2 * bs - frame_width);
          } else {
            memcpy(above_row, above_ref, bs);
            memset(above_row + bs, above_row[bs - 1], bs);
          }
        } else if (x0 <= frame_width) {
          const int r = frame_width - x0;
          memcpy(above_row, above_ref, r);
          memset(above_row + r, above_row[r - 1], x0 + 2 * bs - frame_width);
        }
      } else {
        /* faster path if the block does not need extension */
        if (bs == 4 && right_available && left_available) {
          const_above_row = above_ref;
        } else {
          memcpy(above_row, above_ref, bs);
          if (bs == 4 && right_available)
            memcpy(above_row + bs, above_ref + bs, bs);
          else
            memset(above_row + bs, above_row[bs - 1], bs);
        }
      }
      above_row[-1] = left_available ? above_ref[-1] : 129;
    } else {
      memset(above_row, 127, bs * 2);
      above_row[-1] = 127;
    }
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred[left_available][up_available][tx_size](dst, dst_stride,
                                                   const_above_row, left_col);
  } else {
    pred[mode][tx_size](dst, dst_stride, const_above_row, left_col);
  }
}

void vp9_predict_intra_block(const MACROBLOCKD *xd, int bwl_in,
                             TX_SIZE tx_size, PREDICTION_MODE mode,
                             const uint8_t *ref, int ref_stride,
                             uint8_t *dst, int dst_stride,
                             int aoff, int loff, int plane) {
  const int bw = (1 << bwl_in);
  const int txw = (1 << tx_size);
  const int have_top = loff || (xd->above_mi != NULL);
  const int have_left = aoff || (xd->left_mi != NULL);
  const int have_right = (aoff + txw) < bw;
  const int x = aoff * 4;
  const int y = loff * 4;

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    build_intra_predictors_high(xd, ref, ref_stride, dst, dst_stride, mode,
                                tx_size, have_top, have_left, have_right,
                                x, y, plane, xd->bd);
    return;
  }
#endif
  build_intra_predictors(xd, ref, ref_stride, dst, dst_stride, mode, tx_size,
                         have_top, have_left, have_right, x, y, plane);
}

void vp9_init_intra_predictors(void) {
  once(vp9_init_intra_predictors_internal);
}
