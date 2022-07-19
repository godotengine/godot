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
#include "vp9/common/vp9_loopfilter.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_reconinter.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_seg_common.h"

// 64 bit masks for left transform size. Each 1 represents a position where
// we should apply a loop filter across the left border of an 8x8 block
// boundary.
//
// In the case of TX_16X16->  ( in low order byte first we end up with
// a mask that looks like this
//
//    10101010
//    10101010
//    10101010
//    10101010
//    10101010
//    10101010
//    10101010
//    10101010
//
// A loopfilter should be applied to every other 8x8 horizontally.
static const uint64_t left_64x64_txform_mask[TX_SIZES]= {
  0xffffffffffffffffULL,  // TX_4X4
  0xffffffffffffffffULL,  // TX_8x8
  0x5555555555555555ULL,  // TX_16x16
  0x1111111111111111ULL,  // TX_32x32
};

// 64 bit masks for above transform size. Each 1 represents a position where
// we should apply a loop filter across the top border of an 8x8 block
// boundary.
//
// In the case of TX_32x32 ->  ( in low order byte first we end up with
// a mask that looks like this
//
//    11111111
//    00000000
//    00000000
//    00000000
//    11111111
//    00000000
//    00000000
//    00000000
//
// A loopfilter should be applied to every other 4 the row vertically.
static const uint64_t above_64x64_txform_mask[TX_SIZES]= {
  0xffffffffffffffffULL,  // TX_4X4
  0xffffffffffffffffULL,  // TX_8x8
  0x00ff00ff00ff00ffULL,  // TX_16x16
  0x000000ff000000ffULL,  // TX_32x32
};

// 64 bit masks for prediction sizes (left). Each 1 represents a position
// where left border of an 8x8 block. These are aligned to the right most
// appropriate bit, and then shifted into place.
//
// In the case of TX_16x32 ->  ( low order byte first ) we end up with
// a mask that looks like this :
//
//  10000000
//  10000000
//  10000000
//  10000000
//  00000000
//  00000000
//  00000000
//  00000000
static const uint64_t left_prediction_mask[BLOCK_SIZES] = {
  0x0000000000000001ULL,  // BLOCK_4X4,
  0x0000000000000001ULL,  // BLOCK_4X8,
  0x0000000000000001ULL,  // BLOCK_8X4,
  0x0000000000000001ULL,  // BLOCK_8X8,
  0x0000000000000101ULL,  // BLOCK_8X16,
  0x0000000000000001ULL,  // BLOCK_16X8,
  0x0000000000000101ULL,  // BLOCK_16X16,
  0x0000000001010101ULL,  // BLOCK_16X32,
  0x0000000000000101ULL,  // BLOCK_32X16,
  0x0000000001010101ULL,  // BLOCK_32X32,
  0x0101010101010101ULL,  // BLOCK_32X64,
  0x0000000001010101ULL,  // BLOCK_64X32,
  0x0101010101010101ULL,  // BLOCK_64X64
};

// 64 bit mask to shift and set for each prediction size.
static const uint64_t above_prediction_mask[BLOCK_SIZES] = {
  0x0000000000000001ULL,  // BLOCK_4X4
  0x0000000000000001ULL,  // BLOCK_4X8
  0x0000000000000001ULL,  // BLOCK_8X4
  0x0000000000000001ULL,  // BLOCK_8X8
  0x0000000000000001ULL,  // BLOCK_8X16,
  0x0000000000000003ULL,  // BLOCK_16X8
  0x0000000000000003ULL,  // BLOCK_16X16
  0x0000000000000003ULL,  // BLOCK_16X32,
  0x000000000000000fULL,  // BLOCK_32X16,
  0x000000000000000fULL,  // BLOCK_32X32,
  0x000000000000000fULL,  // BLOCK_32X64,
  0x00000000000000ffULL,  // BLOCK_64X32,
  0x00000000000000ffULL,  // BLOCK_64X64
};
// 64 bit mask to shift and set for each prediction size. A bit is set for
// each 8x8 block that would be in the left most block of the given block
// size in the 64x64 block.
static const uint64_t size_mask[BLOCK_SIZES] = {
  0x0000000000000001ULL,  // BLOCK_4X4
  0x0000000000000001ULL,  // BLOCK_4X8
  0x0000000000000001ULL,  // BLOCK_8X4
  0x0000000000000001ULL,  // BLOCK_8X8
  0x0000000000000101ULL,  // BLOCK_8X16,
  0x0000000000000003ULL,  // BLOCK_16X8
  0x0000000000000303ULL,  // BLOCK_16X16
  0x0000000003030303ULL,  // BLOCK_16X32,
  0x0000000000000f0fULL,  // BLOCK_32X16,
  0x000000000f0f0f0fULL,  // BLOCK_32X32,
  0x0f0f0f0f0f0f0f0fULL,  // BLOCK_32X64,
  0x00000000ffffffffULL,  // BLOCK_64X32,
  0xffffffffffffffffULL,  // BLOCK_64X64
};

// These are used for masking the left and above borders.
static const uint64_t left_border =  0x1111111111111111ULL;
static const uint64_t above_border = 0x000000ff000000ffULL;

// 16 bit masks for uv transform sizes.
static const uint16_t left_64x64_txform_mask_uv[TX_SIZES]= {
  0xffff,  // TX_4X4
  0xffff,  // TX_8x8
  0x5555,  // TX_16x16
  0x1111,  // TX_32x32
};

static const uint16_t above_64x64_txform_mask_uv[TX_SIZES]= {
  0xffff,  // TX_4X4
  0xffff,  // TX_8x8
  0x0f0f,  // TX_16x16
  0x000f,  // TX_32x32
};

// 16 bit left mask to shift and set for each uv prediction size.
static const uint16_t left_prediction_mask_uv[BLOCK_SIZES] = {
  0x0001,  // BLOCK_4X4,
  0x0001,  // BLOCK_4X8,
  0x0001,  // BLOCK_8X4,
  0x0001,  // BLOCK_8X8,
  0x0001,  // BLOCK_8X16,
  0x0001,  // BLOCK_16X8,
  0x0001,  // BLOCK_16X16,
  0x0011,  // BLOCK_16X32,
  0x0001,  // BLOCK_32X16,
  0x0011,  // BLOCK_32X32,
  0x1111,  // BLOCK_32X64
  0x0011,  // BLOCK_64X32,
  0x1111,  // BLOCK_64X64
};
// 16 bit above mask to shift and set for uv each prediction size.
static const uint16_t above_prediction_mask_uv[BLOCK_SIZES] = {
  0x0001,  // BLOCK_4X4
  0x0001,  // BLOCK_4X8
  0x0001,  // BLOCK_8X4
  0x0001,  // BLOCK_8X8
  0x0001,  // BLOCK_8X16,
  0x0001,  // BLOCK_16X8
  0x0001,  // BLOCK_16X16
  0x0001,  // BLOCK_16X32,
  0x0003,  // BLOCK_32X16,
  0x0003,  // BLOCK_32X32,
  0x0003,  // BLOCK_32X64,
  0x000f,  // BLOCK_64X32,
  0x000f,  // BLOCK_64X64
};

// 64 bit mask to shift and set for each uv prediction size
static const uint16_t size_mask_uv[BLOCK_SIZES] = {
  0x0001,  // BLOCK_4X4
  0x0001,  // BLOCK_4X8
  0x0001,  // BLOCK_8X4
  0x0001,  // BLOCK_8X8
  0x0001,  // BLOCK_8X16,
  0x0001,  // BLOCK_16X8
  0x0001,  // BLOCK_16X16
  0x0011,  // BLOCK_16X32,
  0x0003,  // BLOCK_32X16,
  0x0033,  // BLOCK_32X32,
  0x3333,  // BLOCK_32X64,
  0x00ff,  // BLOCK_64X32,
  0xffff,  // BLOCK_64X64
};
static const uint16_t left_border_uv =  0x1111;
static const uint16_t above_border_uv = 0x000f;

static const int mode_lf_lut[MB_MODE_COUNT] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // INTRA_MODES
  1, 1, 0, 1                     // INTER_MODES (ZEROMV == 0)
};

static void update_sharpness(loop_filter_info_n *lfi, int sharpness_lvl) {
  int lvl;

  // For each possible value for the loop filter fill out limits
  for (lvl = 0; lvl <= MAX_LOOP_FILTER; lvl++) {
    // Set loop filter parameters that control sharpness.
    int block_inside_limit = lvl >> ((sharpness_lvl > 0) + (sharpness_lvl > 4));

    if (sharpness_lvl > 0) {
      if (block_inside_limit > (9 - sharpness_lvl))
        block_inside_limit = (9 - sharpness_lvl);
    }

    if (block_inside_limit < 1)
      block_inside_limit = 1;

    memset(lfi->lfthr[lvl].lim, block_inside_limit, SIMD_WIDTH);
    memset(lfi->lfthr[lvl].mblim, (2 * (lvl + 2) + block_inside_limit),
           SIMD_WIDTH);
  }
}

static uint8_t get_filter_level(const loop_filter_info_n *lfi_n,
                                const MODE_INFO *mi) {
  return lfi_n->lvl[mi->segment_id][mi->ref_frame[0]]
                   [mode_lf_lut[mi->mode]];
}

void vp9_loop_filter_init(VP9_COMMON *cm) {
  loop_filter_info_n *lfi = &cm->lf_info;
  struct loopfilter *lf = &cm->lf;
  int lvl;

  // init limits for given sharpness
  update_sharpness(lfi, lf->sharpness_level);
  lf->last_sharpness_level = lf->sharpness_level;

  // init hev threshold const vectors
  for (lvl = 0; lvl <= MAX_LOOP_FILTER; lvl++)
    memset(lfi->lfthr[lvl].hev_thr, (lvl >> 4), SIMD_WIDTH);
}

void vp9_loop_filter_frame_init(VP9_COMMON *cm, int default_filt_lvl) {
  int seg_id;
  // n_shift is the multiplier for lf_deltas
  // the multiplier is 1 for when filter_lvl is between 0 and 31;
  // 2 when filter_lvl is between 32 and 63
  const int scale = 1 << (default_filt_lvl >> 5);
  loop_filter_info_n *const lfi = &cm->lf_info;
  struct loopfilter *const lf = &cm->lf;
  const struct segmentation *const seg = &cm->seg;

  // update limits if sharpness has changed
  if (lf->last_sharpness_level != lf->sharpness_level) {
    update_sharpness(lfi, lf->sharpness_level);
    lf->last_sharpness_level = lf->sharpness_level;
  }

  for (seg_id = 0; seg_id < MAX_SEGMENTS; seg_id++) {
    int lvl_seg = default_filt_lvl;
    if (segfeature_active(seg, seg_id, SEG_LVL_ALT_LF)) {
      const int data = get_segdata(seg, seg_id, SEG_LVL_ALT_LF);
      lvl_seg = clamp(seg->abs_delta == SEGMENT_ABSDATA ?
                      data : default_filt_lvl + data,
                      0, MAX_LOOP_FILTER);
    }

    if (!lf->mode_ref_delta_enabled) {
      // we could get rid of this if we assume that deltas are set to
      // zero when not in use; encoder always uses deltas
      memset(lfi->lvl[seg_id], lvl_seg, sizeof(lfi->lvl[seg_id]));
    } else {
      int ref, mode;
      const int intra_lvl = lvl_seg + lf->ref_deltas[INTRA_FRAME] * scale;
      lfi->lvl[seg_id][INTRA_FRAME][0] = clamp(intra_lvl, 0, MAX_LOOP_FILTER);

      for (ref = LAST_FRAME; ref < MAX_REF_FRAMES; ++ref) {
        for (mode = 0; mode < MAX_MODE_LF_DELTAS; ++mode) {
          const int inter_lvl = lvl_seg + lf->ref_deltas[ref] * scale
                                        + lf->mode_deltas[mode] * scale;
          lfi->lvl[seg_id][ref][mode] = clamp(inter_lvl, 0, MAX_LOOP_FILTER);
        }
      }
    }
  }
}

static void filter_selectively_vert_row2(int subsampling_factor,
                                         uint8_t *s, int pitch,
                                         unsigned int mask_16x16,
                                         unsigned int mask_8x8,
                                         unsigned int mask_4x4,
                                         unsigned int mask_4x4_int,
                                         const loop_filter_thresh *lfthr,
                                         const uint8_t *lfl) {
  const int dual_mask_cutoff = subsampling_factor ? 0xff : 0xffff;
  const int lfl_forward = subsampling_factor ? 4 : 8;
  const unsigned int dual_one = 1 | (1 << lfl_forward);
  unsigned int mask;
  uint8_t *ss[2];
  ss[0] = s;

  for (mask =
           (mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int) & dual_mask_cutoff;
       mask; mask = (mask & ~dual_one) >> 1) {
    if (mask & dual_one) {
      const loop_filter_thresh *lfis[2];
      lfis[0] = lfthr + *lfl;
      lfis[1] = lfthr + *(lfl + lfl_forward);
      ss[1] = ss[0] + 8 * pitch;

      if (mask_16x16 & dual_one) {
        if ((mask_16x16 & dual_one) == dual_one) {
          vpx_lpf_vertical_16_dual(ss[0], pitch, lfis[0]->mblim, lfis[0]->lim,
                                   lfis[0]->hev_thr);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_16x16 & 1)];
          vpx_lpf_vertical_16(ss[!(mask_16x16 & 1)], pitch, lfi->mblim,
                              lfi->lim, lfi->hev_thr);
        }
      }

      if (mask_8x8 & dual_one) {
        if ((mask_8x8 & dual_one) == dual_one) {
          vpx_lpf_vertical_8_dual(ss[0], pitch, lfis[0]->mblim, lfis[0]->lim,
                                  lfis[0]->hev_thr, lfis[1]->mblim,
                                  lfis[1]->lim, lfis[1]->hev_thr);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_8x8 & 1)];
          vpx_lpf_vertical_8(ss[!(mask_8x8 & 1)], pitch, lfi->mblim, lfi->lim,
                             lfi->hev_thr);
        }
      }

      if (mask_4x4 & dual_one) {
        if ((mask_4x4 & dual_one) == dual_one) {
          vpx_lpf_vertical_4_dual(ss[0], pitch, lfis[0]->mblim, lfis[0]->lim,
                                  lfis[0]->hev_thr, lfis[1]->mblim,
                                  lfis[1]->lim, lfis[1]->hev_thr);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_4x4 & 1)];
          vpx_lpf_vertical_4(ss[!(mask_4x4 & 1)], pitch, lfi->mblim, lfi->lim,
                             lfi->hev_thr);
        }
      }

      if (mask_4x4_int & dual_one) {
        if ((mask_4x4_int & dual_one) == dual_one) {
          vpx_lpf_vertical_4_dual(ss[0] + 4, pitch, lfis[0]->mblim,
                                  lfis[0]->lim, lfis[0]->hev_thr,
                                  lfis[1]->mblim, lfis[1]->lim,
                                  lfis[1]->hev_thr);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_4x4_int & 1)];
          vpx_lpf_vertical_4(ss[!(mask_4x4_int & 1)] + 4, pitch, lfi->mblim,
                             lfi->lim, lfi->hev_thr);
        }
      }
    }

    ss[0] += 8;
    lfl += 1;
    mask_16x16 >>= 1;
    mask_8x8 >>= 1;
    mask_4x4 >>= 1;
    mask_4x4_int >>= 1;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_filter_selectively_vert_row2(int subsampling_factor,
                                                uint16_t *s, int pitch,
                                                unsigned int mask_16x16,
                                                unsigned int mask_8x8,
                                                unsigned int mask_4x4,
                                                unsigned int mask_4x4_int,
                                                const loop_filter_thresh *lfthr,
                                                const uint8_t *lfl, int bd) {
  const int dual_mask_cutoff = subsampling_factor ? 0xff : 0xffff;
  const int lfl_forward = subsampling_factor ? 4 : 8;
  const unsigned int dual_one = 1 | (1 << lfl_forward);
  unsigned int mask;
  uint16_t *ss[2];
  ss[0] = s;

  for (mask =
           (mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int) & dual_mask_cutoff;
       mask; mask = (mask & ~dual_one) >> 1) {
    if (mask & dual_one) {
      const loop_filter_thresh *lfis[2];
      lfis[0] = lfthr + *lfl;
      lfis[1] = lfthr + *(lfl + lfl_forward);
      ss[1] = ss[0] + 8 * pitch;

      if (mask_16x16 & dual_one) {
        if ((mask_16x16 & dual_one) == dual_one) {
          vpx_highbd_lpf_vertical_16_dual(ss[0], pitch, lfis[0]->mblim,
                                          lfis[0]->lim, lfis[0]->hev_thr, bd);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_16x16 & 1)];
          vpx_highbd_lpf_vertical_16(ss[!(mask_16x16 & 1)], pitch, lfi->mblim,
                                     lfi->lim, lfi->hev_thr, bd);
        }
      }

      if (mask_8x8 & dual_one) {
        if ((mask_8x8 & dual_one) == dual_one) {
          vpx_highbd_lpf_vertical_8_dual(ss[0], pitch, lfis[0]->mblim,
                                         lfis[0]->lim, lfis[0]->hev_thr,
                                         lfis[1]->mblim, lfis[1]->lim,
                                         lfis[1]->hev_thr, bd);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_8x8 & 1)];
          vpx_highbd_lpf_vertical_8(ss[!(mask_8x8 & 1)], pitch, lfi->mblim,
                                    lfi->lim, lfi->hev_thr, bd);
        }
      }

      if (mask_4x4 & dual_one) {
        if ((mask_4x4 & dual_one) == dual_one) {
          vpx_highbd_lpf_vertical_4_dual(ss[0], pitch, lfis[0]->mblim,
                                         lfis[0]->lim, lfis[0]->hev_thr,
                                         lfis[1]->mblim, lfis[1]->lim,
                                         lfis[1]->hev_thr, bd);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_4x4 & 1)];
          vpx_highbd_lpf_vertical_4(ss[!(mask_4x4 & 1)], pitch, lfi->mblim,
                                    lfi->lim, lfi->hev_thr, bd);
        }
      }

      if (mask_4x4_int & dual_one) {
        if ((mask_4x4_int & dual_one) == dual_one) {
          vpx_highbd_lpf_vertical_4_dual(ss[0] + 4, pitch, lfis[0]->mblim,
                                         lfis[0]->lim, lfis[0]->hev_thr,
                                         lfis[1]->mblim, lfis[1]->lim,
                                         lfis[1]->hev_thr, bd);
        } else {
          const loop_filter_thresh *lfi = lfis[!(mask_4x4_int & 1)];
          vpx_highbd_lpf_vertical_4(ss[!(mask_4x4_int & 1)] + 4, pitch,
                                    lfi->mblim, lfi->lim, lfi->hev_thr, bd);
        }
      }
    }

    ss[0] += 8;
    lfl += 1;
    mask_16x16 >>= 1;
    mask_8x8 >>= 1;
    mask_4x4 >>= 1;
    mask_4x4_int >>= 1;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void filter_selectively_horiz(uint8_t *s, int pitch,
                                     unsigned int mask_16x16,
                                     unsigned int mask_8x8,
                                     unsigned int mask_4x4,
                                     unsigned int mask_4x4_int,
                                     const loop_filter_thresh *lfthr,
                                     const uint8_t *lfl) {
  unsigned int mask;
  int count;

  for (mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
       mask; mask >>= count) {
    count = 1;
    if (mask & 1) {
      const loop_filter_thresh *lfi = lfthr + *lfl;

      if (mask_16x16 & 1) {
        if ((mask_16x16 & 3) == 3) {
          vpx_lpf_horizontal_edge_16(s, pitch, lfi->mblim, lfi->lim,
                                     lfi->hev_thr);
          count = 2;
        } else {
          vpx_lpf_horizontal_edge_8(s, pitch, lfi->mblim, lfi->lim,
                                    lfi->hev_thr);
        }
      } else if (mask_8x8 & 1) {
        if ((mask_8x8 & 3) == 3) {
          // Next block's thresholds.
          const loop_filter_thresh *lfin = lfthr + *(lfl + 1);

          vpx_lpf_horizontal_8_dual(s, pitch, lfi->mblim, lfi->lim,
                                    lfi->hev_thr, lfin->mblim, lfin->lim,
                                    lfin->hev_thr);

          if ((mask_4x4_int & 3) == 3) {
            vpx_lpf_horizontal_4_dual(s + 4 * pitch, pitch, lfi->mblim,
                                      lfi->lim, lfi->hev_thr, lfin->mblim,
                                      lfin->lim, lfin->hev_thr);
          } else {
            if (mask_4x4_int & 1)
              vpx_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                                   lfi->hev_thr);
            else if (mask_4x4_int & 2)
              vpx_lpf_horizontal_4(s + 8 + 4 * pitch, pitch, lfin->mblim,
                                   lfin->lim, lfin->hev_thr);
          }
          count = 2;
        } else {
          vpx_lpf_horizontal_8(s, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);

          if (mask_4x4_int & 1)
            vpx_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                                 lfi->hev_thr);
        }
      } else if (mask_4x4 & 1) {
        if ((mask_4x4 & 3) == 3) {
          // Next block's thresholds.
          const loop_filter_thresh *lfin = lfthr + *(lfl + 1);

          vpx_lpf_horizontal_4_dual(s, pitch, lfi->mblim, lfi->lim,
                                    lfi->hev_thr, lfin->mblim, lfin->lim,
                                    lfin->hev_thr);
          if ((mask_4x4_int & 3) == 3) {
            vpx_lpf_horizontal_4_dual(s + 4 * pitch, pitch, lfi->mblim,
                                      lfi->lim, lfi->hev_thr, lfin->mblim,
                                      lfin->lim, lfin->hev_thr);
          } else {
            if (mask_4x4_int & 1)
              vpx_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                                   lfi->hev_thr);
            else if (mask_4x4_int & 2)
              vpx_lpf_horizontal_4(s + 8 + 4 * pitch, pitch, lfin->mblim,
                                   lfin->lim, lfin->hev_thr);
          }
          count = 2;
        } else {
          vpx_lpf_horizontal_4(s, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);

          if (mask_4x4_int & 1)
            vpx_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                                 lfi->hev_thr);
        }
      } else {
        vpx_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                             lfi->hev_thr);
      }
    }
    s += 8 * count;
    lfl += count;
    mask_16x16 >>= count;
    mask_8x8 >>= count;
    mask_4x4 >>= count;
    mask_4x4_int >>= count;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_filter_selectively_horiz(uint16_t *s, int pitch,
                                            unsigned int mask_16x16,
                                            unsigned int mask_8x8,
                                            unsigned int mask_4x4,
                                            unsigned int mask_4x4_int,
                                            const loop_filter_thresh *lfthr,
                                            const uint8_t *lfl, int bd) {
  unsigned int mask;
  int count;

  for (mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
       mask; mask >>= count) {
    count = 1;
    if (mask & 1) {
      const loop_filter_thresh *lfi = lfthr + *lfl;

      if (mask_16x16 & 1) {
        if ((mask_16x16 & 3) == 3) {
          vpx_highbd_lpf_horizontal_edge_16(s, pitch, lfi->mblim, lfi->lim,
                                            lfi->hev_thr, bd);
          count = 2;
        } else {
          vpx_highbd_lpf_horizontal_edge_8(s, pitch, lfi->mblim, lfi->lim,
                                           lfi->hev_thr, bd);
        }
      } else if (mask_8x8 & 1) {
        if ((mask_8x8 & 3) == 3) {
          // Next block's thresholds.
          const loop_filter_thresh *lfin = lfthr + *(lfl + 1);

          vpx_highbd_lpf_horizontal_8_dual(s, pitch, lfi->mblim, lfi->lim,
                                           lfi->hev_thr, lfin->mblim, lfin->lim,
                                           lfin->hev_thr, bd);

          if ((mask_4x4_int & 3) == 3) {
            vpx_highbd_lpf_horizontal_4_dual(s + 4 * pitch, pitch, lfi->mblim,
                                             lfi->lim, lfi->hev_thr,
                                             lfin->mblim, lfin->lim,
                                             lfin->hev_thr, bd);
          } else {
            if (mask_4x4_int & 1) {
              vpx_highbd_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim,
                                          lfi->lim, lfi->hev_thr, bd);
            } else if (mask_4x4_int & 2) {
              vpx_highbd_lpf_horizontal_4(s + 8 + 4 * pitch, pitch, lfin->mblim,
                                          lfin->lim, lfin->hev_thr, bd);
            }
          }
          count = 2;
        } else {
          vpx_highbd_lpf_horizontal_8(s, pitch, lfi->mblim, lfi->lim,
                                      lfi->hev_thr, bd);

          if (mask_4x4_int & 1) {
            vpx_highbd_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim,
                                        lfi->lim, lfi->hev_thr, bd);
          }
        }
      } else if (mask_4x4 & 1) {
        if ((mask_4x4 & 3) == 3) {
          // Next block's thresholds.
          const loop_filter_thresh *lfin = lfthr + *(lfl + 1);

          vpx_highbd_lpf_horizontal_4_dual(s, pitch, lfi->mblim, lfi->lim,
                                           lfi->hev_thr, lfin->mblim, lfin->lim,
                                           lfin->hev_thr, bd);
          if ((mask_4x4_int & 3) == 3) {
            vpx_highbd_lpf_horizontal_4_dual(s + 4 * pitch, pitch, lfi->mblim,
                                             lfi->lim, lfi->hev_thr,
                                             lfin->mblim, lfin->lim,
                                             lfin->hev_thr, bd);
          } else {
            if (mask_4x4_int & 1) {
              vpx_highbd_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim,
                                          lfi->lim, lfi->hev_thr, bd);
            } else if (mask_4x4_int & 2) {
              vpx_highbd_lpf_horizontal_4(s + 8 + 4 * pitch, pitch, lfin->mblim,
                                          lfin->lim, lfin->hev_thr, bd);
            }
          }
          count = 2;
        } else {
          vpx_highbd_lpf_horizontal_4(s, pitch, lfi->mblim, lfi->lim,
                                      lfi->hev_thr, bd);

          if (mask_4x4_int & 1) {
            vpx_highbd_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim,
                                        lfi->lim, lfi->hev_thr, bd);
          }
        }
      } else {
        vpx_highbd_lpf_horizontal_4(s + 4 * pitch, pitch, lfi->mblim, lfi->lim,
                                    lfi->hev_thr, bd);
      }
    }
    s += 8 * count;
    lfl += count;
    mask_16x16 >>= count;
    mask_8x8 >>= count;
    mask_4x4 >>= count;
    mask_4x4_int >>= count;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

// This function ors into the current lfm structure, where to do loop
// filters for the specific mi we are looking at. It uses information
// including the block_size_type (32x16, 32x32, etc.), the transform size,
// whether there were any coefficients encoded, and the loop filter strength
// block we are currently looking at. Shift is used to position the
// 1's we produce.
static void build_masks(const loop_filter_info_n *const lfi_n,
                        const MODE_INFO *mi, const int shift_y,
                        const int shift_uv,
                        LOOP_FILTER_MASK *lfm) {
  const BLOCK_SIZE block_size = mi->sb_type;
  const TX_SIZE tx_size_y = mi->tx_size;
  const TX_SIZE tx_size_uv = get_uv_tx_size_impl(tx_size_y, block_size, 1, 1);
  const int filter_level = get_filter_level(lfi_n, mi);
  uint64_t *const left_y = &lfm->left_y[tx_size_y];
  uint64_t *const above_y = &lfm->above_y[tx_size_y];
  uint64_t *const int_4x4_y = &lfm->int_4x4_y;
  uint16_t *const left_uv = &lfm->left_uv[tx_size_uv];
  uint16_t *const above_uv = &lfm->above_uv[tx_size_uv];
  uint16_t *const int_4x4_uv = &lfm->int_4x4_uv;
  int i;

  // If filter level is 0 we don't loop filter.
  if (!filter_level) {
    return;
  } else {
    const int w = num_8x8_blocks_wide_lookup[block_size];
    const int h = num_8x8_blocks_high_lookup[block_size];
    int index = shift_y;
    for (i = 0; i < h; i++) {
      memset(&lfm->lfl_y[index], filter_level, w);
      index += 8;
    }
  }

  // These set 1 in the current block size for the block size edges.
  // For instance if the block size is 32x16, we'll set:
  //    above =   1111
  //              0000
  //    and
  //    left  =   1000
  //          =   1000
  // NOTE : In this example the low bit is left most ( 1000 ) is stored as
  //        1,  not 8...
  //
  // U and V set things on a 16 bit scale.
  //
  *above_y |= above_prediction_mask[block_size] << shift_y;
  *above_uv |= above_prediction_mask_uv[block_size] << shift_uv;
  *left_y |= left_prediction_mask[block_size] << shift_y;
  *left_uv |= left_prediction_mask_uv[block_size] << shift_uv;

  // If the block has no coefficients and is not intra we skip applying
  // the loop filter on block edges.
  if (mi->skip && is_inter_block(mi))
    return;

  // Here we are adding a mask for the transform size. The transform
  // size mask is set to be correct for a 64x64 prediction block size. We
  // mask to match the size of the block we are working on and then shift it
  // into place..
  *above_y |= (size_mask[block_size] &
               above_64x64_txform_mask[tx_size_y]) << shift_y;
  *above_uv |= (size_mask_uv[block_size] &
                above_64x64_txform_mask_uv[tx_size_uv]) << shift_uv;

  *left_y |= (size_mask[block_size] &
              left_64x64_txform_mask[tx_size_y]) << shift_y;
  *left_uv |= (size_mask_uv[block_size] &
               left_64x64_txform_mask_uv[tx_size_uv]) << shift_uv;

  // Here we are trying to determine what to do with the internal 4x4 block
  // boundaries.  These differ from the 4x4 boundaries on the outside edge of
  // an 8x8 in that the internal ones can be skipped and don't depend on
  // the prediction block size.
  if (tx_size_y == TX_4X4)
    *int_4x4_y |= size_mask[block_size] << shift_y;

  if (tx_size_uv == TX_4X4)
    *int_4x4_uv |= (size_mask_uv[block_size] & 0xffff) << shift_uv;
}

// This function does the same thing as the one above with the exception that
// it only affects the y masks. It exists because for blocks < 16x16 in size,
// we only update u and v masks on the first block.
static void build_y_mask(const loop_filter_info_n *const lfi_n,
                         const MODE_INFO *mi, const int shift_y,
                         LOOP_FILTER_MASK *lfm) {
  const BLOCK_SIZE block_size = mi->sb_type;
  const TX_SIZE tx_size_y = mi->tx_size;
  const int filter_level = get_filter_level(lfi_n, mi);
  uint64_t *const left_y = &lfm->left_y[tx_size_y];
  uint64_t *const above_y = &lfm->above_y[tx_size_y];
  uint64_t *const int_4x4_y = &lfm->int_4x4_y;
  int i;

  if (!filter_level) {
    return;
  } else {
    const int w = num_8x8_blocks_wide_lookup[block_size];
    const int h = num_8x8_blocks_high_lookup[block_size];
    int index = shift_y;
    for (i = 0; i < h; i++) {
      memset(&lfm->lfl_y[index], filter_level, w);
      index += 8;
    }
  }

  *above_y |= above_prediction_mask[block_size] << shift_y;
  *left_y |= left_prediction_mask[block_size] << shift_y;

  if (mi->skip && is_inter_block(mi))
    return;

  *above_y |= (size_mask[block_size] &
               above_64x64_txform_mask[tx_size_y]) << shift_y;

  *left_y |= (size_mask[block_size] &
              left_64x64_txform_mask[tx_size_y]) << shift_y;

  if (tx_size_y == TX_4X4)
    *int_4x4_y |= size_mask[block_size] << shift_y;
}

void vp9_adjust_mask(VP9_COMMON *const cm, const int mi_row,
                     const int mi_col, LOOP_FILTER_MASK *lfm) {
  int i;

  // The largest loopfilter we have is 16x16 so we use the 16x16 mask
  // for 32x32 transforms also.
  lfm->left_y[TX_16X16] |= lfm->left_y[TX_32X32];
  lfm->above_y[TX_16X16] |= lfm->above_y[TX_32X32];
  lfm->left_uv[TX_16X16] |= lfm->left_uv[TX_32X32];
  lfm->above_uv[TX_16X16] |= lfm->above_uv[TX_32X32];

  // We do at least 8 tap filter on every 32x32 even if the transform size
  // is 4x4. So if the 4x4 is set on a border pixel add it to the 8x8 and
  // remove it from the 4x4.
  lfm->left_y[TX_8X8] |= lfm->left_y[TX_4X4] & left_border;
  lfm->left_y[TX_4X4] &= ~left_border;
  lfm->above_y[TX_8X8] |= lfm->above_y[TX_4X4] & above_border;
  lfm->above_y[TX_4X4] &= ~above_border;
  lfm->left_uv[TX_8X8] |= lfm->left_uv[TX_4X4] & left_border_uv;
  lfm->left_uv[TX_4X4] &= ~left_border_uv;
  lfm->above_uv[TX_8X8] |= lfm->above_uv[TX_4X4] & above_border_uv;
  lfm->above_uv[TX_4X4] &= ~above_border_uv;

  // We do some special edge handling.
  if (mi_row + MI_BLOCK_SIZE > cm->mi_rows) {
    const uint64_t rows = cm->mi_rows - mi_row;

    // Each pixel inside the border gets a 1,
    const uint64_t mask_y = (((uint64_t) 1 << (rows << 3)) - 1);
    const uint16_t mask_uv = (((uint16_t) 1 << (((rows + 1) >> 1) << 2)) - 1);

    // Remove values completely outside our border.
    for (i = 0; i < TX_32X32; i++) {
      lfm->left_y[i] &= mask_y;
      lfm->above_y[i] &= mask_y;
      lfm->left_uv[i] &= mask_uv;
      lfm->above_uv[i] &= mask_uv;
    }
    lfm->int_4x4_y &= mask_y;
    lfm->int_4x4_uv &= mask_uv;

    // We don't apply a wide loop filter on the last uv block row. If set
    // apply the shorter one instead.
    if (rows == 1) {
      lfm->above_uv[TX_8X8] |= lfm->above_uv[TX_16X16];
      lfm->above_uv[TX_16X16] = 0;
    }
    if (rows == 5) {
      lfm->above_uv[TX_8X8] |= lfm->above_uv[TX_16X16] & 0xff00;
      lfm->above_uv[TX_16X16] &= ~(lfm->above_uv[TX_16X16] & 0xff00);
    }
  }

  if (mi_col + MI_BLOCK_SIZE > cm->mi_cols) {
    const uint64_t columns = cm->mi_cols - mi_col;

    // Each pixel inside the border gets a 1, the multiply copies the border
    // to where we need it.
    const uint64_t mask_y  = (((1 << columns) - 1)) * 0x0101010101010101ULL;
    const uint16_t mask_uv = ((1 << ((columns + 1) >> 1)) - 1) * 0x1111;

    // Internal edges are not applied on the last column of the image so
    // we mask 1 more for the internal edges
    const uint16_t mask_uv_int = ((1 << (columns >> 1)) - 1) * 0x1111;

    // Remove the bits outside the image edge.
    for (i = 0; i < TX_32X32; i++) {
      lfm->left_y[i] &= mask_y;
      lfm->above_y[i] &= mask_y;
      lfm->left_uv[i] &= mask_uv;
      lfm->above_uv[i] &= mask_uv;
    }
    lfm->int_4x4_y &= mask_y;
    lfm->int_4x4_uv &= mask_uv_int;

    // We don't apply a wide loop filter on the last uv column. If set
    // apply the shorter one instead.
    if (columns == 1) {
      lfm->left_uv[TX_8X8] |= lfm->left_uv[TX_16X16];
      lfm->left_uv[TX_16X16] = 0;
    }
    if (columns == 5) {
      lfm->left_uv[TX_8X8] |= (lfm->left_uv[TX_16X16] & 0xcccc);
      lfm->left_uv[TX_16X16] &= ~(lfm->left_uv[TX_16X16] & 0xcccc);
    }
  }
  // We don't apply a loop filter on the first column in the image, mask that
  // out.
  if (mi_col == 0) {
    for (i = 0; i < TX_32X32; i++) {
      lfm->left_y[i] &= 0xfefefefefefefefeULL;
      lfm->left_uv[i] &= 0xeeee;
    }
  }

  // Assert if we try to apply 2 different loop filters at the same position.
  assert(!(lfm->left_y[TX_16X16] & lfm->left_y[TX_8X8]));
  assert(!(lfm->left_y[TX_16X16] & lfm->left_y[TX_4X4]));
  assert(!(lfm->left_y[TX_8X8] & lfm->left_y[TX_4X4]));
  assert(!(lfm->int_4x4_y & lfm->left_y[TX_16X16]));
  assert(!(lfm->left_uv[TX_16X16]&lfm->left_uv[TX_8X8]));
  assert(!(lfm->left_uv[TX_16X16] & lfm->left_uv[TX_4X4]));
  assert(!(lfm->left_uv[TX_8X8] & lfm->left_uv[TX_4X4]));
  assert(!(lfm->int_4x4_uv & lfm->left_uv[TX_16X16]));
  assert(!(lfm->above_y[TX_16X16] & lfm->above_y[TX_8X8]));
  assert(!(lfm->above_y[TX_16X16] & lfm->above_y[TX_4X4]));
  assert(!(lfm->above_y[TX_8X8] & lfm->above_y[TX_4X4]));
  assert(!(lfm->int_4x4_y & lfm->above_y[TX_16X16]));
  assert(!(lfm->above_uv[TX_16X16] & lfm->above_uv[TX_8X8]));
  assert(!(lfm->above_uv[TX_16X16] & lfm->above_uv[TX_4X4]));
  assert(!(lfm->above_uv[TX_8X8] & lfm->above_uv[TX_4X4]));
  assert(!(lfm->int_4x4_uv & lfm->above_uv[TX_16X16]));
}

// This function sets up the bit masks for the entire 64x64 region represented
// by mi_row, mi_col.
void vp9_setup_mask(VP9_COMMON *const cm, const int mi_row, const int mi_col,
                    MODE_INFO **mi, const int mode_info_stride,
                    LOOP_FILTER_MASK *lfm) {
  int idx_32, idx_16, idx_8;
  const loop_filter_info_n *const lfi_n = &cm->lf_info;
  MODE_INFO **mip = mi;
  MODE_INFO **mip2 = mi;

  // These are offsets to the next mi in the 64x64 block. It is what gets
  // added to the mi ptr as we go through each loop. It helps us to avoid
  // setting up special row and column counters for each index. The last step
  // brings us out back to the starting position.
  const int offset_32[] = {4, (mode_info_stride << 2) - 4, 4,
                           -(mode_info_stride << 2) - 4};
  const int offset_16[] = {2, (mode_info_stride << 1) - 2, 2,
                           -(mode_info_stride << 1) - 2};
  const int offset[] = {1, mode_info_stride - 1, 1, -mode_info_stride - 1};

  // Following variables represent shifts to position the current block
  // mask over the appropriate block. A shift of 36 to the left will move
  // the bits for the final 32 by 32 block in the 64x64 up 4 rows and left
  // 4 rows to the appropriate spot.
  const int shift_32_y[] = {0, 4, 32, 36};
  const int shift_16_y[] = {0, 2, 16, 18};
  const int shift_8_y[] = {0, 1, 8, 9};
  const int shift_32_uv[] = {0, 2, 8, 10};
  const int shift_16_uv[] = {0, 1, 4, 5};
  const int max_rows = (mi_row + MI_BLOCK_SIZE > cm->mi_rows ?
                        cm->mi_rows - mi_row : MI_BLOCK_SIZE);
  const int max_cols = (mi_col + MI_BLOCK_SIZE > cm->mi_cols ?
                        cm->mi_cols - mi_col : MI_BLOCK_SIZE);

  vp9_zero(*lfm);
  assert(mip[0] != NULL);

  switch (mip[0]->sb_type) {
    case BLOCK_64X64:
      build_masks(lfi_n, mip[0] , 0, 0, lfm);
      break;
    case BLOCK_64X32:
      build_masks(lfi_n, mip[0], 0, 0, lfm);
      mip2 = mip + mode_info_stride * 4;
      if (4 >= max_rows)
        break;
      build_masks(lfi_n, mip2[0], 32, 8, lfm);
      break;
    case BLOCK_32X64:
      build_masks(lfi_n, mip[0], 0, 0, lfm);
      mip2 = mip + 4;
      if (4 >= max_cols)
        break;
      build_masks(lfi_n, mip2[0], 4, 2, lfm);
      break;
    default:
      for (idx_32 = 0; idx_32 < 4; mip += offset_32[idx_32], ++idx_32) {
        const int shift_y = shift_32_y[idx_32];
        const int shift_uv = shift_32_uv[idx_32];
        const int mi_32_col_offset = ((idx_32 & 1) << 2);
        const int mi_32_row_offset = ((idx_32 >> 1) << 2);
        if (mi_32_col_offset >= max_cols || mi_32_row_offset >= max_rows)
          continue;
        switch (mip[0]->sb_type) {
          case BLOCK_32X32:
            build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
            break;
          case BLOCK_32X16:
            build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
            if (mi_32_row_offset + 2 >= max_rows)
              continue;
            mip2 = mip + mode_info_stride * 2;
            build_masks(lfi_n, mip2[0], shift_y + 16, shift_uv + 4, lfm);
            break;
          case BLOCK_16X32:
            build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
            if (mi_32_col_offset + 2 >= max_cols)
              continue;
            mip2 = mip + 2;
            build_masks(lfi_n, mip2[0], shift_y + 2, shift_uv + 1, lfm);
            break;
          default:
            for (idx_16 = 0; idx_16 < 4; mip += offset_16[idx_16], ++idx_16) {
              const int shift_y = shift_32_y[idx_32] + shift_16_y[idx_16];
              const int shift_uv = shift_32_uv[idx_32] + shift_16_uv[idx_16];
              const int mi_16_col_offset = mi_32_col_offset +
                  ((idx_16 & 1) << 1);
              const int mi_16_row_offset = mi_32_row_offset +
                  ((idx_16 >> 1) << 1);

              if (mi_16_col_offset >= max_cols || mi_16_row_offset >= max_rows)
                continue;

              switch (mip[0]->sb_type) {
                case BLOCK_16X16:
                  build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
                  break;
                case BLOCK_16X8:
                  build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
                  if (mi_16_row_offset + 1 >= max_rows)
                    continue;
                  mip2 = mip + mode_info_stride;
                  build_y_mask(lfi_n, mip2[0], shift_y+8, lfm);
                  break;
                case BLOCK_8X16:
                  build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
                  if (mi_16_col_offset +1 >= max_cols)
                    continue;
                  mip2 = mip + 1;
                  build_y_mask(lfi_n, mip2[0], shift_y+1, lfm);
                  break;
                default: {
                  const int shift_y = shift_32_y[idx_32] +
                                      shift_16_y[idx_16] +
                                      shift_8_y[0];
                  build_masks(lfi_n, mip[0], shift_y, shift_uv, lfm);
                  mip += offset[0];
                  for (idx_8 = 1; idx_8 < 4; mip += offset[idx_8], ++idx_8) {
                    const int shift_y = shift_32_y[idx_32] +
                                        shift_16_y[idx_16] +
                                        shift_8_y[idx_8];
                    const int mi_8_col_offset = mi_16_col_offset +
                        ((idx_8 & 1));
                    const int mi_8_row_offset = mi_16_row_offset +
                        ((idx_8 >> 1));

                    if (mi_8_col_offset >= max_cols ||
                        mi_8_row_offset >= max_rows)
                      continue;
                    build_y_mask(lfi_n, mip[0], shift_y, lfm);
                  }
                  break;
                }
              }
            }
            break;
        }
      }
      break;
  }
}

static void filter_selectively_vert(uint8_t *s, int pitch,
                                    unsigned int mask_16x16,
                                    unsigned int mask_8x8,
                                    unsigned int mask_4x4,
                                    unsigned int mask_4x4_int,
                                    const loop_filter_thresh *lfthr,
                                    const uint8_t *lfl) {
  unsigned int mask;

  for (mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
       mask; mask >>= 1) {
    const loop_filter_thresh *lfi = lfthr + *lfl;

    if (mask & 1) {
      if (mask_16x16 & 1) {
        vpx_lpf_vertical_16(s, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);
      } else if (mask_8x8 & 1) {
        vpx_lpf_vertical_8(s, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);
      } else if (mask_4x4 & 1) {
        vpx_lpf_vertical_4(s, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);
      }
    }
    if (mask_4x4_int & 1)
      vpx_lpf_vertical_4(s + 4, pitch, lfi->mblim, lfi->lim, lfi->hev_thr);
    s += 8;
    lfl += 1;
    mask_16x16 >>= 1;
    mask_8x8 >>= 1;
    mask_4x4 >>= 1;
    mask_4x4_int >>= 1;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_filter_selectively_vert(uint16_t *s, int pitch,
                                           unsigned int mask_16x16,
                                           unsigned int mask_8x8,
                                           unsigned int mask_4x4,
                                           unsigned int mask_4x4_int,
                                           const loop_filter_thresh *lfthr,
                                           const uint8_t *lfl, int bd) {
  unsigned int mask;

  for (mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
       mask; mask >>= 1) {
    const loop_filter_thresh *lfi = lfthr + *lfl;

    if (mask & 1) {
      if (mask_16x16 & 1) {
        vpx_highbd_lpf_vertical_16(s, pitch, lfi->mblim, lfi->lim,
                                   lfi->hev_thr, bd);
      } else if (mask_8x8 & 1) {
        vpx_highbd_lpf_vertical_8(s, pitch, lfi->mblim, lfi->lim,
                                  lfi->hev_thr, bd);
      } else if (mask_4x4 & 1) {
        vpx_highbd_lpf_vertical_4(s, pitch, lfi->mblim, lfi->lim,
                                  lfi->hev_thr, bd);
      }
    }
    if (mask_4x4_int & 1)
      vpx_highbd_lpf_vertical_4(s + 4, pitch, lfi->mblim, lfi->lim,
                                lfi->hev_thr, bd);
    s += 8;
    lfl += 1;
    mask_16x16 >>= 1;
    mask_8x8 >>= 1;
    mask_4x4 >>= 1;
    mask_4x4_int >>= 1;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_filter_block_plane_non420(VP9_COMMON *cm,
                                   struct macroblockd_plane *plane,
                                   MODE_INFO **mi_8x8,
                                   int mi_row, int mi_col) {
  const int ss_x = plane->subsampling_x;
  const int ss_y = plane->subsampling_y;
  const int row_step = 1 << ss_y;
  const int col_step = 1 << ss_x;
  const int row_step_stride = cm->mi_stride * row_step;
  struct buf_2d *const dst = &plane->dst;
  uint8_t* const dst0 = dst->buf;
  unsigned int mask_16x16[MI_BLOCK_SIZE] = {0};
  unsigned int mask_8x8[MI_BLOCK_SIZE] = {0};
  unsigned int mask_4x4[MI_BLOCK_SIZE] = {0};
  unsigned int mask_4x4_int[MI_BLOCK_SIZE] = {0};
  uint8_t lfl[MI_BLOCK_SIZE * MI_BLOCK_SIZE];
  int r, c;

  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r += row_step) {
    unsigned int mask_16x16_c = 0;
    unsigned int mask_8x8_c = 0;
    unsigned int mask_4x4_c = 0;
    unsigned int border_mask;

    // Determine the vertical edges that need filtering
    for (c = 0; c < MI_BLOCK_SIZE && mi_col + c < cm->mi_cols; c += col_step) {
      const MODE_INFO *mi = mi_8x8[c];
      const BLOCK_SIZE sb_type = mi[0].sb_type;
      const int skip_this = mi[0].skip && is_inter_block(mi);
      // left edge of current unit is block/partition edge -> no skip
      const int block_edge_left = (num_4x4_blocks_wide_lookup[sb_type] > 1) ?
          !(c & (num_8x8_blocks_wide_lookup[sb_type] - 1)) : 1;
      const int skip_this_c = skip_this && !block_edge_left;
      // top edge of current unit is block/partition edge -> no skip
      const int block_edge_above = (num_4x4_blocks_high_lookup[sb_type] > 1) ?
          !(r & (num_8x8_blocks_high_lookup[sb_type] - 1)) : 1;
      const int skip_this_r = skip_this && !block_edge_above;
      const TX_SIZE tx_size = get_uv_tx_size(mi, plane);
      const int skip_border_4x4_c = ss_x && mi_col + c == cm->mi_cols - 1;
      const int skip_border_4x4_r = ss_y && mi_row + r == cm->mi_rows - 1;

      // Filter level can vary per MI
      if (!(lfl[(r << 3) + (c >> ss_x)] =
            get_filter_level(&cm->lf_info, mi)))
        continue;

      // Build masks based on the transform size of each block
      if (tx_size == TX_32X32) {
        if (!skip_this_c && ((c >> ss_x) & 3) == 0) {
          if (!skip_border_4x4_c)
            mask_16x16_c |= 1 << (c >> ss_x);
          else
            mask_8x8_c |= 1 << (c >> ss_x);
        }
        if (!skip_this_r && ((r >> ss_y) & 3) == 0) {
          if (!skip_border_4x4_r)
            mask_16x16[r] |= 1 << (c >> ss_x);
          else
            mask_8x8[r] |= 1 << (c >> ss_x);
        }
      } else if (tx_size == TX_16X16) {
        if (!skip_this_c && ((c >> ss_x) & 1) == 0) {
          if (!skip_border_4x4_c)
            mask_16x16_c |= 1 << (c >> ss_x);
          else
            mask_8x8_c |= 1 << (c >> ss_x);
        }
        if (!skip_this_r && ((r >> ss_y) & 1) == 0) {
          if (!skip_border_4x4_r)
            mask_16x16[r] |= 1 << (c >> ss_x);
          else
            mask_8x8[r] |= 1 << (c >> ss_x);
        }
      } else {
        // force 8x8 filtering on 32x32 boundaries
        if (!skip_this_c) {
          if (tx_size == TX_8X8 || ((c >> ss_x) & 3) == 0)
            mask_8x8_c |= 1 << (c >> ss_x);
          else
            mask_4x4_c |= 1 << (c >> ss_x);
        }

        if (!skip_this_r) {
          if (tx_size == TX_8X8 || ((r >> ss_y) & 3) == 0)
            mask_8x8[r] |= 1 << (c >> ss_x);
          else
            mask_4x4[r] |= 1 << (c >> ss_x);
        }

        if (!skip_this && tx_size < TX_8X8 && !skip_border_4x4_c)
          mask_4x4_int[r] |= 1 << (c >> ss_x);
      }
    }

    // Disable filtering on the leftmost column
    border_mask = ~(mi_col == 0);
#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_vert(CONVERT_TO_SHORTPTR(dst->buf),
                                     dst->stride,
                                     mask_16x16_c & border_mask,
                                     mask_8x8_c & border_mask,
                                     mask_4x4_c & border_mask,
                                     mask_4x4_int[r],
                                     cm->lf_info.lfthr, &lfl[r << 3],
                                     (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_vert(dst->buf, dst->stride,
                              mask_16x16_c & border_mask,
                              mask_8x8_c & border_mask,
                              mask_4x4_c & border_mask,
                              mask_4x4_int[r],
                              cm->lf_info.lfthr, &lfl[r << 3]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    dst->buf += 8 * dst->stride;
    mi_8x8 += row_step_stride;
  }

  // Now do horizontal pass
  dst->buf = dst0;
  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r += row_step) {
    const int skip_border_4x4_r = ss_y && mi_row + r == cm->mi_rows - 1;
    const unsigned int mask_4x4_int_r = skip_border_4x4_r ? 0 : mask_4x4_int[r];

    unsigned int mask_16x16_r;
    unsigned int mask_8x8_r;
    unsigned int mask_4x4_r;

    if (mi_row + r == 0) {
      mask_16x16_r = 0;
      mask_8x8_r = 0;
      mask_4x4_r = 0;
    } else {
      mask_16x16_r = mask_16x16[r];
      mask_8x8_r = mask_8x8[r];
      mask_4x4_r = mask_4x4[r];
    }
#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_horiz(CONVERT_TO_SHORTPTR(dst->buf),
                                      dst->stride,
                                      mask_16x16_r,
                                      mask_8x8_r,
                                      mask_4x4_r,
                                      mask_4x4_int_r,
                                      cm->lf_info.lfthr, &lfl[r << 3],
                                      (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_horiz(dst->buf, dst->stride,
                               mask_16x16_r,
                               mask_8x8_r,
                               mask_4x4_r,
                               mask_4x4_int_r,
                               cm->lf_info.lfthr, &lfl[r << 3]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    dst->buf += 8 * dst->stride;
  }
}

void vp9_filter_block_plane_ss00(VP9_COMMON *const cm,
                                 struct macroblockd_plane *const plane,
                                 int mi_row,
                                 LOOP_FILTER_MASK *lfm) {
  struct buf_2d *const dst = &plane->dst;
  uint8_t *const dst0 = dst->buf;
  int r;
  uint64_t mask_16x16 = lfm->left_y[TX_16X16];
  uint64_t mask_8x8 = lfm->left_y[TX_8X8];
  uint64_t mask_4x4 = lfm->left_y[TX_4X4];
  uint64_t mask_4x4_int = lfm->int_4x4_y;

  assert(plane->subsampling_x == 0 && plane->subsampling_y == 0);

  // Vertical pass: do 2 rows at one time
  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r += 2) {
    // Disable filtering on the leftmost column.
#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_vert_row2(plane->subsampling_x,
                                          CONVERT_TO_SHORTPTR(dst->buf),
                                          dst->stride,
                                          (unsigned int)mask_16x16,
                                          (unsigned int)mask_8x8,
                                          (unsigned int)mask_4x4,
                                          (unsigned int)mask_4x4_int,
                                          cm->lf_info.lfthr,
                                          &lfm->lfl_y[r << 3],
                                          (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_vert_row2(plane->subsampling_x, dst->buf, dst->stride,
                                   (unsigned int)mask_16x16,
                                   (unsigned int)mask_8x8,
                                   (unsigned int)mask_4x4,
                                   (unsigned int)mask_4x4_int,
                                   cm->lf_info.lfthr, &lfm->lfl_y[r << 3]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    dst->buf += 16 * dst->stride;
    mask_16x16 >>= 16;
    mask_8x8 >>= 16;
    mask_4x4 >>= 16;
    mask_4x4_int >>= 16;
  }

  // Horizontal pass
  dst->buf = dst0;
  mask_16x16 = lfm->above_y[TX_16X16];
  mask_8x8 = lfm->above_y[TX_8X8];
  mask_4x4 = lfm->above_y[TX_4X4];
  mask_4x4_int = lfm->int_4x4_y;

  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r++) {
    unsigned int mask_16x16_r;
    unsigned int mask_8x8_r;
    unsigned int mask_4x4_r;

    if (mi_row + r == 0) {
      mask_16x16_r = 0;
      mask_8x8_r = 0;
      mask_4x4_r = 0;
    } else {
      mask_16x16_r = mask_16x16 & 0xff;
      mask_8x8_r = mask_8x8 & 0xff;
      mask_4x4_r = mask_4x4 & 0xff;
    }

#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_horiz(CONVERT_TO_SHORTPTR(dst->buf),
                                      dst->stride, mask_16x16_r, mask_8x8_r,
                                      mask_4x4_r, mask_4x4_int & 0xff,
                                      cm->lf_info.lfthr, &lfm->lfl_y[r << 3],
                                      (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_horiz(dst->buf, dst->stride, mask_16x16_r, mask_8x8_r,
                               mask_4x4_r, mask_4x4_int & 0xff,
                               cm->lf_info.lfthr, &lfm->lfl_y[r << 3]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH

    dst->buf += 8 * dst->stride;
    mask_16x16 >>= 8;
    mask_8x8 >>= 8;
    mask_4x4 >>= 8;
    mask_4x4_int >>= 8;
  }
}

void vp9_filter_block_plane_ss11(VP9_COMMON *const cm,
                                 struct macroblockd_plane *const plane,
                                 int mi_row,
                                 LOOP_FILTER_MASK *lfm) {
  struct buf_2d *const dst = &plane->dst;
  uint8_t *const dst0 = dst->buf;
  int r, c;
  uint8_t lfl_uv[16];

  uint16_t mask_16x16 = lfm->left_uv[TX_16X16];
  uint16_t mask_8x8 = lfm->left_uv[TX_8X8];
  uint16_t mask_4x4 = lfm->left_uv[TX_4X4];
  uint16_t mask_4x4_int = lfm->int_4x4_uv;

  assert(plane->subsampling_x == 1 && plane->subsampling_y == 1);

  // Vertical pass: do 2 rows at one time
  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r += 4) {
    for (c = 0; c < (MI_BLOCK_SIZE >> 1); c++) {
      lfl_uv[(r << 1) + c] = lfm->lfl_y[(r << 3) + (c << 1)];
      lfl_uv[((r + 2) << 1) + c] = lfm->lfl_y[((r + 2) << 3) + (c << 1)];
    }

    // Disable filtering on the leftmost column.
#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_vert_row2(plane->subsampling_x,
                                          CONVERT_TO_SHORTPTR(dst->buf),
                                          dst->stride,
                                          (unsigned int)mask_16x16,
                                          (unsigned int)mask_8x8,
                                          (unsigned int)mask_4x4,
                                          (unsigned int)mask_4x4_int,
                                          cm->lf_info.lfthr, &lfl_uv[r << 1],
                                          (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_vert_row2(plane->subsampling_x, dst->buf, dst->stride,
                                   (unsigned int)mask_16x16,
                                   (unsigned int)mask_8x8,
                                   (unsigned int)mask_4x4,
                                   (unsigned int)mask_4x4_int,
                                   cm->lf_info.lfthr, &lfl_uv[r << 1]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH

    dst->buf += 16 * dst->stride;
    mask_16x16 >>= 8;
    mask_8x8 >>= 8;
    mask_4x4 >>= 8;
    mask_4x4_int >>= 8;
  }

  // Horizontal pass
  dst->buf = dst0;
  mask_16x16 = lfm->above_uv[TX_16X16];
  mask_8x8 = lfm->above_uv[TX_8X8];
  mask_4x4 = lfm->above_uv[TX_4X4];
  mask_4x4_int = lfm->int_4x4_uv;

  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; r += 2) {
    const int skip_border_4x4_r = mi_row + r == cm->mi_rows - 1;
    const unsigned int mask_4x4_int_r =
        skip_border_4x4_r ? 0 : (mask_4x4_int & 0xf);
    unsigned int mask_16x16_r;
    unsigned int mask_8x8_r;
    unsigned int mask_4x4_r;

    if (mi_row + r == 0) {
      mask_16x16_r = 0;
      mask_8x8_r = 0;
      mask_4x4_r = 0;
    } else {
      mask_16x16_r = mask_16x16 & 0xf;
      mask_8x8_r = mask_8x8 & 0xf;
      mask_4x4_r = mask_4x4 & 0xf;
    }

#if CONFIG_VP9_HIGHBITDEPTH
    if (cm->use_highbitdepth) {
      highbd_filter_selectively_horiz(CONVERT_TO_SHORTPTR(dst->buf),
                                      dst->stride, mask_16x16_r, mask_8x8_r,
                                      mask_4x4_r, mask_4x4_int_r,
                                      cm->lf_info.lfthr, &lfl_uv[r << 1],
                                      (int)cm->bit_depth);
    } else {
#endif  // CONFIG_VP9_HIGHBITDEPTH
      filter_selectively_horiz(dst->buf, dst->stride, mask_16x16_r, mask_8x8_r,
                               mask_4x4_r, mask_4x4_int_r, cm->lf_info.lfthr,
                               &lfl_uv[r << 1]);
#if CONFIG_VP9_HIGHBITDEPTH
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH

    dst->buf += 8 * dst->stride;
    mask_16x16 >>= 4;
    mask_8x8 >>= 4;
    mask_4x4 >>= 4;
    mask_4x4_int >>= 4;
  }
}

static void loop_filter_rows(YV12_BUFFER_CONFIG *frame_buffer, VP9_COMMON *cm,
                             struct macroblockd_plane planes[MAX_MB_PLANE],
                             int start, int stop, int y_only) {
  const int num_planes = y_only ? 1 : MAX_MB_PLANE;
  enum lf_path path;
  int mi_row, mi_col;

  if (y_only)
    path = LF_PATH_444;
  else if (planes[1].subsampling_y == 1 && planes[1].subsampling_x == 1)
    path = LF_PATH_420;
  else if (planes[1].subsampling_y == 0 && planes[1].subsampling_x == 0)
    path = LF_PATH_444;
  else
    path = LF_PATH_SLOW;

  for (mi_row = start; mi_row < stop; mi_row += MI_BLOCK_SIZE) {
    MODE_INFO **mi = cm->mi_grid_visible + mi_row * cm->mi_stride;
    LOOP_FILTER_MASK *lfm = get_lfm(&cm->lf, mi_row, 0);

    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE, ++lfm) {
      int plane;

      vp9_setup_dst_planes(planes, frame_buffer, mi_row, mi_col);

      // TODO(jimbankoski): For 444 only need to do y mask.
      vp9_adjust_mask(cm, mi_row, mi_col, lfm);

      vp9_filter_block_plane_ss00(cm, &planes[0], mi_row, lfm);
      for (plane = 1; plane < num_planes; ++plane) {
        switch (path) {
          case LF_PATH_420:
            vp9_filter_block_plane_ss11(cm, &planes[plane], mi_row, lfm);
            break;
          case LF_PATH_444:
            vp9_filter_block_plane_ss00(cm, &planes[plane], mi_row, lfm);
            break;
          case LF_PATH_SLOW:
            vp9_filter_block_plane_non420(cm, &planes[plane], mi + mi_col,
                                          mi_row, mi_col);
            break;
        }
      }
    }
  }
}

void vp9_loop_filter_frame(YV12_BUFFER_CONFIG *frame,
                           VP9_COMMON *cm, MACROBLOCKD *xd,
                           int frame_filter_level,
                           int y_only, int partial_frame) {
  int start_mi_row, end_mi_row, mi_rows_to_filter;
  if (!frame_filter_level) return;
  start_mi_row = 0;
  mi_rows_to_filter = cm->mi_rows;
  if (partial_frame && cm->mi_rows > 8) {
    start_mi_row = cm->mi_rows >> 1;
    start_mi_row &= 0xfffffff8;
    mi_rows_to_filter = VPXMAX(cm->mi_rows / 8, 8);
  }
  end_mi_row = start_mi_row + mi_rows_to_filter;
  loop_filter_rows(frame, cm, xd->plane, start_mi_row, end_mi_row, y_only);
}

// Used by the encoder to build the loopfilter masks.
// TODO(slavarnway): Do the encoder the same way the decoder does it and
//                   build the masks in line as part of the encode process.
void vp9_build_mask_frame(VP9_COMMON *cm, int frame_filter_level,
                          int partial_frame) {
  int start_mi_row, end_mi_row, mi_rows_to_filter;
  int mi_col, mi_row;
  if (!frame_filter_level) return;
  start_mi_row = 0;
  mi_rows_to_filter = cm->mi_rows;
  if (partial_frame && cm->mi_rows > 8) {
    start_mi_row = cm->mi_rows >> 1;
    start_mi_row &= 0xfffffff8;
    mi_rows_to_filter = VPXMAX(cm->mi_rows / 8, 8);
  }
  end_mi_row = start_mi_row + mi_rows_to_filter;

  vp9_loop_filter_frame_init(cm, frame_filter_level);

  for (mi_row = start_mi_row; mi_row < end_mi_row; mi_row += MI_BLOCK_SIZE) {
    MODE_INFO **mi = cm->mi_grid_visible + mi_row * cm->mi_stride;
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
      // vp9_setup_mask() zeros lfm
      vp9_setup_mask(cm, mi_row, mi_col, mi + mi_col, cm->mi_stride,
                     get_lfm(&cm->lf, mi_row, mi_col));
    }
  }
}

// 8x8 blocks in a superblock.  A "1" represents the first block in a 16x16
// or greater area.
static const uint8_t first_block_in_16x16[8][8] = {
  {1, 0, 1, 0, 1, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0},
  {1, 0, 1, 0, 1, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0},
  {1, 0, 1, 0, 1, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0},
  {1, 0, 1, 0, 1, 0, 1, 0},
  {0, 0, 0, 0, 0, 0, 0, 0}
};

// This function sets up the bit masks for a block represented
// by mi_row, mi_col in a 64x64 region.
// TODO(SJL): This function only works for yv12.
void vp9_build_mask(VP9_COMMON *cm, const MODE_INFO *mi, int mi_row,
                    int mi_col, int bw, int bh) {
  const BLOCK_SIZE block_size = mi->sb_type;
  const TX_SIZE tx_size_y = mi->tx_size;
  const loop_filter_info_n *const lfi_n = &cm->lf_info;
  const int filter_level = get_filter_level(lfi_n, mi);
  const TX_SIZE tx_size_uv = get_uv_tx_size_impl(tx_size_y, block_size, 1, 1);
  LOOP_FILTER_MASK *const lfm = get_lfm(&cm->lf, mi_row, mi_col);
  uint64_t *const left_y = &lfm->left_y[tx_size_y];
  uint64_t *const above_y = &lfm->above_y[tx_size_y];
  uint64_t *const int_4x4_y = &lfm->int_4x4_y;
  uint16_t *const left_uv = &lfm->left_uv[tx_size_uv];
  uint16_t *const above_uv = &lfm->above_uv[tx_size_uv];
  uint16_t *const int_4x4_uv = &lfm->int_4x4_uv;
  const int row_in_sb = (mi_row & 7);
  const int col_in_sb = (mi_col & 7);
  const int shift_y = col_in_sb + (row_in_sb << 3);
  const int shift_uv = (col_in_sb >> 1) + ((row_in_sb >> 1) << 2);
  const int build_uv = first_block_in_16x16[row_in_sb][col_in_sb];

  if (!filter_level) {
    return;
  } else {
    int index = shift_y;
    int i;
    for (i = 0; i < bh; i++) {
      memset(&lfm->lfl_y[index], filter_level, bw);
      index += 8;
    }
  }

  // These set 1 in the current block size for the block size edges.
  // For instance if the block size is 32x16, we'll set:
  //    above =   1111
  //              0000
  //    and
  //    left  =   1000
  //          =   1000
  // NOTE : In this example the low bit is left most ( 1000 ) is stored as
  //        1,  not 8...
  //
  // U and V set things on a 16 bit scale.
  //
  *above_y |= above_prediction_mask[block_size] << shift_y;
  *left_y |= left_prediction_mask[block_size] << shift_y;

  if (build_uv) {
    *above_uv |= above_prediction_mask_uv[block_size] << shift_uv;
    *left_uv |= left_prediction_mask_uv[block_size] << shift_uv;
  }

  // If the block has no coefficients and is not intra we skip applying
  // the loop filter on block edges.
  if (mi->skip && is_inter_block(mi))
    return;

  // Add a mask for the transform size. The transform size mask is set to
  // be correct for a 64x64 prediction block size. Mask to match the size of
  // the block we are working on and then shift it into place.
  *above_y |= (size_mask[block_size] &
               above_64x64_txform_mask[tx_size_y]) << shift_y;
  *left_y |= (size_mask[block_size] &
              left_64x64_txform_mask[tx_size_y]) << shift_y;

  if (build_uv) {
    *above_uv |= (size_mask_uv[block_size] &
                  above_64x64_txform_mask_uv[tx_size_uv]) << shift_uv;

    *left_uv |= (size_mask_uv[block_size] &
                 left_64x64_txform_mask_uv[tx_size_uv]) << shift_uv;
  }

  // Try to determine what to do with the internal 4x4 block boundaries.  These
  // differ from the 4x4 boundaries on the outside edge of an 8x8 in that the
  // internal ones can be skipped and don't depend on the prediction block size.
  if (tx_size_y == TX_4X4)
    *int_4x4_y |= size_mask[block_size] << shift_y;

  if (build_uv && tx_size_uv == TX_4X4)
    *int_4x4_uv |= (size_mask_uv[block_size] & 0xffff) << shift_uv;
}

void vp9_loop_filter_data_reset(
    LFWorkerData *lf_data, YV12_BUFFER_CONFIG *frame_buffer,
    struct VP9Common *cm, const struct macroblockd_plane planes[MAX_MB_PLANE]) {
  lf_data->frame_buffer = frame_buffer;
  lf_data->cm = cm;
  lf_data->start = 0;
  lf_data->stop = 0;
  lf_data->y_only = 0;
  memcpy(lf_data->planes, planes, sizeof(lf_data->planes));
}

void vp9_reset_lfm(VP9_COMMON *const cm) {
  if (cm->lf.filter_level) {
    memset(cm->lf.lfm, 0,
           ((cm->mi_rows + (MI_BLOCK_SIZE - 1)) >> 3) * cm->lf.lfm_stride *
            sizeof(*cm->lf.lfm));
  }
}

int vp9_loop_filter_worker(LFWorkerData *const lf_data, void *unused) {
  (void)unused;
  loop_filter_rows(lf_data->frame_buffer, lf_data->cm, lf_data->planes,
                   lf_data->start, lf_data->stop, lf_data->y_only);
  return 1;
}
