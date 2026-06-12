/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_LOOPFILTER_H_
#define VPX_VP9_COMMON_VP9_LOOPFILTER_H_

#include "vpx_ports/mem.h"
#include "./vpx_config.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_seg_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LOOP_FILTER 63
#define MAX_SHARPNESS 7

#define SIMD_WIDTH 16

#define MAX_REF_LF_DELTAS 4
#define MAX_MODE_LF_DELTAS 2

enum lf_path {
  LF_PATH_420,
  LF_PATH_444,
  LF_PATH_SLOW,
};

// Need to align this structure so when it is declared and
// passed it can be loaded into vector registers.
typedef struct {
  DECLARE_ALIGNED(SIMD_WIDTH, uint8_t, mblim[SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, uint8_t, lim[SIMD_WIDTH]);
  DECLARE_ALIGNED(SIMD_WIDTH, uint8_t, hev_thr[SIMD_WIDTH]);
} loop_filter_thresh;

typedef struct {
  loop_filter_thresh lfthr[MAX_LOOP_FILTER + 1];
  uint8_t lvl[MAX_SEGMENTS][MAX_REF_FRAMES][MAX_MODE_LF_DELTAS];
} loop_filter_info_n;

// This structure holds bit masks for all 8x8 blocks in a 64x64 region.
// Each 1 bit represents a position in which we want to apply the loop filter.
// Left_ entries refer to whether we apply a filter on the border to the
// left of the block.   Above_ entries refer to whether or not to apply a
// filter on the above border.   Int_ entries refer to whether or not to
// apply borders on the 4x4 edges within the 8x8 block that each bit
// represents.
// Since each transform is accompanied by a potentially different type of
// loop filter there is a different entry in the array for each transform size.
typedef struct {
  uint64_t left_y[TX_SIZES];
  uint64_t above_y[TX_SIZES];
  uint64_t int_4x4_y;
  uint16_t left_uv[TX_SIZES];
  uint16_t above_uv[TX_SIZES];
  uint16_t int_4x4_uv;
  uint8_t lfl_y[64];
} LOOP_FILTER_MASK;

struct loopfilter {
  int filter_level;
  int last_filt_level;

  int sharpness_level;
  int last_sharpness_level;

  uint8_t mode_ref_delta_enabled;
  uint8_t mode_ref_delta_update;

  // 0 = Intra, Last, GF, ARF
  signed char ref_deltas[MAX_REF_LF_DELTAS];
  signed char last_ref_deltas[MAX_REF_LF_DELTAS];

  // 0 = ZERO_MV, MV
  signed char mode_deltas[MAX_MODE_LF_DELTAS];
  signed char last_mode_deltas[MAX_MODE_LF_DELTAS];

  LOOP_FILTER_MASK *lfm;
  int lfm_stride;
};

/* assorted loopfilter functions which get used elsewhere */
struct VP9Common;
struct macroblockd;
struct VP9LfSyncData;

// This function sets up the bit masks for the entire 64x64 region represented
// by mi_row, mi_col.
void vp9_setup_mask(struct VP9Common *const cm, const int mi_row,
                    const int mi_col, MODE_INFO **mi8x8,
                    const int mode_info_stride, LOOP_FILTER_MASK *lfm);

void vp9_filter_block_plane_ss00(struct VP9Common *const cm,
                                 struct macroblockd_plane *const plane,
                                 int mi_row, LOOP_FILTER_MASK *lfm);

void vp9_filter_block_plane_ss11(struct VP9Common *const cm,
                                 struct macroblockd_plane *const plane,
                                 int mi_row, LOOP_FILTER_MASK *lfm);

void vp9_filter_block_plane_non420(struct VP9Common *cm,
                                   struct macroblockd_plane *plane,
                                   MODE_INFO **mi_8x8, int mi_row, int mi_col);

void vp9_loop_filter_init(struct VP9Common *cm);

// Update the loop filter for the current frame.
// This should be called before vp9_loop_filter_frame(), vp9_build_mask_frame()
// calls this function directly.
void vp9_loop_filter_frame_init(struct VP9Common *cm, int default_filt_lvl);

void vp9_loop_filter_frame(YV12_BUFFER_CONFIG *frame, struct VP9Common *cm,
                           struct macroblockd *xd, int frame_filter_level,
                           int y_only, int partial_frame);

// Get the superblock lfm for a given mi_row, mi_col.
static INLINE LOOP_FILTER_MASK *get_lfm(const struct loopfilter *lf,
                                        const int mi_row, const int mi_col) {
  return &lf->lfm[(mi_col >> 3) + ((mi_row >> 3) * lf->lfm_stride)];
}

void vp9_build_mask(struct VP9Common *cm, const MODE_INFO *mi, int mi_row,
                    int mi_col, int bw, int bh);
void vp9_adjust_mask(struct VP9Common *const cm, const int mi_row,
                     const int mi_col, LOOP_FILTER_MASK *lfm);
void vp9_build_mask_frame(struct VP9Common *cm, int frame_filter_level,
                          int partial_frame);
void vp9_reset_lfm(struct VP9Common *const cm);

typedef struct LoopFilterWorkerData {
  YV12_BUFFER_CONFIG *frame_buffer;
  struct VP9Common *cm;
  struct macroblockd_plane planes[MAX_MB_PLANE];

  int start;
  int stop;
  int y_only;
} LFWorkerData;

void vp9_loop_filter_data_reset(
    LFWorkerData *lf_data, YV12_BUFFER_CONFIG *frame_buffer,
    struct VP9Common *cm, const struct macroblockd_plane planes[MAX_MB_PLANE]);

// Operates on the rows described by 'arg1' (cast to LFWorkerData *).
int vp9_loop_filter_worker(void *arg1, void *unused);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_LOOPFILTER_H_
