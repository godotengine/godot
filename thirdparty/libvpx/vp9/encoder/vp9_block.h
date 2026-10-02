/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_BLOCK_H_
#define VPX_VP9_ENCODER_VP9_BLOCK_H_

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_entropymv.h"
#include "vp9/common/vp9_entropy.h"

#ifdef __cplusplus
extern "C" {
#endif

struct macroblock_plane {
  DECLARE_ALIGNED(16, int16_t, src_diff[64 * 64]);
  tran_low_t *qcoeff;
  tran_low_t *coeff;
  uint16_t *eobs;
  struct buf_2d src;

  // Quantizer settings
  int16_t *round_fp;
  int16_t *quant_fp;
  int16_t *quant;
  int16_t *quant_shift;
  int16_t *zbin;
  int16_t *round;

  int64_t quant_thred[2];
};

/* The [2] dimension is for whether we skip the EOB node (i.e. if previous
 * coefficient in this block was zero) or not. */
typedef unsigned int vp9_coeff_cost[PLANE_TYPES][REF_TYPES][COEF_BANDS][2]
                                   [COEFF_CONTEXTS][ENTROPY_TOKENS];

typedef struct {
  int_mv ref_mvs[MAX_REF_FRAMES][MAX_MV_REF_CANDIDATES];
  uint8_t mode_context[MAX_REF_FRAMES];
} MB_MODE_INFO_EXT;

typedef struct {
  int col_min;
  int col_max;
  int row_min;
  int row_max;
} MvLimits;

typedef struct macroblock MACROBLOCK;
struct macroblock {
// cf. https://bugs.chromium.org/p/webm/issues/detail?id=1054
#if defined(_MSC_VER) && _MSC_VER < 1900
  int64_t bsse[MAX_MB_PLANE << 2];
#endif

  struct macroblock_plane plane[MAX_MB_PLANE];

  MACROBLOCKD e_mbd;
  MB_MODE_INFO_EXT *mbmi_ext;
  MB_MODE_INFO_EXT *mbmi_ext_base;
  int skip_block;
  int select_tx_size;
  int skip_recode;
  int skip_optimize;
  int q_index;
  double log_block_src_var;
  int block_tx_domain;

  // The equivalent error at the current rdmult of one whole bit (not one
  // bitcost unit).
  int errorperbit;
  // The equivalent SAD error of one (whole) bit at the current quantizer
  // for large blocks.
  int sadperbit16;
  // The equivalent SAD error of one (whole) bit at the current quantizer
  // for sub-8x8 blocks.
  int sadperbit4;
  int rddiv;
  int rdmult;
  int cb_rdmult;
  int segment_id;
  int mb_energy;

  // These are set to their default values at the beginning, and then adjusted
  // further in the encoding process.
  BLOCK_SIZE min_partition_size;
  BLOCK_SIZE max_partition_size;

  int mv_best_ref_index[MAX_REF_FRAMES];
  unsigned int max_mv_context[MAX_REF_FRAMES];
  unsigned int source_variance;
  unsigned int pred_sse[MAX_REF_FRAMES];
  int pred_mv_sad[MAX_REF_FRAMES];

  int nmvjointcost[MV_JOINTS];
  int *nmvcost[2];
  int *nmvcost_hp[2];
  int **mvcost;

  int nmvjointsadcost[MV_JOINTS];
  int *nmvsadcost[2];
  int *nmvsadcost_hp[2];
  int **mvsadcost;

  // sharpness is used to disable skip mode and change rd_mult
  int sharpness;

  // aq mode is used to adjust rd based on segment.
  int adjust_rdmult_by_segment;

  // These define limits to motion vector components to prevent them
  // from extending outside the UMV borders
  MvLimits mv_limits;

  // Notes transform blocks where no coefficients are coded.
  // Set during mode selection. Read during block encoding.
  uint8_t zcoeff_blk[TX_SIZES][256];

  // Accumulate the tx block eobs in a partition block.
  int32_t sum_y_eobs[TX_SIZES];

  int skip;

  int encode_breakout;

  // note that token_costs is the cost when eob node is skipped
  vp9_coeff_cost token_costs[TX_SIZES];

  int optimize;

  // indicate if it is in the rd search loop or encoding process
  int use_lp32x32fdct;
  int skip_encode;

  // In first pass, intra prediction is done based on source pixels
  // at tile boundaries
  int fp_src_pred;

  // use fast quantization process
  int quant_fp;

  // skip forward transform and quantization
  uint8_t skip_txfm[MAX_MB_PLANE << 2];
#define SKIP_TXFM_NONE 0
// TODO(chengchen): consider remove SKIP_TXFM_AC_DC from vp9 completely
// since it increases risks of bad perceptual quality.
// https://crbug.com/webm/1729
#define SKIP_TXFM_AC_DC 1
#define SKIP_TXFM_AC_ONLY 2

// cf. https://bugs.chromium.org/p/webm/issues/detail?id=1054
#if !defined(_MSC_VER) || _MSC_VER >= 1900
  int64_t bsse[MAX_MB_PLANE << 2];
#endif

  // Used to store sub partition's choices.
  MV pred_mv[MAX_REF_FRAMES];

  // Strong color activity detection. Used in RTC coding mode to enhance
  // the visual quality at the boundary of moving color objects.
  uint8_t color_sensitivity[2];

  uint8_t sb_is_skin;

  uint8_t skip_low_source_sad;

  uint8_t lowvar_highsumdiff;

  uint8_t last_sb_high_content;

  int sb_use_mv_part;

  int sb_mvcol_part;

  int sb_mvrow_part;

  int sb_pickmode_part;

  int zero_temp_sad_source;

  // For each superblock: saves the content value (e.g., low/high sad/sumdiff)
  // based on source sad, prior to encoding the frame.
  uint8_t content_state_sb;

  // Used to save the status of whether a block has a low variance in
  // choose_partitioning. 0 for 64x64, 1~2 for 64x32, 3~4 for 32x64, 5~8 for
  // 32x32, 9~24 for 16x16.
  uint8_t variance_low[25];

  uint8_t arf_frame_usage;
  uint8_t lastgolden_frame_usage;

  void (*fwd_txfm4x4)(const int16_t *input, tran_low_t *output, int stride);
  void (*inv_txfm_add)(const tran_low_t *input, uint8_t *dest, int stride,
                       int eob);
#if CONFIG_VP9_HIGHBITDEPTH
  void (*highbd_inv_txfm_add)(const tran_low_t *input, uint16_t *dest,
                              int stride, int eob, int bd);
#endif
  DECLARE_ALIGNED(16, uint8_t, est_pred[64 * 64]);

  struct scale_factors *me_sf;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_BLOCK_H_
