/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_RD_H_
#define VPX_VP9_ENCODER_VP9_RD_H_

#include <limits.h>

#include "vp9/common/vp9_blockd.h"

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_context_tree.h"
#include "vp9/encoder/vp9_cost.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RDDIV_BITS 7
#define RD_EPB_SHIFT 6

#define RDCOST(RM, DM, R, D) \
  ROUND_POWER_OF_TWO(((int64_t)(R)) * (RM), VP9_PROB_COST_SHIFT) + ((D) << (DM))
#define RDCOST_NEG_R(RM, DM, R, D) \
  ((D) << (DM)) - ROUND_POWER_OF_TWO(((int64_t)(R)) * (RM), VP9_PROB_COST_SHIFT)
#define RDCOST_NEG_D(RM, DM, R, D) \
  ROUND_POWER_OF_TWO(((int64_t)(R)) * (RM), VP9_PROB_COST_SHIFT) - ((D) << (DM))

#define QIDX_SKIP_THRESH 115

#define MV_COST_WEIGHT 108
#define MV_COST_WEIGHT_SUB 120

#define MAX_MODES 30
#define MAX_REFS 6

#define RD_THRESH_INIT_FACT 32
#define RD_THRESH_MAX_FACT 64
#define RD_THRESH_INC 1

#define VP9_DIST_SCALE_LOG2 4
#define VP9_DIST_SCALE (1 << VP9_DIST_SCALE_LOG2)

// This enumerator type needs to be kept aligned with the mode order in
// const MODE_DEFINITION vp9_mode_order[MAX_MODES] used in the rd code.
typedef enum {
  THR_NEARESTMV,
  THR_NEARESTA,
  THR_NEARESTG,

  THR_DC,

  THR_NEWMV,
  THR_NEWA,
  THR_NEWG,

  THR_NEARMV,
  THR_NEARA,
  THR_NEARG,

  THR_ZEROMV,
  THR_ZEROG,
  THR_ZEROA,

  THR_COMP_NEARESTLA,
  THR_COMP_NEARESTGA,

  THR_TM,

  THR_COMP_NEARLA,
  THR_COMP_NEWLA,
  THR_COMP_NEARGA,
  THR_COMP_NEWGA,

  THR_COMP_ZEROLA,
  THR_COMP_ZEROGA,

  THR_H_PRED,
  THR_V_PRED,
  THR_D135_PRED,
  THR_D207_PRED,
  THR_D153_PRED,
  THR_D63_PRED,
  THR_D117_PRED,
  THR_D45_PRED,
} THR_MODES;

typedef enum {
  THR_LAST,
  THR_GOLD,
  THR_ALTR,
  THR_COMP_LA,
  THR_COMP_GA,
  THR_INTRA,
} THR_MODES_SUB8X8;

typedef struct {
  // RD multiplier control factors added for Vizier project.
  double rd_mult_inter_qp_fac;
  double rd_mult_arf_qp_fac;
  double rd_mult_key_qp_fac;
} RD_CONTROL;

typedef struct RD_OPT {
  // Thresh_mult is used to set a threshold for the rd score. A higher value
  // means that we will accept the best mode so far more often. This number
  // is used in combination with the current block size, and thresh_freq_fact to
  // pick a threshold.
  int thresh_mult[MAX_MODES];
  int thresh_mult_sub8x8[MAX_REFS];

  int threshes[MAX_SEGMENTS][BLOCK_SIZES][MAX_MODES];

  int64_t prediction_type_threshes[MAX_REF_FRAMES][REFERENCE_MODES];

  int64_t filter_threshes[MAX_REF_FRAMES][SWITCHABLE_FILTER_CONTEXTS];
  int64_t prediction_type_threshes_prev[MAX_REF_FRAMES][REFERENCE_MODES];

  int64_t filter_threshes_prev[MAX_REF_FRAMES][SWITCHABLE_FILTER_CONTEXTS];
  int RDMULT;
  int RDDIV;
  double r0;
} RD_OPT;

typedef struct RD_COST {
  int rate;
  int64_t dist;
  int64_t rdcost;
} RD_COST;

// Reset the rate distortion cost values to maximum (invalid) value.
void vp9_rd_cost_reset(RD_COST *rd_cost);
// Initialize the rate distortion cost values to zero.
void vp9_rd_cost_init(RD_COST *rd_cost);
// It supports negative rate and dist, which is different from RDCOST().
int64_t vp9_calculate_rd_cost(int mult, int div, int rate, int64_t dist);
// Update the cost value based on its rate and distortion.
void vp9_rd_cost_update(int mult, int div, RD_COST *rd_cost);

struct TileInfo;
struct TileDataEnc;
struct VP9_COMP;
struct macroblock;

void vp9_init_rd_parameters(struct VP9_COMP *cpi);

int vp9_compute_rd_mult_based_on_qindex(const struct VP9_COMP *cpi, int qindex);

int vp9_compute_rd_mult(const struct VP9_COMP *cpi, int qindex);

int vp9_get_adaptive_rdmult(const struct VP9_COMP *cpi, double beta);

void vp9_initialize_rd_consts(struct VP9_COMP *cpi);

void vp9_initialize_me_consts(struct VP9_COMP *cpi, MACROBLOCK *x, int qindex);

void vp9_model_rd_from_var_lapndz(unsigned int var, unsigned int n_log2,
                                  unsigned int qstep, int *rate, int64_t *dist);

int vp9_get_switchable_rate(const struct VP9_COMP *cpi,
                            const MACROBLOCKD *const xd);

int vp9_raster_block_offset(BLOCK_SIZE plane_bsize, int raster_block,
                            int stride);

int16_t *vp9_raster_block_offset_int16(BLOCK_SIZE plane_bsize, int raster_block,
                                       int16_t *base);

YV12_BUFFER_CONFIG *vp9_get_scaled_ref_frame(const struct VP9_COMP *cpi,
                                             int ref_frame);

void vp9_init_me_luts(void);

void vp9_get_entropy_contexts(BLOCK_SIZE bsize, TX_SIZE tx_size,
                              const struct macroblockd_plane *pd,
                              ENTROPY_CONTEXT t_above[16],
                              ENTROPY_CONTEXT t_left[16]);

void vp9_set_rd_speed_thresholds(struct VP9_COMP *cpi);

void vp9_set_rd_speed_thresholds_sub8x8(struct VP9_COMP *cpi);

void vp9_update_rd_thresh_fact(int (*factor_buf)[MAX_MODES], int rd_thresh,
                               int bsize, int best_mode_index);

static INLINE int rd_less_than_thresh(int64_t best_rd, int thresh,
                                      const int *const thresh_fact) {
  return best_rd < ((int64_t)thresh * (*thresh_fact) >> 5) || thresh == INT_MAX;
}

static INLINE void set_error_per_bit(MACROBLOCK *x, int rdmult) {
  x->errorperbit = rdmult >> RD_EPB_SHIFT;
  x->errorperbit += (x->errorperbit == 0);
}

void vp9_mv_pred(struct VP9_COMP *cpi, MACROBLOCK *x, uint8_t *ref_y_buffer,
                 int ref_y_stride, int ref_frame, BLOCK_SIZE block_size);

void vp9_setup_pred_block(const MACROBLOCKD *xd,
                          struct buf_2d dst[MAX_MB_PLANE],
                          const YV12_BUFFER_CONFIG *src, int mi_row, int mi_col,
                          const struct scale_factors *scale,
                          const struct scale_factors *scale_uv);

int vp9_get_intra_cost_penalty(const struct VP9_COMP *const cpi,
                               BLOCK_SIZE bsize, int qindex, int qdelta);

unsigned int vp9_get_sby_variance(struct VP9_COMP *cpi,
                                  const struct buf_2d *ref, BLOCK_SIZE bs);
unsigned int vp9_get_sby_perpixel_variance(struct VP9_COMP *cpi,
                                           const struct buf_2d *ref,
                                           BLOCK_SIZE bs);
#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vp9_high_get_sby_variance(struct VP9_COMP *cpi,
                                       const struct buf_2d *ref, BLOCK_SIZE bs,
                                       int bd);
unsigned int vp9_high_get_sby_perpixel_variance(struct VP9_COMP *cpi,
                                                const struct buf_2d *ref,
                                                BLOCK_SIZE bs, int bd);
#endif

void vp9_build_inter_mode_cost(struct VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_RD_H_
