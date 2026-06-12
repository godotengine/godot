/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_MCOMP_H_
#define VPX_VP8_ENCODER_MCOMP_H_

#include "block.h"
#include "vpx_dsp/variance.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The maximum number of steps in a step search given the largest allowed
 * initial step
 */
#define MAX_MVSEARCH_STEPS 8

/* Max full pel mv specified in 1 pel units */
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS)) - 1)

/* Maximum size of the first step in full pel units */
#define MAX_FIRST_STEP (1 << (MAX_MVSEARCH_STEPS - 1))

int vp8_mv_bit_cost(int_mv *mv, int_mv *ref, int *mvcost[2], int Weight);
void vp8_init_dsmotion_compensation(MACROBLOCK *x, int stride);
void vp8_init3smotion_compensation(MACROBLOCK *x, int stride);

int vp8_hex_search(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                   int_mv *best_mv, int search_param, int sad_per_bit,
                   const vp8_variance_fn_ptr_t *vfp, int *mvsadcost[2],
                   int_mv *center_mv);

typedef int(fractional_mv_step_fp)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                   int_mv *bestmv, int_mv *ref_mv,
                                   int error_per_bit,
                                   const vp8_variance_fn_ptr_t *vfp,
                                   int *mvcost[2], int *distortion,
                                   unsigned int *sse);

fractional_mv_step_fp vp8_find_best_sub_pixel_step_iteratively;
fractional_mv_step_fp vp8_find_best_sub_pixel_step;
fractional_mv_step_fp vp8_find_best_half_pixel_step;
fractional_mv_step_fp vp8_skip_fractional_mv_step;

int vp8_full_search_sad(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                        int sad_per_bit, int distance,
                        vp8_variance_fn_ptr_t *fn_ptr, int *mvcost[2],
                        int_mv *center_mv);

typedef int (*vp8_refining_search_fn_t)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                        int_mv *ref_mv, int sad_per_bit,
                                        int distance,
                                        vp8_variance_fn_ptr_t *fn_ptr,
                                        int *mvcost[2], int_mv *center_mv);

typedef int (*vp8_diamond_search_fn_t)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                       int_mv *ref_mv, int_mv *best_mv,
                                       int search_param, int sad_per_bit,
                                       int *num00,
                                       vp8_variance_fn_ptr_t *fn_ptr,
                                       int *mvcost[2], int_mv *center_mv);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_MCOMP_H_
