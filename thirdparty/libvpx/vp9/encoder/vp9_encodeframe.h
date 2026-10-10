/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_
#define VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct macroblock;
struct yv12_buffer_config;
struct VP9_COMP;
struct ThreadData;

void vp9_setup_src_planes(struct macroblock *x,
                          const struct yv12_buffer_config *src, int mi_row,
                          int mi_col);

void vp9_encode_frame(struct VP9_COMP *cpi);

void vp9_init_tile_data(struct VP9_COMP *cpi);
void vp9_encode_tile(struct VP9_COMP *cpi, struct ThreadData *td, int tile_row,
                     int tile_col);

void vp9_encode_sb_row(struct VP9_COMP *cpi, struct ThreadData *td,
                       int tile_row, int tile_col, int mi_row);

void vp9_set_variance_partition_thresholds(struct VP9_COMP *cpi, int q,
                                           int content_state);

struct KMEANS_DATA;
void vp9_kmeans(double *ctr_ls, double *boundary_ls, int *count_ls, int k,
                struct KMEANS_DATA *arr, int size);
int vp9_get_group_idx(double value, double *boundary_ls, int k);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODEFRAME_H_
