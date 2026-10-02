/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_ETHREAD_H_
#define VPX_VP9_ENCODER_VP9_ETHREAD_H_

#include "vpx_util/vpx_pthread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NUM_TILE_COLS (1 << 6)
#define MAX_NUM_TILE_ROWS 4
#define MAX_NUM_THREADS 64

struct VP9_COMP;
struct ThreadData;

typedef struct EncWorkerData {
  struct VP9_COMP *cpi;
  struct ThreadData *td;
  int start;
  int thread_id;
  int tile_completion_status[MAX_NUM_TILE_COLS];
} EncWorkerData;

// Encoder row synchronization
typedef struct VP9RowMTSyncData {
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex;
  pthread_cond_t *cond;
#endif
  // Allocate memory to store the sb/mb block index in each row.
  int *cur_col;
  int sync_range;
  int rows;
} VP9RowMTSync;

// Frees EncWorkerData related allocations made by vp9_encode_*_mt().
// row_mt specific data is freed with vp9_row_mt_mem_dealloc() and is not
// called by this function.
void vp9_encode_free_mt_data(struct VP9_COMP *cpi);

void vp9_encode_tiles_mt(struct VP9_COMP *cpi);

void vp9_encode_tiles_row_mt(struct VP9_COMP *cpi);

void vp9_encode_fp_row_mt(struct VP9_COMP *cpi);

void vp9_row_mt_sync_read(VP9RowMTSync *const row_mt_sync, int r, int c);
void vp9_row_mt_sync_write(VP9RowMTSync *const row_mt_sync, int r, int c,
                           const int cols);

void vp9_row_mt_sync_read_dummy(VP9RowMTSync *const row_mt_sync, int r, int c);
void vp9_row_mt_sync_write_dummy(VP9RowMTSync *const row_mt_sync, int r, int c,
                                 const int cols);

// Allocate memory for row based multi-threading synchronization.
void vp9_row_mt_sync_mem_alloc(VP9RowMTSync *row_mt_sync, struct VP9Common *cm,
                               int rows);

// Deallocate row based multi-threading synchronization related mutex and data.
void vp9_row_mt_sync_mem_dealloc(VP9RowMTSync *row_mt_sync);

void vp9_temporal_filter_row_mt(struct VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ETHREAD_H_
