/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_THREAD_COMMON_H_
#define VPX_VP9_COMMON_VP9_THREAD_COMMON_H_
#include "./vpx_config.h"
#include "vp9/common/vp9_loopfilter.h"
#include "vpx_util/vpx_pthread.h"
#include "vpx_util/vpx_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP9Common;
struct FRAME_COUNTS;

// Loopfilter row synchronization
typedef struct VP9LfSyncData {
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex;
  pthread_cond_t *cond;
#endif
  // Allocate memory to store the loop-filtered superblock index in each row.
  int *cur_sb_col;
  // The optimal sync_range for different resolution and platform should be
  // determined by testing. Currently, it is chosen to be a power-of-2 number.
  int sync_range;
  int rows;

  // Row-based parallel loopfilter data
  LFWorkerData *lfdata;
  int num_workers;         // number of allocated workers.
  int num_active_workers;  // number of scheduled workers.

#if CONFIG_MULTITHREAD
  pthread_mutex_t *lf_mutex;
  pthread_mutex_t *recon_done_mutex;
  pthread_cond_t *recon_done_cond;
#endif
  int *num_tiles_done;
  int corrupted;
} VP9LfSync;

// Allocate memory for loopfilter row synchronization.
void vp9_loop_filter_alloc(VP9LfSync *lf_sync, struct VP9Common *cm, int rows,
                           int width, int num_workers);

// Deallocate loopfilter synchronization related mutex and data.
void vp9_loop_filter_dealloc(VP9LfSync *lf_sync);

// Multi-threaded loopfilter that uses the tile threads.
void vp9_loop_filter_frame_mt(YV12_BUFFER_CONFIG *frame, struct VP9Common *cm,
                              struct macroblockd_plane planes[MAX_MB_PLANE],
                              int frame_filter_level, int y_only,
                              int partial_frame, VPxWorker *workers,
                              int num_workers, VP9LfSync *lf_sync);

// Multi-threaded loopfilter initialisations
void vp9_lpf_mt_init(VP9LfSync *lf_sync, struct VP9Common *cm,
                     int frame_filter_level, int num_workers);

void vp9_loopfilter_rows(LFWorkerData *lf_data, VP9LfSync *lf_sync);

void vp9_set_row(VP9LfSync *lf_sync, int num_tiles, int row, int is_last_row,
                 int corrupted);

void vp9_loopfilter_job(LFWorkerData *lf_data, VP9LfSync *lf_sync);

void vp9_accumulate_frame_counts(struct FRAME_COUNTS *accum,
                                 const struct FRAME_COUNTS *counts, int is_dec);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_THREAD_COMMON_H_
