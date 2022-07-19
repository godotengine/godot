/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_thread_common.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_loopfilter.h"

#if CONFIG_MULTITHREAD
static INLINE void mutex_lock(pthread_mutex_t *const mutex) {
  const int kMaxTryLocks = 4000;
  int locked = 0;
  int i;

  for (i = 0; i < kMaxTryLocks; ++i) {
    if (!pthread_mutex_trylock(mutex)) {
      locked = 1;
      break;
    }
  }

  if (!locked)
    pthread_mutex_lock(mutex);
}
#endif  // CONFIG_MULTITHREAD

static INLINE void sync_read(VP9LfSync *const lf_sync, int r, int c) {
#if CONFIG_MULTITHREAD
  const int nsync = lf_sync->sync_range;

  if (r && !(c & (nsync - 1))) {
    pthread_mutex_t *const mutex = &lf_sync->mutex_[r - 1];
    mutex_lock(mutex);

    while (c > lf_sync->cur_sb_col[r - 1] - nsync) {
      pthread_cond_wait(&lf_sync->cond_[r - 1], mutex);
    }
    pthread_mutex_unlock(mutex);
  }
#else
  (void)lf_sync;
  (void)r;
  (void)c;
#endif  // CONFIG_MULTITHREAD
}

static INLINE void sync_write(VP9LfSync *const lf_sync, int r, int c,
                              const int sb_cols) {
#if CONFIG_MULTITHREAD
  const int nsync = lf_sync->sync_range;
  int cur;
  // Only signal when there are enough filtered SB for next row to run.
  int sig = 1;

  if (c < sb_cols - 1) {
    cur = c;
    if (c % nsync)
      sig = 0;
  } else {
    cur = sb_cols + nsync;
  }

  if (sig) {
    mutex_lock(&lf_sync->mutex_[r]);

    lf_sync->cur_sb_col[r] = cur;

    pthread_cond_signal(&lf_sync->cond_[r]);
    pthread_mutex_unlock(&lf_sync->mutex_[r]);
  }
#else
  (void)lf_sync;
  (void)r;
  (void)c;
  (void)sb_cols;
#endif  // CONFIG_MULTITHREAD
}

// Implement row loopfiltering for each thread.
static INLINE
void thread_loop_filter_rows(const YV12_BUFFER_CONFIG *const frame_buffer,
                             VP9_COMMON *const cm,
                             struct macroblockd_plane planes[MAX_MB_PLANE],
                             int start, int stop, int y_only,
                             VP9LfSync *const lf_sync) {
  const int num_planes = y_only ? 1 : MAX_MB_PLANE;
  const int sb_cols = mi_cols_aligned_to_sb(cm->mi_cols) >> MI_BLOCK_SIZE_LOG2;
  int mi_row, mi_col;
  enum lf_path path;
  if (y_only)
    path = LF_PATH_444;
  else if (planes[1].subsampling_y == 1 && planes[1].subsampling_x == 1)
    path = LF_PATH_420;
  else if (planes[1].subsampling_y == 0 && planes[1].subsampling_x == 0)
    path = LF_PATH_444;
  else
    path = LF_PATH_SLOW;

  for (mi_row = start; mi_row < stop;
       mi_row += lf_sync->num_workers * MI_BLOCK_SIZE) {
    MODE_INFO **const mi = cm->mi_grid_visible + mi_row * cm->mi_stride;
    LOOP_FILTER_MASK *lfm = get_lfm(&cm->lf, mi_row, 0);

    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE, ++lfm) {
      const int r = mi_row >> MI_BLOCK_SIZE_LOG2;
      const int c = mi_col >> MI_BLOCK_SIZE_LOG2;
      int plane;

      sync_read(lf_sync, r, c);

      vp9_setup_dst_planes(planes, frame_buffer, mi_row, mi_col);

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

      sync_write(lf_sync, r, c, sb_cols);
    }
  }
}

// Row-based multi-threaded loopfilter hook
static int loop_filter_row_worker(VP9LfSync *const lf_sync,
                                  LFWorkerData *const lf_data) {
  thread_loop_filter_rows(lf_data->frame_buffer, lf_data->cm, lf_data->planes,
                          lf_data->start, lf_data->stop, lf_data->y_only,
                          lf_sync);
  return 1;
}

static void loop_filter_rows_mt(YV12_BUFFER_CONFIG *frame,
                                VP9_COMMON *cm,
                                struct macroblockd_plane planes[MAX_MB_PLANE],
                                int start, int stop, int y_only,
                                VPxWorker *workers, int nworkers,
                                VP9LfSync *lf_sync) {
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  // Number of superblock rows and cols
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  // Decoder may allocate more threads than number of tiles based on user's
  // input.
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int num_workers = VPXMIN(nworkers, tile_cols);
  int i;

  if (!lf_sync->sync_range || sb_rows != lf_sync->rows ||
      num_workers > lf_sync->num_workers) {
    vp9_loop_filter_dealloc(lf_sync);
    vp9_loop_filter_alloc(lf_sync, cm, sb_rows, cm->width, num_workers);
  }

  // Initialize cur_sb_col to -1 for all SB rows.
  memset(lf_sync->cur_sb_col, -1, sizeof(*lf_sync->cur_sb_col) * sb_rows);

  // Set up loopfilter thread data.
  // The decoder is capping num_workers because it has been observed that using
  // more threads on the loopfilter than there are cores will hurt performance
  // on Android. This is because the system will only schedule the tile decode
  // workers on cores equal to the number of tile columns. Then if the decoder
  // tries to use more threads for the loopfilter, it will hurt performance
  // because of contention. If the multithreading code changes in the future
  // then the number of workers used by the loopfilter should be revisited.
  for (i = 0; i < num_workers; ++i) {
    VPxWorker *const worker = &workers[i];
    LFWorkerData *const lf_data = &lf_sync->lfdata[i];

    worker->hook = (VPxWorkerHook)loop_filter_row_worker;
    worker->data1 = lf_sync;
    worker->data2 = lf_data;

    // Loopfilter data
    vp9_loop_filter_data_reset(lf_data, frame, cm, planes);
    lf_data->start = start + i * MI_BLOCK_SIZE;
    lf_data->stop = stop;
    lf_data->y_only = y_only;

    // Start loopfiltering
    if (i == num_workers - 1) {
      winterface->execute(worker);
    } else {
      winterface->launch(worker);
    }
  }

  // Wait till all rows are finished
  for (i = 0; i < num_workers; ++i) {
    winterface->sync(&workers[i]);
  }
}

void vp9_loop_filter_frame_mt(YV12_BUFFER_CONFIG *frame,
                              VP9_COMMON *cm,
                              struct macroblockd_plane planes[MAX_MB_PLANE],
                              int frame_filter_level,
                              int y_only, int partial_frame,
                              VPxWorker *workers, int num_workers,
                              VP9LfSync *lf_sync) {
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
  vp9_loop_filter_frame_init(cm, frame_filter_level);

  loop_filter_rows_mt(frame, cm, planes, start_mi_row, end_mi_row,
                      y_only, workers, num_workers, lf_sync);
}

// Set up nsync by width.
static INLINE int get_sync_range(int width) {
  // nsync numbers are picked by testing. For example, for 4k
  // video, using 4 gives best performance.
  if (width < 640)
    return 1;
  else if (width <= 1280)
    return 2;
  else if (width <= 4096)
    return 4;
  else
    return 8;
}

// Allocate memory for lf row synchronization
void vp9_loop_filter_alloc(VP9LfSync *lf_sync, VP9_COMMON *cm, int rows,
                           int width, int num_workers) {
  lf_sync->rows = rows;
#if CONFIG_MULTITHREAD
  {
    int i;

    CHECK_MEM_ERROR(cm, lf_sync->mutex_,
                    vpx_malloc(sizeof(*lf_sync->mutex_) * rows));
    if (lf_sync->mutex_) {
      for (i = 0; i < rows; ++i) {
        pthread_mutex_init(&lf_sync->mutex_[i], NULL);
      }
    }

    CHECK_MEM_ERROR(cm, lf_sync->cond_,
                    vpx_malloc(sizeof(*lf_sync->cond_) * rows));
    if (lf_sync->cond_) {
      for (i = 0; i < rows; ++i) {
        pthread_cond_init(&lf_sync->cond_[i], NULL);
      }
    }
  }
#endif  // CONFIG_MULTITHREAD

  CHECK_MEM_ERROR(cm, lf_sync->lfdata,
                  vpx_malloc(num_workers * sizeof(*lf_sync->lfdata)));
  lf_sync->num_workers = num_workers;

  CHECK_MEM_ERROR(cm, lf_sync->cur_sb_col,
                  vpx_malloc(sizeof(*lf_sync->cur_sb_col) * rows));

  // Set up nsync.
  lf_sync->sync_range = get_sync_range(width);
}

// Deallocate lf synchronization related mutex and data
void vp9_loop_filter_dealloc(VP9LfSync *lf_sync) {
  if (lf_sync != NULL) {
#if CONFIG_MULTITHREAD
    int i;

    if (lf_sync->mutex_ != NULL) {
      for (i = 0; i < lf_sync->rows; ++i) {
        pthread_mutex_destroy(&lf_sync->mutex_[i]);
      }
      vpx_free(lf_sync->mutex_);
    }
    if (lf_sync->cond_ != NULL) {
      for (i = 0; i < lf_sync->rows; ++i) {
        pthread_cond_destroy(&lf_sync->cond_[i]);
      }
      vpx_free(lf_sync->cond_);
    }
#endif  // CONFIG_MULTITHREAD
    vpx_free(lf_sync->lfdata);
    vpx_free(lf_sync->cur_sb_col);
    // clear the structure as the source of this call may be a resize in which
    // case this call will be followed by an _alloc() which may fail.
    vp9_zero(*lf_sync);
  }
}

// Accumulate frame counts.
void vp9_accumulate_frame_counts(FRAME_COUNTS *accum,
                                 const FRAME_COUNTS *counts, int is_dec) {
  int i, j, k, l, m;

  for (i = 0; i < BLOCK_SIZE_GROUPS; i++)
    for (j = 0; j < INTRA_MODES; j++)
      accum->y_mode[i][j] += counts->y_mode[i][j];

  for (i = 0; i < INTRA_MODES; i++)
    for (j = 0; j < INTRA_MODES; j++)
      accum->uv_mode[i][j] += counts->uv_mode[i][j];

  for (i = 0; i < PARTITION_CONTEXTS; i++)
    for (j = 0; j < PARTITION_TYPES; j++)
      accum->partition[i][j] += counts->partition[i][j];

  if (is_dec) {
    int n;
    for (i = 0; i < TX_SIZES; i++)
      for (j = 0; j < PLANE_TYPES; j++)
        for (k = 0; k < REF_TYPES; k++)
          for (l = 0; l < COEF_BANDS; l++)
            for (m = 0; m < COEFF_CONTEXTS; m++) {
              accum->eob_branch[i][j][k][l][m] +=
                  counts->eob_branch[i][j][k][l][m];
              for (n = 0; n < UNCONSTRAINED_NODES + 1; n++)
                accum->coef[i][j][k][l][m][n] +=
                    counts->coef[i][j][k][l][m][n];
            }
  } else {
    for (i = 0; i < TX_SIZES; i++)
      for (j = 0; j < PLANE_TYPES; j++)
        for (k = 0; k < REF_TYPES; k++)
          for (l = 0; l < COEF_BANDS; l++)
            for (m = 0; m < COEFF_CONTEXTS; m++)
              accum->eob_branch[i][j][k][l][m] +=
                  counts->eob_branch[i][j][k][l][m];
                // In the encoder, coef is only updated at frame
                // level, so not need to accumulate it here.
                // for (n = 0; n < UNCONSTRAINED_NODES + 1; n++)
                //   accum->coef[i][j][k][l][m][n] +=
                //       counts->coef[i][j][k][l][m][n];
  }

  for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; i++)
    for (j = 0; j < SWITCHABLE_FILTERS; j++)
      accum->switchable_interp[i][j] += counts->switchable_interp[i][j];

  for (i = 0; i < INTER_MODE_CONTEXTS; i++)
    for (j = 0; j < INTER_MODES; j++)
      accum->inter_mode[i][j] += counts->inter_mode[i][j];

  for (i = 0; i < INTRA_INTER_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      accum->intra_inter[i][j] += counts->intra_inter[i][j];

  for (i = 0; i < COMP_INTER_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      accum->comp_inter[i][j] += counts->comp_inter[i][j];

  for (i = 0; i < REF_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      for (k = 0; k < 2; k++)
      accum->single_ref[i][j][k] += counts->single_ref[i][j][k];

  for (i = 0; i < REF_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      accum->comp_ref[i][j] += counts->comp_ref[i][j];

  for (i = 0; i < TX_SIZE_CONTEXTS; i++) {
    for (j = 0; j < TX_SIZES; j++)
      accum->tx.p32x32[i][j] += counts->tx.p32x32[i][j];

    for (j = 0; j < TX_SIZES - 1; j++)
      accum->tx.p16x16[i][j] += counts->tx.p16x16[i][j];

    for (j = 0; j < TX_SIZES - 2; j++)
      accum->tx.p8x8[i][j] += counts->tx.p8x8[i][j];
  }

  for (i = 0; i < TX_SIZES; i++)
    accum->tx.tx_totals[i] += counts->tx.tx_totals[i];

  for (i = 0; i < SKIP_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      accum->skip[i][j] += counts->skip[i][j];

  for (i = 0; i < MV_JOINTS; i++)
    accum->mv.joints[i] += counts->mv.joints[i];

  for (k = 0; k < 2; k++) {
    nmv_component_counts *const comps = &accum->mv.comps[k];
    const nmv_component_counts *const comps_t = &counts->mv.comps[k];

    for (i = 0; i < 2; i++) {
      comps->sign[i] += comps_t->sign[i];
      comps->class0_hp[i] += comps_t->class0_hp[i];
      comps->hp[i] += comps_t->hp[i];
    }

    for (i = 0; i < MV_CLASSES; i++)
      comps->classes[i] += comps_t->classes[i];

    for (i = 0; i < CLASS0_SIZE; i++) {
      comps->class0[i] += comps_t->class0[i];
      for (j = 0; j < MV_FP_SIZE; j++)
        comps->class0_fp[i][j] += comps_t->class0_fp[i][j];
    }

    for (i = 0; i < MV_OFFSET_BITS; i++)
      for (j = 0; j < 2; j++)
        comps->bits[i][j] += comps_t->bits[i][j];

    for (i = 0; i < MV_FP_SIZE; i++)
      comps->fp[i] += comps_t->fp[i];
  }
}
