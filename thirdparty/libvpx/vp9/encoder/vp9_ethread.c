/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/common/vp9_thread_common.h"
#include "vp9/encoder/vp9_bitstream.h"
#include "vp9/encoder/vp9_encodeframe.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_multi_thread.h"
#include "vp9/encoder/vp9_temporal_filter.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_util/vpx_pthread.h"

static void accumulate_rd_opt(ThreadData *td, ThreadData *td_t) {
  int i, j, k, l, m, n;

  for (i = 0; i < REFERENCE_MODES; i++)
    td->rd_counts.comp_pred_diff[i] += td_t->rd_counts.comp_pred_diff[i];

  for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; i++)
    td->rd_counts.filter_diff[i] += td_t->rd_counts.filter_diff[i];

  for (i = 0; i < TX_SIZES; i++)
    for (j = 0; j < PLANE_TYPES; j++)
      for (k = 0; k < REF_TYPES; k++)
        for (l = 0; l < COEF_BANDS; l++)
          for (m = 0; m < COEFF_CONTEXTS; m++)
            for (n = 0; n < ENTROPY_TOKENS; n++)
              td->rd_counts.coef_counts[i][j][k][l][m][n] +=
                  td_t->rd_counts.coef_counts[i][j][k][l][m][n];
}

static int enc_worker_hook(void *arg1, void *unused) {
  EncWorkerData *const thread_data = (EncWorkerData *)arg1;
  VP9_COMP *const cpi = thread_data->cpi;
  const VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  int t;

  (void)unused;

  for (t = thread_data->start; t < tile_rows * tile_cols;
       t += cpi->num_workers) {
    int tile_row = t / tile_cols;
    int tile_col = t % tile_cols;

    vp9_encode_tile(cpi, thread_data->td, tile_row, tile_col);
  }

  return 1;
}

static int get_max_tile_cols(VP9_COMP *cpi) {
  const int aligned_width = ALIGN_POWER_OF_TWO(cpi->oxcf.width, MI_SIZE_LOG2);
  int mi_cols = aligned_width >> MI_SIZE_LOG2;
  int min_log2_tile_cols, max_log2_tile_cols;
  int log2_tile_cols;

  vp9_get_tile_n_bits(mi_cols, &min_log2_tile_cols, &max_log2_tile_cols);
  log2_tile_cols =
      clamp(cpi->oxcf.tile_columns, min_log2_tile_cols, max_log2_tile_cols);
  if (cpi->oxcf.target_level == LEVEL_AUTO) {
    const int level_tile_cols =
        log_tile_cols_from_picsize_level(cpi->common.width, cpi->common.height);
    if (log2_tile_cols > level_tile_cols) {
      log2_tile_cols = VPXMAX(level_tile_cols, min_log2_tile_cols);
    }
  }
  return (1 << log2_tile_cols);
}

static void create_enc_workers(VP9_COMP *cpi, int num_workers) {
  VP9_COMMON *const cm = &cpi->common;
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  int i;
  // While using SVC, we need to allocate threads according to the highest
  // resolution. When row based multithreading is enabled, it is OK to
  // allocate more threads than the number of max tile columns.
  if (cpi->use_svc && !cpi->row_mt) {
    int max_tile_cols = get_max_tile_cols(cpi);
    num_workers = VPXMIN(cpi->oxcf.max_threads, max_tile_cols);
  }
  assert(num_workers > 0);
  if (num_workers == cpi->num_workers) return;
  vp9_loop_filter_dealloc(&cpi->lf_row_sync);
  vp9_bitstream_encode_tiles_buffer_dealloc(cpi);
  vp9_encode_free_mt_data(cpi);

  CHECK_MEM_ERROR(&cm->error, cpi->workers,
                  vpx_malloc(num_workers * sizeof(*cpi->workers)));

  CHECK_MEM_ERROR(&cm->error, cpi->tile_thr_data,
                  vpx_calloc(num_workers, sizeof(*cpi->tile_thr_data)));

  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    EncWorkerData *thread_data = &cpi->tile_thr_data[i];

    ++cpi->num_workers;
    winterface->init(worker);
    worker->thread_name = "vpx enc worker";

    if (i < num_workers - 1) {
      thread_data->cpi = cpi;

      // Allocate thread data.
      CHECK_MEM_ERROR(&cm->error, thread_data->td,
                      vpx_memalign(32, sizeof(*thread_data->td)));
      vp9_zero(*thread_data->td);

      // Set up pc_tree.
      thread_data->td->leaf_tree = NULL;
      thread_data->td->pc_tree = NULL;
      vp9_setup_pc_tree(cm, thread_data->td);

      // Allocate frame counters in thread data.
      CHECK_MEM_ERROR(&cm->error, thread_data->td->counts,
                      vpx_calloc(1, sizeof(*thread_data->td->counts)));

      // Create threads
      if (!winterface->reset(worker))
        vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                           "Tile encoder thread creation failed");
    } else {
      // Main thread acts as a worker and uses the thread data in cpi.
      thread_data->cpi = cpi;
      thread_data->td = &cpi->td;
    }
    winterface->sync(worker);
  }
}

static void launch_enc_workers(VP9_COMP *cpi, VPxWorkerHook hook, void *data2,
                               int num_workers) {
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  int i;

  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    worker->hook = hook;
    worker->data1 = &cpi->tile_thr_data[i];
    worker->data2 = data2;
  }

  // Encode a frame
  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    EncWorkerData *const thread_data = (EncWorkerData *)worker->data1;

    // Set the starting tile for each thread.
    thread_data->start = i;

    if (i == cpi->num_workers - 1)
      winterface->execute(worker);
    else
      winterface->launch(worker);
  }

  // Encoding ends.
  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    winterface->sync(worker);
  }
}

void vp9_encode_free_mt_data(struct VP9_COMP *cpi) {
  int t;
  for (t = 0; t < cpi->num_workers; ++t) {
    VPxWorker *const worker = &cpi->workers[t];
    EncWorkerData *const thread_data = &cpi->tile_thr_data[t];

    // Deallocate allocated threads.
    vpx_get_worker_interface()->end(worker);

    // Deallocate allocated thread data.
    if (t < cpi->num_workers - 1) {
      vpx_free(thread_data->td->counts);
      vp9_free_pc_tree(thread_data->td);
      vpx_free(thread_data->td);
    }
  }
  vpx_free(cpi->tile_thr_data);
  cpi->tile_thr_data = NULL;
  vpx_free(cpi->workers);
  cpi->workers = NULL;
  cpi->num_workers = 0;
}

void vp9_encode_tiles_mt(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int num_workers = VPXMIN(cpi->oxcf.max_threads, tile_cols);
  int i;

  vp9_init_tile_data(cpi);

  create_enc_workers(cpi, num_workers);

  for (i = 0; i < num_workers; i++) {
    EncWorkerData *const thread_data = &cpi->tile_thr_data[i];

    // Before encoding a frame, copy the thread data from cpi.
    if (thread_data->td != &cpi->td) {
      thread_data->td->mb = cpi->td.mb;
      thread_data->td->rd_counts = cpi->td.rd_counts;
    }
    if (thread_data->td->counts != &cpi->common.counts) {
      memcpy(thread_data->td->counts, &cpi->common.counts,
             sizeof(cpi->common.counts));
    }

    // Handle use_nonrd_pick_mode case.
    if (cpi->sf.use_nonrd_pick_mode) {
      MACROBLOCK *const x = &thread_data->td->mb;
      MACROBLOCKD *const xd = &x->e_mbd;
      struct macroblock_plane *const p = x->plane;
      struct macroblockd_plane *const pd = xd->plane;
      PICK_MODE_CONTEXT *ctx = &thread_data->td->pc_root->none;
      int j;

      for (j = 0; j < MAX_MB_PLANE; ++j) {
        p[j].coeff = ctx->coeff_pbuf[j][0];
        p[j].qcoeff = ctx->qcoeff_pbuf[j][0];
        pd[j].dqcoeff = ctx->dqcoeff_pbuf[j][0];
        p[j].eobs = ctx->eobs_pbuf[j][0];
      }
    }
  }

  launch_enc_workers(cpi, enc_worker_hook, NULL, num_workers);

  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    EncWorkerData *const thread_data = (EncWorkerData *)worker->data1;

    // Accumulate counters.
    if (i < cpi->num_workers - 1) {
      vp9_accumulate_frame_counts(&cm->counts, thread_data->td->counts, 0);
      accumulate_rd_opt(&cpi->td, thread_data->td);
    }
  }
}

#if !CONFIG_REALTIME_ONLY
static void accumulate_fp_tile_stat(TileDataEnc *tile_data,
                                    TileDataEnc *tile_data_t) {
  tile_data->fp_data.intra_factor += tile_data_t->fp_data.intra_factor;
  tile_data->fp_data.brightness_factor +=
      tile_data_t->fp_data.brightness_factor;
  tile_data->fp_data.coded_error += tile_data_t->fp_data.coded_error;
  tile_data->fp_data.sr_coded_error += tile_data_t->fp_data.sr_coded_error;
  tile_data->fp_data.frame_noise_energy +=
      tile_data_t->fp_data.frame_noise_energy;
  tile_data->fp_data.intra_error += tile_data_t->fp_data.intra_error;
  tile_data->fp_data.intercount += tile_data_t->fp_data.intercount;
  tile_data->fp_data.second_ref_count += tile_data_t->fp_data.second_ref_count;
  tile_data->fp_data.neutral_count += tile_data_t->fp_data.neutral_count;
  tile_data->fp_data.intra_count_low += tile_data_t->fp_data.intra_count_low;
  tile_data->fp_data.intra_count_high += tile_data_t->fp_data.intra_count_high;
  tile_data->fp_data.intra_skip_count += tile_data_t->fp_data.intra_skip_count;
  tile_data->fp_data.mvcount += tile_data_t->fp_data.mvcount;
  tile_data->fp_data.new_mv_count += tile_data_t->fp_data.new_mv_count;
  tile_data->fp_data.sum_mvr += tile_data_t->fp_data.sum_mvr;
  tile_data->fp_data.sum_mvr_abs += tile_data_t->fp_data.sum_mvr_abs;
  tile_data->fp_data.sum_mvc += tile_data_t->fp_data.sum_mvc;
  tile_data->fp_data.sum_mvc_abs += tile_data_t->fp_data.sum_mvc_abs;
  tile_data->fp_data.sum_mvrs += tile_data_t->fp_data.sum_mvrs;
  tile_data->fp_data.sum_mvcs += tile_data_t->fp_data.sum_mvcs;
  tile_data->fp_data.sum_in_vectors += tile_data_t->fp_data.sum_in_vectors;
  tile_data->fp_data.intra_smooth_count +=
      tile_data_t->fp_data.intra_smooth_count;
  const int min_start_row = VPXMIN(tile_data->fp_data.image_data_start_row,
                                   tile_data_t->fp_data.image_data_start_row);
  tile_data->fp_data.image_data_start_row =
      (min_start_row == INVALID_ROW)
          ? VPXMAX(tile_data->fp_data.image_data_start_row,
                   tile_data_t->fp_data.image_data_start_row)
          : min_start_row;
}
#endif  // !CONFIG_REALTIME_ONLY

// Allocate memory for row synchronization
void vp9_row_mt_sync_mem_alloc(VP9RowMTSync *row_mt_sync, VP9_COMMON *cm,
                               int rows) {
  row_mt_sync->rows = rows;
#if CONFIG_MULTITHREAD
  {
    int i;

    CHECK_MEM_ERROR(&cm->error, row_mt_sync->mutex,
                    vpx_malloc(sizeof(*row_mt_sync->mutex) * rows));
    if (row_mt_sync->mutex) {
      for (i = 0; i < rows; ++i) {
        pthread_mutex_init(&row_mt_sync->mutex[i], NULL);
      }
    }

    CHECK_MEM_ERROR(&cm->error, row_mt_sync->cond,
                    vpx_malloc(sizeof(*row_mt_sync->cond) * rows));
    if (row_mt_sync->cond) {
      for (i = 0; i < rows; ++i) {
        pthread_cond_init(&row_mt_sync->cond[i], NULL);
      }
    }
  }
#endif  // CONFIG_MULTITHREAD

  CHECK_MEM_ERROR(&cm->error, row_mt_sync->cur_col,
                  vpx_malloc(sizeof(*row_mt_sync->cur_col) * rows));

  // Set up nsync.
  row_mt_sync->sync_range = 1;
}

// Deallocate row based multi-threading synchronization related mutex and data
void vp9_row_mt_sync_mem_dealloc(VP9RowMTSync *row_mt_sync) {
  if (row_mt_sync != NULL) {
#if CONFIG_MULTITHREAD
    int i;

    if (row_mt_sync->mutex != NULL) {
      for (i = 0; i < row_mt_sync->rows; ++i) {
        pthread_mutex_destroy(&row_mt_sync->mutex[i]);
      }
      vpx_free(row_mt_sync->mutex);
    }
    if (row_mt_sync->cond != NULL) {
      for (i = 0; i < row_mt_sync->rows; ++i) {
        pthread_cond_destroy(&row_mt_sync->cond[i]);
      }
      vpx_free(row_mt_sync->cond);
    }
#endif  // CONFIG_MULTITHREAD
    vpx_free(row_mt_sync->cur_col);
    // clear the structure as the source of this call may be dynamic change
    // in tiles in which case this call will be followed by an _alloc()
    // which may fail.
    vp9_zero(*row_mt_sync);
  }
}

void vp9_row_mt_sync_read(VP9RowMTSync *const row_mt_sync, int r, int c) {
#if CONFIG_MULTITHREAD
  const int nsync = row_mt_sync->sync_range;

  if (r && !(c & (nsync - 1))) {
    pthread_mutex_t *const mutex = &row_mt_sync->mutex[r - 1];
    pthread_mutex_lock(mutex);

    while (c > row_mt_sync->cur_col[r - 1] - nsync + 1) {
      pthread_cond_wait(&row_mt_sync->cond[r - 1], mutex);
    }
    pthread_mutex_unlock(mutex);
  }
#else
  (void)row_mt_sync;
  (void)r;
  (void)c;
#endif  // CONFIG_MULTITHREAD
}

void vp9_row_mt_sync_read_dummy(VP9RowMTSync *const row_mt_sync, int r, int c) {
  (void)row_mt_sync;
  (void)r;
  (void)c;
  return;
}

void vp9_row_mt_sync_write(VP9RowMTSync *const row_mt_sync, int r, int c,
                           const int cols) {
#if CONFIG_MULTITHREAD
  const int nsync = row_mt_sync->sync_range;
  int cur;
  // Only signal when there are enough encoded blocks for next row to run.
  int sig = 1;

  if (c < cols - 1) {
    cur = c;
    if (c % nsync != nsync - 1) sig = 0;
  } else {
    cur = cols + nsync;
  }

  if (sig) {
    pthread_mutex_lock(&row_mt_sync->mutex[r]);

    row_mt_sync->cur_col[r] = cur;

    pthread_cond_signal(&row_mt_sync->cond[r]);
    pthread_mutex_unlock(&row_mt_sync->mutex[r]);
  }
#else
  (void)row_mt_sync;
  (void)r;
  (void)c;
  (void)cols;
#endif  // CONFIG_MULTITHREAD
}

void vp9_row_mt_sync_write_dummy(VP9RowMTSync *const row_mt_sync, int r, int c,
                                 const int cols) {
  (void)row_mt_sync;
  (void)r;
  (void)c;
  (void)cols;
  return;
}

#if !CONFIG_REALTIME_ONLY
static int first_pass_worker_hook(void *arg1, void *arg2) {
  EncWorkerData *const thread_data = (EncWorkerData *)arg1;
  MultiThreadHandle *multi_thread_ctxt = (MultiThreadHandle *)arg2;
  VP9_COMP *const cpi = thread_data->cpi;
  const VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int tile_row, tile_col;
  TileDataEnc *this_tile;
  int end_of_frame;
  int thread_id = thread_data->thread_id;
  int cur_tile_id = multi_thread_ctxt->thread_id_to_tile_id[thread_id];
  JobNode *proc_job = NULL;
  FIRSTPASS_DATA fp_acc_data;
  MV zero_mv = { 0, 0 };
  MV best_ref_mv;
  int mb_row;

  end_of_frame = 0;
  while (0 == end_of_frame) {
    // Get the next job in the queue
    proc_job =
        (JobNode *)vp9_enc_grp_get_next_job(multi_thread_ctxt, cur_tile_id);
    if (NULL == proc_job) {
      // Query for the status of other tiles
      end_of_frame = vp9_get_tiles_proc_status(
          multi_thread_ctxt, thread_data->tile_completion_status, &cur_tile_id,
          tile_cols);
    } else {
      tile_col = proc_job->tile_col_id;
      tile_row = proc_job->tile_row_id;

      this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
      mb_row = proc_job->vert_unit_row_num;

      best_ref_mv = zero_mv;
      vp9_zero(fp_acc_data);
      fp_acc_data.image_data_start_row = INVALID_ROW;
      vp9_first_pass_encode_tile_mb_row(cpi, thread_data->td, &fp_acc_data,
                                        this_tile, &best_ref_mv, mb_row);
    }
  }
  return 1;
}

void vp9_encode_fp_row_mt(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  TileDataEnc *first_tile_col;
  int num_workers = VPXMAX(cpi->oxcf.max_threads, 1);
  int i;

  if (multi_thread_ctxt->allocated_tile_cols < tile_cols ||
      multi_thread_ctxt->allocated_tile_rows < tile_rows ||
      multi_thread_ctxt->allocated_vert_unit_rows < cm->mb_rows) {
    vp9_row_mt_mem_dealloc(cpi);
    vp9_init_tile_data(cpi);
    vp9_row_mt_mem_alloc(cpi);
  } else {
    vp9_init_tile_data(cpi);
  }

  create_enc_workers(cpi, num_workers);

  vp9_assign_tile_to_thread(multi_thread_ctxt, tile_cols, cpi->num_workers);

  vp9_prepare_job_queue(cpi, FIRST_PASS_JOB);

  vp9_multi_thread_tile_init(cpi);

  for (i = 0; i < num_workers; i++) {
    EncWorkerData *thread_data;
    thread_data = &cpi->tile_thr_data[i];

    // Before encoding a frame, copy the thread data from cpi.
    if (thread_data->td != &cpi->td) {
      thread_data->td->mb = cpi->td.mb;
    }
  }

  launch_enc_workers(cpi, first_pass_worker_hook, multi_thread_ctxt,
                     num_workers);

  first_tile_col = &cpi->tile_data[0];
  for (i = 1; i < tile_cols; i++) {
    TileDataEnc *this_tile = &cpi->tile_data[i];
    accumulate_fp_tile_stat(first_tile_col, this_tile);
  }
}

static int temporal_filter_worker_hook(void *arg1, void *arg2) {
  EncWorkerData *const thread_data = (EncWorkerData *)arg1;
  MultiThreadHandle *multi_thread_ctxt = (MultiThreadHandle *)arg2;
  VP9_COMP *const cpi = thread_data->cpi;
  const VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int tile_row, tile_col;
  int mb_col_start, mb_col_end;
  TileDataEnc *this_tile;
  int end_of_frame;
  int thread_id = thread_data->thread_id;
  int cur_tile_id = multi_thread_ctxt->thread_id_to_tile_id[thread_id];
  JobNode *proc_job = NULL;
  int mb_row;

  end_of_frame = 0;
  while (0 == end_of_frame) {
    // Get the next job in the queue
    proc_job =
        (JobNode *)vp9_enc_grp_get_next_job(multi_thread_ctxt, cur_tile_id);
    if (NULL == proc_job) {
      // Query for the status of other tiles
      end_of_frame = vp9_get_tiles_proc_status(
          multi_thread_ctxt, thread_data->tile_completion_status, &cur_tile_id,
          tile_cols);
    } else {
      tile_col = proc_job->tile_col_id;
      tile_row = proc_job->tile_row_id;
      this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
      mb_col_start = (this_tile->tile_info.mi_col_start) >> TF_SHIFT;
      mb_col_end = (this_tile->tile_info.mi_col_end + TF_ROUND) >> TF_SHIFT;
      mb_row = proc_job->vert_unit_row_num;

      vp9_temporal_filter_iterate_row_c(cpi, thread_data->td, mb_row,
                                        mb_col_start, mb_col_end);
    }
  }
  return 1;
}

void vp9_temporal_filter_row_mt(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  int num_workers = cpi->num_workers ? cpi->num_workers : 1;
  int i;

  if (multi_thread_ctxt->allocated_tile_cols < tile_cols ||
      multi_thread_ctxt->allocated_tile_rows < tile_rows ||
      multi_thread_ctxt->allocated_vert_unit_rows < cm->mb_rows) {
    vp9_row_mt_mem_dealloc(cpi);
    vp9_init_tile_data(cpi);
    vp9_row_mt_mem_alloc(cpi);
  } else {
    vp9_init_tile_data(cpi);
  }

  create_enc_workers(cpi, num_workers);

  vp9_assign_tile_to_thread(multi_thread_ctxt, tile_cols, cpi->num_workers);

  vp9_prepare_job_queue(cpi, ARNR_JOB);

  for (i = 0; i < num_workers; i++) {
    EncWorkerData *thread_data;
    thread_data = &cpi->tile_thr_data[i];

    // Before encoding a frame, copy the thread data from cpi.
    if (thread_data->td != &cpi->td) {
      thread_data->td->mb = cpi->td.mb;
    }
  }

  launch_enc_workers(cpi, temporal_filter_worker_hook, multi_thread_ctxt,
                     num_workers);
}
#endif  // !CONFIG_REALTIME_ONLY

static int enc_row_mt_worker_hook(void *arg1, void *arg2) {
  EncWorkerData *const thread_data = (EncWorkerData *)arg1;
  MultiThreadHandle *multi_thread_ctxt = (MultiThreadHandle *)arg2;
  VP9_COMP *const cpi = thread_data->cpi;
  const VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int tile_row, tile_col;
  int end_of_frame;
  int thread_id = thread_data->thread_id;
  int cur_tile_id = multi_thread_ctxt->thread_id_to_tile_id[thread_id];
  JobNode *proc_job = NULL;
  int mi_row;

  end_of_frame = 0;
  while (0 == end_of_frame) {
    // Get the next job in the queue
    proc_job =
        (JobNode *)vp9_enc_grp_get_next_job(multi_thread_ctxt, cur_tile_id);
    if (NULL == proc_job) {
      // Query for the status of other tiles
      end_of_frame = vp9_get_tiles_proc_status(
          multi_thread_ctxt, thread_data->tile_completion_status, &cur_tile_id,
          tile_cols);
    } else {
      tile_col = proc_job->tile_col_id;
      tile_row = proc_job->tile_row_id;
      mi_row = proc_job->vert_unit_row_num * MI_BLOCK_SIZE;

      vp9_encode_sb_row(cpi, thread_data->td, tile_row, tile_col, mi_row);
    }
  }
  return 1;
}

void vp9_encode_tiles_row_mt(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  int num_workers = VPXMAX(cpi->oxcf.max_threads, 1);
  int i;

  if (multi_thread_ctxt->allocated_tile_cols < tile_cols ||
      multi_thread_ctxt->allocated_tile_rows < tile_rows ||
      multi_thread_ctxt->allocated_vert_unit_rows < cm->mb_rows) {
    vp9_row_mt_mem_dealloc(cpi);
    vp9_init_tile_data(cpi);
    vp9_row_mt_mem_alloc(cpi);
  } else {
    vp9_init_tile_data(cpi);
  }

  create_enc_workers(cpi, num_workers);

  vp9_assign_tile_to_thread(multi_thread_ctxt, tile_cols, cpi->num_workers);

  vp9_prepare_job_queue(cpi, ENCODE_JOB);

  vp9_multi_thread_tile_init(cpi);

  for (i = 0; i < num_workers; i++) {
    EncWorkerData *thread_data;
    thread_data = &cpi->tile_thr_data[i];
    // Before encoding a frame, copy the thread data from cpi.
    if (thread_data->td != &cpi->td) {
      thread_data->td->mb = cpi->td.mb;
      thread_data->td->rd_counts = cpi->td.rd_counts;
    }
    if (thread_data->td->counts != &cpi->common.counts) {
      memcpy(thread_data->td->counts, &cpi->common.counts,
             sizeof(cpi->common.counts));
    }

    // Handle use_nonrd_pick_mode case.
    if (cpi->sf.use_nonrd_pick_mode) {
      MACROBLOCK *const x = &thread_data->td->mb;
      MACROBLOCKD *const xd = &x->e_mbd;
      struct macroblock_plane *const p = x->plane;
      struct macroblockd_plane *const pd = xd->plane;
      PICK_MODE_CONTEXT *ctx = &thread_data->td->pc_root->none;
      int j;

      for (j = 0; j < MAX_MB_PLANE; ++j) {
        p[j].coeff = ctx->coeff_pbuf[j][0];
        p[j].qcoeff = ctx->qcoeff_pbuf[j][0];
        pd[j].dqcoeff = ctx->dqcoeff_pbuf[j][0];
        p[j].eobs = ctx->eobs_pbuf[j][0];
      }
    }
  }

  launch_enc_workers(cpi, enc_row_mt_worker_hook, multi_thread_ctxt,
                     num_workers);

  for (i = 0; i < num_workers; i++) {
    VPxWorker *const worker = &cpi->workers[i];
    EncWorkerData *const thread_data = (EncWorkerData *)worker->data1;

    // Accumulate counters.
    if (i < cpi->num_workers - 1) {
      vp9_accumulate_frame_counts(&cm->counts, thread_data->td->counts, 0);
      accumulate_rd_opt(&cpi->td, thread_data->td);
    }
  }
}
