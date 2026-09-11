/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "vpx_util/vpx_pthread.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vp9/encoder/vp9_multi_thread.h"
#include "vp9/encoder/vp9_temporal_filter.h"

void *vp9_enc_grp_get_next_job(MultiThreadHandle *multi_thread_ctxt,
                               int tile_id) {
  RowMTInfo *row_mt_info;
  JobQueueHandle *job_queue_hdl = NULL;
  void *next = NULL;
  JobNode *job_info = NULL;
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex_handle = NULL;
#endif

  row_mt_info = (RowMTInfo *)(&multi_thread_ctxt->row_mt_info[tile_id]);
  job_queue_hdl = (JobQueueHandle *)&row_mt_info->job_queue_hdl;
#if CONFIG_MULTITHREAD
  mutex_handle = &row_mt_info->job_mutex;
#endif

// lock the mutex for queue access
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(mutex_handle);
#endif
  next = job_queue_hdl->next;
  if (next != NULL) {
    JobQueue *job_queue = (JobQueue *)next;
    job_info = &job_queue->job_info;
    // Update the next job in the queue
    job_queue_hdl->next = job_queue->next;
    job_queue_hdl->num_jobs_acquired++;
  }

#if CONFIG_MULTITHREAD
  pthread_mutex_unlock(mutex_handle);
#endif

  return job_info;
}

void vp9_row_mt_alloc_rd_thresh(VP9_COMP *const cpi,
                                TileDataEnc *const this_tile) {
  VP9_COMMON *const cm = &cpi->common;
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  int i;

  if (this_tile->row_base_thresh_freq_fact != NULL) {
    if (sb_rows <= this_tile->sb_rows) {
      return;
    }
    vpx_free(this_tile->row_base_thresh_freq_fact);
    this_tile->row_base_thresh_freq_fact = NULL;
  }
  CHECK_MEM_ERROR(
      &cm->error, this_tile->row_base_thresh_freq_fact,
      (int *)vpx_calloc(sb_rows * BLOCK_SIZES * MAX_MODES,
                        sizeof(*(this_tile->row_base_thresh_freq_fact))));
  for (i = 0; i < sb_rows * BLOCK_SIZES * MAX_MODES; i++)
    this_tile->row_base_thresh_freq_fact[i] = RD_THRESH_INIT_FACT;
  this_tile->sb_rows = sb_rows;
}

void vp9_row_mt_mem_alloc(VP9_COMP *cpi) {
  struct VP9Common *cm = &cpi->common;
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  int tile_row, tile_col;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  int jobs_per_tile_col, total_jobs;

  // Allocate memory that is large enough for all row_mt stages. First pass
  // uses 16x16 block size.
  jobs_per_tile_col = VPXMAX(cm->mb_rows, sb_rows);
  // Calculate the total number of jobs
  total_jobs = jobs_per_tile_col * tile_cols;

  multi_thread_ctxt->allocated_tile_cols = tile_cols;
  multi_thread_ctxt->allocated_tile_rows = tile_rows;
  multi_thread_ctxt->allocated_vert_unit_rows = jobs_per_tile_col;

  CHECK_MEM_ERROR(&cm->error, multi_thread_ctxt->job_queue,
                  (JobQueue *)vpx_memalign(32, total_jobs * sizeof(JobQueue)));

#if CONFIG_MULTITHREAD
  // Create mutex for each tile
  for (tile_col = 0; tile_col < tile_cols; tile_col++) {
    RowMTInfo *row_mt_info = &multi_thread_ctxt->row_mt_info[tile_col];
    pthread_mutex_init(&row_mt_info->job_mutex, NULL);
  }
#endif

  // Allocate memory for row based multi-threading
  for (tile_col = 0; tile_col < tile_cols; tile_col++) {
    TileDataEnc *this_tile = &cpi->tile_data[tile_col];
    vp9_row_mt_sync_mem_alloc(&this_tile->row_mt_sync, cm, jobs_per_tile_col);
  }

  // Assign the sync pointer of tile row zero for every tile row > 0
  for (tile_row = 1; tile_row < tile_rows; tile_row++) {
    for (tile_col = 0; tile_col < tile_cols; tile_col++) {
      TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols + tile_col];
      TileDataEnc *this_col_tile = &cpi->tile_data[tile_col];
      this_tile->row_mt_sync = this_col_tile->row_mt_sync;
    }
  }

  // Calculate the number of vertical units in the given tile row
  for (tile_row = 0; tile_row < tile_rows; tile_row++) {
    TileDataEnc *this_tile = &cpi->tile_data[tile_row * tile_cols];
    TileInfo *tile_info = &this_tile->tile_info;
    multi_thread_ctxt->num_tile_vert_sbs[tile_row] =
        get_num_vert_units(*tile_info, MI_BLOCK_SIZE_LOG2);
  }
}

void vp9_row_mt_mem_dealloc(VP9_COMP *cpi) {
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  int tile_col;
#if CONFIG_MULTITHREAD
  int tile_row;
#endif

  // Deallocate memory for job queue
  if (multi_thread_ctxt->job_queue) {
    vpx_free(multi_thread_ctxt->job_queue);
    multi_thread_ctxt->job_queue = NULL;
  }

#if CONFIG_MULTITHREAD
  // Destroy mutex for each tile
  for (tile_col = 0; tile_col < multi_thread_ctxt->allocated_tile_cols;
       tile_col++) {
    RowMTInfo *row_mt_info = &multi_thread_ctxt->row_mt_info[tile_col];
    pthread_mutex_destroy(&row_mt_info->job_mutex);
  }
#endif

  // Free row based multi-threading sync memory
  for (tile_col = 0; tile_col < multi_thread_ctxt->allocated_tile_cols;
       tile_col++) {
    TileDataEnc *this_tile = &cpi->tile_data[tile_col];
    vp9_row_mt_sync_mem_dealloc(&this_tile->row_mt_sync);
  }

#if CONFIG_MULTITHREAD
  for (tile_row = 0; tile_row < multi_thread_ctxt->allocated_tile_rows;
       tile_row++) {
    for (tile_col = 0; tile_col < multi_thread_ctxt->allocated_tile_cols;
         tile_col++) {
      TileDataEnc *this_tile =
          &cpi->tile_data[tile_row * multi_thread_ctxt->allocated_tile_cols +
                          tile_col];
      if (this_tile->row_base_thresh_freq_fact != NULL) {
        vpx_free(this_tile->row_base_thresh_freq_fact);
        this_tile->row_base_thresh_freq_fact = NULL;
      }
    }
  }
#endif

  multi_thread_ctxt->allocated_tile_cols = 0;
  multi_thread_ctxt->allocated_tile_rows = 0;
  multi_thread_ctxt->allocated_vert_unit_rows = 0;
}

void vp9_multi_thread_tile_init(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  int i;

  for (i = 0; i < tile_cols; i++) {
    TileDataEnc *this_tile = &cpi->tile_data[i];
    int jobs_per_tile_col = cpi->oxcf.pass == 1 ? cm->mb_rows : sb_rows;

    // Initialize cur_col to -1 for all rows.
    memset(this_tile->row_mt_sync.cur_col, -1,
           sizeof(*this_tile->row_mt_sync.cur_col) * jobs_per_tile_col);
    vp9_zero(this_tile->fp_data);
    this_tile->fp_data.image_data_start_row = INVALID_ROW;
  }
}

void vp9_assign_tile_to_thread(MultiThreadHandle *multi_thread_ctxt,
                               int tile_cols, int num_workers) {
  int tile_id = 0;
  int i;

  // Allocating the threads for the tiles
  for (i = 0; i < num_workers; i++) {
    multi_thread_ctxt->thread_id_to_tile_id[i] = tile_id++;
    if (tile_id == tile_cols) tile_id = 0;
  }
}

int vp9_get_job_queue_status(MultiThreadHandle *multi_thread_ctxt,
                             int cur_tile_id) {
  RowMTInfo *row_mt_info;
  JobQueueHandle *job_queue_hndl;
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex;
#endif
  int num_jobs_remaining;

  row_mt_info = &multi_thread_ctxt->row_mt_info[cur_tile_id];
  job_queue_hndl = &row_mt_info->job_queue_hdl;
#if CONFIG_MULTITHREAD
  mutex = &row_mt_info->job_mutex;
#endif

#if CONFIG_MULTITHREAD
  pthread_mutex_lock(mutex);
#endif
  num_jobs_remaining =
      multi_thread_ctxt->jobs_per_tile_col - job_queue_hndl->num_jobs_acquired;
#if CONFIG_MULTITHREAD
  pthread_mutex_unlock(mutex);
#endif

  return (num_jobs_remaining);
}

void vp9_prepare_job_queue(VP9_COMP *cpi, JOB_TYPE job_type) {
  VP9_COMMON *const cm = &cpi->common;
  MultiThreadHandle *multi_thread_ctxt = &cpi->multi_thread_ctxt;
  JobQueue *job_queue = multi_thread_ctxt->job_queue;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int job_row_num, jobs_per_tile, jobs_per_tile_col = 0, total_jobs;
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  int tile_col, i;

  switch (job_type) {
    case ENCODE_JOB: jobs_per_tile_col = sb_rows; break;
    case FIRST_PASS_JOB: jobs_per_tile_col = cm->mb_rows; break;
    case ARNR_JOB:
      jobs_per_tile_col = ((cm->mi_rows + TF_ROUND) >> TF_SHIFT);
      break;
    default: assert(0);
  }

  total_jobs = jobs_per_tile_col * tile_cols;

  multi_thread_ctxt->jobs_per_tile_col = jobs_per_tile_col;
  // memset the entire job queue buffer to zero
  memset(job_queue, 0, total_jobs * sizeof(JobQueue));

  // Job queue preparation
  for (tile_col = 0; tile_col < tile_cols; tile_col++) {
    RowMTInfo *tile_ctxt = &multi_thread_ctxt->row_mt_info[tile_col];
    JobQueue *job_queue_curr, *job_queue_temp;
    int tile_row = 0;

    tile_ctxt->job_queue_hdl.next = (void *)job_queue;
    tile_ctxt->job_queue_hdl.num_jobs_acquired = 0;

    job_queue_curr = job_queue;
    job_queue_temp = job_queue;

    // loop over all the vertical rows
    for (job_row_num = 0, jobs_per_tile = 0; job_row_num < jobs_per_tile_col;
         job_row_num++, jobs_per_tile++) {
      job_queue_curr->job_info.vert_unit_row_num = job_row_num;
      job_queue_curr->job_info.tile_col_id = tile_col;
      job_queue_curr->job_info.tile_row_id = tile_row;
      job_queue_curr->next = (void *)(job_queue_temp + 1);
      job_queue_curr = ++job_queue_temp;

      if (ENCODE_JOB == job_type) {
        if (jobs_per_tile >=
            multi_thread_ctxt->num_tile_vert_sbs[tile_row] - 1) {
          tile_row++;
          jobs_per_tile = -1;
        }
      }
    }

    // Set the last pointer to NULL
    job_queue_curr += -1;
    job_queue_curr->next = (void *)NULL;

    // Move to the next tile
    job_queue += jobs_per_tile_col;
  }

  for (i = 0; i < cpi->num_workers; i++) {
    EncWorkerData *thread_data;
    thread_data = &cpi->tile_thr_data[i];
    thread_data->thread_id = i;

    for (tile_col = 0; tile_col < tile_cols; tile_col++)
      thread_data->tile_completion_status[tile_col] = 0;
  }
}

int vp9_get_tiles_proc_status(MultiThreadHandle *multi_thread_ctxt,
                              int *tile_completion_status, int *cur_tile_id,
                              int tile_cols) {
  int tile_col;
  int tile_id = -1;  // Stores the tile ID with minimum proc done
  int max_num_jobs_remaining = 0;
  int num_jobs_remaining;

  // Mark the completion to avoid check in the loop
  tile_completion_status[*cur_tile_id] = 1;
  // Check for the status of all the tiles
  for (tile_col = 0; tile_col < tile_cols; tile_col++) {
    if (tile_completion_status[tile_col] == 0) {
      num_jobs_remaining =
          vp9_get_job_queue_status(multi_thread_ctxt, tile_col);
      // Mark the completion to avoid checks during future switches across tiles
      if (num_jobs_remaining == 0) tile_completion_status[tile_col] = 1;
      if (num_jobs_remaining > max_num_jobs_remaining) {
        max_num_jobs_remaining = num_jobs_remaining;
        tile_id = tile_col;
      }
    }
  }

  if (-1 == tile_id) {
    return 1;
  } else {
    // Update the cur ID to the next tile ID that will be processed,
    // which will be the least processed tile
    *cur_tile_id = tile_id;
    return 0;
  }
}
