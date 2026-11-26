/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_
#define VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_job_queue.h"

void *vp9_enc_grp_get_next_job(MultiThreadHandle *multi_thread_ctxt,
                               int tile_id);

void vp9_prepare_job_queue(VP9_COMP *cpi, JOB_TYPE job_type);

int vp9_get_job_queue_status(MultiThreadHandle *multi_thread_ctxt,
                             int cur_tile_id);

void vp9_assign_tile_to_thread(MultiThreadHandle *multi_thread_ctxt,
                               int tile_cols, int num_workers);

void vp9_multi_thread_tile_init(VP9_COMP *cpi);

void vp9_row_mt_mem_alloc(VP9_COMP *cpi);

void vp9_row_mt_alloc_rd_thresh(VP9_COMP *const cpi,
                                TileDataEnc *const this_tile);

void vp9_row_mt_mem_dealloc(VP9_COMP *cpi);

int vp9_get_tiles_proc_status(MultiThreadHandle *multi_thread_ctxt,
                              int *tile_completion_status, int *cur_tile_id,
                              int tile_cols);

#endif  // VPX_VP9_ENCODER_VP9_MULTI_THREAD_H_
