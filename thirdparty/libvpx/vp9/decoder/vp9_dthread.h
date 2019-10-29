/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_DECODER_VP9_DTHREAD_H_
#define VP9_DECODER_VP9_DTHREAD_H_

#include "./vpx_config.h"
#include "vpx_util/vpx_thread.h"
#include "vpx/internal/vpx_codec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP9Common;
struct VP9Decoder;

// WorkerData for the FrameWorker thread. It contains all the information of
// the worker and decode structures for decoding a frame.
typedef struct FrameWorkerData {
  struct VP9Decoder *pbi;
  const uint8_t *data;
  const uint8_t *data_end;
  size_t data_size;
  void *user_priv;
  int result;
  int worker_id;
  int received_frame;

  // scratch_buffer is used in frame parallel mode only.
  // It is used to make a copy of the compressed data.
  uint8_t *scratch_buffer;
  size_t scratch_buffer_size;

#if CONFIG_MULTITHREAD
  pthread_mutex_t stats_mutex;
  pthread_cond_t stats_cond;
#endif

  int frame_context_ready;  // Current frame's context is ready to read.
  int frame_decoded;        // Finished decoding current frame.
} FrameWorkerData;

void vp9_frameworker_lock_stats(VPxWorker *const worker);
void vp9_frameworker_unlock_stats(VPxWorker *const worker);
void vp9_frameworker_signal_stats(VPxWorker *const worker);

// Wait until ref_buf has been decoded to row in real pixel unit.
// Note: worker may already finish decoding ref_buf and release it in order to
// start decoding next frame. So need to check whether worker is still decoding
// ref_buf.
void vp9_frameworker_wait(VPxWorker *const worker, RefCntBuffer *const ref_buf,
                          int row);

// FrameWorker broadcasts its decoding progress so other workers that are
// waiting on it can resume decoding.
void vp9_frameworker_broadcast(RefCntBuffer *const buf, int row);

// Copy necessary decoding context from src worker to dst worker.
void vp9_frameworker_copy_context(VPxWorker *const dst_worker,
                                  VPxWorker *const src_worker);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // VP9_DECODER_VP9_DTHREAD_H_
