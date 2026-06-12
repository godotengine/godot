/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_DECODER_VP9_JOB_QUEUE_H_
#define VPX_VP9_DECODER_VP9_JOB_QUEUE_H_

#include "vpx_util/vpx_pthread.h"

typedef struct {
  // Pointer to buffer base which contains the jobs
  uint8_t *buf_base;

  // Pointer to current address where new job can be added
  uint8_t *volatile buf_wr;

  // Pointer to current address from where next job can be obtained
  uint8_t *volatile buf_rd;

  // Pointer to end of job buffer
  uint8_t *buf_end;

  int terminate;

#if CONFIG_MULTITHREAD
  pthread_mutex_t mutex;
  pthread_cond_t cond;
#endif
} JobQueueRowMt;

void vp9_jobq_init(JobQueueRowMt *jobq, uint8_t *buf, size_t buf_size);
void vp9_jobq_reset(JobQueueRowMt *jobq);
void vp9_jobq_deinit(JobQueueRowMt *jobq);
void vp9_jobq_terminate(JobQueueRowMt *jobq);
int vp9_jobq_queue(JobQueueRowMt *jobq, void *job, size_t job_size);
int vp9_jobq_dequeue(JobQueueRowMt *jobq, void *job, size_t job_size,
                     int blocking);

#endif  // VPX_VP9_DECODER_VP9_JOB_QUEUE_H_
